# Copyright 2026 SAFEFLOW2027 Authors. All Rights Reserved.
"""Tests for checksum-gated host mirroring of sealed SHAPO milestones."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.star.sync_shapo_p47_milestones_to_host import (
    RemoteSeal,
    ensure_capacity,
    parse_remote_records,
    scan_once,
    transfer_job,
    validate_remote_arguments,
    validate_remote_python,
)
from scripts.star.watch_shapo_p47_milestones import MilestoneJob


def _write_seal(
    output: Path,
    *,
    environment: str = 'Ant-v4',
    seed: int = 2027,
    milestone: int = 500_000,
    identity: dict[str, object] | None = None,
) -> None:
    artifact = output / 'artifacts/evaluation.json'
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"episodes":100}\n', encoding='utf-8')
    audit = {
        'status': 'passed',
        'environment': environment,
        'training_seed': seed,
        'milestone': milestone,
        'training_seed_is_statistical_unit': True,
        'diagnostic_only': True,
        'used_for_checkpoint_or_configuration_selection': False,
        'server_only_checkpoint': {'server_only': True},
    }
    audit.update(identity or {})
    audit_path = output / 'milestone_audit.json'
    audit_path.write_text(json.dumps(audit) + '\n', encoding='utf-8')
    files = [artifact, audit_path]
    (output / 'checksums.sha256').write_text(
        ''.join(
            f'{hashlib.sha256(path.read_bytes()).hexdigest()}  '
            f'{path.relative_to(output).as_posix()}\n'
            for path in sorted(files)
        ),
        encoding='utf-8',
    )
    (output / 'milestone_audit_complete').touch()


def _record(relative: str, source: Path) -> RemoteSeal:
    files = [path for path in source.rglob('*') if path.is_file()]
    return RemoteSeal(
        relative=relative,
        bytes=sum(path.stat().st_size for path in files),
        file_count=len(files),
        manifest_sha256=hashlib.sha256((source / 'checksums.sha256').read_bytes()).hexdigest(),
    )


def test_remote_arguments_reject_shell_metacharacters_and_parent_traversal() -> None:
    validate_remote_arguments('seeta-nmb2', '/root/safe/milestones')
    validate_remote_python('/root/safe/env/bin/python')
    with pytest.raises(ValueError, match='unsafe SSH host'):
        validate_remote_arguments('server;touch-pwned', '/root/safe/milestones')
    with pytest.raises(ValueError, match='unsafe remote milestone root'):
        validate_remote_arguments('seeta-nmb2', '/root/safe/../milestones')
    with pytest.raises(ValueError, match='unsafe remote Python interpreter'):
        validate_remote_python('/root/safe/env/bin/python;touch-pwned')


def test_remote_records_are_exact_unique_expected_seals() -> None:
    relative = 'step_500000/ant-2027'
    checksum = 'a' * 64
    line = json.dumps(
        {
            'relative': relative,
            'bytes': 123,
            'file_count': 4,
            'manifest_sha256': checksum,
        },
    )
    records = parse_remote_records(line, {relative})
    assert records[relative].manifest_sha256 == checksum

    with pytest.raises(ValueError, match='duplicate milestone'):
        parse_remote_records(f'{line}\n{line}\n', {relative})
    with pytest.raises(ValueError, match='unexpected milestone'):
        parse_remote_records(line, {'step_500000/walker2d-2027'})


def test_capacity_gate_preserves_requested_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        'scripts.star.sync_shapo_p47_milestones_to_host.shutil.disk_usage',
        lambda _path: type('Usage', (), {'free': 1_000})(),
    )
    assert ensure_capacity(tmp_path, 400, 500) == 1_000
    with pytest.raises(RuntimeError, match='insufficient host space'):
        ensure_capacity(tmp_path, 501, 500)


def test_transfer_is_atomic_validated_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / 'host/milestones'
    output = output_root / 'step_500000/ant-2027'
    source = tmp_path / 'remote-seal'
    identity = {'source_commit': 'a' * 40}
    _write_seal(source, identity=identity)
    relative = 'step_500000/ant-2027'
    record = _record(relative, source)
    job = MilestoneJob(
        environment='Ant-v4',
        seed=2027,
        milestone=500_000,
        container=Path('/server-only/formal/runs/ant-2027'),
        output=output,
    )

    def fake_copy(_host: str, _root: str, observed: str, destination: Path) -> None:
        assert observed == relative
        for path in source.rglob('*'):
            if path.is_file():
                target = destination / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())

    monkeypatch.setattr(
        'scripts.star.sync_shapo_p47_milestones_to_host.copy_remote_tree',
        fake_copy,
    )
    monkeypatch.setattr(
        'scripts.star.sync_shapo_p47_milestones_to_host.shutil.disk_usage',
        lambda _path: type('Usage', (), {'free': 10_000_000})(),
    )
    first = transfer_job(
        job,
        record,
        output_root=output_root,
        host='seeta-nmb2',
        remote_root='/remote/milestones',
        reserve_bytes=1_000,
        expected_identity=identity,
    )
    assert first['state'] == 'transferred'
    assert output.is_dir()
    assert not list(output.rglob('*.pt'))

    second = transfer_job(
        job,
        record,
        output_root=output_root,
        host='seeta-nmb2',
        remote_root='/remote/milestones',
        reserve_bytes=1_000,
        expected_identity=identity,
    )
    assert second['state'] == 'already_present'


def test_existing_host_seal_must_match_remote_manifest(tmp_path: Path) -> None:
    output_root = tmp_path / 'host/milestones'
    output = output_root / 'step_500000/ant-2027'
    identity = {'source_commit': 'a' * 40}
    _write_seal(output, identity=identity)
    job = MilestoneJob('Ant-v4', 2027, 500_000, Path('/server-only'), output)
    record = _record('step_500000/ant-2027', output)
    changed = RemoteSeal(
        relative=record.relative,
        bytes=record.bytes,
        file_count=record.file_count,
        manifest_sha256='f' * 64,
    )
    with pytest.raises(ValueError, match='metadata disagree'):
        transfer_job(
            job,
            changed,
            output_root=output_root,
            host='seeta-nmb2',
            remote_root='/remote/milestones',
            reserve_bytes=0,
            expected_identity=identity,
        )

    changed_size = RemoteSeal(
        relative=record.relative,
        bytes=record.bytes + 1,
        file_count=record.file_count,
        manifest_sha256=record.manifest_sha256,
    )
    with pytest.raises(ValueError, match='metadata disagree'):
        transfer_job(
            job,
            changed_size,
            output_root=output_root,
            host='seeta-nmb2',
            remote_root='/remote/milestones',
            reserve_bytes=0,
            expected_identity=identity,
        )


def test_scan_rejects_remote_seal_disappearing_after_host_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / 'host/milestones'
    output = output_root / 'step_500000/ant-2027'
    identity = {'source_commit': 'a' * 40}
    _write_seal(output, identity=identity)
    job = MilestoneJob('Ant-v4', 2027, 500_000, Path('/server-only'), output)
    monkeypatch.setattr(
        'scripts.star.sync_shapo_p47_milestones_to_host.scan_remote',
        lambda *_args, **_kwargs: {},
    )
    with pytest.raises(ValueError, match='remote milestone seal disappeared'):
        scan_once(
            [job],
            output_root=output_root,
            host='seeta-nmb2',
            remote_root='/remote/milestones',
            remote_python='/remote/env/bin/python',
            reserve_bytes=0,
            expected_identity=identity,
        )

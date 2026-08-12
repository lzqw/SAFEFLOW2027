# Copyright 2026 SAFEFLOW2027 Authors. All Rights Reserved.
"""Tests for completion-aware host transfer of final SHAPO evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.star.watch_shapo_p47_final_sync import (
    REMOTE_HEALTH_FILES,
    REMOTE_PROBE,
    parse_checksum_manifest,
    sync_once,
    validate_local_output,
    validate_remote_arguments,
    validate_remote_python,
)


SOURCE_COMMIT = 'a' * 40


def _write_manifest(root: Path, name: str, excluded: set[str]) -> None:
    files = sorted(
        path
        for path in root.rglob('*')
        if path.is_file() and path.name not in excluded and path.name != name
    )
    (root / name).write_text(
        ''.join(
            f'{hashlib.sha256(path.read_bytes()).hexdigest()}  '
            f'{path.relative_to(root).as_posix()}\n'
            for path in files
        ),
        encoding='utf-8',
    )


def _write_transfer(output: Path) -> None:
    public = output / 'public_artifacts'
    public.mkdir(parents=True)
    (public / 'summary.json').write_text('{"status":"passed"}\n', encoding='utf-8')
    (public / 'shapo_p47_final_summary.json').write_text(
        f'{{"source_commit":"{SOURCE_COMMIT}",'
        '"statistical_unit":"training_seed","n_training_seeds_per_environment":5,'
        '"evaluation_artifact_count":60,"provenance_artifact_count":80,'
        '"trajectory_artifact_count":60,"checkpoint_artifact_count":120}\n',
        encoding='utf-8',
    )
    _write_manifest(public, 'checksums.sha256', {'checksums.sha256', 'finalization_complete'})
    (public / 'finalization_complete').touch()
    runtime = output / 'runtime_health'
    runtime.mkdir()
    health = {
        'state': 'complete',
        'alerts': [],
        'formal_matrix_complete': True,
        'completed_runs': 20,
        'failed_runs': 0,
        'stale_runs': 0,
        'total_runs': 20,
        'active_workers': 0,
        'completed_environment_steps': 150_000_000,
        'planned_environment_steps': 150_000_000,
    }
    for name in REMOTE_HEALTH_FILES:
        (runtime / name).write_text(
            json.dumps(health if name == 'completion_health.json' else {'artifact': name}),
            encoding='utf-8',
        )
    (output / 'runtime_health_manifest.csv').write_text(
        'artifact,remote_path,sha256,bytes\n'
        + ''.join(
            f'{name},/remote/formal/watchdog-test/{name},'
            f'{hashlib.sha256((runtime / name).read_bytes()).hexdigest()},'
            f'{(runtime / name).stat().st_size}\n'
            for name in REMOTE_HEALTH_FILES
        ),
        encoding='utf-8',
    )
    public_sha = hashlib.sha256((public / 'checksums.sha256').read_bytes()).hexdigest()
    (output / 'transfer_manifest.tsv').write_text(
        '\n'.join(
            [
                'remote_host\tseeta-nmb2',
                'remote_run_root\t/remote/formal',
                f'public_checksum_manifest_sha256\t{public_sha}',
            ],
        )
        + '\n',
        encoding='utf-8',
    )
    _write_manifest(
        output,
        'transfer_checksums.sha256',
        {'transfer_checksums.sha256', 'transfer_complete'},
    )
    (output / 'transfer_complete').touch()


def _write_remote_completion(run_root: Path, completion_root: Path) -> None:
    public = run_root / 'public_artifacts'
    public.mkdir(parents=True)
    (public / 'finalization_complete').touch()
    (public / 'checksums.sha256').touch()
    completion_root.mkdir()
    (completion_root / 'completion_watch_complete.json').write_text(
        json.dumps(
            {
                'state': 'complete',
                'formal_source_commit': SOURCE_COMMIT,
                'scientific_protocol_changed': False,
                'training_state_changed': False,
                'used_for_checkpoint_or_configuration_selection': False,
                'final': {
                    'status': 'passed',
                    'source_commit': SOURCE_COMMIT,
                    'statistical_unit': 'training_seed',
                    'n_training_seeds_per_environment': 5,
                    'evaluation_artifact_count': 60,
                    'provenance_artifact_count': 80,
                    'trajectory_artifact_count': 60,
                    'checkpoint_artifact_count': 120,
                    'strict_validation_count': 20,
                },
            },
        ),
        encoding='utf-8',
    )
    watchdog = run_root / 'watchdog-test'
    watchdog.mkdir()
    (watchdog / 'completion_health.json').write_text(
        json.dumps(
            {
                'state': 'complete',
                'alerts': [],
                'formal_matrix_complete': True,
                'completed_runs': 20,
                'failed_runs': 0,
                'stale_runs': 0,
                'total_runs': 20,
                'active_workers': 0,
                'completed_environment_steps': 150_000_000,
                'planned_environment_steps': 150_000_000,
            },
        ),
        encoding='utf-8',
    )


def _run_remote_probe(run_root: Path, completion_root: Path) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, '-', str(run_root), str(completion_root), SOURCE_COMMIT],
        input=REMOTE_PROBE,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def test_remote_arguments_reject_unsafe_values() -> None:
    validate_remote_arguments('seeta-nmb2', '/remote/formal', '/remote/completion')
    validate_remote_python('/remote/env/bin/python')
    with pytest.raises(ValueError, match='unsafe SSH host'):
        validate_remote_arguments('host;bad', '/remote/formal', '/remote/completion')
    with pytest.raises(ValueError, match='unsafe remote run root'):
        validate_remote_arguments('seeta-nmb2', '/remote/../formal', '/remote/completion')
    with pytest.raises(ValueError, match='unsafe remote Python interpreter'):
        validate_remote_python('/remote/env/bin/python;bad')


def test_checksum_manifest_rejects_parent_traversal(tmp_path: Path) -> None:
    manifest = tmp_path / 'checksums.sha256'
    manifest.write_text(f'{"0" * 64}  ../outside\n', encoding='utf-8')
    with pytest.raises(ValueError, match='unsafe checksum line'):
        parse_checksum_manifest(manifest)


def test_remote_probe_requires_complete_statistical_receipt(tmp_path: Path) -> None:
    run_root = tmp_path / 'formal'
    completion_root = tmp_path / 'completion'
    _write_remote_completion(run_root, completion_root)
    assert _run_remote_probe(run_root, completion_root)['state'] == 'ready'

    receipt = completion_root / 'completion_watch_complete.json'
    payload = json.loads(receipt.read_text(encoding='utf-8'))
    payload['final']['statistical_unit'] = 'episode'
    receipt.write_text(json.dumps(payload), encoding='utf-8')
    result = _run_remote_probe(run_root, completion_root)
    assert result['state'] == 'alert'
    assert 'statistical_unit' in str(result['error'])


def test_remote_probe_waits_during_atomic_completion_receipt_handoff(tmp_path: Path) -> None:
    run_root = tmp_path / 'formal'
    completion_root = tmp_path / 'completion'
    _write_remote_completion(run_root, completion_root)
    (completion_root / 'completion_watch_complete.json').unlink()
    assert _run_remote_probe(run_root, completion_root) == {'state': 'waiting'}


def test_waiting_remote_does_not_create_local_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / 'final'
    calls = []
    monkeypatch.setattr(
        'scripts.star.watch_shapo_p47_final_sync.probe_remote',
        lambda *_args: {'state': 'waiting'},
    )
    monkeypatch.setattr(
        'scripts.star.watch_shapo_p47_final_sync.invoke_sync',
        lambda *_args: calls.append('called'),
    )
    result = sync_once(
        remote_host='seeta-nmb2',
        remote_run_root='/remote/formal',
        remote_completion_root='/remote/completion',
        remote_python='/remote/env/bin/python',
        source_commit='a' * 40,
        output_root=output,
        reserve_bytes=1_000,
        sync_script=tmp_path / 'sync.sh',
    )
    assert result['state'] == 'waiting'
    assert calls == []
    assert not output.exists()


def test_ready_remote_transfers_once_then_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / 'final'
    calls = []
    monkeypatch.setattr(
        'scripts.star.watch_shapo_p47_final_sync.probe_remote',
        lambda *_args: {'state': 'ready'},
    )

    def fake_sync(*_args: object) -> dict[str, object]:
        calls.append('called')
        _write_transfer(output)
        return {'command': ['sync'], 'stdout': 'complete', 'stderr': ''}

    monkeypatch.setattr('scripts.star.watch_shapo_p47_final_sync.invoke_sync', fake_sync)
    arguments = {
        'remote_host': 'seeta-nmb2',
        'remote_run_root': '/remote/formal',
        'remote_completion_root': '/remote/completion',
        'remote_python': '/remote/env/bin/python',
        'source_commit': 'a' * 40,
        'output_root': output,
        'reserve_bytes': 1_000,
        'sync_script': tmp_path / 'sync.sh',
    }
    first = sync_once(**arguments)
    assert first['state'] == 'complete'
    assert first['transfer'] == 'invoked'
    assert calls == ['called']
    second = sync_once(**arguments)
    assert second['state'] == 'complete'
    assert second['transfer'] == 'already_present'
    assert calls == ['called']


def test_local_transfer_rejects_checksum_or_remote_identity_drift(tmp_path: Path) -> None:
    output = tmp_path / 'final'
    _write_transfer(output)
    result = validate_local_output(
        output,
        remote_host='seeta-nmb2',
        remote_run_root='/remote/formal',
        source_commit='a' * 40,
    )
    assert result['status'] == 'passed'

    summary = output / 'public_artifacts/summary.json'
    summary.write_text('{"status":"changed"}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum identity mismatch'):
        validate_local_output(
            output,
            remote_host='seeta-nmb2',
            remote_run_root='/remote/formal',
            source_commit='a' * 40,
        )


def test_local_transfer_rejects_server_checkpoint_payload(tmp_path: Path) -> None:
    output = tmp_path / 'final'
    _write_transfer(output)
    checkpoint = output / 'public_artifacts/checkpoint.pt'
    checkpoint.write_bytes(b'checkpoint')
    _write_manifest(
        output / 'public_artifacts',
        'checksums.sha256',
        {'checksums.sha256', 'finalization_complete'},
    )
    public_sha = hashlib.sha256(
        (output / 'public_artifacts/checksums.sha256').read_bytes(),
    ).hexdigest()
    transfer_manifest = output / 'transfer_manifest.tsv'
    lines = transfer_manifest.read_text(encoding='utf-8').splitlines()
    transfer_manifest.write_text(
        '\n'.join(
            (
                f'public_checksum_manifest_sha256\t{public_sha}'
                if line.startswith('public_checksum_manifest_sha256\t')
                else line
            )
            for line in lines
        )
        + '\n',
        encoding='utf-8',
    )
    _write_manifest(
        output,
        'transfer_checksums.sha256',
        {'transfer_checksums.sha256', 'transfer_complete'},
    )
    with pytest.raises(ValueError, match='server-only checkpoints'):
        validate_local_output(
            output,
            remote_host='seeta-nmb2',
            remote_run_root='/remote/formal',
            source_commit='a' * 40,
        )

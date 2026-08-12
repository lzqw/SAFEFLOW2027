# Copyright 2026 SAFEFLOW2027 Authors. All Rights Reserved.
"""Tests for non-invasive auditing of preregistered SHAPO P4.7 milestones."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from scripts.star.watch_shapo_p47_milestones import (
    DIRECT_SOURCES,
    GLOB_SOURCES,
    MilestoneJob,
    invoke_auditor,
    locked_jobs,
    milestone_audit_identity,
    parse_checksum_manifest,
    scan_once,
    stable_sources,
    validate_sealed_output,
)


def _job(tmp_path: Path, *, output_name: str = 'audit') -> MilestoneJob:
    return MilestoneJob(
        environment='Ant-v4',
        seed=2027,
        milestone=500_000,
        container=tmp_path / 'formal/runs/ant-2027',
        output=tmp_path / output_name,
    )


def _write_sources(job: MilestoneJob, modified: float = 100.0) -> None:
    for template in GLOB_SOURCES.values():
        relative = template.format(milestone=job.milestone).replace('**', 'segment')
        path = job.container / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'stable-source')
        os.utime(path, (modified, modified))
    for relative in DIRECT_SOURCES.values():
        path = job.container / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'stable-source')
        os.utime(path, (modified, modified))


def _write_seal(job: MilestoneJob, identity: dict[str, object] | None = None) -> None:
    artifact = job.output / 'artifacts/evaluation.json'
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"mean_return":1.0}\n', encoding='utf-8')
    audit = {
        'status': 'passed',
        'environment': job.environment,
        'training_seed': job.seed,
        'milestone': job.milestone,
        'training_seed_is_statistical_unit': True,
        'diagnostic_only': True,
        'used_for_checkpoint_or_configuration_selection': False,
        'server_only_checkpoint': {'server_only': True},
    }
    audit.update(identity or {})
    audit_path = job.output / 'milestone_audit.json'
    audit_path.write_text(json.dumps(audit) + '\n', encoding='utf-8')
    files = [artifact, audit_path]
    (job.output / 'checksums.sha256').write_text(
        ''.join(
            f'{hashlib.sha256(path.read_bytes()).hexdigest()}  '
            f'{path.relative_to(job.output).as_posix()}\n'
            for path in sorted(files)
        ),
        encoding='utf-8',
    )
    (job.output / 'milestone_audit_complete').touch()


def test_locked_jobs_expands_every_seed_environment_and_registered_step(tmp_path: Path) -> None:
    matrix = {
        'training_seeds': [2027, 2028],
        'environments': {
            'SafetyPointGoal1-v0': {'milestones': [100, 400]},
            'Ant-v4': {'milestones': [50, 200, 500]},
        },
    }
    jobs = locked_jobs(tmp_path / 'formal', tmp_path / 'audits', matrix)
    assert len(jobs) == 10
    assert jobs[0].key == 'SafetyPointGoal1-v0/seed-2027/step-100'
    assert jobs[-1].key == 'Ant-v4/seed-2028/step-500'
    assert jobs[-1].output == tmp_path / 'audits/step_500/ant-2028'


def test_sources_must_all_exist_and_age_past_stability_window(tmp_path: Path) -> None:
    job = _job(tmp_path)
    _write_sources(job)
    ready, missing, age = stable_sources(job, now=250.0, stable_seconds=120.0)
    assert ready is True
    assert missing == []
    assert age == 150.0

    evaluation = next(job.container.glob('training/**/artifacts/evaluation_step_500000.json'))
    evaluation.unlink()
    ready, missing, age = stable_sources(job, now=250.0, stable_seconds=120.0)
    assert ready is False
    assert missing == ['evaluation']
    assert age is None


def test_sealed_output_requires_full_hash_coverage_and_no_checkpoint_payload(
    tmp_path: Path,
) -> None:
    job = _job(tmp_path)
    _write_seal(job)
    assert validate_sealed_output(job)['diagnostic_only'] is True

    artifact = job.output / 'artifacts/evaluation.json'
    artifact.write_text('{"mean_return":2.0}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='identity mismatch'):
        validate_sealed_output(job)

    _write_seal(job)
    checkpoint = job.output / 'checkpoints/milestone.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'checkpoint-payload')
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with (job.output / 'checksums.sha256').open('a', encoding='utf-8') as stream:
        stream.write(f'{checksum}  checkpoints/milestone.pt\n')
    with pytest.raises(ValueError, match='copied checkpoint payloads'):
        validate_sealed_output(job)


def test_checksum_manifest_rejects_parent_traversal(tmp_path: Path) -> None:
    manifest = tmp_path / 'checksums.sha256'
    manifest.write_text(f'{"0" * 64}  ../outside\n', encoding='utf-8')
    with pytest.raises(ValueError, match='unsafe or duplicate'):
        parse_checksum_manifest(manifest)


def test_sealed_output_rejects_different_formal_source_identity(tmp_path: Path) -> None:
    job = _job(tmp_path)
    _write_seal(job, {'source_commit': 'a' * 40})
    with pytest.raises(ValueError, match="identity 'source_commit' changed"):
        validate_sealed_output(job, expected_identity={'source_commit': 'b' * 40})


def test_scan_audits_ready_job_once_then_revalidates_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(tmp_path)
    _write_sources(job)
    calls = []
    matrix_path = tmp_path / 'matrix.yaml'
    matrix_path.write_text(
        yaml.safe_dump(
            {
                'protocol_version': 'test-protocol-v1',
                'protocol_sha256': '1' * 64,
                'frozen_method_canonical_sha256': '2' * 64,
            },
        ),
        encoding='utf-8',
    )
    audit_script = tmp_path / 'audit_shapo_p47_milestone.py'
    audit_script.write_text('# audit\n', encoding='utf-8')
    audit_script.with_name('validate_shapo_p47_run.py').write_text(
        '# validator\n',
        encoding='utf-8',
    )
    identity = milestone_audit_identity(matrix_path, 'a' * 40, audit_script)

    def fake_invoke(current: MilestoneJob, **_: object) -> dict[str, str]:
        calls.append(current.key)
        _write_seal(current, identity)
        return {'stdout': '{"status":"passed"}', 'stderr': ''}

    monkeypatch.setattr(
        'scripts.star.watch_shapo_p47_milestones.invoke_auditor',
        fake_invoke,
    )
    arguments = {
        'matrix_path': matrix_path,
        'source_commit': 'a' * 40,
        'audit_script': audit_script,
        'python_bin': tmp_path / 'python',
        'stable_seconds': 120.0,
        'now': 250.0,
    }
    first = scan_once([job], **arguments)
    assert first['state'] == 'complete'
    assert first['newly_audited_jobs'] == 1
    assert calls == [job.key]

    second = scan_once([job], **arguments)
    assert second['state'] == 'complete'
    assert second['newly_audited_jobs'] == 0
    assert calls == [job.key]


def test_auditor_failure_preserves_child_stdout_and_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(tmp_path)
    monkeypatch.setattr(
        'scripts.star.watch_shapo_p47_milestones.subprocess.run',
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=7,
            stdout='auditor output',
            stderr='strict validation failed',
        ),
    )
    with pytest.raises(RuntimeError, match='strict validation failed'):
        invoke_auditor(
            job,
            matrix_path=tmp_path / 'matrix.yaml',
            source_commit='a' * 40,
            audit_script=tmp_path / 'audit.py',
            python_bin=tmp_path / 'python',
        )

# Copyright 2026 SAFEFLOW2027 Authors. All Rights Reserved.
"""Tests for fail-closed post-training SHAPO finalization orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.star.watch_shapo_p47_completion import (
    complete_once,
    completion_gates,
    expected_counts,
    validate_final_output,
)
from scripts.star.watch_shapo_p47_milestones import locked_jobs, milestone_audit_identity


def _matrix() -> dict[str, object]:
    return {
        'protocol_version': 'completion-test-v1',
        'protocol_sha256': '1' * 64,
        'frozen_method_canonical_sha256': '2' * 64,
        'training_seeds': [2027],
        'environments': {
            'Ant-v4': {
                'total_steps': 10,
                'milestones': [5, 10],
            },
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + '\n', encoding='utf-8')


def _write_seal(job: object, identity: dict[str, object]) -> None:
    artifact = job.output / 'artifacts/evaluation.json'
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"episodes":100}\n', encoding='utf-8')
    audit = {
        'status': 'passed',
        'environment': job.environment,
        'training_seed': job.seed,
        'milestone': job.milestone,
        'training_seed_is_statistical_unit': True,
        'diagnostic_only': True,
        'used_for_checkpoint_or_configuration_selection': False,
        'server_only_checkpoint': {'server_only': True},
        **identity,
    }
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


def _fixture(tmp_path: Path, *, complete: bool) -> dict[str, object]:
    matrix = _matrix()
    matrix_path = tmp_path / 'matrix.yaml'
    matrix_path.write_text(yaml.safe_dump(matrix), encoding='utf-8')
    audit_script = tmp_path / 'audit_shapo_p47_milestone.py'
    audit_script.write_text('# audit\n', encoding='utf-8')
    audit_script.with_name('validate_shapo_p47_run.py').write_text(
        '# validator\n',
        encoding='utf-8',
    )
    source_commit = 'a' * 40
    run_root = tmp_path / 'formal'
    milestone_root = tmp_path / 'milestones'
    watchdog_root = tmp_path / 'watchdog'
    output_root = run_root / 'public_artifacts'
    identity = milestone_audit_identity(matrix_path, source_commit, audit_script)

    if complete:
        (run_root / 'formal_matrix_complete').parent.mkdir(parents=True, exist_ok=True)
        (run_root / 'formal_matrix_complete').touch()
        (run_root / 'matrix_failure_count.txt').write_text('0\n', encoding='utf-8')
        validation = {
            'status': 'passed',
            'environment': 'Ant-v4',
            'training_seed': 2027,
            'training_seed_is_statistical_unit': True,
            'total_steps': 10,
            'protocol_version': matrix['protocol_version'],
            'protocol_sha256': matrix['protocol_sha256'],
            'frozen_method_canonical_sha256': matrix['frozen_method_canonical_sha256'],
        }
        _write_json(run_root / 'runs/ant-2027/formal_validation.json', validation)
        jobs = locked_jobs(run_root, milestone_root, matrix)
        for job in jobs:
            _write_seal(job, identity)
        _write_json(
            milestone_root / 'milestone_watch_latest.json',
            {
                'state': 'complete',
                'total_jobs': 2,
                'sealed_jobs': 2,
                'waiting_jobs': 0,
                'scientific_protocol_changed': False,
                'training_state_changed': False,
                'used_for_checkpoint_or_configuration_selection': False,
                'source_commit': source_commit,
            },
        )
        (milestone_root / 'milestone_watch_complete').touch()
        _write_json(
            watchdog_root / 'completion_health.json',
            {
                'state': 'complete',
                'alerts': [],
                'formal_matrix_complete': True,
                'completed_runs': 1,
                'failed_runs': 0,
                'stale_runs': 0,
                'total_runs': 1,
                'active_workers': 0,
                'completed_environment_steps': 10,
                'planned_environment_steps': 10,
                'filesystem_free_bytes': 10_000,
            },
        )
    else:
        _write_json(
            milestone_root / 'milestone_watch_latest.json',
            {
                'state': 'watching',
                'sealed_jobs': 0,
                'scientific_protocol_changed': False,
                'training_state_changed': False,
                'used_for_checkpoint_or_configuration_selection': False,
            },
        )
        _write_json(watchdog_root / 'latest_health.json', {'state': 'healthy'})

    return {
        'matrix': matrix,
        'matrix_path': matrix_path,
        'audit_script': audit_script,
        'source_commit': source_commit,
        'run_root': run_root,
        'milestone_root': milestone_root,
        'watchdog_root': watchdog_root,
        'output_root': output_root,
    }


def _write_final_output(
    output: Path,
    matrix: dict[str, object],
    source_commit: str,
) -> None:
    counts = expected_counts(matrix)
    summary = {
        'source_commit': source_commit,
        'statistical_unit': 'training_seed',
        'n_training_seeds_per_environment': 1,
        **{
            field: counts[field]
            for field in (
                'evaluation_artifact_count',
                'provenance_artifact_count',
                'trajectory_artifact_count',
                'checkpoint_artifact_count',
            )
        },
    }
    _write_json(output / 'shapo_p47_final_summary.json', summary)
    _write_json(output / 'strict_validations/ant-2027.json', {'status': 'passed'})
    files = [path for path in output.rglob('*') if path.is_file()]
    (output / 'checksums.sha256').write_text(
        ''.join(
            f'{hashlib.sha256(path.read_bytes()).hexdigest()}  '
            f'{path.relative_to(output).as_posix()}\n'
            for path in sorted(files)
        ),
        encoding='utf-8',
    )
    (output / 'finalization_complete').touch()


def test_expected_counts_are_derived_from_matrix() -> None:
    assert expected_counts(_matrix()) == {
        'formal_runs': 1,
        'milestone_jobs': 2,
        'evaluation_artifact_count': 2,
        'provenance_artifact_count': 4,
        'trajectory_artifact_count': 2,
        'checkpoint_artifact_count': 5,
    }


def test_incomplete_matrix_waits_without_creating_final_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, complete=False)
    called = []
    result = complete_once(
        **fixture,
        validation_source=tmp_path,
        python_bin=Path('/python'),
        finalizer_script=Path('/finalizer'),
    )
    assert result['state'] == 'waiting'
    assert result['gates']['ready'] is False
    assert not fixture['output_root'].exists()
    assert called == []


def test_all_three_gates_require_complete_immutable_evidence(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, complete=True)
    gates = completion_gates(
        run_root=fixture['run_root'],
        milestone_root=fixture['milestone_root'],
        watchdog_root=fixture['watchdog_root'],
        matrix_path=fixture['matrix_path'],
        matrix=fixture['matrix'],
        source_commit=fixture['source_commit'],
        audit_script=fixture['audit_script'],
    )
    assert gates['ready'] is True
    assert gates['training']['validated_runs'] == 1
    assert gates['milestones']['sealed_jobs'] == 2
    assert gates['watchdog']['latest_state'] == 'complete'
    assert gates['used_for_checkpoint_or_configuration_selection'] is False

    seal = next(fixture['milestone_root'].glob('step_*/ant-2027/milestone_audit.json'))
    seal.write_text('{}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='identity mismatch'):
        completion_gates(
            run_root=fixture['run_root'],
            milestone_root=fixture['milestone_root'],
            watchdog_root=fixture['watchdog_root'],
            matrix_path=fixture['matrix_path'],
            matrix=fixture['matrix'],
            source_commit=fixture['source_commit'],
            audit_script=fixture['audit_script'],
        )


def test_ready_matrix_invokes_finalizer_once_then_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, complete=True)
    calls = []

    def fake_finalizer(*_args: object) -> dict[str, object]:
        calls.append('invoked')
        _write_final_output(
            fixture['output_root'],
            fixture['matrix'],
            fixture['source_commit'],
        )
        return {'command': ['finalizer'], 'stdout': 'complete', 'stderr': ''}

    monkeypatch.setattr(
        'scripts.star.watch_shapo_p47_completion.invoke_finalizer',
        fake_finalizer,
    )
    arguments = {
        **fixture,
        'validation_source': tmp_path,
        'python_bin': Path('/python'),
        'finalizer_script': Path('/finalizer'),
    }
    first = complete_once(**arguments)
    assert first['state'] == 'complete'
    assert first['finalization'] == 'invoked'
    assert calls == ['invoked']
    second = complete_once(**arguments)
    assert second['state'] == 'complete'
    assert second['finalization'] == 'already_present'
    assert calls == ['invoked']


def test_final_output_rejects_statistical_or_checksum_drift(tmp_path: Path) -> None:
    matrix = _matrix()
    output = tmp_path / 'public_artifacts'
    _write_final_output(output, matrix, 'a' * 40)
    result = validate_final_output(output, matrix, 'a' * 40)
    assert result['strict_validation_count'] == 1

    summary = output / 'shapo_p47_final_summary.json'
    payload = json.loads(summary.read_text(encoding='utf-8'))
    payload['statistical_unit'] = 'episode'
    summary.write_text(json.dumps(payload) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum mismatch'):
        validate_final_output(output, matrix, 'a' * 40)

    checksum_path = output / 'checksums.sha256'
    lines = []
    for line in checksum_path.read_text(encoding='utf-8').splitlines():
        _checksum, relative = line.split('  ', 1)
        path = output / relative
        lines.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}')
    checksum_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match="field 'statistical_unit' changed"):
        validate_final_output(output, matrix, 'a' * 40)


def test_engineering_alerts_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, complete=False)
    fixture['run_root'].mkdir(parents=True, exist_ok=True)
    (fixture['run_root'] / 'matrix_failure_count.txt').write_text('1\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='failed runs'):
        completion_gates(
            run_root=fixture['run_root'],
            milestone_root=fixture['milestone_root'],
            watchdog_root=fixture['watchdog_root'],
            matrix_path=fixture['matrix_path'],
            matrix=fixture['matrix'],
            source_commit=fixture['source_commit'],
            audit_script=fixture['audit_script'],
        )

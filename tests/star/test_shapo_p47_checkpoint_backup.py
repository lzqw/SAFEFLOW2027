# Copyright 2026 SAFEFLOW2027 Authors. All Rights Reserved.
"""Tests for fail-closed SHAPO host checkpoint disaster recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml
from scripts.star.backup_shapo_p47_checkpoints_to_host import (
    BackupJob,
    RemoteCheckpoint,
    backup_once,
    ensure_capacity,
    expected_jobs,
    parse_remote_inventory,
    recover_interrupted_swap,
    transfer_checkpoint,
    validate_remote_arguments,
    validate_remote_python,
)
from scripts.star.materialize_shapo_p47_config import materialize
from scripts.star.validate_shapo_p47_checkpoint_backup import validate_checkpoint_backup


SOURCE_COMMIT = 'a' * 40
REMOTE_ROOT = Path('/root/autodl-tmp/SAFE/shapo_p47_four_env/formal')
REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX = REPO_ROOT / 'configs/star/shapo_p47_four_env.yaml'


def _checkpoint_payload(matrix: Path, environment: str, seed: int, steps: int) -> dict:
    config = materialize(matrix, environment, smoke=False)['custom_cfgs']
    config['seed'] = seed
    config['env_id'] = environment
    return {
        'format': 'star-checkpoint',
        'version': 1,
        'model': {'actor.weight': torch.zeros(1)},
        'optimizers': {
            name: {'state': {}, 'param_groups': []}
            for name in (
                'actor',
                'reward_critic',
                'cost_critic',
                'violation_critic',
                'energy_dual',
            )
        },
        'duals': {'energy': {}, 'global': {}},
        'normalizer': {'_count': torch.tensor(1.0)},
        'counters': {
            'env_steps': steps,
            'updates': steps,
            'actor_updates': steps // 2,
            'reward_critic_updates': steps * 2,
            'utd_carry': 0.0,
            'violations': 0,
            'cumulative_training_cost': 0.0,
        },
        'warmup': {
            'stage': 'W2',
            'geometry_enabled': False,
            'geometry_activation_step': None,
            'geometry_eligibility_step': None,
            'geometry_last_disable_step': None,
            'consecutive_passes': 0,
            'consecutive_failures': 0,
            'gate_evaluations': 0,
            'ema': {},
            'latest': {},
        },
        'rng': {'python': (), 'numpy': (), 'torch': torch.zeros(1), 'cuda': None},
        'replay': {
            'version': 3,
            'capacity': 400_000,
            'obs_dim': 1,
            'act_dim': 1,
            'context_dim': 1,
            'size': min(steps, 400_000),
            'pointer': min(steps, 400_000) % 400_000,
            'data': {
                name: torch.zeros(1).expand(min(steps, 400_000))
                for name in (
                    'obs',
                    'action',
                    'reward',
                    'cost',
                    'terminated',
                    'truncated',
                    'reward_done',
                    'cost_done',
                    'next_obs',
                    'context',
                    'next_context',
                    'risk_threshold',
                    'next_risk_threshold',
                    'empirical_cost_to_go',
                    'violation_label',
                    'boundary_score',
                    'is_boundary',
                    'is_failure',
                    'source_random',
                    'episode_id',
                    'is_heldout',
                    'category',
                    'generation',
                )
            },
            'episode_slots': [],
            'active_episode_ids': [],
            'next_episode_id': 0,
            'heldout_episode_period': 5,
        },
        'extra': {
            'environment': environment,
            'seed': seed,
            'resolved_config': config,
            'rollout_state': {},
            'completed_milestones': [],
            'completed_checkpoint_milestones': [],
            'last_heldout_evidence': {},
            'last_heldout_evidence_status': 'insufficient_evidence',
            'reward_td_diagnostics': [],
            'best_safe_checkpoint_selection': None,
        },
    }


def _write_checkpoint(
    path: Path,
    matrix: Path,
    *,
    environment: str = 'Ant-v4',
    seed: int = 2027,
    steps: int = 50_000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(matrix, environment, seed, steps), path)


def _record(job: str, checkpoint: Path) -> RemoteCheckpoint:
    return RemoteCheckpoint(
        job=job,
        relative=f'runs/{job}/training/segment/checkpoints/latest_full.pt',
        bytes=checkpoint.stat().st_size,
        mtime_ns=checkpoint.stat().st_mtime_ns,
        sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )


def test_expected_jobs_expand_frozen_four_by_five_matrix() -> None:
    matrix = MATRIX
    jobs = expected_jobs(matrix)
    assert len(jobs) == 20
    assert jobs[0] == BackupJob('SafetyPointGoal1-v0', 2027, 'point-goal-2027')
    assert jobs[-1] == BackupJob('Walker2d-v4', 2031, 'walker2d-2031')


def test_remote_arguments_and_records_are_strict() -> None:
    validate_remote_arguments('seeta-nmb2', str(REMOTE_ROOT))
    with pytest.raises(ValueError, match='unsafe SSH host'):
        validate_remote_arguments('server;touch-pwned', str(REMOTE_ROOT))
    with pytest.raises(ValueError, match='unsafe remote run root'):
        validate_remote_arguments('seeta-nmb2', '/root/safe/../formal')
    with pytest.raises(ValueError, match='unsafe remote run root'):
        validate_remote_arguments('seeta-nmb2', '/')
    validate_remote_python('/root/safe/env/bin/python')
    with pytest.raises(ValueError, match='unsafe remote Python interpreter'):
        validate_remote_python('/root/safe/../python')

    job = BackupJob('Ant-v4', 2027, 'ant-2027')
    payload = {
        'job': job.job,
        'relative': 'runs/ant-2027/training/segment/checkpoints/latest_full.pt',
        'bytes': 123,
        'mtime_ns': 456,
        'sha256': 'b' * 64,
    }
    records = parse_remote_inventory(json.dumps(payload), {job.job: job}, str(REMOTE_ROOT))
    assert records[job.job].bytes == 123
    with pytest.raises(ValueError, match='unexpected or duplicate'):
        parse_remote_inventory(
            f'{json.dumps(payload)}\n{json.dumps(payload)}',
            {job.job: job},
            str(REMOTE_ROOT),
        )
    payload['relative'] = 'runs/walker2d-2027/training/segment/checkpoints/latest_full.pt'
    with pytest.raises(ValueError, match='escaped its job'):
        parse_remote_inventory(json.dumps(payload), {job.job: job}, str(REMOTE_ROOT))


def test_checkpoint_validator_accepts_recovery_state_and_rejects_identity(
    tmp_path: Path,
) -> None:
    matrix = MATRIX
    checkpoint = tmp_path / 'latest_full.pt'
    _write_checkpoint(checkpoint, matrix)
    receipt = validate_checkpoint_backup(
        checkpoint,
        matrix_path=matrix,
        environment='Ant-v4',
        seed=2027,
        job='ant-2027',
        remote_run_root=REMOTE_ROOT,
        source_commit=SOURCE_COMMIT,
    )
    assert receipt['status'] == 'passed'
    assert receipt['checkpoint_env_steps'] == 50_000
    assert receipt['has_replay'] is True
    assert receipt['inspected_scientific_metrics'] is False

    bad = torch.load(checkpoint, map_location='cpu', weights_only=False)
    bad['extra']['environment'] = 'Walker2d-v4'
    torch.save(bad, checkpoint)
    with pytest.raises(ValueError, match='different formal run'):
        validate_checkpoint_backup(
            checkpoint,
            matrix_path=matrix,
            environment='Ant-v4',
            seed=2027,
            job='ant-2027',
            remote_run_root=REMOTE_ROOT,
            source_commit=SOURCE_COMMIT,
        )


def test_checkpoint_validator_rejects_replay_and_resume_escape(tmp_path: Path) -> None:
    matrix = MATRIX
    checkpoint = tmp_path / 'latest_full.pt'
    payload = _checkpoint_payload(matrix, 'Ant-v4', 2027, 50_000)
    payload['replay']['capacity'] = 1
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match='replay capacity changed'):
        validate_checkpoint_backup(
            checkpoint,
            matrix_path=matrix,
            environment='Ant-v4',
            seed=2027,
            job='ant-2027',
            remote_run_root=REMOTE_ROOT,
            source_commit=SOURCE_COMMIT,
        )

    payload = _checkpoint_payload(matrix, 'Ant-v4', 2027, 50_000)
    payload['extra']['resolved_config']['algo_cfgs'][
        'resume_from'
    ] = '/root/autodl-tmp/SAFE/other/latest_full.pt'
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match='escaped its formal run'):
        validate_checkpoint_backup(
            checkpoint,
            matrix_path=matrix,
            environment='Ant-v4',
            seed=2027,
            job='ant-2027',
            remote_run_root=REMOTE_ROOT,
            source_commit=SOURCE_COMMIT,
        )


def test_capacity_gate_protects_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        'scripts.star.backup_shapo_p47_checkpoints_to_host.shutil.disk_usage',
        lambda _path: type('Usage', (), {'free': 1_000})(),
    )
    assert ensure_capacity(tmp_path, 400, 500) == 1_000
    with pytest.raises(RuntimeError, match='insufficient host space'):
        ensure_capacity(tmp_path, 501, 500)


def test_transfer_is_validated_atomic_idempotent_and_monotonic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = MATRIX
    output_root = tmp_path / 'host-backups'
    remote_checkpoint = tmp_path / 'remote/latest_full.pt'
    _write_checkpoint(remote_checkpoint, matrix, steps=50_000)
    job = BackupJob('Ant-v4', 2027, 'ant-2027')
    first_record = _record(job.job, remote_checkpoint)
    monkeypatch.setattr(
        'scripts.star.backup_shapo_p47_checkpoints_to_host.copy_remote_checkpoint',
        lambda _host, _root, _record, destination: destination.write_bytes(
            remote_checkpoint.read_bytes(),
        ),
    )
    monkeypatch.setattr(
        'scripts.star.backup_shapo_p47_checkpoints_to_host.scan_remote',
        lambda *_args, **_kwargs: {job.job: first_record},
    )
    monkeypatch.setattr(
        'scripts.star.backup_shapo_p47_checkpoints_to_host.shutil.disk_usage',
        lambda _path: type('Usage', (), {'free': 10_000_000})(),
    )
    first = transfer_checkpoint(
        job,
        first_record,
        host='seeta-nmb2',
        remote_run_root=str(REMOTE_ROOT),
        remote_python='/root/safe/env/bin/python',
        output_root=output_root,
        matrix_path=matrix,
        source_commit=SOURCE_COMMIT,
        reserve_bytes=1_000,
    )
    assert first['state'] == 'transferred'
    assert (output_root / job.job / 'backup_complete').is_file()

    second = transfer_checkpoint(
        job,
        first_record,
        host='seeta-nmb2',
        remote_run_root=str(REMOTE_ROOT),
        remote_python='/root/safe/env/bin/python',
        output_root=output_root,
        matrix_path=matrix,
        source_commit=SOURCE_COMMIT,
        reserve_bytes=1_000,
    )
    assert second['state'] == 'already_current'

    _write_checkpoint(remote_checkpoint, matrix, steps=100_000)
    second_record = _record(job.job, remote_checkpoint)
    monkeypatch.setattr(
        'scripts.star.backup_shapo_p47_checkpoints_to_host.scan_remote',
        lambda *_args, **_kwargs: {job.job: second_record},
    )
    advanced = transfer_checkpoint(
        job,
        second_record,
        host='seeta-nmb2',
        remote_run_root=str(REMOTE_ROOT),
        remote_python='/root/safe/env/bin/python',
        output_root=output_root,
        matrix_path=matrix,
        source_commit=SOURCE_COMMIT,
        reserve_bytes=1_000,
    )
    assert advanced['checkpoint_env_steps'] == 100_000
    receipt = json.loads(
        (output_root / job.job / 'backup_receipt.json').read_text(encoding='utf-8'),
    )
    assert receipt['checkpoint_env_steps'] == 100_000
    assert not list(output_root.glob('.*.previous'))


def test_interrupted_swap_restores_previous_valid_backup(tmp_path: Path) -> None:
    matrix = MATRIX
    output_root = tmp_path / 'backups'
    output_root.mkdir()
    job = BackupJob('Ant-v4', 2027, 'ant-2027')
    previous = output_root / f'.{job.job}.previous'
    checkpoint = previous / 'latest_full.pt'
    _write_checkpoint(checkpoint, matrix)
    validation = validate_checkpoint_backup(
        checkpoint,
        matrix_path=matrix,
        environment=job.environment,
        seed=job.seed,
        job=job.job,
        remote_run_root=REMOTE_ROOT,
        source_commit=SOURCE_COMMIT,
    )
    validation.update(
        {
            'backup_version': 'shapo-p47-host-checkpoint-backup-v1',
            'backed_up_at_utc': '2026-08-13T00:00:00+00:00',
        },
    )
    (previous / 'backup_receipt.json').write_text(json.dumps(validation), encoding='utf-8')
    (previous / 'backup_complete').touch()
    (output_root / f'.{job.job}.staging').mkdir()
    recover_interrupted_swap(
        job,
        output_root=output_root,
        matrix_path=matrix,
        remote_run_root=REMOTE_ROOT,
        source_commit=SOURCE_COMMIT,
    )
    assert (output_root / job.job / 'latest_full.pt').is_file()
    assert not previous.exists()
    assert not (output_root / f'.{job.job}.staging').exists()


def test_backup_once_waits_for_unstarted_jobs_without_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = MATRIX
    all_jobs = expected_jobs(matrix)
    job = next(item for item in all_jobs if item.job == 'ant-2027')
    checkpoint = tmp_path / 'remote/latest_full.pt'
    _write_checkpoint(checkpoint, matrix)
    record = _record(job.job, checkpoint)
    monkeypatch.setattr(
        'scripts.star.backup_shapo_p47_checkpoints_to_host.scan_remote',
        lambda *_args, **_kwargs: {job.job: record},
    )
    monkeypatch.setattr(
        'scripts.star.backup_shapo_p47_checkpoints_to_host.copy_remote_checkpoint',
        lambda _host, _root, _record, destination: destination.write_bytes(
            checkpoint.read_bytes(),
        ),
    )
    monkeypatch.setattr(
        'scripts.star.backup_shapo_p47_checkpoints_to_host.shutil.disk_usage',
        lambda _path: type('Usage', (), {'free': 10_000_000})(),
    )
    snapshot = backup_once(
        all_jobs,
        host='seeta-nmb2',
        remote_run_root=str(REMOTE_ROOT),
        remote_python='/root/safe/env/bin/python',
        output_root=tmp_path / 'backups',
        matrix_path=matrix,
        source_commit=SOURCE_COMMIT,
        reserve_bytes=0,
        matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
    )
    assert snapshot['state'] == 'watching'
    assert snapshot['host_backup_jobs'] == 1
    assert snapshot['waiting_jobs'] == 19
    assert snapshot['inspected_scientific_metrics'] is False


def test_matrix_fixture_remains_yaml_mapping() -> None:
    matrix = yaml.safe_load(MATRIX.read_text(encoding='utf-8'))
    assert matrix['checkpoint']['latest_full_frequency_steps'] == 50_000

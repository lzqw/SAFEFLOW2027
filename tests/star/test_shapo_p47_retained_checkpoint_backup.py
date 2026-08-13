# Copyright 2026 SAFEFLOW2027 Authors. All Rights Reserved.
"""Tests for private SHAPO preregistered-checkpoint retention."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from scripts.star import backup_shapo_p47_retained_checkpoints_to_host as retained_backup
from scripts.star.backup_shapo_p47_retained_checkpoints_to_host import (
    DEFAULT_MAX_FINAL_PATCH_BYTES,
    FileIdentity,
    RemoteArtifact,
    RetainedJob,
    backup_once,
    create_final_patch,
    ensure_capacity,
    expected_jobs,
    parse_remote_inventory,
    transfer_final_patch,
    transfer_lightweight,
)
from scripts.star.materialize_shapo_p47_config import materialize
from scripts.star.restore_shapo_p47_final_checkpoint import restore_final_checkpoint
from scripts.star.validate_shapo_p47_retained_checkpoint_backup import (
    validate_retained_checkpoint_backup,
)


SOURCE_COMMIT = 'a' * 40
REMOTE_ROOT = Path('/root/autodl-tmp/SAFE/shapo_p47_four_env/formal')
REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX = REPO_ROOT / 'configs/star/shapo_p47_four_env.yaml'


def _checkpoint_payload(
    environment: str,
    seed: int,
    steps: int,
    *,
    replay: bool,
    rng_marker: int = 0,
) -> dict:
    config = materialize(MATRIX, environment, smoke=False)['custom_cfgs']
    config['seed'] = seed
    config['env_id'] = environment
    replay_payload = None
    if replay:
        resident = min(steps, 400_000)
        replay_payload = {
            'version': 3,
            'capacity': 400_000,
            'obs_dim': 1,
            'act_dim': 1,
            'context_dim': 1,
            'size': resident,
            'pointer': steps % 400_000,
            'data': {
                name: torch.zeros(1).expand(resident)
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
        }
    return {
        'format': 'star-checkpoint',
        'version': 1,
        'model': {'actor.weight': torch.tensor([1.0])},
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
            'violations': 1,
            'cumulative_training_cost': 2.0,
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
        'rng': {
            'python': (rng_marker,),
            'numpy': (rng_marker,),
            'torch': torch.tensor([rng_marker]),
            'cuda': None,
        },
        'replay': replay_payload,
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
    *,
    environment: str = 'Ant-v4',
    seed: int = 2027,
    steps: int = 500_000,
    replay: bool = False,
    rng_marker: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(environment, seed, steps, replay=replay, rng_marker=rng_marker),
        path,
    )


def _identity(path: Path, relative: str) -> FileIdentity:
    return FileIdentity(
        relative=relative,
        bytes=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _job() -> RetainedJob:
    return RetainedJob('Ant-v4', 2027, 'ant-2027', (500_000, 2_000_000, 5_000_000), 5_000_000)


def test_expected_jobs_expand_exact_retention_contract() -> None:
    jobs = expected_jobs(MATRIX)
    assert len(jobs) == 20
    assert all(len(job.labels) == 5 for job in jobs)
    assert sum(len(job.labels) for job in jobs) == 100
    assert jobs[0].labels == (
        'milestone_1000000',
        'milestone_4000000',
        'milestone_10000000',
        'best_safe_return',
        'final_full',
    )


def test_inventory_parser_requires_same_segment_final_patch_base() -> None:
    job = _job()
    evidence = {
        'receipt': {
            'relative': 'runs/ant-2027/receipt.json',
            'bytes': 10,
            'mtime_ns': 202,
            'sha256': 'd' * 64,
        },
        'training_summary': {
            'relative': 'runs/ant-2027/training/segment/artifacts/training_summary.json',
            'bytes': 11,
            'mtime_ns': 203,
            'sha256': 'e' * 64,
        },
        'formal_validation': {
            'relative': 'runs/ant-2027/formal_validation.json',
            'bytes': 12,
            'mtime_ns': 204,
            'sha256': 'f' * 64,
        },
        'exit_code': {
            'relative': 'runs/ant-2027/exit_code.txt',
            'bytes': 2,
            'mtime_ns': 205,
            'sha256': '0' * 64,
        },
    }
    payload = {
        'job': job.job,
        'label': 'final_full',
        'relative': 'runs/ant-2027/training/segment/checkpoints/final_full.pt',
        'bytes': 100,
        'mtime_ns': 200,
        'sha256': 'b' * 64,
        'base': {
            'relative': 'runs/ant-2027/training/segment/checkpoints/latest_full.pt',
            'bytes': 101,
            'mtime_ns': 201,
            'sha256': 'c' * 64,
        },
        'completion_evidence': evidence,
    }
    records = parse_remote_inventory(json.dumps(payload), {job.job: job}, str(REMOTE_ROOT))
    assert records[(job.job, 'final_full')].base is not None
    assert records[(job.job, 'final_full')].completion_evidence is not None
    without_evidence = dict(payload)
    without_evidence.pop('completion_evidence')
    with pytest.raises(ValueError, match='lacks exact completion evidence'):
        parse_remote_inventory(json.dumps(without_evidence), {job.job: job}, str(REMOTE_ROOT))
    wrong_path = json.loads(json.dumps(payload))
    wrong_path['completion_evidence']['receipt'][
        'relative'
    ] = 'runs/ant-2027/training/segment/receipt.json'
    with pytest.raises(ValueError, match='completion evidence path changed'):
        parse_remote_inventory(json.dumps(wrong_path), {job.job: job}, str(REMOTE_ROOT))
    payload['base']['relative'] = 'runs/ant-2027/training/other/checkpoints/latest_full.pt'
    with pytest.raises(ValueError, match='does not share'):
        parse_remote_inventory(json.dumps(payload), {job.job: job}, str(REMOTE_ROOT))


def test_retained_validator_distinguishes_lightweight_and_full(tmp_path: Path) -> None:
    lightweight = tmp_path / 'milestone_500000.pt'
    _write_checkpoint(lightweight)
    receipt = validate_retained_checkpoint_backup(
        lightweight,
        matrix_path=MATRIX,
        environment='Ant-v4',
        seed=2027,
        job='ant-2027',
        label='milestone_500000',
        expected_steps={500_000},
        replay_required=False,
        remote_run_root=REMOTE_ROOT,
        source_commit=SOURCE_COMMIT,
    )
    assert receipt['status'] == 'passed'
    assert receipt['has_replay'] is False
    assert receipt['scientific_metric_values_used_for_decision'] is False
    assert receipt['scientific_metric_values_reported'] is False

    full = tmp_path / 'final_full.pt'
    _write_checkpoint(full, steps=5_000_000, replay=True)
    full_receipt = validate_retained_checkpoint_backup(
        full,
        matrix_path=MATRIX,
        environment='Ant-v4',
        seed=2027,
        job='ant-2027',
        label='final_full',
        expected_steps={5_000_000},
        replay_required=True,
        remote_run_root=REMOTE_ROOT,
        source_commit=SOURCE_COMMIT,
    )
    assert full_receipt['replay_size'] == 400_000
    with pytest.raises(ValueError, match='unexpectedly contains replay'):
        validate_retained_checkpoint_backup(
            full,
            matrix_path=MATRIX,
            environment='Ant-v4',
            seed=2027,
            job='ant-2027',
            label='milestone_500000',
            expected_steps={5_000_000},
            replay_required=False,
            remote_run_root=REMOTE_ROOT,
            source_commit=SOURCE_COMMIT,
        )


def test_lightweight_transfer_is_atomic_idempotent_and_best_is_monotonic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job()
    remote = tmp_path / 'remote/best_safe_return.pt'
    _write_checkpoint(remote)
    relative = 'runs/ant-2027/training/segment/checkpoints/best_safe_return.pt'
    record = RemoteArtifact(job.job, 'best_safe_return', _identity(remote, relative))
    monkeypatch.setattr(
        'scripts.star.backup_shapo_p47_retained_checkpoints_to_host.copy_remote_file',
        lambda _host, _root, _identity, destination: destination.write_bytes(remote.read_bytes()),
    )
    monkeypatch.setattr(
        'scripts.star.backup_shapo_p47_retained_checkpoints_to_host.scan_remote',
        lambda *_args, **_kwargs: {(job.job, 'best_safe_return'): record},
    )
    monkeypatch.setattr(
        'scripts.star.backup_shapo_p47_retained_checkpoints_to_host.shutil.disk_usage',
        lambda _path: type('Usage', (), {'free': 10_000_000})(),
    )
    output_root = tmp_path / 'retained'
    first = transfer_lightweight(
        job,
        record,
        host='seeta-nmb2',
        remote_run_root=str(REMOTE_ROOT),
        remote_python='/root/safe/env/bin/python',
        output_root=output_root,
        matrix_path=MATRIX,
        source_commit=SOURCE_COMMIT,
        reserve_bytes=0,
        matrix_sha256=hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
    )
    assert first['state'] == 'transferred'
    second = transfer_lightweight(
        job,
        record,
        host='seeta-nmb2',
        remote_run_root=str(REMOTE_ROOT),
        remote_python='/root/safe/env/bin/python',
        output_root=output_root,
        matrix_path=MATRIX,
        source_commit=SOURCE_COMMIT,
        reserve_bytes=0,
        matrix_sha256=hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
    )
    assert second['state'] == 'already_current'

    _write_checkpoint(remote, steps=2_000_000, rng_marker=1)
    advanced = RemoteArtifact(job.job, 'best_safe_return', _identity(remote, relative))
    monkeypatch.setattr(
        'scripts.star.backup_shapo_p47_retained_checkpoints_to_host.scan_remote',
        lambda *_args, **_kwargs: {(job.job, 'best_safe_return'): advanced},
    )
    result = transfer_lightweight(
        job,
        advanced,
        host='seeta-nmb2',
        remote_run_root=str(REMOTE_ROOT),
        remote_python='/root/safe/env/bin/python',
        output_root=output_root,
        matrix_path=MATRIX,
        source_commit=SOURCE_COMMIT,
        reserve_bytes=0,
        matrix_sha256=hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
    )
    assert result['checkpoint_env_steps'] == 2_000_000


def test_zstd_patch_reconstructs_final_checkpoint_byte_exactly(tmp_path: Path) -> None:
    zstd = Path('/home/lzqw/anaconda3/bin/zstd')
    if not zstd.is_file():
        pytest.skip('zstd unavailable')
    base = tmp_path / 'latest_full.pt'
    final = tmp_path / 'final_full.pt'
    _write_checkpoint(base, steps=5_000_000, replay=True, rng_marker=0)
    _write_checkpoint(final, steps=5_000_000, replay=True, rng_marker=1)
    identity = create_final_patch(
        base,
        final,
        tmp_path / 'final_full.pt.zstpatch',
        tmp_path / 'reconstructed.pt',
        zstd_bin=zstd,
    )
    assert identity['reconstruction_verified'] is True
    assert identity['final_checkpoint_sha256'] == hashlib.sha256(final.read_bytes()).hexdigest()
    assert (tmp_path / 'reconstructed.pt').read_bytes() == final.read_bytes()


def test_zstd_patch_fails_closed_above_storage_cap(tmp_path: Path) -> None:
    zstd = Path('/home/lzqw/anaconda3/bin/zstd')
    if not zstd.is_file():
        pytest.skip('zstd unavailable')
    base = tmp_path / 'latest_full.pt'
    final = tmp_path / 'final_full.pt'
    _write_checkpoint(base, steps=5_000_000, replay=True, rng_marker=0)
    _write_checkpoint(final, steps=5_000_000, replay=True, rng_marker=1)
    with pytest.raises(ValueError, match='storage cap'):
        create_final_patch(
            base,
            final,
            tmp_path / 'final_full.pt.zstpatch',
            tmp_path / 'reconstructed.pt',
            zstd_bin=zstd,
            max_patch_bytes=1,
        )
    assert DEFAULT_MAX_FINAL_PATCH_BYTES == 4 * 1024**2


def test_restore_command_rejects_overwrite_and_reconstructs_exactly(tmp_path: Path) -> None:
    zstd = Path('/home/lzqw/anaconda3/bin/zstd')
    if not zstd.is_file():
        pytest.skip('zstd unavailable')
    base = tmp_path / 'latest_full.pt'
    final = tmp_path / 'source-final_full.pt'
    patch = tmp_path / 'final_full.pt.zstpatch'
    reconstructed = tmp_path / 'probe-reconstructed.pt'
    output = tmp_path / 'restored/final_full.pt'
    receipt = tmp_path / 'backup_receipt.json'
    _write_checkpoint(base, steps=5_000_000, replay=True, rng_marker=0)
    _write_checkpoint(final, steps=5_000_000, replay=True, rng_marker=1)
    identity = create_final_patch(
        base,
        final,
        patch,
        reconstructed,
        zstd_bin=zstd,
    )
    receipt.write_text(
        json.dumps(
            {
                **identity,
                'backup_version': 'shapo-p47-retained-final-patch-v1',
                'status': 'passed',
                'job': 'ant-2027',
                'environment': 'Ant-v4',
                'training_seed': 2027,
                'checkpoint_env_steps': 5_000_000,
                'label': 'final_full',
                'storage_representation': 'zstd_patch_from_validated_latest_full',
                'base_checkpoint_bytes': base.stat().st_size,
                'base_checkpoint_sha256': hashlib.sha256(base.read_bytes()).hexdigest(),
                'scientific_result': False,
                'used_for_checkpoint_or_configuration_selection': False,
                'used_for_preregistered_checkpoint_retention': True,
                'scientific_metric_values_used_for_decision': False,
                'scientific_metric_values_reported': False,
                'checkpoint_counters_validated': True,
                'training_state_changed': False,
                'scientific_protocol_changed': False,
            },
        ),
        encoding='utf-8',
    )
    result = restore_final_checkpoint(
        base=base,
        patch=patch,
        receipt_path=receipt,
        output=output,
        zstd_bin=zstd,
    )
    assert result['sha256'] == hashlib.sha256(final.read_bytes()).hexdigest()
    assert output.read_bytes() == final.read_bytes()
    with pytest.raises(FileExistsError, match='refusing to overwrite'):
        restore_final_checkpoint(
            base=base,
            patch=patch,
            receipt_path=receipt,
            output=output,
            zstd_bin=zstd,
        )


def test_final_patch_waits_for_exact_recovery_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job()
    final = tmp_path / 'remote/final_full.pt'
    base = tmp_path / 'remote/latest_full.pt'
    _write_checkpoint(base, steps=5_000_000, replay=True, rng_marker=0)
    _write_checkpoint(final, steps=5_000_000, replay=True, rng_marker=1)
    record = RemoteArtifact(
        job.job,
        'final_full',
        _identity(final, 'runs/ant-2027/training/segment/checkpoints/final_full.pt'),
        _identity(base, 'runs/ant-2027/training/segment/checkpoints/latest_full.pt'),
    )
    result = transfer_final_patch(
        job,
        record,
        host='seeta-nmb2',
        remote_run_root=str(REMOTE_ROOT),
        remote_python='/root/safe/env/bin/python',
        output_root=tmp_path / 'retained',
        recovery_root=tmp_path / 'missing-recovery',
        matrix_path=MATRIX,
        source_commit=SOURCE_COMMIT,
        reserve_bytes=0,
        matrix_sha256=hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
        zstd_bin=Path('/home/lzqw/anaconda3/bin/zstd'),
    )
    assert result['state'] == 'waiting_for_matching_latest_full_backup'


def test_final_patch_retains_and_revalidates_exact_completion_capsule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zstd = Path('/home/lzqw/anaconda3/bin/zstd')
    if not zstd.is_file():
        pytest.skip('zstd unavailable')
    job = _job()
    remote = tmp_path / 'remote'
    base = remote / 'latest_full.pt'
    final = remote / 'final_full.pt'
    _write_checkpoint(base, steps=5_000_000, replay=True, rng_marker=0)
    _write_checkpoint(final, steps=5_000_000, replay=True, rng_marker=1)
    log_dir = REMOTE_ROOT / 'runs/ant-2027/training/segment'
    evidence_sources = {
        'receipt': remote / 'receipt.json',
        'training_summary': remote / 'training_summary.json',
        'formal_validation': remote / 'formal_validation.json',
        'exit_code': remote / 'exit_code.txt',
    }
    evidence_sources['receipt'].write_text(
        json.dumps(
            {
                'status': 'passed',
                'environment': 'Ant-v4',
                'seed': 2027,
                'steps': 5_000_000,
                'run_steps': 5_000_000,
                'source_commit': SOURCE_COMMIT,
                'source_dirty': False,
                'log_dir': str(log_dir),
            },
        ),
        encoding='utf-8',
    )
    evidence_sources['training_summary'].write_text(
        json.dumps(
            {
                'status': 'completed',
                'environment': 'Ant-v4',
                'seed': 2027,
                'env_steps': 5_000_000,
                'run_env_steps': 5_000_000,
                'engineering_hard_stop_reason': None,
            },
        ),
        encoding='utf-8',
    )
    evidence_sources['formal_validation'].write_text(
        json.dumps(
            {
                'status': 'passed',
                'environment': 'Ant-v4',
                'training_seed': 2027,
                'total_steps': 5_000_000,
                'source_commit': SOURCE_COMMIT,
                'matrix_sha256': hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
                'training_seed_is_statistical_unit': True,
            },
        ),
        encoding='utf-8',
    )
    evidence_sources['exit_code'].write_text('0\n', encoding='utf-8')
    relatives = {
        'receipt': 'runs/ant-2027/receipt.json',
        'training_summary': ('runs/ant-2027/training/segment/artifacts/training_summary.json'),
        'formal_validation': 'runs/ant-2027/formal_validation.json',
        'exit_code': 'runs/ant-2027/exit_code.txt',
    }
    evidence = {
        label: _identity(source, relatives[label]) for label, source in evidence_sources.items()
    }
    record = RemoteArtifact(
        job.job,
        'final_full',
        _identity(final, 'runs/ant-2027/training/segment/checkpoints/final_full.pt'),
        _identity(base, 'runs/ant-2027/training/segment/checkpoints/latest_full.pt'),
        evidence,
    )
    by_relative = {
        record.file.relative: final,
        **{identity.relative: evidence_sources[label] for label, identity in evidence.items()},
    }

    def fake_copy(
        _host: str,
        _root: str,
        identity: FileIdentity,
        destination: Path,
    ) -> None:
        destination.write_bytes(by_relative[identity.relative].read_bytes())

    monkeypatch.setattr(retained_backup, 'copy_remote_file', fake_copy)
    monkeypatch.setattr(retained_backup, 'matching_recovery_base', lambda *_a, **_k: (base, {}))
    monkeypatch.setattr(
        retained_backup,
        'scan_remote',
        lambda *_a, **_k: {(job.job, 'final_full'): record},
    )
    monkeypatch.setattr(
        retained_backup.shutil,
        'disk_usage',
        lambda _path: type('Usage', (), {'free': 2_000_000_000})(),
    )
    output_root = tmp_path / 'retained'
    recovery_base = tmp_path / 'recovery/ant-2027/latest_full.pt'
    recovery_base.parent.mkdir(parents=True)
    recovery_base.write_bytes(base.read_bytes())
    result = transfer_final_patch(
        job,
        record,
        host='seeta-nmb2',
        remote_run_root=str(REMOTE_ROOT),
        remote_python='/root/safe/env/bin/python',
        output_root=output_root,
        recovery_root=tmp_path / 'recovery',
        matrix_path=MATRIX,
        source_commit=SOURCE_COMMIT,
        reserve_bytes=0,
        matrix_sha256=hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
        zstd_bin=zstd,
    )
    assert result['state'] == 'transferred'
    capsule = output_root / 'ant-2027/final_full/completion_evidence'
    assert {path.name for path in capsule.iterdir()} == {
        'receipt.json',
        'training_summary.json',
        'formal_validation.json',
        'exit_code.txt',
    }
    repeated = transfer_final_patch(
        job,
        record,
        host='seeta-nmb2',
        remote_run_root=str(REMOTE_ROOT),
        remote_python='/root/safe/env/bin/python',
        output_root=output_root,
        recovery_root=tmp_path / 'recovery',
        matrix_path=MATRIX,
        source_commit=SOURCE_COMMIT,
        reserve_bytes=0,
        matrix_sha256=hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
        zstd_bin=zstd,
    )
    assert repeated['state'] == 'already_current'
    (capsule / 'exit_code.txt').write_text('1\n', encoding='utf-8')
    with pytest.raises(ValueError, match='completion evidence changed'):
        transfer_final_patch(
            job,
            record,
            host='seeta-nmb2',
            remote_run_root=str(REMOTE_ROOT),
            remote_python='/root/safe/env/bin/python',
            output_root=output_root,
            recovery_root=tmp_path / 'recovery',
            matrix_path=MATRIX,
            source_commit=SOURCE_COMMIT,
            reserve_bytes=0,
            matrix_sha256=hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
            zstd_bin=zstd,
        )


def test_backup_once_waits_without_reading_scientific_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = expected_jobs(MATRIX)
    job = next(item for item in jobs if item.job == 'ant-2027')
    remote = tmp_path / 'remote/milestone_500000.pt'
    _write_checkpoint(remote)
    record = RemoteArtifact(
        job.job,
        'milestone_500000',
        _identity(remote, 'runs/ant-2027/training/segment/checkpoints/milestone_500000.pt'),
    )
    monkeypatch.setattr(
        'scripts.star.backup_shapo_p47_retained_checkpoints_to_host.scan_remote',
        lambda *_args, **_kwargs: {(job.job, 'milestone_500000'): record},
    )
    monkeypatch.setattr(
        'scripts.star.backup_shapo_p47_retained_checkpoints_to_host.copy_remote_file',
        lambda _host, _root, _identity, destination: destination.write_bytes(remote.read_bytes()),
    )
    monkeypatch.setattr(
        'scripts.star.backup_shapo_p47_retained_checkpoints_to_host.shutil.disk_usage',
        lambda _path: type('Usage', (), {'free': 100_000_000})(),
    )
    snapshot = backup_once(
        jobs,
        host='seeta-nmb2',
        remote_run_root=str(REMOTE_ROOT),
        remote_python='/root/safe/env/bin/python',
        output_root=tmp_path / 'retained',
        recovery_root=tmp_path / 'recovery',
        matrix_path=MATRIX,
        source_commit=SOURCE_COMMIT,
        reserve_bytes=0,
        matrix_sha256=hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
        zstd_bin=Path('/home/lzqw/anaconda3/bin/zstd'),
    )
    assert snapshot['state'] == 'watching'
    assert snapshot['total_retained_artifacts'] == 100
    assert snapshot['host_retained_artifacts'] == 1
    assert snapshot['waiting_artifacts'] == 99
    assert snapshot['scientific_metric_values_used_for_decision'] is False
    assert snapshot['scientific_metric_values_reported'] is False


def test_capacity_gate_protects_private_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        'scripts.star.backup_shapo_p47_retained_checkpoints_to_host.shutil.disk_usage',
        lambda _path: type('Usage', (), {'free': 1_000})(),
    )
    assert ensure_capacity(tmp_path, 400, 500) == 1_000
    with pytest.raises(RuntimeError, match='insufficient host space'):
        ensure_capacity(tmp_path, 501, 500)

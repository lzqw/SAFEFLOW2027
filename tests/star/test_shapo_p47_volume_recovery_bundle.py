# Copyright 2026 SAFEFLOW2027 Authors. All Rights Reserved.
"""Tests for exact-layout recovery after total SHAPO server-volume loss."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.star import build_shapo_p47_volume_recovery_bundle as recovery
from scripts.star.backup_shapo_p47_checkpoints_to_host import BackupJob
from scripts.star.backup_shapo_p47_retained_checkpoints_to_host import (
    RetainedJob,
    create_final_patch,
)


SOURCE_COMMIT = 'a' * 40
REMOTE_ROOT = Path('/root/autodl-tmp/SAFE/shapo_p47_four_env/formal')
REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX = REPO_ROOT / 'configs/star/shapo_p47_four_env.yaml'
AUDITOR = REPO_ROOT / 'scripts/star/audit_shapo_p47_milestone.py'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(job: str, *, relative: str, step: int, path: Path) -> dict[str, object]:
    return {
        'status': 'passed',
        'scientific_result': False,
        'used_for_checkpoint_or_configuration_selection': False,
        'training_state_changed': False,
        'scientific_protocol_changed': False,
        'job': job,
        'environment': 'Ant-v4',
        'training_seed': 2027,
        'checkpoint_env_steps': step,
        'checkpoint_bytes': path.stat().st_size,
        'checkpoint_sha256': _sha256(path),
        'matrix_sha256': _sha256(MATRIX),
        'source_commit': SOURCE_COMMIT,
        'remote_run_root': str(REMOTE_ROOT),
        'remote_relative_path': relative,
    }


def _compact_name(label: str, milestone: int) -> tuple[str, str]:
    direct = {
        'materialization_receipt': ('provenance/materialize.json', 'materialize.json'),
        'resolved_config': (
            'provenance/resolved_formal_config.yaml',
            'resolved_formal_config.yaml',
        ),
        'source_commit': ('provenance/source_commit.txt', 'source_commit.txt'),
    }
    if label in direct:
        return direct[label]
    names = {
        'best_safe_return_selection': 'best_safe_return_selection.json',
        'counterfactual': f'counterfactual_step_{milestone}.json',
        'evaluation': f'evaluation_step_{milestone}.json',
        'evaluation_curve': 'evaluation.csv',
        'run_manifest': 'run_manifest.json',
        'trajectory': f'trajectory_step_{milestone}.csv',
        'warmup_status': 'warmup_status.json',
    }
    return f'artifacts/{names[label]}', f'training/segment/artifacts/{names[label]}'


def _active_plan_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    best_step: int = 500_000,
) -> recovery.RecoveryPlan:
    job_name = 'ant-2027'
    job = RetainedJob('Ant-v4', 2027, job_name, (500_000, 2_000_000, 5_000_000), 5_000_000)
    recovery_job = BackupJob('Ant-v4', 2027, job_name)
    checkpoint_root = tmp_path / 'checkpoints'
    retained_root = tmp_path / 'retained'
    milestone_root = tmp_path / 'milestones'
    full = checkpoint_root / job_name / 'latest_full.pt'
    full.parent.mkdir(parents=True)
    full.write_bytes(b'validated-current-replay-state')
    current_relative = 'runs/ant-2027/training/current/checkpoints/latest_full.pt'
    anchor_absolute = REMOTE_ROOT / 'runs/ant-2027/training/original/checkpoints/latest_full.pt'
    full_receipt = _receipt(
        job_name,
        relative=current_relative,
        step=550_000,
        path=full,
    )

    seal = milestone_root / 'step_500000/ant-2027'
    compact = {}
    for label in sorted(recovery.EXPECTED_COMPACT_LABELS):
        sealed_relative, target_relative = _compact_name(label, 500_000)
        source = seal / sealed_relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f'{label}-sealed\n', encoding='utf-8')
        compact[label] = {
            'artifact': sealed_relative,
            'source_artifact': str(REMOTE_ROOT / 'runs/ant-2027' / target_relative),
            'bytes': source.stat().st_size,
            'sha256': _sha256(source),
        }
    (seal / 'milestone_audit.json').write_text('{}\n', encoding='utf-8')
    (seal / 'checksums.sha256').write_text('sealed-checksums\n', encoding='utf-8')
    (seal / 'milestone_audit_complete').touch()

    milestone_checkpoint = retained_root / job_name / 'milestone_500000/milestone_500000.pt'
    milestone_checkpoint.parent.mkdir(parents=True)
    milestone_checkpoint.write_bytes(b'milestone-state')
    milestone_relative = 'runs/ant-2027/training/segment/checkpoints/milestone_500000.pt'
    milestone_receipt = {
        **_receipt(
            job_name,
            relative=milestone_relative,
            step=500_000,
            path=milestone_checkpoint,
        ),
        'label': 'milestone_500000',
        'has_replay': False,
        'replay_size': 0,
        'replay_capacity': 400_000,
        'resume_from': str(anchor_absolute),
    }
    best = retained_root / job_name / 'best_safe_return/best_safe_return.pt'
    best.parent.mkdir(parents=True)
    best.write_bytes(b'best-safe-state')
    best_receipt = {
        **_receipt(
            job_name,
            relative='runs/ant-2027/training/segment/checkpoints/best_safe_return.pt',
            step=best_step,
            path=best,
        ),
        'label': 'best_safe_return',
        'has_replay': False,
        'replay_size': 0,
        'replay_capacity': 400_000,
        'resume_from': str(anchor_absolute),
    }
    audit = {
        'compact_artifacts': compact,
        'best_safe_return_selection': {'selected': {'milestone': 500_000}},
        'server_only_checkpoint': {
            'label': 'milestone_500000',
            'artifact': str(REMOTE_ROOT / milestone_relative),
            'bytes': milestone_checkpoint.stat().st_size,
            'sha256': _sha256(milestone_checkpoint),
            'env_steps': 500_000,
            'replay_included': False,
            'replay_size': 0,
            'replay_capacity': 400_000,
            'resume_from': str(anchor_absolute),
        },
    }

    monkeypatch.setattr(recovery, 'expected_recovery_jobs', lambda _matrix: [recovery_job])
    monkeypatch.setattr(recovery, 'expected_retained_jobs', lambda _matrix: [job])
    monkeypatch.setattr(
        recovery,
        'validate_existing_backup',
        lambda *_args, **_kwargs: full_receipt,
    )
    monkeypatch.setattr(
        recovery,
        '_checkpoint_metadata',
        lambda _path: {
            'env_steps': 550_000,
            'completed_milestones': [500_000],
            'completed_checkpoint_milestones': [0, 500_000],
            'selected_milestone': 500_000,
            'resume_from': str(anchor_absolute),
        },
    )
    monkeypatch.setattr(
        recovery,
        'validate_sealed_output',
        lambda *_args, **_kwargs: audit,
    )

    def fake_retained(
        _job: RetainedJob,
        label: str,
        _output: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        return milestone_receipt if label == 'milestone_500000' else best_receipt

    monkeypatch.setattr(recovery, 'validate_existing_lightweight', fake_retained)
    return recovery.build_recovery_plan(
        checkpoint_root=checkpoint_root,
        retained_root=retained_root,
        milestone_root=milestone_root,
        matrix_path=MATRIX,
        source_commit=SOURCE_COMMIT,
        remote_run_root=REMOTE_ROOT,
        audit_script=AUDITOR,
        zstd_bin=tmp_path / 'unused-zstd',
    )


def test_plan_restores_history_resume_anchor_and_best_without_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _active_plan_fixture(tmp_path, monkeypatch)
    manifest = plan.manifest
    assert manifest['status'] == 'passed'
    assert manifest['jobs_recoverable'] == 1
    assert manifest['jobs'][0]['completed_milestones'] == [500_000]
    assert manifest['scientific_metric_values_inspected'] is False
    assert manifest['used_for_checkpoint_or_configuration_selection'] is False
    serialized = json.dumps(manifest)
    assert str(tmp_path) not in serialized
    destinations = {item.destination.as_posix(): item for item in plan.files}
    current = 'runs/ant-2027/training/current/checkpoints/latest_full.pt'
    anchor = 'runs/ant-2027/training/original/checkpoints/latest_full.pt'
    assert destinations[current].sha256 == destinations[anchor].sha256
    assert 'validated_current_payload_resume_path_anchor' in destinations[anchor].roles
    assert 'runs/ant-2027/training/segment/checkpoints/milestone_500000.pt' in destinations
    assert 'runs/ant-2027/training/segment/checkpoints/best_safe_return.pt' in destinations
    assert 'milestone-audits-fca7eca/step_500000/ant-2027/milestone_audit.json' in destinations


def test_bundle_materialization_is_atomic_private_verified_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _active_plan_fixture(tmp_path, monkeypatch)
    bundle = tmp_path / 'bundle'
    result = recovery.materialize_bundle(plan, bundle)
    assert result['status'] == 'passed'
    assert result['jobs_recoverable'] == 1
    current = bundle / 'runs/ant-2027/training/current/checkpoints/latest_full.pt'
    anchor = bundle / 'runs/ant-2027/training/original/checkpoints/latest_full.pt'
    assert current.stat().st_ino == anchor.stat().st_ino
    assert bundle.stat().st_mode & 0o077 == 0
    recovery.verify_bundle(
        bundle,
        matrix_path=MATRIX,
        source_commit=SOURCE_COMMIT,
        remote_run_root=REMOTE_ROOT,
    )
    with pytest.raises(FileExistsError, match='refusing to overwrite'):
        recovery.materialize_bundle(plan, bundle)

    evaluation = next(bundle.glob('runs/ant-2027/training/**/evaluation_step_500000.json'))
    evaluation.write_text('tampered\n', encoding='utf-8')
    with pytest.raises(ValueError, match='bundle file changed'):
        recovery.verify_bundle(
            bundle,
            matrix_path=MATRIX,
            source_commit=SOURCE_COMMIT,
            remote_run_root=REMOTE_ROOT,
        )


def test_verify_rejects_wrong_target_path_and_unmanifested_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _active_plan_fixture(tmp_path, monkeypatch)
    bundle = tmp_path / 'bundle'
    recovery.materialize_bundle(plan, bundle)
    with pytest.raises(ValueError, match="'target_remote_run_root' changed"):
        recovery.verify_bundle(
            bundle,
            matrix_path=MATRIX,
            source_commit=SOURCE_COMMIT,
            remote_run_root=Path('/root/autodl-tmp/SAFE/wrong/formal'),
        )
    (bundle / 'unexpected').write_text('not sealed\n', encoding='utf-8')
    with pytest.raises(ValueError, match='coverage changed'):
        recovery.verify_bundle(
            bundle,
            matrix_path=MATRIX,
            source_commit=SOURCE_COMMIT,
            remote_run_root=REMOTE_ROOT,
        )


def test_plan_rejects_nonprefix_history_and_stale_best_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = RetainedJob('Ant-v4', 2027, 'ant-2027', (500_000, 2_000_000, 5_000_000), 5_000_000)
    with pytest.raises(ValueError, match='not a frozen prefix'):
        recovery._validate_progress_metadata(  # pylint: disable=protected-access
            {
                'env_steps': 2_050_000,
                'completed_milestones': [2_000_000],
                'completed_checkpoint_milestones': [0, 2_000_000],
                'selected_milestone': 2_000_000,
            },
            job=job,
        )
    with pytest.raises(ValueError, match='stale'):
        _active_plan_fixture(tmp_path / 'stale', monkeypatch, best_step=2_000_000)


def _minimal_manifest(item: recovery.PlannedFile) -> dict[str, object]:
    return {
        'version': recovery.VERSION,
        'status': 'passed',
        'target_remote_run_root': str(REMOTE_ROOT),
        'matrix_sha256': _sha256(MATRIX),
        'source_commit': SOURCE_COMMIT,
        'scientific_result': False,
        'scientific_metric_values_inspected': False,
        'scientific_metric_values_reported': False,
        'used_for_checkpoint_or_configuration_selection': False,
        'training_state_changed': False,
        'scientific_protocol_changed': False,
        'jobs_recoverable': 1,
        'file_count': 1,
        'installed_bytes': item.byte_count,
        'unique_payload_bytes': item.byte_count,
        'files': [
            {
                'destination': item.destination.as_posix(),
                'bytes': item.byte_count,
                'sha256': item.sha256,
                'roles': list(item.roles),
            },
        ],
    }


def test_materializer_reconstructs_deferred_final_patch_byte_exactly(tmp_path: Path) -> None:
    zstd = Path('/home/lzqw/anaconda3/bin/zstd')
    if not zstd.is_file():
        pytest.skip('zstd unavailable')
    base = tmp_path / 'latest_full.pt'
    final = tmp_path / 'source-final_full.pt'
    patch = tmp_path / 'final_full.pt.zstpatch'
    probe = tmp_path / 'probe.pt'
    receipt = tmp_path / 'backup_receipt.json'
    base.write_bytes(b'base-checkpoint-' * 100)
    final.write_bytes(b'final-checkpoint-' * 100)
    identity = create_final_patch(base, final, patch, probe, zstd_bin=zstd)
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
                'base_checkpoint_sha256': _sha256(base),
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
    item = recovery.PlannedFile(
        source=None,
        destination=Path('runs/ant-2027/training/final/checkpoints/final_full.pt'),
        byte_count=final.stat().st_size,
        sha256=_sha256(final),
        roles=('reconstructed_byte_exact_final_full',),
        final_patch=recovery.FinalPatchRecipe(base, patch, receipt, zstd),
    )
    plan = recovery.RecoveryPlan(files=(item,), manifest=_minimal_manifest(item))
    bundle = tmp_path / 'final-bundle'
    result = recovery.materialize_bundle(plan, bundle)
    restored = bundle / item.destination
    assert result['status'] == 'passed'
    assert restored.read_bytes() == final.read_bytes()


def test_completed_capsule_is_restored_to_original_strict_validation_paths(
    tmp_path: Path,
) -> None:
    retained_root = tmp_path / 'retained'
    capsule = retained_root / 'ant-2027/final_full/completion_evidence'
    capsule.mkdir(parents=True)
    destinations = {
        'receipt': 'runs/ant-2027/receipt.json',
        'training_summary': ('runs/ant-2027/training/final/artifacts/training_summary.json'),
        'formal_validation': 'runs/ant-2027/formal_validation.json',
        'exit_code': 'runs/ant-2027/exit_code.txt',
    }
    identities = {}
    for label, name in recovery.COMPLETION_EVIDENCE_NAMES.items():
        source = capsule / name
        source.write_text(f'{label}\n', encoding='utf-8')
        identities[label] = {
            'remote_relative_path': destinations[label],
            'bytes': source.stat().st_size,
            'sha256': _sha256(source),
        }
    files: dict[str, recovery.PlannedFile] = {}
    recovery._plan_completion_evidence(  # pylint: disable=protected-access
        files,
        job=RetainedJob(
            'Ant-v4',
            2027,
            'ant-2027',
            (500_000, 2_000_000, 5_000_000),
            5_000_000,
        ),
        retained_root=retained_root,
        receipt={'completion_evidence': identities},
    )
    assert set(files) == set(destinations.values())
    assert files[destinations['receipt']].roles == ('completed_run_receipt',)

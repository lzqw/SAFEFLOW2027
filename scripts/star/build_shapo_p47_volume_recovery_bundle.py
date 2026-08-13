#!/usr/bin/env python3
# pylint: disable=too-many-lines,wrong-import-position
"""Build a fail-closed SHAPO bundle for loss of the entire server data volume.

The ordinary recovery watcher assumes ``/root/autodl-tmp`` survived a server
restart.  This tool covers the stronger failure mode in which that volume is
empty: it reconstructs the exact formal-run layout from private host recovery
checkpoints, preregistered lightweight checkpoints, and sealed compact
milestone evidence.  It never contacts or mutates the server.

The bundle must be installed only at the exact recorded formal run root on a
new, empty volume.  The existing formal launcher then remains the sole owner of
checkpoint choice and process startup.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.star.backup_shapo_p47_checkpoints_to_host import BackupJob  # noqa: E402
from scripts.star.backup_shapo_p47_checkpoints_to_host import validate_existing_backup  # noqa: E402
from scripts.star.backup_shapo_p47_checkpoints_to_host import (  # noqa: E402
    expected_jobs as expected_recovery_jobs,
)
from scripts.star.backup_shapo_p47_retained_checkpoints_to_host import (  # noqa: E402
    COMPLETION_EVIDENCE_NAMES,
    DEFAULT_MAX_FINAL_PATCH_BYTES,
    RetainedJob,
)
from scripts.star.backup_shapo_p47_retained_checkpoints_to_host import (  # noqa: E402
    expected_jobs as expected_retained_jobs,
)
from scripts.star.backup_shapo_p47_retained_checkpoints_to_host import (  # noqa: E402
    validate_completion_evidence_relative,
    validate_existing_final_patch,
    validate_existing_lightweight,
)
from scripts.star.restore_shapo_p47_final_checkpoint import restore_final_checkpoint  # noqa: E402
from scripts.star.validate_shapo_p47_checkpoint_backup import sha256_file  # noqa: E402
from scripts.star.watch_shapo_p47_milestones import (  # noqa: E402
    MilestoneJob,
    milestone_audit_identity,
    validate_sealed_output,
)


VERSION = 'shapo-p47-volume-recovery-bundle-v1'
MANIFEST_NAME = 'volume_recovery_manifest.json'
COMPLETE_NAME = 'volume_recovery_bundle_complete'
DEFAULT_MILESTONE_RELATIVE_ROOT = Path('milestone-audits-fca7eca')
DEFAULT_ZSTD_BIN = Path('/usr/bin/zstd')
EXPECTED_RECEIPT_FIELDS = {
    'status': 'passed',
    'scientific_result': False,
    'used_for_checkpoint_or_configuration_selection': False,
    'training_state_changed': False,
    'scientific_protocol_changed': False,
}
EXPECTED_COMPACT_LABELS = {
    'best_safe_return_selection',
    'counterfactual',
    'evaluation',
    'evaluation_curve',
    'materialization_receipt',
    'resolved_config',
    'run_manifest',
    'source_commit',
    'trajectory',
    'warmup_status',
}


@dataclass(frozen=True)
class FinalPatchRecipe:
    """Inputs for one byte-exact deferred final-checkpoint reconstruction."""

    base: Path
    patch: Path
    receipt: Path
    zstd_bin: Path


@dataclass(frozen=True)
class PlannedFile:
    """One validated host file and its exact target-relative destination."""

    source: Path | None
    destination: Path
    byte_count: int
    sha256: str
    roles: tuple[str, ...]
    final_patch: FinalPatchRecipe | None = None


@dataclass(frozen=True)
class RecoveryPlan:
    """Private sources plus the publishable, metric-free recovery manifest."""

    files: tuple[PlannedFile, ...]
    manifest: dict[str, Any]


def utc_now() -> str:
    """Return an aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    """Load one strict JSON mapping from a regular file."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'{label} is not a regular file: {path}')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{label} is not a mapping: {path}')
    return payload


def _safe_relative(value: Any, *, label: str) -> Path:
    """Require one nonempty target-relative path without traversal."""
    candidate = Path(str(value))
    if not str(candidate) or candidate.is_absolute() or '..' in candidate.parts:
        raise ValueError(f'{label} is not a safe relative path: {value!r}')
    return candidate


def _relative_to_run(value: Any, *, run_root: Path, label: str) -> Path:
    """Map one recorded absolute formal path to its target-relative path."""
    candidate = Path(str(value))
    if not candidate.is_absolute() or '..' in candidate.parts:
        raise ValueError(f'{label} is not an absolute path without traversal')
    try:
        relative = candidate.relative_to(run_root)
    except ValueError as error:
        raise ValueError(f'{label} escaped the exact formal run root') from error
    return _safe_relative(relative, label=label)


def _require_job_destination(relative: Path, *, job: str, filename: str) -> None:
    """Constrain a restored checkpoint to one exact matrix cell."""
    prefix = Path('runs') / job / 'training'
    try:
        remainder = relative.relative_to(prefix)
    except ValueError as error:
        raise ValueError(f'checkpoint destination escaped its job: {relative}') from error
    if len(remainder.parts) < 3 or remainder.parts[-2:] != ('checkpoints', filename):
        raise ValueError(f'checkpoint destination has the wrong layout: {relative}')


def _require_compact_destination(
    relative: Path,
    *,
    job: str,
    label: str,
    milestone: int,
) -> None:
    """Constrain one compact source snapshot to its exact same-job location."""
    direct = {
        'materialization_receipt': 'materialize.json',
        'resolved_config': 'resolved_formal_config.yaml',
        'source_commit': 'source_commit.txt',
    }
    artifact_names = {
        'best_safe_return_selection': 'best_safe_return_selection.json',
        'counterfactual': f'counterfactual_step_{milestone}.json',
        'evaluation': f'evaluation_step_{milestone}.json',
        'evaluation_curve': 'evaluation.csv',
        'run_manifest': 'run_manifest.json',
        'trajectory': f'trajectory_step_{milestone}.csv',
        'warmup_status': 'warmup_status.json',
    }
    job_root = Path('runs') / job
    if label in direct:
        if relative != job_root / direct[label]:
            raise ValueError(f'compact provenance destination changed: {relative}')
        return
    expected_name = artifact_names.get(label)
    try:
        remainder = relative.relative_to(job_root / 'training')
    except ValueError as error:
        raise ValueError(f'compact artifact escaped its job: {relative}') from error
    if (
        expected_name is None
        or len(remainder.parts) < 3
        or remainder.parts[-2:] != ('artifacts', expected_name)
    ):
        raise ValueError(f'compact artifact destination changed: {relative}')


def _require_receipt_identity(
    receipt: dict[str, Any],
    *,
    job: str,
    run_root: Path,
    matrix_sha256: str,
    source_commit: str,
) -> None:
    """Bind a host receipt to this exact formal run and frozen source."""
    for key, expected in EXPECTED_RECEIPT_FIELDS.items():
        if receipt.get(key) != expected:
            raise ValueError(f'host receipt field {key!r} changed for {job}')
    expected = {
        'job': job,
        'remote_run_root': str(run_root),
        'matrix_sha256': matrix_sha256,
        'source_commit': source_commit,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f'host receipt identity {key!r} changed for {job}')


def _regular_identity(path: Path, *, allow_empty: bool = False) -> tuple[int, str]:
    """Hash one stable regular file and reject symbolic links or mutation."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'recovery source is not a regular file: {path}')
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError(f'recovery source changed while hashing: {path}')
    if after.st_size <= 0 and not allow_empty:
        raise ValueError(f'recovery source is empty: {path}')
    return after.st_size, digest


def _add_file(
    files: dict[str, PlannedFile],
    *,
    source: Path,
    destination: Path,
    role: str,
    replace: bool = False,
    allow_empty: bool = False,
) -> None:
    """Add one checked mapping, optionally replacing an older compact snapshot."""
    relative = _safe_relative(destination, label='bundle destination')
    byte_count, digest = _regular_identity(source, allow_empty=allow_empty)
    key = relative.as_posix()
    existing = files.get(key)
    if existing is None:
        files[key] = PlannedFile(source, relative, byte_count, digest, (role,))
        return
    if existing.sha256 == digest and existing.byte_count == byte_count:
        roles = tuple(sorted(set(existing.roles) | {role}))
        files[key] = PlannedFile(
            existing.source,
            relative,
            byte_count,
            digest,
            roles,
            existing.final_patch,
        )
        return
    if not replace:
        raise ValueError(f'conflicting recovery payloads target {relative}')
    files[key] = PlannedFile(source, relative, byte_count, digest, (role,))


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Read only recovery-control metadata, never scientific metric values."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    try:
        extra = payload['extra']
        selected = extra.get('best_safe_checkpoint_selection')
        return {
            'env_steps': int(payload['counters']['env_steps']),
            'completed_milestones': [int(value) for value in extra['completed_milestones']],
            'completed_checkpoint_milestones': [
                int(value) for value in extra['completed_checkpoint_milestones']
            ],
            'selected_milestone': (
                int(selected['milestone']) if isinstance(selected, dict) else None
            ),
            'resume_from': str(extra['resolved_config']['algo_cfgs'].get('resume_from') or ''),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'checkpoint recovery-control metadata is invalid: {path}') from error
    finally:
        del payload


def _validate_progress_metadata(
    metadata: dict[str, Any],
    *,
    job: RetainedJob,
) -> tuple[int, ...]:
    """Require completed milestones to be an exact prefix of the frozen schedule."""
    completed = metadata['completed_milestones']
    checkpointed = metadata['completed_checkpoint_milestones']
    if completed != sorted(set(completed)) or checkpointed != sorted(set(checkpointed)):
        raise ValueError(f'checkpoint milestone sets are not canonical for {job.job}')
    locked = tuple(job.milestones)
    completed_tuple = tuple(completed)
    if completed_tuple != locked[: len(completed_tuple)]:
        raise ValueError(f'completed milestones are not a frozen prefix for {job.job}')
    locked_checkpointed = tuple(value for value in checkpointed if value != 0)
    if any(value not in locked for value in checkpointed if value != 0):
        raise ValueError(f'checkpoint contains an unregistered milestone for {job.job}')
    if locked_checkpointed != completed_tuple:
        raise ValueError(f'evaluation/checkpoint milestone history diverged for {job.job}')
    step = int(metadata['env_steps'])
    if any(value > step for value in completed_tuple):
        raise ValueError(f'checkpoint claims a future completed milestone for {job.job}')
    selected = metadata['selected_milestone']
    if completed_tuple and selected not in set(completed_tuple):
        raise ValueError(f'best-safe selection is outside completed milestones for {job.job}')
    if not completed_tuple and selected is not None:
        raise ValueError(f'best-safe selection exists before a completed milestone for {job.job}')
    return completed_tuple


def _validate_retained(
    job: RetainedJob,
    label: str,
    retained_root: Path,
    *,
    matrix_path: Path,
    run_root: Path,
    source_commit: str,
    matrix_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    """Deeply revalidate one host-retained lightweight checkpoint and receipt."""
    output = retained_root / job.job / label
    receipt = validate_existing_lightweight(
        job,
        label,
        output,
        matrix_path=matrix_path,
        remote_run_root=run_root,
        source_commit=source_commit,
    )
    if receipt is None:
        raise ValueError(f'missing retained checkpoint for {job.job}/{label}')
    _require_receipt_identity(
        receipt,
        job=job.job,
        run_root=run_root,
        matrix_sha256=matrix_sha256,
        source_commit=source_commit,
    )
    if receipt.get('label') != label:
        raise ValueError(f'retained checkpoint label changed for {job.job}/{label}')
    checkpoint = output / f'{label}.pt'
    return checkpoint, receipt


def _plan_final_checkpoint(
    files: dict[str, PlannedFile],
    job: RetainedJob,
    *,
    checkpoint_root: Path,
    retained_root: Path,
    matrix_path: Path,
    source_commit: str,
    zstd_bin: Path,
    destination: Path,
) -> dict[str, Any]:
    """Validate and defer one byte-exact final-full patch reconstruction."""
    output = retained_root / job.job / 'final_full'
    receipt = validate_existing_final_patch(
        job,
        output,
        recovery_root=checkpoint_root,
        record=None,
        matrix_path=matrix_path,
        source_commit=source_commit,
        max_final_patch_bytes=DEFAULT_MAX_FINAL_PATCH_BYTES,
    )
    if receipt is None:
        raise ValueError(f'missing final-full host patch for completed job {job.job}')
    relative = _safe_relative(destination, label=f'{job.job} final-full destination')
    key = relative.as_posix()
    if key in files:
        raise ValueError(f'conflicting recovery payloads target {relative}')
    files[key] = PlannedFile(
        source=None,
        destination=relative,
        byte_count=int(receipt['final_checkpoint_bytes']),
        sha256=str(receipt['final_checkpoint_sha256']),
        roles=('reconstructed_byte_exact_final_full',),
        final_patch=FinalPatchRecipe(
            base=checkpoint_root / job.job / 'latest_full.pt',
            patch=output / 'final_full.pt.zstpatch',
            receipt=output / 'backup_receipt.json',
            zstd_bin=zstd_bin,
        ),
    )
    return receipt


def _plan_completion_evidence(
    files: dict[str, PlannedFile],
    *,
    job: RetainedJob,
    retained_root: Path,
    receipt: dict[str, Any],
) -> None:
    """Restore the exact small final-run capsule required by strict validation."""
    identities = receipt.get('completion_evidence')
    if not isinstance(identities, dict) or set(identities) != set(COMPLETION_EVIDENCE_NAMES):
        raise ValueError(f'final-full receipt lacks completion evidence for {job.job}')
    evidence_root = retained_root / job.job / 'final_full/completion_evidence'
    for label, expected_name in COMPLETION_EVIDENCE_NAMES.items():
        identity = identities[label]
        if not isinstance(identity, dict):
            raise ValueError(f'completion evidence identity is invalid for {job.job}/{label}')
        source = evidence_root / expected_name
        destination = _safe_relative(
            identity.get('remote_relative_path'),
            label=f'{job.job}/{label} completion path',
        )
        validate_completion_evidence_relative(job.job, label, destination)
        byte_count, digest = _regular_identity(source)
        if byte_count != int(identity.get('bytes', -1)) or digest != identity.get('sha256'):
            raise ValueError(f'completion evidence changed for {job.job}/{label}')
        _add_file(
            files,
            source=source,
            destination=destination,
            role=f'completed_run_{label}',
            replace=True,
        )


def _restore_compact_milestone(
    files: dict[str, PlannedFile],
    *,
    seal: Path,
    audit: dict[str, Any],
    run_root: Path,
    milestone_relative_root: Path,
    job: RetainedJob,
    milestone: int,
) -> None:
    """Plan both raw historical artifacts and the immutable server audit seal."""
    compact = audit.get('compact_artifacts')
    if not isinstance(compact, dict) or set(compact) != EXPECTED_COMPACT_LABELS:
        raise ValueError(f'milestone seal compact-artifact contract changed: {seal}')
    for label, identity in compact.items():
        if not isinstance(identity, dict):
            raise ValueError(f'invalid compact identity {label!r}: {seal}')
        source = seal / _safe_relative(identity.get('artifact'), label='sealed artifact')
        destination = _relative_to_run(
            identity.get('source_artifact'),
            run_root=run_root,
            label=f'{job.job} milestone source artifact',
        )
        _require_compact_destination(
            destination,
            job=job.job,
            label=label,
            milestone=milestone,
        )
        byte_count, digest = _regular_identity(source)
        if byte_count != int(identity.get('bytes', -1)) or digest != identity.get('sha256'):
            raise ValueError(f'compact milestone identity changed: {source}')
        _add_file(
            files,
            source=source,
            destination=destination,
            role=f'raw_milestone_{milestone}_{label}',
            replace=True,
        )
    seal_destination = milestone_relative_root / f'step_{milestone}' / job.job
    for source in sorted(path for path in seal.rglob('*') if path.is_file()):
        _add_file(
            files,
            source=source,
            destination=seal_destination / source.relative_to(seal),
            role=f'sealed_milestone_{milestone}',
            allow_empty=source.name == 'milestone_audit_complete',
        )


def build_recovery_plan(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    *,
    checkpoint_root: Path,
    retained_root: Path,
    milestone_root: Path,
    matrix_path: Path,
    source_commit: str,
    remote_run_root: Path,
    audit_script: Path,
    milestone_relative_root: Path = DEFAULT_MILESTONE_RELATIVE_ROOT,
    zstd_bin: Path = DEFAULT_ZSTD_BIN,
) -> RecoveryPlan:
    """Validate every available host slot and assemble an exact-layout plan."""
    inputs = [checkpoint_root, retained_root, milestone_root, matrix_path, audit_script]
    if any(path.is_symlink() for path in inputs):
        raise ValueError('recovery inputs must not be symbolic links')
    if not checkpoint_root.is_dir() or not retained_root.is_dir() or not milestone_root.is_dir():
        raise ValueError('checkpoint, retained, and milestone roots must all exist')
    if not matrix_path.is_file() or not audit_script.is_file():
        raise ValueError('frozen matrix or milestone auditor is unavailable')
    if zstd_bin.exists() and (zstd_bin.is_symlink() or not os.access(zstd_bin, os.X_OK)):
        raise ValueError('Zstandard binary is unsafe or not executable')
    if not remote_run_root.is_absolute() or '..' in remote_run_root.parts:
        raise ValueError('remote run root must be one exact absolute path')
    milestone_relative_root = _safe_relative(
        milestone_relative_root,
        label='milestone relative root',
    )
    if len(source_commit) != 40 or any(value not in '0123456789abcdef' for value in source_commit):
        raise ValueError('source commit must be a full lowercase Git identity')

    matrix = yaml.safe_load(matrix_path.read_text(encoding='utf-8'))
    if not isinstance(matrix, dict):
        raise ValueError('frozen matrix is not a mapping')
    matrix_sha256 = sha256_file(matrix_path)
    recovery_jobs = {job.job: job for job in expected_recovery_jobs(matrix_path)}
    retained_jobs = {job.job: job for job in expected_retained_jobs(matrix_path)}
    if set(recovery_jobs) != set(retained_jobs):
        raise ValueError('recovery and retention matrix expansions disagree')
    present_jobs = sorted(
        path.name
        for path in checkpoint_root.iterdir()
        if path.is_dir() and not path.name.startswith('.')
    )
    if not present_jobs:
        raise ValueError('no host recovery checkpoints are available')
    unknown = sorted(set(present_jobs) - set(recovery_jobs))
    if unknown:
        raise ValueError(f'host checkpoint root contains unknown jobs: {unknown}')

    expected_audit_identity = milestone_audit_identity(
        matrix_path,
        source_commit,
        audit_script,
    )
    files: dict[str, PlannedFile] = {}
    job_rows = []
    for job_name in present_jobs:
        recovery_job: BackupJob = recovery_jobs[job_name]
        retained_job: RetainedJob = retained_jobs[job_name]
        host_checkpoint_dir = checkpoint_root / job_name
        receipt = validate_existing_backup(
            recovery_job,
            host_checkpoint_dir,
            matrix_path=matrix_path,
            remote_run_root=remote_run_root,
            source_commit=source_commit,
        )
        if receipt is None:
            raise ValueError(f'incomplete host recovery checkpoint for {job_name}')
        _require_receipt_identity(
            receipt,
            job=job_name,
            run_root=remote_run_root,
            matrix_sha256=matrix_sha256,
            source_commit=source_commit,
        )
        full_checkpoint = host_checkpoint_dir / 'latest_full.pt'
        metadata = _checkpoint_metadata(full_checkpoint)
        if int(receipt['checkpoint_env_steps']) != metadata['env_steps']:
            raise ValueError(f'checkpoint receipt counter changed for {job_name}')
        completed = _validate_progress_metadata(metadata, job=retained_job)

        current_relative = _safe_relative(
            receipt.get('remote_relative_path'),
            label=f'{job_name} latest-full remote path',
        )
        _require_job_destination(current_relative, job=job_name, filename='latest_full.pt')
        if metadata['env_steps'] == retained_job.total_steps:
            final_base = retained_root / job_name / 'final_full'
            if not final_base.exists():
                raise ValueError(f'missing final-full host patch for completed job {job_name}')
            final_base_receipt = _load_json(
                final_base / 'backup_receipt.json',
                label=f'{job_name} final-full receipt',
            )
            if final_base_receipt.get('base_remote_relative_path') != current_relative.as_posix():
                raise ValueError(f'final-full patch base path changed for {job_name}')
        _add_file(
            files,
            source=full_checkpoint,
            destination=current_relative,
            role='current_replay_bearing_latest_full',
        )
        resume_from = metadata['resume_from']
        resume_anchor = None
        if resume_from:
            resume_relative = _relative_to_run(
                resume_from,
                run_root=remote_run_root,
                label=f'{job_name} embedded resume path',
            )
            _require_job_destination(resume_relative, job=job_name, filename='latest_full.pt')
            _add_file(
                files,
                source=full_checkpoint,
                destination=resume_relative,
                role='validated_current_payload_resume_path_anchor',
            )
            resume_anchor = resume_relative.as_posix()

        latest_sealed_selection = None
        for milestone in completed:
            seal = milestone_root / f'step_{milestone}' / job_name
            milestone_job = MilestoneJob(
                environment=retained_job.environment,
                seed=retained_job.seed,
                milestone=milestone,
                container=remote_run_root / 'runs' / job_name,
                output=seal,
            )
            audit = validate_sealed_output(
                milestone_job,
                expected_identity=expected_audit_identity,
            )
            sealed_selection = audit.get('best_safe_return_selection', {}).get('selected', {})
            try:
                latest_sealed_selection = int(sealed_selection['milestone'])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'milestone seal selection is invalid: {seal}') from error
            _restore_compact_milestone(
                files,
                seal=seal,
                audit=audit,
                run_root=remote_run_root,
                milestone_relative_root=milestone_relative_root,
                job=retained_job,
                milestone=milestone,
            )
            label = f'milestone_{milestone}'
            checkpoint, retained_receipt = _validate_retained(
                retained_job,
                label,
                retained_root,
                matrix_path=matrix_path,
                run_root=remote_run_root,
                source_commit=source_commit,
                matrix_sha256=matrix_sha256,
            )
            server_checkpoint = audit.get('server_only_checkpoint')
            if not isinstance(server_checkpoint, dict):
                raise ValueError(f'missing server checkpoint identity in seal: {seal}')
            destination = _relative_to_run(
                server_checkpoint.get('artifact'),
                run_root=remote_run_root,
                label=f'{job_name}/{label} server checkpoint',
            )
            receipt_relative = _safe_relative(
                retained_receipt.get('remote_relative_path'),
                label=f'{job_name}/{label} retained path',
            )
            expected_checkpoint = {
                'label': label,
                'bytes': int(retained_receipt['checkpoint_bytes']),
                'sha256': retained_receipt['checkpoint_sha256'],
                'env_steps': int(retained_receipt['checkpoint_env_steps']),
                'replay_included': bool(retained_receipt['has_replay']),
                'replay_size': int(retained_receipt['replay_size']),
                'replay_capacity': int(retained_receipt['replay_capacity']),
                'resume_from': retained_receipt['resume_from'],
            }
            observed_checkpoint = {
                'label': server_checkpoint.get('label'),
                'bytes': int(server_checkpoint.get('bytes', -1)),
                'sha256': server_checkpoint.get('sha256'),
                'env_steps': int(server_checkpoint.get('env_steps', -1)),
                'replay_included': bool(server_checkpoint.get('replay_included')),
                'replay_size': int(server_checkpoint.get('replay_size', -1)),
                'replay_capacity': int(server_checkpoint.get('replay_capacity', -1)),
                'resume_from': server_checkpoint.get('resume_from'),
            }
            if destination != receipt_relative or observed_checkpoint != expected_checkpoint:
                raise ValueError(f'milestone seal/host checkpoint disagree for {job_name}/{label}')
            _require_job_destination(destination, job=job_name, filename=f'{label}.pt')
            _add_file(
                files,
                source=checkpoint,
                destination=destination,
                role=f'lightweight_{label}',
            )

        if completed:
            if latest_sealed_selection != metadata['selected_milestone']:
                raise ValueError(f'latest sealed selection is stale for {job_name}')
            best_checkpoint, best_receipt = _validate_retained(
                retained_job,
                'best_safe_return',
                retained_root,
                matrix_path=matrix_path,
                run_root=remote_run_root,
                source_commit=source_commit,
                matrix_sha256=matrix_sha256,
            )
            if int(best_receipt['checkpoint_env_steps']) != metadata['selected_milestone']:
                raise ValueError(f'best-safe host checkpoint is stale for {job_name}')
            best_destination = _safe_relative(
                best_receipt.get('remote_relative_path'),
                label=f'{job_name} best-safe retained path',
            )
            _require_job_destination(
                best_destination,
                job=job_name,
                filename='best_safe_return.pt',
            )
            _add_file(
                files,
                source=best_checkpoint,
                destination=best_destination,
                role='current_best_safe_return',
            )

        if metadata['env_steps'] == retained_job.total_steps:
            if not zstd_bin.is_file() or zstd_bin.is_symlink() or not os.access(zstd_bin, os.X_OK):
                raise ValueError('Zstandard binary is unavailable or not executable')
            final_slot = retained_root / retained_job.job / 'final_full'
            final_receipt_preview = _load_json(
                final_slot / 'backup_receipt.json',
                label=f'{job_name} final-full receipt',
            )
            final_destination = _safe_relative(
                final_receipt_preview.get('remote_relative_path'),
                label=f'{job_name} final-full retained path',
            )
            _require_job_destination(
                final_destination,
                job=job_name,
                filename='final_full.pt',
            )
            final_receipt = _plan_final_checkpoint(
                files,
                retained_job,
                checkpoint_root=checkpoint_root,
                retained_root=retained_root,
                matrix_path=matrix_path,
                source_commit=source_commit,
                zstd_bin=zstd_bin,
                destination=final_destination,
            )
            _require_receipt_identity(
                final_receipt,
                job=job_name,
                run_root=remote_run_root,
                matrix_sha256=matrix_sha256,
                source_commit=source_commit,
            )
            _plan_completion_evidence(
                files,
                job=retained_job,
                retained_root=retained_root,
                receipt=final_receipt,
            )

        job_rows.append(
            {
                'job': job_name,
                'environment': retained_job.environment,
                'training_seed': retained_job.seed,
                'checkpoint_env_steps': metadata['env_steps'],
                'total_steps': retained_job.total_steps,
                'completed_milestones': list(completed),
                'selected_milestone': metadata['selected_milestone'],
                'resume_anchor_relative_path': resume_anchor,
                'status': (
                    'completed_checkpoint_recovery_ready'
                    if metadata['env_steps'] == retained_job.total_steps
                    else 'active_recovery_ready'
                ),
            },
        )

    ordered_files = tuple(files[key] for key in sorted(files))
    installed_bytes = sum(item.byte_count for item in ordered_files)
    unique_payload_bytes = sum(
        byte_count
        for byte_count, _digest in {(item.byte_count, item.sha256) for item in ordered_files}
    )
    manifest = {
        'version': VERSION,
        'status': 'passed',
        'created_at_utc': utc_now(),
        'target_remote_run_root': str(remote_run_root),
        'requires_exact_target_path': True,
        'requires_new_empty_server_volume': True,
        'deployment_executed': False,
        'formal_launcher_remains_checkpoint_authority': True,
        'resume_path_anchor_semantics': (
            'The current semantically validated replay-bearing payload is also installed at '
            'its immutable embedded resume_from path. The anchor restores path provenance; it '
            'does not claim to reproduce the overwritten historical payload at that path.'
        ),
        'protocol_version': matrix['protocol_version'],
        'protocol_sha256': matrix['protocol_sha256'],
        'matrix_sha256': matrix_sha256,
        'frozen_method_canonical_sha256': matrix['frozen_method_canonical_sha256'],
        'source_commit': source_commit,
        'audit_script_sha256': expected_audit_identity['audit_script_sha256'],
        'strict_evaluation_validator_sha256': expected_audit_identity[
            'strict_evaluation_validator_sha256'
        ],
        'scientific_result': False,
        'scientific_metric_values_inspected': False,
        'scientific_metric_values_reported': False,
        'used_for_checkpoint_or_configuration_selection': False,
        'training_state_changed': False,
        'scientific_protocol_changed': False,
        'jobs_recoverable': len(job_rows),
        'jobs': job_rows,
        'file_count': len(ordered_files),
        'installed_bytes': installed_bytes,
        'unique_payload_bytes': unique_payload_bytes,
        'files': [
            {
                'destination': item.destination.as_posix(),
                'bytes': item.byte_count,
                'sha256': item.sha256,
                'roles': list(item.roles),
            }
            for item in ordered_files
        ],
    }
    return RecoveryPlan(files=ordered_files, manifest=manifest)


def _fsync_directory(path: Path) -> None:
    """Fsync one directory after an atomic recovery-bundle operation."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_or_link(
    item: PlannedFile,
    destination: Path,
    installed: dict[tuple[int, str], Path],
) -> None:
    """Copy each unique payload once and hard-link duplicate recovery anchors."""
    identity = (item.byte_count, item.sha256)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    prior = installed.get(identity)
    if prior is not None:
        os.link(prior, destination)
    elif item.source is not None:
        shutil.copy2(item.source, destination)
        installed[identity] = destination
    else:
        recipe = item.final_patch
        if recipe is None:
            raise ValueError(f'planned recovery file has no source: {item.destination}')
        restored = restore_final_checkpoint(
            base=recipe.base,
            patch=recipe.patch,
            receipt_path=recipe.receipt,
            output=destination,
            zstd_bin=recipe.zstd_bin,
        )
        if restored['sha256'] != item.sha256 or int(restored['bytes']) != item.byte_count:
            raise ValueError(f'final checkpoint reconstruction changed: {item.destination}')
        installed[identity] = destination
    os.chmod(destination, 0o600)
    if destination.stat().st_size != item.byte_count or sha256_file(destination) != item.sha256:
        raise ValueError(f'materialized recovery file changed: {destination}')


def materialize_bundle(plan: RecoveryPlan, output_root: Path) -> dict[str, Any]:
    """Atomically create an owner-private bundle at one new local path."""
    output_root = output_root.absolute()
    if output_root == Path('/') or '..' in output_root.parts:
        raise ValueError('bundle output path is unsafe')
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f'refusing to overwrite bundle output: {output_root}')
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f'.{output_root.name}.staging-', dir=output_root.parent),
    )
    os.chmod(staging, 0o700)
    try:
        installed: dict[tuple[int, str], Path] = {}
        for item in plan.files:
            destination = staging / item.destination
            _copy_or_link(
                item,
                destination,
                installed,
            )
        manifest_path = staging / MANIFEST_NAME
        manifest_path.write_text(
            f'{json.dumps(plan.manifest, indent=2, sort_keys=True, allow_nan=False)}\n',
            encoding='utf-8',
        )
        os.chmod(manifest_path, 0o600)
        with manifest_path.open('rb') as stream:
            os.fsync(stream.fileno())
        manifest_sha256 = sha256_file(manifest_path)
        complete = staging / COMPLETE_NAME
        complete.write_text(f'{manifest_sha256}  {MANIFEST_NAME}\n', encoding='utf-8')
        os.chmod(complete, 0o600)
        with complete.open('rb') as stream:
            os.fsync(stream.fileno())
        for directory in sorted(
            (path for path in staging.rglob('*') if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            os.chmod(directory, 0o700)
            _fsync_directory(directory)
        _fsync_directory(staging)
        os.replace(staging, output_root)
        _fsync_directory(output_root.parent)
    except Exception:
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return verify_bundle(
        output_root,
        matrix_path=None,
        source_commit=str(plan.manifest['source_commit']),
        remote_run_root=Path(str(plan.manifest['target_remote_run_root'])),
        expected_matrix_sha256=str(plan.manifest['matrix_sha256']),
    )


def verify_bundle(  # pylint: disable=too-many-locals
    bundle_root: Path,
    *,
    matrix_path: Path | None,
    source_commit: str,
    remote_run_root: Path,
    expected_matrix_sha256: str | None = None,
) -> dict[str, Any]:
    """Re-hash a complete bundle without reading or reporting scientific metrics."""
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise ValueError('recovery bundle is not a regular directory')
    manifest_path = bundle_root / MANIFEST_NAME
    complete = bundle_root / COMPLETE_NAME
    manifest = _load_json(manifest_path, label='volume recovery manifest')
    marker = complete.read_text(encoding='utf-8').strip().split()
    if marker != [sha256_file(manifest_path), MANIFEST_NAME]:
        raise ValueError('volume recovery completion marker changed')
    matrix_sha256 = (
        sha256_file(matrix_path) if matrix_path is not None else str(expected_matrix_sha256 or '')
    )
    expected = {
        'version': VERSION,
        'status': 'passed',
        'target_remote_run_root': str(remote_run_root),
        'matrix_sha256': matrix_sha256,
        'source_commit': source_commit,
        'scientific_result': False,
        'scientific_metric_values_inspected': False,
        'scientific_metric_values_reported': False,
        'used_for_checkpoint_or_configuration_selection': False,
        'training_state_changed': False,
        'scientific_protocol_changed': False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'volume recovery manifest field {key!r} changed')
    entries = manifest.get('files')
    if not isinstance(entries, list) or len(entries) != int(manifest.get('file_count', -1)):
        raise ValueError('volume recovery file manifest is invalid')
    expected_files = {MANIFEST_NAME, COMPLETE_NAME}
    installed_bytes = 0
    identities = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('volume recovery file entry is not a mapping')
        relative = _safe_relative(entry.get('destination'), label='manifest destination')
        if relative.as_posix() in expected_files:
            raise ValueError('volume recovery manifest shadows its own control file')
        expected_files.add(relative.as_posix())
        path = bundle_root / relative
        byte_count, digest = _regular_identity(
            path,
            allow_empty=path.name == 'milestone_audit_complete',
        )
        if byte_count != int(entry.get('bytes', -1)) or digest != entry.get('sha256'):
            raise ValueError(f'volume recovery bundle file changed: {relative}')
        installed_bytes += byte_count
        identities.add((byte_count, digest))
    actual_files = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob('*')
        if path.is_file()
    }
    symlinks = [path for path in bundle_root.rglob('*') if path.is_symlink()]
    if symlinks or actual_files != expected_files:
        raise ValueError('volume recovery bundle coverage changed')
    if installed_bytes != int(manifest.get('installed_bytes', -1)):
        raise ValueError('volume recovery installed-byte total changed')
    unique_bytes = sum(byte_count for byte_count, _digest in identities)
    if unique_bytes != int(manifest.get('unique_payload_bytes', -1)):
        raise ValueError('volume recovery unique-byte total changed')
    return {
        'status': 'passed',
        'version': VERSION,
        'bundle_root': str(bundle_root),
        'target_remote_run_root': str(remote_run_root),
        'jobs_recoverable': int(manifest['jobs_recoverable']),
        'file_count': len(entries),
        'installed_bytes': installed_bytes,
        'unique_payload_bytes': unique_bytes,
        'manifest_sha256': sha256_file(manifest_path),
        'scientific_result': False,
        'scientific_metric_values_inspected': False,
        'used_for_checkpoint_or_configuration_selection': False,
    }


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    """Register common plan/build CLI arguments."""
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--retained-root', type=Path, required=True)
    parser.add_argument('--milestone-root', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--remote-run-root', type=Path, required=True)
    parser.add_argument(
        '--audit-script',
        type=Path,
        default=Path(__file__).resolve().with_name('audit_shapo_p47_milestone.py'),
    )
    parser.add_argument(
        '--milestone-relative-root',
        type=Path,
        default=DEFAULT_MILESTONE_RELATIVE_ROOT,
    )
    parser.add_argument('--zstd-bin', type=Path, default=DEFAULT_ZSTD_BIN)


def main() -> None:
    """Plan, atomically build, or independently verify a recovery bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    plan_parser = subparsers.add_parser('plan', help='validate sources and print a dry plan')
    _add_plan_arguments(plan_parser)
    build_parser = subparsers.add_parser('build', help='create a new owner-private bundle')
    _add_plan_arguments(build_parser)
    build_parser.add_argument('--output-root', type=Path, required=True)
    verify_parser = subparsers.add_parser('verify', help='re-hash an existing bundle')
    verify_parser.add_argument('--bundle-root', type=Path, required=True)
    verify_parser.add_argument('--matrix', type=Path, required=True)
    verify_parser.add_argument('--source-commit', required=True)
    verify_parser.add_argument('--remote-run-root', type=Path, required=True)
    args = parser.parse_args()

    if args.command == 'verify':
        result = verify_bundle(
            args.bundle_root.absolute(),
            matrix_path=args.matrix.resolve(),
            source_commit=args.source_commit,
            remote_run_root=args.remote_run_root,
        )
    else:
        plan = build_recovery_plan(
            checkpoint_root=args.checkpoint_root.resolve(),
            retained_root=args.retained_root.resolve(),
            milestone_root=args.milestone_root.resolve(),
            matrix_path=args.matrix.resolve(),
            source_commit=args.source_commit,
            remote_run_root=args.remote_run_root,
            audit_script=args.audit_script.resolve(),
            milestone_relative_root=args.milestone_relative_root,
            zstd_bin=args.zstd_bin.resolve(),
        )
        result = (
            plan.manifest
            if args.command == 'plan'
            else materialize_bundle(plan, args.output_root.absolute())
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()

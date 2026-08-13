#!/usr/bin/env python3
"""Privately mirror every checkpoint retained by the frozen SHAPO protocol."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.star.validate_shapo_p47_checkpoint_backup import (  # noqa: E402
    validate_checkpoint_backup,
)
from scripts.star.validate_shapo_p47_retained_checkpoint_backup import (  # noqa: E402
    validate_retained_checkpoint_backup,
)


SAFE_HOST = re.compile(r'^[A-Za-z0-9_.@-]+$')
SAFE_REMOTE_ROOT = re.compile(r'^/[A-Za-z0-9_./-]+$')
FULL_SHA256 = re.compile(r'^[0-9a-f]{64}$')
DEFAULT_POLL_SECONDS = 3600.0
DEFAULT_MAX_FINAL_PATCH_BYTES = 4 * 1024**2
SLUGS = {
    'SafetyPointGoal1-v0': 'point-goal',
    'SafetyPointButton1-v0': 'point-button',
    'Ant-v4': 'ant',
    'Walker2d-v4': 'walker2d',
}
REMOTE_INVENTORY_SOURCE = r"""
import hashlib
import json
import sys
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def stable_identity(path, run_root):
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f'retained checkpoint is not a regular file: {path}')
    try:
        relative = path.resolve().relative_to(run_root).as_posix()
    except ValueError as error:
        raise SystemExit(f'retained checkpoint escaped run root: {path}') from error
    before = path.stat()
    checksum = digest(path)
    after = path.stat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise SystemExit(f'retained checkpoint changed while hashing: {path}')
    return {
        'relative': relative,
        'bytes': after.st_size,
        'mtime_ns': after.st_mtime_ns,
        'sha256': checksum,
    }


run_root = Path(sys.argv[1]).resolve()
jobs = json.loads(sys.argv[2])
expected_source_commit = sys.argv[3]
expected_matrix_sha256 = sys.argv[4]
manifest = {}
for line in (run_root / 'launch_manifest.txt').read_text(encoding='utf-8').splitlines():
    key, separator, value = line.partition('=')
    if not separator or not key or key in manifest:
        raise SystemExit('formal launch manifest is invalid')
    manifest[key] = value
if (
    manifest.get('source_commit') != expected_source_commit
    or manifest.get('matrix_sha256') != expected_matrix_sha256
):
    raise SystemExit('formal launch manifest identity changed')

for spec in jobs:
    job = spec['job']
    container = run_root / 'runs' / job
    if not container.exists():
        continue
    if (container / 'source_commit.txt').read_text(encoding='utf-8').strip() != expected_source_commit:
        raise SystemExit(f'formal run source identity changed: {container}')
    labels = [(f'milestone_{step}', f'milestone_{step}.pt') for step in spec['milestones']]
    labels.extend([('best_safe_return', 'best_safe_return.pt'), ('final_full', 'final_full.pt')])
    for label, filename in labels:
        candidates = list(container.glob(f'training/**/checkpoints/{filename}'))
        if not candidates:
            continue
        newest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        record = {'job': job, 'label': label, **stable_identity(newest, run_root)}
        if label == 'final_full':
            base = newest.with_name('latest_full.pt')
            if not base.is_file():
                raise SystemExit(f'final checkpoint lacks its same-segment latest-full base: {newest}')
            record['base'] = stable_identity(base, run_root)
        print(json.dumps(record, sort_keys=True))
"""


@dataclass(frozen=True)
class RetainedJob:
    """One frozen environment and training-seed retention slot."""

    environment: str
    seed: int
    job: str
    milestones: tuple[int, ...]
    total_steps: int

    @property
    def labels(self) -> tuple[str, ...]:
        """Return every retained label in protocol order."""
        return (
            *(f'milestone_{step}' for step in self.milestones),
            'best_safe_return',
            'final_full',
        )

    def expected_steps(self, label: str) -> set[int]:
        """Return the permitted counter steps for one retained label."""
        if label.startswith('milestone_'):
            return {int(label.removeprefix('milestone_'))}
        if label == 'best_safe_return':
            return set(self.milestones)
        if label == 'final_full':
            return {self.total_steps}
        raise ValueError(f'unknown retained label: {label!r}')


@dataclass(frozen=True)
class FileIdentity:
    """Stable identity of one remote regular file."""

    relative: str
    bytes: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class RemoteArtifact:
    """Stable identity of one remotely retained checkpoint."""

    job: str
    label: str
    file: FileIdentity
    base: FileIdentity | None = None


def utc_now() -> str:
    """Return one timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Return a streaming lowercase SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    """Persist directory-entry changes for one exact local directory."""
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_json(path: Path, payload: Any) -> None:
    """Atomically replace one strict JSON file and fsync its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    fsync_directory(path.parent)


def append_jsonl(path: Path, payload: Any) -> None:
    """Append and fsync one compact operational journal row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_remote_arguments(host: str, run_root: str, remote_python: str) -> None:
    """Reject SSH inputs that could expand the read-only remote command."""
    if not SAFE_HOST.fullmatch(host):
        raise ValueError(f'unsafe SSH host or alias: {host!r}')
    for label, value in (
        ('remote run root', run_root),
        ('remote Python interpreter', remote_python),
    ):
        if (
            not SAFE_REMOTE_ROOT.fullmatch(value)
            or '..' in Path(value).parts
            or Path(value) == Path('/')
        ):
            raise ValueError(f'unsafe {label}: {value!r}')


def expected_jobs(matrix_path: Path) -> list[RetainedJob]:
    """Expand the complete frozen 4-by-5 retained-checkpoint contract."""
    matrix = yaml.safe_load(matrix_path.read_text(encoding='utf-8'))
    if not isinstance(matrix, dict) or not isinstance(matrix.get('environments'), dict):
        raise ValueError('frozen matrix is invalid')
    environments = matrix['environments']
    seeds = [int(seed) for seed in matrix.get('training_seeds', [])]
    if set(environments) != set(SLUGS) or len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError('frozen matrix does not contain exactly four environments and five seeds')
    jobs = []
    for seed in seeds:
        for environment, spec in environments.items():
            milestones = tuple(int(step) for step in spec['milestones'])
            total_steps = int(spec['total_steps'])
            if len(milestones) != 3 or milestones[-1] != total_steps:
                raise ValueError(f'frozen retention milestones are invalid for {environment}')
            jobs.append(
                RetainedJob(
                    environment=environment,
                    seed=seed,
                    job=f'{SLUGS[environment]}-{seed}',
                    milestones=milestones,
                    total_steps=total_steps,
                ),
            )
    return jobs


def _parse_file_identity(payload: Any, *, label: str, remote_run_root: str) -> FileIdentity:
    """Parse one constrained remote file identity."""
    if not isinstance(payload, dict):
        raise ValueError(f'{label} identity is not a mapping')
    try:
        relative = str(payload['relative'])
        byte_count = int(payload['bytes'])
        mtime_ns = int(payload['mtime_ns'])
        checksum = str(payload['sha256'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'{label} identity is invalid') from error
    candidate = Path(relative)
    root = Path(remote_run_root)
    if (
        candidate.is_absolute()
        or '..' in candidate.parts
        or byte_count <= 0
        or mtime_ns <= 0
        or not FULL_SHA256.fullmatch(checksum)
        or not str((root / candidate).resolve()).startswith(f'{root.resolve()}/')
    ):
        raise ValueError(f'{label} identity is unsafe: {relative!r}')
    return FileIdentity(relative, byte_count, mtime_ns, checksum)


def parse_remote_inventory(
    text: str,
    jobs: dict[str, RetainedJob],
    remote_run_root: str,
) -> dict[tuple[str, str], RemoteArtifact]:
    """Parse and constrain the remote retained-checkpoint JSONL response."""
    records: dict[tuple[str, str], RemoteArtifact] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            job_name = str(payload['job'])
            artifact_label = str(payload['label'])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f'invalid remote retained record on line {line_number}') from error
        if job_name not in jobs or artifact_label not in jobs[job_name].labels:
            raise ValueError(f'unexpected remote retained artifact: {job_name}/{artifact_label}')
        key = (job_name, artifact_label)
        if key in records:
            raise ValueError(f'duplicate remote retained artifact: {job_name}/{artifact_label}')
        file_identity = _parse_file_identity(
            payload,
            label=f'{job_name}/{artifact_label}',
            remote_run_root=remote_run_root,
        )
        expected_name = f'{artifact_label}.pt'
        if artifact_label == 'best_safe_return':
            expected_name = 'best_safe_return.pt'
        if artifact_label == 'final_full':
            expected_name = 'final_full.pt'
        relative = Path(file_identity.relative)
        expected_prefix = Path('runs') / job_name / 'training'
        try:
            relative.relative_to(expected_prefix)
        except ValueError as error:
            raise ValueError(
                f'retained artifact escaped its job: {file_identity.relative!r}',
            ) from error
        if relative.name != expected_name or relative.parent.name != 'checkpoints':
            raise ValueError(
                f'retained artifact has the wrong filename: {file_identity.relative!r}',
            )
        base = None
        if artifact_label == 'final_full':
            base = _parse_file_identity(
                payload.get('base'),
                label=f'{job_name}/final_full base',
                remote_run_root=remote_run_root,
            )
            base_relative = Path(base.relative)
            if base_relative.parent != relative.parent or base_relative.name != 'latest_full.pt':
                raise ValueError(
                    f'final-full base does not share its checkpoint directory: {job_name}',
                )
        elif 'base' in payload:
            raise ValueError(f'non-final artifact unexpectedly has a patch base: {key}')
        records[key] = RemoteArtifact(job_name, artifact_label, file_identity, base)
    return records


def scan_remote(
    host: str,
    remote_run_root: str,
    remote_python: str,
    jobs: list[RetainedJob],
    *,
    source_commit: str,
    matrix_sha256: str,
) -> dict[tuple[str, str], RemoteArtifact]:
    """Hash all currently available retained checkpoints without mutating the server."""
    validate_remote_arguments(host, remote_run_root, remote_python)
    specifications = [
        {
            'job': job.job,
            'milestones': list(job.milestones),
            'total_steps': job.total_steps,
        }
        for job in jobs
    ]
    remote_command = (
        f'nice -n 19 ionice -c 3 {shlex.quote(remote_python)} - '
        f'{shlex.quote(remote_run_root)} {shlex.quote(json.dumps(specifications))} '
        f'{shlex.quote(source_commit)} {shlex.quote(matrix_sha256)}'
    )
    completed = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, remote_command],
        input=REMOTE_INVENTORY_SOURCE,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f'remote retained-checkpoint inventory failed with exit {completed.returncode}: '
            f'{completed.stderr.strip()}',
        )
    return parse_remote_inventory(
        completed.stdout,
        {job.job: job for job in jobs},
        remote_run_root,
    )


def ensure_capacity(parent: Path, transfer_bytes: int, reserve_bytes: int) -> int:
    """Require room for one staging operation plus the protected host reserve."""
    available = shutil.disk_usage(parent).free
    if transfer_bytes + reserve_bytes > available:
        raise RuntimeError(
            f'insufficient host space: need {transfer_bytes + reserve_bytes} bytes '
            f'including reserve, have {available}',
        )
    return available


def copy_remote_file(
    host: str,
    remote_run_root: str,
    identity: FileIdentity,
    destination: Path,
) -> None:
    """Copy one immutable server file into private staging storage."""
    remote_path = f'{host}:{remote_run_root.rstrip("/")}/{identity.relative}'
    completed = subprocess.run(
        [
            'rsync',
            '-a',
            '--protect-args',
            '--inplace',
            '--bwlimit=16384',
            '--rsync-path=nice -n 19 ionice -c 3 rsync',
            '-e',
            'ssh -o BatchMode=yes -o ConnectTimeout=10',
            '--',
            remote_path,
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f'rsync failed for {identity.relative!r} with exit {completed.returncode}: '
            f'{completed.stderr.strip()}',
        )


def _slot_path(output_root: Path, job: RetainedJob, label: str) -> Path:
    """Return one exact private retained-artifact directory."""
    if label not in job.labels:
        raise ValueError(f'unknown retained label for {job.job}: {label!r}')
    return output_root / job.job / label


def _checkpoint_name(label: str) -> str:
    """Return the unmodified lightweight checkpoint filename."""
    return f'{label}.pt'


def _remove_private_slot(path: Path, job_root: Path) -> None:
    """Remove one exact private staging/previous directory."""
    if path.parent != job_root or path.is_symlink():
        raise ValueError(f'unsafe retained-checkpoint path: {path}')
    if path.exists():
        if not path.is_dir():
            raise ValueError(f'retained-checkpoint path is not a directory: {path}')
        shutil.rmtree(path)


def _install_staging(staging: Path, output: Path, job_root: Path) -> None:
    """Atomically publish one validated retained artifact with rollback."""
    previous = job_root / f'.{output.name}.previous'
    if previous.exists():
        if output.exists():
            _remove_private_slot(previous, job_root)
        else:
            os.replace(previous, output)
    destination_old = None
    if output.exists():
        destination_old = previous
        os.replace(output, destination_old)
    try:
        os.replace(staging, output)
    except Exception:
        if destination_old is not None and destination_old.exists() and not output.exists():
            os.replace(destination_old, output)
        raise
    fsync_directory(job_root)
    if destination_old is not None:
        _remove_private_slot(destination_old, job_root)
        fsync_directory(job_root)


def _load_receipt(output: Path) -> dict[str, Any]:
    """Load a complete retained-artifact receipt."""
    receipt_path = output / 'backup_receipt.json'
    complete = output / 'backup_complete'
    if (
        output.is_symlink()
        or not output.is_dir()
        or not receipt_path.is_file()
        or not complete.is_file()
    ):
        raise ValueError(f'incomplete private retained-checkpoint backup: {output}')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if not isinstance(receipt, dict):
        raise ValueError(f'retained-checkpoint receipt is invalid: {output}')
    return receipt


def validate_existing_lightweight(
    job: RetainedJob,
    label: str,
    output: Path,
    *,
    matrix_path: Path,
    remote_run_root: Path,
    source_commit: str,
) -> dict[str, Any] | None:
    """Deeply revalidate one existing lightweight host checkpoint."""
    if not output.exists():
        return None
    receipt = _load_receipt(output)
    checkpoint = output / _checkpoint_name(label)
    validated = validate_retained_checkpoint_backup(
        checkpoint,
        matrix_path=matrix_path,
        environment=job.environment,
        seed=job.seed,
        job=job.job,
        label=label,
        expected_steps=job.expected_steps(label),
        replay_required=False,
        remote_run_root=remote_run_root,
        source_commit=source_commit,
    )
    for key, value in validated.items():
        if receipt.get(key) != value:
            raise ValueError(f'lightweight host receipt changed for {job.job}/{label}: {key}')
    return receipt


def stable_local_identity(path: Path) -> tuple[int, str]:
    """Hash one regular local file while proving its inode view stayed stable."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'local patch base is not a regular file: {path}')
    before = path.stat()
    checksum = sha256_file(path)
    after = path.stat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError(f'local patch base changed while hashing: {path}')
    return after.st_size, checksum


def matching_recovery_base(
    job: RetainedJob,
    record: RemoteArtifact,
    recovery_root: Path,
    *,
    matrix_path: Path,
    remote_run_root: Path,
    source_commit: str,
) -> tuple[Path, dict[str, Any]] | None:
    """Return the semantically validated host latest-full when it matches final-full's base."""
    if record.base is None:
        raise ValueError(f'final-full record lacks a base: {job.job}')
    root = recovery_root / job.job
    if not root.exists():
        return None
    receipt_path = root / 'backup_receipt.json'
    complete = root / 'backup_complete'
    checkpoint = root / 'latest_full.pt'
    if not receipt_path.is_file() or not complete.is_file() or not checkpoint.is_file():
        raise ValueError(f'incomplete recovery checkpoint base: {root}')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if (
        receipt.get('status') != 'passed'
        or receipt.get('job') != job.job
        or receipt.get('source_commit') != source_commit
        or receipt.get('checkpoint_sha256') != record.base.sha256
        or int(receipt.get('checkpoint_bytes', -1)) != record.base.bytes
    ):
        return None
    validated = validate_checkpoint_backup(
        checkpoint,
        matrix_path=matrix_path,
        environment=job.environment,
        seed=job.seed,
        job=job.job,
        remote_run_root=remote_run_root,
        source_commit=source_commit,
    )
    for key in (
        'status',
        'job',
        'environment',
        'training_seed',
        'checkpoint_env_steps',
        'checkpoint_bytes',
        'checkpoint_sha256',
        'matrix_sha256',
        'source_commit',
    ):
        if receipt.get(key) != validated.get(key):
            raise ValueError(f'recovery checkpoint receipt changed for patch base: {job.job}/{key}')
    byte_count, checksum = stable_local_identity(checkpoint)
    if byte_count != record.base.bytes or checksum != record.base.sha256:
        raise ValueError(f'validated recovery checkpoint bytes changed: {job.job}')
    return checkpoint, receipt


def validate_existing_final_patch(
    job: RetainedJob,
    output: Path,
    *,
    recovery_root: Path,
    record: RemoteArtifact | None,
    matrix_path: Path,
    source_commit: str,
    max_final_patch_bytes: int,
) -> dict[str, Any] | None:
    """Revalidate a final-full patch, its immutable receipt, and its exact base."""
    if not output.exists():
        return None
    receipt = _load_receipt(output)
    patch = output / 'final_full.pt.zstpatch'
    if patch.is_symlink() or not patch.is_file():
        raise ValueError(f'final-full patch is unavailable: {output}')
    patch_bytes, patch_sha = stable_local_identity(patch)
    expected = {
        'validator_version': 'shapo-p47-retained-checkpoint-validator-v1',
        'status': 'passed',
        'job': job.job,
        'environment': job.environment,
        'training_seed': job.seed,
        'label': 'final_full',
        'checkpoint_env_steps': job.total_steps,
        'expected_steps': [job.total_steps],
        'checkpoint_format': 'star-checkpoint-v1',
        'has_replay': True,
        'matrix_sha256': sha256_file(matrix_path),
        'source_commit': source_commit,
        'backup_version': 'shapo-p47-retained-final-patch-v1',
        'storage_representation': 'zstd_patch_from_validated_latest_full',
        'reconstruction_verified': True,
        'patch_bytes': patch_bytes,
        'patch_sha256': patch_sha,
        'scientific_result': False,
        'used_for_checkpoint_or_configuration_selection': False,
        'used_for_preregistered_checkpoint_retention': True,
        'scientific_metric_values_used_for_decision': False,
        'scientific_metric_values_reported': False,
        'checkpoint_counters_validated': True,
        'training_state_changed': False,
        'scientific_protocol_changed': False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f'final-full patch receipt changed for {job.job}: {key}')
    if patch_bytes > max_final_patch_bytes:
        raise ValueError(f'final-full patch exceeds its storage cap for {job.job}')
    if receipt.get('checkpoint_sha256') != receipt.get('final_checkpoint_sha256') or int(
        receipt.get('checkpoint_bytes', -1),
    ) != int(receipt.get('final_checkpoint_bytes', -2)):
        raise ValueError(f'final-full receipt checkpoint identities disagree for {job.job}')
    base = recovery_root / job.job / 'latest_full.pt'
    base_bytes, base_sha = stable_local_identity(base)
    if (
        receipt.get('base_checkpoint_sha256') != base_sha
        or int(receipt.get('base_checkpoint_bytes', -1)) != base_bytes
    ):
        raise ValueError(f'final-full patch base changed for {job.job}')
    if record is not None:
        if record.base is None:
            raise ValueError(f'current final-full record lacks a base: {job.job}')
        if (
            receipt.get('final_checkpoint_sha256') != record.file.sha256
            or int(receipt.get('final_checkpoint_bytes', -1)) != record.file.bytes
            or base_sha != record.base.sha256
            or base_bytes != record.base.bytes
        ):
            raise ValueError(f'host/server final-full patch identity changed for {job.job}')
    return receipt


def _second_scan_matches(
    job: RetainedJob,
    record: RemoteArtifact,
    *,
    host: str,
    remote_run_root: str,
    remote_python: str,
    source_commit: str,
    matrix_sha256: str,
) -> None:
    """Require the server artifact and optional patch base to remain byte-identical."""
    second = scan_remote(
        host,
        remote_run_root,
        remote_python,
        [job],
        source_commit=source_commit,
        matrix_sha256=matrix_sha256,
    ).get((job.job, record.label))
    if second != record:
        raise ValueError(
            f'remote retained checkpoint changed during transfer: {job.job}/{record.label}',
        )


def transfer_lightweight(  # pylint: disable=too-many-arguments,too-many-locals
    job: RetainedJob,
    record: RemoteArtifact,
    *,
    host: str,
    remote_run_root: str,
    remote_python: str,
    output_root: Path,
    matrix_path: Path,
    source_commit: str,
    reserve_bytes: int,
    matrix_sha256: str,
) -> dict[str, Any]:
    """Atomically retain one deeply validated lightweight checkpoint."""
    label = record.label
    output = _slot_path(output_root, job, label)
    existing = validate_existing_lightweight(
        job,
        label,
        output,
        matrix_path=matrix_path,
        remote_run_root=Path(remote_run_root),
        source_commit=source_commit,
    )
    if existing is not None and existing['checkpoint_sha256'] == record.file.sha256:
        if int(existing['checkpoint_bytes']) != record.file.bytes:
            raise ValueError(f'host/server lightweight size changed: {job.job}/{label}')
        return {
            'job': job.job,
            'label': label,
            'state': 'already_current',
            'checkpoint_env_steps': int(existing['checkpoint_env_steps']),
            'checkpoint_bytes': record.file.bytes,
            'checkpoint_sha256': record.file.sha256,
        }
    if existing is not None and label != 'best_safe_return':
        raise ValueError(f'immutable retained checkpoint changed: {job.job}/{label}')

    available_before = ensure_capacity(output_root, record.file.bytes, reserve_bytes)
    job_root = output_root / job.job
    job_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if job_root.is_symlink() or not job_root.is_dir():
        raise ValueError(f'private retained-checkpoint job root is unsafe: {job_root}')
    os.chmod(job_root, 0o700)
    staging = job_root / f'.{label}.staging'
    _remove_private_slot(staging, job_root)
    staging.mkdir(mode=0o700)
    try:
        checkpoint = staging / _checkpoint_name(label)
        copy_remote_file(host, remote_run_root, record.file, checkpoint)
        os.chmod(checkpoint, 0o600)
        copied_bytes, copied_sha = stable_local_identity(checkpoint)
        if copied_bytes != record.file.bytes or copied_sha != record.file.sha256:
            raise ValueError(f'lightweight checkpoint changed in transit: {job.job}/{label}')
        _second_scan_matches(
            job,
            record,
            host=host,
            remote_run_root=remote_run_root,
            remote_python=remote_python,
            source_commit=source_commit,
            matrix_sha256=matrix_sha256,
        )
        receipt = validate_retained_checkpoint_backup(
            checkpoint,
            matrix_path=matrix_path,
            environment=job.environment,
            seed=job.seed,
            job=job.job,
            label=label,
            expected_steps=job.expected_steps(label),
            replay_required=False,
            remote_run_root=Path(remote_run_root),
            source_commit=source_commit,
        )
        if existing is not None and int(receipt['checkpoint_env_steps']) <= int(
            existing['checkpoint_env_steps'],
        ):
            raise ValueError(f'best-safe checkpoint did not advance: {job.job}')
        receipt.update(
            {
                'backup_version': 'shapo-p47-retained-checkpoint-backup-v1',
                'backed_up_at_utc': utc_now(),
                'remote_host': host,
                'remote_run_root': remote_run_root,
                'remote_relative_path': record.file.relative,
                'remote_mtime_ns': record.file.mtime_ns,
                'host_available_bytes_before_transfer': available_before,
            },
        )
        atomic_json(staging / 'backup_receipt.json', receipt)
        (staging / 'backup_complete').touch()
        os.chmod(staging / 'backup_complete', 0o600)
        with (staging / 'backup_complete').open('rb') as stream:
            os.fsync(stream.fileno())
        fsync_directory(staging)
        _install_staging(staging, output, job_root)
    except Exception:
        _remove_private_slot(staging, job_root)
        raise
    return {
        'job': job.job,
        'label': label,
        'state': 'transferred',
        'checkpoint_env_steps': int(receipt['checkpoint_env_steps']),
        'checkpoint_bytes': record.file.bytes,
        'checkpoint_sha256': record.file.sha256,
        'host_available_bytes_before_transfer': available_before,
    }


def create_final_patch(
    base: Path,
    final_checkpoint: Path,
    patch: Path,
    reconstructed: Path,
    *,
    zstd_bin: Path,
    max_patch_bytes: int = DEFAULT_MAX_FINAL_PATCH_BYTES,
) -> dict[str, Any]:
    """Create and byte-exactly verify one Zstandard final-full patch."""
    if max_patch_bytes <= 0:
        raise ValueError('final-full patch cap must be positive')
    encode = subprocess.run(
        [
            str(zstd_bin),
            '-q',
            '-T1',
            '-3',
            f'--patch-from={base}',
            str(final_checkpoint),
            '-o',
            str(patch),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if encode.returncode != 0:
        raise RuntimeError(f'zstd final-full patch encoding failed: {encode.stderr.strip()}')
    patch_bytes, patch_sha = stable_local_identity(patch)
    if patch_bytes > max_patch_bytes:
        raise ValueError(
            f'final-full patch exceeds {max_patch_bytes}-byte storage cap: {patch_bytes}',
        )
    decode = subprocess.run(
        [
            str(zstd_bin),
            '-q',
            '-d',
            f'--patch-from={base}',
            str(patch),
            '-o',
            str(reconstructed),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if decode.returncode != 0:
        raise RuntimeError(f'zstd final-full patch decoding failed: {decode.stderr.strip()}')
    final_bytes, final_sha = stable_local_identity(final_checkpoint)
    reconstructed_bytes, reconstructed_sha = stable_local_identity(reconstructed)
    if reconstructed_bytes != final_bytes or reconstructed_sha != final_sha:
        raise ValueError('final-full patch did not reconstruct the exact source checkpoint')
    verified_patch_bytes, verified_patch_sha = stable_local_identity(patch)
    if verified_patch_bytes != patch_bytes or verified_patch_sha != patch_sha:
        raise ValueError('final-full patch changed while its reconstruction was verified')
    return {
        'final_checkpoint_bytes': final_bytes,
        'final_checkpoint_sha256': final_sha,
        'patch_bytes': patch_bytes,
        'patch_sha256': patch_sha,
        'reconstruction_verified': True,
    }


def transfer_final_patch(  # pylint: disable=too-many-arguments,too-many-locals
    job: RetainedJob,
    record: RemoteArtifact,
    *,
    host: str,
    remote_run_root: str,
    remote_python: str,
    output_root: Path,
    recovery_root: Path,
    matrix_path: Path,
    source_commit: str,
    reserve_bytes: int,
    matrix_sha256: str,
    zstd_bin: Path,
    max_final_patch_bytes: int = DEFAULT_MAX_FINAL_PATCH_BYTES,
) -> dict[str, Any]:
    """Retain final-full as a byte-exact patch against its validated latest-full base."""
    output = _slot_path(output_root, job, 'final_full')
    existing = validate_existing_final_patch(
        job,
        output,
        recovery_root=recovery_root,
        record=record,
        matrix_path=matrix_path,
        source_commit=source_commit,
        max_final_patch_bytes=max_final_patch_bytes,
    )
    if existing is not None:
        return {
            'job': job.job,
            'label': 'final_full',
            'state': 'already_current',
            'checkpoint_env_steps': job.total_steps,
            'final_checkpoint_bytes': int(existing['final_checkpoint_bytes']),
            'final_checkpoint_sha256': existing['final_checkpoint_sha256'],
            'patch_bytes': int(existing['patch_bytes']),
            'patch_sha256': existing['patch_sha256'],
        }
    base_match = matching_recovery_base(
        job,
        record,
        recovery_root,
        matrix_path=matrix_path,
        remote_run_root=Path(remote_run_root),
        source_commit=source_commit,
    )
    if base_match is None:
        return {
            'job': job.job,
            'label': 'final_full',
            'state': 'waiting_for_matching_latest_full_backup',
        }
    base, _base_receipt = base_match
    # Staging can briefly contain source, reconstruction, and a worst-case encoded patch.
    available_before = ensure_capacity(
        output_root,
        3 * record.file.bytes + max_final_patch_bytes,
        reserve_bytes,
    )
    job_root = output_root / job.job
    job_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if job_root.is_symlink() or not job_root.is_dir():
        raise ValueError(f'private retained-checkpoint job root is unsafe: {job_root}')
    os.chmod(job_root, 0o700)
    staging = job_root / '.final_full.staging'
    _remove_private_slot(staging, job_root)
    staging.mkdir(mode=0o700)
    try:
        final_checkpoint = staging / 'final_full.pt'
        reconstructed = staging / '.final_full.reconstructed.pt'
        patch = staging / 'final_full.pt.zstpatch'
        copy_remote_file(host, remote_run_root, record.file, final_checkpoint)
        os.chmod(final_checkpoint, 0o600)
        copied_bytes, copied_sha = stable_local_identity(final_checkpoint)
        if copied_bytes != record.file.bytes or copied_sha != record.file.sha256:
            raise ValueError(f'final-full checkpoint changed in transit: {job.job}')
        validation = validate_retained_checkpoint_backup(
            final_checkpoint,
            matrix_path=matrix_path,
            environment=job.environment,
            seed=job.seed,
            job=job.job,
            label='final_full',
            expected_steps={job.total_steps},
            replay_required=True,
            remote_run_root=Path(remote_run_root),
            source_commit=source_commit,
        )
        patch_identity = create_final_patch(
            base,
            final_checkpoint,
            patch,
            reconstructed,
            zstd_bin=zstd_bin,
            max_patch_bytes=max_final_patch_bytes,
        )
        _second_scan_matches(
            job,
            record,
            host=host,
            remote_run_root=remote_run_root,
            remote_python=remote_python,
            source_commit=source_commit,
            matrix_sha256=matrix_sha256,
        )
        base_bytes, base_sha = stable_local_identity(base)
        if record.base is None or base_bytes != record.base.bytes or base_sha != record.base.sha256:
            raise ValueError(f'final-full patch base changed during transfer: {job.job}')
        final_checkpoint.unlink()
        reconstructed.unlink()
        os.chmod(patch, 0o600)
        receipt = {
            **validation,
            **patch_identity,
            'backup_version': 'shapo-p47-retained-final-patch-v1',
            'backed_up_at_utc': utc_now(),
            'remote_host': host,
            'remote_run_root': remote_run_root,
            'remote_relative_path': record.file.relative,
            'remote_mtime_ns': record.file.mtime_ns,
            'base_remote_relative_path': record.base.relative,
            'base_checkpoint_bytes': base_bytes,
            'base_checkpoint_sha256': base_sha,
            'zstd_binary': str(zstd_bin),
            'host_available_bytes_before_transfer': available_before,
            'storage_representation': 'zstd_patch_from_validated_latest_full',
        }
        atomic_json(staging / 'backup_receipt.json', receipt)
        (staging / 'backup_complete').touch()
        os.chmod(staging / 'backup_complete', 0o600)
        with (staging / 'backup_complete').open('rb') as stream:
            os.fsync(stream.fileno())
        fsync_directory(staging)
        _install_staging(staging, output, job_root)
    except Exception:
        _remove_private_slot(staging, job_root)
        raise
    return {
        'job': job.job,
        'label': 'final_full',
        'state': 'transferred',
        'checkpoint_env_steps': job.total_steps,
        **patch_identity,
        'host_available_bytes_before_transfer': available_before,
    }


def backup_once(  # pylint: disable=too-many-arguments,too-many-locals
    jobs: list[RetainedJob],
    *,
    host: str,
    remote_run_root: str,
    remote_python: str,
    output_root: Path,
    recovery_root: Path,
    matrix_path: Path,
    source_commit: str,
    reserve_bytes: int,
    matrix_sha256: str,
    zstd_bin: Path,
    max_final_patch_bytes: int = DEFAULT_MAX_FINAL_PATCH_BYTES,
) -> dict[str, Any]:
    """Refresh every available retained slot and revalidate every host copy."""
    records = scan_remote(
        host,
        remote_run_root,
        remote_python,
        jobs,
        source_commit=source_commit,
        matrix_sha256=matrix_sha256,
    )
    rows = []
    host_artifacts = 0
    complete_jobs = 0
    for job in jobs:
        job_complete = True
        for label in job.labels:
            record = records.get((job.job, label))
            output = _slot_path(output_root, job, label)
            if record is None:
                if output.exists():
                    if label == 'final_full':
                        validate_existing_final_patch(
                            job,
                            output,
                            recovery_root=recovery_root,
                            record=None,
                            matrix_path=matrix_path,
                            source_commit=source_commit,
                            max_final_patch_bytes=max_final_patch_bytes,
                        )
                    else:
                        validate_existing_lightweight(
                            job,
                            label,
                            output,
                            matrix_path=matrix_path,
                            remote_run_root=Path(remote_run_root),
                            source_commit=source_commit,
                        )
                    raise ValueError(
                        f'remote retained artifact disappeared after host backup: {job.job}/{label}',
                    )
                rows.append(
                    {'job': job.job, 'label': label, 'state': 'waiting_for_remote_artifact'},
                )
                job_complete = False
                continue
            if label == 'final_full':
                row = transfer_final_patch(
                    job,
                    record,
                    host=host,
                    remote_run_root=remote_run_root,
                    remote_python=remote_python,
                    output_root=output_root,
                    recovery_root=recovery_root,
                    matrix_path=matrix_path,
                    source_commit=source_commit,
                    reserve_bytes=reserve_bytes,
                    matrix_sha256=matrix_sha256,
                    zstd_bin=zstd_bin,
                    max_final_patch_bytes=max_final_patch_bytes,
                )
            else:
                row = transfer_lightweight(
                    job,
                    record,
                    host=host,
                    remote_run_root=remote_run_root,
                    remote_python=remote_python,
                    output_root=output_root,
                    matrix_path=matrix_path,
                    source_commit=source_commit,
                    reserve_bytes=reserve_bytes,
                    matrix_sha256=matrix_sha256,
                )
            rows.append(row)
            retained = row['state'] in {'transferred', 'already_current'}
            host_artifacts += int(retained)
            job_complete &= retained
        complete_jobs += int(job_complete)
    total_artifacts = sum(len(job.labels) for job in jobs)
    return {
        'backup_version': 'shapo-p47-retained-checkpoint-backup-v1',
        'observed_at_utc': utc_now(),
        'state': 'complete' if complete_jobs == len(jobs) else 'watching',
        'scientific_result': False,
        'used_for_checkpoint_or_configuration_selection': False,
        'used_for_preregistered_checkpoint_retention': True,
        'scientific_metric_values_used_for_decision': False,
        'scientific_metric_values_reported': False,
        'training_state_changed': False,
        'scientific_protocol_changed': False,
        'checkpoint_payloads_published_to_github': False,
        'remote_host': host,
        'remote_run_root': remote_run_root,
        'total_jobs': len(jobs),
        'complete_jobs': complete_jobs,
        'total_retained_artifacts': total_artifacts,
        'remote_retained_artifacts': len(records),
        'host_retained_artifacts': host_artifacts,
        'newly_transferred_artifacts': sum(row['state'] == 'transferred' for row in rows),
        'waiting_artifacts': total_artifacts - host_artifacts,
        'rows': rows,
    }


def main() -> None:  # pylint: disable=too-many-locals
    """Poll retained identities and maintain private host copies until all 20 jobs finish."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-host', default='seeta-nmb2')
    parser.add_argument('--remote-run-root', required=True)
    parser.add_argument('--remote-python-bin', required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--recovery-backup-root', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--zstd-bin', type=Path, required=True)
    parser.add_argument('--reserve-bytes', type=int, default=2 * 1024**3)
    parser.add_argument(
        '--max-final-patch-bytes',
        type=int,
        default=DEFAULT_MAX_FINAL_PATCH_BYTES,
    )
    parser.add_argument('--poll-seconds', type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.reserve_bytes < 0 or args.max_final_patch_bytes <= 0 or args.poll_seconds <= 0.0:
        parser.error('reserve must be nonnegative; patch cap and poll seconds must be positive')
    if len(args.source_commit) != 40 or any(
        value not in '0123456789abcdef' for value in args.source_commit
    ):
        parser.error('--source-commit must be a full lowercase Git identity')
    validate_remote_arguments(args.remote_host, args.remote_run_root, args.remote_python_bin)
    for executable in ('ssh', 'rsync'):
        if shutil.which(executable) is None:
            parser.error(f'{executable} is unavailable')

    matrix_path = args.matrix.resolve()
    output_root = args.output_root.resolve()
    recovery_root = args.recovery_backup_root.resolve()
    state_dir = args.state_dir.resolve()
    zstd_bin = args.zstd_bin.resolve()
    if not matrix_path.is_file() or not zstd_bin.is_file() or not os.access(zstd_bin, os.X_OK):
        parser.error('frozen matrix or executable zstd binary is unavailable')
    if not recovery_root.is_dir():
        parser.error('recovery checkpoint backup root is unavailable')
    roots = (output_root, recovery_root, state_dir)
    if any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ):
        parser.error('retained output, recovery backup, and state directories must be separate')
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_root, 0o700)
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state_dir, 0o700)
    matrix_sha256 = sha256_file(matrix_path)
    backup_script_sha256 = sha256_file(Path(__file__).resolve())
    validator_path = Path(__file__).with_name('validate_shapo_p47_retained_checkpoint_backup.py')
    validator_sha256 = sha256_file(validator_path)
    jobs = expected_jobs(matrix_path)

    with (state_dir / '.retained-checkpoint-backup.lock').open('w', encoding='utf-8') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit('another retained-checkpoint backup holds the state lock') from error
        while True:
            try:
                if sha256_file(matrix_path) != matrix_sha256:
                    raise ValueError('locked matrix changed while retained backup was running')
                if sha256_file(Path(__file__).resolve()) != backup_script_sha256:
                    raise ValueError('retained-checkpoint backup source changed while running')
                if sha256_file(validator_path) != validator_sha256:
                    raise ValueError('retained-checkpoint validator changed while running')
                snapshot = backup_once(
                    jobs,
                    host=args.remote_host,
                    remote_run_root=args.remote_run_root,
                    remote_python=args.remote_python_bin,
                    output_root=output_root,
                    recovery_root=recovery_root,
                    matrix_path=matrix_path,
                    source_commit=args.source_commit,
                    reserve_bytes=args.reserve_bytes,
                    matrix_sha256=matrix_sha256,
                    zstd_bin=zstd_bin,
                    max_final_patch_bytes=args.max_final_patch_bytes,
                )
                snapshot.update(
                    {
                        'matrix_sha256': matrix_sha256,
                        'source_commit': args.source_commit,
                        'backup_script_sha256': backup_script_sha256,
                        'validator_sha256': validator_sha256,
                    },
                )
            except Exception as error:
                alert = {
                    'observed_at_utc': utc_now(),
                    'state': 'alert',
                    'error_type': type(error).__name__,
                    'error': str(error),
                    'scientific_result': False,
                    'used_for_checkpoint_or_configuration_selection': False,
                    'scientific_metric_values_used_for_decision': False,
                    'scientific_metric_values_reported': False,
                    'training_state_changed': False,
                    'scientific_protocol_changed': False,
                }
                atomic_json(state_dir / 'retained_checkpoint_backup_alert.json', alert)
                raise
            atomic_json(state_dir / 'retained_checkpoint_backup_latest.json', snapshot)
            append_jsonl(
                state_dir / 'retained_checkpoint_backup_journal.jsonl',
                {
                    key: snapshot[key]
                    for key in (
                        'observed_at_utc',
                        'state',
                        'remote_retained_artifacts',
                        'host_retained_artifacts',
                        'newly_transferred_artifacts',
                        'waiting_artifacts',
                        'complete_jobs',
                        'scientific_result',
                        'used_for_checkpoint_or_configuration_selection',
                        'scientific_metric_values_used_for_decision',
                        'scientific_metric_values_reported',
                        'training_state_changed',
                        'scientific_protocol_changed',
                    )
                },
            )
            print(
                json.dumps(
                    {
                        key: snapshot[key]
                        for key in (
                            'observed_at_utc',
                            'state',
                            'remote_retained_artifacts',
                            'host_retained_artifacts',
                            'newly_transferred_artifacts',
                            'waiting_artifacts',
                            'complete_jobs',
                        )
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if snapshot['state'] == 'complete' or args.once:
                return
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()

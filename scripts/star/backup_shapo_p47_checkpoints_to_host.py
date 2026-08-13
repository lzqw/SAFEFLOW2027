#!/usr/bin/env python3
"""Mirror the newest replay-bearing SHAPO checkpoint for disaster recovery."""

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


SAFE_HOST = re.compile(r'^[A-Za-z0-9_.@-]+$')
SAFE_REMOTE_ROOT = re.compile(r'^/[A-Za-z0-9_./-]+$')
FULL_SHA256 = re.compile(r'^[0-9a-f]{64}$')
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
for job in jobs:
    container = run_root / 'runs' / job
    candidates = list(container.glob('training/**/checkpoints/latest_full.pt'))
    if not candidates:
        continue
    newest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    if newest.is_symlink() or not newest.is_file():
        raise SystemExit(f'checkpoint is not a regular file: {newest}')
    if (container / 'source_commit.txt').read_text(encoding='utf-8').strip() != expected_source_commit:
        raise SystemExit(f'formal run source identity changed: {container}')
    try:
        relative = newest.resolve().relative_to(run_root).as_posix()
    except ValueError as error:
        raise SystemExit(f'checkpoint escaped run root: {newest}') from error
    before = newest.stat()
    checksum = digest(newest)
    after = newest.stat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise SystemExit(f'checkpoint changed while hashing: {newest}')
    print(json.dumps({
        'job': job,
        'relative': relative,
        'bytes': after.st_size,
        'mtime_ns': after.st_mtime_ns,
        'sha256': checksum,
    }, sort_keys=True))
"""


@dataclass(frozen=True)
class BackupJob:
    """One frozen environment and training-seed recovery slot."""

    environment: str
    seed: int
    job: str


@dataclass(frozen=True)
class RemoteCheckpoint:
    """Stable identity of one server-side full recovery checkpoint."""

    job: str
    relative: str
    bytes: int
    mtime_ns: int
    sha256: str


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    """Atomically replace one strict JSON receipt and fsync its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def append_jsonl(path: Path, payload: Any) -> None:
    """Append and fsync one compact operational journal row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    """Persist directory-entry changes for one exact local directory."""
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def validate_remote_arguments(host: str, run_root: str) -> None:
    """Reject SSH inputs that could expand the read-only remote command."""
    if not SAFE_HOST.fullmatch(host):
        raise ValueError(f'unsafe SSH host or alias: {host!r}')
    if (
        not SAFE_REMOTE_ROOT.fullmatch(run_root)
        or '..' in Path(run_root).parts
        or Path(run_root) == Path('/')
    ):
        raise ValueError(f'unsafe remote run root: {run_root!r}')


def validate_remote_python(path: str) -> None:
    """Require one explicit absolute remote interpreter path."""
    if not SAFE_REMOTE_ROOT.fullmatch(path) or '..' in Path(path).parts or Path(path) == Path('/'):
        raise ValueError(f'unsafe remote Python interpreter: {path!r}')


def expected_jobs(matrix_path: Path) -> list[BackupJob]:
    """Expand the complete frozen 4-by-5 matrix in registered order."""
    matrix = yaml.safe_load(matrix_path.read_text(encoding='utf-8'))
    if not isinstance(matrix, dict) or not isinstance(matrix.get('environments'), dict):
        raise ValueError('frozen matrix is invalid')
    environments = matrix['environments']
    seeds = [int(seed) for seed in matrix.get('training_seeds', [])]
    if set(environments) != set(SLUGS) or len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError('frozen matrix does not contain exactly four environments and five seeds')
    return [
        BackupJob(environment=environment, seed=seed, job=f'{SLUGS[environment]}-{seed}')
        for seed in seeds
        for environment in environments
    ]


def parse_remote_inventory(
    text: str,
    jobs: dict[str, BackupJob],
    remote_run_root: str,
) -> dict[str, RemoteCheckpoint]:
    """Parse and constrain the server inventory JSONL response."""
    records = {}
    root = Path(remote_run_root)
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            job = str(payload['job'])
            relative = str(payload['relative'])
            byte_count = int(payload['bytes'])
            mtime_ns = int(payload['mtime_ns'])
            checksum = str(payload['sha256'])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f'invalid remote checkpoint record on line {line_number}') from error
        if job not in jobs or job in records:
            raise ValueError(f'unexpected or duplicate remote checkpoint job: {job!r}')
        candidate = Path(relative)
        expected_prefix = Path('runs') / job / 'training'
        try:
            candidate.relative_to(expected_prefix)
        except ValueError as error:
            raise ValueError(f'remote checkpoint escaped its job: {relative!r}') from error
        if (
            candidate.is_absolute()
            or '..' in candidate.parts
            or candidate.name != 'latest_full.pt'
            or candidate.parent.name != 'checkpoints'
            or byte_count <= 0
            or mtime_ns <= 0
            or not FULL_SHA256.fullmatch(checksum)
            or not str((root / candidate).resolve()).startswith(f'{root.resolve()}/')
        ):
            raise ValueError(f'remote checkpoint metadata is unsafe: {relative!r}')
        records[job] = RemoteCheckpoint(job, relative, byte_count, mtime_ns, checksum)
    return records


def scan_remote(
    host: str,
    remote_run_root: str,
    remote_python: str,
    jobs: list[BackupJob],
    *,
    source_commit: str,
    matrix_sha256: str,
) -> dict[str, RemoteCheckpoint]:
    """Hash every current server checkpoint without mutating the training tree."""
    validate_remote_arguments(host, remote_run_root)
    validate_remote_python(remote_python)
    job_names = [job.job for job in jobs]
    remote_command = (
        f'nice -n 19 ionice -c 3 {shlex.quote(remote_python)} - '
        f'{shlex.quote(remote_run_root)} '
        f'{shlex.quote(json.dumps(job_names))} {shlex.quote(source_commit)} '
        f'{shlex.quote(matrix_sha256)}'
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
            f'remote checkpoint inventory failed with exit {completed.returncode}: '
            f'{completed.stderr.strip()}',
        )
    return parse_remote_inventory(
        completed.stdout,
        {job.job: job for job in jobs},
        remote_run_root,
    )


def ensure_capacity(parent: Path, transfer_bytes: int, reserve_bytes: int) -> int:
    """Require room for a staging copy plus the protected host reserve."""
    available = shutil.disk_usage(parent).free
    if transfer_bytes + reserve_bytes > available:
        raise RuntimeError(
            f'insufficient host space: need {transfer_bytes + reserve_bytes} bytes '
            f'including reserve, have {available}',
        )
    return available


def copy_remote_checkpoint(
    host: str,
    remote_run_root: str,
    record: RemoteCheckpoint,
    destination: Path,
) -> None:
    """Copy one immutable inode view to a private staging file."""
    validate_remote_arguments(host, remote_run_root)
    remote_path = f'{host}:{remote_run_root.rstrip("/")}/{record.relative}'
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
            f'rsync failed for {record.job!r} with exit {completed.returncode}: '
            f'{completed.stderr.strip()}',
        )


def validate_existing_backup(
    job: BackupJob,
    output: Path,
    *,
    matrix_path: Path,
    remote_run_root: Path,
    source_commit: str,
) -> dict[str, Any] | None:
    """Validate an existing host backup and return its receipt when complete."""
    checkpoint = output / 'latest_full.pt'
    receipt_path = output / 'backup_receipt.json'
    complete = output / 'backup_complete'
    if output.is_symlink():
        raise ValueError(f'host checkpoint backup is a symbolic link: {output}')
    if output.exists() and not output.is_dir():
        raise ValueError(f'host checkpoint backup is not a directory: {output}')
    present = [checkpoint.exists(), receipt_path.exists(), complete.exists()]
    if not output.exists() and not any(present):
        return None
    if not all(present):
        raise ValueError(f'incomplete host checkpoint backup: {output}')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
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
            raise ValueError(f'host checkpoint backup receipt changed for {job.job}: {key}')
    return receipt


def remove_private_tree(path: Path, output_root: Path) -> None:
    """Remove only one exact private staging or previous-backup directory."""
    if path.parent != output_root or path.is_symlink():
        raise ValueError(f'unsafe private backup path: {path}')
    if path.exists():
        if not path.is_dir():
            raise ValueError(f'private backup path is not a directory: {path}')
        shutil.rmtree(path)


def recover_interrupted_swap(
    job: BackupJob,
    *,
    output_root: Path,
    matrix_path: Path,
    remote_run_root: Path,
    source_commit: str,
) -> None:
    """Restore a prior valid directory after an interrupted atomic handoff."""
    output = output_root / job.job
    staging = output_root / f'.{job.job}.staging'
    previous = output_root / f'.{job.job}.previous'
    if previous.exists():
        if previous.is_symlink() or not previous.is_dir():
            raise ValueError(f'interrupted previous backup is unsafe: {previous}')
        if output.exists():
            validate_existing_backup(
                job,
                output,
                matrix_path=matrix_path,
                remote_run_root=remote_run_root,
                source_commit=source_commit,
            )
            remove_private_tree(previous, output_root)
        else:
            os.replace(previous, output)
    remove_private_tree(staging, output_root)


def transfer_checkpoint(  # pylint: disable=too-many-arguments,too-many-locals
    job: BackupJob,
    record: RemoteCheckpoint,
    *,
    host: str,
    remote_run_root: str,
    remote_python: str,
    output_root: Path,
    matrix_path: Path,
    source_commit: str,
    reserve_bytes: int,
) -> dict[str, Any]:
    """Atomically install one newer, fully validated disaster-recovery backup."""
    output = output_root / job.job
    output_root.mkdir(parents=True, exist_ok=True)
    recover_interrupted_swap(
        job,
        output_root=output_root,
        matrix_path=matrix_path,
        remote_run_root=Path(remote_run_root),
        source_commit=source_commit,
    )
    existing = validate_existing_backup(
        job,
        output,
        matrix_path=matrix_path,
        remote_run_root=Path(remote_run_root),
        source_commit=source_commit,
    )
    if existing is not None:
        existing_step = int(existing['checkpoint_env_steps'])
        existing_sha = str(existing['checkpoint_sha256'])
        if existing_sha == record.sha256:
            if int(existing['checkpoint_bytes']) != record.bytes:
                raise ValueError(f'host/server byte identity changed for {job.job}')
            return {
                'job': job.job,
                'state': 'already_current',
                'checkpoint_env_steps': existing_step,
                'checkpoint_bytes': record.bytes,
                'checkpoint_sha256': record.sha256,
            }

    available_before = ensure_capacity(output_root, record.bytes, reserve_bytes)
    staging = output_root / f'.{job.job}.staging'
    previous = output_root / f'.{job.job}.previous'
    staging.mkdir(mode=0o700)
    try:
        staged_checkpoint = staging / 'latest_full.pt'
        copy_remote_checkpoint(host, remote_run_root, record, staged_checkpoint)
        if staged_checkpoint.is_symlink():
            raise ValueError(f'copied checkpoint is a symbolic link: {job.job}')
        copied_sha = sha256_file(staged_checkpoint)
        if staged_checkpoint.stat().st_size != record.bytes or copied_sha != record.sha256:
            raise ValueError(f'checkpoint changed or was corrupted in transit: {job.job}')
        second_scan = scan_remote(
            host,
            remote_run_root,
            remote_python,
            [job],
            source_commit=source_commit,
            matrix_sha256=sha256_file(matrix_path),
        ).get(job.job)
        if second_scan != record:
            raise ValueError(f'remote checkpoint changed during transfer: {job.job}')
        receipt = validate_checkpoint_backup(
            staged_checkpoint,
            matrix_path=matrix_path,
            environment=job.environment,
            seed=job.seed,
            job=job.job,
            remote_run_root=Path(remote_run_root),
            source_commit=source_commit,
        )
        receipt.update(
            {
                'backup_version': 'shapo-p47-host-checkpoint-backup-v1',
                'backed_up_at_utc': utc_now(),
                'remote_host': host,
                'remote_run_root': remote_run_root,
                'remote_relative_path': record.relative,
                'remote_mtime_ns': record.mtime_ns,
                'host_available_bytes_before_transfer': available_before,
                'scientific_result': False,
                'used_for_checkpoint_or_configuration_selection': False,
                'inspected_scientific_metrics': False,
                'training_state_changed': False,
                'scientific_protocol_changed': False,
            },
        )
        if existing is not None and int(receipt['checkpoint_env_steps']) <= int(
            existing['checkpoint_env_steps'],
        ):
            raise ValueError(
                f'remote checkpoint did not advance beyond a different host backup: {job.job}',
            )
        atomic_json(staging / 'backup_receipt.json', receipt)
        (staging / 'backup_complete').touch()
        with (staging / 'backup_complete').open('rb') as stream:
            os.fsync(stream.fileno())
        fsync_directory(staging)
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
        fsync_directory(output_root)
        if destination_old is not None:
            remove_private_tree(destination_old, output_root)
            fsync_directory(output_root)
    except Exception:
        remove_private_tree(staging, output_root)
        raise
    return {
        'job': job.job,
        'state': 'transferred',
        'checkpoint_env_steps': int(receipt['checkpoint_env_steps']),
        'checkpoint_bytes': record.bytes,
        'checkpoint_sha256': record.sha256,
        'host_available_bytes_before_transfer': available_before,
    }


def backup_once(  # pylint: disable=too-many-arguments
    jobs: list[BackupJob],
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
    """Refresh every available run slot and revalidate all retained backups."""
    records = scan_remote(
        host,
        remote_run_root,
        remote_python,
        jobs,
        source_commit=source_commit,
        matrix_sha256=matrix_sha256,
    )
    rows = []
    backed_up_jobs = 0
    for job in jobs:
        record = records.get(job.job)
        if record is None:
            existing = validate_existing_backup(
                job,
                output_root / job.job,
                matrix_path=matrix_path,
                remote_run_root=Path(remote_run_root),
                source_commit=source_commit,
            )
            if existing is not None:
                raise ValueError(f'remote checkpoint disappeared after host backup: {job.job}')
            rows.append({'job': job.job, 'state': 'waiting_for_remote_checkpoint'})
            continue
        row = transfer_checkpoint(
            job,
            record,
            host=host,
            remote_run_root=remote_run_root,
            remote_python=remote_python,
            output_root=output_root,
            matrix_path=matrix_path,
            source_commit=source_commit,
            reserve_bytes=reserve_bytes,
        )
        rows.append(row)
        backed_up_jobs += 1
    return {
        'backup_version': 'shapo-p47-host-checkpoint-backup-v1',
        'observed_at_utc': utc_now(),
        'state': 'complete' if backed_up_jobs == len(jobs) else 'watching',
        'scientific_result': False,
        'used_for_checkpoint_or_configuration_selection': False,
        'inspected_scientific_metrics': False,
        'training_state_changed': False,
        'scientific_protocol_changed': False,
        'remote_host': host,
        'remote_run_root': remote_run_root,
        'total_jobs': len(jobs),
        'remote_checkpoint_jobs': len(records),
        'host_backup_jobs': backed_up_jobs,
        'newly_transferred_jobs': sum(row['state'] == 'transferred' for row in rows),
        'waiting_jobs': len(jobs) - backed_up_jobs,
        'rows': rows,
    }


def main() -> None:  # pylint: disable=too-many-locals
    """Poll immutable checkpoint identities and maintain one host copy per run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-host', default='seeta-nmb2')
    parser.add_argument('--remote-run-root', required=True)
    parser.add_argument('--remote-python-bin', required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--reserve-bytes', type=int, default=2 * 1024**3)
    parser.add_argument('--poll-seconds', type=float, default=21600.0)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.reserve_bytes < 0 or args.poll_seconds <= 0.0:
        parser.error('reserve bytes must be non-negative and poll seconds must be positive')
    if len(args.source_commit) != 40 or any(
        value not in '0123456789abcdef' for value in args.source_commit
    ):
        parser.error('--source-commit must be a full lowercase Git identity')
    for executable in ('ssh', 'rsync'):
        if shutil.which(executable) is None:
            parser.error(f'{executable} is unavailable')
    validate_remote_arguments(args.remote_host, args.remote_run_root)
    validate_remote_python(args.remote_python_bin)

    matrix_path = args.matrix.resolve()
    output_root = args.output_root.resolve()
    state_dir = args.state_dir.resolve()
    if not matrix_path.is_file():
        parser.error('frozen matrix is unavailable')
    if (
        output_root == state_dir
        or output_root in state_dir.parents
        or state_dir in output_root.parents
    ):
        parser.error('backup output and state directories must be separate')
    output_root.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    matrix_sha256 = sha256_file(matrix_path)
    backup_script_sha256 = sha256_file(Path(__file__).resolve())
    validator_path = Path(__file__).with_name('validate_shapo_p47_checkpoint_backup.py')
    validator_sha256 = sha256_file(validator_path)
    jobs = expected_jobs(matrix_path)

    with (state_dir / '.checkpoint-backup.lock').open('w', encoding='utf-8') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit('another host checkpoint backup holds the state lock') from error
        while True:
            try:
                if sha256_file(matrix_path) != matrix_sha256:
                    raise ValueError('locked matrix changed while checkpoint backup was running')
                if sha256_file(Path(__file__).resolve()) != backup_script_sha256:
                    raise ValueError('checkpoint backup source changed while running')
                if sha256_file(validator_path) != validator_sha256:
                    raise ValueError('checkpoint backup validator changed while running')
                snapshot = backup_once(
                    jobs,
                    host=args.remote_host,
                    remote_run_root=args.remote_run_root,
                    remote_python=args.remote_python_bin,
                    output_root=output_root,
                    matrix_path=matrix_path,
                    source_commit=args.source_commit,
                    reserve_bytes=args.reserve_bytes,
                    matrix_sha256=matrix_sha256,
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
                    'inspected_scientific_metrics': False,
                    'training_state_changed': False,
                    'scientific_protocol_changed': False,
                }
                atomic_json(state_dir / 'checkpoint_backup_alert.json', alert)
                raise
            atomic_json(state_dir / 'checkpoint_backup_latest.json', snapshot)
            append_jsonl(
                state_dir / 'checkpoint_backup_journal.jsonl',
                {
                    key: snapshot[key]
                    for key in (
                        'observed_at_utc',
                        'state',
                        'remote_checkpoint_jobs',
                        'host_backup_jobs',
                        'newly_transferred_jobs',
                        'waiting_jobs',
                        'scientific_result',
                        'used_for_checkpoint_or_configuration_selection',
                        'inspected_scientific_metrics',
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
                            'remote_checkpoint_jobs',
                            'host_backup_jobs',
                            'newly_transferred_jobs',
                            'waiting_jobs',
                        )
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if args.once:
                return
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()

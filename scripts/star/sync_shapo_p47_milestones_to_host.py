#!/usr/bin/env python3
"""Continuously mirror sealed SHAPO milestone evidence from the server to the host."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.star.watch_shapo_p47_milestones import (  # noqa: E402
    MilestoneJob,
    locked_jobs,
    milestone_audit_identity,
    sha256_file,
    validate_sealed_output,
)


SAFE_HOST = re.compile(r'^[A-Za-z0-9_.@-]+$')
SAFE_REMOTE_ROOT = re.compile(r'^/[A-Za-z0-9_./-]+$')
REMOTE_SCAN_SOURCE = r"""
import hashlib
import json
import sys
from pathlib import Path

EXPECTED = __EXPECTED__
EXCLUDED = {'checksums.sha256', 'milestone_audit_complete'}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def manifest(path):
    entries = {}
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        try:
            checksum, relative = line.split('  ', 1)
        except ValueError as error:
            raise ValueError(f'invalid checksum line {line_number}: {line!r}') from error
        candidate = Path(relative)
        if (
            len(checksum) != 64
            or any(character not in '0123456789abcdef' for character in checksum)
            or candidate.is_absolute()
            or '..' in candidate.parts
            or relative in entries
        ):
            raise ValueError(f'unsafe checksum line {line_number}: {line!r}')
        entries[relative] = checksum
    return entries


root = Path(sys.argv[1]).resolve()
if not root.is_dir():
    raise SystemExit(f'remote milestone root is absent: {root}')
for relative in EXPECTED:
    candidate = Path(relative)
    if candidate.is_absolute() or '..' in candidate.parts:
        raise ValueError(f'unsafe expected milestone path: {relative}')
    output = root / candidate
    if not (output / 'milestone_audit_complete').is_file():
        continue
    files = [path for path in output.rglob('*') if path.is_file()]
    if any(path.is_symlink() for path in output.rglob('*')):
        raise ValueError(f'remote milestone contains a symbolic link: {output}')
    if any(path.suffix == '.pt' for path in files):
        raise ValueError(f'remote compact milestone copied a checkpoint: {output}')
    checksum_path = output / 'checksums.sha256'
    if not checksum_path.is_file():
        raise ValueError(f'remote milestone lacks its checksum manifest: {output}')
    entries = manifest(checksum_path)
    actual = {
        path.relative_to(output).as_posix()
        for path in files
        if path.name not in EXCLUDED
    }
    if set(entries) != actual:
        raise ValueError(f'remote milestone checksum coverage changed: {output}')
    for name, expected in entries.items():
        if digest(output / name) != expected:
            raise ValueError(f'remote milestone checksum mismatch: {output / name}')
    print(
        json.dumps(
            {
                'relative': relative,
                'bytes': sum(path.stat().st_size for path in files),
                'file_count': len(files),
                'manifest_sha256': digest(checksum_path),
            },
            sort_keys=True,
        ),
    )
"""


@dataclass(frozen=True)
class RemoteSeal:
    """Portable identity of one remotely sealed compact milestone."""

    relative: str
    bytes: int
    file_count: int
    manifest_sha256: str


def atomic_json(path: Path, payload: Any) -> None:
    """Atomically replace one strict JSON status file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    temporary.write_text(
        f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Any) -> None:
    """Append and fsync one bounded transfer journal record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_remote_arguments(host: str, root: str) -> None:
    """Reject SSH destinations that could alter the intended read-only command."""
    if not SAFE_HOST.fullmatch(host):
        raise ValueError(f'unsafe SSH host or alias: {host!r}')
    if not SAFE_REMOTE_ROOT.fullmatch(root) or '..' in Path(root).parts:
        raise ValueError(f'unsafe remote milestone root: {root!r}')


def validate_remote_python(path: str) -> None:
    """Require one explicit absolute interpreter path for the remote checksum scan."""
    if not SAFE_REMOTE_ROOT.fullmatch(path) or '..' in Path(path).parts:
        raise ValueError(f'unsafe remote Python interpreter: {path!r}')


def expected_jobs(matrix_path: Path, output_root: Path) -> list[MilestoneJob]:
    """Expand the frozen matrix into the exact host-side transfer destinations."""
    matrix = yaml.safe_load(matrix_path.read_text(encoding='utf-8'))
    return locked_jobs(Path('/server-only/formal'), output_root, matrix)


def parse_remote_records(
    text: str,
    expected_relatives: set[str],
) -> dict[str, RemoteSeal]:
    """Parse and constrain the server scanner's JSONL response."""
    records: dict[str, RemoteSeal] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f'invalid remote scan JSON on line {line_number}') from error
        relative = str(payload.get('relative', ''))
        checksum = str(payload.get('manifest_sha256', ''))
        byte_count = int(payload.get('bytes', -1))
        file_count = int(payload.get('file_count', -1))
        if relative not in expected_relatives:
            raise ValueError(f'remote scan returned an unexpected milestone: {relative!r}')
        if relative in records:
            raise ValueError(f'remote scan returned a duplicate milestone: {relative!r}')
        if (
            len(checksum) != 64
            or any(character not in '0123456789abcdef' for character in checksum)
            or byte_count < 0
            or file_count <= 0
        ):
            raise ValueError(f'remote scan returned invalid metadata for {relative!r}')
        records[relative] = RemoteSeal(
            relative=relative,
            bytes=byte_count,
            file_count=file_count,
            manifest_sha256=checksum,
        )
    return records


def scan_remote(
    host: str,
    remote_root: str,
    relatives: list[str],
    remote_python: str,
) -> dict[str, RemoteSeal]:
    """Read and independently checksum all currently sealed remote milestones."""
    validate_remote_arguments(host, remote_root)
    validate_remote_python(remote_python)
    source = REMOTE_SCAN_SOURCE.replace('__EXPECTED__', json.dumps(relatives))
    remote_command = f'{shlex.quote(remote_python)} - {shlex.quote(remote_root)}'
    completed = subprocess.run(
        [
            'ssh',
            '-o',
            'BatchMode=yes',
            '-o',
            'ConnectTimeout=10',
            host,
            remote_command,
        ],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f'remote milestone scan failed with exit {completed.returncode}: '
            f'{completed.stderr.strip()}',
        )
    return parse_remote_records(completed.stdout, set(relatives))


def ensure_capacity(parent: Path, transfer_bytes: int, reserve_bytes: int) -> int:
    """Require enough host space for one staging copy plus a retained reserve."""
    available = shutil.disk_usage(parent).free
    required = transfer_bytes + reserve_bytes
    if required > available:
        raise RuntimeError(
            f'insufficient host space: need {required} bytes including reserve, '
            f'have {available}',
        )
    return available


def copy_remote_tree(
    host: str,
    remote_root: str,
    relative: str,
    destination: Path,
) -> None:
    """Copy one compact remote directory into an otherwise empty staging path."""
    validate_remote_arguments(host, remote_root)
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError(f'unsafe milestone relative path: {relative!r}')
    remote_path = f'{host}:{remote_root.rstrip("/")}/{relative}/'
    completed = subprocess.run(
        [
            'rsync',
            '-a',
            '--protect-args',
            '-e',
            'ssh -o BatchMode=yes -o ConnectTimeout=10',
            '--',
            remote_path,
            f'{destination}/',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f'rsync failed for {relative!r} with exit {completed.returncode}: '
            f'{completed.stderr.strip()}',
        )


def transfer_job(
    job: MilestoneJob,
    record: RemoteSeal,
    *,
    output_root: Path,
    host: str,
    remote_root: str,
    reserve_bytes: int,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    """Validate or atomically materialize one host-side compact milestone seal."""
    relative = job.output.relative_to(output_root).as_posix()
    if relative != record.relative:
        raise ValueError(f'job and remote milestone identity disagree: {job.key}')
    if job.output.exists():
        audit = validate_sealed_output(job, expected_identity=expected_identity)
        files = [path for path in job.output.rglob('*') if path.is_file()]
        local_bytes = sum(path.stat().st_size for path in files)
        manifest_sha256 = sha256_file(job.output / 'checksums.sha256')
        if (
            manifest_sha256 != record.manifest_sha256
            or len(files) != record.file_count
            or local_bytes != record.bytes
        ):
            raise ValueError(f'host and server milestone metadata disagree: {job.key}')
        return {
            'job': job.key,
            'state': 'already_present',
            'relative': relative,
            'bytes': local_bytes,
            'file_count': len(files),
            'manifest_sha256': manifest_sha256,
            'diagnostic_only': audit['diagnostic_only'],
        }

    job.output.parent.mkdir(parents=True, exist_ok=True)
    available_before = ensure_capacity(job.output.parent, record.bytes, reserve_bytes)
    staging = Path(tempfile.mkdtemp(prefix=f'.{job.output.name}.', dir=job.output.parent))
    try:
        copy_remote_tree(host, remote_root, relative, staging)
        staged_job = MilestoneJob(
            environment=job.environment,
            seed=job.seed,
            milestone=job.milestone,
            container=job.container,
            output=staging,
        )
        audit = validate_sealed_output(staged_job, expected_identity=expected_identity)
        files = [path for path in staging.rglob('*') if path.is_file()]
        local_bytes = sum(path.stat().st_size for path in files)
        local_manifest = sha256_file(staging / 'checksums.sha256')
        if (
            local_bytes != record.bytes
            or len(files) != record.file_count
            or local_manifest != record.manifest_sha256
        ):
            raise ValueError(f'copied milestone metadata changed in transit: {job.key}')
        os.replace(staging, job.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        'job': job.key,
        'state': 'transferred',
        'relative': relative,
        'bytes': record.bytes,
        'file_count': record.file_count,
        'manifest_sha256': record.manifest_sha256,
        'host_available_bytes_before_transfer': available_before,
        'diagnostic_only': audit['diagnostic_only'],
    }


def scan_once(  # pylint: disable=too-many-arguments
    jobs: list[MilestoneJob],
    *,
    output_root: Path,
    host: str,
    remote_root: str,
    remote_python: str,
    reserve_bytes: int,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    """Mirror every new server seal and revalidate every existing host seal."""
    relative_jobs = {
        job.output.relative_to(output_root).as_posix(): job
        for job in jobs
    }
    records = scan_remote(host, remote_root, list(relative_jobs), remote_python)
    rows = []
    for relative, job in relative_jobs.items():
        record = records.get(relative)
        if record is not None:
            rows.append(
                transfer_job(
                    job,
                    record,
                    output_root=output_root,
                    host=host,
                    remote_root=remote_root,
                    reserve_bytes=reserve_bytes,
                    expected_identity=expected_identity,
                ),
            )
        elif job.output.exists():
            validate_sealed_output(job, expected_identity=expected_identity)
            raise ValueError(
                f'remote milestone seal disappeared after its host copy was verified: {job.key}',
            )
        else:
            rows.append({'job': job.key, 'state': 'waiting_for_remote_seal'})
    local_sealed = sum(row['state'] != 'waiting_for_remote_seal' for row in rows)
    transferred = sum(row['state'] == 'transferred' for row in rows)
    return {
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'state': 'complete' if local_sealed == len(jobs) else 'watching',
        'scientific_protocol_changed': False,
        'training_state_changed': False,
        'used_for_checkpoint_or_configuration_selection': False,
        'remote_host': host,
        'remote_milestone_root': remote_root,
        'total_jobs': len(jobs),
        'remote_sealed_jobs': len(records),
        'host_sealed_jobs': local_sealed,
        'newly_transferred_jobs': transferred,
        'waiting_jobs': len(jobs) - local_sealed,
        'rows': rows,
    }


def main() -> None:  # pylint: disable=too-many-locals
    """Poll the remote seal root without mutating any formal training container."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-host', default='seeta-nmb2')
    parser.add_argument('--remote-milestone-root', required=True)
    parser.add_argument('--remote-python-bin', required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--audit-script', type=Path, default=Path(__file__).with_name(
        'audit_shapo_p47_milestone.py',
    ))
    parser.add_argument('--reserve-bytes', type=int, default=2 * 1024**3)
    parser.add_argument('--poll-seconds', type=float, default=600.0)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if len(args.source_commit) != 40:
        parser.error('--source-commit must be a full commit identity')
    if args.reserve_bytes < 0 or args.poll_seconds <= 0.0:
        parser.error('reserve bytes must be non-negative and poll seconds must be positive')
    for executable in ('ssh', 'rsync'):
        if shutil.which(executable) is None:
            parser.error(f'{executable} is unavailable')
    validate_remote_arguments(args.remote_host, args.remote_milestone_root)
    validate_remote_python(args.remote_python_bin)

    matrix_path = args.matrix.resolve()
    audit_script = args.audit_script.resolve()
    validator_path = audit_script.with_name('validate_shapo_p47_run.py')
    if not matrix_path.is_file() or not audit_script.is_file() or not validator_path.is_file():
        parser.error('matrix, milestone auditor, or strict validator is unavailable')
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_dir = (args.state_dir or output_root.parent / 'milestone_sync_state').resolve()
    if state_dir == output_root or output_root in state_dir.parents:
        parser.error('--state-dir must not be inside the milestone output tree')
    state_dir.mkdir(parents=True, exist_ok=True)

    matrix_sha256 = sha256_file(matrix_path)
    audit_sha256 = sha256_file(audit_script)
    validator_sha256 = sha256_file(validator_path)
    sync_sha256 = sha256_file(Path(__file__).resolve())
    expected_identity = milestone_audit_identity(
        matrix_path,
        args.source_commit,
        audit_script,
    )
    jobs = expected_jobs(matrix_path, output_root)
    identities = {
        'matrix_sha256': matrix_sha256,
        'audit_script_sha256': audit_sha256,
        'strict_evaluation_validator_sha256': validator_sha256,
        'sync_script_sha256': sync_sha256,
        'source_commit': args.source_commit,
    }

    with (state_dir / '.milestone-sync.lock').open('w', encoding='utf-8') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit('another host milestone sync holds the output lock') from error
        while True:
            try:
                if sha256_file(matrix_path) != matrix_sha256:
                    raise ValueError('locked matrix changed while milestone sync was running')
                if sha256_file(audit_script) != audit_sha256:
                    raise ValueError('milestone auditor changed while sync was running')
                if sha256_file(validator_path) != validator_sha256:
                    raise ValueError('strict validator changed while sync was running')
                if sha256_file(Path(__file__).resolve()) != sync_sha256:
                    raise ValueError('host milestone sync source changed while running')
                snapshot = scan_once(
                    jobs,
                    output_root=output_root,
                    host=args.remote_host,
                    remote_root=args.remote_milestone_root,
                    remote_python=args.remote_python_bin,
                    reserve_bytes=args.reserve_bytes,
                    expected_identity=expected_identity,
                )
                snapshot.update(identities)
            except Exception as error:
                alert = {
                    'observed_at_utc': datetime.now(timezone.utc).isoformat(),
                    'state': 'alert',
                    'error_type': type(error).__name__,
                    'error': str(error),
                    'scientific_protocol_changed': False,
                    'training_state_changed': False,
                }
                atomic_json(state_dir / 'milestone_sync_alert.json', alert)
                raise
            atomic_json(state_dir / 'milestone_sync_latest.json', snapshot)
            append_jsonl(
                state_dir / 'milestone_sync_journal.jsonl',
                {
                    key: snapshot[key]
                    for key in (
                        'observed_at_utc',
                        'state',
                        'remote_sealed_jobs',
                        'host_sealed_jobs',
                        'newly_transferred_jobs',
                        'waiting_jobs',
                        'scientific_protocol_changed',
                        'training_state_changed',
                        'used_for_checkpoint_or_configuration_selection',
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
                            'remote_sealed_jobs',
                            'host_sealed_jobs',
                            'newly_transferred_jobs',
                            'waiting_jobs',
                        )
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if snapshot['state'] == 'complete':
                (state_dir / 'milestone_sync_complete').touch()
                return
            if args.once:
                return
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Wait for sealed SHAPO final evidence, then mirror it to the host exactly once."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFE_HOST = re.compile(r'^[A-Za-z0-9_.@-]+$')
SAFE_REMOTE_PATH = re.compile(r'^/[A-Za-z0-9_./-]+$')
FULL_COMMIT = re.compile(r'^[0-9a-f]{40}$')
TRANSFER_EXCLUSIONS = {'transfer_checksums.sha256', 'transfer_complete'}
PUBLIC_EXCLUSIONS = {'checksums.sha256', 'finalization_complete'}
REMOTE_HEALTH_FILES = (
    'completion_health.json',
    'latest_health.json',
    'health_journal.jsonl',
    'launch_audit.json',
    'persistence_audit.json',
)
REMOTE_PROBE = r"""
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1]).resolve()
completion_root = Path(sys.argv[2]).resolve()
source_commit = sys.argv[3]
alert = completion_root / 'completion_watch_alert.json'
if alert.is_file():
    print(json.dumps({'state': 'alert', 'error': alert.read_text(encoding='utf-8')}))
    raise SystemExit(0)
public = run_root / 'public_artifacts'
if not public.exists():
    print(json.dumps({'state': 'waiting'}))
    raise SystemExit(0)
required = [
    public / 'finalization_complete',
    public / 'checksums.sha256',
    completion_root / 'completion_watch_complete.json',
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    print(json.dumps({'state': 'alert', 'error': f'incomplete final output: {missing}'}))
    raise SystemExit(0)
watchdogs = sorted(path for path in run_root.glob('watchdog-*') if path.is_dir())
if len(watchdogs) != 1 or not (watchdogs[0] / 'completion_health.json').is_file():
    print(json.dumps({'state': 'alert', 'error': 'watchdog completion health is unavailable'}))
    raise SystemExit(0)
completion = json.loads(
    (completion_root / 'completion_watch_complete.json').read_text(encoding='utf-8'),
)
expected_completion = {
    'state': 'complete',
    'formal_source_commit': source_commit,
    'scientific_protocol_changed': False,
    'training_state_changed': False,
    'used_for_checkpoint_or_configuration_selection': False,
}
for field, expected in expected_completion.items():
    if completion.get(field) != expected:
        print(
            json.dumps(
                {
                    'state': 'alert',
                    'error': f'completion receipt field {field!r} changed',
                },
            ),
        )
        raise SystemExit(0)
final = completion.get('final')
expected_final = {
    'status': 'passed',
    'source_commit': source_commit,
    'statistical_unit': 'training_seed',
    'n_training_seeds_per_environment': 5,
    'evaluation_artifact_count': 60,
    'provenance_artifact_count': 80,
    'trajectory_artifact_count': 60,
    'checkpoint_artifact_count': 120,
    'strict_validation_count': 20,
}
if not isinstance(final, dict):
    print(json.dumps({'state': 'alert', 'error': 'completion final receipt is invalid'}))
    raise SystemExit(0)
for field, expected in expected_final.items():
    if final.get(field) != expected:
        print(
            json.dumps(
                {'state': 'alert', 'error': f'completion final field {field!r} changed'},
            ),
        )
        raise SystemExit(0)
health = json.loads(
    (watchdogs[0] / 'completion_health.json').read_text(encoding='utf-8'),
)
expected_health = {
    'state': 'complete',
    'alerts': [],
    'formal_matrix_complete': True,
    'completed_runs': 20,
    'failed_runs': 0,
    'stale_runs': 0,
    'total_runs': 20,
    'active_workers': 0,
}
for field, expected in expected_health.items():
    if health.get(field) != expected:
        print(
            json.dumps(
                {'state': 'alert', 'error': f'completion health field {field!r} changed'},
            ),
        )
        raise SystemExit(0)
if health.get('completed_environment_steps') != health.get('planned_environment_steps'):
    print(json.dumps({'state': 'alert', 'error': 'completion health budget is incomplete'}))
    raise SystemExit(0)
print(
    json.dumps(
        {
            'state': 'ready',
            'public_root': str(public),
            'watchdog_root': str(watchdogs[0]),
        },
        sort_keys=True,
    ),
)
"""


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    """Atomically replace one strict JSON receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    temporary.write_text(
        f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Any) -> None:
    """Append and fsync one compact final-sync journal row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_remote_arguments(host: str, run_root: str, completion_root: str) -> None:
    """Reject SSH arguments that could escape the expected read-only command."""
    if not SAFE_HOST.fullmatch(host):
        raise ValueError(f'unsafe SSH host or alias: {host!r}')
    for label, value in (('run root', run_root), ('completion root', completion_root)):
        if not SAFE_REMOTE_PATH.fullmatch(value) or '..' in Path(value).parts:
            raise ValueError(f'unsafe remote {label}: {value!r}')


def validate_remote_python(path: str) -> None:
    """Require one explicit absolute interpreter for the remote read-only probe."""
    if not SAFE_REMOTE_PATH.fullmatch(path) or '..' in Path(path).parts:
        raise ValueError(f'unsafe remote Python interpreter: {path!r}')


def probe_remote(
    host: str,
    run_root: str,
    completion_root: str,
    remote_python: str,
    source_commit: str,
) -> dict[str, Any]:
    """Read the remote finalization and watchdog seals without changing remote state."""
    validate_remote_arguments(host, run_root, completion_root)
    validate_remote_python(remote_python)
    command = (
        f'{shlex.quote(remote_python)} - {shlex.quote(run_root)} '
        f'{shlex.quote(completion_root)} {shlex.quote(source_commit)}'
    )
    completed = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, command],
        input=REMOTE_PROBE,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f'remote final-sync probe failed with exit {completed.returncode}: '
            f'{completed.stderr.strip()}',
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError('remote final-sync probe did not return strict JSON') from error
    if not isinstance(payload, dict) or payload.get('state') not in {'waiting', 'ready', 'alert'}:
        raise ValueError(f'remote final-sync probe returned an invalid state: {payload!r}')
    if payload['state'] == 'alert':
        raise RuntimeError(f'remote finalization recorded an alert: {payload.get("error")}')
    return payload


def parse_checksum_manifest(path: Path) -> dict[str, str]:
    """Parse one two-space SHA-256 manifest with safe relative paths."""
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        try:
            digest, relative = line.split('  ', 1)
        except ValueError as error:
            raise ValueError(f'invalid checksum line {line_number}: {line!r}') from error
        normalized = relative[2:] if relative.startswith('./') else relative
        candidate = Path(normalized)
        if (
            len(digest) != 64
            or any(character not in '0123456789abcdef' for character in digest)
            or candidate.is_absolute()
            or '..' in candidate.parts
            or normalized in entries
        ):
            raise ValueError(f'unsafe checksum line {line_number}: {line!r}')
        entries[normalized] = digest
    return entries


def validate_checksum_tree(
    root: Path,
    manifest_name: str,
    exclusions: set[str],
) -> dict[str, Any]:
    """Require complete checksum coverage and byte identity below one tree."""
    manifest_path = root / manifest_name
    entries = parse_checksum_manifest(manifest_path)
    if any(path.is_symlink() for path in root.rglob('*')):
        raise ValueError(f'checksum tree contains a symbolic link: {root}')
    if any(
        path.is_file()
        and path.relative_to(root).as_posix() not in exclusions
        and path.name in exclusions
        for path in root.rglob('*')
    ):
        raise ValueError(f'checksum exclusion appears below the tree root: {root}')
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob('*')
        if path.is_file() and path.relative_to(root).as_posix() not in exclusions
    }
    if set(entries) != actual:
        raise ValueError(f'checksum coverage mismatch: {root}')
    for relative, expected in entries.items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f'checksum identity mismatch: {root / relative}')
    return {
        'manifest_sha256': sha256_file(manifest_path),
        'covered_files': len(entries),
    }


def parse_transfer_manifest(path: Path) -> dict[str, str]:
    """Parse the final-sync wrapper's exact tab-separated transfer identity."""
    records: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        try:
            key, value = line.split('\t', 1)
        except ValueError as error:
            raise ValueError(f'invalid transfer manifest line {line_number}') from error
        if not key or key in records:
            raise ValueError(f'duplicate transfer manifest key on line {line_number}')
        records[key] = value
    return records


def validate_runtime_health(output_root: Path, remote_run_root: str) -> None:
    """Validate the transferred watchdog receipt and its per-file source manifest."""
    health_root = output_root / 'runtime_health'
    manifest_path = output_root / 'runtime_health_manifest.csv'
    if not health_root.is_dir() or not manifest_path.is_file():
        raise ValueError('host final transfer lacks runtime health evidence')
    with manifest_path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['artifact', 'remote_path', 'sha256', 'bytes']:
            raise ValueError('runtime health manifest columns changed')
        rows = list(reader)
    if len(rows) != len(REMOTE_HEALTH_FILES):
        raise ValueError('runtime health manifest coverage changed')
    seen: set[str] = set()
    for row in rows:
        name = row['artifact']
        path = health_root / name
        remote_path = Path(row['remote_path'])
        if (
            name not in REMOTE_HEALTH_FILES
            or name in seen
            or not path.is_file()
            or remote_path.name != name
            or remote_path.parent.parent.as_posix() != remote_run_root
            or not remote_path.parent.name.startswith('watchdog-')
            or row['sha256'] != sha256_file(path)
            or row['bytes'] != str(path.stat().st_size)
        ):
            raise ValueError(f'runtime health evidence changed for {name!r}')
        seen.add(name)
    health = json.loads((health_root / 'completion_health.json').read_text(encoding='utf-8'))
    expected = {
        'state': 'complete',
        'alerts': [],
        'formal_matrix_complete': True,
        'completed_runs': 20,
        'failed_runs': 0,
        'stale_runs': 0,
        'total_runs': 20,
        'active_workers': 0,
    }
    if not isinstance(health, dict):
        raise ValueError('runtime completion health is not a JSON object')
    for field, value in expected.items():
        if health.get(field) != value:
            raise ValueError(f'runtime completion health field {field!r} changed')
    if health.get('completed_environment_steps') != health.get('planned_environment_steps'):
        raise ValueError('runtime completion health budget is incomplete')


def validate_local_output(
    output_root: Path,
    *,
    remote_host: str,
    remote_run_root: str,
    source_commit: str,
) -> dict[str, Any]:
    """Revalidate one complete host transfer without trusting its completion marker alone."""
    if (
        output_root.is_symlink()
        or not output_root.is_dir()
        or not (output_root / 'transfer_complete').is_file()
    ):
        raise ValueError(f'incomplete host final transfer: {output_root}')
    transfer = validate_checksum_tree(
        output_root,
        'transfer_checksums.sha256',
        TRANSFER_EXCLUSIONS,
    )
    public = output_root / 'public_artifacts'
    if not (public / 'finalization_complete').is_file():
        raise ValueError('host final transfer lacks the server finalization seal')
    public_identity = validate_checksum_tree(public, 'checksums.sha256', PUBLIC_EXCLUSIONS)
    records = parse_transfer_manifest(output_root / 'transfer_manifest.tsv')
    expected = {
        'remote_host': remote_host,
        'remote_run_root': remote_run_root,
        'public_checksum_manifest_sha256': public_identity['manifest_sha256'],
    }
    for field, value in expected.items():
        if records.get(field) != value:
            raise ValueError(f'host final transfer field {field!r} changed')
    validate_runtime_health(output_root, remote_run_root)
    if list(output_root.rglob('*.pt')):
        raise ValueError('host final transfer copied server-only checkpoints')
    summary_path = public / 'shapo_p47_final_summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if not isinstance(summary, dict):
        raise ValueError('host final summary is not a JSON object')
    expected_summary = {
        'source_commit': source_commit,
        'statistical_unit': 'training_seed',
        'n_training_seeds_per_environment': 5,
        'evaluation_artifact_count': 60,
        'provenance_artifact_count': 80,
        'trajectory_artifact_count': 60,
        'checkpoint_artifact_count': 120,
    }
    for field, value in expected_summary.items():
        if summary.get(field) != value:
            raise ValueError(f'host final summary field {field!r} changed')
    return {
        'status': 'passed',
        'output_root': str(output_root),
        'bytes': sum(path.stat().st_size for path in output_root.rglob('*') if path.is_file()),
        'file_count': sum(path.is_file() for path in output_root.rglob('*')),
        'transfer_manifest_sha256': transfer['manifest_sha256'],
        'public_manifest_sha256': public_identity['manifest_sha256'],
        'remote_host': remote_host,
        'remote_run_root': remote_run_root,
    }


def invoke_sync(
    sync_script: Path,
    remote_host: str,
    remote_run_root: str,
    output_root: Path,
    reserve_bytes: int,
) -> dict[str, Any]:
    """Invoke the existing capacity-gated atomic final-sync wrapper once."""
    command = [
        'bash',
        str(sync_script),
        remote_host,
        remote_run_root,
        str(output_root),
        str(reserve_bytes),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f'final host sync failed with exit {completed.returncode}; '
            f'stdout={completed.stdout.strip()!r}; stderr={completed.stderr.strip()!r}',
        )
    return {
        'command': command,
        'stdout': completed.stdout.strip(),
        'stderr': completed.stderr.strip(),
    }


def sync_once(  # pylint: disable=too-many-arguments
    *,
    remote_host: str,
    remote_run_root: str,
    remote_completion_root: str,
    remote_python: str,
    source_commit: str,
    output_root: Path,
    reserve_bytes: int,
    sync_script: Path,
) -> dict[str, Any]:
    """Wait without mutation, or transfer and validate the sealed final evidence once."""
    if output_root.exists():
        local = validate_local_output(
            output_root,
            remote_host=remote_host,
            remote_run_root=remote_run_root,
            source_commit=source_commit,
        )
        remote = probe_remote(
            remote_host,
            remote_run_root,
            remote_completion_root,
            remote_python,
            source_commit,
        )
        if remote['state'] != 'ready':
            raise ValueError('host final output exists while remote final seal is unavailable')
        return {'state': 'complete', 'transfer': 'already_present', 'local': local}
    remote = probe_remote(
        remote_host,
        remote_run_root,
        remote_completion_root,
        remote_python,
        source_commit,
    )
    if remote['state'] == 'waiting':
        return {
            'state': 'waiting',
            'remote_state': 'waiting',
            'scientific_protocol_changed': False,
            'training_state_changed': False,
            'used_for_checkpoint_or_configuration_selection': False,
        }
    invocation = invoke_sync(
        sync_script,
        remote_host,
        remote_run_root,
        output_root,
        reserve_bytes,
    )
    local = validate_local_output(
        output_root,
        remote_host=remote_host,
        remote_run_root=remote_run_root,
        source_commit=source_commit,
    )
    return {
        'state': 'complete',
        'transfer': 'invoked',
        'invocation': invocation,
        'local': local,
        'scientific_protocol_changed': False,
        'training_state_changed': False,
        'used_for_checkpoint_or_configuration_selection': False,
    }


def main() -> None:  # pylint: disable=too-many-locals
    """Poll finalization evidence and terminate after one verified host transfer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-host', default='seeta-nmb2')
    parser.add_argument('--remote-run-root', required=True)
    parser.add_argument('--remote-completion-root', required=True)
    parser.add_argument('--remote-python-bin', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--reserve-bytes', type=int, default=2 * 1024**3)
    parser.add_argument('--poll-seconds', type=float, default=600.0)
    parser.add_argument(
        '--sync-script',
        type=Path,
        default=Path(__file__).resolve().with_name('sync_shapo_p47_final_to_host.sh'),
    )
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.reserve_bytes < 0 or args.poll_seconds <= 0.0:
        parser.error('reserve bytes must be non-negative and poll seconds must be positive')
    validate_remote_arguments(
        args.remote_host,
        args.remote_run_root,
        args.remote_completion_root,
    )
    validate_remote_python(args.remote_python_bin)
    if not FULL_COMMIT.fullmatch(args.source_commit):
        parser.error('--source-commit must be a full commit identity')
    for executable in ('ssh', 'rsync', 'sha256sum'):
        if shutil.which(executable) is None:
            parser.error(f'{executable} is unavailable')

    output_root = args.output_root.resolve()
    state_dir = args.state_dir.resolve()
    sync_script = args.sync_script.resolve()
    if (
        output_root == state_dir
        or output_root in state_dir.parents
        or state_dir in output_root.parents
    ):
        parser.error('--state-dir must not overlap final output')
    if not sync_script.is_file():
        parser.error('--sync-script is unavailable')
    state_dir.mkdir(parents=True, exist_ok=True)
    immutable_paths = {
        'final_sync_watcher': Path(__file__).resolve(),
        'sync_script': sync_script,
    }
    immutable_hashes = {name: sha256_file(path) for name, path in immutable_paths.items()}
    identities = {
        **{f'{name}_sha256': digest for name, digest in immutable_hashes.items()},
        'scientific_protocol_changed': False,
        'training_state_changed': False,
        'used_for_checkpoint_or_configuration_selection': False,
    }

    with (state_dir / '.final-sync-watch.lock').open('w', encoding='utf-8') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit('another final-sync watcher holds the state lock') from error
        while True:
            observed_at = datetime.now(timezone.utc).isoformat()
            try:
                for name, path in immutable_paths.items():
                    if sha256_file(path) != immutable_hashes[name]:
                        raise ValueError(f'final-sync dependency changed while running: {name}')
                snapshot = sync_once(
                    remote_host=args.remote_host,
                    remote_run_root=args.remote_run_root,
                    remote_completion_root=args.remote_completion_root,
                    remote_python=args.remote_python_bin,
                    source_commit=args.source_commit,
                    output_root=output_root,
                    reserve_bytes=args.reserve_bytes,
                    sync_script=sync_script,
                )
                snapshot.update({'observed_at_utc': observed_at, **identities})
            except Exception as error:
                alert = {
                    'observed_at_utc': observed_at,
                    'state': 'alert',
                    'error_type': type(error).__name__,
                    'error': str(error),
                    **identities,
                }
                atomic_json(state_dir / 'final_sync_watch_alert.json', alert)
                raise
            atomic_json(state_dir / 'final_sync_watch_latest.json', snapshot)
            append_jsonl(
                state_dir / 'final_sync_watch_journal.jsonl',
                {
                    'observed_at_utc': observed_at,
                    'state': snapshot['state'],
                    'scientific_protocol_changed': False,
                    'training_state_changed': False,
                    'used_for_checkpoint_or_configuration_selection': False,
                },
            )
            print(
                json.dumps(
                    {'observed_at_utc': observed_at, 'state': snapshot['state']},
                    sort_keys=True,
                ),
                flush=True,
            )
            if snapshot['state'] == 'complete':
                atomic_json(state_dir / 'final_sync_watch_complete.json', snapshot)
                return
            if args.once:
                return
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()

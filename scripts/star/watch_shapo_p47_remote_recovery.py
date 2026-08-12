#!/usr/bin/env python3
"""Keep the frozen remote SHAPO process graph recoverable across server restarts."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FULL_COMMIT = re.compile(r'^[0-9a-f]{40}$')
FULL_SHA256 = re.compile(r'^[0-9a-f]{64}$')
SAFE_HOST = re.compile(r'^[A-Za-z0-9_.@-]+$')
REMOTE_AGENT = Path(__file__).resolve().with_name('shapo_p47_remote_recovery_agent.py')
VERSION = 'shapo-p47-host-recovery-watch-v1'


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
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
    """Append and fsync one compact recovery watcher row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_remote_path(value: str, label: str) -> str:
    """Require one absolute remote path with no traversal components."""
    path = Path(value)
    if not path.is_absolute() or path == Path('/') or '..' in path.parts:
        raise ValueError(f'{label} must be a safe absolute remote path')
    return value


def parse_lock(value: str, *, digest: bool) -> tuple[str, str]:
    """Parse one ``absolute-path=identity`` lock from the command line."""
    try:
        path, identity = value.rsplit('=', 1)
    except ValueError as error:
        raise ValueError(f'invalid path identity lock: {value!r}') from error
    validate_remote_path(path, 'locked path')
    pattern = FULL_SHA256 if digest else FULL_COMMIT
    if not pattern.fullmatch(identity):
        raise ValueError(f'invalid identity in lock: {value!r}')
    return path, identity


def parse_named_digest(value: str) -> tuple[str, str]:
    """Parse one safe ``name=sha256`` identity."""
    try:
        name, digest = value.rsplit('=', 1)
    except ValueError as error:
        raise ValueError(f'invalid named digest lock: {value!r}') from error
    if not re.fullmatch(r'[a-z0-9-]+', name) or not FULL_SHA256.fullmatch(digest):
        raise ValueError(f'invalid named digest lock: {value!r}')
    return name, digest


def build_payload(args: argparse.Namespace, *, allow_recovery: bool) -> dict[str, Any]:
    """Build the complete remote recovery contract from validated CLI arguments."""
    path_fields = (
        'remote_python_bin',
        'formal_source',
        'run_root',
        'smoke_root',
        'monitor_source',
        'monitor_output',
        'milestone_source',
        'milestone_output',
        'validation_source',
        'completion_state',
        'recovery_root',
    )
    for field in path_fields:
        validate_remote_path(getattr(args, field), field)
    worktrees = {
        args.formal_source: args.formal_source_commit,
        args.monitor_source: args.monitor_source_commit,
        args.milestone_source: args.milestone_source_commit,
        args.validation_source: args.validation_source_commit,
    }
    if len(worktrees) != 4 or any(not FULL_COMMIT.fullmatch(value) for value in worktrees.values()):
        raise ValueError('four distinct locked worktree commits are required')
    file_locks = dict(parse_lock(value, digest=True) for value in args.locked_file)
    expected_files = {
        args.remote_python_bin,
        f'{args.formal_source}/scripts/star/launch_shapo_p47_formal.sh',
        f'{args.formal_source}/configs/star/shapo_p47_four_env.yaml',
        f'{args.smoke_root}/smoke_gate.json',
        f'{args.monitor_source}/scripts/star/watch_shapo_p47_formal.py',
        f'{args.monitor_source}/scripts/star/status_shapo_p47_formal.py',
        f'{args.milestone_source}/scripts/star/watch_shapo_p47_milestones.py',
        f'{args.milestone_source}/scripts/star/audit_shapo_p47_milestone.py',
        f'{args.milestone_source}/scripts/star/validate_shapo_p47_run.py',
        f'{args.validation_source}/scripts/star/watch_shapo_p47_completion.py',
        f'{args.validation_source}/scripts/star/finalize_shapo_p47_formal.sh',
        f'{args.validation_source}/scripts/star/finalize_shapo_p47_formal.py',
        f'{args.validation_source}/configs/star/shapo_p47_four_env.yaml',
        f'{args.validation_source}/scripts/star/audit_shapo_p47_milestone.py',
        f'{args.validation_source}/scripts/star/validate_shapo_p47_run.py',
    }
    if len(file_locks) != len(args.locked_file) or set(file_locks) != expected_files:
        raise ValueError('locked remote files must be nonempty and unique')
    resolved_configs = dict(parse_named_digest(value) for value in args.resolved_config_sha256)
    if set(resolved_configs) != {'point-goal', 'point-button', 'ant', 'walker2d'}:
        raise ValueError('exactly four resolved formal config identities are required')
    if not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib < 0.0:
        raise ValueError('minimum free GiB must be non-negative')
    if not args.expected_torch_version or not args.expected_cuda_device:
        raise ValueError('expected Torch and CUDA device identities are required')
    return {
        'version': 'shapo-p47-remote-recovery-v1',
        'allow_recovery': allow_recovery,
        'python_bin': args.remote_python_bin,
        'formal_source': args.formal_source,
        'run_root': args.run_root,
        'smoke_root': args.smoke_root,
        'monitor_source': args.monitor_source,
        'monitor_output': args.monitor_output,
        'milestone_source': args.milestone_source,
        'milestone_output': args.milestone_output,
        'validation_source': args.validation_source,
        'completion_state': args.completion_state,
        'recovery_root': args.recovery_root,
        'worktree_commits': worktrees,
        'file_sha256': file_locks,
        'resolved_config_sha256': resolved_configs,
        'expected_torch_version': args.expected_torch_version,
        'expected_cuda_device': args.expected_cuda_device,
        'minimum_free_bytes': int(args.minimum_free_gib * 1024**3),
    }


def invoke_remote(
    *,
    host: str,
    remote_python: str,
    payload: dict[str, Any],
    agent_source: str,
) -> dict[str, Any]:
    """Execute one agent transaction over noninteractive SSH and parse strict JSON."""
    if not SAFE_HOST.fullmatch(host):
        raise ValueError(f'unsafe SSH host or alias: {host!r}')
    validate_remote_path(remote_python, 'remote Python')
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8'),
    ).decode('ascii')
    remote_command = f'{shlex.quote(remote_python)} - {shlex.quote(encoded)}'
    completed = subprocess.run(
        [
            'ssh',
            '-o',
            'BatchMode=yes',
            '-o',
            'ConnectTimeout=15',
            host,
            remote_command,
        ],
        input=agent_source,
        capture_output=True,
        text=True,
        check=False,
    )
    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 and not rows:
        raise ConnectionError(
            f'remote recovery SSH failed with exit {completed.returncode}: '
            f'{completed.stderr.strip()}',
        )
    if len(rows) != 1:
        raise RuntimeError(f'remote recovery returned {len(rows)} stdout records')
    try:
        result = json.loads(rows[0])
    except json.JSONDecodeError as error:
        raise RuntimeError('remote recovery returned invalid JSON') from error
    if not isinstance(result, dict) or result.get('version') != payload['version']:
        raise RuntimeError('remote recovery response identity changed')
    if completed.returncode != 0 or result.get('state') == 'alert':
        raise RemoteRecoveryError(result, completed.stderr.strip())
    if result.get('state') not in {
        'running',
        'recovered',
        'recovery_deferred',
        'recovery_required',
        'complete',
        'busy',
    }:
        raise RuntimeError(f'remote recovery returned invalid state: {result.get("state")!r}')
    return result


class RemoteRecoveryError(RuntimeError):
    """A machine-readable fail-closed response from the remote agent."""

    def __init__(self, payload: dict[str, Any], stderr: str = '') -> None:
        self.payload = payload
        self.stderr = stderr
        message = f"remote recovery alert: {payload.get('error', 'unknown error')}"
        if stderr:
            message = f'{message}; stderr={stderr}'
        super().__init__(message)


def compact_record(result: dict[str, Any], identities: dict[str, Any]) -> dict[str, Any]:
    """Build a bounded append-only monitoring record."""
    return {
        'observed_at_utc': result['observed_at_utc'],
        'state': result['state'],
        'formal_matrix_complete': result.get('formal_matrix_complete'),
        'formal_worker_count': len(result.get('formal_worker_pids', [])),
        'recoverable_checkpoint_runs': result.get('recoverable_checkpoint_runs'),
        'needed_processes': result.get('needed_processes', []),
        'deferred_processes': result.get('deferred_processes', []),
        'launched_processes': sorted(result.get('launched_processes', {})),
        'adopted_pid_files': result.get('adopted_pid_files', []),
        'scientific_protocol_changed': False,
        'training_configuration_changed': False,
        'used_for_checkpoint_or_configuration_selection': False,
        **identities,
    }


def main() -> None:  # pylint: disable=too-many-locals
    """Continuously inspect and recover the exact remote process graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-host', required=True)
    parser.add_argument('--remote-python-bin', required=True)
    parser.add_argument('--formal-source', required=True)
    parser.add_argument('--formal-source-commit', required=True)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--smoke-root', required=True)
    parser.add_argument('--monitor-source', required=True)
    parser.add_argument('--monitor-source-commit', required=True)
    parser.add_argument('--monitor-output', required=True)
    parser.add_argument('--milestone-source', required=True)
    parser.add_argument('--milestone-source-commit', required=True)
    parser.add_argument('--milestone-output', required=True)
    parser.add_argument('--validation-source', required=True)
    parser.add_argument('--validation-source-commit', required=True)
    parser.add_argument('--completion-state', required=True)
    parser.add_argument('--recovery-root', required=True)
    parser.add_argument('--locked-file', action='append', default=[], metavar='PATH=SHA256')
    parser.add_argument(
        '--resolved-config-sha256',
        action='append',
        default=[],
        metavar='SLUG=SHA256',
    )
    parser.add_argument('--expected-torch-version', required=True)
    parser.add_argument('--expected-cuda-device', required=True)
    parser.add_argument('--minimum-free-gib', type=float, default=4.0)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=300.0)
    parser.add_argument('--observe-only', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not SAFE_HOST.fullmatch(args.remote_host):
        parser.error('remote host or alias is unsafe')
    if args.poll_seconds <= 0.0:
        parser.error('--poll-seconds must be positive')
    if not REMOTE_AGENT.is_file():
        parser.error(f'remote recovery agent is unavailable: {REMOTE_AGENT}')
    try:
        payload = build_payload(args, allow_recovery=not args.observe_only)
    except ValueError as error:
        parser.error(str(error))

    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    watcher_path = Path(__file__).resolve()
    immutable_hashes = {
        'host_recovery_watcher_sha256': sha256_file(watcher_path),
        'remote_recovery_agent_sha256': sha256_file(REMOTE_AGENT),
        'remote_recovery_contract_sha256': hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8'),
        ).hexdigest(),
    }
    identities = {
        'watch_version': VERSION,
        'remote_host': args.remote_host,
        'observe_only': args.observe_only,
        **immutable_hashes,
    }
    agent_source = REMOTE_AGENT.read_text(encoding='utf-8')
    with (state_dir / '.remote-recovery-watch.lock').open('w', encoding='utf-8') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit('another remote recovery watcher holds the state lock') from error
        while True:
            observed_at = datetime.now(timezone.utc).isoformat()
            try:
                if sha256_file(watcher_path) != immutable_hashes['host_recovery_watcher_sha256']:
                    raise ValueError('host recovery watcher source changed while running')
                if sha256_file(REMOTE_AGENT) != immutable_hashes['remote_recovery_agent_sha256']:
                    raise ValueError('remote recovery agent source changed while running')
                result = invoke_remote(
                    host=args.remote_host,
                    remote_python=args.remote_python_bin,
                    payload=payload,
                    agent_source=agent_source,
                )
                result.update(identities)
            except ConnectionError as error:
                unreachable = {
                    'observed_at_utc': observed_at,
                    'state': 'unreachable',
                    'error_type': type(error).__name__,
                    'error': str(error),
                    'scientific_protocol_changed': False,
                    'training_configuration_changed': False,
                    'used_for_checkpoint_or_configuration_selection': False,
                    **identities,
                }
                atomic_json(state_dir / 'remote_recovery_watch_latest.json', unreachable)
                append_jsonl(state_dir / 'remote_recovery_watch_journal.jsonl', unreachable)
                print(
                    json.dumps(
                        {'observed_at_utc': observed_at, 'state': 'unreachable'},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if args.once:
                    raise
                time.sleep(args.poll_seconds)
                continue
            except Exception as error:
                details = error.payload if isinstance(error, RemoteRecoveryError) else {}
                alert = {
                    'observed_at_utc': observed_at,
                    'state': 'alert',
                    'error_type': type(error).__name__,
                    'error': str(error),
                    'remote_alert': details,
                    'scientific_protocol_changed': False,
                    'training_configuration_changed': False,
                    'used_for_checkpoint_or_configuration_selection': False,
                    **identities,
                }
                atomic_json(state_dir / 'remote_recovery_watch_alert.json', alert)
                atomic_json(state_dir / 'remote_recovery_watch_latest.json', alert)
                append_jsonl(state_dir / 'remote_recovery_watch_journal.jsonl', alert)
                print(
                    json.dumps(
                        {'observed_at_utc': observed_at, 'state': 'alert'},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if args.once:
                    raise SystemExit(2) from error
                time.sleep(args.poll_seconds)
                continue
            atomic_json(state_dir / 'remote_recovery_watch_latest.json', result)
            if result.get('launched_processes'):
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                atomic_json(state_dir / 'recovery_actions' / f'action-{stamp}.json', result)
            append_jsonl(
                state_dir / 'remote_recovery_watch_journal.jsonl',
                compact_record(result, identities),
            )
            print(
                json.dumps(
                    {
                        'observed_at_utc': result['observed_at_utc'],
                        'state': result['state'],
                        'formal_worker_count': len(result.get('formal_worker_pids', [])),
                        'launched_processes': sorted(result.get('launched_processes', {})),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if result['state'] == 'complete' or args.once:
                if result['state'] == 'complete':
                    atomic_json(state_dir / 'remote_recovery_watch_complete.json', result)
                return
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()

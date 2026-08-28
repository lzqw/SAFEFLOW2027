#!/usr/bin/env python3
"""Stage portable single-GPU evidence from the server when queues become terminal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

JOBS = {
    'pointcircle_geometry_enhancement': {
        'directory': 'pointcircle-geometry-enhancement-v1',
        'state_file': 'queue_state.json',
    },
    'pointcircle_formal': {
        'directory': 'pointcircle-enhanced-formal-v1',
        'state_file': 'queue_state.json',
    },
    'pointcircle_priors': {
        'directory': 'pointcircle-enhanced-priors-v1',
        'state_file': 'queue_state.json',
    },
    'pointcircle_postprocess': {
        'directory': 'pointcircle-enhanced-postprocess-v1',
        'state_file': 'postprocessing_state.json',
    },
    'pointcircle_controls': {
        'directory': 'pointcircle-enhanced-controls-v1',
        'state_file': 'queue_state.json',
    },
    'pointcircle_control_postprocess': {
        'directory': 'pointcircle-enhanced-control-postprocess-v1',
        'state_file': 'postprocessing_state.json',
    },
    'point_robustness': {
        'directory': 'point-robustness-enhanced-v1',
        'state_file': 'queue_state.json',
    },
    'pointgoal_robustness': {
        'directory': 'pointgoal-robustness-v1',
        'state_file': 'queue_state.json',
    },
    'ant_2m': {
        'directory': 'ant-2m-v1',
        'state_file': 'queue_state.json',
    },
}
TERMINAL_STATES = frozenset({'complete', 'screen_passed', 'screen_not_passed', 'failed'})
PORTABLE_PATTERNS = (
    '*.json',
    '*.csv',
    '*.md',
    '*.yaml',
    '*.yml',
    '*.txt',
    '*.sha256',
    '*_complete',
)


def utc_now() -> str:
    """Return one UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON through an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def remote_states(
    ssh_host: str,
    remote_python: Path,
    remote_root: Path,
) -> dict[str, dict[str, Any] | None]:
    """Read every queue state in one SSH round trip."""
    relative_files = {
        name: f"{spec['directory']}/{spec['state_file']}" for name, spec in JOBS.items()
    }
    remote_code = (
        'import json,sys; from pathlib import Path; '
        'root=Path(sys.argv[1]); files=json.loads(sys.argv[2]); out={}; '
        'exec("for name,rel in files.items():\\n p=root/rel\\n '
        'out[name]=json.loads(p.read_text()) if p.is_file() else None"); '
        'print(json.dumps(out,sort_keys=True))'
    )
    command = (
        f'{shlex.quote(str(remote_python))} -c {shlex.quote(remote_code)} '
        f'{shlex.quote(str(remote_root))} '
        f'{shlex.quote(json.dumps(relative_files, sort_keys=True))}'
    )
    completed = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', ssh_host, command],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError('remote queue status is not a JSON object')
    return payload


def prior_stage_matches(destination: Path, terminal_state: str) -> bool:
    """Return whether this exact terminal state was already staged."""
    receipt = destination / 'stage_receipt.json'
    if not receipt.is_file():
        return False
    payload = json.loads(receipt.read_text(encoding='utf-8'))
    return payload.get('terminal_state') == terminal_state


def rsync_portable(
    ssh_host: str,
    remote_directory: Path,
    destination: Path,
) -> None:
    """Copy only portable evidence, excluding checkpoints and logs."""
    destination.mkdir(parents=True, exist_ok=True)
    command = ['rsync', '-a', '--prune-empty-dirs', '--include=*/']
    for pattern in PORTABLE_PATTERNS:
        command.append(f'--include={pattern}')
    command.extend(
        [
            '--exclude=*',
            f'{ssh_host}:{remote_directory}/',
            f'{destination}/',
        ],
    )
    subprocess.run(command, check=True)


def staged_files(destination: Path) -> list[dict[str, Any]]:
    """Inventory staged evidence without hashing the stage receipt itself."""
    rows = []
    for path in sorted(destination.rglob('*')):
        if not path.is_file() or path.name == 'stage_receipt.json':
            continue
        rows.append(
            {
                'path': str(path.relative_to(destination)),
                'bytes': path.stat().st_size,
                'sha256': sha256_file(path),
            },
        )
    return rows


def stage_terminal_jobs(
    ssh_host: str,
    remote_root: Path,
    staging_root: Path,
    states: dict[str, dict[str, Any] | None],
) -> list[str]:
    """Stage newly terminal jobs and return their names."""
    staged = []
    for name, spec in JOBS.items():
        payload = states.get(name)
        if not isinstance(payload, dict):
            continue
        state = str(payload.get('state'))
        if state not in TERMINAL_STATES:
            continue
        destination = staging_root / name
        if prior_stage_matches(destination, state):
            continue
        remote_directory = remote_root / str(spec['directory'])
        rsync_portable(ssh_host, remote_directory, destination)
        files = staged_files(destination)
        atomic_json(
            destination / 'stage_receipt.json',
            {
                'status': 'complete',
                'job': name,
                'ssh_host': ssh_host,
                'remote_directory': str(remote_directory),
                'terminal_state': state,
                'remote_state_payload': payload,
                'staged_at_utc': utc_now(),
                'portable_files': files,
                'portable_file_count': len(files),
                'portable_bytes': sum(int(row['bytes']) for row in files),
                'checkpoints_copied': False,
                'logs_copied': False,
                'git_worktree_modified': False,
            },
        )
        staged.append(name)
    return staged


def parser() -> argparse.ArgumentParser:
    """Build watcher arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--ssh-host', required=True)
    result.add_argument('--remote-python', type=Path, required=True)
    result.add_argument('--remote-root', type=Path, required=True)
    result.add_argument('--staging-root', type=Path, required=True)
    result.add_argument('--poll-seconds', type=int, default=60)
    result.add_argument('--once', action='store_true')
    return result


def main() -> None:
    """Poll until all jobs are terminal, staging each terminal result exactly once."""
    args = parser().parse_args()
    args.remote_python = Path(str(args.remote_python))
    args.remote_root = Path(str(args.remote_root))
    args.staging_root = args.staging_root.resolve()
    if args.poll_seconds <= 0:
        raise ValueError('poll interval must be positive')
    args.staging_root.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            observed = remote_states(args.ssh_host, args.remote_python, args.remote_root)
        except subprocess.CalledProcessError:
            if args.once:
                raise
            time.sleep(args.poll_seconds)
            continue
        newly_staged = stage_terminal_jobs(
            args.ssh_host,
            args.remote_root,
            args.staging_root,
            observed,
        )
        state_names = {
            name: payload.get('state') if isinstance(payload, dict) else None
            for name, payload in observed.items()
        }
        all_terminal = all(state in TERMINAL_STATES for state in state_names.values())
        atomic_json(
            args.staging_root / 'watcher_state.json',
            {
                'status': 'complete' if all_terminal else 'monitoring',
                'observed_at_utc': utc_now(),
                'remote_states': state_names,
                'newly_staged': newly_staged,
                'all_terminal': all_terminal,
            },
        )
        if args.once or all_terminal:
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()

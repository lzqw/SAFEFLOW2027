#!/usr/bin/env python3
"""Collect completed single-GPU paper-core evidence from server to the host repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REMOTE_BASE = Path('/root/autodl-tmp/SAFE/single_gpu_paper_core')
REMOTE_ROOTS = {
    'pointgoal': REMOTE_BASE / 'v3',
    'pointcircle': REMOTE_BASE / 'pointcircle-v1',
    'pointcircle_priors': REMOTE_BASE / 'pointcircle-priors-v1',
    'pointcircle_postprocess': REMOTE_BASE / 'pointcircle-postprocess-v1',
    'pointcircle_controls': REMOTE_BASE / 'pointcircle-controls-v1',
    'pointcircle_controls_postprocess': REMOTE_BASE / 'pointcircle-controls-postprocess-v1',
    'point_robustness': REMOTE_BASE / 'point-robustness-v1',
}
TERMINAL_STATES = {
    'pointgoal': ('postprocess/postprocessing_state.json',),
    'pointcircle': ('queue_state.json',),
    'pointcircle_priors': ('queue_state.json',),
    'pointcircle_postprocess': ('postprocessing_state.json',),
    'pointcircle_controls': ('queue_state.json',),
    'pointcircle_controls_postprocess': ('postprocessing_state.json',),
    'point_robustness': ('queue_state.json',),
}


def sha256_file(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON through an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def remote_json(remote: str, path: Path) -> dict[str, Any]:
    """Read one remote JSON object without shell interpolation."""
    script = 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))'
    remote_command = ' '.join(
        (
            shlex.quote('/root/autodl-tmp/SAFE/envs/star-p0-p1/bin/python'),
            '-c',
            shlex.quote(script),
            shlex.quote(str(path)),
        )
    )
    command = ['ssh', remote, remote_command]
    payload = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    result = json.loads(payload)
    if not isinstance(result, dict):
        raise ValueError(f'{remote}:{path} does not contain a JSON object')
    return result


def terminal_states(remote: str) -> dict[str, dict[str, Any]]:
    """Read every required remote terminal state."""
    states: dict[str, dict[str, Any]] = {}
    for label, relative_paths in TERMINAL_STATES.items():
        if len(relative_paths) != 1:
            raise AssertionError('each collection root must have one terminal state')
        states[label] = remote_json(remote, REMOTE_ROOTS[label] / relative_paths[0])
    return states


def validate_states(
    states: dict[str, dict[str, Any]],
    allow_screen_failure: bool,
) -> str:
    """Require a coherent successful or registered-screen-failure terminal chain."""
    values = {label: str(payload.get('state')) for label, payload in states.items()}
    if all(value == 'complete' for value in values.values()):
        return 'complete'
    allowed = {'complete', 'screen_not_passed'}
    if allow_screen_failure and set(values.values()) <= allowed and 'screen_not_passed' in values.values():
        if values['pointgoal'] != 'complete':
            raise ValueError('PointCircle screen failure is valid only after PointGoal completion')
        return 'screen_not_passed'
    raise RuntimeError(f'remote paper-core evidence is not terminal: {values}')


def rsync_root(remote: str, remote_root: Path, destination: Path) -> None:
    """Copy only portable evidence, excluding checkpoints, training trees, work, and logs."""
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        'rsync',
        '-a',
        '--prune-empty-dirs',
        '--include=*/',
        '--include=*.json',
        '--include=*.csv',
        '--include=*.md',
        '--exclude=**/training/***',
        '--exclude=**/work/***',
        '--exclude=*.pt',
        '--exclude=*.log',
        '--exclude=*',
        f'{remote}:{remote_root}/',
        f'{destination}/',
    ]
    subprocess.run(command, check=True)


def parser() -> argparse.ArgumentParser:
    """Build collection arguments."""
    default_output = (
        Path(__file__).resolve().parents[2]
        / 'reports/star/single_gpu_paper_core_artifacts'
    )
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--remote', default='seeta-nmb2')
    result.add_argument('--output-root', type=Path, default=default_output)
    result.add_argument('--allow-screen-failure', action='store_true')
    return result


def main() -> None:
    """Authenticate terminal states, copy evidence, and seal a host-side manifest."""
    args = parser().parse_args()
    args.output_root = args.output_root.resolve()
    states = terminal_states(args.remote)
    outcome = validate_states(states, args.allow_screen_failure)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for label, remote_root in REMOTE_ROOTS.items():
        rsync_root(args.remote, remote_root, args.output_root / label)
    files = sorted(
        path
        for path in args.output_root.rglob('*')
        if path.is_file() and path.name != 'COLLECTION_MANIFEST.json'
    )
    forbidden = [
        path
        for path in files
        if path.suffix == '.pt' or 'training' in path.relative_to(args.output_root).parts
    ]
    if forbidden:
        raise ValueError(f'collector copied forbidden training/checkpoint files: {forbidden}')
    manifest = {
        'status': outcome,
        'collected_at_utc': datetime.now(timezone.utc).isoformat(),
        'remote': args.remote,
        'remote_roots': {label: str(path) for label, path in REMOTE_ROOTS.items()},
        'terminal_states': states,
        'portable_evidence_only': True,
        'checkpoints_copied': False,
        'files': [
            {
                'path': str(path.relative_to(args.output_root)),
                'bytes': path.stat().st_size,
                'sha256': sha256_file(path),
            }
            for path in files
        ],
    }
    atomic_json(args.output_root / 'COLLECTION_MANIFEST.json', manifest)
    print(
        json.dumps(
            {
                'status': outcome,
                'files': len(files),
                'output_root': str(args.output_root),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    main()

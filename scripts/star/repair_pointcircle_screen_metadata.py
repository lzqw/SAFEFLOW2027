#!/usr/bin/env python3
"""Audit and repair missing PointCircle screen environment metadata only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ENVIRONMENT = 'SafetyPointCircle1-v0'
ARMS = ('C0', 'P47')
MILESTONES = (150_000, 300_000)
BASE_SEEDS = (22_027, 22_028, 22_029, 22_030, 22_031)
EPISODES = 20
EPISODES_PER_BASE_SEED = 4


def utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    return sha256_bytes(path.read_bytes())


def scientific_sha256(payload: dict[str, Any]) -> str:
    """Fingerprint every field except the repaired environment label."""
    scientific = dict(payload)
    scientific.pop('environment', None)
    encoded = json.dumps(
        scientific,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return sha256_bytes(encoded)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON through an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} does not contain a JSON object')
    return payload


def git_commit(source: Path) -> str:
    """Require a clean repair source and return its exact commit."""
    status = subprocess.run(
        ['git', '-C', str(source), 'status', '--porcelain'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError('metadata repair source must be clean')
    return subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def artifact_path(screen_root: Path, arm: str, milestone: int) -> Path | None:
    """Resolve at most one artifact at an exact preregistered arm and milestone."""
    matches = sorted(
        (screen_root / f'{arm}-2027').glob(
            'training/STAR-{SafetyPointCircle1-v0}/seed-2027-*/artifacts/'
            f'evaluation_step_{milestone}.json',
        ),
    )
    if len(matches) > 1:
        raise ValueError(f'expected at most one {arm}@{milestone} artifact, found {len(matches)}')
    return matches[0].resolve() if matches else None


def validate_payload(payload: dict[str, Any], arm: str, milestone: int) -> None:
    """Reject any artifact that is not the exact frozen development evaluation."""
    label = f'{arm}@{milestone}'
    environment = payload.get('environment')
    if environment not in (None, ENVIRONMENT):
        raise ValueError(f'{label} carries unexpected environment {environment!r}')
    if int(payload.get('milestone', -1)) != milestone:
        raise ValueError(f'{label} milestone mismatch')
    if int(payload.get('episodes', -1)) != EPISODES:
        raise ValueError(f'{label} does not contain {EPISODES} episodes')
    if [int(seed) for seed in payload.get('base_seeds', [])] != list(BASE_SEEDS):
        raise ValueError(f'{label} uses unexpected base seeds')
    if int(payload.get('episodes_per_base_seed', -1)) != EPISODES_PER_BASE_SEED:
        raise ValueError(f'{label} has an unexpected episodes-per-seed value')


def initial_state(
    screen_root: Path,
    training_source_commit: str,
    repair_source_commit: str,
) -> dict[str, Any]:
    """Build the immutable identity portion of the audit state."""
    return {
        'status': 'watching',
        'environment': ENVIRONMENT,
        'screen_root': str(screen_root),
        'training_source_commit': training_source_commit,
        'repair_source_commit': repair_source_commit,
        'scientific_metrics_changed': False,
        'required_arms': list(ARMS),
        'required_milestones': list(MILESTONES),
        'artifacts': {},
    }


def load_state(
    state_path: Path,
    screen_root: Path,
    training_source_commit: str,
    repair_source_commit: str,
) -> dict[str, Any]:
    """Load a restartable state while enforcing its source identities."""
    if not state_path.is_file():
        return initial_state(screen_root, training_source_commit, repair_source_commit)
    state = read_json(state_path)
    expected = {
        'environment': ENVIRONMENT,
        'screen_root': str(screen_root),
        'training_source_commit': training_source_commit,
        'repair_source_commit': repair_source_commit,
        'scientific_metrics_changed': False,
        'required_arms': list(ARMS),
        'required_milestones': list(MILESTONES),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f'repair state identity mismatch for {key}')
    if not isinstance(state.get('artifacts'), dict):
        raise ValueError('repair state has no artifact mapping')
    return state


def repair_one(
    path: Path,
    arm: str,
    milestone: int,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    """Add only the missing environment field with crash-safe audit evidence."""
    payload = read_json(path)
    validate_payload(payload, arm, milestone)
    current_sha256 = sha256_file(path)
    content_sha256 = scientific_sha256(payload)
    key = f'{arm}:{milestone}'
    existing = state['artifacts'].get(key)
    if existing is not None:
        if existing.get('path') != str(path):
            raise ValueError(f'{key} repair path changed')
        if existing.get('scientific_payload_sha256') != content_sha256:
            raise ValueError(f'{key} scientific payload changed')
        if existing.get('status') == 'complete':
            if payload.get('environment') != ENVIRONMENT:
                raise ValueError(f'{key} lost its repaired environment label')
            if existing.get('repaired_sha256') != current_sha256:
                raise ValueError(f'{key} bytes changed after repair')
            return
        if existing.get('status') != 'repairing':
            raise ValueError(f'{key} has an unexpected repair status')
        if payload.get('environment') is None:
            if existing.get('original_sha256') != current_sha256:
                raise ValueError(f'{key} changed before its pending repair')
        elif payload.get('environment') != ENVIRONMENT:
            raise ValueError(f'{key} has unexpected environment metadata')

    if payload.get('environment') == ENVIRONMENT:
        original_sha256 = (
            existing.get('original_sha256') if existing is not None else current_sha256
        )
        action = (
            existing.get('action') if existing is not None else 'environment_already_present'
        )
        state['artifacts'][key] = {
            'status': 'complete',
            'action': action,
            'arm': arm,
            'milestone': milestone,
            'path': str(path),
            'original_sha256': original_sha256,
            'repaired_sha256': current_sha256,
            'scientific_payload_sha256': content_sha256,
            'scientific_metrics_changed': False,
            'completed_at_utc': utc_now(),
        }
        atomic_json(state_path, state)
        return

    if existing is None:
        state['artifacts'][key] = {
            'status': 'repairing',
            'action': 'added_missing_environment',
            'arm': arm,
            'milestone': milestone,
            'path': str(path),
            'original_sha256': current_sha256,
            'scientific_payload_sha256': content_sha256,
            'scientific_metrics_changed': False,
            'started_at_utc': utc_now(),
        }
        atomic_json(state_path, state)

    repaired = dict(payload)
    repaired['environment'] = ENVIRONMENT
    if scientific_sha256(repaired) != content_sha256:
        raise AssertionError(f'{key} repair altered scientific content')
    atomic_json(path, repaired)
    repaired_payload = read_json(path)
    validate_payload(repaired_payload, arm, milestone)
    repaired_sha256 = sha256_file(path)
    if repaired_payload.get('environment') != ENVIRONMENT:
        raise AssertionError(f'{key} environment repair did not persist')
    if scientific_sha256(repaired_payload) != content_sha256:
        raise AssertionError(f'{key} scientific content changed after atomic write')
    entry = state['artifacts'][key]
    entry.update(
        {
            'status': 'complete',
            'repaired_sha256': repaired_sha256,
            'completed_at_utc': utc_now(),
        },
    )
    atomic_json(state_path, state)


def repair_available(
    screen_root: Path,
    state_path: Path,
    training_source_commit: str,
    repair_source_commit: str,
) -> dict[str, Any]:
    """Repair all currently available exact artifacts and return the audit state."""
    screen_root = screen_root.resolve()
    state_path = state_path.resolve()
    state = load_state(
        state_path,
        screen_root,
        training_source_commit,
        repair_source_commit,
    )
    for arm in ARMS:
        for milestone in MILESTONES:
            path = artifact_path(screen_root, arm, milestone)
            if path is not None:
                repair_one(path, arm, milestone, state, state_path)
    required = {f'{arm}:{milestone}' for arm in ARMS for milestone in MILESTONES}
    completed = {
        key
        for key, entry in state['artifacts'].items()
        if isinstance(entry, dict) and entry.get('status') == 'complete'
    }
    state['status'] = 'complete' if completed == required else 'watching'
    state['last_checked_at_utc'] = utc_now()
    if state['status'] == 'complete':
        state['completed_at_utc'] = utc_now()
    atomic_json(state_path, state)
    return state


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--screen-root', type=Path, required=True)
    result.add_argument('--state', type=Path, required=True)
    result.add_argument('--repair-source', type=Path, required=True)
    result.add_argument('--training-source-commit', required=True)
    result.add_argument('--poll-seconds', type=float, default=0.25)
    result.add_argument('--once', action='store_true')
    return result


def main() -> None:
    """Patch existing artifacts and optionally watch for the 300k pair."""
    args = parser().parse_args()
    args.screen_root = args.screen_root.resolve()
    args.state = args.state.resolve()
    args.repair_source = args.repair_source.resolve()
    if args.poll_seconds <= 0.0:
        raise ValueError('poll interval must be positive')
    repair_source_commit = git_commit(args.repair_source)
    while True:
        state = repair_available(
            args.screen_root,
            args.state,
            args.training_source_commit,
            repair_source_commit,
        )
        print(json.dumps(state, sort_keys=True, allow_nan=False), flush=True)
        if args.once or state['status'] == 'complete':
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Report resumable progress for the formal SHAPO P4.7 matrix."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import yaml


SLUGS = {
    'SafetyPointGoal1-v0': 'point-goal',
    'SafetyPointButton1-v0': 'point-button',
    'Ant-v4': 'ant',
    'Walker2d-v4': 'walker2d',
}


def latest_progress(container: Path) -> dict[str, Any]:
    """Return the greatest total-step row across all recovery segments."""
    best: dict[str, Any] = {}
    best_identity = (-1.0, -1.0)
    for path in container.glob('training/**/progress.csv'):
        modified = path.stat().st_mtime
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows[-1:]:
            try:
                steps = float(row['TotalEnvSteps'])
            except (KeyError, TypeError, ValueError):
                continue
            identity = (steps, modified)
            if identity > best_identity:
                best_identity = identity
                best = {
                    'env_steps': steps,
                    'updates': float(row.get('TotalUpdates', 0.0)),
                    'fps': float(row.get('Time/FPS', 0.0)),
                    'cumulative_cost': float(row.get('Train/CumulativeCost', 0.0)),
                    'wall_seconds': float(row.get('Time/Total', 0.0)),
                    'progress_mtime_unix': modified,
                    'progress_path': str(path),
                }
    return best


def newest_full_checkpoint(container: Path) -> dict[str, Any]:
    """Return lightweight identity fields for the newest resumable full checkpoint."""
    matches = list(container.glob('training/**/checkpoints/latest_full.pt'))
    if not matches:
        return {'has_full_checkpoint': False}
    path = max(matches, key=lambda candidate: candidate.stat().st_mtime_ns)
    stat = path.stat()
    return {
        'has_full_checkpoint': True,
        'latest_full_checkpoint': str(path),
        'latest_full_checkpoint_bytes': stat.st_size,
        'latest_full_checkpoint_mtime_unix': stat.st_mtime,
    }


def active_worker_containers(proc_root: Path | None = None) -> set[str]:
    """Discover formal run containers from live ``train_star.py --config`` commands."""
    proc_root = Path('/proc') if proc_root is None else proc_root
    containers = set()
    for path in proc_root.glob('[0-9]*/cmdline'):
        try:
            arguments = [
                value.decode('utf-8', errors='replace')
                for value in path.read_bytes().split(b'\0')
                if value
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not any(value.endswith('scripts/star/train_star.py') for value in arguments):
            continue
        try:
            config = Path(arguments[arguments.index('--config') + 1])
        except (ValueError, IndexError):
            continue
        containers.add(str(config.parent))
    return containers


def run_state(
    container: Path,
    total_steps: int,
    *,
    now: float | None = None,
    stale_after_seconds: float = 1800.0,
    worker_alive: bool | None = None,
) -> dict[str, Any]:
    """Classify one seed run from its immutable receipts and append-only segments."""
    snapshot_time = time.time() if now is None else now
    checkpoint = newest_full_checkpoint(container)
    validation = container / 'formal_validation.json'
    if validation.is_file():
        payload = json.loads(validation.read_text(encoding='utf-8'))
        if payload.get('status') == 'passed':
            return {
                'state': 'completed',
                'env_steps': total_steps,
                'fraction': 1.0,
                'worker_alive': bool(worker_alive),
                **checkpoint,
            }
    progress = latest_progress(container)
    exit_path = container / 'exit_code.txt'
    configs = list(container.glob('training/**/config.json'))
    state = 'pending'
    activity_mtimes = [float(progress['progress_mtime_unix'])] if progress else []
    activity_mtimes.extend(path.stat().st_mtime for path in configs)
    activity_age = None
    if activity_mtimes:
        activity_age = max(0.0, snapshot_time - max(activity_mtimes))
    if progress or configs or worker_alive:
        state = 'running_or_recoverable'
        if (
            not worker_alive
            and stale_after_seconds > 0.0
            and activity_age is not None
            and activity_age > stale_after_seconds
        ):
            state = 'stale_recoverable'
    if exit_path.is_file():
        exit_code = int(exit_path.read_text(encoding='utf-8').strip())
        state = 'completed_unvalidated' if exit_code == 0 else 'engineering_failure'
    steps = int(progress.get('env_steps', 0.0))
    return {
        'state': state,
        'env_steps': steps,
        'fraction': steps / total_steps,
        'worker_alive': bool(worker_alive),
        'activity_age_seconds': activity_age,
        **checkpoint,
        **{key: value for key, value in progress.items() if key != 'env_steps'},
    }


def main() -> None:
    """Print a machine-readable status snapshot and a compact human table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument(
        '--stale-after-seconds',
        type=float,
        default=1800.0,
        help='mark active-looking runs stale after this inactivity window; <=0 disables',
    )
    parser.add_argument('--json-only', action='store_true')
    args = parser.parse_args()
    matrix = yaml.safe_load(args.matrix.read_text(encoding='utf-8'))
    active_containers = active_worker_containers()
    rows = []
    for seed in matrix['training_seeds']:
        for env_id, environment in matrix['environments'].items():
            total_steps = int(environment['total_steps'])
            container = args.run_root / 'runs' / f'{SLUGS[env_id]}-{seed}'
            rows.append(
                {
                    'environment': env_id,
                    'training_seed': int(seed),
                    'total_steps': total_steps,
                    **run_state(
                        container,
                        total_steps,
                        stale_after_seconds=args.stale_after_seconds,
                        worker_alive=str(container) in active_containers,
                    ),
                },
            )
    payload = {
        'formal_matrix_complete': (args.run_root / 'formal_matrix_complete').is_file(),
        'completed_runs': sum(row['state'] == 'completed' for row in rows),
        'failed_runs': sum(row['state'] == 'engineering_failure' for row in rows),
        'stale_runs': sum(row['state'] == 'stale_recoverable' for row in rows),
        'full_checkpoint_runs': sum(bool(row['has_full_checkpoint']) for row in rows),
        'total_runs': len(rows),
        'completed_environment_steps': sum(int(row['env_steps']) for row in rows),
        'planned_environment_steps': sum(int(row['total_steps']) for row in rows),
        'runs': rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not args.json_only:
        print('\nenvironment                   seed  state                         steps       %')
        for row in rows:
            print(
                f"{row['environment']:<29} {row['training_seed']:>4}  "
                f"{row['state']:<27} {row['env_steps']:>10} "
                f"{100.0 * row['fraction']:>6.2f}",
            )


if __name__ == '__main__':
    main()

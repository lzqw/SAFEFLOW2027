#!/usr/bin/env python3
"""Report resumable progress for the formal SHAPO P4.7 matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


SLUGS = {
    'SafetyPointGoal1-v0': 'point-goal',
    'SafetyPointButton1-v0': 'point-button',
    'Ant-v4': 'ant',
    'Walker2d-v4': 'walker2d',
}


def latest_progress(container: Path) -> dict[str, float]:
    """Return the greatest total-step row across all recovery segments."""
    best: dict[str, float] = {}
    for path in container.glob('training/**/progress.csv'):
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows[-1:]:
            try:
                steps = float(row['TotalEnvSteps'])
            except (KeyError, TypeError, ValueError):
                continue
            if steps > best.get('env_steps', -1.0):
                best = {
                    'env_steps': steps,
                    'updates': float(row.get('TotalUpdates', 0.0)),
                    'fps': float(row.get('Time/FPS', 0.0)),
                    'cumulative_cost': float(row.get('Train/CumulativeCost', 0.0)),
                    'wall_seconds': float(row.get('Time/Total', 0.0)),
                }
    return best


def run_state(container: Path, total_steps: int) -> dict[str, Any]:
    """Classify one seed run from its immutable receipts and append-only segments."""
    validation = container / 'formal_validation.json'
    if validation.is_file():
        payload = json.loads(validation.read_text(encoding='utf-8'))
        if payload.get('status') == 'passed':
            return {'state': 'completed', 'env_steps': total_steps, 'fraction': 1.0}
    progress = latest_progress(container)
    exit_path = container / 'exit_code.txt'
    state = 'pending'
    if progress or any(container.glob('training/**/config.json')):
        state = 'running_or_recoverable'
    if exit_path.is_file():
        exit_code = int(exit_path.read_text(encoding='utf-8').strip())
        state = 'completed_unvalidated' if exit_code == 0 else 'engineering_failure'
    steps = int(progress.get('env_steps', 0.0))
    return {
        'state': state,
        'env_steps': steps,
        'fraction': steps / total_steps,
        **{key: value for key, value in progress.items() if key != 'env_steps'},
    }


def main() -> None:
    """Print a machine-readable status snapshot and a compact human table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--json-only', action='store_true')
    args = parser.parse_args()
    matrix = yaml.safe_load(args.matrix.read_text(encoding='utf-8'))
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
                    **run_state(container, total_steps),
                },
            )
    payload = {
        'formal_matrix_complete': (args.run_root / 'formal_matrix_complete').is_file(),
        'completed_runs': sum(row['state'] == 'completed' for row in rows),
        'failed_runs': sum(row['state'] == 'engineering_failure' for row in rows),
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

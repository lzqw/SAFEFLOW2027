#!/usr/bin/env python3
"""Persist compact health evidence for the long frozen SHAPO P4.7 matrix."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from scripts.star.status_shapo_p47_formal import formal_matrix_status


def process_alive(pid_file: Path | None) -> bool | None:
    """Return whether a recorded supervisor PID exists, or ``None`` when unconfigured."""
    if pid_file is None:
        return None
    try:
        pid = int(pid_file.read_text(encoding='utf-8').strip())
        os.kill(pid, 0)
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return False
    return True


def build_health_snapshot(
    status: dict[str, Any],
    *,
    run_root: Path,
    supervisor_alive: bool | None,
    min_free_bytes: int,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Classify one status payload without changing any training state."""
    free_bytes = shutil.disk_usage(run_root).free
    matrix_complete = bool(status['formal_matrix_complete'])
    alerts = []
    if int(status['failed_runs']):
        alerts.append('engineering_failure')
    if int(status['stale_runs']):
        alerts.append('stale_recoverable')
    if free_bytes < min_free_bytes:
        alerts.append('low_disk_space')
    if not matrix_complete and supervisor_alive is False:
        alerts.append('supervisor_not_alive')
    state = 'complete' if matrix_complete and not alerts else ('alert' if alerts else 'healthy')
    timestamp = observed_at or datetime.now(timezone.utc).isoformat()
    active_runs = [row for row in status['runs'] if row['worker_alive']]
    return {
        'observed_at_utc': timestamp,
        'state': state,
        'alerts': alerts,
        'supervisor_alive': supervisor_alive,
        'filesystem_free_bytes': free_bytes,
        'minimum_free_bytes': min_free_bytes,
        'formal_matrix_complete': matrix_complete,
        'completed_runs': int(status['completed_runs']),
        'failed_runs': int(status['failed_runs']),
        'stale_runs': int(status['stale_runs']),
        'full_checkpoint_runs': int(status['full_checkpoint_runs']),
        'total_runs': int(status['total_runs']),
        'completed_environment_steps': int(status['completed_environment_steps']),
        'planned_environment_steps': int(status['planned_environment_steps']),
        'active_workers': len(active_runs),
        'active_runs': [
            {
                'environment': row['environment'],
                'training_seed': int(row['training_seed']),
                'env_steps': int(row['env_steps']),
                'total_steps': int(row['total_steps']),
                'activity_age_seconds': row.get('activity_age_seconds'),
                'has_full_checkpoint': bool(row['has_full_checkpoint']),
            }
            for row in active_runs
        ],
        'run_root': str(run_root),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace one JSON receipt atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def append_journal(path: Path, snapshot: dict[str, Any]) -> None:
    """Append a compact status record and flush it to disk for runtime auditing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    compact = {
        key: snapshot[key]
        for key in (
            'observed_at_utc',
            'state',
            'alerts',
            'supervisor_alive',
            'filesystem_free_bytes',
            'completed_runs',
            'failed_runs',
            'stale_runs',
            'full_checkpoint_runs',
            'completed_environment_steps',
            'planned_environment_steps',
            'active_workers',
            'active_runs',
        )
    }
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(compact, sort_keys=True, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    """Watch until completion or a durable engineering alert is produced."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--supervisor-pid-file', type=Path)
    parser.add_argument('--interval-seconds', type=float, default=600.0)
    parser.add_argument('--stale-after-seconds', type=float, default=1800.0)
    parser.add_argument('--min-free-gib', type=float, default=4.0)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval_seconds <= 0.0:
        parser.error('--interval-seconds must be positive')
    if args.min_free_gib < 0.0:
        parser.error('--min-free-gib must be non-negative')
    matrix = yaml.safe_load(args.matrix.read_text(encoding='utf-8'))
    min_free_bytes = int(args.min_free_gib * (1024**3))
    while True:
        status = formal_matrix_status(
            args.run_root,
            matrix,
            stale_after_seconds=args.stale_after_seconds,
        )
        snapshot = build_health_snapshot(
            status,
            run_root=args.run_root,
            supervisor_alive=process_alive(args.supervisor_pid_file),
            min_free_bytes=min_free_bytes,
        )
        atomic_json(args.output_dir / 'latest_health.json', snapshot)
        append_journal(args.output_dir / 'health_journal.jsonl', snapshot)
        if snapshot['state'] == 'alert':
            atomic_json(args.output_dir / 'alert.json', snapshot)
            raise SystemExit(2)
        if snapshot['state'] == 'complete':
            atomic_json(args.output_dir / 'completion_health.json', snapshot)
            return
        if args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == '__main__':
    main()

# Copyright 2026 SAFEFLOW2027 Authors. All Rights Reserved.
"""Resumable and stale-state tests for the long SHAPO P4.7 formal matrix."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from scripts.star.status_shapo_p47_formal import (
    active_worker_containers,
    formal_matrix_status,
    latest_progress,
    run_state,
)
from scripts.star.watch_shapo_p47_formal import (
    append_journal,
    atomic_json,
    build_health_snapshot,
    process_alive,
)


def _progress(path: Path, steps: int, modified: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        'TotalEnvSteps': steps,
        'TotalUpdates': steps - 2000,
        'Time/FPS': 7.0,
        'Train/CumulativeCost': 12.0,
        'Time/Total': 100.0,
    }
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    os.utime(path, (modified, modified))


def test_latest_progress_prefers_step_then_newest_recovery_segment(tmp_path: Path) -> None:
    first = tmp_path / 'training/segment-a/progress.csv'
    second = tmp_path / 'training/segment-b/progress.csv'
    _progress(first, 10_000, 100.0)
    _progress(second, 10_000, 200.0)
    assert latest_progress(tmp_path)['progress_path'] == str(second)
    _progress(first, 15_000, 100.0)
    assert latest_progress(tmp_path)['progress_path'] == str(first)


def test_run_state_marks_stale_progress_and_exposes_full_checkpoint(tmp_path: Path) -> None:
    progress = tmp_path / 'training/segment/progress.csv'
    checkpoint = tmp_path / 'training/segment/checkpoints/latest_full.pt'
    _progress(progress, 50_000, 100.0)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'full-checkpoint')
    os.utime(checkpoint, (150.0, 150.0))
    state = run_state(
        tmp_path,
        10_000_000,
        now=2000.0,
        stale_after_seconds=1800.0,
        worker_alive=False,
    )
    assert state['state'] == 'stale_recoverable'
    assert state['env_steps'] == 50_000
    assert state['activity_age_seconds'] == 1900.0
    assert state['has_full_checkpoint'] is True
    assert state['latest_full_checkpoint_bytes'] == len(b'full-checkpoint')


def test_run_state_keeps_fresh_or_explicit_exit_states_authoritative(tmp_path: Path) -> None:
    progress = tmp_path / 'training/segment/progress.csv'
    _progress(progress, 5000, 900.0)
    fresh = run_state(tmp_path, 100_000, now=1000.0, stale_after_seconds=1800.0)
    assert fresh['state'] == 'running_or_recoverable'
    (tmp_path / 'exit_code.txt').write_text('137\n', encoding='utf-8')
    failed = run_state(tmp_path, 100_000, now=5000.0, stale_after_seconds=1800.0)
    assert failed['state'] == 'engineering_failure'


def test_live_worker_prevents_false_stale_during_long_evaluation(tmp_path: Path) -> None:
    progress = tmp_path / 'training/segment/progress.csv'
    _progress(progress, 500_000, 100.0)
    state = run_state(
        tmp_path,
        5_000_000,
        now=10_000.0,
        stale_after_seconds=1800.0,
        worker_alive=True,
    )
    assert state['state'] == 'running_or_recoverable'
    assert state['worker_alive'] is True


def test_active_worker_containers_parses_train_config_from_proc(tmp_path: Path) -> None:
    proc = tmp_path / 'proc'
    worker = proc / '123/cmdline'
    unrelated = proc / '456/cmdline'
    worker.parent.mkdir(parents=True)
    unrelated.parent.mkdir(parents=True)
    config = tmp_path / 'formal/runs/ant-2027/resolved_formal_config.yaml'
    worker.write_bytes(
        b'python\0scripts/star/train_star.py\0--config\0'
        + str(config).encode('utf-8')
        + b'\0--seed\0'
        + b'2027\0',
    )
    unrelated.write_bytes(b'python\0other.py\0')
    assert active_worker_containers(proc) == {str(config.parent)}


def test_completed_validation_overrides_old_activity(tmp_path: Path) -> None:
    progress = tmp_path / 'training/segment/progress.csv'
    _progress(progress, 100_000, 10.0)
    (tmp_path / 'formal_validation.json').write_text(
        json.dumps({'status': 'passed'}),
        encoding='utf-8',
    )
    state = run_state(tmp_path, 100_000, now=10_000.0)
    assert state['state'] == 'completed'
    assert state['fraction'] == 1.0


def _matrix() -> dict[str, object]:
    return {
        'training_seeds': [2027],
        'environments': {
            'Ant-v4': {'total_steps': 100_000},
            'Walker2d-v4': {'total_steps': 100_000},
        },
    }


def test_formal_matrix_status_reuses_run_classification(tmp_path: Path) -> None:
    run_root = tmp_path / 'formal'
    _progress(run_root / 'runs/ant-2027/training/a/progress.csv', 5000, 900.0)
    payload = formal_matrix_status(
        run_root,
        _matrix(),
        now=1000.0,
        proc_root=tmp_path / 'empty-proc',
    )
    assert payload['total_runs'] == 2
    assert payload['completed_environment_steps'] == 5000
    assert payload['failed_runs'] == 0
    assert payload['runs'][0]['state'] == 'running_or_recoverable'
    assert payload['runs'][1]['state'] == 'pending'


def test_health_snapshot_alerts_without_mutating_run(tmp_path: Path) -> None:
    status = {
        'formal_matrix_complete': False,
        'completed_runs': 0,
        'failed_runs': 1,
        'stale_runs': 0,
        'full_checkpoint_runs': 1,
        'total_runs': 2,
        'completed_environment_steps': 5000,
        'planned_environment_steps': 200_000,
        'runs': [
            {
                'environment': 'Ant-v4',
                'training_seed': 2027,
                'env_steps': 5000,
                'total_steps': 100_000,
                'worker_alive': True,
                'activity_age_seconds': 10.0,
                'has_full_checkpoint': True,
            },
        ],
    }
    snapshot = build_health_snapshot(
        status,
        run_root=tmp_path,
        supervisor_alive=False,
        min_free_bytes=0,
        observed_at='2026-08-10T00:00:00+00:00',
    )
    assert snapshot['state'] == 'alert'
    assert snapshot['alerts'] == ['engineering_failure', 'supervisor_not_alive']
    assert snapshot['active_workers'] == 1
    assert not list(tmp_path.iterdir())


def test_health_snapshot_complete_ignores_stopped_supervisor(tmp_path: Path) -> None:
    status = {
        'formal_matrix_complete': True,
        'completed_runs': 2,
        'failed_runs': 0,
        'stale_runs': 0,
        'full_checkpoint_runs': 2,
        'total_runs': 2,
        'completed_environment_steps': 200_000,
        'planned_environment_steps': 200_000,
        'runs': [],
    }
    snapshot = build_health_snapshot(
        status,
        run_root=tmp_path,
        supervisor_alive=False,
        min_free_bytes=0,
    )
    assert snapshot['state'] == 'complete'
    assert snapshot['alerts'] == []


def test_watchdog_receipts_are_atomic_append_only_and_pid_aware(tmp_path: Path) -> None:
    latest = tmp_path / 'monitor/latest.json'
    journal = tmp_path / 'monitor/journal.jsonl'
    payload = {'state': 'healthy', 'value': 1}
    atomic_json(latest, payload)
    assert json.loads(latest.read_text(encoding='utf-8')) == payload
    assert not latest.with_suffix('.json.tmp').exists()

    snapshot = {
        'observed_at_utc': '2026-08-10T00:00:00+00:00',
        'state': 'healthy',
        'alerts': [],
        'supervisor_alive': True,
        'filesystem_free_bytes': 10,
        'completed_runs': 0,
        'failed_runs': 0,
        'stale_runs': 0,
        'full_checkpoint_runs': 1,
        'completed_environment_steps': 5000,
        'planned_environment_steps': 100_000,
        'active_workers': 1,
        'active_runs': [],
    }
    append_journal(journal, snapshot)
    append_journal(journal, snapshot)
    assert len(journal.read_text(encoding='utf-8').splitlines()) == 2

    pid_file = tmp_path / 'pid'
    pid_file.write_text(f'{os.getpid()}\n', encoding='utf-8')
    assert process_alive(pid_file) is True
    pid_file.write_text('not-a-pid\n', encoding='utf-8')
    assert process_alive(pid_file) is False

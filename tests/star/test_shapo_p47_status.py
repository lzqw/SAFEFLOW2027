# Copyright 2026 SAFEFLOW2027 Authors. All Rights Reserved.
"""Resumable and stale-state tests for the long SHAPO P4.7 formal matrix."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from scripts.star.status_shapo_p47_formal import (
    active_worker_containers,
    latest_progress,
    run_state,
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

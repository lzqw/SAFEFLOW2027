#!/usr/bin/env python3
"""Run the four 20k STAR compatibility smokes with compact receipts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ENVIRONMENTS = (
    'SafetyPointGoal1-v0',
    'SafetyPointButton1-v0',
    'Ant-v4',
    'Walker2d-v4',
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a receipt without exposing partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(f'{json.dumps(payload, indent=2, sort_keys=True)}\n', encoding='utf-8')
    os.replace(temporary, path)


def git_commit(repo_root: Path) -> str:
    """Return the source commit used by the server worktree."""
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else 'unknown'


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one tabular artifact."""
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def assert_finite(value: Any, location: str = 'receipt') -> None:
    """Reject non-finite numbers recursively while allowing strings and nulls."""
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f'{location}.{key}')
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite(child, f'{location}[{index}]')
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f'non-finite value at {location}: {value}')


def matching_receipt(path: Path, signature: dict[str, Any]) -> dict[str, Any] | None:
    """Return a prior successful receipt only when its immutable signature matches."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('status') != 'passed':
        return None
    return payload if all(payload.get(key) == value for key, value in signature.items()) else None


def run_environment(  # pylint: disable=too-many-locals
    args: argparse.Namespace,
    repo_root: Path,
    env_id: str,
) -> dict[str, Any]:
    """Run and validate one real STAR training closure."""
    environment_dir = args.output_dir / env_id
    result_path = environment_dir / 'result.json'
    signature = {
        'algorithm': 'STAR',
        'environment': env_id,
        'seed': args.seed,
        'steps': args.steps,
    }
    if args.resume:
        existing = matching_receipt(result_path, signature)
        if existing is not None:
            return existing

    command = [
        str(args.python),
        str(repo_root / 'scripts/star/train_star.py'),
        '--config',
        str(repo_root / f'configs/star/{env_id}.yaml'),
        '--steps',
        str(args.steps),
        '--steps-per-epoch',
        str(args.steps_per_epoch),
        '--seed',
        str(args.seed),
        '--device',
        args.device,
        '--batch-size',
        str(args.batch_size),
        '--random-steps',
        str(args.random_steps),
        '--replay-size',
        str(max(args.batch_size + 1, args.steps * 2)),
        '--eval-episodes',
        str(args.eval_episodes),
        '--eval-seeds',
        str(args.eval_seed),
        '--eval-milestones',
        f'0,{args.steps}',
        '--log-dir',
        str(environment_dir / 'runs'),
        '--result',
        str(result_path),
        '--hourly-rate-cny',
        str(args.hourly_rate_cny),
        '--no-tensorboard',
    ]
    environment_dir.mkdir(parents=True, exist_ok=True)
    log_path = environment_dir / 'launcher.log'
    started = time.monotonic()
    with log_path.open('w', encoding='utf-8') as stream:
        result = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        tail = '\n'.join(log_path.read_text(encoding='utf-8').splitlines()[-40:])
        raise RuntimeError(f'{env_id} exited {result.returncode}:\n{tail}')

    receipt = json.loads(result_path.read_text(encoding='utf-8'))
    assert_finite(receipt)
    if any(receipt.get(key) != value for key, value in signature.items()):
        raise RuntimeError(f'{env_id} receipt signature does not match the requested run')
    expected_updates = max(0, args.steps - args.random_steps)
    if receipt['updates'] != expected_updates:
        raise RuntimeError(
            f'{env_id} expected {expected_updates} updates, observed {receipt["updates"]}',
        )
    summary = receipt['training_summary']
    if summary['status'] != 'completed' or summary['run_env_steps'] != args.steps:
        raise RuntimeError(f'{env_id} did not complete the requested training segment')

    log_dir = Path(receipt['log_dir'])
    diagnostics = read_csv(log_dir / 'artifacts/diagnostics.csv')
    if not diagnostics:
        raise RuntimeError(f'{env_id} emitted no diagnostics rows')
    for row_index, row in enumerate(diagnostics):
        for key, value in row.items():
            if value and not math.isfinite(float(value)):
                raise RuntimeError(f'{env_id} diagnostics row {row_index} has {key}={value}')
    warmup = json.loads((log_dir / 'artifacts/warmup_status.json').read_text(encoding='utf-8'))
    evaluation = json.loads(
        (log_dir / f'artifacts/evaluation_step_{args.steps}.json').read_text(encoding='utf-8'),
    )
    receipt.update(
        {
            'matrix_elapsed_seconds': time.monotonic() - started,
            'diagnostics_rows': len(diagnostics),
            'last_diagnostics': diagnostics[-1],
            'final_warmup': warmup,
            'final_evaluation': evaluation,
            'launcher_log': str(log_path),
        },
    )
    atomic_json(result_path, receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments for the compatibility matrix."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--env-ids', nargs='+', default=list(ENVIRONMENTS))
    result.add_argument('--steps', type=int, default=20_000)
    result.add_argument('--steps-per-epoch', type=int, default=5_000)
    result.add_argument('--random-steps', type=int, default=2_000)
    result.add_argument('--batch-size', type=int, default=256)
    result.add_argument('--seed', type=int, default=2027)
    result.add_argument('--eval-seed', type=int, default=12_027)
    result.add_argument('--eval-episodes', type=int, default=1)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--python', type=Path, default=Path(sys.executable))
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument('--summary', type=Path, required=True)
    result.add_argument('--hourly-rate-cny', type=float, default=1.48)
    result.add_argument('--resume', action='store_true')
    return result


def main() -> None:
    """Execute all requested environments sequentially and checkpoint the summary."""
    args = parser().parse_args()
    if min(args.steps, args.steps_per_epoch, args.random_steps, args.batch_size) <= 0:
        raise ValueError('step and batch arguments must be positive')
    if args.steps % args.steps_per_epoch:
        raise ValueError('--steps must be divisible by --steps-per-epoch')
    repo_root = Path(__file__).resolve().parents[2]
    commit = git_commit(repo_root)
    results: list[dict[str, Any]] = []
    matrix_started = time.monotonic()
    for env_id in args.env_ids:
        print(f'[{utc_now()}] starting STAR smoke for {env_id}', flush=True)
        results.append(run_environment(args, repo_root, env_id))
        payload = {
            'status': 'running',
            'captured_at_utc': utc_now(),
            'git_commit': commit,
            'steps_per_environment': args.steps,
            'results': results,
        }
        atomic_json(args.summary, payload)
        print(f'[{utc_now()}] completed STAR smoke for {env_id}', flush=True)

    elapsed = time.monotonic() - matrix_started
    payload.update(
        {
            'status': 'passed',
            'captured_at_utc': utc_now(),
            'matrix_elapsed_seconds': elapsed,
            'estimated_cost_cny': elapsed * args.hourly_rate_cny / 3600.0,
        },
    )
    atomic_json(args.summary, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()

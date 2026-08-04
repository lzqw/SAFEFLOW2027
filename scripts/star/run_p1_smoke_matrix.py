#!/usr/bin/env python3
"""Run checkpointed 20k SACLag smoke tests on all four P1 environments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DEFAULT_ENVS = (
    'SafetyPointGoal1-v0',
    'SafetyPointButton1-v0',
    'Ant-v4',
    'Walker2d-v4',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(f'{json.dumps(payload, indent=2, sort_keys=True)}\n', encoding='utf-8')
    os.replace(temporary, path)


def read_progress(log_dir: Path) -> list[dict[str, str]]:
    with (log_dir / 'progress.csv').open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def assert_finite_progress(rows: list[dict[str, str]], env_id: str) -> None:
    if not rows:
        raise RuntimeError(f'{env_id} produced no progress rows')
    prefixes = ('Loss/', 'Metrics/', 'Time/', 'Value/')
    for row_index, row in enumerate(rows):
        for key, value in row.items():
            if key.startswith(prefixes) and value and not math.isfinite(float(value)):
                raise RuntimeError(f'{env_id} row {row_index} has non-finite {key}={value}')


def checkpoint_epoch(path: Path) -> int:
    match = re.fullmatch(r'epoch-([0-9]+)\.pt', path.name)
    return int(match.group(1)) if match else -1


def run_environment(args: argparse.Namespace, env_id: str, env_index: int) -> dict[str, Any]:
    import omnisafe

    seed = args.seed + env_index
    cost_limit = 10.0 if env_id.startswith('SafetyPoint') else 0.0
    env_output = args.output_dir / env_id
    log_root = env_output / 'runs'
    result_path = env_output / 'result.json'
    resume_signature = {
        'status': 'passed',
        'algorithm': 'SACLag',
        'env_id': env_id,
        'seed': seed,
        'total_steps': args.steps,
        'steps_per_epoch': args.steps_per_epoch,
        'start_learning_steps': args.start_learning_steps,
        'batch_size': args.batch_size,
        'checkpoint_interval_epochs': args.checkpoint_interval_epochs,
        'device': args.device,
    }
    if args.resume and result_path.exists():
        existing = json.loads(result_path.read_text(encoding='utf-8'))
        if all(existing.get(key) == value for key, value in resume_signature.items()):
            return existing

    custom_cfgs: dict[str, Any] = {
        'seed': seed,
        'train_cfgs': {
            'device': args.device,
            'torch_threads': 1,
            'vector_env_nums': 1,
            'parallel': 1,
            'total_steps': args.steps,
            'eval_episodes': 1,
        },
        'algo_cfgs': {
            'steps_per_epoch': args.steps_per_epoch,
            'update_cycle': 1,
            'update_iters': 1,
            'size': max(50_000, args.steps * 2),
            'batch_size': args.batch_size,
            'start_learning_steps': args.start_learning_steps,
            'reward_normalize': False,
            'cost_normalize': False,
            'obs_normalize': False,
        },
        'logger_cfgs': {
            'log_dir': str(log_root),
            'use_tensorboard': True,
            'use_wandb': False,
            'save_model_freq': args.checkpoint_interval_epochs,
        },
        'lagrange_cfgs': {'cost_limit': cost_limit},
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    agent = omnisafe.Agent('SACLag', env_id, custom_cfgs=custom_cfgs)
    final_return, final_cost, final_length = agent.learn()
    training_elapsed = time.monotonic() - started
    expected_updates = max(0, args.steps - (args.start_learning_steps + 1))
    actual_updates = agent.agent._update_count  # pylint: disable=protected-access
    if actual_updates != expected_updates:
        raise RuntimeError(
            f'{env_id} expected {expected_updates} updates, observed {actual_updates}',
        )
    log_dir = Path(agent.agent.logger.log_dir)
    progress = read_progress(log_dir)
    assert_finite_progress(progress, env_id)
    checkpoints = sorted((log_dir / 'torch_save').glob('*.pt'), key=checkpoint_epoch)
    if not checkpoints:
        raise RuntimeError(f'{env_id} did not save a checkpoint')
    latest_checkpoint = checkpoints[-1]
    expected_final_epoch = args.steps // args.steps_per_epoch
    if checkpoint_epoch(latest_checkpoint) != expected_final_epoch:
        raise RuntimeError(
            f'{env_id} latest checkpoint is {latest_checkpoint.name}, '
            f'expected epoch-{expected_final_epoch}.pt',
        )

    restore_started = time.monotonic()
    evaluator = omnisafe.Evaluator(render_mode='rgb_array')
    evaluator.load_saved(save_dir=str(log_dir), model_name=latest_checkpoint.name)
    restored_returns, restored_costs = evaluator.evaluate(num_episodes=1)
    restore_elapsed = time.monotonic() - restore_started
    if not all(math.isfinite(value) for value in [*restored_returns, *restored_costs]):
        raise RuntimeError(f'{env_id} restored evaluation returned non-finite metrics')

    result = {
        'status': 'passed',
        'timestamp_utc': utc_now(),
        'algorithm': 'SACLag',
        'env_id': env_id,
        'seed': seed,
        'cost_limit': cost_limit,
        'total_steps': args.steps,
        'steps_per_epoch': args.steps_per_epoch,
        'start_learning_steps': args.start_learning_steps,
        'batch_size': args.batch_size,
        'checkpoint_interval_epochs': args.checkpoint_interval_epochs,
        'device': args.device,
        'training_elapsed_seconds': training_elapsed,
        'training_end_to_end_fps': args.steps / training_elapsed,
        'updates_expected': expected_updates,
        'updates_actual': actual_updates,
        'final_episode_return': final_return,
        'final_episode_cost': final_cost,
        'final_episode_length': final_length,
        'peak_cuda_memory_mib': (
            torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        ),
        'log_dir': str(log_dir),
        'progress_rows': len(progress),
        'last_progress_row': progress[-1],
        'checkpoint_count': len(checkpoints),
        'restored_checkpoint': latest_checkpoint.name,
        'restore_evaluation_elapsed_seconds': restore_elapsed,
        'restored_episode_return': restored_returns[0],
        'restored_episode_cost': restored_costs[0],
        'train_eval_seed_offset': 10_000,
    }
    atomic_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-ids', nargs='+', default=list(DEFAULT_ENVS))
    parser.add_argument('--steps', type=int, default=20_000)
    parser.add_argument('--steps-per-epoch', type=int, default=1000)
    parser.add_argument('--start-learning-steps', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--checkpoint-interval-epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.steps <= 0 or args.steps_per_epoch <= 0:
        parser.error('step counts must be positive')
    if args.steps % args.steps_per_epoch:
        parser.error('--steps must be divisible by --steps-per-epoch')
    epochs = args.steps // args.steps_per_epoch
    if epochs % args.checkpoint_interval_epochs:
        parser.error('epoch count must be divisible by --checkpoint-interval-epochs')

    results: list[dict[str, Any]] = []
    for env_index, env_id in enumerate(args.env_ids):
        print(f'[{utc_now()}] starting {args.steps}-step gate for {env_id}', flush=True)
        results.append(run_environment(args, env_id, env_index))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        atomic_json(
            args.summary,
            {
                'status': 'running',
                'timestamp_utc': utc_now(),
                'requested_environments': args.env_ids,
                'results': results,
            },
        )

    payload = {
        'status': 'passed',
        'timestamp_utc': utc_now(),
        'requested_environments': args.env_ids,
        'steps_per_environment': args.steps,
        'results': results,
    }
    atomic_json(args.summary, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()

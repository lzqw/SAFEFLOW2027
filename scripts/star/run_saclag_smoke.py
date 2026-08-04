#!/usr/bin/env python3
"""Run a short, checkpointed OmniSafe SACLag training/evaluation smoke test."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def last_progress_row(log_dir: Path) -> dict[str, str]:
    progress = log_dir / 'progress.csv'
    if not progress.exists():
        return {}
    with progress.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-id', default='SafetyPointGoal1-v0')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--steps-per-epoch', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.steps_per_epoch <= 0:
        parser.error('step counts must be positive')
    if args.steps % args.steps_per_epoch:
        parser.error('--steps must be divisible by --steps-per-epoch')

    import omnisafe

    args.log_dir.mkdir(parents=True, exist_ok=True)
    custom_cfgs: dict[str, Any] = {
        'seed': args.seed,
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
            'size': max(20_000, args.steps * 2),
            'batch_size': 256,
            'start_learning_steps': 256,
            'reward_normalize': False,
            'cost_normalize': False,
            'obs_normalize': False,
        },
        'logger_cfgs': {
            'log_dir': str(args.log_dir),
            'use_tensorboard': True,
            'use_wandb': False,
            'save_model_freq': 1,
        },
        'lagrange_cfgs': {'cost_limit': 10.0},
    }

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    started = time.monotonic()
    agent = omnisafe.Agent('SACLag', args.env_id, custom_cfgs=custom_cfgs)
    ep_ret, ep_cost, ep_len = agent.learn()
    elapsed = time.monotonic() - started
    expected_updates = max(0, args.steps - 257)
    actual_updates = agent.agent._update_count  # pylint: disable=protected-access
    if actual_updates != expected_updates:
        raise RuntimeError(
            f'expected {expected_updates} real updates, observed {actual_updates}',
        )
    resolved_log_dir = Path(agent.agent.logger.log_dir)
    checkpoints = sorted(
        str(path.relative_to(resolved_log_dir))
        for path in (resolved_log_dir / 'torch_save').glob('*.pt')
    )
    if not checkpoints:
        raise RuntimeError('SACLag smoke completed without writing a checkpoint')

    result = {
        'status': 'passed',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'algorithm': 'SACLag',
        'env_id': args.env_id,
        'seed': args.seed,
        'total_steps': args.steps,
        'steps_per_epoch': args.steps_per_epoch,
        'updates_expected': expected_updates,
        'updates_actual': actual_updates,
        'elapsed_seconds': elapsed,
        'end_to_end_fps': args.steps / elapsed,
        'final_episode_return': ep_ret,
        'final_episode_cost': ep_cost,
        'final_episode_length': ep_len,
        'peak_cuda_memory_mib': (
            torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        ),
        'log_dir': str(resolved_log_dir),
        'checkpoints': checkpoints,
        'last_progress_row': last_progress_row(resolved_log_dir),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(f'{json.dumps(result, indent=2, sort_keys=True)}\n', encoding='utf-8')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

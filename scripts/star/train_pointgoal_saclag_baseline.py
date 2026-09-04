#!/usr/bin/env python3
"""Train one frozen OmniSafe SACLag Point-task baseline run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_ENVIRONMENTS = ('SafetyPointGoal1-v0', 'SafetyPointCircle1-v0')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def last_progress_row(log_dir: Path) -> dict[str, str]:
    progress = log_dir / 'progress.csv'
    if not progress.is_file():
        return {}
    with progress.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--environment',
        choices=SUPPORTED_ENVIRONMENTS,
        default='SafetyPointGoal1-v0',
    )
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps-per-epoch', type=int, default=2_000)
    parser.add_argument('--replay-size', type=int, default=400_000)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--phase', choices=('throughput', 'formal'), required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.steps_per_epoch <= 0:
        parser.error('step counts must be positive')
    if args.steps % args.steps_per_epoch:
        parser.error('--steps must be divisible by --steps-per-epoch')
    if args.replay_size < args.steps:
        parser.error('--replay-size must retain every transition for cumulative-cost audit')
    return args


def main() -> None:
    args = parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

    import torch

    import omnisafe

    epochs = args.steps // args.steps_per_epoch
    custom_cfgs: dict[str, Any] = {
        'seed': args.seed,
        'train_cfgs': {
            'device': args.device,
            'torch_threads': 1,
            'vector_env_nums': 1,
            'parallel': 1,
            'total_steps': args.steps,
            'eval_episodes': 0,
        },
        'algo_cfgs': {
            'steps_per_epoch': args.steps_per_epoch,
            'update_cycle': 1,
            'update_iters': 1,
            'size': args.replay_size,
            'batch_size': 256,
            'start_learning_steps': 10_000,
            'reward_normalize': False,
            'cost_normalize': False,
            'obs_normalize': False,
        },
        'logger_cfgs': {
            'log_dir': str(args.log_dir),
            'use_tensorboard': False,
            'use_wandb': False,
            'save_model_freq': epochs,
        },
        'lagrange_cfgs': {'cost_limit': 10.0},
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    wrapper = omnisafe.Agent('SACLag', args.environment, custom_cfgs=custom_cfgs)
    algorithm = wrapper.agent
    ep_ret, ep_cost, ep_len = wrapper.learn()
    elapsed = time.monotonic() - started

    replay_size = int(algorithm._buf.size)  # pylint: disable=protected-access
    cumulative_cost = float(  # pylint: disable=protected-access
        algorithm._buf.data['cost'][:replay_size].sum().item()
    )
    resolved_log_dir = Path(algorithm.logger.log_dir)
    checkpoint = resolved_log_dir / 'torch_save' / f'epoch-{epochs}.pt'
    if not checkpoint.is_file():
        raise RuntimeError(f'final SACLag checkpoint is missing: {checkpoint}')
    if replay_size != args.steps:
        raise RuntimeError(f'expected {args.steps} replay transitions, observed {replay_size}')

    payload = {
        'algorithm': 'SACLag',
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256_file(checkpoint),
        'elapsed_seconds': elapsed,
        'end_to_end_fps': args.steps / elapsed,
        'environment': args.environment,
        'final_training_episode_cost': float(ep_cost),
        'final_training_episode_length': float(ep_len),
        'final_training_episode_return': float(ep_ret),
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'last_progress_row': last_progress_row(resolved_log_dir),
        'learner_vector_envs': 1,
        'off_policy_utd_ratio': 1.0,
        'phase': args.phase,
        'protocol_sha256': args.protocol_sha256,
        'replay_transitions': replay_size,
        'seed': args.seed,
        'source_commit': args.source_commit,
        'status': 'complete',
        'steps': args.steps,
        'training_cumulative_cost': cumulative_cost,
        'training_configuration': custom_cfgs,
    }
    if torch.cuda.is_available():
        payload['peak_cuda_memory_mib'] = torch.cuda.max_memory_allocated() / 2**20
    atomic_json(args.result, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

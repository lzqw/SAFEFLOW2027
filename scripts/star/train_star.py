#!/usr/bin/env python3
"""Launch one reproducible STAR route-A training or full-checkpoint recovery run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import torch
import yaml

import omnisafe
from omnisafe.algorithms.off_policy.star import STAR


def deep_merge(destination: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar/list leaves."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            deep_merge(destination[key], value)
        else:
            destination[key] = value
    return destination


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated list of integer steps or seeds."""
    return [int(item.strip()) for item in value.split(',') if item.strip()]


def load_config(path: Path | None) -> dict[str, Any]:
    """Load an optional small experiment override file."""
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} must contain a YAML mapping')
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a launcher result without exposing partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(f'{json.dumps(payload, indent=2, sort_keys=True)}\n', encoding='utf-8')
    os.replace(temporary, path)


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 fingerprint."""
    return hashlib.sha256(payload).hexdigest()


def source_revision() -> dict[str, Any]:
    """Return the exact Git source identity without mutating the worktree."""
    try:
        revision = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ['git', 'status', '--porcelain'],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {'source_commit': None, 'source_dirty': None}
    return {'source_commit': revision, 'source_dirty': bool(status.strip())}


def parser() -> argparse.ArgumentParser:
    """Build the single-run launcher parser."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', type=Path)
    result.add_argument('--env-id')
    result.add_argument('--steps', type=int)
    result.add_argument('--steps-per-epoch', type=int, default=5000)
    result.add_argument('--seed', type=int)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--vector-envs', type=int, default=1)
    result.add_argument('--batch-size', type=int, default=256)
    result.add_argument('--replay-size', type=int)
    result.add_argument('--eval-episodes', type=int)
    result.add_argument('--eval-seeds')
    result.add_argument('--eval-milestones')
    result.add_argument('--random-steps', type=int)
    result.add_argument('--terminal-utility-steps', type=int)
    result.add_argument('--geometry-min-steps', type=int)
    result.add_argument('--hidden-size', type=int)
    result.add_argument('--log-dir', type=Path, default=Path('./runs/star'))
    result.add_argument('--result', type=Path)
    result.add_argument('--resume', type=Path)
    checkpoint = result.add_mutually_exclusive_group()
    checkpoint.add_argument('--checkpoint-replay', dest='checkpoint_replay', action='store_true')
    checkpoint.add_argument(
        '--no-checkpoint-replay',
        dest='checkpoint_replay',
        action='store_false',
    )
    result.set_defaults(checkpoint_replay=None)
    result.add_argument('--full-checkpoint-freq', type=int)
    result.add_argument('--no-tensorboard', action='store_true')
    result.add_argument('--hourly-rate-cny', type=float, default=1.48)
    return result


def main() -> None:  # pylint: disable=too-many-locals,too-many-branches
    """Resolve CLI/config precedence, train, and emit a compact run receipt."""
    args = parser().parse_args()
    file_config = load_config(args.config)
    env_id = args.env_id or file_config.get('env_id') or 'SafetyPointGoal1-v0'
    seed = args.seed if args.seed is not None else int(file_config.get('seed', 2027))
    custom: dict[str, Any] = {}
    deep_merge(custom, file_config.get('custom_cfgs', {}))
    configured_steps = custom.get('train_cfgs', {}).get('total_steps')
    steps = args.steps if args.steps is not None else int(configured_steps or 20_000)
    if steps <= 0 or args.steps_per_epoch <= 0:
        raise ValueError('training and logging step counts must be positive')
    eval_episodes = (
        args.eval_episodes
        if args.eval_episodes is not None
        else int(custom.get('train_cfgs', {}).get('eval_episodes', 5))
    )
    if args.eval_milestones:
        milestones = parse_int_list(args.eval_milestones)
    elif steps >= 300_000 and env_id == 'SafetyPointGoal1-v0':
        milestones = [0, 50_000, 100_000, 200_000, 300_000]
    else:
        milestones = [0, steps]
    milestones = sorted({value for value in milestones if 0 <= value <= steps} | {0, steps})
    eval_seeds = parse_int_list(args.eval_seeds) if args.eval_seeds else None
    replay_size = args.replay_size or max(args.batch_size + 1, min(1_000_000, steps * 2))

    overrides: dict[str, Any] = {
        'seed': seed,
        'train_cfgs': {
            'device': args.device,
            'torch_threads': 1,
            'vector_env_nums': args.vector_envs,
            'parallel': 1,
            'total_steps': steps,
            'eval_episodes': eval_episodes,
        },
        'algo_cfgs': {
            'steps_per_epoch': min(args.steps_per_epoch, steps),
            'size': replay_size,
            'batch_size': args.batch_size,
            'eval_milestones': milestones,
            'resume_from': str(args.resume or ''),
        },
        'logger_cfgs': {
            'log_dir': str(args.log_dir),
            'use_tensorboard': not args.no_tensorboard,
            'use_wandb': False,
        },
    }
    optional_algo = {
        'random_steps': args.random_steps,
        'terminal_utility_steps': args.terminal_utility_steps,
        'geometry_min_steps': args.geometry_min_steps,
        'checkpoint_replay': args.checkpoint_replay,
        'full_checkpoint_freq': args.full_checkpoint_freq,
    }
    overrides['algo_cfgs'].update(
        {key: value for key, value in optional_algo.items() if value is not None},
    )
    if eval_seeds is not None:
        overrides['algo_cfgs']['eval_seeds'] = eval_seeds
    if args.hidden_size is not None:
        overrides['model_cfgs'] = {
            'actor': {'hidden_sizes': [args.hidden_size, args.hidden_size]},
            'critic': {'hidden_sizes': [args.hidden_size, args.hidden_size]},
        }
    deep_merge(custom, overrides)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    wrapper = omnisafe.Agent('STAR', env_id, custom_cfgs=custom)
    final_return, final_cost, final_length = wrapper.learn()
    elapsed = time.monotonic() - started
    algorithm = cast(STAR, wrapper.agent)
    log_dir = Path(algorithm.logger.log_dir)
    summary = json.loads(
        (log_dir / 'artifacts/training_summary.json').read_text(encoding='utf-8'),
    )
    receipt = {
        'status': 'passed',
        'algorithm': 'STAR',
        'environment': env_id,
        'seed': seed,
        'steps': algorithm.env_steps,
        'run_steps': algorithm.run_env_steps,
        'updates': algorithm.update_count,
        'elapsed_seconds': elapsed,
        'end_to_end_fps': algorithm.run_env_steps / elapsed,
        'estimated_cost_cny': elapsed * args.hourly_rate_cny / 3600.0,
        'final_episode_return': final_return,
        'final_episode_cost': final_cost,
        'final_episode_length': final_length,
        'peak_cuda_memory_mib': (
            torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        ),
        'config_file': str(args.config) if args.config is not None else None,
        'config_sha256': (
            sha256_bytes(args.config.read_bytes()) if args.config is not None else None
        ),
        'resolved_overrides_sha256': sha256_bytes(
            json.dumps(custom, sort_keys=True, separators=(',', ':')).encode('utf-8'),
        ),
        **source_revision(),
        'log_dir': str(log_dir),
        'training_summary': summary,
    }
    if args.result is not None:
        atomic_json(args.result, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

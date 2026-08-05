#!/usr/bin/env python3
"""Isolate reward-target self-bootstrap cadence on a frozen STAR replay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from scripts.star.analyze_pointgoal_replay import build_algorithm, checkpoint_metadata
from scripts.star.diagnose_reward_critic_tracking import (
    probe,
    reward_update,
    uniform_replay_indices,
)


DEFAULT_COEFFICIENTS = (0.0, 0.00025, 0.0005, 0.001, 0.0025, 0.005)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one compact diagnostic receipt atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(f'{json.dumps(payload, indent=2, sort_keys=True)}\n', encoding='utf-8')
    os.replace(temporary, path)


@torch.no_grad()
def soft_update_reward_target(algorithm: Any, coefficient: float) -> None:
    """Apply one isolated Polyak step to the reward target critic only."""
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError('reward target coefficient must lie in [0, 1]')
    if coefficient == 0.0:
        return
    actor_critic = algorithm._actor_critic  # pylint: disable=protected-access
    for target, online in zip(
        actor_critic.target_reward_critic.parameters(),
        actor_critic.reward_critic.parameters(),
    ):
        target.lerp_(online, coefficient)


@torch.no_grad()
def reward_target_parameter_gap(algorithm: Any) -> dict[str, float]:
    """Measure absolute and online-scale-relative reward target lag."""
    actor_critic = algorithm._actor_critic  # pylint: disable=protected-access
    squared_gap = torch.zeros((), dtype=torch.float64)
    squared_online = torch.zeros((), dtype=torch.float64)
    parameter_count = 0
    for target, online in zip(
        actor_critic.target_reward_critic.parameters(),
        actor_critic.reward_critic.parameters(),
    ):
        target_cpu = target.detach().double().cpu()
        online_cpu = online.detach().double().cpu()
        squared_gap += (target_cpu - online_cpu).square().sum()
        squared_online += online_cpu.square().sum()
        parameter_count += online.numel()
    if parameter_count == 0:
        raise ValueError('reward critic has no parameters')
    gap_rms = float((squared_gap / parameter_count).sqrt().item())
    online_rms = float((squared_online / parameter_count).sqrt().item())
    return {
        'parameter_gap_rms': gap_rms,
        'online_parameter_rms': online_rms,
        'relative_parameter_gap': gap_rms / max(online_rms, torch.finfo(torch.float64).eps),
    }


def run_variant(  # pylint: disable=too-many-arguments
    checkpoint: Path,
    metadata: dict[str, Any],
    work_dir: Path,
    device: str,
    indices: torch.Tensor,
    outer_updates: int,
    coefficient: float,
    probe_samples: int,
    recent_window: int,
    seed: int,
) -> dict[str, Any]:
    """Run exact R2 critic updates while changing only reward-target cadence."""
    algorithm = build_algorithm(checkpoint, metadata, device, work_dir)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    target_generator = torch.Generator(device='cpu')
    target_generator.manual_seed(seed + 1)
    checkpoints = sorted({0, 250, 500, 1000, outer_updates})
    observations: list[dict[str, Any]] = []
    losses: list[float] = []
    for outer in range(outer_updates + 1):
        if outer in checkpoints:
            observations.append(
                {
                    'outer_updates': outer,
                    'reward_gradient_updates': 2 * outer,
                    **probe(algorithm, indices, probe_samples, seed + 2),
                    **reward_target_parameter_gap(algorithm),
                    'recent_training_loss_mean': (
                        sum(losses[-100:]) / min(len(losses), 100) if losses else None
                    ),
                },
            )
        if outer == outer_updates:
            break
        for _ in range(2):
            losses.append(
                reward_update(
                    algorithm,
                    target_samples=4,
                    generator=target_generator,
                    recent_fraction=0.5,
                    recent_window=recent_window,
                ),
            )
        soft_update_reward_target(algorithm, coefficient)
    return {
        'reward_target_polyak_per_outer_update': coefficient,
        'outer_updates': outer_updates,
        'reward_updates_per_outer': 2,
        'target_samples': 4,
        'recent_fraction': 0.5,
        'recent_window': recent_window,
        'actor_updates': 0,
        'energy_dual_updates': 0,
        'observations': observations,
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the target-cadence counterfactual on one immutable checkpoint."""
    coefficients = tuple(float(value) for value in args.coefficients)
    if not coefficients or any(not 0.0 <= value <= 1.0 for value in coefficients):
        raise ValueError('all reward target coefficients must lie in [0, 1]')
    if len(set(coefficients)) != len(coefficients):
        raise ValueError('reward target coefficients must be unique')
    metadata = checkpoint_metadata(args.checkpoint)
    indices = uniform_replay_indices(metadata['replay_size'], args.transition_count)
    variants: dict[str, Any] = {}
    for coefficient in coefficients:
        label = format(coefficient, '.8g').replace('.', 'p')
        variants[f'reward_target_polyak_{label}'] = run_variant(
            args.checkpoint,
            metadata,
            args.work_dir / f'reward_target_polyak_{label}',
            args.device,
            indices,
            args.outer_updates,
            coefficient,
            args.probe_samples,
            args.recent_window,
            args.seed,
        )
    payload = {
        'schema_version': 1,
        'diagnostic': 'frozen_replay_reward_target_timescale',
        'checkpoint': str(args.checkpoint),
        'checkpoint_bytes': args.checkpoint.stat().st_size,
        'environment': metadata['environment'],
        'training_seed': metadata['seed'],
        'env_steps': metadata['env_steps'],
        'replay_size': metadata['replay_size'],
        'transition_count': len(indices),
        'probe_target_samples': args.probe_samples,
        'torch_seed': args.seed,
        'new_environment_transitions': 0,
        'actor_updates': 0,
        'energy_dual_updates': 0,
        'fixed_r2_mechanism': {
            'target_samples': 4,
            'reward_updates_per_outer': 2,
            'recent_fraction': 0.5,
            'recent_window': args.recent_window,
        },
        'variants': variants,
        'interpretation_scope': (
            'post-hoc frozen-policy/frozen-replay target-timescale diagnostic; not training '
            'continuation, a promotion metric, or a formal verdict'
        ),
    }
    atomic_json(args.output, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--checkpoint', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--work-dir', type=Path, required=True)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--transition-count', type=int, default=4096)
    result.add_argument('--outer-updates', type=int, default=2000)
    result.add_argument('--probe-samples', type=int, default=16)
    result.add_argument('--recent-window', type=int, default=10000)
    result.add_argument('--seed', type=int, default=616161)
    result.add_argument(
        '--coefficients',
        type=float,
        nargs='+',
        default=DEFAULT_COEFFICIENTS,
    )
    return result


def main() -> None:
    """Run and print the compact target-timescale diagnostic."""
    payload = diagnose(parser().parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

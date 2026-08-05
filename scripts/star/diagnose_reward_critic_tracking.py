#!/usr/bin/env python3
"""Counterfactually measure STAR reward-critic tracking on a frozen final replay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from scripts.star.analyze_pointgoal_replay import build_algorithm, checkpoint_metadata


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one compact diagnostic receipt atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(f'{json.dumps(payload, indent=2, sort_keys=True)}\n', encoding='utf-8')
    os.replace(temporary, path)


def uniform_replay_indices(size: int, sample_count: int) -> torch.Tensor:
    """Select an evenly spaced replay subset without replacement."""
    count = min(size, sample_count)
    if size <= 0 or count <= 0:
        raise ValueError('replay size and sample count must be positive')
    if count == 1:
        return torch.zeros(1, dtype=torch.long)
    return torch.linspace(0, size - 1, count, dtype=torch.float64).round().long()


@torch.no_grad()
def expected_target(
    algorithm: Any,
    next_augmented_obs: torch.Tensor,
    reward: torch.Tensor,
    reward_done: torch.Tensor,
    sample_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Estimate the stochastic-policy reward backup with caller-isolated randomness."""
    batch_size = next_augmented_obs.shape[0]
    actor_critic = algorithm._actor_critic  # pylint: disable=protected-access
    base = torch.empty(
        (sample_count, batch_size, actor_critic.act_dim),
        dtype=next_augmented_obs.dtype,
        device='cpu',
    ).uniform_(-1.0, 1.0, generator=generator)
    expanded_obs = (
        next_augmented_obs.unsqueeze(0)
        .expand(sample_count, -1, -1)
        .reshape(sample_count * batch_size, -1)
    )
    flow = actor_critic.target_flow(
        expanded_obs,
        base_action=base.reshape(sample_count * batch_size, -1).to(
            next_augmented_obs.device,
        ),
    )
    first, second = actor_critic.target_reward_critic(expanded_obs, flow.action)
    multiplier = algorithm._energy_dual.value.detach()  # pylint: disable=protected-access
    value = (torch.minimum(first, second) - multiplier * flow.kinetic_energy).reshape(
        sample_count,
        batch_size,
    )
    continuation = value.mean(dim=0)
    gamma = float(algorithm._cfgs.algo_cfgs.gamma)  # pylint: disable=protected-access
    return reward + gamma * (1.0 - reward_done) * continuation


@torch.no_grad()
def probe(
    algorithm: Any,
    indices: torch.Tensor,
    probe_samples: int,
    probe_seed: int,
    chunk_size: int = 512,
) -> dict[str, float]:
    """Measure expected-target TD on fixed uniform replay and its newest decile."""
    buffer = algorithm._buf  # pylint: disable=protected-access
    device = algorithm._device  # pylint: disable=protected-access
    generator = torch.Generator(device='cpu')
    generator.manual_seed(probe_seed)
    td_parts: list[torch.Tensor] = []
    q_parts: list[torch.Tensor] = []
    source_indices: list[torch.Tensor] = []
    for start in range(0, indices.numel(), chunk_size):
        selected = indices[start : start + chunk_size]
        data = buffer.data
        obs = data['obs'][selected].to(device)
        context = data['context'][selected].to(device)
        next_obs = data['next_obs'][selected].to(device)
        next_context = data['next_context'][selected].to(device)
        action = data['action'][selected].to(device)
        reward = data['reward'][selected].to(device)
        reward_done = data['reward_done'][selected].to(device)
        augmented = algorithm._augment(obs, context)  # pylint: disable=protected-access
        next_augmented = algorithm._augment(  # pylint: disable=protected-access
            next_obs,
            next_context,
        )
        target = expected_target(
            algorithm,
            next_augmented,
            reward,
            reward_done,
            probe_samples,
            generator,
        )
        first, second = algorithm._actor_critic.reward_critic(  # pylint: disable=protected-access
            augmented,
            action,
        )
        td_parts.append((0.5 * ((first - target).abs() + (second - target).abs())).cpu())
        q_parts.append((0.5 * (first.abs() + second.abs())).cpu())
        source_indices.append(selected)
    td = torch.cat(td_parts)
    q_abs = torch.cat(q_parts)
    replay_indices = torch.cat(source_indices)
    newest = replay_indices >= 9 * buffer.size // 10
    oldest = replay_indices < buffer.size // 10
    return {
        'uniform_td': float(td.mean().item()),
        'uniform_td_p90': float(torch.quantile(td, 0.9).item()),
        'uniform_q_abs_mean': float(q_abs.mean().item()),
        'oldest_decile_td': float(td[oldest].mean().item()),
        'newest_decile_td': float(td[newest].mean().item()),
        'newest_to_oldest_td_ratio': float(td[newest].mean().item() / td[oldest].mean().item()),
    }


def sample_reward_batch(
    algorithm: Any,
    recent_fraction: float,
    recent_window: int,
) -> dict[str, torch.Tensor]:
    """Draw a uniform/recent replay mixture without safety-category stratification."""
    buffer = algorithm._buf  # pylint: disable=protected-access
    requested = buffer.batch_size
    if not 0.0 <= recent_fraction <= 1.0 or recent_window <= 0:
        raise ValueError('recent replay mixture is invalid')
    recent_count = round(requested * recent_fraction)
    uniform_count = requested - recent_count
    parts: list[torch.Tensor] = []
    if uniform_count:
        parts.append(torch.randint(buffer.size, (uniform_count,)))
    if recent_count:
        available = min(buffer.size, recent_window)
        offsets = torch.randint(available, (recent_count,))
        parts.append((buffer.pointer - 1 - offsets).remainder(buffer.capacity))
    indices = torch.cat(parts)
    indices = indices[torch.randperm(indices.numel())]
    device = algorithm._device  # pylint: disable=protected-access
    batch = {key: value[indices].to(device) for key, value in buffer.data.items()}
    batch['indices'] = indices.to(device)
    batch['importance_weights'] = torch.ones(requested, device=device)
    return batch


def reward_update(
    algorithm: Any,
    target_samples: int,
    generator: torch.Generator,
    recent_fraction: float,
    recent_window: int,
) -> float:
    """Apply one isolated reward-critic update without changing the actor or replay."""
    batch = sample_reward_batch(algorithm, recent_fraction, recent_window)
    augmented = algorithm._augment(  # pylint: disable=protected-access
        batch['obs'],
        batch['context'],
    )
    next_augmented = algorithm._augment(  # pylint: disable=protected-access
        batch['next_obs'],
        batch['next_context'],
    )
    target = expected_target(
        algorithm,
        next_augmented,
        batch['reward'],
        batch['reward_done'],
        target_samples,
        generator,
    )
    critic = algorithm._actor_critic.reward_critic  # pylint: disable=protected-access
    first, second = critic(augmented, batch['action'])
    per_sample = (first - target).square() + (second - target).square()
    weights = batch['importance_weights']
    loss = (per_sample * weights).sum() / weights.sum().clamp_min(1.0)
    optimizer = algorithm._actor_critic.reward_critic_optimizer  # pylint: disable=protected-access
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(
        critic.parameters(),
        float(algorithm._cfgs.algo_cfgs.max_grad_norm),  # pylint: disable=protected-access
    )
    optimizer.step()
    return float(loss.item())


def run_variant(  # pylint: disable=too-many-arguments
    checkpoint: Path,
    metadata: dict[str, Any],
    work_dir: Path,
    device: str,
    indices: torch.Tensor,
    outer_updates: int,
    reward_updates_per_outer: int,
    target_samples: int,
    recent_fraction: float,
    recent_window: int,
    probe_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Run one frozen-policy, frozen-replay reward-tracking counterfactual."""
    algorithm = build_algorithm(checkpoint, metadata, device, work_dir)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    target_generator = torch.Generator(device='cpu')
    target_generator.manual_seed(seed + 1)
    checkpoints = sorted({0, 500, 1000, outer_updates})
    observations: list[dict[str, Any]] = []
    losses: list[float] = []
    for outer in range(outer_updates + 1):
        if outer in checkpoints:
            observations.append(
                {
                    'outer_updates': outer,
                    'reward_gradient_updates': outer * reward_updates_per_outer,
                    **probe(algorithm, indices, probe_samples, seed + 2),
                    'recent_training_loss_mean': (
                        sum(losses[-100:]) / min(len(losses), 100) if losses else None
                    ),
                },
            )
        if outer == outer_updates:
            break
        for _ in range(reward_updates_per_outer):
            losses.append(
                reward_update(
                    algorithm,
                    target_samples,
                    target_generator,
                    recent_fraction,
                    recent_window,
                ),
            )
        algorithm._actor_critic.polyak_update(  # pylint: disable=protected-access
            float(algorithm._cfgs.algo_cfgs.polyak),  # pylint: disable=protected-access
            update_actor=False,
        )
    return {
        'target_samples': target_samples,
        'reward_updates_per_outer': reward_updates_per_outer,
        'outer_updates': outer_updates,
        'recent_fraction': recent_fraction,
        'recent_window': recent_window,
        'observations': observations,
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the registered 2x2 exploratory tracking decomposition."""
    metadata = checkpoint_metadata(args.checkpoint)
    indices = uniform_replay_indices(metadata['replay_size'], args.transition_count)
    variants: dict[str, Any] = {}
    variant_specs = (
        (1, 1, 0.0),
        (4, 1, 0.0),
        (1, 2, 0.0),
        (4, 2, 0.0),
        (4, 2, 0.5),
        (4, 2, 0.75),
    )
    for target_samples, updates, recent_fraction in variant_specs:
        recent_label = str(recent_fraction).replace('.', 'p')
        name = (
            f'target_samples_{target_samples}__reward_updates_{updates}'
            f'__recent_fraction_{recent_label}'
        )
        variants[name] = run_variant(
            args.checkpoint,
            metadata,
            args.work_dir / name,
            args.device,
            indices,
            args.outer_updates,
            updates,
            target_samples,
            recent_fraction,
            args.recent_window,
            args.probe_samples,
            args.seed,
        )
    payload = {
        'schema_version': 1,
        'diagnostic': 'frozen_replay_reward_critic_tracking',
        'checkpoint': str(args.checkpoint),
        'checkpoint_bytes': args.checkpoint.stat().st_size,
        'environment': metadata['environment'],
        'training_seed': metadata['seed'],
        'env_steps': metadata['env_steps'],
        'replay_size': metadata['replay_size'],
        'transition_count': len(indices),
        'probe_target_samples': args.probe_samples,
        'torch_seed': args.seed,
        'actor_updates': 0,
        'new_environment_transitions': 0,
        'variants': variants,
        'interpretation_scope': (
            'post-hoc frozen-policy/frozen-replay capacity diagnostic; not training continuation, '
            'a promotion metric, or a formal verdict'
        ),
    }
    atomic_json(args.output, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    """Build CLI arguments for frozen-replay reward tracking."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--checkpoint', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--work-dir', type=Path, required=True)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--transition-count', type=int, default=4096)
    result.add_argument('--outer-updates', type=int, default=2000)
    result.add_argument('--probe-samples', type=int, default=16)
    result.add_argument('--recent-window', type=int, default=10000)
    result.add_argument('--seed', type=int, default=515151)
    return result


def main() -> None:
    """Run the diagnostic and print its receipt."""
    payload = diagnose(parser().parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Measure stochastic-flow variance in STAR reward Bellman targets."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import torch
from scripts.star.analyze_pointgoal_replay import build_algorithm, checkpoint_metadata


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one compact diagnostic receipt atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(f'{json.dumps(payload, indent=2, sort_keys=True)}\n', encoding='utf-8')
    os.replace(temporary, path)


def uniform_replay_indices(size: int, sample_count: int) -> torch.Tensor:
    """Select an evenly spaced, deterministic replay subset without replacement."""
    if size <= 0 or sample_count <= 0:
        raise ValueError('replay size and sample count must be positive')
    count = min(size, sample_count)
    if count == 1:
        return torch.zeros(1, dtype=torch.long)
    return torch.linspace(0, size - 1, count, dtype=torch.float64).round().long()


def summarize_draws(
    targets: torch.Tensor,
    first_q: torch.Tensor,
    second_q: torch.Tensor,
    group_sizes: list[int],
) -> dict[str, Any]:
    """Decompose Monte Carlo target noise and TD residuals for several sample counts."""
    if targets.ndim != 2:
        raise ValueError('targets must have shape [draws, transitions]')
    draws = targets.shape[0]
    reference = targets.mean(dim=0)
    reference_td = 0.5 * ((first_q - reference).abs() + (second_q - reference).abs())
    target_std = targets.std(dim=0, unbiased=False)
    result: dict[str, Any] = {
        'draws': int(draws),
        'target_mean': float(reference.mean().item()),
        'target_std_across_transitions': float(reference.std(unbiased=False).item()),
        'target_sampling_std_mean': float(target_std.mean().item()),
        'target_sampling_std_rms': float(target_std.square().mean().sqrt().item()),
        'target_sampling_std_p90': float(torch.quantile(target_std, 0.9).item()),
        'td_to_all_draw_mean': float(reference_td.mean().item()),
        'td_to_all_draw_p90': float(torch.quantile(reference_td, 0.9).item()),
        'sample_count_estimates': {},
    }
    for count in group_sizes:
        if count <= 0 or draws % count:
            raise ValueError(f'draw count {draws} must be divisible by sample count {count}')
        estimates = targets.reshape(draws // count, count, -1).mean(dim=1)
        errors = estimates - reference
        td = 0.5 * (
            (first_q.unsqueeze(0) - estimates).abs() + (second_q.unsqueeze(0) - estimates).abs()
        )
        per_estimate_td = td.mean(dim=1)
        result['sample_count_estimates'][str(count)] = {
            'independent_estimates': int(estimates.shape[0]),
            'estimator_mae_to_all_draw_mean': float(errors.abs().mean().item()),
            'estimator_rmse_to_all_draw_mean': float(errors.square().mean().sqrt().item()),
            'mean_td': float(td.mean().item()),
            'mean_td_between_estimate_std': float(per_estimate_td.std(unbiased=False).item()),
            'mean_td_min': float(per_estimate_td.min().item()),
            'mean_td_max': float(per_estimate_td.max().item()),
        }
    return result


def summarize_cohort(
    targets: torch.Tensor,
    first_q: torch.Tensor,
    second_q: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float | int]:
    """Summarize expected-target TD and sampling noise for one replay cohort."""
    selected = int(mask.sum().item())
    if selected == 0:
        return {'transitions': 0}
    cohort_targets = targets[:, mask]
    reference = cohort_targets.mean(dim=0)
    td = 0.5 * ((first_q[mask] - reference).abs() + (second_q[mask] - reference).abs())
    return {
        'transitions': selected,
        'reward_mean': float(rewards[mask].mean().item()),
        'q_abs_mean': float((0.5 * (first_q[mask].abs() + second_q[mask].abs())).mean().item()),
        'target_mean': float(reference.mean().item()),
        'td_to_all_draw_mean': float(td.mean().item()),
        'target_sampling_std_mean': float(cohort_targets.std(dim=0, unbiased=False).mean().item()),
    }


def replay_cohorts(
    targets: torch.Tensor,
    first_q: torch.Tensor,
    second_q: torch.Tensor,
    rewards: torch.Tensor,
    indices: torch.Tensor,
    replay_size: int,
    categories: torch.Tensor,
    source_random: torch.Tensor,
) -> dict[str, Any]:
    """Partition expected-target residuals by chronology and replay provenance."""
    chronological: list[dict[str, Any]] = []
    for decile in range(10):
        lower = decile * replay_size // 10
        upper = (decile + 1) * replay_size // 10
        mask = (indices >= lower) & (indices < upper)
        chronological.append(
            {
                'replay_index_lower': lower,
                'replay_index_upper_exclusive': upper,
                **summarize_cohort(targets, first_q, second_q, rewards, mask),
            },
        )
    provenance = {
        name: summarize_cohort(
            targets,
            first_q,
            second_q,
            rewards,
            categories == category,
        )
        for name, category in (('normal', 0), ('boundary', 1), ('failure', 2))
    }
    provenance['source_random'] = summarize_cohort(
        targets,
        first_q,
        second_q,
        rewards,
        source_random,
    )
    provenance['source_policy'] = summarize_cohort(
        targets,
        first_q,
        second_q,
        rewards,
        ~source_random,
    )
    provenance['positive_reward'] = summarize_cohort(
        targets,
        first_q,
        second_q,
        rewards,
        rewards > 0.0,
    )
    provenance['nonpositive_reward'] = summarize_cohort(
        targets,
        first_q,
        second_q,
        rewards,
        rewards <= 0.0,
    )
    return {'chronological_deciles': chronological, 'provenance': provenance}


@torch.no_grad()
def actor_target_draws(
    algorithm: Any,
    next_augmented_obs: torch.Tensor,
    reward: torch.Tensor,
    reward_done: torch.Tensor,
    flow_function: Callable[..., Any],
    draws: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw reward targets, target values, and flow energies from one actor."""
    target_critic = algorithm._actor_critic.target_reward_critic  # pylint: disable=protected-access
    gamma = float(algorithm._cfgs.algo_cfgs.gamma)  # pylint: disable=protected-access
    multiplier = algorithm._energy_dual.value.detach()  # pylint: disable=protected-access
    targets: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    energies: list[torch.Tensor] = []
    for _ in range(draws):
        flow = flow_function(next_augmented_obs)
        first, second = target_critic(next_augmented_obs, flow.action)
        value = torch.minimum(first, second) - multiplier * flow.kinetic_energy
        target = reward + gamma * (1.0 - reward_done) * value
        targets.append(target.cpu())
        values.append(value.cpu())
        energies.append(flow.kinetic_energy.cpu())
    return torch.stack(targets), torch.stack(values), torch.stack(energies)


@torch.no_grad()
def diagnose(
    checkpoint: Path,
    output: Path,
    work_dir: Path,
    device: str,
    transition_count: int,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Restore one full checkpoint and compute the fixed-replay variance decomposition."""
    metadata = checkpoint_metadata(checkpoint)
    algorithm = build_algorithm(checkpoint, metadata, device, work_dir)
    buffer = algorithm._buf  # pylint: disable=protected-access
    actual_count = min(transition_count, buffer.size)
    indices = uniform_replay_indices(buffer.size, actual_count)
    compute_device = algorithm._device  # pylint: disable=protected-access
    data = buffer.data
    obs = data['obs'][indices].to(compute_device)
    context = data['context'][indices].to(compute_device)
    next_obs = data['next_obs'][indices].to(compute_device)
    next_context = data['next_context'][indices].to(compute_device)
    action = data['action'][indices].to(compute_device)
    reward = data['reward'][indices].to(compute_device)
    reward_done = data['reward_done'][indices].to(compute_device)
    augmented_obs = algorithm._augment(obs, context)  # pylint: disable=protected-access
    next_augmented_obs = algorithm._augment(  # pylint: disable=protected-access
        next_obs,
        next_context,
    )
    first_q, second_q = algorithm._actor_critic.reward_critic(  # pylint: disable=protected-access
        augmented_obs,
        action,
    )
    first_q = first_q.cpu()
    second_q = second_q.cpu()
    reward_cpu = reward.cpu()
    categories = data['category'][indices].cpu()
    source_random = data['source_random'][indices].cpu()

    group_sizes = [count for count in (1, 2, 4, 8, 16, 32) if count <= draws]
    if any(draws % count for count in group_sizes):
        raise ValueError('draw count must be divisible by every reported target sample count')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    actor_critic = algorithm._actor_critic  # pylint: disable=protected-access
    actor_results: dict[str, Any] = {}
    for name, flow_function in (
        ('target_actor', actor_critic.target_flow),
        ('online_actor', actor_critic.flow),
    ):
        targets, values, energies = actor_target_draws(
            algorithm,
            next_augmented_obs,
            reward,
            reward_done,
            flow_function,
            draws,
        )
        actor_results[name] = {
            **summarize_draws(targets, first_q, second_q, group_sizes),
            'next_value_sampling_std_mean': float(values.std(dim=0, unbiased=False).mean().item()),
            'energy_mean': float(energies.mean().item()),
            'energy_sampling_std_mean': float(energies.std(dim=0, unbiased=False).mean().item()),
            'replay_cohorts': replay_cohorts(
                targets,
                first_q,
                second_q,
                reward_cpu,
                indices,
                buffer.size,
                categories,
                source_random,
            ),
        }

    payload: dict[str, Any] = {
        'schema_version': 1,
        'diagnostic': 'reward_target_sampling_variance',
        'checkpoint': str(checkpoint),
        'checkpoint_bytes': checkpoint.stat().st_size,
        'environment': metadata['environment'],
        'training_seed': metadata['seed'],
        'env_steps': metadata['env_steps'],
        'replay_size': buffer.size,
        'selection': 'evenly_spaced_resident_replay_indices',
        'transition_count': actual_count,
        'draws': draws,
        'torch_seed': seed,
        'device': str(compute_device),
        'reward_q_abs_mean': float((0.5 * (first_q.abs() + second_q.abs())).mean().item()),
        'reward_q_twin_gap_mean': float((first_q - second_q).abs().mean().item()),
        'reward_mean': float(reward.mean().item()),
        'reward_done_fraction': float(reward_done.mean().item()),
        'energy_multiplier': float(
            algorithm._energy_dual.value.detach().item(),  # pylint: disable=protected-access
        ),
        'actors': actor_results,
        'interpretation_scope': (
            'read-only final-replay diagnostic; not a registered promotion metric or formal verdict'
        ),
    }
    for actor in actor_results.values():
        reference_td = actor['td_to_all_draw_mean']
        sampling_std = actor['target_sampling_std_mean']
        actor['sampling_std_to_reference_td_ratio'] = (
            sampling_std / reference_td if not math.isclose(reference_td, 0.0) else None
        )
    atomic_json(output, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the read-only variance diagnostic."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--checkpoint', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--work-dir', type=Path, required=True)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--transition-count', type=int, default=4096)
    result.add_argument('--draws', type=int, default=32)
    result.add_argument('--seed', type=int, default=424242)
    return result


def main() -> None:
    """Run the diagnostic and print its machine-readable receipt."""
    args = parser().parse_args()
    payload = diagnose(
        args.checkpoint,
        args.output,
        args.work_dir,
        args.device,
        args.transition_count,
        args.draws,
        args.seed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

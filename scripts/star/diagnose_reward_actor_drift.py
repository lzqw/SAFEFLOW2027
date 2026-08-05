#!/usr/bin/env python3
"""Isolate actor and energy-dual drift in the STAR reward subsystem."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from scripts.star.analyze_pointgoal_replay import build_algorithm, checkpoint_metadata
from scripts.star.diagnose_reward_critic_tracking import probe, uniform_replay_indices
from scripts.star.diagnose_reward_target_timescale import (
    reward_target_parameter_gap,
    soft_update_reward_target,
)


VARIANT_SPECS = (
    {
        'name': 'target_0p001__actor_frozen__dual_frozen',
        'reward_target_polyak': 0.001,
        'actor_interval': None,
        'active_energy_dual': False,
    },
    {
        'name': 'target_0p001__actor_2__dual_frozen',
        'reward_target_polyak': 0.001,
        'actor_interval': 2,
        'active_energy_dual': False,
    },
    {
        'name': 'target_0p001__actor_2__dual_active',
        'reward_target_polyak': 0.001,
        'actor_interval': 2,
        'active_energy_dual': True,
    },
    {
        'name': 'target_0p005__actor_2__dual_active',
        'reward_target_polyak': 0.005,
        'actor_interval': 2,
        'active_energy_dual': True,
    },
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one compact diagnostic receipt atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(f'{json.dumps(payload, indent=2, sort_keys=True)}\n', encoding='utf-8')
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    """Reset CPU and CUDA generators so subsystem variants share exogenous draws."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def soft_update_module(target: nn.Module, online: nn.Module, coefficient: float) -> None:
    """Apply a validated Polyak update to one matching module pair."""
    if not 0.0 < coefficient <= 1.0:
        raise ValueError('module target coefficient must lie in (0, 1]')
    for target_parameter, online_parameter in zip(target.parameters(), online.parameters()):
        target_parameter.lerp_(online_parameter, coefficient)


@torch.no_grad()
def module_parameter_gap(target: nn.Module, online: nn.Module) -> dict[str, float]:
    """Return absolute and online-scale-relative parameter lag for a module pair."""
    squared_gap = torch.zeros((), dtype=torch.float64)
    squared_online = torch.zeros((), dtype=torch.float64)
    parameter_count = 0
    for target_parameter, online_parameter in zip(target.parameters(), online.parameters()):
        target_cpu = target_parameter.detach().double().cpu()
        online_cpu = online_parameter.detach().double().cpu()
        squared_gap += (target_cpu - online_cpu).square().sum()
        squared_online += online_cpu.square().sum()
        parameter_count += online_parameter.numel()
    if parameter_count == 0:
        raise ValueError('module has no parameters')
    gap_rms = float((squared_gap / parameter_count).sqrt().item())
    online_rms = float((squared_online / parameter_count).sqrt().item())
    return {
        'parameter_gap_rms': gap_rms,
        'online_parameter_rms': online_rms,
        'relative_parameter_gap': gap_rms / max(online_rms, torch.finfo(torch.float64).eps),
    }


@torch.no_grad()
def policy_probe(
    algorithm: Any,
    indices: torch.Tensor,
    base_seed: int,
) -> dict[str, float]:
    """Measure online/target flow and terminal value on one fixed replay cohort."""
    buffer = algorithm._buf  # pylint: disable=protected-access
    device = algorithm._device  # pylint: disable=protected-access
    selected = indices[: min(indices.numel(), 512)]
    data = buffer.data
    augmented = algorithm._augment(  # pylint: disable=protected-access
        data['obs'][selected].to(device),
        data['context'][selected].to(device),
    )
    actor_critic = algorithm._actor_critic  # pylint: disable=protected-access
    generator = torch.Generator(device='cpu')
    generator.manual_seed(base_seed)
    base_action = torch.empty(
        (selected.numel(), actor_critic.act_dim),
        dtype=augmented.dtype,
        device='cpu',
    ).uniform_(-1.0, 1.0, generator=generator).to(device)
    online_flow = actor_critic.flow(augmented, base_action=base_action)
    target_flow = actor_critic.target_flow(augmented, base_action=base_action)
    first, second = actor_critic.reward_critic(augmented, online_flow.action)
    actor_gap = module_parameter_gap(actor_critic.target_actor, actor_critic.actor)
    return {
        'online_flow_energy_mean': float(online_flow.kinetic_energy.mean().item()),
        'target_flow_energy_mean': float(target_flow.kinetic_energy.mean().item()),
        'target_online_action_rms': float(
            (target_flow.action - online_flow.action).square().mean().sqrt().item(),
        ),
        'online_actor_terminal_reward_mean': float(torch.minimum(first, second).mean().item()),
        'energy_dual': float(algorithm._energy_dual.value.item()),  # pylint: disable=protected-access
        'actor_target_parameter_gap_rms': actor_gap['parameter_gap_rms'],
        'actor_target_relative_parameter_gap': actor_gap['relative_parameter_gap'],
    }


def reward_update(algorithm: Any) -> torch.Tensor:
    """Apply one exact R2 reward update and return its actor observation batch."""
    batch = algorithm._sample_reward_batch()  # pylint: disable=protected-access
    augmented = algorithm._augment(  # pylint: disable=protected-access
        batch['obs'],
        batch['context'],
    )
    next_augmented = algorithm._augment(  # pylint: disable=protected-access
        batch['next_obs'],
        batch['next_context'],
    )
    algorithm._update_reward_critics(  # pylint: disable=protected-access
        batch,
        augmented,
        next_augmented,
    )
    return augmented


def actor_update(algorithm: Any, augmented: torch.Tensor, active_energy_dual: bool) -> None:
    """Apply an actor step, optionally restoring the energy dual and its optimizer."""
    if active_energy_dual:
        algorithm._update_actor(augmented)  # pylint: disable=protected-access
        return
    dual = algorithm._energy_dual  # pylint: disable=protected-access
    dual_state = copy.deepcopy(dual.state_dict())
    optimizer_state = copy.deepcopy(dual.optimizer.state_dict())
    algorithm._update_actor(augmented)  # pylint: disable=protected-access
    dual.load_state_dict(dual_state)
    dual.optimizer.load_state_dict(optimizer_state)


def run_variant(  # pylint: disable=too-many-arguments,too-many-locals
    checkpoint: Path,
    metadata: dict[str, Any],
    work_dir: Path,
    device: str,
    indices: torch.Tensor,
    outer_updates: int,
    probe_samples: int,
    recent_window: int,
    seed: int,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Run one isolated reward/actor closed-loop counterfactual."""
    algorithm = build_algorithm(checkpoint, metadata, device, work_dir)
    algo = algorithm._cfgs.algo_cfgs  # pylint: disable=protected-access
    actual_r2 = {
        'target_samples': int(algo.reward_target_samples),
        'separate_reward_replay': bool(algo.separate_reward_replay),
        'recent_fraction': float(algo.reward_replay_recent_fraction),
        'recent_window': int(algo.reward_replay_recent_window),
        'target_actor': bool(algo.use_target_flow_for_reward_backup),
    }
    expected_r2 = {
        'target_samples': 4,
        'separate_reward_replay': True,
        'recent_fraction': 0.5,
        'recent_window': recent_window,
        'target_actor': True,
    }
    if actual_r2 != expected_r2:
        raise ValueError(f'checkpoint does not restore the frozen R2 mechanism: {actual_r2}')
    reward_target_polyak = float(spec['reward_target_polyak'])
    actor_interval = spec['actor_interval']
    active_energy_dual = bool(spec['active_energy_dual'])
    checkpoints = sorted({0, 250, 500, 1000, outer_updates})
    observations: list[dict[str, Any]] = []
    actor_updates = 0
    energy_dual_updates = 0
    for outer in range(outer_updates + 1):
        if outer in checkpoints:
            actor_critic = algorithm._actor_critic  # pylint: disable=protected-access
            observations.append(
                {
                    'outer_updates': outer,
                    'reward_gradient_updates': 2 * outer,
                    'actor_updates': actor_updates,
                    'energy_dual_updates': energy_dual_updates,
                    **probe(algorithm, indices, probe_samples, seed + 2),
                    **reward_target_parameter_gap(algorithm),
                    **policy_probe(algorithm, indices, seed + 3),
                    'reward_target_polyak_per_outer_update': reward_target_polyak,
                    'actor_target_polyak_per_actor_update': float(
                        algorithm._cfgs.algo_cfgs.polyak,  # pylint: disable=protected-access
                    ),
                },
            )
        if outer == outer_updates:
            break
        actor_augmented = None
        for reward_index in range(2):
            seed_all(seed + 10_000 * outer + reward_index)
            actor_augmented = reward_update(algorithm)
        assert actor_augmented is not None
        actor_applied = actor_interval is not None and outer % int(actor_interval) == 0
        if actor_applied:
            seed_all(seed + 1_000_000 + actor_updates)
            actor_update(algorithm, actor_augmented, active_energy_dual)
            actor_updates += 1
            energy_dual_updates += int(active_energy_dual)
        soft_update_reward_target(algorithm, reward_target_polyak)
        if actor_applied:
            actor_critic = algorithm._actor_critic  # pylint: disable=protected-access
            soft_update_module(
                actor_critic.target_actor,
                actor_critic.actor,
                float(algorithm._cfgs.algo_cfgs.polyak),  # pylint: disable=protected-access
            )
    return {
        **spec,
        'outer_updates': outer_updates,
        'reward_updates_per_outer': 2,
        'target_samples': 4,
        'recent_fraction': 0.5,
        'recent_window': recent_window,
        'actor_updates': actor_updates,
        'energy_dual_updates': energy_dual_updates,
        'observations': observations,
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the isolated reward/actor subsystem variants."""
    metadata = checkpoint_metadata(args.checkpoint)
    indices = uniform_replay_indices(metadata['replay_size'], args.transition_count)
    variants = {
        spec['name']: run_variant(
            args.checkpoint,
            metadata,
            args.work_dir / spec['name'],
            args.device,
            indices,
            args.outer_updates,
            args.probe_samples,
            args.recent_window,
            args.seed,
            spec,
        )
        for spec in VARIANT_SPECS
    }
    payload = {
        'schema_version': 1,
        'diagnostic': 'frozen_replay_reward_actor_drift',
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
        'safety_critic_updates': 0,
        'fixed_r2_mechanism': {
            'target_samples': 4,
            'reward_updates_per_outer': 2,
            'recent_fraction': 0.5,
            'recent_window': args.recent_window,
        },
        'variants': variants,
        'interpretation_scope': (
            'post-hoc fixed-replay reward/actor subsystem diagnostic; not training continuation, '
            'a promotion metric, or a formal verdict'
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
    result.add_argument('--seed', type=int, default=717171)
    return result


def main() -> None:
    """Run and print the compact actor-drift diagnostic."""
    payload = diagnose(parser().parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

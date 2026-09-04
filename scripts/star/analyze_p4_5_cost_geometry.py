#!/usr/bin/env python3
"""Run the preregistered P4.5 64-state by 256-action Q-cost geometry audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
from scripts.star.evaluate_pointgoal_checkpoints import (
    atomic_json,
    build_algorithm,
    checkpoint_payload,
)


PROTOCOL_VERSION = 'p4.5-unified-cost-geometry-v1'
PROTOCOL_SHA256 = '8abebc101021ec71a00fdf6d2c5580b5f97dada710e58de00936a75490e85c11'
DEFAULT_STATE_COUNT = 64
DEFAULT_ACTION_SAMPLES = 256
DEFAULT_AUDIT_SEED = 4_500_007
MODE_SPECS = {
    'A': {'learned_drift': False, 'wiener_noise': True, 'action_clipping': True},
    'B': {'learned_drift': True, 'wiener_noise': False, 'action_clipping': True},
    'C': {'learned_drift': True, 'wiener_noise': True, 'action_clipping': True},
}


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _q_margin(algorithm: Any, augmented: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Return mean distributional Q-cost divided by budget scale, minus remaining budget."""
    with torch.no_grad():
        cost = (
            algorithm._actor_critic.cost_critic.expected_cost(  # pylint: disable=protected-access
                augmented,
                action,
            ).mean(dim=0)
        )
    context_dim = algorithm._actor_critic.context_dim  # pylint: disable=protected-access
    budget = augmented[..., -context_dim]
    limit = float(algorithm._cfgs.lagrange_cfgs.cost_limit)  # pylint: disable=protected-access
    return cost / limit - budget


def _select_states(
    algorithm: Any,
    replay: dict[str, Any],
    state_count: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Select finalized resident transitions by stable absolute normalized Q-budget margin."""
    data = replay['data']
    eligible = torch.isfinite(data['empirical_cost_to_go']) & (data['episode_id'] >= 0)
    resident_indices = torch.arange(data['obs'].shape[0])[eligible]
    if resident_indices.numel() < state_count:
        raise ValueError(
            f'checkpoint has {resident_indices.numel()} finalized resident transitions, '
            f'fewer than required {state_count}',
        )
    device = algorithm._device  # pylint: disable=protected-access
    scores: list[tuple[float, int, float]] = []
    for start in range(0, resident_indices.numel(), 4096):
        indices = resident_indices[start : start + 4096]
        obs = data['obs'][indices].to(device)
        context = data['context'][indices].to(device)
        action = data['action'][indices].to(device)
        augmented = algorithm._augment(obs, context)  # pylint: disable=protected-access
        margin = _q_margin(algorithm, augmented, action).cpu()
        scores.extend(
            (abs(float(value)), int(index), float(value)) for value, index in zip(margin, indices)
        )
    scores.sort(key=lambda item: (item[0], item[1]))
    selected = scores[:state_count]
    indices = torch.tensor([item[1] for item in selected], dtype=torch.long)
    margins = torch.tensor([item[2] for item in selected], dtype=torch.float32)
    bandwidth = float(
        algorithm._cfgs.algo_cfgs.cost_geometry_bandwidth,
    )  # pylint: disable=protected-access
    return (
        {key: value[indices].to(device) for key, value in data.items()},
        {
            'eligible_finalized_resident_transitions': int(resident_indices.numel()),
            'selected_state_count': state_count,
            'selected_replay_indices': indices.tolist(),
            'selected_start_margins': margins.tolist(),
            'in_band_state_count': int((margins.abs() <= bandwidth).sum().item()),
            'in_band_definition': f'|Q_C/cost_limit-remaining_budget_ratio| <= {bandwidth}',
            'selection': (
                'ascending absolute normalized Q-cost budget margin with stable '
                'replay-index tie break'
            ),
        },
    )


def _covariance_metrics(actions: torch.Tensor, normal: torch.Tensor) -> dict[str, float]:
    """Return endpoint covariance resolved along one fixed Q-cost normal."""
    centered = actions - actions.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(actions.shape[0] - 1, 1)
    normal_variance = normal @ covariance @ normal
    total_variance = torch.trace(covariance)
    tangent_variance = (total_variance - normal_variance).clamp_min(0.0) / max(
        actions.shape[-1] - 1,
        1,
    )
    return {
        'endpoint_normal_variance': float(normal_variance.item()),
        'endpoint_tangent_variance': float(tangent_variance.item()),
        'endpoint_total_variance': float(total_variance.item()),
    }


def _integrate_mode(  # pylint: disable=too-many-locals
    algorithm: Any,
    augmented: torch.Tensor,
    fixed_normal: torch.Tensor,
    base_actions: torch.Tensor,
    common_noise: torch.Tensor,
    mode: str,
) -> dict[str, float]:
    """Integrate one learned-drift/noise mode with clipping enabled in every mode."""
    spec = MODE_SPECS[mode]
    actor = algorithm._actor_critic.actor  # pylint: disable=protected-access
    sample_count = base_actions.shape[0]
    repeated_obs = augmented.expand(sample_count, -1)
    action = base_actions.clone()
    start_margin = _q_margin(algorithm, repeated_obs, action)
    step_size = 1.0 / int(actor.num_steps)
    clipping_events = 0
    normal_velocities: list[torch.Tensor] = []
    normal_variances: list[torch.Tensor] = []
    tangent_variances: list[torch.Tensor] = []
    normal_energies: list[torch.Tensor] = []
    tangent_energies: list[torch.Tensor] = []
    band_samples: list[torch.Tensor] = []
    gradient_norms: list[torch.Tensor] = []
    final_proposal = action
    reference = algorithm._flow_reference(  # pylint: disable=protected-access
        repeated_obs,
        target=False,
        anisotropy_enabled=True,
        inward_drift_enabled=False,
    )
    for index in range(int(actor.num_steps)):
        start_time = torch.full(
            (sample_count, 1),
            index * step_size,
            device=action.device,
            dtype=action.dtype,
        )
        with torch.no_grad():
            start_velocity = (
                actor.velocity(repeated_obs, action, start_time)
                if spec['learned_drift']
                else torch.zeros_like(action)
            )
            midpoint = (action + 0.5 * step_size * start_velocity).clamp(-1.0, 1.0)
            midpoint_time = start_time + 0.5 * step_size
            learned_velocity = (
                actor.velocity(repeated_obs, midpoint, midpoint_time)
                if spec['learned_drift']
                else torch.zeros_like(action)
            )
        local = reference(midpoint)
        normal_velocity = (learned_velocity * local.normal).sum(dim=-1)
        tangent_squared = (
            learned_velocity.square().sum(dim=-1) - normal_velocity.square()
        ).clamp_min(0.0)
        normal_velocities.append(normal_velocity)
        normal_variances.append(local.normal_variance)
        tangent_variances.append(local.tangent_variance)
        normal_energies.append(0.5 * step_size * normal_velocity.square() / local.normal_variance)
        tangent_energies.append(0.5 * step_size * tangent_squared / local.tangent_variance)
        band_samples.append(
            (
                local.margin.abs()
                <= float(
                    algorithm._cfgs.algo_cfgs.cost_geometry_bandwidth,
                )  # pylint: disable=protected-access
            ).float(),
        )
        gradient_norms.append(
            local.gradient_norm
            * float(algorithm._cfgs.lagrange_cfgs.cost_limit),  # pylint: disable=protected-access
        )
        increment = (
            torch.matmul(
                local.noise_transform,
                common_noise[:, index, :].unsqueeze(-1),
            ).squeeze(-1)
            if spec['wiener_noise']
            else torch.zeros_like(action)
        )
        final_proposal = action + step_size * learned_velocity + math.sqrt(step_size) * increment
        clipping_events += int((final_proposal.abs() > 1.0).any(dim=-1).sum().item())
        action = final_proposal.clamp(-1.0, 1.0)

    velocity = torch.stack(normal_velocities, dim=-1)
    vn = torch.stack(normal_variances, dim=-1)
    vt = torch.stack(tangent_variances, dim=-1)
    end_margin = _q_margin(algorithm, repeated_obs, action)
    metrics = _covariance_metrics(action, fixed_normal)
    pre_clipping = _covariance_metrics(final_proposal, fixed_normal)
    metrics.update(
        {
            'pre_clipping_endpoint_normal_variance': pre_clipping['endpoint_normal_variance'],
            'pre_clipping_endpoint_tangent_variance': pre_clipping['endpoint_tangent_variance'],
            'normal_variance': float(vn.mean().item()),
            'tangent_variance': float(vt.mean().item()),
            'normal_to_tangent_variance_ratio': float((vn / vt).mean().item()),
            'dangerous_normal_velocity': float(velocity.clamp_min(0.0).mean().item()),
            'outward_velocity_ratio': float((velocity > 0.0).float().mean().item()),
            'mean_normal_velocity': float(velocity.mean().item()),
            'normal_energy': float(torch.stack(normal_energies, dim=-1).mean().item()),
            'tangent_energy': float(torch.stack(tangent_energies, dim=-1).mean().item()),
            'crossing_rate': float(
                ((start_margin <= 0.0) & (end_margin > 0.0)).float().mean().item(),
            ),
            'action_clipping_rate': clipping_events / (sample_count * int(actor.num_steps)),
            'final_endpoint_clipping_rate': float(
                (final_proposal.abs() > 1.0).any(dim=-1).float().mean().item(),
            ),
            'safety_band_sample_ratio': float(
                torch.stack(band_samples, dim=-1).mean().item(),
            ),
            'action_gradient_norm': float(
                torch.stack(gradient_norms, dim=-1).mean().item(),
            ),
            'mean_start_margin': float(start_margin.mean().item()),
            'mean_end_margin': float(end_margin.mean().item()),
            'mean_endpoint_normal_displacement': float(
                ((action - base_actions) * fixed_normal).sum(dim=-1).mean().item(),
            ),
        },
    )
    return metrics


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average one scalar metric mapping across equally weighted selected states."""
    if not rows:
        raise ValueError('cannot average an empty audit result')
    return {key: statistics.fmean(row[key] for row in rows) for key in rows[0]}


def _arm_from_flags(algo_cfgs: dict[str, Any]) -> str:
    """Validate the P4.5 method seal and map the sole ablation flag to C0/C1."""
    required = {
        'unified_cost_geometry_enabled': True,
        'stochastic_flow_enabled': True,
        'safety_anisotropy_enabled': True,
        'safety_inward_drift_enabled': False,
        'safety_outward_rectification_enabled': False,
        'candidate_probing_enabled': False,
        'dynamic_geometry_gate': False,
    }
    mismatches = {
        key: algo_cfgs.get(key)
        for key, expected in required.items()
        if algo_cfgs.get(key) != expected
    }
    if mismatches:
        raise ValueError(f'checkpoint violates P4.5 method flags: {mismatches}')
    return 'C1' if bool(algo_cfgs.get('directional_loss_enabled')) else 'C0'


def audit_checkpoint(  # pylint: disable=too-many-locals,too-many-statements
    checkpoint: Path,
    output: Path,
    work_dir: Path,
    device: str,
    state_count: int = DEFAULT_STATE_COUNT,
    sample_count: int = DEFAULT_ACTION_SAMPLES,
    audit_seed: int = DEFAULT_AUDIT_SEED,
) -> dict[str, Any]:
    """Audit one sealed P4.5 final full-replay checkpoint."""
    payload = checkpoint_payload(checkpoint)
    if payload.get('replay') is None:
        raise ValueError('P4.5 audit requires the final full checkpoint with replay')
    algo_cfgs = payload['extra']['resolved_config']['algo_cfgs']
    if algo_cfgs.get('safety_protocol_version') != PROTOCOL_VERSION:
        raise ValueError('checkpoint is not a P4.5 preregistered run')
    if algo_cfgs.get('safety_protocol_sha256') != PROTOCOL_SHA256:
        raise ValueError('checkpoint has the wrong P4.5 protocol seal')
    if int(payload['counters']['env_steps']) != 100_000:
        raise ValueError('P4.5 audit requires the final 100k checkpoint')
    arm = _arm_from_flags(algo_cfgs)
    algorithm = build_algorithm(payload, device, [12_027], work_dir)
    algorithm._actor_critic.eval()  # pylint: disable=protected-access
    try:
        batch, selection = _select_states(algorithm, payload['replay'], state_count)
        augmented = algorithm._augment(  # pylint: disable=protected-access
            batch['obs'],
            batch['context'],
        )
        replay_actions = batch['action']
        action_dim = int(replay_actions.shape[-1])
        mode_rows: dict[str, list[dict[str, float]]] = {mode: [] for mode in MODE_SPECS}
        per_state: list[dict[str, Any]] = []
        valid_normals = 0
        for state_index in range(state_count):
            state_obs = augmented[state_index : state_index + 1]
            replay_action = replay_actions[state_index : state_index + 1]
            local = algorithm._flow_reference(  # pylint: disable=protected-access
                state_obs,
                target=False,
                anisotropy_enabled=True,
                inward_drift_enabled=False,
            )(replay_action)
            normal = local.normal[0]
            valid_normal = normal.norm().item() > 0.5
            valid_normals += int(valid_normal)
            generator = torch.Generator(device='cpu')
            generator.manual_seed(audit_seed + state_index)
            base_actions = torch.empty(
                (sample_count, action_dim),
                dtype=replay_action.dtype,
            ).uniform_(
                -1.0,
                1.0,
                generator=generator,
            )
            common_noise = torch.randn(
                (
                    sample_count,
                    int(algorithm._actor_critic.actor.num_steps),
                    action_dim,
                ),  # pylint: disable=protected-access
                generator=generator,
                dtype=replay_action.dtype,
            )
            base_actions = base_actions.to(algorithm._device)  # pylint: disable=protected-access
            common_noise = common_noise.to(algorithm._device)  # pylint: disable=protected-access
            state_record: dict[str, Any] = {
                'ordinal': state_index,
                'replay_index': selection['selected_replay_indices'][state_index],
                'selected_replay_margin': selection['selected_start_margins'][state_index],
                'normal': [float(value) for value in normal.tolist()],
                'normal_gradient_norm': float(
                    (
                        local.gradient_norm
                        * float(
                            algorithm._cfgs.lagrange_cfgs.cost_limit,
                        )  # pylint: disable=protected-access
                    ).item(),
                ),
                'valid_normal': valid_normal,
                'modes': {},
            }
            for mode in MODE_SPECS:
                metrics = _integrate_mode(
                    algorithm,
                    state_obs,
                    normal,
                    base_actions,
                    common_noise,
                    mode,
                )
                mode_rows[mode].append(metrics)
                state_record['modes'][mode] = metrics
            per_state.append(state_record)

        result = {
            'stage': 'P4.5-final-cost-geometry-audit',
            'protocol_version': PROTOCOL_VERSION,
            'protocol_sha256': PROTOCOL_SHA256,
            'checkpoint': str(checkpoint),
            'checkpoint_sha256': sha256(checkpoint),
            'arm': arm,
            'training_seed': int(payload['extra']['seed']),
            'env_steps': int(payload['counters']['env_steps']),
            'device': device,
            'state_count': state_count,
            'action_samples_per_state': sample_count,
            'audit_seed': audit_seed,
            'flow_steps': int(
                algorithm._actor_critic.actor.num_steps,
            ),  # pylint: disable=protected-access
            'selection': selection,
            'valid_fixed_normal_state_count': valid_normals,
            'flags': {
                'unified_cost_geometry_enabled': bool(
                    algo_cfgs['unified_cost_geometry_enabled'],
                ),
                'directional_loss_enabled': bool(algo_cfgs['directional_loss_enabled']),
                'safety_anisotropy_enabled': bool(algo_cfgs['safety_anisotropy_enabled']),
                'probability_critic_policy_access': False,
                'outward_rectification_enabled': bool(
                    algo_cfgs['safety_outward_rectification_enabled'],
                ),
                'inward_drift_enabled': bool(algo_cfgs['safety_inward_drift_enabled']),
            },
            'modes': {
                mode: {'specification': MODE_SPECS[mode], 'metrics': _mean_rows(mode_rows[mode])}
                for mode in MODE_SPECS
            },
            'common_randomness': (
                'one uniform base-action tensor and one standard-normal Wiener tensor per state; '
                'all modes reuse the base tensor, A/C reuse Wiener noise, and B uses zeros'
            ),
            'clipping_definition': (
                'fraction of sample-step proposals with any action component outside [-1,1]'
            ),
            'crossing_definition': (
                'fraction with start normalized Q-budget margin <= 0 and endpoint margin > 0'
            ),
            'per_state': per_state,
        }
        atomic_json(output, result)
        return result
    finally:
        algorithm._logger.close()  # pylint: disable=protected-access
        algorithm._env.close()  # pylint: disable=protected-access


def parse_args() -> argparse.Namespace:
    """Parse one final-checkpoint audit invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--state-count', type=int, default=DEFAULT_STATE_COUNT)
    parser.add_argument('--action-samples', type=int, default=DEFAULT_ACTION_SAMPLES)
    parser.add_argument('--audit-seed', type=int, default=DEFAULT_AUDIT_SEED)
    args = parser.parse_args()
    if args.state_count <= 0 or args.action_samples <= 1:
        parser.error('state count must be positive and action samples must exceed one')
    return args


def main() -> None:
    """Audit one checkpoint and print its compact identity and mode summaries."""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result = audit_checkpoint(
        args.checkpoint,
        args.output,
        args.work_dir,
        args.device,
        args.state_count,
        args.action_samples,
        args.audit_seed,
    )
    print(
        json.dumps(
            {
                'output': str(args.output),
                'identity': f"{result['arm']}:{result['training_seed']}",
                'modes': result['modes'],
            },
        ),
    )


if __name__ == '__main__':
    main()

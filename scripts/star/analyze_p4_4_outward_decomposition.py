#!/usr/bin/env python3
"""Run the locked P4.4 four-mode decomposition on one final P4.3 checkpoint."""

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


PROTOCOL_VERSION = 'p4.4-outward-rectification-v1'
PROTOCOL_SHA256 = '005c42348a22f664448c688b3cecc16eba7cbbea55d7727537caeef5d3b69457'
PARENT_PROTOCOL_VERSION = 'p4.3-safety-anisotropy-v2'
DEFAULT_STATE_COUNT = 64
DEFAULT_ACTION_SAMPLES = 256
DEFAULT_AUDIT_SEED = 4_000_007
ALIGNMENT_EPSILON = 1e-8
MODE_SPECS = {
    'A': {'learned_drift': False, 'wiener_noise': True, 'action_clipping': False},
    'B': {'learned_drift': False, 'wiener_noise': True, 'action_clipping': True},
    'C': {'learned_drift': True, 'wiener_noise': False, 'action_clipping': True},
    'D': {'learned_drift': True, 'wiener_noise': True, 'action_clipping': True},
}


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _risk_ucb(algorithm: Any, augmented: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Evaluate the online Bellman probability UCB without mutating model state."""
    with torch.no_grad():
        head = algorithm._violation_critic().probabilities(  # pylint: disable=protected-access
            augmented,
            action,
        )
        return (
            head.mean(dim=0)
            + float(algorithm._cfgs.algo_cfgs.risk_ucb_beta)  # pylint: disable=protected-access
            * head.std(dim=0, unbiased=False)
        ).clamp(0.0, 1.0)


def _heldout_boundary_states(
    algorithm: Any,
    replay: dict[str, Any],
    state_count: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Select held-out transitions by ascending absolute probability-UCB margin."""
    data = replay['data']
    eligible = (
        data['is_heldout'].bool()
        & torch.isfinite(data['violation_label'])
        & (data['episode_id'] >= 0)
    )
    resident_indices = torch.arange(data['obs'].shape[0])[eligible]
    if resident_indices.numel() < state_count:
        raise ValueError(
            f'checkpoint has {resident_indices.numel()} finalized held-out transitions, '
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
        margin = _risk_ucb(algorithm, augmented, action).cpu() - float(
            algorithm._cfgs.algo_cfgs.risk_boundary,  # pylint: disable=protected-access
        )
        scores.extend(
            (abs(float(value)), int(index), float(value)) for value, index in zip(margin, indices)
        )
    scores.sort(key=lambda item: (item[0], item[1]))
    selected = scores[:state_count]
    selected_indices = torch.tensor([item[1] for item in selected], dtype=torch.long)
    selected_margin = torch.tensor([item[2] for item in selected], dtype=torch.float32)
    bandwidth = float(
        algorithm._cfgs.algo_cfgs.boundary_bandwidth,  # pylint: disable=protected-access
    )
    return (
        {key: value[selected_indices].to(device) for key, value in data.items()},
        {
            'eligible_heldout_transitions': int(resident_indices.numel()),
            'selected_state_count': state_count,
            'selected_replay_indices': selected_indices.tolist(),
            'selected_start_margins': selected_margin.tolist(),
            'in_band_state_count': int((selected_margin.abs() <= bandwidth).sum().item()),
            'in_band_definition': f'|F_UCB-delta_C| <= {bandwidth}',
            'selection': 'ascending |F_UCB-delta_C|, stable replay-index tie break',
        },
    )


def _covariance_metrics(
    actions: torch.Tensor,
    normal: torch.Tensor,
) -> dict[str, float]:
    """Return fixed-normal endpoint covariance components."""
    centered = actions - actions.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(actions.shape[0] - 1, 1)
    normal_variance = normal @ covariance @ normal
    total_variance = torch.trace(covariance)
    tangent_variance = (total_variance - normal_variance).clamp_min(0.0) / (
        actions.shape[-1] - 1
    )
    return {
        'normal_variance': float(normal_variance.item()),
        'tangent_variance': float(tangent_variance.item()),
        'total_empirical_variance': float(total_variance.item()),
    }


def _mode_metrics(
    actions: torch.Tensor,
    final_proposal: torch.Tensor,
    normal: torch.Tensor,
    boundary_action: torch.Tensor,
    end_margin: torch.Tensor,
    start_margin: float,
    normal_velocity: torch.Tensor,
    path_normal_valid: torch.Tensor,
    clipping_events: int,
) -> dict[str, float]:
    """Summarize one decomposition mode on one fixed state."""
    result = _covariance_metrics(actions, normal)
    pre_clipping = _covariance_metrics(final_proposal, normal)
    normal_displacement = ((actions - boundary_action) * normal).sum(dim=-1)
    result.update(
        {
            'pre_clipping_normal_variance': pre_clipping['normal_variance'],
            'pre_clipping_tangent_variance': pre_clipping['tangent_variance'],
            'pre_clipping_total_empirical_variance': pre_clipping[
                'total_empirical_variance'
            ],
            'crossing_rate': float((end_margin > start_margin).float().mean().item()),
            'mean_normal_displacement': float(normal_displacement.mean().item()),
            'clipping_rate': clipping_events / normal_velocity.numel(),
            'final_endpoint_clipping_rate': float(
                (final_proposal.abs() > 1.0).any(dim=-1).float().mean().item(),
            ),
            'outward_velocity_ratio': float((normal_velocity > 0.0).float().mean().item()),
            'outward_speed': float(normal_velocity.clamp_min(0.0).mean().item()),
            'mean_learned_normal_velocity': float(normal_velocity.mean().item()),
            'path_valid_normal_ratio': float(path_normal_valid.float().mean().item()),
            'mean_end_margin': float(end_margin.mean().item()),
        },
    )
    return result


def _integrate_mode(  # pylint: disable=too-many-locals
    algorithm: Any,
    augmented: torch.Tensor,
    boundary_action: torch.Tensor,
    normal: torch.Tensor,
    start_margin: float,
    sample_count: int,
    common_noise: torch.Tensor,
    mode: str,
) -> dict[str, float]:
    """Integrate one explicitly toggled drift/noise/clipping decomposition mode."""
    spec = MODE_SPECS[mode]
    actor = algorithm._actor_critic.actor  # pylint: disable=protected-access
    device = algorithm._device  # pylint: disable=protected-access
    action = boundary_action.expand(sample_count, -1).clone()
    repeated_obs = augmented.expand(sample_count, -1)
    step_size = 1.0 / int(actor.num_steps)
    clipping_events = 0
    normal_velocities: list[torch.Tensor] = []
    path_normal_valid: list[torch.Tensor] = []
    final_proposal = action
    reference = algorithm._flow_reference(  # pylint: disable=protected-access
        repeated_obs,
        target=False,
        anisotropy_enabled=algorithm._safety_anisotropy_enabled(),  # pylint: disable=protected-access
        inward_drift_enabled=False,
    )
    diagnostic_normal_reference = (
        reference
        if algorithm._safety_anisotropy_enabled()  # pylint: disable=protected-access
        else algorithm._flow_reference(  # pylint: disable=protected-access
            repeated_obs,
            target=False,
            anisotropy_enabled=True,
            inward_drift_enabled=False,
        )
    )
    for index in range(int(actor.num_steps)):
        start_time = torch.full(
            (sample_count, 1),
            index * step_size,
            device=device,
            dtype=action.dtype,
        )
        if spec['learned_drift']:
            with torch.no_grad():
                start_velocity = actor.velocity(repeated_obs, action, start_time)
            midpoint = action + 0.5 * step_size * start_velocity
            if spec['action_clipping']:
                midpoint = midpoint.clamp(-1.0, 1.0)
            midpoint_time = start_time + 0.5 * step_size
            with torch.no_grad():
                learned_velocity = actor.velocity(repeated_obs, midpoint, midpoint_time)
        else:
            midpoint = action
            learned_velocity = torch.zeros_like(action)
        local_reference = reference(midpoint)
        diagnostic_geometry = diagnostic_normal_reference(midpoint)
        local_normal_velocity = (learned_velocity * diagnostic_geometry.normal).sum(dim=-1)
        normal_velocities.append(local_normal_velocity)
        path_normal_valid.append(diagnostic_geometry.normal.norm(dim=-1) > 0.5)
        if spec['wiener_noise']:
            increment = torch.matmul(
                local_reference.noise_transform,
                common_noise[:, index, :].unsqueeze(-1),
            ).squeeze(-1)
        else:
            increment = torch.zeros_like(action)
        final_proposal = (
            action
            + step_size * learned_velocity
            + math.sqrt(step_size) * increment
        )
        clipping_events += int((final_proposal.abs() > 1.0).any(dim=-1).sum().item())
        action = final_proposal.clamp(-1.0, 1.0) if spec['action_clipping'] else final_proposal
    end_margin = _risk_ucb(algorithm, repeated_obs, action) - float(
        algorithm._cfgs.algo_cfgs.risk_boundary,  # pylint: disable=protected-access
    )
    return _mode_metrics(
        action,
        final_proposal,
        normal,
        boundary_action,
        end_margin,
        start_margin,
        torch.stack(normal_velocities, dim=-1),
        torch.stack(path_normal_valid, dim=-1),
        clipping_events,
    )


def _gradient_alignment(
    algorithm: Any,
    augmented: torch.Tensor,
    boundary_action: torch.Tensor,
    epsilon: float,
) -> dict[str, float | bool | None]:
    """Compare probability-UCB and mean-head expected-cost action gradients."""
    with torch.enable_grad():
        probability_action = boundary_action.detach().requires_grad_(True)
        head_probability = algorithm._violation_critic().probabilities(  # pylint: disable=protected-access
            augmented.detach(),
            probability_action,
        )
        probability_ucb = (
            head_probability.mean(dim=0)
            + float(algorithm._cfgs.algo_cfgs.risk_ucb_beta)  # pylint: disable=protected-access
            * head_probability.std(dim=0, unbiased=False)
        ).clamp(0.0, 1.0)
        probability_gradient = torch.autograd.grad(
            probability_ucb.sum(),
            probability_action,
        )[0][0]
        cost_action = boundary_action.detach().requires_grad_(True)
        expected_cost = algorithm._actor_critic.cost_critic.expected_cost(  # pylint: disable=protected-access
            augmented.detach(),
            cost_action,
        ).mean(dim=0)
        cost_gradient = torch.autograd.grad(expected_cost.sum(), cost_action)[0][0]
    probability_norm = float(probability_gradient.norm().item())
    cost_norm = float(cost_gradient.norm().item())
    finite = bool(
        torch.isfinite(probability_gradient).all() and torch.isfinite(cost_gradient).all(),
    )
    valid = finite and probability_norm > epsilon and cost_norm > epsilon
    cosine = None
    if valid:
        cosine = float(
            torch.dot(probability_gradient, cost_gradient).div(
                probability_gradient.norm() * cost_gradient.norm(),
            ).clamp(-1.0, 1.0).item(),
        )
    return {
        'probability_gradient_norm': probability_norm,
        'cost_gradient_norm': cost_norm,
        'valid': valid,
        'cosine': cosine,
        'opposite': bool(cosine is not None and cosine < 0.0),
    }


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average a non-empty collection of finite scalar mappings."""
    if not rows:
        raise ValueError('cannot average an empty metric collection')
    return {
        key: statistics.fmean(row[key] for row in rows)
        for key in rows[0]
    }


def _arm_from_flags(algo_cfgs: dict[str, Any]) -> str:
    """Map the immutable P4.3 safety flags to their preregistered arm."""
    flags = (
        bool(algo_cfgs['stochastic_flow_enabled']),
        bool(algo_cfgs['safety_anisotropy_enabled']),
        bool(algo_cfgs['safety_inward_drift_enabled']),
        bool(algo_cfgs['candidate_probing_enabled']),
    )
    arms = {
        (True, False, False, False): 'S0',
        (True, True, False, False): 'S1',
        (True, True, True, False): 'S2',
    }
    if flags not in arms:
        raise ValueError(f'checkpoint flags {flags} do not identify a P4.3 arm')
    return arms[flags]


def audit_checkpoint(  # pylint: disable=too-many-locals,too-many-statements
    checkpoint: Path,
    output: Path,
    work_dir: Path,
    device: str,
    state_count: int = DEFAULT_STATE_COUNT,
    sample_count: int = DEFAULT_ACTION_SAMPLES,
    audit_seed: int = DEFAULT_AUDIT_SEED,
) -> dict[str, Any]:
    """Audit one immutable final P4.3 checkpoint and write strict JSON evidence."""
    payload = checkpoint_payload(checkpoint)
    if payload.get('replay') is None:
        raise ValueError('P4.4 decomposition requires a full checkpoint with replay')
    resolved = payload['extra']['resolved_config']
    algo_cfgs = resolved['algo_cfgs']
    if algo_cfgs.get('safety_protocol_version') != PARENT_PROTOCOL_VERSION:
        raise ValueError('checkpoint is not a P4.3 preregistered run')
    if int(payload['counters']['env_steps']) != 50_000:
        raise ValueError('P4.4 pretraining decomposition requires the final 50k checkpoint')
    arm = _arm_from_flags(algo_cfgs)
    algorithm = build_algorithm(payload, device, [12_027], work_dir)
    algorithm._actor_critic.eval()  # pylint: disable=protected-access
    try:
        batch, selection = _heldout_boundary_states(algorithm, payload['replay'], state_count)
        augmented = algorithm._augment(  # pylint: disable=protected-access
            batch['obs'],
            batch['context'],
        )
        boundary_actions = batch['action']
        mode_rows: dict[str, list[dict[str, float]]] = {mode: [] for mode in MODE_SPECS}
        per_state: list[dict[str, Any]] = []
        alignment_rows: list[dict[str, float | bool | None]] = []
        valid_fixed_normals = 0
        action_dim = int(boundary_actions.shape[-1])
        for state_index in range(state_count):
            state_obs = augmented[state_index : state_index + 1]
            boundary_action = boundary_actions[state_index : state_index + 1]
            geometry = algorithm._flow_reference(  # pylint: disable=protected-access
                state_obs,
                target=False,
                anisotropy_enabled=True,
                inward_drift_enabled=False,
            )(boundary_action)
            normal = geometry.normal[0]
            valid_normal = bool(normal.norm().item() > 0.5)
            valid_fixed_normals += int(valid_normal)
            alignment = _gradient_alignment(
                algorithm,
                state_obs,
                boundary_action,
                ALIGNMENT_EPSILON,
            )
            alignment_rows.append(alignment)
            generator = torch.Generator(device='cpu')
            generator.manual_seed(audit_seed + state_index)
            common_noise = torch.randn(
                (sample_count, int(algorithm._actor_critic.actor.num_steps), action_dim),  # pylint: disable=protected-access
                generator=generator,
            ).to(device=algorithm._device, dtype=boundary_action.dtype)  # pylint: disable=protected-access
            state_record: dict[str, Any] = {
                'ordinal': state_index,
                'replay_index': selection['selected_replay_indices'][state_index],
                'start_margin': float(geometry.margin.item()),
                'normal': [float(value) for value in normal.tolist()],
                'valid_normal': valid_normal,
                'critic_alignment': alignment,
                'modes': {},
            }
            for mode in MODE_SPECS:
                metrics = _integrate_mode(
                    algorithm,
                    state_obs,
                    boundary_action,
                    normal,
                    float(geometry.margin.item()),
                    sample_count,
                    common_noise,
                    mode,
                )
                mode_rows[mode].append(metrics)
                state_record['modes'][mode] = metrics
            per_state.append(state_record)
        valid_alignments = [row for row in alignment_rows if bool(row['valid'])]
        cosines = [float(row['cosine']) for row in valid_alignments if row['cosine'] is not None]
        negative_count = sum(value < 0.0 for value in cosines)
        result = {
            'stage': 'P4.4-pretraining-outward-decomposition',
            'protocol_version': PROTOCOL_VERSION,
            'protocol_sha256': PROTOCOL_SHA256,
            'parent_protocol_version': PARENT_PROTOCOL_VERSION,
            'parent_protocol_sha256': str(algo_cfgs['safety_protocol_sha256']),
            'checkpoint': str(checkpoint),
            'checkpoint_sha256': sha256(checkpoint),
            'parent_arm': arm,
            'training_seed': int(payload['extra']['seed']),
            'env_steps': int(payload['counters']['env_steps']),
            'device': device,
            'state_count': state_count,
            'action_samples_per_state': sample_count,
            'audit_seed': audit_seed,
            'flow_steps': int(algorithm._actor_critic.actor.num_steps),  # pylint: disable=protected-access
            'selection': selection,
            'valid_fixed_normal_state_count': valid_fixed_normals,
            'flags': {
                'stochastic_flow_enabled': bool(algo_cfgs['stochastic_flow_enabled']),
                'safety_anisotropy_enabled': bool(algo_cfgs['safety_anisotropy_enabled']),
                'checkpoint_inward_drift_enabled': bool(
                    algo_cfgs['safety_inward_drift_enabled'],
                ),
                'decomposition_inward_drift_enabled': False,
                'candidate_probing_enabled': bool(algo_cfgs['candidate_probing_enabled']),
            },
            'modes': {
                mode: {'specification': MODE_SPECS[mode], 'metrics': _mean_rows(mode_rows[mode])}
                for mode in MODE_SPECS
            },
            'critic_alignment': {
                'epsilon': ALIGNMENT_EPSILON,
                'valid_state_count': len(cosines),
                'invalid_state_count': state_count - len(cosines),
                'negative_count': negative_count,
                'negative_fraction': negative_count / len(cosines) if cosines else None,
                'mean_cosine': statistics.fmean(cosines) if cosines else None,
                'median_cosine': statistics.median(cosines) if cosines else None,
            },
            'common_randomness': (
                'one standard-normal tensor per state shared by modes A, B, and D; '
                'mode C uses exact zeros'
            ),
            'clipping_definition': (
                'fraction of sample-step proposals with any component outside [-1,1]'
            ),
            'crossing_definition': 'Pr[F_UCB(s,A_1)-delta_C > start margin]',
            'heldout_evidence': payload['extra'].get('last_heldout_evidence', {}),
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
    """Audit one checkpoint and print its compact identity and mode-D result."""
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
                'identity': f"{result['parent_arm']}:{result['training_seed']}",
                'mode_D': result['modes']['D']['metrics'],
                'critic_alignment': result['critic_alignment'],
            },
        ),
    )


if __name__ == '__main__':
    main()

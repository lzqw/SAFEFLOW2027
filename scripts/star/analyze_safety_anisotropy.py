#!/usr/bin/env python3
"""Audit P4.3 directional policy variance on fixed held-out near-boundary states."""

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


PROTOCOL_VERSION = 'p4.3-safety-anisotropy-v2'
DEFAULT_STATE_COUNT = 64
DEFAULT_ACTION_SAMPLES = 256
DEFAULT_AUDIT_SEED = 3_000_007


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _risk_ucb(algorithm: Any, augmented: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Evaluate the unified probability UCB without changing any model state."""
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
    """Select the deterministic closest-margin held-out replay transitions."""
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
    batch_size = 4096
    for start in range(0, resident_indices.numel(), batch_size):
        indices = resident_indices[start : start + batch_size]
        obs = data['obs'][indices].to(device)
        context = data['context'][indices].to(device)
        action = data['action'][indices].to(device)
        augmented = algorithm._augment(obs, context)  # pylint: disable=protected-access
        risk = _risk_ucb(algorithm, augmented, action).cpu()
        margin = risk - float(
            algorithm._cfgs.algo_cfgs.risk_boundary,
        )  # pylint: disable=protected-access
        scores.extend(
            (abs(float(value)), int(index), float(value)) for value, index in zip(margin, indices)
        )
    scores.sort(key=lambda item: (item[0], item[1]))
    selected = scores[:state_count]
    selected_indices = torch.tensor([item[1] for item in selected], dtype=torch.long)
    selected_margin = torch.tensor([item[2] for item in selected], dtype=torch.float32)
    bandwidth = float(
        algorithm._cfgs.algo_cfgs.boundary_bandwidth,
    )  # pylint: disable=protected-access
    batch = {key: value[selected_indices].to(device) for key, value in data.items()}
    metadata = {
        'eligible_heldout_transitions': int(resident_indices.numel()),
        'selected_state_count': state_count,
        'selected_replay_indices': selected_indices.tolist(),
        'selected_start_margins': selected_margin.tolist(),
        'in_band_state_count': int((selected_margin.abs() <= bandwidth).sum().item()),
        'in_band_definition': f'|F_UCB-delta_C| <= {bandwidth}',
        'selection': 'ascending |F_UCB-delta_C|, stable replay-index tie break',
    }
    return batch, metadata


def _sample_policy(
    algorithm: Any,
    augmented: torch.Tensor,
    boundary_action: torch.Tensor,
    sample_count: int,
    generator: torch.Generator,
    *,
    conditional_wiener: bool,
) -> Any:
    """Sample either the complete policy or only its Wiener randomness."""
    action_dim = boundary_action.shape[-1]
    flow_steps = int(algorithm._cfgs.model_cfgs.flow.num_steps)  # pylint: disable=protected-access
    repeated_obs = augmented.expand(sample_count, -1)
    if conditional_wiener:
        base_action = boundary_action.expand(sample_count, -1)
    else:
        base_action = 2.0 * torch.rand((sample_count, action_dim), generator=generator) - 1.0
        base_action = base_action.to(algorithm._device)  # pylint: disable=protected-access
    flow_noise = torch.randn(
        (sample_count, flow_steps, action_dim),
        generator=generator,
    ).to(
        algorithm._device,
    )  # pylint: disable=protected-access
    with torch.no_grad():
        return algorithm._policy_flow(  # pylint: disable=protected-access
            repeated_obs,
            base_action=base_action,
            flow_noise=flow_noise,
        )


def _directional_metrics(
    actions: torch.Tensor,
    normal: torch.Tensor,
    boundary_action: torch.Tensor,
    end_margin: torch.Tensor,
    start_margin: float,
) -> dict[str, float]:
    """Return empirical covariance, entropy, movement, and boundary-crossing metrics."""
    centered = actions - actions.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(actions.shape[0] - 1, 1)
    normal_variance = normal @ covariance @ normal
    trace = torch.trace(covariance)
    tangent_variance = (trace - normal_variance).clamp_min(0.0) / (actions.shape[-1] - 1)
    constant = 2.0 * math.pi * math.e
    normal_entropy = 0.5 * math.log(constant * max(float(normal_variance.item()), 1e-12))
    tangent_entropy = (
        0.5
        * (actions.shape[-1] - 1)
        * math.log(constant * max(float(tangent_variance.item()), 1e-12))
    )
    normal_displacement = ((actions - boundary_action) * normal).sum(dim=-1)
    return {
        'normal_variance': float(normal_variance.item()),
        'tangent_variance': float(tangent_variance.item()),
        'trace': float(trace.item()),
        'normal_entropy': normal_entropy,
        'tangent_entropy': tangent_entropy,
        'total_directional_entropy': normal_entropy + tangent_entropy,
        'mean_normal_displacement': float(normal_displacement.mean().item()),
        'crossing_rate': float((end_margin > start_margin).float().mean().item()),
        'predicted_unsafe_rate': float((end_margin > 0.0).float().mean().item()),
        'mean_end_margin': float(end_margin.mean().item()),
        'clipping_rate': float((actions.abs() >= 1.0 - 1e-6).any(dim=-1).float().mean().item()),
    }


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average every finite scalar field over valid fixed states."""
    keys = rows[0]
    result = {}
    for key in keys:
        values = [row[key] for row in rows if math.isfinite(row[key])]
        result[key] = statistics.fmean(values) if values else float('nan')
    return result


def audit_checkpoint(  # pylint: disable=too-many-locals,too-many-statements
    checkpoint: Path,
    output: Path,
    work_dir: Path,
    device: str,
    state_count: int,
    sample_count: int,
    audit_seed: int,
) -> dict[str, Any]:
    """Run the preregistered held-out directional audit and write strict JSON."""
    payload = checkpoint_payload(checkpoint)
    if payload.get('replay') is None:
        raise ValueError('directional audit requires a full checkpoint with replay')
    resolved = payload['extra']['resolved_config']
    algo_cfgs = resolved['algo_cfgs']
    if algo_cfgs.get('safety_protocol_version') != PROTOCOL_VERSION:
        raise ValueError('checkpoint is not a P4.3 preregistered run')
    if int(payload['counters']['env_steps']) != 50_000:
        raise ValueError('P4.3 directional audit requires the final 50k checkpoint')
    algorithm = build_algorithm(payload, device, [12_027], work_dir)
    algorithm._actor_critic.eval()  # pylint: disable=protected-access
    try:
        batch, selection = _heldout_boundary_states(algorithm, payload['replay'], state_count)
        obs = batch['obs']
        context = batch['context']
        boundary_actions = batch['action']
        augmented = algorithm._augment(obs, context)  # pylint: disable=protected-access
        full_rows: list[dict[str, float]] = []
        conditional_rows: list[dict[str, float]] = []
        local_reference_rows: list[dict[str, float]] = []
        valid_normal_count = 0
        maximum_trace_error = 0.0
        per_state: list[dict[str, Any]] = []
        for state_index in range(state_count):
            state_obs = augmented[state_index : state_index + 1]
            boundary_action = boundary_actions[state_index : state_index + 1]
            # The critic-derived normal is measured for every arm, including isotropic S0.
            geometry = algorithm._flow_reference(  # pylint: disable=protected-access
                state_obs,
                target=False,
                anisotropy_enabled=True,
                inward_drift_enabled=False,
            )(boundary_action)
            normal = geometry.normal[0]
            valid_normal = float(normal.norm().item()) >= 0.5
            if valid_normal:
                valid_normal_count += 1
            actual_reference = algorithm._flow_reference(  # pylint: disable=protected-access
                state_obs,
                target=False,
            )(boundary_action)
            action_dim = int(boundary_action.shape[-1])
            normal_variance = float(actual_reference.normal_variance.item())
            tangent_variance = float(actual_reference.tangent_variance.item())
            trace = normal_variance + (action_dim - 1) * tangent_variance
            trace_budget = action_dim * float(algo_cfgs['flow_reference_variance'])
            trace_error = abs(trace - trace_budget) / trace_budget
            maximum_trace_error = max(maximum_trace_error, trace_error)
            constant = 2.0 * math.pi * math.e
            anisotropic_entropy = 0.5 * math.log(constant * normal_variance) + (
                0.5 * (action_dim - 1) * math.log(constant * tangent_variance)
            )
            isotropic_low_entropy = (
                0.5
                * action_dim
                * math.log(
                    constant * normal_variance,
                )
            )
            local_reference = {
                'reliability': float(actual_reference.reliability.item()),
                'boundary_reliability': float(actual_reference.boundary_reliability.item()),
                'gradient_reliability': float(actual_reference.gradient_reliability.item()),
                'ensemble_reliability': float(actual_reference.ensemble_reliability.item()),
                'calibration_reliability': float(
                    actual_reference.calibration_reliability.item(),
                ),
                'normal_variance': normal_variance,
                'tangent_variance': tangent_variance,
                'trace': trace,
                'trace_relative_error': trace_error,
                'entropy': anisotropic_entropy,
                'matched_normal_isotropic_low_entropy': isotropic_low_entropy,
                'entropy_advantage_vs_matched_normal_isotropic_low': (
                    anisotropic_entropy - isotropic_low_entropy
                ),
                'inward_drift': float(
                    (-(actual_reference.drift * actual_reference.normal).sum()).item(),
                ),
            }
            local_reference_rows.append(local_reference)
            state_record: dict[str, Any] = {
                'ordinal': state_index,
                'replay_index': selection['selected_replay_indices'][state_index],
                'start_margin': float(geometry.margin.item()),
                'normal': [float(value) for value in normal.tolist()],
                'valid_normal': valid_normal,
                'local_reference': local_reference,
            }
            if valid_normal:
                for mode_index, (name, conditional) in enumerate(
                    (('full_policy', False), ('conditional_wiener', True)),
                ):
                    generator = torch.Generator(device='cpu')
                    generator.manual_seed(
                        audit_seed + mode_index * 1_000_000 + state_index,
                    )
                    flow = _sample_policy(
                        algorithm,
                        state_obs,
                        boundary_action,
                        sample_count,
                        generator,
                        conditional_wiener=conditional,
                    )
                    repeated_obs = state_obs.expand(sample_count, -1)
                    end_margin = _risk_ucb(algorithm, repeated_obs, flow.action) - float(
                        algo_cfgs['risk_boundary'],
                    )
                    metrics = _directional_metrics(
                        flow.action,
                        normal,
                        boundary_action,
                        end_margin,
                        float(geometry.margin.item()),
                    )
                    if flow.reference is None:
                        raise RuntimeError('audited stochastic flow lost its reference trace')
                    path_trace = (
                        flow.reference.normal_variance
                        + (action_dim - 1) * flow.reference.tangent_variance
                    )
                    path_error = float(
                        ((path_trace - trace_budget).abs() / trace_budget).max().item(),
                    )
                    maximum_trace_error = max(maximum_trace_error, path_error)
                    metrics['maximum_path_trace_relative_error'] = path_error
                    (conditional_rows if conditional else full_rows).append(metrics)
                    state_record[name] = metrics
            per_state.append(state_record)
        if not full_rows or not conditional_rows:
            raise ValueError('no held-out state had a usable probability-critic action normal')
        flags = {
            'stochastic_flow_enabled': bool(algo_cfgs['stochastic_flow_enabled']),
            'safety_anisotropy_enabled': bool(algo_cfgs['safety_anisotropy_enabled']),
            'safety_inward_drift_enabled': bool(algo_cfgs['safety_inward_drift_enabled']),
            'candidate_probing_enabled': bool(algo_cfgs['candidate_probing_enabled']),
        }
        result = {
            'stage': 'P4.3-safety-anisotropy-heldout-audit',
            'protocol_version': PROTOCOL_VERSION,
            'protocol_sha256': algo_cfgs['safety_protocol_sha256'],
            'checkpoint': str(checkpoint),
            'checkpoint_sha256': sha256(checkpoint),
            'training_seed': int(payload['extra']['seed']),
            'env_steps': int(payload['counters']['env_steps']),
            'device': device,
            'flags': flags,
            'risk_tolerance_delta_c': float(algo_cfgs['risk_boundary']),
            'boundary_bandwidth': float(algo_cfgs['boundary_bandwidth']),
            'state_count': state_count,
            'action_samples_per_state': sample_count,
            'audit_seed': audit_seed,
            'selection': selection,
            'valid_normal_state_count': valid_normal_count,
            'maximum_reference_trace_relative_error': maximum_trace_error,
            'local_reference': _mean_rows(local_reference_rows),
            'full_policy': _mean_rows(full_rows),
            'conditional_wiener': _mean_rows(conditional_rows),
            'crossing_definition': 'Pr[m_C(s,A_1)-m_C(s,a_boundary)>0]',
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
    """Audit one checkpoint."""
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
    print(json.dumps({'output': str(args.output), 'summary': result['conditional_wiener']}))


if __name__ == '__main__':
    main()

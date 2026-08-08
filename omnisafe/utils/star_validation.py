# Copyright 2026 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pure numerical helpers for auditable STAR replay validation artifacts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch


def checkpoint_reconstruction_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Return saved STAR construction settings, with a legacy shape-based fallback."""
    saved = payload.get('extra', {}).get('resolved_config')
    sections = ('algo_cfgs', 'model_cfgs', 'dual_cfgs', 'lagrange_cfgs')
    if isinstance(saved, dict):
        return {key: deepcopy(saved[key]) for key in sections if key in saved}

    state = payload.get('model')
    if not isinstance(state, dict):
        raise ValueError('STAR checkpoint has no model state for configuration inference')

    def linear_shapes(prefix: str) -> list[torch.Size]:
        keys = [key for key in state if key.startswith(prefix) and key.endswith('.weight')]
        keys.sort(key=lambda key: int(key.rsplit('.', maxsplit=2)[-2]))
        return [state[key].shape for key in keys]

    actor_shapes = linear_shapes('actor.velocity_net.')
    reward_shapes = linear_shapes('reward_critic.critics.0.')
    cost_shapes = linear_shapes('cost_critic.heads.0.')
    if min(len(actor_shapes), len(reward_shapes), len(cost_shapes)) < 2:
        raise ValueError('STAR checkpoint network shapes are incomplete')
    head_indices = {
        int(key.split('.')[2])
        for key in state
        if key.startswith('cost_critic.heads.') and key.endswith('.weight')
    }
    return {
        'model_cfgs': {
            'actor': {'hidden_sizes': [int(shape[0]) for shape in actor_shapes[:-1]]},
            'critic': {
                'hidden_sizes': [int(shape[0]) for shape in reward_shapes[:-1]],
                'num_quantiles': int(cost_shapes[-1][0]),
                'ensemble_size': len(head_indices),
            },
        },
    }


def deterministic_indices(size: int, maximum: int) -> torch.Tensor:
    """Select the full replay or a deterministic time-stratified subset."""
    count = min(size, maximum)
    if count <= 0:
        raise ValueError('resident replay and maximum sample count must be non-empty')
    if count == size:
        return torch.arange(size, dtype=torch.long)
    indices = torch.linspace(0, size - 1, count, dtype=torch.float64).round().long()
    if indices.unique().numel() != count:
        raise RuntimeError('deterministic replay subsampling produced duplicate indices')
    return indices


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return average ranks with exact tie handling."""
    order = torch.argsort(values)
    sorted_values = values[order]
    _, inverse, counts = torch.unique_consecutive(
        sorted_values,
        return_inverse=True,
        return_counts=True,
    )
    ends = counts.cumsum(dim=0).to(torch.float64)
    starts = ends - counts
    group_ranks = 0.5 * (starts + ends - 1.0)
    ranks = torch.empty(values.shape[0], dtype=torch.float64, device=values.device)
    ranks[order] = group_ranks[inverse]
    return ranks


def spearman(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return tie-correct rank correlation or NaN for a degenerate sample."""
    if first.numel() < 2:
        return float('nan')
    first_rank = average_ranks(first.to(torch.float64))
    second_rank = average_ranks(second.to(torch.float64))
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = first_centered.norm() * second_centered.norm()
    if float(denominator.item()) == 0.0:
        return float('nan')
    return float((first_centered * second_centered).sum().div(denominator).item())


def calibration_rows(
    predicted: torch.Tensor,
    empirical: torch.Tensor,
    bins: int,
) -> list[dict[str, Any]]:
    """Aggregate reliability values in equal-width predicted-risk bins."""
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    assignments = torch.bucketize(predicted, boundaries[1:-1])
    rows = []
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum().item())
        if count:
            predicted_mean = float(predicted[selected].mean().item())
            empirical_mean = float(empirical[selected].mean().item())
            absolute_gap = abs(predicted_mean - empirical_mean)
        else:
            predicted_mean = None
            empirical_mean = None
            absolute_gap = None
        rows.append(
            {
                'bin': index,
                'lower': float(boundaries[index].item()),
                'upper': float(boundaries[index + 1].item()),
                'count': count,
                'predicted_risk_mean': predicted_mean,
                'empirical_risk_mean': empirical_mean,
                'absolute_gap': absolute_gap,
            },
        )
    return rows


def threshold_rows(
    audit: dict[str, torch.Tensor],
    current_low: float,
    current_safe: float,
    boundary: float,
) -> list[dict[str, Any]]:
    """Measure conservative inner-boundary coverage under a small fixed threshold grid."""
    pairs = {
        (0.02, 0.30),
        (0.02, 0.40),
        (0.02, 0.45),
        (0.02, 0.49),
        (0.05, 0.40),
        (0.05, 0.45),
        (0.05, 0.49),
        (0.10, 0.40),
        (0.10, 0.45),
        (0.10, 0.49),
        (current_low, current_safe),
    }
    rows = []
    for low, safe in sorted(pairs):
        if not 0.0 <= low < safe < boundary:
            continue
        selected = torch.logical_and(audit['risk_ucb'] >= low, audit['risk_ucb'] <= safe)
        count = int(selected.sum().item())
        rows.append(
            {
                'boundary_low': low,
                'boundary_safe': safe,
                'is_current': low == current_low and safe == current_safe,
                'sample_count': count,
                'sample_fraction': count / audit['risk_ucb'].numel(),
                'source_random_fraction': (
                    float(audit['source_random'][selected].float().mean().item()) if count else 0.0
                ),
                'immediate_violation_fraction': (
                    float(audit['is_failure'][selected].float().mean().item()) if count else 0.0
                ),
            },
        )
    return rows


def boundary_score_rows(
    audit: dict[str, torch.Tensor],
    current_threshold: float,
) -> list[dict[str, Any]]:
    """Measure replay coverage under fixed thresholds of the continuous Gaussian score."""
    if not 0.0 <= current_threshold <= 1.0:
        raise ValueError('current boundary-score threshold must lie in [0, 1]')
    thresholds = {0.1, 0.25, 0.5, 0.75, 0.9, current_threshold}
    rows = []
    for threshold in sorted(thresholds):
        selected = audit['boundary_score'] >= threshold
        count = int(selected.sum().item())
        selected_labels = audit['violation_label'][selected]
        finalized = torch.isfinite(selected_labels)
        rows.append(
            {
                'boundary_score_threshold': threshold,
                'is_current': threshold == current_threshold,
                'sample_count': count,
                'sample_fraction': count / audit['boundary_score'].numel(),
                'source_random_fraction': (
                    float(audit['source_random'][selected].float().mean().item()) if count else 0.0
                ),
                'immediate_violation_fraction': (
                    float(audit['is_failure'][selected].float().mean().item()) if count else 0.0
                ),
                'positive_label_fraction': (
                    float(selected_labels[finalized].mean().item())
                    if bool(finalized.any())
                    else None
                ),
            },
        )
    return rows

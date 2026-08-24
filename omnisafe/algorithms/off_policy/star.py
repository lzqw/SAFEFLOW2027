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
"""Safety-critical reference flow training loop for STAR route A."""

# The training loop intentionally keeps the nine-step update order in one auditable module.
# pylint: disable=too-many-lines

from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import deque
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from omnisafe.adapter.star_adapter import FAILURE_COST_ENVS, POINT_PREFIX, STARAdapter, STARContext
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.algorithms.off_policy.star_warmup import STARWarmup
from omnisafe.common.buffer.star_replay_buffer import (
    BOUNDARY,
    STARReplayBuffer,
    updates_for_transitions,
)
from omnisafe.common.logger import Logger
from omnisafe.common.normalizer import Normalizer
from omnisafe.common.star_checkpoint import load_star_checkpoint, save_star_checkpoint
from omnisafe.common.star_dual import EnergyDual, GlobalSafetyDual
from omnisafe.models.actor_critic.star_actor_q_critic import (
    STARActorQCritic,
    flac_sampled_reward_target,
)
from omnisafe.models.critic.distributional_cost_critic import (
    distributional_bellman_target,
    quantile_huber_loss,
)
from omnisafe.models.critic.violation_probability_critic import (
    ViolationProbabilityCritic,
    binary_auroc,
    binary_brier_score,
    binary_brier_skill,
    masked_bce_loss,
    violation_bellman_target,
)
from omnisafe.models.safety.distributional_backend import (
    SafetyGeometry,
    boundary_gate,
    calibration_reliability,
    compute_cost_stochastic_reference,
    compute_safety_geometry,
    compute_stochastic_reference,
    gaussian_directional_entropies,
    normalized_directional_loss,
    safety_aware_directional_loss,
)
from omnisafe.utils.star_validation import spearman as tie_correct_spearman


UPDATE_ORDER = (
    'distributional_cost_critic',
    'violation_probability_ensemble',
    'twin_reward_critics',
    'differentiable_flow',
    'terminal_reward_cost_utility',
    'late_path_safety_geometry',
    'flow_actor',
    'energy_dual',
    'true_episode_global_dual',
    'target_networks',
)

LOG_KEYS = (
    'Train/Epoch',
    'Train/WarmupStage',
    'Train/GeometryEnabled',
    'Train/CumulativeCost',
    'Train/ViolationRatio',
    'Metrics/EpRet',
    'Metrics/EpCost',
    'Metrics/EpLen',
    'Evaluation/Return',
    'Evaluation/Cost',
    'Evaluation/CostCVaR',
    'Evaluation/FailureCount',
    'Evaluation/FailureRate',
    'Evaluation/ViolationRatio',
    'Evaluation/CalibrationError',
    'Evaluation/BrierScore',
    'Evaluation/BrierSkill',
    'Evaluation/AUROC',
    'Evaluation/Spearman',
    'Evaluation/GoalsPerEpisode',
    'Evaluation/StallRatio',
    'Evaluation/Displacement',
    'Evaluation/StepDisplacement',
    'Evaluation/ActionNorm',
    'Evaluation/SuccessRate',
    'Evaluation/PathLength',
    'Evaluation/CostPerSuccessfulEpisode',
    'Evaluation/CostPerUnitPathLength',
    'Loss/CostCritic',
    'Loss/ViolationCritic',
    'Loss/ViolationBellman',
    'Loss/ViolationMCAuxiliary',
    'Loss/RewardCritic',
    'Loss/Actor',
    'Loss/Direction',
    'Direction/Numerator',
    'Direction/Denominator',
    'Loss/EnergyDual',
    'Critic/CostTD',
    'Critic/CostQMean',
    'Critic/CostQStd',
    'Critic/CostQAbsMax',
    'Critic/CostTargetQMean',
    'Critic/CostTargetQStd',
    'Critic/CostTargetQAbsMax',
    'Critic/CostToGoMAE',
    'Critic/RewardTD',
    'Critic/RewardQAbsMax',
    'Critic/RewardTargetQAbsMax',
    'Critic/RewardTargetSamplingStd',
    'Numerics/CostGradientNorm',
    'Numerics/CostActionGradientNorm',
    'Numerics/ViolationGradientNorm',
    'Numerics/RewardGradientNorm',
    'Numerics/ActorGradientNorm',
    'Critic/CalibrationError',
    'Critic/Spearman',
    'Critic/PredictedRisk',
    'Critic/EmpiricalRisk',
    'Critic/LabeledRatio',
    'Critic/BrierScore',
    'Critic/BrierSkill',
    'Critic/AUROC',
    'Critic/PositiveLabels',
    'Critic/NegativeLabels',
    'Critic/RiskP10',
    'Critic/RiskP50',
    'Critic/RiskP90',
    'Critic/RiskUCBMean',
    'Critic/RiskUCBBoundaryRate',
    'ReplayProbability/CalibrationError',
    'ReplayProbability/BrierScore',
    'ReplayProbability/BrierSkill',
    'ReplayProbability/AUROC',
    'ReplayProbability/BellmanError',
    'Heldout/CalibrationError',
    'Heldout/BrierScore',
    'Heldout/BrierSkill',
    'Heldout/AUROC',
    'Heldout/BellmanError',
    'Heldout/GradientNorm',
    'Heldout/EnsembleDisagreement',
    'Heldout/BoundarySamples',
    'Heldout/PositiveLabels',
    'Heldout/NegativeLabels',
    'Heldout/Episodes',
    'Heldout/Transitions',
    'Heldout/EvidenceSufficient',
    'Heldout/CalibrationReliability',
    'Geometry/GateActivation',
    'Geometry/CandidateGateActivation',
    'Geometry/ValidGradientRate',
    'Geometry/EnsembleDisagreement',
    'Geometry/GradientNorm',
    'Geometry/VN',
    'Geometry/VT',
    'Geometry/VNVT',
    'Geometry/RawJVP',
    'Geometry/NormalizedJVP',
    'Geometry/NormalVelocity',
    'Geometry/DangerousNormalVelocity',
    'Geometry/TangentEnergy',
    'Geometry/NormalEnergy',
    'Reference/Reliability',
    'Reference/BoundaryReliability',
    'Reference/GradientReliability',
    'Reference/EnsembleReliability',
    'Reference/CalibrationReliability',
    'Reference/RiskUCB',
    'Reference/Margin',
    'Reference/VN',
    'Reference/VT',
    'Reference/VNVT',
    'Reference/Trace',
    'Reference/TraceRelativeError',
    'Reference/HN',
    'Reference/HT',
    'Reference/NormalVelocity',
    'Reference/DangerousNormalVelocity',
    'Reference/ExecutedOutwardVelocityRatio',
    'Reference/RawNormalVelocity',
    'Reference/RawDangerousNormalVelocity',
    'Reference/RawOutwardVelocityRatio',
    'Reference/ClippingRate',
    'Reference/PreClippingVariance',
    'Reference/PostClippingVariance',
    'Reference/InwardDrift',
    'Reference/TangentEnergy',
    'Reference/NormalEnergy',
    'Reference/CrossingRate',
    'Reference/SafetyBandSampleRatio',
    'Reference/CostScore',
    'Reference/CostMargin',
    'Probing/CandidateRiskP10',
    'Probing/CandidateRiskP50',
    'Probing/CandidateRiskP90',
    'Probing/SelectedRisk',
    'Probing/SelectedDisagreement',
    'Probing/ActualCost',
    'Probing/ActualViolation',
    'Probing/EffectiveBoundarySamples',
    'Probing/TangentRatio',
    'Probing/DangerousNormalVelocity',
    'Actor/TerminalReward',
    'Actor/TerminalCost',
    'Actor/UpdateApplied',
    'Energy/Path',
    'Dual/Energy',
    'Dual/GlobalSafety',
    'Replay/FailureRatio',
    'Replay/BoundaryRatio',
    'Replay/ViolationRatio',
    'TotalEnvSteps',
    'TotalUpdates',
    'TotalRewardCriticUpdates',
    'Time/FPS',
    'Time/Total',
)


def cost_geometry_contract(env_id: str, cost_limit: float) -> tuple[str, float]:
    """Return the frozen geometry interpretation and its strictly positive score scale.

    Point tasks use cumulative cost divided by their episode budget.  The two SHAPO
    locomotion tasks have the prescribed formal budget ``d=0`` and a single binary cost on an
    unhealthy terminal transition, so their expected cost-to-go is the future-failure
    probability and is already on a unit scale.
    """
    if env_id.startswith(POINT_PREFIX) and cost_limit > 0.0:
        return 'remaining_cost_budget', float(cost_limit)
    if env_id in FAILURE_COST_ENVS and math.isclose(cost_limit, 0.0, abs_tol=1e-12):
        return 'future_failure_probability', 1.0
    raise ValueError(
        'unified cost geometry requires a positive Point-task budget or a zero-budget '
        'binary-failure Ant/Walker task',
    )


def best_safe_checkpoint_candidate_is_better(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
) -> bool:
    """Apply the preregistered feasible-return checkpoint ordering deterministically."""
    if incumbent is None:
        return True
    candidate_safe = bool(candidate['constraint_satisfied'])
    incumbent_safe = bool(incumbent['constraint_satisfied'])
    if candidate_safe != incumbent_safe:
        return candidate_safe
    if candidate_safe:
        candidate_rank = (
            -float(candidate['mean_return']),
            float(candidate['mean_cost']),
            int(candidate['milestone']),
        )
        incumbent_rank = (
            -float(incumbent['mean_return']),
            float(incumbent['mean_cost']),
            int(incumbent['milestone']),
        )
    else:
        candidate_rank = (
            float(candidate['mean_cost']),
            -float(candidate['mean_return']),
            int(candidate['milestone']),
        )
        incumbent_rank = (
            float(incumbent['mean_cost']),
            -float(incumbent['mean_return']),
            int(incumbent['milestone']),
        )
    return candidate_rank < incumbent_rank


MAX_LOG_KEYS = frozenset(
    {
        'Critic/RewardQAbsMax',
        'Critic/RewardTargetQAbsMax',
        'Critic/CostQAbsMax',
        'Critic/CostTargetQAbsMax',
    },
)
REQUIRED_FINITE_KEYS = frozenset(
    {
        'Loss/CostCritic',
        'Loss/ViolationCritic',
        'Loss/ViolationBellman',
        'Loss/RewardCritic',
        'Loss/Actor',
        'Loss/Direction',
        'Loss/EnergyDual',
        'Critic/CostTD',
        'Critic/CostQMean',
        'Critic/CostQStd',
        'Critic/CostQAbsMax',
        'Critic/CostTargetQMean',
        'Critic/CostTargetQStd',
        'Critic/CostTargetQAbsMax',
        'Critic/RewardTD',
        'Critic/RewardQAbsMax',
        'Critic/RewardTargetQAbsMax',
        'ReplayProbability/BellmanError',
        'Numerics/CostGradientNorm',
        'Numerics/CostActionGradientNorm',
        'Numerics/ViolationGradientNorm',
        'Numerics/RewardGradientNorm',
        'Numerics/ActorGradientNorm',
        'Actor/TerminalReward',
        'Actor/TerminalCost',
        'Energy/Path',
        'Dual/Energy',
        'Dual/GlobalSafety',
        'Reference/Reliability',
        'Reference/TraceRelativeError',
        'Reference/NormalVelocity',
        'Reference/DangerousNormalVelocity',
        'Reference/RawNormalVelocity',
        'Reference/RawDangerousNormalVelocity',
    },
)


@contextmanager
def _frozen_parameters(*modules: nn.Module) -> Iterator[None]:
    """Freeze module parameters while preserving first action derivatives."""
    parameters = [parameter for module in modules for parameter in module.parameters()]
    flags = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a small JSON artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        f'{json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)}\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    """Replace undefined floating diagnostics with JSON ``null`` recursively."""
    if isinstance(value, dict):
        return {key: _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    """Append a consistently headed CSV record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _spearman(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return tie-correct rank correlation without a SciPy dependency."""
    if second.numel() != first.numel():
        return float('nan')
    return tie_correct_spearman(first, second)


def _tail_mean(values: list[float], fraction: float = 0.2) -> float:
    """Return the upper-tail mean used as empirical cost CVaR."""
    if not values:
        return 0.0
    count = max(1, math.ceil(len(values) * fraction))
    return sum(sorted(values, reverse=True)[:count]) / count


def high_risk_geometry_summary(
    margin: torch.Tensor,
    metrics: dict[str, torch.Tensor],
    fraction: float = 0.2,
) -> dict[str, Any]:
    """Average geometry on the highest-margin fraction with stable tie handling."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError('high-risk geometry fraction must lie in (0, 1]')
    flattened_margin = margin.detach().flatten().cpu()
    flattened_metrics = {name: value.detach().flatten().cpu() for name, value in metrics.items()}
    for name, value in flattened_metrics.items():
        if value.numel() != flattened_margin.numel():
            raise ValueError(f'high-risk geometry metric {name!r} has incompatible size')
    finite_margin = torch.isfinite(flattened_margin)
    population_indices = torch.nonzero(finite_margin, as_tuple=False).flatten()
    if population_indices.numel() == 0:
        return {
            'fraction': fraction,
            'population_samples': 0,
            'selected_samples': 0,
            'ranking': 'descending_local_cost_margin_stable_ties',
            'minimum_selected_margin': float('nan'),
            'mean_selected_margin': float('nan'),
            'maximum_selected_margin': float('nan'),
            **{name: float('nan') for name in sorted(flattened_metrics)},
        }
    population_margin = flattened_margin[population_indices]
    count = max(1, math.ceil(int(population_indices.numel()) * fraction))
    relative_order = torch.argsort(population_margin, descending=True, stable=True)
    selected = population_indices[relative_order[:count]]
    selected_margin = flattened_margin[selected]
    result: dict[str, Any] = {
        'fraction': fraction,
        'population_samples': int(population_indices.numel()),
        'selected_samples': int(selected.numel()),
        'ranking': 'descending_local_cost_margin_stable_ties',
        'minimum_selected_margin': float(selected_margin.min().item()),
        'mean_selected_margin': float(selected_margin.mean().item()),
        'maximum_selected_margin': float(selected_margin.max().item()),
    }
    for name, values in sorted(flattened_metrics.items()):
        selected_values = values[selected]
        selected_values = selected_values[torch.isfinite(selected_values)]
        result[name] = (
            float(selected_values.mean().item()) if selected_values.numel() else float('nan')
        )
    return result


def equal_count_risk_bins(
    predicted: torch.Tensor,
    empirical: torch.Tensor,
    count: int = 5,
) -> list[dict[str, float | int]]:
    """Return stable equal-count predicted-risk bins with observed cost-to-go means."""
    predicted = predicted.detach().flatten().cpu()
    empirical = empirical.detach().flatten().cpu()
    if predicted.numel() != empirical.numel():
        raise ValueError('predicted and empirical cost tensors must have equal size')
    if count <= 0:
        raise ValueError('risk-bin count must be positive')
    finite = torch.isfinite(predicted) & torch.isfinite(empirical)
    predicted = predicted[finite]
    empirical = empirical[finite]
    if predicted.numel() == 0:
        return []
    # ``stable=True`` makes ties reproducible in their original evaluation order.
    order = torch.argsort(predicted, stable=True)
    bins: list[dict[str, float | int]] = []
    for index, selected in enumerate(torch.tensor_split(order, min(count, order.numel()))):
        if selected.numel() == 0:
            continue
        predicted_bin = predicted[selected]
        empirical_bin = empirical[selected]
        bins.append(
            {
                'bin': index,
                'count': int(selected.numel()),
                'predicted_min': float(predicted_bin.min().item()),
                'predicted_max': float(predicted_bin.max().item()),
                'predicted_mean': float(predicted_bin.mean().item()),
                'true_cost_to_go_mean': float(empirical_bin.mean().item()),
            },
        )
    return bins


def _truthy(value: Any) -> bool:
    """Interpret scalar or tensor environment flags without ambiguous tensor truth values."""
    return bool(torch.as_tensor(value).bool().any().item())


def normalized_actor_objective(
    energy: torch.Tensor,
    reward_value: torch.Tensor,
    cost_value: torch.Tensor,
    energy_multiplier: torch.Tensor | float,
    safety_multiplier: torch.Tensor | float,
) -> torch.Tensor:
    """Return the dual-scale-stable STAR actor objective."""
    safety = torch.as_tensor(safety_multiplier, device=reward_value.device)
    if bool((safety < 0).any()):
        raise ValueError('safety multiplier must be non-negative')
    numerator = (
        torch.as_tensor(energy_multiplier, device=reward_value.device) * energy
        - reward_value
        + safety * cost_value
    )
    return numerator / (1.0 + safety)


def select_boundary_candidates(  # pylint: disable=too-many-arguments
    risk_ucb: torch.Tensor,
    disagreement: torch.Tensor,
    reward_value: torch.Tensor,
    *,
    risk_boundary: float,
    disagreement_threshold: float,
    risk_tie_band: float,
) -> torch.Tensor:
    """Select the closest trusted inside-boundary candidate, breaking ties by reward value."""
    if risk_ucb.ndim != 2 or disagreement.shape != risk_ucb.shape:
        raise ValueError('candidate risk and disagreement must share [env, candidate] shape')
    if reward_value.shape != risk_ucb.shape:
        raise ValueError('candidate reward values must match candidate risks')
    if disagreement_threshold < 0.0 or risk_tie_band < 0.0:
        raise ValueError('candidate disagreement and tie thresholds cannot be negative')
    selections = []
    for env_index in range(risk_ucb.shape[0]):
        risk = risk_ucb[env_index]
        trusted = disagreement[env_index] <= disagreement_threshold
        safe = trusted & (risk <= risk_boundary)
        if bool(safe.any()):
            frontier = risk[safe].max()
            eligible = safe & (risk >= frontier - risk_tie_band)
        else:
            pool = trusted if bool(trusted.any()) else torch.ones_like(trusted)
            safest = risk[pool].min()
            eligible = pool & (risk <= safest + risk_tie_band)
        score = reward_value[env_index].masked_fill(~eligible, -torch.inf)
        selections.append(score.argmax())
    return torch.stack(selections)


def tangent_projection(
    noise: torch.Tensor,
    gradient: torch.Tensor,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project action noise onto the local risk tangent and return its retained-norm ratio."""
    if noise.shape != gradient.shape or noise.ndim != 2:
        raise ValueError('tangent projection expects matching rank-two action tensors')
    normal = gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(epsilon)
    tangent = noise - (noise * normal).sum(dim=-1, keepdim=True) * normal
    ratio = tangent.norm(dim=-1) / noise.norm(dim=-1).clamp_min(epsilon)
    return tangent, ratio


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class STAR(BaseAlgo):
    """First-order route-A STAR with distributional cost and reference-flow geometry."""

    _env_steps: int
    _update_count: int
    _actor_update_count: int
    _reward_critic_update_count: int
    _utd_carry: float
    _violation_count: int
    _cumulative_training_cost: float
    _last_update_order: tuple[str, ...]
    _training_seconds: float
    _completed_milestones: set[int]
    _completed_checkpoint_milestones: set[int]
    _next_log_step: int
    _last_log_step: int
    _last_log_time: float
    _run_env_steps: int
    _last_heldout_evidence: dict[str, float]
    _last_heldout_evidence_status: str
    _reward_td_diagnostics: list[float]
    _hard_stop_reason: str | None
    _pending_probe_metrics: dict[str, torch.Tensor] | None
    _best_safe_checkpoint_selection: dict[str, Any] | None

    def _init_env(self) -> None:
        """Create isolated STAR train/evaluation adapters."""
        self._env = STARAdapter(
            self._env_id,
            int(self._cfgs.train_cfgs.vector_env_nums),
            self._seed,
            self._cfgs,
        )
        obs_shape = self._env.observation_space.shape
        action_shape = self._env.action_space.shape
        if (
            obs_shape is None
            or action_shape is None
            or len(obs_shape) != 1
            or len(action_shape) != 1
        ):
            raise NotImplementedError('STAR currently requires flat Box observations and actions')
        self._obs_dim = obs_shape[0]
        self._act_dim = action_shape[0]
        self._num_envs = int(self._cfgs.train_cfgs.vector_env_nums)

    def _init_model(self) -> None:
        """Create action flow, twin reward critics, and quantile cost ensemble."""
        model = self._cfgs.model_cfgs
        algo = self._cfgs.algo_cfgs
        actor_lr = model.actor.lr
        assert actor_lr is not None
        legacy_compatibility = bool(getattr(algo, 'legacy_checkpoint_compatibility', False))
        context_dim = 1 if legacy_compatibility else 3 if self._bellman_probability_enabled() else 2
        self._actor_critic = STARActorQCritic(
            obs_dim=self._obs_dim,
            act_dim=self._act_dim,
            actor_hidden_sizes=list(model.actor.hidden_sizes),
            critic_hidden_sizes=list(model.critic.hidden_sizes),
            actor_lr=float(actor_lr),
            reward_critic_lr=float(model.critic.reward_lr),
            cost_critic_lr=float(model.critic.cost_lr),
            violation_critic_lr=float(model.critic.violation_lr),
            flow_steps=int(model.flow.num_steps),
            max_velocity=float(model.flow.max_velocity),
            num_quantiles=int(model.critic.num_quantiles),
            safety_ensemble_size=int(model.critic.ensemble_size),
            context_dim=context_dim,
            legacy_compatibility=legacy_compatibility,
        ).to(self._device)

    def _init(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
    ) -> None:
        """Initialize CPU replay, warmup evidence, duals, normalizer, and counters."""
        algo = self._cfgs.algo_cfgs
        self._buf = STARReplayBuffer(
            obs_dim=self._obs_dim,
            act_dim=self._act_dim,
            capacity=int(algo.size),
            batch_size=int(algo.batch_size),
            num_envs=self._num_envs,
            failure_fraction=float(algo.replay_failure_fraction),
            boundary_fraction=float(algo.replay_boundary_fraction),
            importance_beta=float(algo.replay_importance_beta),
            boundary_score_threshold=float(algo.boundary_score_threshold),
            context_dim=self._actor_critic.context_dim,
            heldout_episode_period=(
                int(algo.heldout_episode_period) if self._bellman_probability_enabled() else 0
            ),
        )
        self._obs_normalizer = Normalizer((self._obs_dim,), clip=10.0).to(self._device)
        dual = self._cfgs.dual_cfgs
        self._energy_dual = EnergyDual(
            float(dual.energy_initial),
            float(dual.energy_lr),
        ).to(self._device)
        self._global_dual = GlobalSafetyDual(
            initial_value=float(dual.global_initial),
            learning_rate=float(dual.global_lr),
            ema_rate=float(dual.global_ema_rate),
            cost_limit=float(self._cfgs.lagrange_cfgs.cost_limit),
            maximum_value=float(dual.global_max),
        )
        self._warmup = STARWarmup(
            random_steps=int(algo.random_steps),
            terminal_utility_steps=int(algo.terminal_utility_steps),
            geometry_min_steps=int(algo.geometry_min_steps),
            td_error_threshold=float(algo.w3_td_error_threshold),
            calibration_threshold=float(algo.w3_calibration_threshold),
            valid_gradient_threshold=float(algo.w3_valid_gradient_threshold),
            disagreement_threshold=float(algo.w3_disagreement_threshold),
            boundary_samples_threshold=int(algo.w3_boundary_samples_threshold),
            positive_labels_threshold=int(algo.w3_positive_labels_threshold),
            negative_labels_threshold=int(algo.w3_negative_labels_threshold),
            brier_skill_threshold=float(algo.w3_brier_skill_threshold),
            auroc_threshold=float(algo.w3_auroc_threshold),
            ema_rate=float(algo.warmup_ema_rate),
            dynamic_geometry_gate=self._dynamic_geometry_gate_enabled(),
            geometry_enable_passes=int(algo.geometry_enable_passes),
            geometry_disable_failures=int(algo.geometry_disable_failures),
        )
        self._env_steps = 0
        self._update_count = 0
        self._actor_update_count = 0
        self._reward_critic_update_count = 0
        self._utd_carry = 0.0
        self._violation_count = 0
        self._cumulative_training_cost = 0.0
        self._pending_episode_costs: deque[float] = deque()
        self._episode_history: deque[tuple[float, float, int]] = deque(maxlen=100)
        self._metric_sums: dict[str, float] = {}
        self._metric_counts: dict[str, int] = {}
        self._latest_values: dict[str, float] = {}
        self._last_heldout_evidence: dict[str, float] = {}
        self._last_heldout_evidence_status = 'insufficient_evidence'
        self._reward_td_diagnostics: list[float] = []
        self._hard_stop_reason = None
        self._pending_probe_metrics: dict[str, torch.Tensor] | None = None
        self._best_safe_checkpoint_selection = None
        self._last_update_order: tuple[str, ...] = ()
        self._training_seconds = 0.0
        self._run_env_steps = 0
        total_steps = int(self._cfgs.train_cfgs.total_steps)
        configured_milestones = [int(step) for step in algo.eval_milestones]
        required_milestones = {total_steps}
        if bool(getattr(algo, 'evaluate_initial_policy', True)):
            required_milestones.add(0)
        self._milestones = sorted({*required_milestones, *configured_milestones})
        self._milestones = [step for step in self._milestones if 0 <= step <= total_steps]
        self._completed_milestones: set[int] = set()
        configured_checkpoints = [int(step) for step in getattr(algo, 'checkpoint_milestones', [])]
        self._checkpoint_milestones = sorted(
            {
                0,
                total_steps,
                *self._milestones,
                *configured_checkpoints,
            },
        )
        self._checkpoint_milestones = [
            step for step in self._checkpoint_milestones if 0 <= step <= total_steps
        ]
        self._completed_checkpoint_milestones: set[int] = set()
        self._next_log_step = int(algo.steps_per_epoch)
        self._last_log_step = 0
        self._last_log_time = time.monotonic()

        eval_episodes_per_seed = int(getattr(algo, 'eval_episodes_per_seed', 1))
        if eval_episodes_per_seed <= 0 or eval_episodes_per_seed != float(
            getattr(algo, 'eval_episodes_per_seed', 1),
        ):
            raise ValueError('eval_episodes_per_seed must be a positive integer')
        if eval_episodes_per_seed > 1:
            configured_eval_seeds = [int(seed) for seed in algo.eval_seeds]
            expected_eval_episodes = len(configured_eval_seeds) * eval_episodes_per_seed
            if int(self._cfgs.train_cfgs.eval_episodes) != expected_eval_episodes:
                raise ValueError(
                    'multi-episode evaluation requires eval_episodes equal to '
                    'len(eval_seeds) * eval_episodes_per_seed',
                )

        actor_update_interval = int(algo.actor_update_interval)
        if actor_update_interval <= 0 or actor_update_interval != float(
            algo.actor_update_interval,
        ):
            raise ValueError('actor_update_interval must be a positive integer')
        reward_target_samples = int(algo.reward_target_samples)
        if reward_target_samples <= 0 or reward_target_samples != float(
            algo.reward_target_samples,
        ):
            raise ValueError('reward_target_samples must be a positive integer')
        reward_updates = int(algo.reward_critic_updates_per_update)
        if reward_updates <= 0 or reward_updates != float(
            algo.reward_critic_updates_per_update,
        ):
            raise ValueError('reward_critic_updates_per_update must be a positive integer')
        reward_target_polyak = self._reward_target_polyak()
        if not 0.0 < reward_target_polyak <= 1.0:
            raise ValueError('reward_target_polyak must lie in (0, 1]')
        recent_fraction = float(algo.reward_replay_recent_fraction)
        if not 0.0 <= recent_fraction <= 1.0:
            raise ValueError('reward_replay_recent_fraction must lie in [0, 1]')
        recent_window = int(algo.reward_replay_recent_window)
        if recent_window <= 0 or recent_window != float(algo.reward_replay_recent_window):
            raise ValueError('reward_replay_recent_window must be a positive integer')
        if recent_fraction > 0.0 and not bool(algo.separate_reward_replay):
            raise ValueError('recent reward replay requires separate_reward_replay')
        probability_target_polyak = self._probability_target_polyak()
        if not 0.0 < probability_target_polyak <= 1.0:
            raise ValueError('probability_target_polyak must lie in (0, 1]')
        auxiliary_weight = float(algo.probability_mc_aux_weight)
        if auxiliary_weight < 0.0:
            raise ValueError('probability_mc_aux_weight cannot be negative')
        heldout_period = int(algo.heldout_episode_period)
        if self._bellman_probability_enabled() and heldout_period < 2:
            raise ValueError('Bellman probability training requires heldout_episode_period >= 2')
        if int(algo.heldout_recent_episodes) <= 0 or int(algo.heldout_eval_batch_size) <= 0:
            raise ValueError('held-out evaluation sizes must be positive')
        heldout_minimums = (
            int(algo.heldout_min_positive_labels),
            int(algo.heldout_min_negative_labels),
        )
        if min(heldout_minimums) <= 0:
            raise ValueError('held-out positive/negative evidence minimums must be positive')
        if self._complete_heldout_evidence_enabled() and not self._bellman_probability_enabled():
            raise ValueError('complete held-out evidence requires Bellman probability training')
        if self._dynamic_geometry_gate_enabled() and (
            not self._bellman_probability_enabled() or not self._local_geometry_enabled()
        ):
            raise ValueError('dynamic geometry gating requires Bellman probability and geometry')
        candidate_count = int(algo.candidate_probing_count)
        if candidate_count <= 0 or candidate_count != float(algo.candidate_probing_count):
            raise ValueError('candidate_probing_count must be a positive integer')
        if self._candidate_probing_enabled() and not self._bellman_probability_enabled():
            raise ValueError('candidate probing requires Bellman probability training')
        if float(algo.candidate_disagreement_threshold) < 0.0:
            raise ValueError('candidate_disagreement_threshold cannot be negative')
        if float(algo.candidate_risk_tie_band) < 0.0:
            raise ValueError('candidate_risk_tie_band cannot be negative')
        if float(algo.candidate_tangent_scale) < 0.0:
            raise ValueError('candidate_tangent_scale cannot be negative')
        if self._safety_anisotropy_enabled() and not self._stochastic_flow_enabled():
            raise ValueError('safety anisotropy requires stochastic flow')
        random_anisotropy_seed = int(algo.random_anisotropy_seed)
        if random_anisotropy_seed < 0 or random_anisotropy_seed != float(
            algo.random_anisotropy_seed,
        ):
            raise ValueError('random anisotropy seed must be a non-negative integer')
        if self._random_anisotropy_enabled() and not (
            self._stochastic_flow_enabled()
            and self._safety_anisotropy_enabled()
            and self._unified_cost_geometry_enabled()
            and self._safety_aware_cost_geometry_enabled()
        ):
            raise ValueError(
                'random anisotropy control requires the complete safety-aware Q-cost geometry',
            )
        if self._safety_inward_drift_enabled() and not self._safety_anisotropy_enabled():
            raise ValueError('inward safety drift requires safety anisotropy')
        if self._safety_outward_rectification_enabled() and not (
            self._stochastic_flow_enabled()
            and self._safety_anisotropy_enabled()
            and self._bellman_probability_enabled()
        ):
            raise ValueError(
                'outward rectification requires stochastic anisotropic Bellman probability flow',
            )
        if self._safety_outward_rectification_enabled() and self._safety_inward_drift_enabled():
            raise ValueError('P4.4 outward rectification excludes safety inward drift')
        if self._safety_anisotropy_enabled() and not (
            self._bellman_probability_enabled() or self._unified_cost_geometry_enabled()
        ):
            raise ValueError('safety anisotropy requires probability or Q-cost geometry')
        if self._unified_cost_geometry_enabled() and not (
            self._stochastic_flow_enabled() and self._safety_anisotropy_enabled()
        ):
            raise ValueError('unified Q-cost geometry requires anisotropic stochastic flow')
        if self._unified_cost_geometry_enabled() and (
            self._safety_inward_drift_enabled() or self._safety_outward_rectification_enabled()
        ):
            raise ValueError('unified Q-cost geometry excludes inward drift and rectification')
        if self._directional_loss_enabled() and not self._unified_cost_geometry_enabled():
            raise ValueError('direction loss requires unified Q-cost geometry')
        if self._directional_loss_enabled() and float(algo.directional_loss_weight) <= 0.0:
            raise ValueError('direction loss weight must be positive when enabled')
        if float(algo.directional_loss_epsilon) <= 0.0:
            raise ValueError('direction loss epsilon must be positive')
        if self._safety_aware_cost_geometry_enabled() and not (
            self._unified_cost_geometry_enabled()
            and self._directional_loss_enabled()
            and self._safety_anisotropy_enabled()
        ):
            raise ValueError(
                'safety-aware cost geometry requires unified anisotropy and direction loss',
            )
        if self._safety_aware_cost_geometry_enabled() and not math.isclose(
            float(algo.directional_loss_weight),
            1.0,
        ):
            raise ValueError('P4.7 safety-aware direction weight is fixed at 1.0')
        if self._unified_cost_geometry_enabled():
            if float(algo.cost_geometry_bandwidth) <= 0.0:
                raise ValueError('Q-cost geometry requires a positive boundary bandwidth')
            self._cost_geometry_contract()
        if self._stochastic_flow_enabled() and (
            self._candidate_probing_enabled()
            or self._dynamic_geometry_gate_enabled()
            or self._local_geometry_enabled()
        ):
            raise ValueError(
                'stochastic safety flow is one policy and cannot use legacy geometry/probing',
            )
        if self._stochastic_flow_enabled():
            if not 0.0 < float(algo.risk_boundary) < 0.5:
                raise ValueError('stochastic-flow risk tolerance must lie in (0, 0.5)')
            if float(algo.flow_reference_variance) <= 0.0:
                raise ValueError('flow reference variance must be positive')
            if not 0.0 < float(algo.flow_normal_variance_ratio) <= 1.0:
                raise ValueError('normal/reference variance ratio must lie in (0, 1]')
            if (
                min(
                    float(algo.safety_gradient_scale),
                    float(algo.safety_ensemble_scale),
                    float(algo.safety_calibration_brier_score_scale),
                    float(algo.safety_drift_temperature),
                )
                <= 0.0
            ):
                raise ValueError('stochastic-flow reliability scales must be positive')
            if not 0.5 < float(algo.safety_calibration_auroc_full) <= 1.0:
                raise ValueError('full calibration AUROC must lie in (0.5, 1]')
            if float(algo.safety_drift_strength) < 0.0:
                raise ValueError('safety drift strength cannot be negative')
            if self._safety_anisotropy_enabled() and self._act_dim < 2:
                raise ValueError('safety anisotropy requires action dimension at least two')
        if (
            self._numerical_hard_stops_enabled()
            and min(
                float(algo.reward_td_absolute_hard_stop),
                float(algo.reward_q_absolute_hard_stop),
                float(algo.reward_td_final_to_minimum_ratio_hard_stop),
            )
            <= 0.0
        ):
            raise ValueError('P4.4 numerical hard-stop thresholds must be positive')
        if (
            self._engineering_hard_stops_enabled()
            and float(
                algo.q_absolute_overflow_threshold,
            )
            <= 0.0
        ):
            raise ValueError('engineering Q-overflow threshold must be positive')

        resume_from = str(algo.resume_from)
        if resume_from:
            self._restore_checkpoint(Path(resume_from))
        else:
            with torch.no_grad():
                initial_obs = self._env.current_obs
                if self._bellman_probability_enabled():
                    initial_obs = initial_obs[~self._buf.current_episode_is_heldout(self._device)]
                if initial_obs.shape[0]:
                    self._obs_normalizer.normalize(initial_obs)

    def _init_log(self) -> None:
        """Initialize tabular, JSON, evaluation, and checkpoint artifact directories."""
        self._logger = Logger(
            output_dir=str(self._cfgs.logger_cfgs.log_dir),
            exp_name=self._cfgs.exp_name,
            seed=int(self._cfgs.seed),
            use_tensorboard=bool(self._cfgs.logger_cfgs.use_tensorboard),
            use_wandb=bool(self._cfgs.logger_cfgs.use_wandb),
            config=self._cfgs,
        )
        for key in LOG_KEYS:
            self._logger.register_key(key)
        self._artifact_dir = Path(self._logger.log_dir) / 'artifacts'
        self._checkpoint_dir = Path(self._logger.log_dir) / 'checkpoints'
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        geometry_mode, geometry_scale = (
            self._cost_geometry_contract()
            if self._unified_cost_geometry_enabled()
            else ('disabled', None)
        )
        _atomic_json(
            self._artifact_dir / 'run_manifest.json',
            {
                'algorithm': 'STAR',
                'environment': self._env_id,
                'seed': self._seed,
                'route': 'A',
                'second_order': False,
                'latent_flow': False,
                'amp': False,
                'torch_compile': False,
                'update_order': list(UPDATE_ORDER),
                'eval_milestones': self._milestones,
                'checkpoint_milestones': self._checkpoint_milestones,
                'evaluation_protocol': {
                    'base_seeds': [int(seed) for seed in self._cfgs.algo_cfgs.eval_seeds],
                    'episodes_per_base_seed': int(
                        getattr(self._cfgs.algo_cfgs, 'eval_episodes_per_seed', 1),
                    ),
                    'seed_once_then_sequential_resets': True,
                    'flow_seed_offset': 1_000_003,
                },
                'checkpoint_protocol': {
                    'latest_full_recovery': bool(self._cfgs.algo_cfgs.checkpoint_replay),
                    'best_safe_return': (
                        'feasible_first;maximum_mean_return;minimum_mean_cost;earliest_milestone'
                    ),
                    'best_safe_return_pre_feasibility_fallback': (
                        'minimum_mean_cost;maximum_mean_return;earliest_milestone'
                    ),
                    'formal_cost_budget': float(self._cfgs.lagrange_cfgs.cost_limit),
                    'milestones': self._checkpoint_milestones,
                },
                'p4_2_backbone_ablation': {
                    'reward_flow_enabled': True,
                    'global_safety_constraint_enabled': self._global_safety_enabled(),
                    'local_safety_geometry_enabled': self._local_geometry_enabled(),
                },
                'p4_2_reward_backbone_repair': {
                    'target_flow_actor_for_reward_backup': (self._reward_target_actor_enabled()),
                    'actor_update_interval': int(self._cfgs.algo_cfgs.actor_update_interval),
                    'reward_target_samples': int(
                        self._cfgs.algo_cfgs.reward_target_samples,
                    ),
                    'reward_critic_updates_per_update': int(
                        self._cfgs.algo_cfgs.reward_critic_updates_per_update,
                    ),
                    'separate_reward_replay': bool(
                        self._cfgs.algo_cfgs.separate_reward_replay,
                    ),
                    'reward_replay_recent_fraction': float(
                        self._cfgs.algo_cfgs.reward_replay_recent_fraction,
                    ),
                    'reward_replay_recent_window': int(
                        self._cfgs.algo_cfgs.reward_replay_recent_window,
                    ),
                },
                'p4_2_reward_target_timescale': {
                    'reward_target_polyak': self._reward_target_polyak(),
                    'cost_target_polyak': float(self._cfgs.algo_cfgs.polyak),
                    'actor_target_polyak': float(self._cfgs.algo_cfgs.polyak),
                },
                'p4_2_probability_critic': {
                    'bellman_probability_enabled': self._bellman_probability_enabled(),
                    'target_actor': self._bellman_probability_enabled(),
                    'target_probability_critic': self._bellman_probability_enabled(),
                    'probability_target_polyak': self._probability_target_polyak(),
                    'mc_auxiliary_weight': float(
                        self._cfgs.algo_cfgs.probability_mc_aux_weight,
                    ),
                    'remaining_budget_clipped': not self._bellman_probability_enabled(),
                    'violation_so_far_context': self._bellman_probability_enabled(),
                    'episode_split': {
                        'heldout_period': (
                            int(self._cfgs.algo_cfgs.heldout_episode_period)
                            if self._bellman_probability_enabled()
                            else 0
                        ),
                        'optimizer_access_to_heldout': False,
                        'global_dual_access_to_heldout': False,
                        'observation_normalizer_access_to_heldout': False,
                        'complete_episode_reader': self._complete_heldout_evidence_enabled(),
                        'recent_complete_episodes': int(
                            self._cfgs.algo_cfgs.heldout_recent_episodes,
                        ),
                        'transition_limit': int(self._cfgs.algo_cfgs.heldout_eval_batch_size),
                    },
                },
                'p4_2_dynamic_geometry': {
                    'enabled': self._dynamic_geometry_gate_enabled(),
                    'enable_after_consecutive_passes': int(
                        self._cfgs.algo_cfgs.geometry_enable_passes,
                    ),
                    'disable_after_consecutive_failures': int(
                        self._cfgs.algo_cfgs.geometry_disable_failures,
                    ),
                    'gate_evidence_split': 'heldout_episode',
                },
                'p4_2_candidate_probing': {
                    'enabled': self._candidate_probing_enabled(),
                    'candidate_count': int(self._cfgs.algo_cfgs.candidate_probing_count),
                    'risk_selection': 'closest_trusted_safe_boundary_then_reward_q',
                    'tangent_requires_geometry_enabled': True,
                },
                'p4_3_stochastic_safety_reference': {
                    'protocol_version': str(
                        getattr(self._cfgs.algo_cfgs, 'safety_protocol_version', ''),
                    ),
                    'protocol_sha256': str(
                        getattr(self._cfgs.algo_cfgs, 'safety_protocol_sha256', ''),
                    ),
                    'stochastic_flow_enabled': self._stochastic_flow_enabled(),
                    'safety_anisotropy_enabled': self._safety_anisotropy_enabled(),
                    'safety_inward_drift_enabled': self._safety_inward_drift_enabled(),
                    'candidate_probing_enabled': self._candidate_probing_enabled(),
                    'risk_tolerance_delta_c': float(self._cfgs.algo_cfgs.risk_boundary),
                    'reference_variance': float(
                        getattr(self._cfgs.algo_cfgs, 'flow_reference_variance', 0.0),
                    ),
                    'normal_variance_ratio': float(
                        getattr(self._cfgs.algo_cfgs, 'flow_normal_variance_ratio', 1.0),
                    ),
                    'fixed_trace_budget': (
                        self._act_dim
                        * float(getattr(self._cfgs.algo_cfgs, 'flow_reference_variance', 0.0))
                    ),
                    'reliability': 'g_B*r_grad*r_ens*r_cal_continuous',
                    'actor_energy': '0.5*integral(u_theta^T*D_C^-1*u_theta)',
                    'inward_drift_excluded_from_actor_energy': True,
                },
                'p4_4_outward_rectification': {
                    'protocol_version': str(
                        getattr(self._cfgs.algo_cfgs, 'safety_protocol_version', ''),
                    ),
                    'protocol_sha256': str(
                        getattr(self._cfgs.algo_cfgs, 'safety_protocol_sha256', ''),
                    ),
                    'complete_heldout_evidence': self._complete_heldout_evidence_enabled(),
                    'minimum_positive_labels': int(
                        self._cfgs.algo_cfgs.heldout_min_positive_labels,
                    ),
                    'minimum_negative_labels': int(
                        self._cfgs.algo_cfgs.heldout_min_negative_labels,
                    ),
                    'outward_rectification_enabled': (self._safety_outward_rectification_enabled()),
                    'rectification': 'u_theta-q_C*max(0,n^T*u_theta)*n',
                    'actor_energy': 'executed_u_rect',
                    'numerical_hard_stops_enabled': self._numerical_hard_stops_enabled(),
                    'reward_td_absolute_hard_stop': float(
                        self._cfgs.algo_cfgs.reward_td_absolute_hard_stop,
                    ),
                    'reward_q_absolute_hard_stop': float(
                        self._cfgs.algo_cfgs.reward_q_absolute_hard_stop,
                    ),
                },
                'p4_5_unified_cost_geometry': {
                    'protocol_version': str(
                        getattr(self._cfgs.algo_cfgs, 'safety_protocol_version', ''),
                    ),
                    'protocol_sha256': str(
                        getattr(self._cfgs.algo_cfgs, 'safety_protocol_sha256', ''),
                    ),
                    'enabled': self._unified_cost_geometry_enabled(),
                    'geometry_mode': geometry_mode,
                    'formal_cost_budget': float(self._cfgs.lagrange_cfgs.cost_limit),
                    'cost_normalization_scale': geometry_scale,
                    'cost_score': 'mean_heads(mean_quantiles(Q_C))/cost_normalization_scale',
                    'local_margin': 'cost_score-geometry_budget_ratio',
                    'geometry_budget_ratio': (
                        'zero'
                        if geometry_mode == 'future_failure_probability'
                        else 'remaining_budget_ratio'
                    ),
                    'terminal_failure_cost': (
                        'one_on_unhealthy_termination;zero_on_time_limit'
                        if geometry_mode == 'future_failure_probability'
                        else 'not_applicable'
                    ),
                    'cost_geometry_bandwidth': float(
                        getattr(self._cfgs.algo_cfgs, 'cost_geometry_bandwidth', 0.0),
                    ),
                    'directional_loss_enabled': self._directional_loss_enabled(),
                    'directional_loss_weight': float(
                        getattr(self._cfgs.algo_cfgs, 'directional_loss_weight', 0.0),
                    ),
                    'directional_loss_velocity': 'raw_learned_midpoint_velocity',
                    'probability_critic_policy_access': False,
                    'outward_rectification_enabled': (self._safety_outward_rectification_enabled()),
                    'inward_drift_enabled': self._safety_inward_drift_enabled(),
                    'engineering_hard_stops_enabled': self._engineering_hard_stops_enabled(),
                    'q_absolute_overflow_threshold': float(
                        getattr(self._cfgs.algo_cfgs, 'q_absolute_overflow_threshold', 0.0),
                    ),
                    'cost_per_successful_episode': (
                        'sum_environment_cost/divide_by_goal_met_episode_count; undefined_if_zero'
                    ),
                    'cost_per_unit_path_length': 'sum_environment_cost/sum_xy_path_length',
                },
                'p4_7_safety_aware_geometry': {
                    'protocol_version': str(
                        getattr(self._cfgs.algo_cfgs, 'safety_protocol_version', ''),
                    ),
                    'protocol_sha256': str(
                        getattr(self._cfgs.algo_cfgs, 'safety_protocol_sha256', ''),
                    ),
                    'enabled': self._safety_aware_cost_geometry_enabled(),
                    'geometry_mode': geometry_mode,
                    'cost_normalization_scale': geometry_scale,
                    'one_sided_safety_weight': 'sigmoid(cost_margin/cost_geometry_bandwidth)',
                    'directional_loss_numerator': (
                        'mean(reliability*positive_part(normal^T*raw_velocity)^2)'
                    ),
                    'directional_loss_denominator': (
                        'stop_gradient(mean(squared_raw_velocity_norm))'
                    ),
                    'shared_direction_and_diffusion_reliability': True,
                    'high_risk_geometry_fraction': 0.2,
                    'high_risk_geometry_selection_gate': False,
                    'random_anisotropy_control_enabled': self._random_anisotropy_enabled(),
                    'random_anisotropy_seed': int(self._cfgs.algo_cfgs.random_anisotropy_seed),
                    'random_control_semantics': (
                        'deterministic_pseudorandom_covariance_axis_with_safety_normal_retained'
                    ),
                },
                'p4_1_correctness': {
                    'context': (
                        '[observation, remaining_budget_ratio, remaining_horizon_ratio, '
                        'violation_so_far]'
                        if self._bellman_probability_enabled()
                        else '[observation, remaining_budget_ratio, remaining_horizon_ratio]'
                    ),
                    'reward_done': 'terminated',
                    'cost_done': 'terminated_or_truncated',
                    'geometry_risk_source': 'independent_violation_probability_ensemble',
                    'boundary_score': 'gaussian_centered_at_probability_0.5',
                    'actor_objective': '(alpha_E*E-Q_R+lambda*Q_C)/(1+lambda)',
                },
            },
        )

    @property
    def update_count(self) -> int:
        """Return the number of gradient update iterations."""
        return self._update_count

    @property
    def env_steps(self) -> int:
        """Return the number of real environment transitions."""
        return self._env_steps

    @property
    def training_seconds(self) -> float:
        """Return measured end-to-end training wall time."""
        return self._training_seconds

    @property
    def run_env_steps(self) -> int:
        """Return transitions collected by this process segment, excluding restored history."""
        return self._run_env_steps

    @property
    def last_update_order(self) -> tuple[str, ...]:
        """Return the last complete update order for integration tests."""
        return self._last_update_order

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize without mutating running statistics during replay or evaluation."""
        count = int(self._obs_normalizer.state_dict()['_count'].item())
        if count <= 1:
            return obs
        normalized = (obs - self._obs_normalizer.mean) / self._obs_normalizer.std
        return normalized.clamp(-10.0, 10.0)

    def _augment(self, obs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Append the identical budget/horizon context consumed by every STAR network."""
        context = context.reshape(obs.shape[0], -1)
        if self._actor_critic.legacy_compatibility:
            context = context[:, :1]
        if context.shape[-1] != self._actor_critic.context_dim:
            raise ValueError(
                f'expected STAR context width {self._actor_critic.context_dim}, '
                f'got {context.shape[-1]}',
            )
        return torch.cat((self._normalize_obs(obs), context), dim=-1)

    def _violation_critic(self) -> ViolationProbabilityCritic:
        """Return the P4.1 probability head or reject legacy-only training."""
        critic = self._actor_critic.violation_critic
        if critic is None:
            raise RuntimeError('legacy checkpoint compatibility is evaluation-only')
        return critic

    def _target_violation_critic(self) -> ViolationProbabilityCritic:
        """Return the frozen probability target required by Bellman risk learning."""
        critic = self._actor_critic.target_violation_critic
        if critic is None:
            raise RuntimeError('legacy checkpoint compatibility has no probability target')
        return critic

    def _bellman_probability_enabled(self) -> bool:
        """Return whether probability learning uses the absorbing Bellman target."""
        return bool(getattr(self._cfgs.algo_cfgs, 'bellman_probability_enabled', False))

    def _dynamic_geometry_gate_enabled(self) -> bool:
        """Return whether held-out evidence controls a reversible geometry switch."""
        return bool(getattr(self._cfgs.algo_cfgs, 'dynamic_geometry_gate', False))

    def _candidate_probing_enabled(self) -> bool:
        """Return whether behavior actions use inside-boundary candidate probing."""
        return bool(getattr(self._cfgs.algo_cfgs, 'candidate_probing_enabled', False))

    def _stochastic_flow_enabled(self) -> bool:
        """Return whether the policy follows a Brownian rather than ODE reference."""
        return bool(getattr(self._cfgs.algo_cfgs, 'stochastic_flow_enabled', False))

    def _safety_anisotropy_enabled(self) -> bool:
        """Return whether continuous risk reliability shapes the flow covariance."""
        return bool(getattr(self._cfgs.algo_cfgs, 'safety_anisotropy_enabled', False))

    def _random_anisotropy_enabled(self) -> bool:
        """Return whether the mechanism control rotates covariance away from safety normals."""
        return bool(getattr(self._cfgs.algo_cfgs, 'random_anisotropy_enabled', False))

    def _safety_inward_drift_enabled(self) -> bool:
        """Return whether the stochastic reference also carries an inward drift."""
        return bool(getattr(self._cfgs.algo_cfgs, 'safety_inward_drift_enabled', False))

    def _safety_outward_rectification_enabled(self) -> bool:
        """Return whether learned outward normal velocity is reliability-rectified."""
        return bool(
            getattr(self._cfgs.algo_cfgs, 'safety_outward_rectification_enabled', False),
        )

    def _unified_cost_geometry_enabled(self) -> bool:
        """Return whether stochastic local geometry uses the cumulative cost critic."""
        return bool(getattr(self._cfgs.algo_cfgs, 'unified_cost_geometry_enabled', False))

    def _directional_loss_enabled(self) -> bool:
        """Return whether the normalized Q-cost outward-direction loss affects the actor."""
        return bool(getattr(self._cfgs.algo_cfgs, 'directional_loss_enabled', False))

    def _safety_aware_cost_geometry_enabled(self) -> bool:
        """Return whether one-sided reliability and stop-gradient normalization are active."""
        return bool(
            getattr(self._cfgs.algo_cfgs, 'safety_aware_cost_geometry_enabled', False),
        )

    def _cost_geometry_contract(self) -> tuple[str, float]:
        """Return the environment-specific interpretation of the frozen cost geometry."""
        return cost_geometry_contract(
            self._env_id,
            float(self._cfgs.lagrange_cfgs.cost_limit),
        )

    def _cost_geometry_scale(self) -> float:
        """Return the positive divisor used only to normalize the cost-critic score."""
        return self._cost_geometry_contract()[1]

    def _cost_geometry_budget_ratio(self, augmented_obs: torch.Tensor) -> torch.Tensor:
        """Return the local boundary target for budgeted-cost or terminal-failure geometry."""
        mode, _ = self._cost_geometry_contract()
        context_budget = augmented_obs[..., -self._actor_critic.context_dim]
        if mode == 'future_failure_probability':
            return torch.zeros_like(context_budget)
        return context_budget

    def _complete_heldout_evidence_enabled(self) -> bool:
        """Return whether balanced complete held-out evidence is mandatory for geometry."""
        return bool(getattr(self._cfgs.algo_cfgs, 'complete_heldout_evidence', False))

    def _numerical_hard_stops_enabled(self) -> bool:
        """Return whether the locked P4.4 numerical stop rules are active."""
        return bool(getattr(self._cfgs.algo_cfgs, 'numerical_hard_stops_enabled', False))

    def _engineering_hard_stops_enabled(self) -> bool:
        """Return whether P4.5 non-finite and gross-overflow stops are active."""
        return bool(getattr(self._cfgs.algo_cfgs, 'engineering_hard_stops_enabled', False))

    def _calibration_reliability(self) -> float:
        """Return the held-out-only continuous probability-critic reliability."""
        algo = self._cfgs.algo_cfgs
        if self._complete_heldout_evidence_enabled() and not bool(
            self._last_heldout_evidence.get('Heldout/EvidenceSufficient', 0.0),
        ):
            return 0.0
        return calibration_reliability(
            self._last_heldout_evidence.get('Heldout/AUROC'),
            self._last_heldout_evidence.get('Heldout/BrierScore'),
            auroc_full=float(algo.safety_calibration_auroc_full),
            brier_score_scale=float(algo.safety_calibration_brier_score_scale),
        )

    def _flow_reference(
        self,
        augmented_obs: torch.Tensor,
        *,
        target: bool,
        anisotropy_enabled: bool | None = None,
        inward_drift_enabled: bool | None = None,
    ) -> Any:
        """Close over observations and return the midpoint reference callback."""
        algo = self._cfgs.algo_cfgs
        anisotropy = (
            self._safety_anisotropy_enabled() if anisotropy_enabled is None else anisotropy_enabled
        )
        inward_drift = (
            self._safety_inward_drift_enabled()
            if inward_drift_enabled is None
            else inward_drift_enabled
        )
        if self._unified_cost_geometry_enabled():
            if inward_drift:
                raise ValueError('P4.5 Q-cost geometry excludes inward reference drift')
            cost_critic = (
                self._actor_critic.target_cost_critic if target else self._actor_critic.cost_critic
            )
            budget_ratio = self._cost_geometry_budget_ratio(augmented_obs)
            cost_scale = self._cost_geometry_scale()

            def cost_reference(action: torch.Tensor) -> Any:
                return compute_cost_stochastic_reference(
                    cost_critic,
                    augmented_obs,
                    action,
                    budget_ratio,
                    anisotropy_enabled=anisotropy,
                    safety_aware_geometry=self._safety_aware_cost_geometry_enabled(),
                    cost_normalization_scale=cost_scale,
                    boundary_bandwidth=float(algo.cost_geometry_bandwidth),
                    reference_variance=float(algo.flow_reference_variance),
                    normal_variance_ratio=float(algo.flow_normal_variance_ratio),
                    gradient_scale=float(algo.safety_gradient_scale),
                    ensemble_scale=float(algo.safety_ensemble_scale),
                    random_anisotropy_seed=(
                        int(algo.random_anisotropy_seed)
                        if self._random_anisotropy_enabled()
                        else None
                    ),
                )

            return cost_reference
        critic: ViolationProbabilityCritic | None = None
        if anisotropy:
            critic = self._target_violation_critic() if target else self._violation_critic()
        calibration = self._calibration_reliability() if anisotropy else 0.0

        def reference(action: torch.Tensor) -> Any:
            return compute_stochastic_reference(
                critic,
                augmented_obs,
                action,
                anisotropy_enabled=anisotropy,
                inward_drift_enabled=inward_drift,
                calibration=calibration,
                risk_tolerance=float(algo.risk_boundary),
                boundary_bandwidth=float(algo.boundary_bandwidth),
                risk_ucb_beta=float(algo.risk_ucb_beta),
                reference_variance=float(algo.flow_reference_variance),
                normal_variance_ratio=float(algo.flow_normal_variance_ratio),
                gradient_scale=float(algo.safety_gradient_scale),
                ensemble_scale=float(algo.safety_ensemble_scale),
                drift_strength=float(algo.safety_drift_strength),
                drift_temperature=float(algo.safety_drift_temperature),
            )

        return reference

    def _policy_flow(  # pylint: disable=too-many-arguments
        self,
        augmented_obs: torch.Tensor,
        *,
        target: bool = False,
        deterministic: bool = False,
        base_action: torch.Tensor | None = None,
        flow_noise: torch.Tensor | None = None,
        anisotropy_enabled: bool | None = None,
        inward_drift_enabled: bool | None = None,
        outward_rectification_enabled: bool | None = None,
    ) -> Any:
        """Sample the one policy used by behavior, targets, actor learning, and evaluation."""
        flow_function = self._actor_critic.target_flow if target else self._actor_critic.flow
        if not self._stochastic_flow_enabled():
            return flow_function(
                augmented_obs,
                deterministic=deterministic,
                base_action=base_action,
            )
        return flow_function(
            augmented_obs,
            deterministic=deterministic,
            base_action=base_action,
            reference=self._flow_reference(
                augmented_obs,
                target=target,
                anisotropy_enabled=anisotropy_enabled,
                inward_drift_enabled=inward_drift_enabled,
            ),
            flow_noise=flow_noise,
            outward_rectification_enabled=(
                self._safety_outward_rectification_enabled()
                if outward_rectification_enabled is None
                else outward_rectification_enabled
            ),
        )

    def _global_safety_enabled(self) -> bool:
        """Return whether the global cost constraint may affect training."""
        return bool(self._cfgs.algo_cfgs.enable_global_safety_constraint)

    def _local_geometry_enabled(self) -> bool:
        """Return whether evidence-ready local geometry may affect the flow."""
        return bool(self._cfgs.algo_cfgs.enable_local_safety_geometry)

    def _training_geometry_enabled(self) -> bool:
        """Return the actual local-geometry state, including the W3 evidence gate."""
        gate_open = (
            self._warmup.geometry_enabled
            if self._dynamic_geometry_gate_enabled()
            else self._warmup.stage == 'W3'
        )
        return self._local_geometry_enabled() and gate_open

    def _effective_global_safety_multiplier(self) -> float:
        """Return zero for B0 and the learned true-episode multiplier otherwise."""
        return self._global_dual.value if self._global_safety_enabled() else 0.0

    def _reward_target_actor_enabled(self) -> bool:
        """Return whether reward Bellman backups use the slowly moving target flow."""
        return bool(self._cfgs.algo_cfgs.use_target_flow_for_reward_backup)

    def _reward_target_polyak(self) -> float:
        """Return the independent reward-target step with legacy-config fallback."""
        return float(
            getattr(
                self._cfgs.algo_cfgs,
                'reward_target_polyak',
                self._cfgs.algo_cfgs.polyak,
            ),
        )

    def _probability_target_polyak(self) -> float:
        """Return the independent probability-target Polyak coefficient."""
        return float(
            getattr(
                self._cfgs.algo_cfgs,
                'probability_target_polyak',
                self._cfgs.algo_cfgs.polyak,
            ),
        )

    def _sample_reward_batch(self) -> dict[str, torch.Tensor]:
        """Draw the reward/actor replay mixture independently of safety strata."""
        algo = self._cfgs.algo_cfgs
        return self._buf.sample_uniform_recent_batch(
            self._device,
            recent_fraction=float(algo.reward_replay_recent_fraction),
            recent_window=int(algo.reward_replay_recent_window),
        )

    def _select_action(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Use bounded random actions in W0/W1 and learned flow actions in W2/W3."""
        context, _ = self._env.current_context()
        random_action = self._warmup.stage in {'W0', 'W1'}
        self._pending_probe_metrics = None
        if random_action:
            action = torch.empty(
                (self._num_envs, self._act_dim),
                dtype=torch.float32,
                device=self._device,
            ).uniform_(-1.0, 1.0)
        else:
            augmented = self._augment(self._env.current_obs, context)
            if self._candidate_probing_enabled() and self._update_count > 0:
                action, self._pending_probe_metrics = self._candidate_probe(augmented)
            else:
                with torch.no_grad():
                    action = self._policy_flow(augmented).action
        source_random = torch.full(
            (self._num_envs,),
            random_action,
            dtype=torch.bool,
            device=self._device,
        )
        return action, source_random

    def _candidate_probe(  # pylint: disable=too-many-locals
        self,
        augmented_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Choose one trusted action from K stochastic flow candidates per environment."""
        algo = self._cfgs.algo_cfgs
        candidate_count = int(algo.candidate_probing_count)
        expanded_obs = (
            augmented_obs.unsqueeze(1)
            .expand(-1, candidate_count, -1)
            .reshape(self._num_envs * candidate_count, -1)
        )
        with torch.no_grad():
            candidate_action = self._actor_critic.flow(expanded_obs).action
            head_probability = self._violation_critic().probabilities(
                expanded_obs,
                candidate_action,
            )
            head_probability = head_probability.reshape(
                head_probability.shape[0],
                self._num_envs,
                candidate_count,
            )
            mean_risk = head_probability.mean(dim=0)
            disagreement = head_probability.std(dim=0, unbiased=False)
            risk_ucb = (mean_risk + float(algo.risk_ucb_beta) * disagreement).clamp(0.0, 1.0)
            reward_first, reward_second = self._actor_critic.reward_critic(
                expanded_obs,
                candidate_action,
            )
            reward_value = torch.minimum(reward_first, reward_second).reshape(
                self._num_envs,
                candidate_count,
            )
            selected_index = select_boundary_candidates(
                risk_ucb,
                disagreement,
                reward_value,
                risk_boundary=float(algo.risk_boundary),
                disagreement_threshold=float(algo.candidate_disagreement_threshold),
                risk_tie_band=float(algo.candidate_risk_tie_band),
            )
            candidate_action = candidate_action.reshape(
                self._num_envs,
                candidate_count,
                self._act_dim,
            )
            env_index = torch.arange(self._num_envs, device=self._device)
            selected_action = candidate_action[env_index, selected_index]
            deterministic_action = self._actor_critic.flow(
                augmented_obs,
                deterministic=True,
            ).action
            trusted_boundary = (
                (risk_ucb <= float(algo.risk_boundary))
                & (disagreement <= float(algo.candidate_disagreement_threshold))
                & ((risk_ucb - float(algo.risk_boundary)).abs() <= float(algo.boundary_bandwidth))
            )
            effective_boundary = trusted_boundary.sum(dim=1).float()

        differentiable_action = selected_action.detach().requires_grad_(True)
        with _frozen_parameters(self._violation_critic()):
            selected_probability = self._violation_critic().probabilities(
                augmented_obs.detach(),
                differentiable_action,
            )
            selected_mean = selected_probability.mean(dim=0)
            risk_gradient = torch.autograd.grad(selected_mean.sum(), differentiable_action)[0]
        gradient_norm = risk_gradient.norm(dim=-1)
        normal = risk_gradient / gradient_norm.unsqueeze(-1).clamp_min(1e-8)
        dangerous_normal_velocity = (
            ((selected_action - deterministic_action) * normal).sum(dim=-1).clamp_min(0.0)
        )

        tangent_ratio = torch.zeros(self._num_envs, device=self._device)
        final_action = selected_action
        selected_disagreement = selected_probability.std(dim=0, unbiased=False).detach()
        trusted_gradient = (
            self._training_geometry_enabled()
            & (gradient_norm >= float(algo.minimum_gradient_norm))
            & (selected_disagreement <= float(algo.candidate_disagreement_threshold))
        )
        if bool(trusted_gradient.any()) and float(algo.candidate_tangent_scale) > 0.0:
            tangent, retained_ratio = tangent_projection(
                torch.randn_like(selected_action),
                risk_gradient.detach(),
            )
            tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            proposal = (selected_action + float(algo.candidate_tangent_scale) * tangent).clamp(
                -1.0,
                1.0,
            )
            with torch.no_grad():
                proposal_probability = self._violation_critic().probabilities(
                    augmented_obs,
                    proposal,
                )
                proposal_disagreement = proposal_probability.std(dim=0, unbiased=False)
                proposal_ucb = (
                    proposal_probability.mean(dim=0)
                    + float(algo.risk_ucb_beta) * proposal_disagreement
                ).clamp(0.0, 1.0)
                accepted = (
                    trusted_gradient
                    & (proposal_ucb <= float(algo.risk_boundary))
                    & (proposal_disagreement <= float(algo.candidate_disagreement_threshold))
                )
                final_action = torch.where(accepted.unsqueeze(-1), proposal, selected_action)
                tangent_ratio = torch.where(accepted, retained_ratio, tangent_ratio)

        with torch.no_grad():
            final_probability = self._violation_critic().probabilities(
                augmented_obs,
                final_action,
            )
            final_disagreement = final_probability.std(dim=0, unbiased=False)
            final_risk = (
                final_probability.mean(dim=0) + float(algo.risk_ucb_beta) * final_disagreement
            ).clamp(0.0, 1.0)
        metrics = {
            'candidate_risk_p10': torch.quantile(risk_ucb, 0.1, dim=1),
            'candidate_risk_p50': torch.quantile(risk_ucb, 0.5, dim=1),
            'candidate_risk_p90': torch.quantile(risk_ucb, 0.9, dim=1),
            'selected_risk': final_risk,
            'selected_disagreement': final_disagreement,
            'effective_boundary_samples': effective_boundary,
            'tangent_ratio': tangent_ratio,
            'dangerous_normal_velocity': dangerous_normal_velocity,
        }
        return final_action.detach(), metrics

    @torch.no_grad()
    def _boundary_scores(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Return the configured continuous replay-boundary score."""
        if self._update_count == 0:
            return torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
        algo = self._cfgs.algo_cfgs
        augmented = self._augment(obs, context)
        if self._unified_cost_geometry_enabled():
            head_cost = self._actor_critic.cost_critic.expected_cost(augmented, action)
            score = head_cost.mean(dim=0) / self._cost_geometry_scale()
            margin = score - self._cost_geometry_budget_ratio(augmented)
            return torch.exp(
                -0.5 * (margin / float(algo.cost_geometry_bandwidth)).square(),
            )
        if self._actor_critic.violation_critic is None:
            return torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
        risk_ucb = self._violation_critic().ucb(
            augmented,
            action,
            float(algo.risk_ucb_beta),
        )
        return boundary_gate(
            risk_ucb,
            float(algo.risk_boundary),
            float(algo.boundary_bandwidth),
        )

    def _advance_warmup(self) -> str:
        """Advance warmup with resident finalized-label evidence."""
        if self._unified_cost_geometry_enabled():
            # P4.5 keeps the shared random/critic-only W0--W2 schedule, but deliberately removes
            # probability-evidence access to the legacy W3 policy switch.  Stochastic Q-cost
            # geometry is configured directly and does not depend on ``geometry_enabled``.
            if self._env_steps >= self._warmup.random_steps and self._warmup.stage == 'W0':
                self._warmup.stage = 'W1'
            if self._env_steps >= self._warmup.terminal_utility_steps and self._warmup.stage in {
                'W0',
                'W1',
            }:
                self._warmup.stage = 'W2'
            return self._warmup.stage
        counts = self._buf.label_counts
        return self._warmup.advance(
            self._env_steps,
            self._buf.pool_sizes['boundary'],
            counts['positive'],
            counts['negative'],
        )

    def _collect_once(self) -> None:
        """Collect one vector transition and execute the explicit UTD entitlement."""
        stage_before_collection = self._warmup.stage
        episode_heldout = self._buf.current_episode_is_heldout(self._device)
        context, _ = self._env.current_context()
        action, source_random = self._select_action()
        boundary_score = self._boundary_scores(
            self._env.current_obs,
            action,
            context,
        )
        transition, completed = self._env.collect_step(action)
        failure = transition['cost'] > 0
        self._cumulative_training_cost += float(transition['cost'].sum().item())
        if self._pending_probe_metrics is not None:
            actual_violation = (
                transition['next_context'][:, -1]
                if self._bellman_probability_enabled()
                else failure.float()
            )
            probe_metrics = {
                'Probing/CandidateRiskP10': self._pending_probe_metrics[
                    'candidate_risk_p10'
                ].mean(),
                'Probing/CandidateRiskP50': self._pending_probe_metrics[
                    'candidate_risk_p50'
                ].mean(),
                'Probing/CandidateRiskP90': self._pending_probe_metrics[
                    'candidate_risk_p90'
                ].mean(),
                'Probing/SelectedRisk': self._pending_probe_metrics['selected_risk'].mean(),
                'Probing/SelectedDisagreement': self._pending_probe_metrics[
                    'selected_disagreement'
                ].mean(),
                'Probing/ActualCost': transition['cost'].mean(),
                'Probing/ActualViolation': actual_violation.float().mean(),
                'Probing/EffectiveBoundarySamples': self._pending_probe_metrics[
                    'effective_boundary_samples'
                ].mean(),
                'Probing/TangentRatio': self._pending_probe_metrics['tangent_ratio'].mean(),
                'Probing/DangerousNormalVelocity': self._pending_probe_metrics[
                    'dangerous_normal_velocity'
                ].mean(),
            }
            for key, probe_value in probe_metrics.items():
                self._accumulate(key, float(probe_value.item()))
        self._buf.store_batch(
            **transition,
            boundary_score=boundary_score,
            failure=failure,
            source_random=source_random,
        )
        self._violation_count += int(failure.sum().item())
        for episode in completed:
            self._episode_history.append(
                (episode.episode_return, episode.episode_cost, episode.episode_length),
            )
            if not bool(episode_heldout[episode.env_index].item()):
                self._pending_episode_costs.append(episode.episode_cost)
            episode_metrics = {
                'Metrics/EpRet': episode.episode_return,
                'Metrics/EpCost': episode.episode_cost,
                'Metrics/EpLen': float(episode.episode_length),
            }
            self._latest_values.update(episode_metrics)
            for key, episode_value in episode_metrics.items():
                self._accumulate(key, episode_value)

        with torch.no_grad():
            if self._bellman_probability_enabled():
                next_episode_heldout = self._buf.current_episode_is_heldout(self._device)
                training_obs = self._env.current_obs[~next_episode_heldout]
                if training_obs.shape[0]:
                    self._obs_normalizer.normalize(training_obs)
            else:
                self._obs_normalizer.normalize(self._env.current_obs)
        self._env_steps += self._num_envs
        self._advance_warmup()
        if stage_before_collection == 'W0':
            self._utd_carry = 0.0
            return
        updates, self._utd_carry = updates_for_transitions(
            float(self._cfgs.algo_cfgs.utd_ratio),
            self._num_envs,
            self._utd_carry,
        )
        for _ in range(updates):
            self._update()

    def _bootstrap_mask(
        self,
        ensemble_size: int,
        batch_size: int,
        eligible: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample non-empty independent bootstrap masks for ensemble diversity."""
        probability = float(self._cfgs.algo_cfgs.bootstrap_probability)
        mask = (torch.rand(ensemble_size, batch_size, device=self._device) < probability).float()
        candidates = torch.arange(batch_size, device=self._device)
        if eligible is not None:
            candidates = candidates[eligible.reshape(-1).bool()]
        for head in range(ensemble_size):
            if candidates.numel() and mask[head, candidates].sum() == 0:
                selected = candidates[int(torch.randint(candidates.numel(), ()).item())]
                mask[head, selected] = 1.0
        return mask

    def _update_cost_critic(  # pylint: disable=too-many-locals
        self,
        batch: dict[str, torch.Tensor],
        augmented_obs: torch.Tensor,
        next_augmented_obs: torch.Tensor,
    ) -> dict[str, float]:
        """Update the globally retained cost-distribution ensemble."""
        with torch.no_grad():
            next_action = self._policy_flow(next_augmented_obs).action
            next_quantiles = self._actor_critic.target_cost_critic(
                next_augmented_obs,
                next_action,
            )
            target = distributional_bellman_target(
                batch['cost'],
                batch['cost_done'],
                next_quantiles,
                float(self._cfgs.algo_cfgs.gamma_cost),
            )
        prediction = self._actor_critic.cost_critic(augmented_obs, batch['action'])
        mask = self._bootstrap_mask(prediction.shape[0], prediction.shape[1])
        loss = quantile_huber_loss(
            prediction,
            target,
            self._actor_critic.cost_critic.quantile_fractions,
            importance_weights=batch['importance_weights'],
            bootstrap_mask=mask,
        )
        optimizer = self._actor_critic.cost_critic_optimizer
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = clip_grad_norm_(
            self._actor_critic.cost_critic.parameters(),
            float(self._cfgs.algo_cfgs.max_grad_norm),
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError('non-finite cost-critic gradient norm')
        optimizer.step()

        with torch.no_grad():
            predicted_cost = prediction.mean(dim=(0, 2))
            target_cost = target.mean(dim=(0, 2))
            td_error = (predicted_cost - target_cost).abs()
            empirical = batch['empirical_cost_to_go']
            labeled = torch.isfinite(empirical)
            cost_to_go_mae = (
                float((predicted_cost[labeled] - empirical[labeled]).abs().mean().item())
                if bool(labeled.any())
                else float('nan')
            )
            correlation = (
                _spearman(predicted_cost[labeled], empirical[labeled])
                if int(labeled.sum().item()) >= 3
                else float('nan')
            )
        return {
            'Loss/CostCritic': float(loss.item()),
            'Critic/CostTD': float(td_error.mean().item()),
            'Critic/CostQMean': float(prediction.mean().item()),
            'Critic/CostQStd': float(prediction.std(unbiased=False).item()),
            'Critic/CostQAbsMax': float(prediction.abs().max().item()),
            'Critic/CostTargetQMean': float(target.mean().item()),
            'Critic/CostTargetQStd': float(target.std(unbiased=False).item()),
            'Critic/CostTargetQAbsMax': float(target.abs().max().item()),
            'Critic/CostToGoMAE': cost_to_go_mae,
            'Critic/Spearman': correlation,
            'Numerics/CostGradientNorm': float(gradient_norm.item()),
        }

    @torch.no_grad()
    def _violation_bellman_target(
        self,
        batch: dict[str, torch.Tensor],
        next_augmented_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Bootstrap event probability under the target actor and target probability critic."""
        next_action = self._policy_flow(next_augmented_obs, target=True).action
        next_probability = self._target_violation_critic().probabilities(
            next_augmented_obs,
            next_action,
        )
        next_violation = batch['next_context'][:, -1] > 0.5
        return violation_bellman_target(
            next_probability,
            next_violation,
            batch['cost_done'],
        )

    def _update_violation_critic(  # pylint: disable=too-many-locals
        self,
        batch: dict[str, torch.Tensor],
        augmented_obs: torch.Tensor,
        next_augmented_obs: torch.Tensor,
    ) -> dict[str, float]:
        """Fit Bellman event probabilities with MC labels used only as calibration auxiliary."""
        critic = self._violation_critic()
        optimizer = self._actor_critic.violation_critic_optimizer
        assert optimizer is not None
        labels = batch['violation_label']
        training = ~batch['is_heldout'].bool()
        labeled = torch.isfinite(labels) & training
        logits = critic(augmented_obs, batch['action'])
        mc_bootstrap = self._bootstrap_mask(logits.shape[0], logits.shape[1], eligible=labeled)
        mc_loss = masked_bce_loss(
            logits,
            labels.nan_to_num(0.0),
            labeled,
            importance_weights=batch['importance_weights'],
            bootstrap_mask=mc_bootstrap,
        )
        bellman_loss = logits.sum() * 0.0
        bellman_error = float('nan')
        if self._bellman_probability_enabled():
            bellman_target = self._violation_bellman_target(batch, next_augmented_obs)
            bellman_bootstrap = self._bootstrap_mask(
                logits.shape[0],
                logits.shape[1],
                eligible=training,
            )
            bellman_loss = masked_bce_loss(
                logits,
                bellman_target,
                training,
                importance_weights=batch['importance_weights'],
                bootstrap_mask=bellman_bootstrap,
            )
            with torch.no_grad():
                bellman_error = float(
                    (logits.sigmoid().mean(dim=0) - bellman_target.mean(dim=0))
                    .abs()[training]
                    .mean()
                    .item(),
                )
            loss = (
                bellman_loss
                + float(
                    self._cfgs.algo_cfgs.probability_mc_aux_weight,
                )
                * mc_loss
            )
            update_eligible = training
        else:
            loss = mc_loss
            update_eligible = labeled
        optimizer.zero_grad()
        gradient_norm = torch.zeros((), device=self._device)
        if bool(update_eligible.any()):
            loss.backward()
            gradient_norm = clip_grad_norm_(
                critic.parameters(),
                float(self._cfgs.algo_cfgs.max_grad_norm),
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError('non-finite violation-critic gradient norm')
            optimizer.step()

        with torch.no_grad():
            probabilities = logits.sigmoid()
            mean_risk = probabilities.mean(dim=0)
            risk_ucb = (
                mean_risk
                + float(self._cfgs.algo_cfgs.risk_ucb_beta)
                * probabilities.std(dim=0, unbiased=False)
            ).clamp(0.0, 1.0)
            scores = boundary_gate(
                risk_ucb,
                float(self._cfgs.algo_cfgs.risk_boundary),
                float(self._cfgs.algo_cfgs.boundary_bandwidth),
            )
            self._buf.relabel_boundaries(batch['indices'], scores)

        labeled_ratio = float(labeled.float().mean().item())
        calibration_error = float('nan')
        brier_score = float('nan')
        brier_skill = float('nan')
        auroc = float('nan')
        empirical_risk = float('nan')
        if bool(labeled.any()):
            selected_probability = mean_risk[labeled]
            selected_labels = labels[labeled]
            calibration_error = float(
                (selected_probability - selected_labels).abs().mean().item(),
            )
            brier_score = binary_brier_score(selected_probability, selected_labels)
            brier_skill = binary_brier_skill(selected_probability, selected_labels)
            auroc = binary_auroc(selected_probability, selected_labels)
            empirical_risk = float(selected_labels.mean().item())
        counts = self._buf.label_counts
        return {
            'Loss/ViolationCritic': float(loss.item()),
            'Loss/ViolationBellman': float(bellman_loss.item()),
            'Loss/ViolationMCAuxiliary': float(mc_loss.item()),
            'Critic/CalibrationError': calibration_error,
            'Critic/PredictedRisk': float(mean_risk.mean().item()),
            'Critic/EmpiricalRisk': empirical_risk,
            'Critic/LabeledRatio': labeled_ratio,
            'Critic/BrierScore': brier_score,
            'Critic/BrierSkill': brier_skill,
            'Critic/AUROC': auroc,
            'Critic/PositiveLabels': float(counts['positive']),
            'Critic/NegativeLabels': float(counts['negative']),
            'Critic/RiskP10': float(torch.quantile(mean_risk, 0.1).item()),
            'Critic/RiskP50': float(torch.quantile(mean_risk, 0.5).item()),
            'Critic/RiskP90': float(torch.quantile(mean_risk, 0.9).item()),
            'Critic/RiskUCBMean': float(risk_ucb.mean().item()),
            'Critic/RiskUCBBoundaryRate': float(
                (scores >= float(self._cfgs.algo_cfgs.boundary_score_threshold))
                .float()
                .mean()
                .item(),
            ),
            'ReplayProbability/CalibrationError': calibration_error,
            'ReplayProbability/BrierScore': brier_score,
            'ReplayProbability/BrierSkill': brier_skill,
            'ReplayProbability/AUROC': auroc,
            'ReplayProbability/BellmanError': bellman_error,
            'Numerics/ViolationGradientNorm': float(gradient_norm.item()),
        }

    def _next_path(
        self,
        next_augmented_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate target action and matching FLAC future reference energy."""
        flow_function = (
            self._actor_critic.target_flow
            if self._reward_target_actor_enabled()
            else self._actor_critic.flow
        )
        if self._stochastic_flow_enabled():
            with torch.no_grad():
                flow = self._policy_flow(
                    next_augmented_obs,
                    target=self._reward_target_actor_enabled(),
                )
                return flow.action, flow.kinetic_energy
        if not self._training_geometry_enabled():
            with torch.no_grad():
                flow = flow_function(next_augmented_obs)
                return flow.action, flow.kinetic_energy
        geometry_obs = next_augmented_obs.detach().requires_grad_(True)
        flow = flow_function(geometry_obs)
        geometry = self._geometry(geometry_obs, flow, enabled=True)
        return flow.action.detach(), geometry.per_sample_energy.detach()

    def _sampled_reward_backup(
        self,
        batch: dict[str, torch.Tensor],
        next_augmented_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the expected stochastic-flow backup and its Monte Carlo target noise."""
        samples = int(self._cfgs.algo_cfgs.reward_target_samples)
        batch_size = next_augmented_obs.shape[0]
        expanded_obs = (
            next_augmented_obs.unsqueeze(0)
            .expand(samples, -1, -1)
            .reshape(samples * batch_size, -1)
        )
        next_action, next_energy = self._next_path(expanded_obs)
        with torch.no_grad():
            next_first, next_second = self._actor_critic.target_reward_critic(
                expanded_obs,
                next_action,
            )
            first_samples = next_first.reshape(samples, batch_size)
            second_samples = next_second.reshape(samples, batch_size)
            energy_samples = next_energy.reshape(samples, batch_size)
            target = flac_sampled_reward_target(
                batch['reward'],
                batch['reward_done'],
                first_samples,
                second_samples,
                energy_samples,
                self._energy_dual.value.detach(),
                float(self._cfgs.algo_cfgs.gamma),
            )
            sampled_continuation = (
                torch.minimum(first_samples, second_samples)
                - self._energy_dual.value.detach() * energy_samples
            )
            sampled_targets = (
                batch['reward'].unsqueeze(0)
                + float(
                    self._cfgs.algo_cfgs.gamma,
                )
                * (1.0 - batch['reward_done']).unsqueeze(0)
                * sampled_continuation
            )
            sampling_std = sampled_targets.std(dim=0, unbiased=False).mean()
        return target, sampling_std

    def _update_reward_critics(  # pylint: disable=too-many-locals
        self,
        batch: dict[str, torch.Tensor],
        augmented_obs: torch.Tensor,
        next_augmented_obs: torch.Tensor,
    ) -> dict[str, float]:
        """Apply the FLAC target to both reward critics after the safety update."""
        target, sampling_std = self._sampled_reward_backup(batch, next_augmented_obs)
        first, second = self._actor_critic.reward_critic(augmented_obs, batch['action'])
        per_sample = (first - target).square() + (second - target).square()
        weights = batch['importance_weights']
        loss = (per_sample * weights).sum() / weights.sum().clamp_min(1.0)
        optimizer = self._actor_critic.reward_critic_optimizer
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = clip_grad_norm_(
            self._actor_critic.reward_critic.parameters(),
            float(self._cfgs.algo_cfgs.max_grad_norm),
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError('non-finite reward-critic gradient norm')
        optimizer.step()
        self._reward_critic_update_count += 1
        td_error = 0.5 * ((first - target).abs() + (second - target).abs())
        return {
            'Loss/RewardCritic': float(loss.item()),
            'Critic/RewardTD': float(td_error.mean().item()),
            'Critic/RewardQAbsMax': float(
                torch.maximum(first.abs().max(), second.abs().max()).item(),
            ),
            'Critic/RewardTargetQAbsMax': float(target.abs().max().item()),
            'Critic/RewardTargetSamplingStd': float(sampling_std.item()),
            'Numerics/RewardGradientNorm': float(gradient_norm.item()),
        }

    def _geometry(
        self,
        augmented_obs: torch.Tensor,
        flow: Any,
        enabled: bool,
    ) -> SafetyGeometry:
        """Apply the configured first-order distributional reference geometry."""
        algo = self._cfgs.algo_cfgs
        return compute_safety_geometry(
            self._violation_critic(),
            augmented_obs,
            flow,
            enabled=enabled,
            late_time=float(algo.late_flow_time),
            boundary=float(algo.risk_boundary),
            boundary_bandwidth=float(algo.boundary_bandwidth),
            minimum_gradient_norm=float(algo.minimum_gradient_norm),
            minimum_consistency=float(algo.minimum_consistency),
            tangent_variance=float(algo.tangent_variance),
            normal_contraction=float(algo.normal_contraction),
            reference_strength=float(algo.reference_strength),
            reference_temperature=float(algo.reference_temperature),
        )

    def _stochastic_reference_metrics(
        self,
        flow: Any,
        augmented_obs: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Summarize the covariance, reliability, drift, and residual energy actually used."""
        reference = flow.reference
        if reference is None:
            raise RuntimeError('stochastic policy flow did not retain its reference trace')
        normal_entropy, tangent_entropy = gaussian_directional_entropies(
            reference.normal_variance,
            reference.tangent_variance,
            self._act_dim,
        )
        trace = reference.normal_variance + (self._act_dim - 1) * reference.tangent_variance
        trace_budget = self._act_dim * float(self._cfgs.algo_cfgs.flow_reference_variance)
        normal_velocity = (flow.velocities * reference.normals).sum(dim=-1)
        raw_normal_velocity = (flow.raw_velocities * reference.normals).sum(dim=-1)
        inward_drift = -(reference.drifts * reference.normals).sum(dim=-1)
        pre_clipping_variance = flow.proposals[:, -1].var(dim=0, unbiased=False).sum()
        post_clipping_variance = flow.action.var(dim=0, unbiased=False).sum()
        metrics = {
            'Reference/Reliability': float(reference.reliability.mean().item()),
            'Reference/BoundaryReliability': float(
                reference.boundary_reliability.mean().item(),
            ),
            'Reference/GradientReliability': float(
                reference.gradient_reliability.mean().item(),
            ),
            'Reference/EnsembleReliability': float(
                reference.ensemble_reliability.mean().item(),
            ),
            'Reference/CalibrationReliability': float(
                reference.calibration_reliability.mean().item(),
            ),
            'Reference/RiskUCB': float(reference.risk_ucb.mean().item()),
            'Reference/Margin': float(reference.margins.mean().item()),
            'Reference/VN': float(reference.normal_variance.mean().item()),
            'Reference/VT': float(reference.tangent_variance.mean().item()),
            'Reference/VNVT': float(
                (reference.normal_variance / reference.tangent_variance).mean().item(),
            ),
            'Reference/Trace': float(trace.mean().item()),
            'Reference/TraceRelativeError': float(
                ((trace - trace_budget).abs() / max(trace_budget, 1e-12)).max().item(),
            ),
            'Reference/HN': float(normal_entropy.mean().item()),
            'Reference/HT': float(tangent_entropy.mean().item()),
            'Reference/NormalVelocity': float(normal_velocity.mean().item()),
            'Reference/DangerousNormalVelocity': float(
                normal_velocity.clamp_min(0.0).mean().item(),
            ),
            'Reference/ExecutedOutwardVelocityRatio': float(
                (normal_velocity > 0.0).float().mean().item(),
            ),
            'Reference/RawNormalVelocity': float(raw_normal_velocity.mean().item()),
            'Reference/RawDangerousNormalVelocity': float(
                raw_normal_velocity.clamp_min(0.0).mean().item(),
            ),
            'Reference/RawOutwardVelocityRatio': float(
                (raw_normal_velocity > 0.0).float().mean().item(),
            ),
            'Reference/ClippingRate': float(
                (flow.proposals.abs() > 1.0).any(dim=-1).float().mean().item(),
            ),
            'Reference/PreClippingVariance': float(pre_clipping_variance.item()),
            'Reference/PostClippingVariance': float(post_clipping_variance.item()),
            'Reference/InwardDrift': float(inward_drift.mean().item()),
            'Reference/TangentEnergy': float(reference.tangent_energy.mean().item()),
            'Reference/NormalEnergy': float(reference.normal_energy.mean().item()),
            'Reference/CrossingRate': 0.0,
            'Reference/SafetyBandSampleRatio': 0.0,
            'Reference/CostScore': 0.0,
            'Reference/CostMargin': 0.0,
            'Numerics/CostActionGradientNorm': 0.0,
        }
        if self._unified_cost_geometry_enabled():
            bandwidth = float(self._cfgs.algo_cfgs.cost_geometry_bandwidth)
            metrics.update(
                {
                    'Reference/SafetyBandSampleRatio': float(
                        (reference.margins.abs() <= bandwidth).float().mean().item(),
                    ),
                    'Reference/CostScore': float(reference.risk_ucb.mean().item()),
                    'Reference/CostMargin': float(reference.margins.mean().item()),
                    'Numerics/CostActionGradientNorm': float(
                        (reference.gradient_norm * self._cost_geometry_scale()).mean().item(),
                    ),
                },
            )
            if augmented_obs is not None:
                with torch.no_grad():
                    start_cost = self._actor_critic.cost_critic.expected_cost(
                        augmented_obs,
                        flow.points[:, 0],
                    ).mean(dim=0)
                    end_cost = self._actor_critic.cost_critic.expected_cost(
                        augmented_obs,
                        flow.action,
                    ).mean(dim=0)
                    budget_ratio = self._cost_geometry_budget_ratio(augmented_obs)
                    start_margin = start_cost / self._cost_geometry_scale() - budget_ratio
                    end_margin = end_cost / self._cost_geometry_scale() - budget_ratio
                metrics['Reference/CrossingRate'] = float(
                    ((start_margin <= 0.0) & (end_margin > 0.0)).float().mean().item(),
                )
        return metrics

    def _update_actor(  # pylint: disable=too-many-locals
        self,
        augmented_obs: torch.Tensor,
    ) -> dict[str, float]:
        """Regenerate the one flow policy, terminal utility, actor, then energy dual."""
        if self._stochastic_flow_enabled():
            flow = self._policy_flow(augmented_obs)
            per_sample_energy = flow.kinetic_energy
            path_energy = per_sample_energy.mean()
            reference_metrics = self._stochastic_reference_metrics(flow, augmented_obs)
            geometry = None
        else:
            flow = self._actor_critic.flow(augmented_obs)
            geometry = self._geometry(
                augmented_obs,
                flow,
                enabled=self._training_geometry_enabled(),
            )
            per_sample_energy = geometry.per_sample_energy
            path_energy = geometry.energy
            reference_metrics = {}
        direction_loss = torch.zeros((), device=augmented_obs.device)
        direction_numerator = torch.zeros_like(direction_loss)
        direction_denominator = torch.zeros_like(direction_loss)
        if self._unified_cost_geometry_enabled():
            if flow.reference is None:
                raise RuntimeError('unified Q-cost geometry requires a retained reference trace')
            directional_loss_function = (
                safety_aware_directional_loss
                if self._safety_aware_cost_geometry_enabled()
                else normalized_directional_loss
            )
            direction_loss, direction_numerator, direction_denominator = directional_loss_function(
                flow.raw_velocities,
                flow.reference.normals,
                flow.reference.reliability,
                epsilon=float(self._cfgs.algo_cfgs.directional_loss_epsilon),
            )
        with _frozen_parameters(
            self._actor_critic.reward_critic,
            self._actor_critic.cost_critic,
        ):
            reward_first, reward_second = self._actor_critic.reward_critic(
                augmented_obs,
                flow.action,
            )
            terminal_reward = torch.minimum(reward_first, reward_second)
            if self._global_safety_enabled():
                head_cost = self._actor_critic.cost_critic.expected_cost(
                    augmented_obs,
                    flow.action,
                )
                terminal_cost = head_cost.mean(dim=0) + float(
                    self._cfgs.algo_cfgs.cost_ucb_beta,
                ) * head_cost.std(dim=0, unbiased=False)
            else:
                terminal_cost = torch.zeros_like(terminal_reward)
            base_actor_loss = normalized_actor_objective(
                per_sample_energy,
                terminal_reward,
                terminal_cost,
                self._energy_dual.value.detach(),
                self._effective_global_safety_multiplier(),
            ).mean()
            actor_loss = base_actor_loss
            if self._directional_loss_enabled():
                actor_loss = (
                    actor_loss
                    + float(
                        self._cfgs.algo_cfgs.directional_loss_weight,
                    )
                    * direction_loss
                )
            optimizer = self._actor_critic.actor_optimizer
            optimizer.zero_grad()
            actor_loss.backward()
            gradient_norm = clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                float(self._cfgs.algo_cfgs.max_grad_norm),
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError('non-finite actor gradient norm')
            optimizer.step()

        target_energy = float(self._cfgs.dual_cfgs.target_energy_per_action) * self._act_dim
        dual_loss = self._energy_dual.update(path_energy, target_energy)
        common_metrics = {
            'Loss/Actor': float(actor_loss.item()),
            'Loss/Direction': float(direction_loss.item()),
            'Direction/Numerator': float(direction_numerator.item()),
            'Direction/Denominator': float(direction_denominator.item()),
            'Loss/EnergyDual': float(dual_loss.item()),
            'Actor/TerminalReward': float(terminal_reward.mean().item()),
            'Actor/TerminalCost': float(terminal_cost.mean().item()),
            'Energy/Path': float(path_energy.item()),
            'Dual/Energy': float(self._energy_dual.value.detach().item()),
            'Numerics/ActorGradientNorm': float(gradient_norm.item()),
        }
        if self._stochastic_flow_enabled():
            return {**common_metrics, **reference_metrics}
        assert geometry is not None
        valid = geometry.gradient_norm >= float(self._cfgs.algo_cfgs.minimum_gradient_norm)
        gate_activation = (geometry.gate > 0.1).float().mean()
        candidate_activation = (geometry.candidate_gate > 0.1).float().mean()
        return {
            **common_metrics,
            'Geometry/GateActivation': float(gate_activation.item()),
            'Geometry/CandidateGateActivation': float(candidate_activation.item()),
            'Geometry/ValidGradientRate': float(valid.float().mean().item()),
            'Geometry/EnsembleDisagreement': float((1.0 - geometry.consistency).mean().item()),
            'Geometry/GradientNorm': float(geometry.gradient_norm.mean().item()),
            'Geometry/VN': float(geometry.normal_variance.mean().item()),
            'Geometry/VT': float(geometry.tangent_variance.mean().item()),
            'Geometry/VNVT': float(
                (geometry.normal_variance / geometry.tangent_variance).mean().item(),
            ),
            'Geometry/RawJVP': float(geometry.raw_jvp_value.mean().item()),
            'Geometry/NormalizedJVP': float(geometry.normalized_jvp_value.mean().item()),
            'Geometry/NormalVelocity': float(geometry.normal_velocity.mean().item()),
            'Geometry/DangerousNormalVelocity': float(
                geometry.dangerous_normal_velocity.mean().item(),
            ),
            'Geometry/TangentEnergy': float(geometry.tangent_energy.mean().item()),
            'Geometry/NormalEnergy': float(geometry.normal_energy.mean().item()),
        }

    def _consume_episode_dual_updates(self) -> None:
        """Record all episodes, activating multiplier steps only from W2 onward."""
        while self._pending_episode_costs:
            episode_cost = self._pending_episode_costs.popleft()
            if self._global_safety_enabled() and self._warmup.stage in {'W2', 'W3'}:
                self._global_dual.update_episode(episode_cost)
            else:
                self._global_dual.record_episode(episode_cost)

    def _update(self) -> None:
        """Execute the fixed P3 update order exactly once."""
        safety_batch = self._buf.sample_batch(self._device)
        safety_augmented_obs = self._augment(safety_batch['obs'], safety_batch['context'])
        safety_next_augmented_obs = self._augment(
            safety_batch['next_obs'],
            safety_batch['next_context'],
        )
        metrics = self._update_cost_critic(
            safety_batch,
            safety_augmented_obs,
            safety_next_augmented_obs,
        )
        metrics.update(
            self._update_violation_critic(
                safety_batch,
                safety_augmented_obs,
                safety_next_augmented_obs,
            ),
        )

        reward_metric_sums: dict[str, float] = {}
        reward_updates = int(self._cfgs.algo_cfgs.reward_critic_updates_per_update)
        actor_augmented_obs = safety_augmented_obs
        for reward_index in range(reward_updates):
            if bool(self._cfgs.algo_cfgs.separate_reward_replay):
                reward_batch = self._sample_reward_batch()
            elif reward_index == 0:
                reward_batch = safety_batch
            else:
                reward_batch = self._buf.sample_batch(self._device)
            reward_augmented_obs = self._augment(
                reward_batch['obs'],
                reward_batch['context'],
            )
            reward_next_augmented_obs = self._augment(
                reward_batch['next_obs'],
                reward_batch['next_context'],
            )
            reward_metrics = self._update_reward_critics(
                reward_batch,
                reward_augmented_obs,
                reward_next_augmented_obs,
            )
            for key, value in reward_metrics.items():
                if key in MAX_LOG_KEYS:
                    reward_metric_sums[key] = max(reward_metric_sums.get(key, 0.0), value)
                else:
                    reward_metric_sums[key] = reward_metric_sums.get(key, 0.0) + value
            actor_augmented_obs = reward_augmented_obs
        metrics.update(
            {
                key: value if key in MAX_LOG_KEYS else value / reward_updates
                for key, value in reward_metric_sums.items()
            },
        )
        actor_updated = False
        actor_interval = int(self._cfgs.algo_cfgs.actor_update_interval)
        if self._warmup.stage in {'W2', 'W3'} and self._update_count % actor_interval == 0:
            metrics.update(self._update_actor(actor_augmented_obs))
            self._actor_update_count += 1
            actor_updated = True
        metrics['Actor/UpdateApplied'] = float(actor_updated)
        self._consume_episode_dual_updates()
        self._actor_critic.polyak_update(
            float(self._cfgs.algo_cfgs.polyak),
            update_actor=actor_updated,
            reward_coefficient=self._reward_target_polyak(),
            violation_coefficient=self._probability_target_polyak(),
        )
        self._update_count += 1
        self._last_update_order = UPDATE_ORDER

        if not self._dynamic_geometry_gate_enabled():
            actor_evidence = 'Geometry/ValidGradientRate' in metrics
            evidence = {
                'cost_td_error': metrics['Critic/CostTD'],
                'calibration_error': metrics['Critic/CalibrationError'],
                'brier_skill': metrics['Critic/BrierSkill'],
                'auroc': metrics['Critic/AUROC'],
            }
            if actor_evidence:
                evidence.update(
                    {
                        'valid_gradient_rate': metrics['Geometry/ValidGradientRate'],
                        'ensemble_disagreement': metrics['Geometry/EnsembleDisagreement'],
                    },
                )
            self._warmup.observe(**evidence)
        self._advance_warmup()
        metrics['Dual/GlobalSafety'] = self._effective_global_safety_multiplier()
        metrics['Train/WarmupStage'] = float(self._warmup.stage_index)
        metrics['Train/GeometryEnabled'] = float(self._training_geometry_enabled())
        ratios = self._buf.category_ratios()
        metrics['Replay/FailureRatio'] = ratios['failure']
        metrics['Replay/BoundaryRatio'] = ratios['boundary']
        metrics['Replay/ViolationRatio'] = self._violation_count / max(self._env_steps, 1)
        for key, value in metrics.items():
            self._accumulate(key, value)

    def _accumulate(self, key: str, value: float) -> None:
        """Accumulate only finite diagnostics for the next tabular row."""
        numeric = float(value)
        if not math.isfinite(numeric):
            if (
                self._numerical_hard_stops_enabled() or self._engineering_hard_stops_enabled()
            ) and key in REQUIRED_FINITE_KEYS:
                raise FloatingPointError(f'non-finite required diagnostic {key}')
            return
        if (
            self._engineering_hard_stops_enabled()
            and key in MAX_LOG_KEYS
            and abs(numeric) > float(self._cfgs.algo_cfgs.q_absolute_overflow_threshold)
        ):
            raise FloatingPointError(
                f'Q-value numerical overflow in {key}: {numeric:.9g}',
            )
        if key in MAX_LOG_KEYS:
            self._metric_sums[key] = max(self._metric_sums.get(key, 0.0), numeric)
            self._metric_counts[key] = 1
        else:
            self._metric_sums[key] = self._metric_sums.get(key, 0.0) + numeric
            self._metric_counts[key] = self._metric_counts.get(key, 0) + 1

    def _mean_metric(self, key: str) -> float:
        """Return interval mean or the most recent persistent value."""
        count = self._metric_counts.get(key, 0)
        if count:
            return self._metric_sums[key] / count
        return self._latest_values.get(key, 0.0)

    @staticmethod
    def _ordinary_least_squares_slope(values: list[float]) -> float:
        """Return the equally spaced least-squares slope of a non-empty scalar sequence."""
        if not values:
            raise ValueError('least-squares slope requires at least one value')
        center = 0.5 * (len(values) - 1)
        denominator = sum((index - center) ** 2 for index in range(len(values)))
        if denominator == 0.0:
            return 0.0
        return sum((index - center) * value for index, value in enumerate(values)) / denominator

    def _check_numerical_hard_stops(self, row: dict[str, float]) -> None:
        """Apply and persist the locked P4.4 5k Reward-TD/Q stopping rules."""
        if not self._numerical_hard_stops_enabled():
            return
        required = (
            'Critic/RewardTD',
            'Critic/RewardQAbsMax',
            'Critic/RewardTargetQAbsMax',
        )
        values = {key: float(row[key]) for key in required}
        if not all(math.isfinite(value) for value in values.values()):
            reasons = ['nonfinite_required_reward_diagnostic']
        else:
            reasons = []
        reward_td = values['Critic/RewardTD']
        self._reward_td_diagnostics.append(reward_td)
        historical_minimum = min(self._reward_td_diagnostics)
        final_ratio = reward_td / max(historical_minimum, 1e-12)
        last_four = self._reward_td_diagnostics[-4:]
        last_four_slope = self._ordinary_least_squares_slope(last_four)
        if abs(reward_td) > float(self._cfgs.algo_cfgs.reward_td_absolute_hard_stop):
            reasons.append('reward_td_absolute_threshold_exceeded')
        q_limit = float(self._cfgs.algo_cfgs.reward_q_absolute_hard_stop)
        if values['Critic/RewardQAbsMax'] > q_limit:
            reasons.append('online_reward_q_absolute_threshold_exceeded')
        if values['Critic/RewardTargetQAbsMax'] > q_limit:
            reasons.append('reward_target_q_absolute_threshold_exceeded')
        ratio_limit = float(
            self._cfgs.algo_cfgs.reward_td_final_to_minimum_ratio_hard_stop,
        )
        if len(last_four) == 4 and final_ratio > ratio_limit and last_four_slope > 0.0:
            reasons.append('reward_td_ratio_and_positive_last_four_slope')
        warning = (
            'positive_last_four_slope_without_ratio_trigger'
            if len(last_four) == 4 and last_four_slope > 0.0 and final_ratio <= ratio_limit
            else None
        )
        status = {
            'env_steps': self._env_steps,
            'enabled': True,
            'reward_td_history': self._reward_td_diagnostics,
            'historical_minimum': historical_minimum,
            'final': reward_td,
            'final_to_historical_minimum_ratio': final_ratio,
            'last_four': last_four,
            'last_four_ols_slope': last_four_slope,
            'online_reward_q_abs_max': values['Critic/RewardQAbsMax'],
            'reward_target_q_abs_max': values['Critic/RewardTargetQAbsMax'],
            'warning': warning,
            'hard_stop_reasons': reasons,
            'passed': not reasons,
        }
        _atomic_json(self._artifact_dir / 'numerical_status_latest.json', status)
        _append_csv(self._artifact_dir / 'numerical_status.csv', status)
        if reasons:
            raise FloatingPointError(f"P4.4 numerical hard stop: {', '.join(reasons)}")

    def _heldout_probability_evidence(self) -> dict[str, float]:  # pylint: disable=too-many-locals
        """Evaluate recent episode-separated held-out data without touching any optimizer."""
        if not self._bellman_probability_enabled():
            return {}
        algo = self._cfgs.algo_cfgs
        batch: dict[str, torch.Tensor] | None = None
        with suppress(RuntimeError):
            batch = self._buf.recent_heldout_batch(
                self._device,
                recent_episodes=int(algo.heldout_recent_episodes),
                batch_size=int(algo.heldout_eval_batch_size),
            )

        calibration_error = float('nan')
        brier_score = float('nan')
        brier_skill = float('nan')
        auroc = float('nan')
        bellman_error = float('nan')
        gradient_norm = float('nan')
        disagreement = float('nan')
        boundary_samples = 0
        positives = 0
        negatives = 0
        episodes = 0
        transitions = 0
        if batch is not None:
            augmented = self._augment(batch['obs'], batch['context'])
            next_augmented = self._augment(batch['next_obs'], batch['next_context'])
            labels = batch['violation_label']
            with torch.no_grad():
                head_probability = self._violation_critic().probabilities(
                    augmented,
                    batch['action'],
                )
                probability = head_probability.mean(dim=0)
                probability_disagreement = head_probability.std(dim=0, unbiased=False)
                risk_ucb = (
                    probability + float(algo.risk_ucb_beta) * probability_disagreement
                ).clamp(0.0, 1.0)
                target = self._violation_bellman_target(batch, next_augmented).mean(dim=0)
                calibration_error = float((probability - labels).abs().mean().item())
                brier_score = binary_brier_score(probability, labels)
                brier_skill = binary_brier_skill(probability, labels)
                auroc = binary_auroc(probability, labels)
                bellman_error = float((probability - target).abs().mean().item())
                disagreement = float(probability_disagreement.mean().item())
                boundary_samples = int(
                    (
                        boundary_gate(
                            risk_ucb,
                            float(algo.risk_boundary),
                            float(algo.boundary_bandwidth),
                        )
                        >= float(algo.boundary_score_threshold)
                    )
                    .sum()
                    .item(),
                )
                positives = int((labels > 0.5).sum().item())
                negatives = int(labels.numel()) - positives
                episodes = int(torch.unique(batch['episode_id']).numel())
                transitions = int(labels.numel())

            heldout_action = batch['action'].detach().requires_grad_(True)
            with _frozen_parameters(self._violation_critic()):
                differentiable_risk = (
                    self._violation_critic()
                    .probabilities(
                        augmented.detach(),
                        heldout_action,
                    )
                    .mean(dim=0)
                )
                action_gradient = torch.autograd.grad(
                    differentiable_risk.sum(),
                    heldout_action,
                )[0]
            gradient_norm = float(action_gradient.norm(dim=-1).mean().item())

        evidence_sufficient = positives >= int(
            algo.heldout_min_positive_labels,
        ) and negatives >= int(algo.heldout_min_negative_labels)
        evidence_status = 'sufficient' if evidence_sufficient else 'insufficient_evidence'
        heldout_calibration_reliability = 0.0
        if evidence_sufficient or not self._complete_heldout_evidence_enabled():
            heldout_calibration_reliability = calibration_reliability(
                auroc,
                brier_score,
                auroc_full=float(getattr(algo, 'safety_calibration_auroc_full', 0.75)),
                brier_score_scale=float(
                    getattr(algo, 'safety_calibration_brier_score_scale', 0.25),
                ),
            )
        evidence = {
            'Heldout/CalibrationError': calibration_error,
            'Heldout/BrierScore': brier_score,
            'Heldout/BrierSkill': brier_skill,
            'Heldout/AUROC': auroc,
            'Heldout/BellmanError': bellman_error,
            'Heldout/GradientNorm': gradient_norm,
            'Heldout/EnsembleDisagreement': disagreement,
            'Heldout/BoundarySamples': float(boundary_samples),
            'Heldout/PositiveLabels': float(positives),
            'Heldout/NegativeLabels': float(negatives),
            'Heldout/Episodes': float(episodes),
            'Heldout/Transitions': float(transitions),
            'Heldout/EvidenceSufficient': float(evidence_sufficient),
            'Heldout/CalibrationReliability': heldout_calibration_reliability,
        }
        gate_passed: bool | None = None
        if self._dynamic_geometry_gate_enabled():
            gate_passed = self._warmup.evaluate_geometry_gate(
                self._env_steps,
                boundary_samples,
                positives,
                negatives,
                cost_td_error=bellman_error,
                calibration_error=calibration_error,
                valid_gradient_rate=gradient_norm,
                ensemble_disagreement=disagreement,
                brier_skill=brier_skill,
                auroc=auroc,
            )
        for key, value in evidence.items():
            if math.isfinite(value):
                self._latest_values[key] = value
                self._accumulate(key, value)
        artifact = {
            'env_steps': self._env_steps,
            'split': 'heldout_episode',
            'recent_episode_limit': int(algo.heldout_recent_episodes),
            'transition_limit': int(algo.heldout_eval_batch_size),
            'partial_episode_allowed': False,
            'evidence_status': evidence_status,
            'gate_passed': gate_passed,
            'curriculum_stage': self._warmup.stage,
            'geometry_enabled': self._training_geometry_enabled(),
            **{key[len('Heldout/') :]: value for key, value in evidence.items()},
        }
        _append_csv(self._artifact_dir / 'heldout_probability.csv', artifact)
        _atomic_json(self._artifact_dir / 'heldout_probability_latest.json', artifact)
        self._last_heldout_evidence = evidence
        self._last_heldout_evidence_status = evidence_status
        return evidence

    def _log_epoch(self, start_time: float) -> None:
        """Dump one complete metrics row and auditable warmup status."""
        self._heldout_probability_evidence()
        now = time.monotonic()
        elapsed = now - self._last_log_time
        new_steps = self._env_steps - self._last_log_step
        row = {key: self._mean_metric(key) for key in LOG_KEYS}
        row.update(
            {
                'Train/Epoch': self._env_steps // int(self._cfgs.algo_cfgs.steps_per_epoch),
                'Train/WarmupStage': float(self._warmup.stage_index),
                'Train/GeometryEnabled': float(self._training_geometry_enabled()),
                'Train/CumulativeCost': self._cumulative_training_cost,
                'Train/ViolationRatio': self._violation_count / max(self._env_steps, 1),
                'Dual/Energy': float(self._energy_dual.value.detach().item()),
                'Dual/GlobalSafety': self._effective_global_safety_multiplier(),
                'TotalEnvSteps': self._env_steps,
                'TotalUpdates': self._update_count,
                'TotalRewardCriticUpdates': self._reward_critic_update_count,
                'Time/FPS': new_steps / max(elapsed, 1e-9),
                'Time/Total': now - start_time,
            },
        )
        self._logger.store(row)
        self._logger.dump_tabular()
        _append_csv(
            self._artifact_dir / 'diagnostics.csv',
            {'wall_time_utc': time.time(), **row},
        )
        self._check_numerical_hard_stops(row)
        _atomic_json(
            self._artifact_dir / 'warmup_status.json',
            {
                'env_steps': self._env_steps,
                'stage': self._warmup.stage,
                'geometry_activation_step': (
                    self._warmup.geometry_activation_step
                    if self._local_geometry_enabled()
                    else None
                ),
                'warmup_geometry_eligibility_step': self._warmup.geometry_eligibility_step,
                'geometry_last_disable_step': self._warmup.geometry_last_disable_step,
                'geometry_enabled': self._training_geometry_enabled(),
                'local_geometry_configured': self._local_geometry_enabled(),
                'ema': self._warmup.ema,
                'latest_heldout_gate_values': self._warmup.latest,
                'conditions': self._warmup.conditions(
                    self._env_steps,
                    self._buf.pool_sizes['boundary'],
                    self._buf.label_counts['positive'],
                    self._buf.label_counts['negative'],
                ),
                'gate_conditions': self._warmup.gate_conditions(
                    self._env_steps,
                    int(self._last_heldout_evidence.get('Heldout/BoundarySamples', 0.0)),
                    int(self._last_heldout_evidence.get('Heldout/PositiveLabels', 0.0)),
                    int(self._last_heldout_evidence.get('Heldout/NegativeLabels', 0.0)),
                ),
                'consecutive_passes': self._warmup.consecutive_passes,
                'consecutive_failures': self._warmup.consecutive_failures,
                'gate_evaluations': self._warmup.gate_evaluations,
                'boundary_samples': self._buf.pool_sizes['boundary'],
                'label_counts': {
                    'replay_train': self._buf.label_counts,
                    'heldout': self._buf.label_counts_for('heldout'),
                },
                'heldout_evidence': self._last_heldout_evidence,
            },
        )
        self._metric_sums.clear()
        self._metric_counts.clear()
        self._last_log_step = self._env_steps
        self._last_log_time = now

    def _evaluate(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
        self,
        milestone: int,
    ) -> dict[str, Any]:
        """Evaluate fixed seeds and fixed stochastic flow bases without touching training state."""
        requested = int(self._cfgs.train_cfgs.eval_episodes)
        configured_seeds = [int(seed) for seed in self._cfgs.algo_cfgs.eval_seeds]
        episodes_per_seed = int(
            getattr(self._cfgs.algo_cfgs, 'eval_episodes_per_seed', 1),
        )
        base_seeds = configured_seeds[:requested] if episodes_per_seed == 1 else configured_seeds
        evaluation_schedule = [
            (base_seed, episode_index)
            for base_seed in base_seeds
            for episode_index in range(episodes_per_seed)
        ]
        if not evaluation_schedule:
            return {
                'milestone': milestone,
                'probability_split': 'external_isolated_environment',
                'episodes': 0,
                'base_seeds': [],
                'episodes_per_base_seed': episodes_per_seed,
                'mean_return': 0.0,
                'mean_cost': 0.0,
                'cost_cvar_20': 0.0,
                'failure_count': 0,
                'failure_rate': 0.0,
                'failure_reason_counts': {},
                'failure_times': [],
                'mean_failure_time': float('nan'),
                'time_limit_count': 0,
                'time_limit_rate': 0.0,
                'violation_ratio': 0.0,
                'calibration_error': float('nan'),
                'brier_score': float('nan'),
                'brier_skill': float('nan'),
                'auroc': float('nan'),
                'spearman': float('nan'),
                'goals_per_episode': 0.0,
                'goals_per_episode_mean': 0.0,
                'stall_ratio': 0.0,
                'mean_displacement': 0.0,
                'mean_start_end_displacement': 0.0,
                'mean_step_displacement': 0.0,
                'mean_path_length': 0.0,
                'mean_action_norm': 0.0,
                'success_rate': 0.0,
                'cost_per_successful_episode': float('nan'),
                'cost_per_unit_path_length': float('nan'),
                'training_cumulative_cost': self._cumulative_training_cost,
                'training_violation_ratio': self._violation_count / max(self._env_steps, 1),
                'cost_to_go_mae': float('nan'),
                'cost_risk_bins': [],
                'direction_geometry': {},
                'direction_geometry_high_risk_top_20_percent': {},
                'trajectory': [],
            }
        evaluation_env = self._env.evaluation_env
        returns: list[float] = []
        costs: list[float] = []
        lengths: list[int] = []
        predicted_risks: list[float] = []
        predicted_costs: list[float] = []
        empirical_costs: list[float] = []
        thresholds: list[float] = []
        goal_counts: list[int] = []
        displacements: list[float] = []
        path_lengths: list[float] = []
        minimum_goal_distances: list[float] = []
        episode_action_norms: list[float] = []
        stall_flags: list[bool] = []
        episode_step_stall_ratios: list[float] = []
        all_action_norms: list[float] = []
        episode_failure_flags: list[bool] = []
        episode_end_reasons: list[str] = []
        failure_times: list[float] = []
        failure_steps: list[int] = []
        failure_heights: list[float] = []
        failure_torso_angles: list[float] = []
        trajectory: list[dict[str, Any]] = []
        trajectory_seeds = {
            int(seed) for seed in getattr(self._cfgs.algo_cfgs, 'trajectory_eval_seeds', [])
        }
        geometry_sums: dict[str, float] = {}
        geometry_counts: dict[str, int] = {}
        high_risk_margin_chunks: list[torch.Tensor] = []
        high_risk_metric_chunks: dict[str, list[torch.Tensor]] = {
            'dangerous_outward_velocity': [],
            'normal_to_tangent_variance_ratio': [],
            'outward_velocity_ratio': [],
            'reliability': [],
            'safety_weight': [],
        }
        cost_geometry_scale = (
            self._cost_geometry_scale() if self._unified_cost_geometry_enabled() else 1.0
        )

        def accumulate_geometry(name: str, values: torch.Tensor) -> None:
            finite = values.detach().flatten()
            finite = finite[torch.isfinite(finite)]
            if finite.numel():
                geometry_sums[name] = geometry_sums.get(name, 0.0) + float(finite.sum().item())
                geometry_counts[name] = geometry_counts.get(name, 0) + int(finite.numel())

        violations = 0
        transitions = 0
        total_movement_distance = 0.0
        total_measured_movements = 0
        actor_was_training = self._actor_critic.training
        self._actor_critic.eval()
        try:
            flow_generators: dict[int, torch.Generator] = {}
            stream_observations: dict[int, torch.Tensor] = {}
            for seed, episode_index in evaluation_schedule:
                if episode_index == 0:
                    obs, _ = evaluation_env.reset(seed=seed)
                    generator = torch.Generator(device='cpu')
                    generator.manual_seed(1_000_003 + seed)
                    flow_generators[seed] = generator
                else:
                    obs = stream_observations[seed]
                    generator = flow_generators[seed]
                context_state = STARContext(
                    self._env_id,
                    1,
                    float(self._cfgs.lagrange_cfgs.cost_limit),
                    int(self._cfgs.algo_cfgs.max_episode_steps),
                    self._device,
                    include_violation=self._bellman_probability_enabled(),
                )
                episode_return = 0.0
                episode_cost = 0.0
                episode_goals = 0
                episode_failed = False
                episode_end_reason = 'max_episode_steps'
                episode_failure_time = float('nan')
                episode_failure_step = -1
                episode_failure_height = float('nan')
                episode_failure_torso_angle = float('nan')
                step_costs: list[float] = []
                step_predictions: list[float] = []
                step_predicted_costs: list[float] = []
                step_thresholds: list[float] = []
                action_norms: list[float] = []
                start_position: tuple[float, float] | None = None
                last_position: tuple[float, float] | None = None
                previous_position: tuple[float, float] | None = None
                minimum_goal_distance = float('inf')
                measured_movements = 0
                stalled_movements = 0
                path_length = 0.0
                for step in range(int(self._cfgs.algo_cfgs.max_episode_steps)):
                    position = self._env.evaluation_position()
                    position_source = 'simulator_xy'
                    if position is None:
                        position = (float(obs[0, 0].item()), float(obs[0, 1].item()))
                        position_source = 'observation_projection'
                    if start_position is None:
                        start_position = position
                    last_position = position
                    if previous_position is not None:
                        movement = math.dist(previous_position, position)
                        measured_movements += 1
                        stalled_movements += int(movement <= 1e-3)
                        path_length += movement
                    previous_position = position
                    layout = self._env.evaluation_layout() or {}
                    goal = layout.get('goal')
                    if goal is not None:
                        minimum_goal_distance = min(
                            minimum_goal_distance,
                            math.dist(position, goal),
                        )
                    context, threshold = context_state.values()
                    augmented = self._augment(obs, context)
                    geometry_budget_ratio = (
                        self._cost_geometry_budget_ratio(augmented)
                        if self._unified_cost_geometry_enabled()
                        else context[..., 0]
                    )
                    base_action = torch.empty(
                        (1, self._act_dim),
                        dtype=augmented.dtype,
                        device='cpu',
                    ).uniform_(-1.0, 1.0, generator=generator)
                    base_action = base_action.to(self._device)
                    flow_noise = None
                    if self._stochastic_flow_enabled():
                        flow_noise = torch.randn(
                            (1, int(self._cfgs.model_cfgs.flow.num_steps), self._act_dim),
                            dtype=augmented.dtype,
                            device='cpu',
                            generator=generator,
                        ).to(self._device)
                    with torch.no_grad():
                        flow = self._policy_flow(
                            augmented,
                            base_action=base_action,
                            flow_noise=flow_noise,
                        )
                        action = flow.action
                        quantiles = self._actor_critic.cost_critic(augmented, action)
                        risk = self._violation_critic().probabilities(augmented, action).mean()
                    trajectory_normal = torch.zeros(self._act_dim, device=self._device)
                    if self._unified_cost_geometry_enabled():
                        reference = flow.reference
                        if reference is None:
                            raise RuntimeError('unified evaluation flow has no Q-cost reference')
                        raw_normal_velocity = (flow.raw_velocities * reference.normals).sum(dim=-1)
                        accumulate_geometry('normal_variance', reference.normal_variance)
                        accumulate_geometry('tangent_variance', reference.tangent_variance)
                        accumulate_geometry(
                            'normal_to_tangent_variance_ratio',
                            reference.normal_variance / reference.tangent_variance,
                        )
                        accumulate_geometry(
                            'dangerous_normal_velocity',
                            raw_normal_velocity.clamp_min(0.0),
                        )
                        accumulate_geometry(
                            'outward_velocity_ratio',
                            (raw_normal_velocity > 0.0).float(),
                        )
                        accumulate_geometry('tangent_energy', reference.tangent_energy)
                        accumulate_geometry('normal_energy', reference.normal_energy)
                        accumulate_geometry(
                            'action_clipping_rate',
                            (flow.proposals.abs() > 1.0).any(dim=-1).float(),
                        )
                        accumulate_geometry(
                            'safety_band_sample_ratio',
                            (
                                reference.margins.abs()
                                <= float(self._cfgs.algo_cfgs.cost_geometry_bandwidth)
                            ).float(),
                        )
                        accumulate_geometry('safety_weight', reference.boundary_reliability)
                        accumulate_geometry(
                            'danger_side_sample_ratio',
                            (reference.margins >= 0.0).float(),
                        )
                        high_risk_margin_chunks.append(reference.margins.detach().flatten().cpu())
                        high_risk_values = {
                            'dangerous_outward_velocity': raw_normal_velocity.clamp_min(0.0),
                            'normal_to_tangent_variance_ratio': (
                                reference.normal_variance / reference.tangent_variance
                            ),
                            'outward_velocity_ratio': (raw_normal_velocity > 0.0).float(),
                            'reliability': reference.reliability,
                            'safety_weight': reference.boundary_reliability,
                        }
                        for name, values in high_risk_values.items():
                            high_risk_metric_chunks[name].append(
                                values.detach().flatten().cpu(),
                            )
                        accumulate_geometry(
                            'action_gradient_norm',
                            reference.gradient_norm * cost_geometry_scale,
                        )
                        point_count = flow.points.shape[-2]
                        path_obs = (
                            augmented.unsqueeze(-2)
                            .expand(
                                -1,
                                point_count,
                                -1,
                            )
                            .reshape(-1, augmented.shape[-1])
                        )
                        path_actions = flow.points.reshape(-1, self._act_dim)
                        with torch.no_grad():
                            path_cost = self._actor_critic.cost_critic.expected_cost(
                                path_obs,
                                path_actions,
                            ).mean(dim=0)
                        path_margin = path_cost.reshape(
                            1,
                            point_count,
                        ) / cost_geometry_scale - geometry_budget_ratio.unsqueeze(-1)
                        accumulate_geometry(
                            'crossing_rate',
                            ((path_margin[:, :-1] <= 0.0) & (path_margin[:, 1:] > 0.0)).float(),
                        )
                        trajectory_normal = reference.normals.mean(dim=-2)[0]
                        trajectory_normal = trajectory_normal / trajectory_normal.norm().clamp_min(
                            1e-8,
                        )
                    action_norm = float(
                        torch.linalg.vector_norm(  # pylint: disable=not-callable
                            action,
                            dim=-1,
                        ).item(),
                    )
                    action_norms.append(action_norm)
                    all_action_norms.append(action_norm)
                    trajectory_row: dict[str, Any] | None = None
                    if seed in trajectory_seeds and episode_index == 0:
                        tangent_x = (
                            -float(trajectory_normal[1].item()) if self._act_dim >= 2 else 0.0
                        )
                        tangent_y = (
                            float(trajectory_normal[0].item()) if self._act_dim >= 2 else 0.0
                        )
                        trajectory_row = {
                            'milestone': milestone,
                            'seed': seed,
                            'base_seed': seed,
                            'episode_index': episode_index,
                            'step': step,
                            'x': position[0],
                            'y': position[1],
                            'position_source': position_source,
                            'goal_x': goal[0] if goal is not None else None,
                            'goal_y': goal[1] if goal is not None else None,
                            'action_norm': action_norm,
                            'action_0': float(action[0, 0].item()),
                            'action_1': float(action[0, 1].item()),
                            'base_action_0': float(base_action[0, 0].item()),
                            'base_action_1': float(base_action[0, 1].item()),
                            'flow_noise_norm': (
                                float(flow_noise.norm().item()) if flow_noise is not None else 0.0
                            ),
                            'predicted_violation_probability': float(risk.item()),
                            'predicted_cost_to_go': float(quantiles.mean().item()),
                            'remaining_budget_ratio': float(context[0, 0].item()),
                            'geometry_budget_ratio': float(geometry_budget_ratio[0].item()),
                            'cost_score': float(
                                quantiles.mean().item() / cost_geometry_scale,
                            ),
                            'cost_margin': float(
                                quantiles.mean().item() / cost_geometry_scale
                                - geometry_budget_ratio[0].item(),
                            ),
                            'cost_normal_0': float(trajectory_normal[0].item()),
                            'cost_normal_1': (
                                float(trajectory_normal[1].item()) if self._act_dim >= 2 else 0.0
                            ),
                            'cost_tangent_0': tangent_x,
                            'cost_tangent_1': tangent_y,
                            'hazards_json': (
                                json.dumps(layout.get('hazards', []), separators=(',', ':'))
                                if step == 0
                                else ''
                            ),
                            'goal_radius': layout.get('goal_radius'),
                            'hazard_radius': layout.get('hazard_radius'),
                        }
                    next_obs, reward, cost, terminated, truncated, info = evaluation_env.step(
                        action,
                    )
                    final_info = info.get('final_info')
                    diagnostic_info = final_info if isinstance(final_info, dict) else info
                    original_reward = float(
                        torch.as_tensor(
                            info.get(
                                'original_reward',
                                diagnostic_info.get('original_reward', reward),
                            ),
                        )
                        .reshape(-1)[0]
                        .item(),
                    )
                    original_cost = float(
                        torch.as_tensor(
                            info.get(
                                'original_cost',
                                diagnostic_info.get('original_cost', cost),
                            ),
                        )
                        .reshape(-1)[0]
                        .item(),
                    )
                    goal_met = int(
                        _truthy(info.get('goal_met', diagnostic_info.get('goal_met', False))),
                    )
                    is_failure = _truthy(
                        diagnostic_info.get(
                            'is_failure',
                            original_cost > 0.0 and _truthy(terminated),
                        ),
                    )
                    if is_failure:
                        episode_failed = True
                        episode_failure_time = float(
                            torch.as_tensor(
                                diagnostic_info.get('episode_time', step + 1),
                            )
                            .reshape(-1)[0]
                            .item(),
                        )
                        episode_failure_step = int(
                            torch.as_tensor(
                                diagnostic_info.get('episode_step', step + 1),
                            )
                            .reshape(-1)[0]
                            .item(),
                        )
                        episode_failure_height = float(
                            torch.as_tensor(
                                diagnostic_info.get('health_height', float('nan')),
                            )
                            .reshape(-1)[0]
                            .item(),
                        )
                        episode_failure_torso_angle = float(
                            torch.as_tensor(
                                diagnostic_info.get('torso_angle', float('nan')),
                            )
                            .reshape(-1)[0]
                            .item(),
                        )
                    if _truthy(terminated) or _truthy(truncated):
                        episode_end_reason = str(
                            diagnostic_info.get(
                                'failure_reason',
                                'terminated' if _truthy(terminated) else 'time_limit',
                            ),
                        )
                    episode_goals += goal_met
                    if trajectory_row is not None:
                        trajectory_row.update(
                            {
                                'reward': original_reward,
                                'cost': original_cost,
                                'goal_met': goal_met,
                                'terminated': int(_truthy(terminated)),
                                'truncated': int(_truthy(truncated)),
                                'is_failure': int(is_failure),
                                'failure_reason': str(
                                    diagnostic_info.get('failure_reason', 'none'),
                                ),
                                'is_healthy': int(
                                    _truthy(diagnostic_info.get('is_healthy', True)),
                                ),
                                'health_height': diagnostic_info.get('health_height'),
                                'torso_angle': diagnostic_info.get('torso_angle'),
                                'torso_upright': diagnostic_info.get('torso_upright'),
                                'episode_time': diagnostic_info.get('episode_time'),
                            },
                        )
                        trajectory.append(trajectory_row)
                    episode_return += original_reward
                    episode_cost += original_cost
                    step_costs.append(original_cost)
                    step_predictions.append(float(risk.item()))
                    step_predicted_costs.append(float(quantiles.mean().item()))
                    step_thresholds.append(float(threshold.item()))
                    violations += int(original_cost > 0)
                    transitions += 1
                    context_state.advance(
                        torch.tensor([original_cost], device=self._device),
                        terminated,
                        truncated,
                    )
                    obs = next_obs
                    if bool(terminated[0].item()) or bool(truncated[0].item()):
                        break
                stream_observations[seed] = obs
                future_cost = 0.0
                episode_empirical = []
                for value in reversed(step_costs):
                    future_cost += value
                    episode_empirical.append(future_cost)
                episode_empirical.reverse()
                predicted_risks.extend(step_predictions)
                predicted_costs.extend(step_predicted_costs)
                empirical_costs.extend(episode_empirical)
                thresholds.extend(step_thresholds)
                returns.append(episode_return)
                costs.append(episode_cost)
                lengths.append(len(step_costs))
                episode_failure_flags.append(episode_failed)
                episode_end_reasons.append(episode_end_reason)
                if episode_failed:
                    failure_times.append(episode_failure_time)
                    failure_steps.append(episode_failure_step)
                    failure_heights.append(episode_failure_height)
                    failure_torso_angles.append(episode_failure_torso_angle)
                goal_counts.append(episode_goals)
                displacement = (
                    math.dist(start_position, last_position)
                    if start_position is not None and last_position is not None
                    else 0.0
                )
                displacements.append(displacement)
                path_lengths.append(path_length)
                minimum_goal_distances.append(minimum_goal_distance)
                episode_action_norms.append(
                    sum(action_norms) / len(action_norms) if action_norms else 0.0,
                )
                stall_flags.append(episode_goals == 0 and displacement <= 0.25)
                episode_step_stall_ratios.append(
                    stalled_movements / max(measured_movements, 1),
                )
                total_movement_distance += path_length
                total_measured_movements += measured_movements
        finally:
            if actor_was_training:
                self._actor_critic.train()

        predicted_tensor = torch.tensor(predicted_risks)
        predicted_cost_tensor = torch.tensor(predicted_costs)
        empirical_tensor = torch.tensor(empirical_costs)
        threshold_tensor = torch.tensor(thresholds)
        empirical_risk = (empirical_tensor > threshold_tensor).float()
        geometry = {
            name: geometry_sums[name] / max(geometry_counts[name], 1)
            for name in sorted(geometry_sums)
        }
        high_risk_geometry = {}
        if high_risk_margin_chunks:
            high_risk_geometry = high_risk_geometry_summary(
                torch.cat(high_risk_margin_chunks),
                {name: torch.cat(chunks) for name, chunks in high_risk_metric_chunks.items()},
                fraction=0.2,
            )
        successful_episodes = sum(count > 0 for count in goal_counts)
        total_path_length = sum(path_lengths)
        failure_count = sum(episode_failure_flags)
        failure_reason_counts: dict[str, int] = {}
        for reason in episode_end_reasons:
            failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
        time_limit_count = failure_reason_counts.get('time_limit', 0)

        def finite_mean(values: list[float]) -> float:
            finite_values = [value for value in values if math.isfinite(value)]
            return sum(finite_values) / len(finite_values) if finite_values else float('nan')

        return {
            'milestone': milestone,
            'probability_split': 'external_isolated_environment',
            'episodes': len(returns),
            'base_seeds': base_seeds,
            'episodes_per_base_seed': episodes_per_seed,
            'seeds': [seed for seed, _ in evaluation_schedule],
            'episode_indices': [index for _, index in evaluation_schedule],
            'returns': returns,
            'costs': costs,
            'lengths': lengths,
            'mean_return': sum(returns) / len(returns),
            'mean_cost': sum(costs) / len(costs),
            'cost_cvar_20': _tail_mean(costs),
            'failure_count': failure_count,
            'failure_rate': failure_count / len(returns),
            'episode_failure_flags': episode_failure_flags,
            'episode_end_reasons': episode_end_reasons,
            'failure_reason_counts': failure_reason_counts,
            'failure_times': failure_times,
            'failure_steps': failure_steps,
            'failure_heights': failure_heights,
            'failure_torso_angles': failure_torso_angles,
            'mean_failure_time': finite_mean(failure_times),
            'mean_failure_step': finite_mean([float(value) for value in failure_steps]),
            'mean_failure_height': finite_mean(failure_heights),
            'mean_failure_torso_angle': finite_mean(failure_torso_angles),
            'time_limit_count': time_limit_count,
            'time_limit_rate': time_limit_count / len(returns),
            'violation_ratio': violations / max(transitions, 1),
            'calibration_error': float((predicted_tensor - empirical_risk).abs().mean().item()),
            'brier_score': binary_brier_score(predicted_tensor, empirical_risk),
            'brier_skill': binary_brier_skill(predicted_tensor, empirical_risk),
            'auroc': binary_auroc(predicted_tensor, empirical_risk),
            'spearman': _spearman(predicted_cost_tensor, empirical_tensor),
            'cost_to_go_mae': float((predicted_cost_tensor - empirical_tensor).abs().mean().item()),
            'cost_risk_bins': equal_count_risk_bins(
                predicted_cost_tensor,
                empirical_tensor,
                count=5,
            ),
            'predicted_risk_mean': float(predicted_tensor.mean().item()),
            'empirical_risk_mean': float(empirical_risk.mean().item()),
            'goal_counts': goal_counts,
            'goals_per_episode': sum(goal_counts) / len(goal_counts),
            'goals_per_episode_mean': sum(goal_counts) / len(goal_counts),
            'goal_met_episodes': successful_episodes,
            'success_rate': successful_episodes / len(goal_counts),
            'displacements': displacements,
            'mean_displacement': sum(displacements) / len(displacements),
            'mean_start_end_displacement': sum(displacements) / len(displacements),
            'mean_step_displacement': total_movement_distance / max(total_measured_movements, 1),
            'mean_path_length': sum(path_lengths) / len(path_lengths),
            'path_lengths': path_lengths,
            'cost_per_successful_episode': (
                sum(costs) / successful_episodes if successful_episodes else float('nan')
            ),
            'cost_per_unit_path_length': (
                sum(costs) / total_path_length if total_path_length > 0.0 else float('nan')
            ),
            'mean_action_norm': sum(all_action_norms) / max(len(all_action_norms), 1),
            'episode_action_norms': episode_action_norms,
            'minimum_goal_distances': minimum_goal_distances,
            'stall_ratio': sum(stall_flags) / len(stall_flags),
            'step_stall_ratio': sum(episode_step_stall_ratios) / len(episode_step_stall_ratios),
            'training_cumulative_cost': self._cumulative_training_cost,
            'training_violation_ratio': self._violation_count / max(self._env_steps, 1),
            'direction_geometry': geometry,
            'direction_geometry_high_risk_top_20_percent': high_risk_geometry,
            'flow_mode': (
                'fixed-random-base-and-wiener-flow'
                if self._stochastic_flow_enabled()
                else 'fixed-random-flow'
            ),
            'flow_seed_offset': 1_000_003,
            'trajectory_seeds': sorted(trajectory_seeds),
            'trajectory': trajectory,
        }

    def _write_trajectory(self, milestone: int, rows: list[dict[str, Any]]) -> None:
        """Write a compact fixed-seed trajectory CSV."""
        if not rows:
            return
        path = self._artifact_dir / f'trajectory_step_{milestone}.csv'
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _stochastic_reference_counterfactual(self, milestone: int) -> dict[str, Any]:
        """Audit fixed-trace entropy and reference use on one same-state replay batch."""
        if self._buf.size == 0:
            return {'milestone': milestone, 'available': False}
        requested = min(int(self._cfgs.algo_cfgs.counterfactual_batch_size), self._buf.size)
        batch = self._buf.sample_batch(self._device, batch_size=requested)
        augmented = self._augment(batch['obs'], batch['context'])
        base_action = torch.zeros((requested, self._act_dim), device=self._device)
        flow_noise = torch.zeros(
            (requested, int(self._cfgs.model_cfgs.flow.num_steps), self._act_dim),
            device=self._device,
        )
        with torch.no_grad():
            flow = self._policy_flow(
                augmented,
                base_action=base_action,
                flow_noise=flow_noise,
            )
        reference = flow.reference
        if reference is None:
            raise RuntimeError('stochastic counterfactual has no reference trace')
        normal_entropy, tangent_entropy = gaussian_directional_entropies(
            reference.normal_variance,
            reference.tangent_variance,
            self._act_dim,
        )
        entropy_total = normal_entropy + tangent_entropy
        constant = torch.as_tensor(
            2.0 * torch.pi * torch.e,
            device=reference.normal_variance.device,
        )
        isotropic_low_entropy = (
            0.5 * self._act_dim * torch.log(constant * reference.normal_variance)
        )
        trace = reference.normal_variance + (self._act_dim - 1) * reference.tangent_variance
        trace_budget = self._act_dim * float(self._cfgs.algo_cfgs.flow_reference_variance)
        payload = {
            'milestone': milestone,
            'available': True,
            'method': 'single_stochastic_flow_policy',
            'same_base_action': True,
            'base_action': 'zero',
            'wiener_noise': 'zero_for_same_path_reference_audit',
            'sample_count': requested,
            'safety_anisotropy_enabled': self._safety_anisotropy_enabled(),
            'safety_inward_drift_enabled': self._safety_inward_drift_enabled(),
            'safety_outward_rectification_enabled': (self._safety_outward_rectification_enabled()),
            'candidate_probing_enabled': self._candidate_probing_enabled(),
            'risk_tolerance_delta_c': float(self._cfgs.algo_cfgs.risk_boundary),
            'calibration_reliability': self._calibration_reliability(),
            'mean_reference_reliability': float(reference.reliability.mean().item()),
            'mean_boundary_reliability': float(
                reference.boundary_reliability.mean().item(),
            ),
            'mean_gradient_reliability': float(
                reference.gradient_reliability.mean().item(),
            ),
            'mean_ensemble_reliability': float(
                reference.ensemble_reliability.mean().item(),
            ),
            'mean_risk_ucb': float(reference.risk_ucb.mean().item()),
            'mean_margin': float(reference.margins.mean().item()),
            'mean_normal_variance': float(reference.normal_variance.mean().item()),
            'mean_tangent_variance': float(reference.tangent_variance.mean().item()),
            'mean_normal_entropy': float(normal_entropy.mean().item()),
            'mean_tangent_entropy': float(tangent_entropy.mean().item()),
            'mean_total_gaussian_entropy': float(entropy_total.mean().item()),
            'matched_normal_isotropic_low_entropy': float(
                isotropic_low_entropy.mean().item(),
            ),
            'entropy_advantage_vs_matched_normal_isotropic_low': float(
                (entropy_total - isotropic_low_entropy).mean().item(),
            ),
            'trace_budget': trace_budget,
            'mean_trace': float(trace.mean().item()),
            'maximum_trace_relative_error': float(
                ((trace - trace_budget).abs() / max(trace_budget, 1e-12)).max().item(),
            ),
            'mean_inward_drift': float(
                (-(reference.drifts * reference.normals).sum(dim=-1)).mean().item(),
            ),
            'mean_path_energy': float(flow.kinetic_energy.mean().item()),
            'mean_raw_normal_velocity': float(
                (flow.raw_velocities * reference.normals).sum(dim=-1).mean().item(),
            ),
            'mean_executed_normal_velocity': float(
                (flow.velocities * reference.normals).sum(dim=-1).mean().item(),
            ),
            'clipping_rate': float(
                (flow.proposals.abs() > 1.0).any(dim=-1).float().mean().item(),
            ),
            'pre_clipping_variance': float(
                flow.proposals[:, -1].var(dim=0, unbiased=False).sum().item(),
            ),
            'post_clipping_variance': float(
                flow.action.var(dim=0, unbiased=False).sum().item(),
            ),
        }
        _atomic_json(self._artifact_dir / f'counterfactual_step_{milestone}.json', payload)
        return payload

    def _counterfactual(  # pylint: disable=too-many-locals
        self,
        milestone: int,
    ) -> dict[str, Any]:
        """Compare same-state gate/covariance/JVP settings without extra training."""
        if self._stochastic_flow_enabled():
            return self._stochastic_reference_counterfactual(milestone)
        if self._buf.size == 0:
            return {'milestone': milestone, 'available': False}
        requested = min(int(self._cfgs.algo_cfgs.counterfactual_batch_size), self._buf.size)
        batch = self._buf.sample_batch(self._device, batch_size=requested)
        augmented = self._augment(batch['obs'], batch['context'])
        base_action = torch.zeros((requested, self._act_dim), device=self._device)
        off_flow = self._actor_critic.flow(augmented, base_action=base_action)
        off_geometry = self._geometry(
            augmented,
            off_flow,
            enabled=False,
        )
        on_flow = self._actor_critic.flow(augmented, base_action=base_action)
        on_geometry = self._geometry(
            augmented,
            on_flow,
            enabled=True,
        )
        near = (on_geometry.risk - float(self._cfgs.algo_cfgs.risk_boundary)).abs() <= float(
            self._cfgs.algo_cfgs.boundary_bandwidth,
        )
        interior = on_geometry.risk < (
            float(self._cfgs.algo_cfgs.risk_boundary)
            - float(self._cfgs.algo_cfgs.boundary_bandwidth)
        )

        valid = self._buf.capacity if self._buf.size == self._buf.capacity else self._buf.size
        resident_category = self._buf.data['category'][:valid]
        resident_boundary = resident_category == BOUNDARY
        boundary_count = int(resident_boundary.sum().item())
        source_random_fraction = 0.0
        immediate_violation_fraction = 0.0
        if boundary_count:
            source_random_fraction = float(
                self._buf.data['source_random'][:valid][resident_boundary].float().mean().item(),
            )
            immediate_violation_fraction = float(
                self._buf.data['is_failure'][:valid][resident_boundary].float().mean().item(),
            )

        def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
            return float(values[mask].mean().item()) if bool(mask.any()) else float('nan')

        fixed_budget_grid = []
        boundary = float(self._cfgs.algo_cfgs.risk_boundary)
        bandwidth = float(self._cfgs.algo_cfgs.boundary_bandwidth)
        for budget_ratio in (1.0, 0.5, 0.2, 0.0):
            fixed_context = batch['context'].clone()
            fixed_context[:, 0] = budget_ratio
            fixed_augmented = self._augment(batch['obs'], fixed_context)
            fixed_flow = self._actor_critic.flow(fixed_augmented, base_action=base_action)
            fixed_geometry = self._geometry(fixed_augmented, fixed_flow, enabled=True)
            strata = {
                'safe': fixed_geometry.risk < boundary - bandwidth,
                'boundary': (fixed_geometry.risk - boundary).abs() <= bandwidth,
                'high_risk': fixed_geometry.risk > boundary + bandwidth,
            }
            for risk_stratum, selected in strata.items():
                count = int(selected.sum().item())
                risk_span = float('nan')
                normal = [float('nan')] * self._act_dim
                if count:
                    selected_risk = fixed_geometry.risk[selected]
                    risk_span = float(
                        (
                            torch.quantile(selected_risk, 0.95)
                            - torch.quantile(selected_risk, 0.05)
                        ).item(),
                    )
                    normal = [
                        float(value)
                        for value in fixed_geometry.normals[selected].mean(dim=0).tolist()
                    ]
                fixed_budget_grid.append(
                    {
                        'budget_ratio': budget_ratio,
                        'risk_stratum': risk_stratum,
                        'count': count,
                        'risk_p95_minus_p05': risk_span,
                        'gradient_norm': masked_mean(fixed_geometry.gradient_norm, selected),
                        'ensemble_disagreement': masked_mean(
                            1.0 - fixed_geometry.consistency,
                            selected,
                        ),
                        'normal': normal,
                        'dangerous_normal_velocity': masked_mean(
                            fixed_geometry.dangerous_normal_velocity,
                            selected,
                        ),
                    },
                )

        payload = {
            'milestone': milestone,
            'available': True,
            'warmup_stage': self._warmup.stage,
            'training_geometry_enabled': self._training_geometry_enabled(),
            'local_safety_geometry_enabled': self._local_geometry_enabled(),
            'same_base_action': True,
            'gate_off_energy': float(off_geometry.energy.item()),
            'gate_on_energy': float(on_geometry.energy.item()),
            'isotropic_vn': float(off_geometry.normal_variance.mean().item()),
            'directional_vn': float(on_geometry.normal_variance.mean().item()),
            'isotropic_vt': float(off_geometry.tangent_variance.mean().item()),
            'directional_vt': float(on_geometry.tangent_variance.mean().item()),
            'vt_preservation_ratio': float(
                (on_geometry.tangent_variance / off_geometry.tangent_variance).mean().item(),
            ),
            'normalized_jvp': float(on_geometry.normalized_jvp_value.mean().item()),
            'raw_jvp': float(on_geometry.raw_jvp_value.mean().item()),
            'gate_boundary_mean': masked_mean(on_geometry.candidate_gate, near),
            'gate_interior_mean': masked_mean(on_geometry.candidate_gate, interior),
            'gate_activation_rate': float((on_geometry.gate > 0.1).float().mean().item()),
            'valid_gradient_rate': float(
                (on_geometry.gradient_norm >= float(self._cfgs.algo_cfgs.minimum_gradient_norm))
                .float()
                .mean()
                .item(),
            ),
            'dangerous_normal_velocity': float(
                on_geometry.dangerous_normal_velocity.mean().item(),
            ),
            'normal_velocity': float(on_geometry.normal_velocity.mean().item()),
            'fixed_budget_risk_grid': fixed_budget_grid,
            'replay_boundary_count': boundary_count,
            'boundary_source_random_fraction': source_random_fraction,
            'boundary_immediate_violation_fraction': immediate_violation_fraction,
        }
        _atomic_json(self._artifact_dir / f'counterfactual_step_{milestone}.json', payload)
        return payload

    def _checkpoint_optimizers(self) -> dict[str, Any]:
        """Return network and energy dual optimizers for complete recovery."""
        return {**self._actor_critic.optimizers, 'energy_dual': self._energy_dual.optimizer}

    def _save_checkpoint(self, path: Path, include_replay: bool) -> None:
        """Save lightweight milestone or full local recovery state."""
        save_star_checkpoint(
            path,
            model=self._actor_critic,
            optimizers=self._checkpoint_optimizers(),
            dual_state={
                'energy': self._energy_dual.state_dict(),
                'global': self._global_dual.state_dict(),
            },
            normalizer=self._obs_normalizer,
            counters={
                'env_steps': self._env_steps,
                'updates': self._update_count,
                'actor_updates': self._actor_update_count,
                'reward_critic_updates': self._reward_critic_update_count,
                'utd_carry': self._utd_carry,
                'violations': self._violation_count,
                'cumulative_training_cost': self._cumulative_training_cost,
            },
            warmup_state=self._warmup.state_dict(),
            replay=self._buf if include_replay else None,
            extra={
                'environment': self._env_id,
                'seed': self._seed,
                'resolved_config': self._cfgs.todict(),
                'rollout_state': self._env.rollout_state(),
                'completed_milestones': sorted(self._completed_milestones),
                'completed_checkpoint_milestones': sorted(
                    self._completed_checkpoint_milestones,
                ),
                'last_heldout_evidence': dict(self._last_heldout_evidence),
                'last_heldout_evidence_status': self._last_heldout_evidence_status,
                'reward_td_diagnostics': list(self._reward_td_diagnostics),
                'best_safe_checkpoint_selection': self._best_safe_checkpoint_selection,
            },
        )

    def _restore_checkpoint(self, path: Path) -> None:
        """Restore learning state; the simulator starts a fresh episode by design."""
        metadata = load_star_checkpoint(
            path,
            model=self._actor_critic,
            optimizers=self._checkpoint_optimizers(),
            normalizer=self._obs_normalizer,
            replay=self._buf,
            map_location=self._device,
        )
        self._energy_dual.load_state_dict(metadata['duals']['energy'])
        self._global_dual.load_state_dict(metadata['duals']['global'])
        counters = metadata['counters']
        self._env_steps = int(counters['env_steps'])
        self._update_count = int(counters['updates'])
        self._actor_update_count = int(counters.get('actor_updates', 0))
        self._reward_critic_update_count = int(
            counters.get('reward_critic_updates', self._update_count),
        )
        self._utd_carry = float(counters['utd_carry'])
        self._violation_count = int(counters['violations'])
        self._cumulative_training_cost = float(counters.get('cumulative_training_cost', 0.0))
        self._warmup.load_state_dict(metadata['warmup'])
        self._completed_milestones = {
            int(step) for step in metadata['extra'].get('completed_milestones', [])
        }
        self._completed_checkpoint_milestones = {
            int(step) for step in metadata['extra'].get('completed_checkpoint_milestones', [])
        }
        self._last_heldout_evidence = {
            str(key): float(value)
            for key, value in metadata['extra'].get('last_heldout_evidence', {}).items()
            if value is not None
        }
        self._last_heldout_evidence_status = str(
            metadata['extra'].get(
                'last_heldout_evidence_status',
                (
                    'sufficient'
                    if bool(self._last_heldout_evidence.get('Heldout/EvidenceSufficient', 0.0))
                    else 'insufficient_evidence'
                ),
            ),
        )
        self._reward_td_diagnostics = [
            float(value) for value in metadata['extra'].get('reward_td_diagnostics', [])
        ]
        restored_selection = metadata['extra'].get('best_safe_checkpoint_selection')
        self._best_safe_checkpoint_selection = (
            dict(restored_selection) if isinstance(restored_selection, dict) else None
        )
        self._buf.reset_pending_episodes()
        if not metadata['has_replay'] and self._env_steps < int(self._cfgs.train_cfgs.total_steps):
            raise ValueError('training resume requires a full STAR checkpoint with replay')
        log_interval = int(self._cfgs.algo_cfgs.steps_per_epoch)
        self._next_log_step = (self._env_steps // log_interval + 1) * log_interval
        self._last_log_step = self._env_steps
        self._last_log_time = time.monotonic()

    def _update_best_safe_return_checkpoint(
        self,
        milestone: int,
        result: dict[str, Any],
    ) -> None:
        """Retain the highest-return feasible checkpoint, with a safety-first fallback."""
        if int(result['episodes']) <= 0:
            return
        cost_budget = float(self._cfgs.lagrange_cfgs.cost_limit)
        mean_cost = float(result['mean_cost'])
        candidate = {
            'milestone': int(milestone),
            'mean_return': float(result['mean_return']),
            'mean_cost': mean_cost,
            'cost_cvar_20': float(result['cost_cvar_20']),
            'failure_rate': float(result['failure_rate']),
            'formal_cost_budget': cost_budget,
            'constraint_satisfied': mean_cost <= cost_budget + 1e-12,
            'selection_status': (
                'best_safe_return'
                if mean_cost <= cost_budget + 1e-12
                else 'provisional_minimum_cost_until_a_safe_checkpoint_exists'
            ),
        }
        updated = best_safe_checkpoint_candidate_is_better(
            candidate,
            self._best_safe_checkpoint_selection,
        )
        if updated:
            self._best_safe_checkpoint_selection = candidate
            if bool(self._cfgs.algo_cfgs.milestone_checkpoints):
                self._save_checkpoint(
                    self._checkpoint_dir / 'best_safe_return.pt',
                    include_replay=False,
                )
        _atomic_json(
            self._artifact_dir / 'best_safe_return_selection.json',
            {
                'selection_rule': (
                    'feasible_first;maximum_mean_return;minimum_mean_cost;earliest_milestone;'
                    'before_feasibility_minimum_mean_cost_then_maximum_mean_return'
                ),
                'last_candidate': candidate,
                'last_candidate_selected': updated,
                'selected': self._best_safe_checkpoint_selection,
            },
        )

    def _evaluate_milestone(self, milestone: int) -> None:
        """Save fixed evaluation, same-state counterfactual, and lightweight model state."""
        result = self._evaluate(milestone)
        result['environment'] = self._env_id
        trajectory = result.pop('trajectory')
        self._write_trajectory(milestone, trajectory)
        counterfactual = self._counterfactual(milestone)
        result['counterfactual_file'] = (
            f'counterfactual_step_{milestone}.json' if counterfactual['available'] else None
        )
        _atomic_json(self._artifact_dir / f'evaluation_step_{milestone}.json', result)
        _append_csv(
            self._artifact_dir / 'evaluation.csv',
            {
                'milestone': milestone,
                'episodes': result['episodes'],
                'mean_return': result['mean_return'],
                'mean_cost': result['mean_cost'],
                'cost_cvar_20': result['cost_cvar_20'],
                'failure_count': result['failure_count'],
                'failure_rate': result['failure_rate'],
                'mean_failure_time': result['mean_failure_time'],
                'time_limit_count': result['time_limit_count'],
                'time_limit_rate': result['time_limit_rate'],
                'violation_ratio': result['violation_ratio'],
                'calibration_error': result['calibration_error'],
                'brier_score': result['brier_score'],
                'brier_skill': result['brier_skill'],
                'auroc': result['auroc'],
                'spearman': result['spearman'],
                'goals_per_episode_mean': result['goals_per_episode_mean'],
                'stall_ratio': result['stall_ratio'],
                'mean_start_end_displacement': result['mean_start_end_displacement'],
                'mean_step_displacement': result['mean_step_displacement'],
                'mean_path_length': result['mean_path_length'],
                'mean_action_norm': result['mean_action_norm'],
                'success_rate': result['success_rate'],
                'cost_per_successful_episode': result['cost_per_successful_episode'],
                'cost_per_unit_path_length': result['cost_per_unit_path_length'],
                'training_cumulative_cost': result['training_cumulative_cost'],
                'training_violation_ratio': result['training_violation_ratio'],
                'cost_to_go_mae': result['cost_to_go_mae'],
                'normal_variance': result['direction_geometry'].get('normal_variance'),
                'tangent_variance': result['direction_geometry'].get('tangent_variance'),
                'normal_to_tangent_variance_ratio': result['direction_geometry'].get(
                    'normal_to_tangent_variance_ratio',
                ),
                'dangerous_normal_velocity': result['direction_geometry'].get(
                    'dangerous_normal_velocity',
                ),
                'outward_velocity_ratio': result['direction_geometry'].get(
                    'outward_velocity_ratio',
                ),
                'tangent_energy': result['direction_geometry'].get('tangent_energy'),
                'normal_energy': result['direction_geometry'].get('normal_energy'),
                'crossing_rate': result['direction_geometry'].get('crossing_rate'),
                'action_clipping_rate': result['direction_geometry'].get(
                    'action_clipping_rate',
                ),
                'safety_band_sample_ratio': result['direction_geometry'].get(
                    'safety_band_sample_ratio',
                ),
            },
        )
        self._update_best_safe_return_checkpoint(milestone, result)
        for key, value in {
            'Evaluation/Return': result['mean_return'],
            'Evaluation/Cost': result['mean_cost'],
            'Evaluation/CostCVaR': result['cost_cvar_20'],
            'Evaluation/FailureCount': result['failure_count'],
            'Evaluation/FailureRate': result['failure_rate'],
            'Evaluation/ViolationRatio': result['violation_ratio'],
            'Evaluation/CalibrationError': result['calibration_error'],
            'Evaluation/BrierScore': result['brier_score'],
            'Evaluation/BrierSkill': result['brier_skill'],
            'Evaluation/AUROC': result['auroc'],
            'Evaluation/Spearman': result['spearman'],
            'Evaluation/GoalsPerEpisode': result['goals_per_episode'],
            'Evaluation/StallRatio': result['stall_ratio'],
            'Evaluation/Displacement': result['mean_displacement'],
            'Evaluation/StepDisplacement': result['mean_step_displacement'],
            'Evaluation/ActionNorm': result['mean_action_norm'],
            'Evaluation/SuccessRate': result['success_rate'],
            'Evaluation/PathLength': result['mean_path_length'],
            'Evaluation/CostPerSuccessfulEpisode': result['cost_per_successful_episode'],
            'Evaluation/CostPerUnitPathLength': result['cost_per_unit_path_length'],
        }.items():
            if math.isfinite(float(value)):
                self._latest_values[key] = float(value)
        self._completed_milestones.add(milestone)

    def _save_checkpoint_milestone(self, milestone: int) -> None:
        """Save one preregistered lightweight checkpoint independently of evaluation."""
        if not bool(self._cfgs.algo_cfgs.milestone_checkpoints):
            return
        self._completed_checkpoint_milestones.add(milestone)
        self._save_checkpoint(
            self._checkpoint_dir / f'milestone_{milestone}.pt',
            include_replay=False,
        )

    def learn(self) -> tuple[float, float, float]:  # pylint: disable=too-many-branches
        """Run STAR collection, ordered UTD-one updates, milestones, and atomic recovery."""
        self._logger.log('INFO: Start STAR route-A training')
        started = time.monotonic()
        starting_env_steps = self._env_steps
        if 0 in self._milestones and 0 not in self._completed_milestones:
            self._evaluate_milestone(0)
        if 0 in self._checkpoint_milestones and 0 not in self._completed_checkpoint_milestones:
            self._save_checkpoint_milestone(0)
        total_steps = int(self._cfgs.train_cfgs.total_steps)
        try:
            while self._env_steps < total_steps:
                self._collect_once()
                for milestone in self._milestones:
                    if milestone <= self._env_steps and milestone not in self._completed_milestones:
                        self._evaluate_milestone(milestone)
                for milestone in self._checkpoint_milestones:
                    if (
                        milestone <= self._env_steps
                        and milestone not in self._completed_checkpoint_milestones
                    ):
                        self._save_checkpoint_milestone(milestone)
                if self._env_steps >= self._next_log_step or self._env_steps >= total_steps:
                    self._log_epoch(started)
                    while self._next_log_step <= self._env_steps:
                        self._next_log_step += int(self._cfgs.algo_cfgs.steps_per_epoch)
                frequency = int(self._cfgs.algo_cfgs.full_checkpoint_freq)
                if frequency > 0 and self._env_steps % frequency < self._num_envs:
                    self._save_checkpoint(
                        self._checkpoint_dir / 'latest_full.pt',
                        include_replay=bool(self._cfgs.algo_cfgs.checkpoint_replay),
                    )
            if bool(self._cfgs.algo_cfgs.checkpoint_replay):
                self._save_checkpoint(
                    self._checkpoint_dir / 'final_full.pt',
                    include_replay=True,
                )
        except FloatingPointError as error:
            self._hard_stop_reason = str(error)
            engineering_category = (
                'q_value_numerical_overflow'
                if 'overflow' in self._hard_stop_reason.lower()
                else 'nan_or_inf'
            )
            _atomic_json(
                self._artifact_dir
                / (
                    'engineering_hard_stop.json'
                    if self._engineering_hard_stops_enabled()
                    else 'numerical_hard_stop.json'
                ),
                {
                    'env_steps': self._env_steps,
                    'updates': self._update_count,
                    'reason': self._hard_stop_reason,
                    'category': engineering_category,
                },
            )
            raise
        finally:
            self._training_seconds = time.monotonic() - started
            self._run_env_steps = self._env_steps - starting_env_steps
            summary = {
                'status': (
                    'hard_stopped'
                    if self._hard_stop_reason is not None
                    else 'completed' if self._env_steps >= total_steps else 'interrupted'
                ),
                'environment': self._env_id,
                'seed': self._seed,
                'env_steps': self._env_steps,
                'run_env_steps': self._run_env_steps,
                'updates': self._update_count,
                'actor_updates': self._actor_update_count,
                'reward_critic_updates': self._reward_critic_update_count,
                'warmup_stage': self._warmup.stage,
                'training_geometry_enabled': self._training_geometry_enabled(),
                'geometry_activation_step': (
                    self._warmup.geometry_activation_step
                    if self._local_geometry_enabled()
                    else None
                ),
                'warmup_geometry_eligibility_step': self._warmup.geometry_eligibility_step,
                'global_safety_constraint_enabled': self._global_safety_enabled(),
                'local_safety_geometry_enabled': self._local_geometry_enabled(),
                'reward_target_actor_enabled': self._reward_target_actor_enabled(),
                'reward_target_polyak': self._reward_target_polyak(),
                'cost_target_polyak': float(self._cfgs.algo_cfgs.polyak),
                'actor_target_polyak': float(self._cfgs.algo_cfgs.polyak),
                'probability_target_polyak': self._probability_target_polyak(),
                'bellman_probability_enabled': self._bellman_probability_enabled(),
                'dynamic_geometry_gate': self._dynamic_geometry_gate_enabled(),
                'candidate_probing_enabled': self._candidate_probing_enabled(),
                'stochastic_flow_enabled': self._stochastic_flow_enabled(),
                'safety_anisotropy_enabled': self._safety_anisotropy_enabled(),
                'random_anisotropy_enabled': self._random_anisotropy_enabled(),
                'random_anisotropy_seed': int(self._cfgs.algo_cfgs.random_anisotropy_seed),
                'safety_inward_drift_enabled': self._safety_inward_drift_enabled(),
                'safety_outward_rectification_enabled': (
                    self._safety_outward_rectification_enabled()
                ),
                'unified_cost_geometry_enabled': self._unified_cost_geometry_enabled(),
                'cost_geometry_mode': (
                    self._cost_geometry_contract()[0]
                    if self._unified_cost_geometry_enabled()
                    else 'disabled'
                ),
                'cost_geometry_normalization_scale': (
                    self._cost_geometry_scale() if self._unified_cost_geometry_enabled() else None
                ),
                'formal_cost_budget': float(self._cfgs.lagrange_cfgs.cost_limit),
                'directional_loss_enabled': self._directional_loss_enabled(),
                'safety_aware_cost_geometry_enabled': (self._safety_aware_cost_geometry_enabled()),
                'directional_loss_weight': float(
                    self._cfgs.algo_cfgs.directional_loss_weight,
                ),
                'directional_loss_denominator_stop_gradient': (
                    self._safety_aware_cost_geometry_enabled()
                ),
                'complete_heldout_evidence': self._complete_heldout_evidence_enabled(),
                'heldout_evidence_status': self._last_heldout_evidence_status,
                'numerical_hard_stops_enabled': self._numerical_hard_stops_enabled(),
                'engineering_hard_stops_enabled': self._engineering_hard_stops_enabled(),
                'engineering_hard_stop_reason': (
                    self._hard_stop_reason if self._engineering_hard_stops_enabled() else None
                ),
                'numerical_hard_stop_reason': self._hard_stop_reason,
                'reward_td_diagnostics': self._reward_td_diagnostics,
                'risk_tolerance_delta_c': float(self._cfgs.algo_cfgs.risk_boundary),
                'flow_reference_variance': float(
                    getattr(self._cfgs.algo_cfgs, 'flow_reference_variance', 0.0),
                ),
                'flow_normal_variance_ratio': float(
                    getattr(self._cfgs.algo_cfgs, 'flow_normal_variance_ratio', 1.0),
                ),
                'calibration_reliability': (
                    self._calibration_reliability() if self._stochastic_flow_enabled() else 0.0
                ),
                'actor_update_interval': int(self._cfgs.algo_cfgs.actor_update_interval),
                'reward_target_samples': int(self._cfgs.algo_cfgs.reward_target_samples),
                'reward_critic_updates_per_update': int(
                    self._cfgs.algo_cfgs.reward_critic_updates_per_update,
                ),
                'separate_reward_replay': bool(
                    self._cfgs.algo_cfgs.separate_reward_replay,
                ),
                'reward_replay_recent_fraction': float(
                    self._cfgs.algo_cfgs.reward_replay_recent_fraction,
                ),
                'reward_replay_recent_window': int(
                    self._cfgs.algo_cfgs.reward_replay_recent_window,
                ),
                'effective_global_safety_multiplier': (self._effective_global_safety_multiplier()),
                'global_dual_state': self._global_dual.state_dict(),
                'training_seconds': self._training_seconds,
                'end_to_end_fps': self._run_env_steps / max(self._training_seconds, 1e-9),
                'replay': self._buf.category_ratios(),
                'replay_split_counts': {
                    'train_transitions': self._buf.training_size,
                    'heldout_transitions': self._buf.heldout_size,
                    'train_labels': self._buf.label_counts,
                    'heldout_labels': self._buf.label_counts_for('heldout'),
                },
                'heldout_probability_latest': self._last_heldout_evidence,
                'violation_ratio': self._violation_count / max(self._env_steps, 1),
                'cumulative_training_cost': self._cumulative_training_cost,
                'completed_evaluation_milestones': sorted(self._completed_milestones),
                'completed_checkpoint_milestones': sorted(
                    self._completed_checkpoint_milestones,
                ),
                'best_safe_checkpoint_selection': self._best_safe_checkpoint_selection,
                'last_update_order': list(self._last_update_order),
            }
            _atomic_json(self._artifact_dir / 'training_summary.json', summary)
            self._logger.close()
            self._env.close()

        recent = list(self._episode_history)[-10:]
        if not recent:
            return 0.0, 0.0, 0.0
        episode_return = sum(item[0] for item in recent) / len(recent)
        episode_cost = sum(item[1] for item in recent) / len(recent)
        episode_length = sum(item[2] for item in recent) / len(recent)
        return episode_return, episode_cost, episode_length

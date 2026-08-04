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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from omnisafe.adapter.star_adapter import STARAdapter, STARContext
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
from omnisafe.models.actor_critic.star_actor_q_critic import STARActorQCritic, flac_reward_target
from omnisafe.models.critic.distributional_cost_critic import (
    distributional_bellman_target,
    quantile_huber_loss,
)
from omnisafe.models.safety.distributional_backend import SafetyGeometry, compute_safety_geometry


UPDATE_ORDER = (
    'distributional_cost_critic',
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
    'Metrics/EpRet',
    'Metrics/EpCost',
    'Metrics/EpLen',
    'Evaluation/Return',
    'Evaluation/Cost',
    'Evaluation/CostCVaR',
    'Evaluation/ViolationRatio',
    'Evaluation/CalibrationError',
    'Evaluation/Spearman',
    'Loss/CostCritic',
    'Loss/RewardCritic',
    'Loss/Actor',
    'Loss/EnergyDual',
    'Critic/CostTD',
    'Critic/RewardTD',
    'Critic/CalibrationError',
    'Critic/Spearman',
    'Critic/PredictedRisk',
    'Critic/EmpiricalRisk',
    'Critic/LabeledRatio',
    'Critic/RiskP10',
    'Critic/RiskP50',
    'Critic/RiskP90',
    'Critic/RiskUCBMean',
    'Critic/RiskUCBBoundaryRate',
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
    'Actor/TerminalReward',
    'Actor/TerminalCost',
    'Energy/Path',
    'Dual/Energy',
    'Dual/GlobalSafety',
    'Replay/FailureRatio',
    'Replay/BoundaryRatio',
    'Replay/ViolationRatio',
    'TotalEnvSteps',
    'TotalUpdates',
    'Time/FPS',
    'Time/Total',
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
    """Return rank correlation without a SciPy dependency."""
    if first.numel() < 3 or second.numel() != first.numel():
        return float('nan')
    first_rank = torch.argsort(torch.argsort(first)).float()
    second_rank = torch.argsort(torch.argsort(second)).float()
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = first_centered.norm() * second_centered.norm()
    if denominator <= 0:
        return float('nan')
    return float((first_centered * second_centered).sum().div(denominator).item())


def _tail_mean(values: list[float], fraction: float = 0.2) -> float:
    """Return the upper-tail mean used as empirical cost CVaR."""
    if not values:
        return 0.0
    count = max(1, math.ceil(len(values) * fraction))
    return sum(sorted(values, reverse=True)[:count]) / count


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class STAR(BaseAlgo):
    """First-order route-A STAR with distributional cost and reference-flow geometry."""

    _env_steps: int
    _update_count: int
    _utd_carry: float
    _violation_count: int
    _last_update_order: tuple[str, ...]
    _training_seconds: float
    _completed_milestones: set[int]
    _next_log_step: int
    _last_log_step: int
    _last_log_time: float
    _run_env_steps: int

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
        actor_lr = model.actor.lr
        assert actor_lr is not None
        self._actor_critic = STARActorQCritic(
            obs_dim=self._obs_dim,
            act_dim=self._act_dim,
            actor_hidden_sizes=list(model.actor.hidden_sizes),
            critic_hidden_sizes=list(model.critic.hidden_sizes),
            actor_lr=float(actor_lr),
            reward_critic_lr=float(model.critic.reward_lr),
            cost_critic_lr=float(model.critic.cost_lr),
            flow_steps=int(model.flow.num_steps),
            max_velocity=float(model.flow.max_velocity),
            num_quantiles=int(model.critic.num_quantiles),
            safety_ensemble_size=int(model.critic.ensemble_size),
        ).to(self._device)

    def _init(self) -> None:  # pylint: disable=too-many-locals
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
            ema_rate=float(algo.warmup_ema_rate),
        )
        self._env_steps = 0
        self._update_count = 0
        self._utd_carry = 0.0
        self._violation_count = 0
        self._pending_episode_costs: deque[float] = deque()
        self._episode_history: deque[tuple[float, float, int]] = deque(maxlen=100)
        self._metric_sums: dict[str, float] = {}
        self._metric_counts: dict[str, int] = {}
        self._latest_values: dict[str, float] = {}
        self._last_update_order: tuple[str, ...] = ()
        self._training_seconds = 0.0
        self._run_env_steps = 0
        total_steps = int(self._cfgs.train_cfgs.total_steps)
        configured_milestones = [int(step) for step in algo.eval_milestones]
        self._milestones = sorted({0, total_steps, *configured_milestones})
        self._milestones = [step for step in self._milestones if 0 <= step <= total_steps]
        self._completed_milestones: set[int] = set()
        self._next_log_step = int(algo.steps_per_epoch)
        self._last_log_step = 0
        self._last_log_time = time.monotonic()

        resume_from = str(algo.resume_from)
        if resume_from:
            self._restore_checkpoint(Path(resume_from))
        else:
            with torch.no_grad():
                self._obs_normalizer.normalize(self._env.current_obs)

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
        """Append remaining-budget or remaining-horizon context to normalized observations."""
        return torch.cat((self._normalize_obs(obs), context.reshape(-1, 1)), dim=-1)

    def _select_action(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Use bounded random actions in W0/W1 and learned flow actions in W2/W3."""
        context, _ = self._env.current_context()
        random_action = self._warmup.stage in {'W0', 'W1'}
        if random_action:
            action = torch.empty(
                (self._num_envs, self._act_dim),
                dtype=torch.float32,
                device=self._device,
            ).uniform_(-1.0, 1.0)
        else:
            action = self._actor_critic.step(self._augment(self._env.current_obs, context))
        source_random = torch.full(
            (self._num_envs,),
            random_action,
            dtype=torch.bool,
            device=self._device,
        )
        return action, source_random

    @torch.no_grad()
    def _boundary_labels(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        threshold: torch.Tensor,
    ) -> torch.Tensor:
        """Identify conservative inner-boundary samples from ensemble UCB risk."""
        if self._update_count == 0:
            return torch.zeros(action.shape[0], dtype=torch.bool, device=action.device)
        risks = self._actor_critic.cost_critic.exceedance_probability(
            self._augment(obs, context),
            action,
            threshold,
            float(self._cfgs.algo_cfgs.risk_temperature),
        )
        ucb = risks.mean(dim=0) + float(self._cfgs.algo_cfgs.risk_ucb_beta) * risks.std(
            dim=0,
            unbiased=False,
        )
        return torch.logical_and(
            ucb >= float(self._cfgs.algo_cfgs.boundary_low),
            ucb <= float(self._cfgs.algo_cfgs.boundary_safe),
        )

    def _collect_once(self) -> None:
        """Collect one vector transition and execute the explicit UTD entitlement."""
        stage_before_collection = self._warmup.stage
        context, threshold = self._env.current_context()
        action, source_random = self._select_action()
        boundary = self._boundary_labels(
            self._env.current_obs,
            action,
            context,
            threshold,
        )
        transition, completed = self._env.collect_step(action)
        failure = transition['cost'] > 0
        self._buf.store_batch(
            **transition,
            boundary=boundary,
            failure=failure,
            source_random=source_random,
        )
        self._violation_count += int(failure.sum().item())
        for episode in completed:
            self._episode_history.append(
                (episode.episode_return, episode.episode_cost, episode.episode_length),
            )
            self._pending_episode_costs.append(episode.episode_cost)
            episode_metrics = {
                'Metrics/EpRet': episode.episode_return,
                'Metrics/EpCost': episode.episode_cost,
                'Metrics/EpLen': float(episode.episode_length),
            }
            self._latest_values.update(episode_metrics)
            for key, value in episode_metrics.items():
                self._accumulate(key, value)

        with torch.no_grad():
            self._obs_normalizer.normalize(self._env.current_obs)
        self._env_steps += self._num_envs
        self._warmup.advance(self._env_steps, self._buf.pool_sizes['boundary'])
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

    def _bootstrap_mask(self, ensemble_size: int, batch_size: int) -> torch.Tensor:
        """Sample non-empty independent bootstrap masks for ensemble diversity."""
        probability = float(self._cfgs.algo_cfgs.bootstrap_probability)
        mask = (torch.rand(ensemble_size, batch_size, device=self._device) < probability).float()
        for head in range(ensemble_size):
            if mask[head].sum() == 0:
                mask[head, int(torch.randint(batch_size, ()).item())] = 1.0
        return mask

    def _update_cost_critic(  # pylint: disable=too-many-locals
        self,
        batch: dict[str, torch.Tensor],
        obs: torch.Tensor,
        augmented_obs: torch.Tensor,
        next_augmented_obs: torch.Tensor,
    ) -> tuple[dict[str, float], torch.Tensor]:
        """Update ensemble quantiles first and refresh sampled boundary labels."""
        del obs
        with torch.no_grad():
            next_action = self._actor_critic.flow(next_augmented_obs).action
            next_quantiles = self._actor_critic.target_cost_critic(
                next_augmented_obs,
                next_action,
            )
            target = distributional_bellman_target(
                batch['cost'],
                batch['done'],
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
        clip_grad_norm_(
            self._actor_critic.cost_critic.parameters(),
            float(self._cfgs.algo_cfgs.max_grad_norm),
        )
        optimizer.step()

        with torch.no_grad():
            threshold = batch['risk_threshold'].reshape(1, -1, 1)
            risk_heads = torch.sigmoid(
                (prediction - threshold) / float(self._cfgs.algo_cfgs.risk_temperature),
            ).mean(dim=-1)
            mean_risk = risk_heads.mean(dim=0)
            ucb = mean_risk + float(self._cfgs.algo_cfgs.risk_ucb_beta) * (
                risk_heads.std(dim=0, unbiased=False)
            )
            near_boundary = torch.logical_and(
                ucb >= float(self._cfgs.algo_cfgs.boundary_low),
                ucb <= float(self._cfgs.algo_cfgs.boundary_safe),
            )
            self._buf.relabel_boundaries(batch['indices'], near_boundary)
            td_error = (prediction.mean(dim=-1).mean(dim=0) - target.mean(dim=-1).mean(dim=0)).abs()
        calibration = self._calibration(batch, prediction.detach(), risk_heads.detach())
        metrics = {
            'Loss/CostCritic': float(loss.item()),
            'Critic/CostTD': float(td_error.mean().item()),
            'Critic/CalibrationError': calibration['calibration_error'],
            'Critic/Spearman': calibration['spearman'],
            'Critic/PredictedRisk': calibration['predicted_risk'],
            'Critic/EmpiricalRisk': calibration['empirical_risk'],
            'Critic/LabeledRatio': calibration['labeled_ratio'],
            'Critic/RiskP10': float(torch.quantile(mean_risk, 0.1).item()),
            'Critic/RiskP50': float(torch.quantile(mean_risk, 0.5).item()),
            'Critic/RiskP90': float(torch.quantile(mean_risk, 0.9).item()),
            'Critic/RiskUCBMean': float(ucb.mean().item()),
            'Critic/RiskUCBBoundaryRate': float(near_boundary.float().mean().item()),
        }
        return metrics, prediction.detach()

    def _next_path(
        self,
        next_augmented_obs: torch.Tensor,
        threshold: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate target action and matching FLAC future reference energy."""
        if self._warmup.stage != 'W3':
            with torch.no_grad():
                flow = self._actor_critic.flow(next_augmented_obs)
                return flow.action, flow.kinetic_energy
        flow = self._actor_critic.flow(next_augmented_obs)
        geometry = self._geometry(next_augmented_obs, threshold, flow, enabled=True)
        return flow.action.detach(), geometry.per_sample_energy.detach()

    def _update_reward_critics(  # pylint: disable=too-many-locals
        self,
        batch: dict[str, torch.Tensor],
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        next_augmented_obs: torch.Tensor,
    ) -> dict[str, float]:
        """Apply the FLAC target to both reward critics after the safety update."""
        next_action, next_energy = self._next_path(
            next_augmented_obs,
            batch['next_risk_threshold'],
        )
        with torch.no_grad():
            next_first, next_second = self._actor_critic.target_reward_critic(
                next_obs,
                next_action,
            )
            target = flac_reward_target(
                batch['reward'],
                batch['done'],
                next_first,
                next_second,
                next_energy,
                self._energy_dual.value.detach(),
                float(self._cfgs.algo_cfgs.gamma),
            )
        first, second = self._actor_critic.reward_critic(obs, batch['action'])
        per_sample = (first - target).square() + (second - target).square()
        weights = batch['importance_weights']
        loss = (per_sample * weights).sum() / weights.sum().clamp_min(1.0)
        optimizer = self._actor_critic.reward_critic_optimizer
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(
            self._actor_critic.reward_critic.parameters(),
            float(self._cfgs.algo_cfgs.max_grad_norm),
        )
        optimizer.step()
        td_error = 0.5 * ((first - target).abs() + (second - target).abs())
        return {
            'Loss/RewardCritic': float(loss.item()),
            'Critic/RewardTD': float(td_error.mean().item()),
        }

    def _geometry(
        self,
        augmented_obs: torch.Tensor,
        threshold: torch.Tensor,
        flow: Any,
        enabled: bool,
    ) -> SafetyGeometry:
        """Apply the configured first-order distributional reference geometry."""
        algo = self._cfgs.algo_cfgs
        return compute_safety_geometry(
            self._actor_critic.cost_critic,
            augmented_obs,
            threshold,
            flow,
            enabled=enabled,
            late_time=float(algo.late_flow_time),
            risk_temperature=float(algo.risk_temperature),
            boundary=float(algo.risk_boundary),
            boundary_bandwidth=float(algo.boundary_bandwidth),
            minimum_gradient_norm=float(algo.minimum_gradient_norm),
            minimum_consistency=float(algo.minimum_consistency),
            tangent_variance=float(algo.tangent_variance),
            normal_contraction=float(algo.normal_contraction),
            reference_strength=float(algo.reference_strength),
            reference_temperature=float(algo.reference_temperature),
        )

    def _update_actor(  # pylint: disable=too-many-locals
        self,
        batch: dict[str, torch.Tensor],
        obs: torch.Tensor,
        augmented_obs: torch.Tensor,
    ) -> dict[str, float]:
        """Regenerate flow, terminal utility, late geometry, actor, then energy dual."""
        flow = self._actor_critic.flow(augmented_obs)
        enabled = self._warmup.stage == 'W3'
        geometry = self._geometry(
            augmented_obs,
            batch['risk_threshold'],
            flow,
            enabled=enabled,
        )
        with _frozen_parameters(
            self._actor_critic.reward_critic,
            self._actor_critic.cost_critic,
        ):
            reward_first, reward_second = self._actor_critic.reward_critic(obs, flow.action)
            terminal_reward = torch.minimum(reward_first, reward_second)
            head_cost = self._actor_critic.cost_critic.expected_cost(augmented_obs, flow.action)
            terminal_cost = head_cost.mean(dim=0) + float(
                self._cfgs.algo_cfgs.cost_ucb_beta,
            ) * head_cost.std(dim=0, unbiased=False)
            actor_loss = (
                self._energy_dual.value.detach() * geometry.energy
                - terminal_reward.mean()
                + self._global_dual.value * terminal_cost.mean()
            )
            optimizer = self._actor_critic.actor_optimizer
            optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                float(self._cfgs.algo_cfgs.max_grad_norm),
            )
            optimizer.step()

        target_energy = float(self._cfgs.dual_cfgs.target_energy_per_action) * self._act_dim
        dual_loss = self._energy_dual.update(geometry.energy, target_energy)
        valid = geometry.gradient_norm >= float(self._cfgs.algo_cfgs.minimum_gradient_norm)
        gate_activation = (geometry.gate > 0.1).float().mean()
        candidate_activation = (geometry.candidate_gate > 0.1).float().mean()
        return {
            'Loss/Actor': float(actor_loss.item()),
            'Loss/EnergyDual': float(dual_loss.item()),
            'Actor/TerminalReward': float(terminal_reward.mean().item()),
            'Actor/TerminalCost': float(terminal_cost.mean().item()),
            'Energy/Path': float(geometry.energy.item()),
            'Dual/Energy': float(self._energy_dual.value.detach().item()),
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
        """Update the global dual only from queued true completed episode costs."""
        while self._pending_episode_costs:
            self._global_dual.update_episode(self._pending_episode_costs.popleft())

    def _update(self) -> None:
        """Execute the fixed P3 update order exactly once."""
        batch = self._buf.sample_batch(self._device)
        obs = self._normalize_obs(batch['obs'])
        next_obs = self._normalize_obs(batch['next_obs'])
        augmented_obs = torch.cat((obs, batch['context'].reshape(-1, 1)), dim=-1)
        next_augmented_obs = torch.cat(
            (next_obs, batch['next_context'].reshape(-1, 1)),
            dim=-1,
        )
        metrics, _ = self._update_cost_critic(batch, obs, augmented_obs, next_augmented_obs)
        metrics.update(self._update_reward_critics(batch, obs, next_obs, next_augmented_obs))
        if self._warmup.stage in {'W2', 'W3'}:
            metrics.update(self._update_actor(batch, obs, augmented_obs))
        self._consume_episode_dual_updates()
        self._actor_critic.polyak_update(float(self._cfgs.algo_cfgs.polyak))
        self._update_count += 1
        self._last_update_order = UPDATE_ORDER

        actor_evidence = 'Geometry/ValidGradientRate' in metrics
        evidence = {
            'cost_td_error': metrics['Critic/CostTD'],
            'calibration_error': metrics['Critic/CalibrationError'],
        }
        if actor_evidence:
            evidence.update(
                {
                    'valid_gradient_rate': metrics['Geometry/ValidGradientRate'],
                    'ensemble_disagreement': metrics['Geometry/EnsembleDisagreement'],
                },
            )
        self._warmup.observe(**evidence)
        self._warmup.advance(self._env_steps, self._buf.pool_sizes['boundary'])
        metrics['Dual/GlobalSafety'] = self._global_dual.value
        metrics['Train/WarmupStage'] = float(self._warmup.stage_index)
        ratios = self._buf.category_ratios()
        metrics['Replay/FailureRatio'] = ratios['failure']
        metrics['Replay/BoundaryRatio'] = ratios['boundary']
        metrics['Replay/ViolationRatio'] = self._violation_count / max(self._env_steps, 1)
        for key, value in metrics.items():
            self._accumulate(key, value)

    def _calibration(
        self,
        batch: dict[str, torch.Tensor],
        prediction: torch.Tensor,
        predicted_risk_heads: torch.Tensor,
    ) -> dict[str, float]:
        """Compare predicted future risk against finalized empirical cost-to-go labels."""
        empirical = batch['empirical_cost_to_go']
        labeled = torch.isfinite(empirical)
        labeled_ratio = float(labeled.float().mean().item())
        if int(labeled.sum().item()) < 3:
            return {
                'calibration_error': float('nan'),
                'spearman': float('nan'),
                'predicted_risk': float(predicted_risk_heads.mean().item()),
                'empirical_risk': float('nan'),
                'labeled_ratio': labeled_ratio,
            }
        predicted_risk = predicted_risk_heads.mean(dim=0)[labeled]
        empirical_cost = empirical[labeled]
        threshold = batch['risk_threshold'][labeled]
        empirical_risk = (empirical_cost > threshold).float()
        predicted_cost = prediction.mean(dim=(0, 2))[labeled]
        return {
            'calibration_error': float((predicted_risk - empirical_risk).abs().mean().item()),
            'spearman': _spearman(predicted_cost, empirical_cost),
            'predicted_risk': float(predicted_risk.mean().item()),
            'empirical_risk': float(empirical_risk.mean().item()),
            'labeled_ratio': labeled_ratio,
        }

    def _accumulate(self, key: str, value: float) -> None:
        """Accumulate only finite diagnostics for the next tabular row."""
        numeric = float(value)
        if not math.isfinite(numeric):
            return
        self._metric_sums[key] = self._metric_sums.get(key, 0.0) + numeric
        self._metric_counts[key] = self._metric_counts.get(key, 0) + 1

    def _mean_metric(self, key: str) -> float:
        """Return interval mean or the most recent persistent value."""
        count = self._metric_counts.get(key, 0)
        if count:
            return self._metric_sums[key] / count
        return self._latest_values.get(key, 0.0)

    def _log_epoch(self, start_time: float) -> None:
        """Dump one complete metrics row and auditable warmup status."""
        now = time.monotonic()
        elapsed = now - self._last_log_time
        new_steps = self._env_steps - self._last_log_step
        row = {key: self._mean_metric(key) for key in LOG_KEYS}
        row.update(
            {
                'Train/Epoch': self._env_steps // int(self._cfgs.algo_cfgs.steps_per_epoch),
                'Train/WarmupStage': float(self._warmup.stage_index),
                'Dual/Energy': float(self._energy_dual.value.detach().item()),
                'Dual/GlobalSafety': self._global_dual.value,
                'TotalEnvSteps': self._env_steps,
                'TotalUpdates': self._update_count,
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
        _atomic_json(
            self._artifact_dir / 'warmup_status.json',
            {
                'env_steps': self._env_steps,
                'stage': self._warmup.stage,
                'geometry_activation_step': self._warmup.geometry_activation_step,
                'ema': self._warmup.ema,
                'conditions': self._warmup.conditions(
                    self._env_steps,
                    self._buf.pool_sizes['boundary'],
                ),
                'boundary_samples': self._buf.pool_sizes['boundary'],
            },
        )
        self._metric_sums.clear()
        self._metric_counts.clear()
        self._last_log_step = self._env_steps
        self._last_log_time = now

    def _evaluate(  # pylint: disable=too-many-locals,too-many-statements
        self,
        milestone: int,
    ) -> dict[str, Any]:
        """Evaluate fixed disjoint seeds without touching training environment state."""
        requested = int(self._cfgs.train_cfgs.eval_episodes)
        seeds = [int(seed) for seed in self._cfgs.algo_cfgs.eval_seeds][:requested]
        if not seeds:
            return {
                'milestone': milestone,
                'episodes': 0,
                'mean_return': 0.0,
                'mean_cost': 0.0,
                'cost_cvar_20': 0.0,
                'violation_ratio': 0.0,
                'calibration_error': float('nan'),
                'spearman': float('nan'),
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
        trajectory: list[dict[str, Any]] = []
        violations = 0
        transitions = 0
        self._actor_critic.eval()
        try:
            for episode_index, seed in enumerate(seeds):
                obs, _ = evaluation_env.reset(seed=seed)
                context_state = STARContext(
                    self._env_id,
                    1,
                    float(self._cfgs.lagrange_cfgs.cost_limit),
                    int(self._cfgs.algo_cfgs.max_episode_steps),
                    self._device,
                )
                episode_return = 0.0
                episode_cost = 0.0
                step_costs: list[float] = []
                step_predictions: list[float] = []
                step_predicted_costs: list[float] = []
                step_thresholds: list[float] = []
                for step in range(int(self._cfgs.algo_cfgs.max_episode_steps)):
                    context, threshold = context_state.values()
                    augmented = self._augment(obs, context)
                    with torch.no_grad():
                        action = self._actor_critic.step(augmented, deterministic=True)
                        quantiles = self._actor_critic.cost_critic(augmented, action)
                        risk = torch.sigmoid(
                            (quantiles - threshold.reshape(1, 1, 1))
                            / float(self._cfgs.algo_cfgs.risk_temperature),
                        ).mean()
                    if episode_index == 0:
                        position = self._env.evaluation_position()
                        position_source = 'simulator_xy'
                        if position is None:
                            position = (float(obs[0, 0].item()), float(obs[0, 1].item()))
                            position_source = 'observation_projection'
                        trajectory.append(
                            {
                                'milestone': milestone,
                                'seed': seed,
                                'step': step,
                                'x': position[0],
                                'y': position[1],
                                'position_source': position_source,
                            },
                        )
                    next_obs, reward, cost, terminated, truncated, info = evaluation_env.step(
                        action,
                    )
                    original_reward = float(
                        torch.as_tensor(info.get('original_reward', reward)).reshape(-1)[0].item(),
                    )
                    original_cost = float(
                        torch.as_tensor(info.get('original_cost', cost)).reshape(-1)[0].item(),
                    )
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
        finally:
            self._actor_critic.train()

        predicted_tensor = torch.tensor(predicted_risks)
        predicted_cost_tensor = torch.tensor(predicted_costs)
        empirical_tensor = torch.tensor(empirical_costs)
        threshold_tensor = torch.tensor(thresholds)
        empirical_risk = (empirical_tensor > threshold_tensor).float()
        return {
            'milestone': milestone,
            'episodes': len(returns),
            'seeds': seeds,
            'returns': returns,
            'costs': costs,
            'lengths': lengths,
            'mean_return': sum(returns) / len(returns),
            'mean_cost': sum(costs) / len(costs),
            'cost_cvar_20': _tail_mean(costs),
            'violation_ratio': violations / max(transitions, 1),
            'calibration_error': float((predicted_tensor - empirical_risk).abs().mean().item()),
            'spearman': _spearman(predicted_cost_tensor, empirical_tensor),
            'predicted_risk_mean': float(predicted_tensor.mean().item()),
            'empirical_risk_mean': float(empirical_risk.mean().item()),
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

    def _counterfactual(  # pylint: disable=too-many-locals
        self,
        milestone: int,
    ) -> dict[str, Any]:
        """Compare same-state gate/covariance/JVP settings without extra training."""
        if self._buf.size == 0:
            return {'milestone': milestone, 'available': False}
        requested = min(int(self._cfgs.algo_cfgs.counterfactual_batch_size), self._buf.size)
        batch = self._buf.sample_batch(self._device, batch_size=requested)
        obs = self._normalize_obs(batch['obs'])
        augmented = torch.cat((obs, batch['context'].reshape(-1, 1)), dim=-1)
        base_action = torch.zeros((requested, self._act_dim), device=self._device)
        off_flow = self._actor_critic.flow(augmented, base_action=base_action)
        off_geometry = self._geometry(
            augmented,
            batch['risk_threshold'],
            off_flow,
            enabled=False,
        )
        on_flow = self._actor_critic.flow(augmented, base_action=base_action)
        on_geometry = self._geometry(
            augmented,
            batch['risk_threshold'],
            on_flow,
            enabled=True,
        )
        near = (on_geometry.risk - float(self._cfgs.algo_cfgs.risk_boundary)).abs() <= float(
            self._cfgs.algo_cfgs.boundary_bandwidth,
        )
        interior = on_geometry.risk < float(self._cfgs.algo_cfgs.boundary_low)

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

        payload = {
            'milestone': milestone,
            'available': True,
            'warmup_stage': self._warmup.stage,
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
                'utd_carry': self._utd_carry,
                'violations': self._violation_count,
            },
            warmup_state=self._warmup.state_dict(),
            replay=self._buf if include_replay else None,
            extra={
                'environment': self._env_id,
                'seed': self._seed,
                'rollout_state': self._env.rollout_state(),
                'completed_milestones': sorted(self._completed_milestones),
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
        self._utd_carry = float(counters['utd_carry'])
        self._violation_count = int(counters['violations'])
        self._warmup.load_state_dict(metadata['warmup'])
        self._completed_milestones = {
            int(step) for step in metadata['extra'].get('completed_milestones', [])
        }
        self._buf.reset_pending_episodes()
        if not metadata['has_replay'] and self._env_steps < int(self._cfgs.train_cfgs.total_steps):
            raise ValueError('training resume requires a full STAR checkpoint with replay')
        log_interval = int(self._cfgs.algo_cfgs.steps_per_epoch)
        self._next_log_step = (self._env_steps // log_interval + 1) * log_interval
        self._last_log_step = self._env_steps
        self._last_log_time = time.monotonic()

    def _evaluate_milestone(self, milestone: int) -> None:
        """Save fixed evaluation, same-state counterfactual, and lightweight model state."""
        result = self._evaluate(milestone)
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
                'violation_ratio': result['violation_ratio'],
                'calibration_error': result['calibration_error'],
                'spearman': result['spearman'],
            },
        )
        for key, value in {
            'Evaluation/Return': result['mean_return'],
            'Evaluation/Cost': result['mean_cost'],
            'Evaluation/CostCVaR': result['cost_cvar_20'],
            'Evaluation/ViolationRatio': result['violation_ratio'],
            'Evaluation/CalibrationError': result['calibration_error'],
            'Evaluation/Spearman': result['spearman'],
        }.items():
            if math.isfinite(float(value)):
                self._latest_values[key] = float(value)
        self._completed_milestones.add(milestone)
        if bool(self._cfgs.algo_cfgs.milestone_checkpoints):
            self._save_checkpoint(
                self._checkpoint_dir / f'milestone_{milestone}.pt',
                include_replay=False,
            )

    def learn(self) -> tuple[float, float, float]:
        """Run STAR collection, ordered UTD-one updates, milestones, and atomic recovery."""
        self._logger.log('INFO: Start STAR route-A training')
        started = time.monotonic()
        starting_env_steps = self._env_steps
        if 0 in self._milestones and 0 not in self._completed_milestones:
            self._evaluate_milestone(0)
        total_steps = int(self._cfgs.train_cfgs.total_steps)
        try:
            while self._env_steps < total_steps:
                self._collect_once()
                for milestone in self._milestones:
                    if milestone <= self._env_steps and milestone not in self._completed_milestones:
                        self._evaluate_milestone(milestone)
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
        finally:
            self._training_seconds = time.monotonic() - started
            self._run_env_steps = self._env_steps - starting_env_steps
            summary = {
                'status': 'completed' if self._env_steps >= total_steps else 'interrupted',
                'environment': self._env_id,
                'seed': self._seed,
                'env_steps': self._env_steps,
                'run_env_steps': self._run_env_steps,
                'updates': self._update_count,
                'warmup_stage': self._warmup.stage,
                'geometry_activation_step': self._warmup.geometry_activation_step,
                'training_seconds': self._training_seconds,
                'end_to_end_fps': self._run_env_steps / max(self._training_seconds, 1e-9),
                'replay': self._buf.category_ratios(),
                'violation_ratio': self._violation_count / max(self._env_steps, 1),
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

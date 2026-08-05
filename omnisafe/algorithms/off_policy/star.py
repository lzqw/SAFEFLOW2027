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
)
from omnisafe.models.safety.distributional_backend import (
    SafetyGeometry,
    boundary_gate,
    compute_safety_geometry,
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
    'Metrics/EpRet',
    'Metrics/EpCost',
    'Metrics/EpLen',
    'Evaluation/Return',
    'Evaluation/Cost',
    'Evaluation/CostCVaR',
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
    'Loss/CostCritic',
    'Loss/ViolationCritic',
    'Loss/RewardCritic',
    'Loss/Actor',
    'Loss/EnergyDual',
    'Critic/CostTD',
    'Critic/RewardTD',
    'Critic/RewardTargetSamplingStd',
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


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class STAR(BaseAlgo):
    """First-order route-A STAR with distributional cost and reference-flow geometry."""

    _env_steps: int
    _update_count: int
    _actor_update_count: int
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
        algo = self._cfgs.algo_cfgs
        actor_lr = model.actor.lr
        assert actor_lr is not None
        legacy_compatibility = bool(getattr(algo, 'legacy_checkpoint_compatibility', False))
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
            context_dim=1 if legacy_compatibility else 2,
            legacy_compatibility=legacy_compatibility,
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
            boundary_score_threshold=float(algo.boundary_score_threshold),
            context_dim=self._actor_critic.context_dim,
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
        )
        self._env_steps = 0
        self._update_count = 0
        self._actor_update_count = 0
        self._reward_critic_update_count = 0
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
                'p4_1_correctness': {
                    'context': '[observation, remaining_budget_ratio, remaining_horizon_ratio]',
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

    def _global_safety_enabled(self) -> bool:
        """Return whether the global cost constraint may affect training."""
        return bool(self._cfgs.algo_cfgs.enable_global_safety_constraint)

    def _local_geometry_enabled(self) -> bool:
        """Return whether evidence-ready local geometry may affect the flow."""
        return bool(self._cfgs.algo_cfgs.enable_local_safety_geometry)

    def _training_geometry_enabled(self) -> bool:
        """Return the actual local-geometry state, including the W3 evidence gate."""
        return self._local_geometry_enabled() and self._warmup.stage == 'W3'

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
    def _boundary_scores(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Return a continuous Gaussian score around the unified probability boundary."""
        if self._update_count == 0 or self._actor_critic.violation_critic is None:
            return torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
        algo = self._cfgs.algo_cfgs
        risk_ucb = self._violation_critic().ucb(
            self._augment(obs, context),
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
        context, _ = self._env.current_context()
        action, source_random = self._select_action()
        boundary_score = self._boundary_scores(
            self._env.current_obs,
            action,
            context,
        )
        transition, completed = self._env.collect_step(action)
        failure = transition['cost'] > 0
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
            next_action = self._actor_critic.flow(next_augmented_obs).action
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
        clip_grad_norm_(
            self._actor_critic.cost_critic.parameters(),
            float(self._cfgs.algo_cfgs.max_grad_norm),
        )
        optimizer.step()

        with torch.no_grad():
            predicted_cost = prediction.mean(dim=(0, 2))
            target_cost = target.mean(dim=(0, 2))
            td_error = (predicted_cost - target_cost).abs()
            empirical = batch['empirical_cost_to_go']
            labeled = torch.isfinite(empirical)
            correlation = (
                _spearman(predicted_cost[labeled], empirical[labeled])
                if int(labeled.sum().item()) >= 3
                else float('nan')
            )
        return {
            'Loss/CostCritic': float(loss.item()),
            'Critic/CostTD': float(td_error.mean().item()),
            'Critic/Spearman': correlation,
        }

    def _update_violation_critic(  # pylint: disable=too-many-locals
        self,
        batch: dict[str, torch.Tensor],
        augmented_obs: torch.Tensor,
    ) -> dict[str, float]:
        """Fit completed-episode violation labels and refresh Gaussian boundary scores."""
        critic = self._violation_critic()
        optimizer = self._actor_critic.violation_critic_optimizer
        assert optimizer is not None
        labels = batch['violation_label']
        labeled = torch.isfinite(labels)
        logits = critic(augmented_obs, batch['action'])
        bootstrap = self._bootstrap_mask(logits.shape[0], logits.shape[1], eligible=labeled)
        loss = masked_bce_loss(
            logits,
            labels.nan_to_num(0.0),
            labeled,
            importance_weights=batch['importance_weights'],
            bootstrap_mask=bootstrap,
        )
        optimizer.zero_grad()
        if bool(labeled.any()):
            loss.backward()
            clip_grad_norm_(critic.parameters(), float(self._cfgs.algo_cfgs.max_grad_norm))
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
        clip_grad_norm_(
            self._actor_critic.reward_critic.parameters(),
            float(self._cfgs.algo_cfgs.max_grad_norm),
        )
        optimizer.step()
        self._reward_critic_update_count += 1
        td_error = 0.5 * ((first - target).abs() + (second - target).abs())
        return {
            'Loss/RewardCritic': float(loss.item()),
            'Critic/RewardTD': float(td_error.mean().item()),
            'Critic/RewardTargetSamplingStd': float(sampling_std.item()),
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

    def _update_actor(  # pylint: disable=too-many-locals
        self,
        augmented_obs: torch.Tensor,
    ) -> dict[str, float]:
        """Regenerate flow, terminal utility, late geometry, actor, then energy dual."""
        flow = self._actor_critic.flow(augmented_obs)
        enabled = self._training_geometry_enabled()
        geometry = self._geometry(
            augmented_obs,
            flow,
            enabled=enabled,
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
            actor_loss = normalized_actor_objective(
                geometry.per_sample_energy,
                terminal_reward,
                terminal_cost,
                self._energy_dual.value.detach(),
                self._effective_global_safety_multiplier(),
            ).mean()
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
        metrics.update(self._update_violation_critic(safety_batch, safety_augmented_obs))

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
                reward_metric_sums[key] = reward_metric_sums.get(key, 0.0) + value
            actor_augmented_obs = reward_augmented_obs
        metrics.update(
            {key: value / reward_updates for key, value in reward_metric_sums.items()},
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
        )
        self._update_count += 1
        self._last_update_order = UPDATE_ORDER

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
                'warmup_geometry_eligibility_step': self._warmup.geometry_activation_step,
                'geometry_enabled': self._training_geometry_enabled(),
                'local_geometry_configured': self._local_geometry_enabled(),
                'ema': self._warmup.ema,
                'conditions': self._warmup.conditions(
                    self._env_steps,
                    self._buf.pool_sizes['boundary'],
                    self._buf.label_counts['positive'],
                    self._buf.label_counts['negative'],
                ),
                'boundary_samples': self._buf.pool_sizes['boundary'],
                'label_counts': self._buf.label_counts,
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
        """Evaluate fixed seeds and fixed stochastic flow bases without touching training state."""
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
        trajectory: list[dict[str, Any]] = []
        violations = 0
        transitions = 0
        total_movement_distance = 0.0
        total_measured_movements = 0
        actor_was_training = self._actor_critic.training
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
                episode_goals = 0
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
                generator = torch.Generator(device='cpu')
                generator.manual_seed(1_000_003 + seed)
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
                    base_action = torch.empty(
                        (1, self._act_dim),
                        dtype=augmented.dtype,
                        device='cpu',
                    ).uniform_(-1.0, 1.0, generator=generator)
                    base_action = base_action.to(self._device)
                    with torch.no_grad():
                        action = self._actor_critic.flow(
                            augmented,
                            base_action=base_action,
                        ).action
                        quantiles = self._actor_critic.cost_critic(augmented, action)
                        risk = self._violation_critic().probabilities(augmented, action).mean()
                    action_norm = float(torch.linalg.vector_norm(action, dim=-1).item())
                    action_norms.append(action_norm)
                    all_action_norms.append(action_norm)
                    if episode_index == 0:
                        trajectory.append(
                            {
                                'milestone': milestone,
                                'seed': seed,
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
                                'predicted_violation_probability': float(risk.item()),
                                'hazards_json': (
                                    json.dumps(layout.get('hazards', []), separators=(',', ':'))
                                    if step == 0
                                    else ''
                                ),
                                'goal_radius': layout.get('goal_radius'),
                                'hazard_radius': layout.get('hazard_radius'),
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
                    episode_goals += int(_truthy(info.get('goal_met', False)))
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
            'brier_score': binary_brier_score(predicted_tensor, empirical_risk),
            'brier_skill': binary_brier_skill(predicted_tensor, empirical_risk),
            'auroc': binary_auroc(predicted_tensor, empirical_risk),
            'spearman': _spearman(predicted_cost_tensor, empirical_tensor),
            'predicted_risk_mean': float(predicted_tensor.mean().item()),
            'empirical_risk_mean': float(empirical_risk.mean().item()),
            'goal_counts': goal_counts,
            'goals_per_episode': sum(goal_counts) / len(goal_counts),
            'goals_per_episode_mean': sum(goal_counts) / len(goal_counts),
            'goal_met_episodes': sum(count > 0 for count in goal_counts),
            'displacements': displacements,
            'mean_displacement': sum(displacements) / len(displacements),
            'mean_start_end_displacement': sum(displacements) / len(displacements),
            'mean_step_displacement': total_movement_distance / max(total_measured_movements, 1),
            'mean_path_length': sum(path_lengths) / len(path_lengths),
            'mean_action_norm': sum(all_action_norms) / max(len(all_action_norms), 1),
            'episode_action_norms': episode_action_norms,
            'minimum_goal_distances': minimum_goal_distances,
            'stall_ratio': sum(stall_flags) / len(stall_flags),
            'step_stall_ratio': sum(episode_step_stall_ratios) / len(episode_step_stall_ratios),
            'flow_mode': 'fixed-random-flow',
            'flow_seed_offset': 1_000_003,
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
            },
            warmup_state=self._warmup.state_dict(),
            replay=self._buf if include_replay else None,
            extra={
                'environment': self._env_id,
                'seed': self._seed,
                'resolved_config': self._cfgs.todict(),
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
        self._actor_update_count = int(counters.get('actor_updates', 0))
        self._reward_critic_update_count = int(
            counters.get('reward_critic_updates', self._update_count),
        )
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
            },
        )
        for key, value in {
            'Evaluation/Return': result['mean_return'],
            'Evaluation/Cost': result['mean_cost'],
            'Evaluation/CostCVaR': result['cost_cvar_20'],
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
                'actor_updates': self._actor_update_count,
                'reward_critic_updates': self._reward_critic_update_count,
                'warmup_stage': self._warmup.stage,
                'training_geometry_enabled': self._training_geometry_enabled(),
                'geometry_activation_step': (
                    self._warmup.geometry_activation_step
                    if self._local_geometry_enabled()
                    else None
                ),
                'warmup_geometry_eligibility_step': self._warmup.geometry_activation_step,
                'global_safety_constraint_enabled': self._global_safety_enabled(),
                'local_safety_geometry_enabled': self._local_geometry_enabled(),
                'reward_target_actor_enabled': self._reward_target_actor_enabled(),
                'reward_target_polyak': self._reward_target_polyak(),
                'cost_target_polyak': float(self._cfgs.algo_cfgs.polyak),
                'actor_target_polyak': float(self._cfgs.algo_cfgs.polyak),
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

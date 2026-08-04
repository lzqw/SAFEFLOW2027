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
"""STAR environment adapter with route-A budget and future-failure context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.utils.config import Config


POINT_PREFIX = 'SafetyPoint'


@dataclass
class CompletedEpisode:
    """Unnormalized statistics emitted exactly once at an episode boundary."""

    env_index: int
    episode_return: float
    episode_cost: float
    episode_length: int
    terminated: bool
    truncated: bool


class STARContext:
    """Maintain remaining budget for Point or remaining horizon for failure tasks."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        cost_limit: float,
        max_episode_steps: int,
        device: torch.device | str,
    ) -> None:
        """Initialize per-environment context without inventing locomotion budgets."""
        if num_envs <= 0 or max_episode_steps <= 0:
            raise ValueError('environment count and horizon must be positive')
        if env_id.startswith(POINT_PREFIX) and cost_limit <= 0:
            raise ValueError('Point tasks require a positive episode cost limit')
        self.is_point = env_id.startswith(POINT_PREFIX)
        self.num_envs = num_envs
        self.cost_limit = float(cost_limit)
        self.max_episode_steps = max_episode_steps
        self.device = torch.device(device)
        self.remaining_budget = torch.full(
            (num_envs,),
            self.cost_limit if self.is_point else 0.0,
            device=self.device,
        )
        self.episode_step = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def values(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized actor context and raw risk threshold."""
        if self.is_point:
            context = self.remaining_budget / self.cost_limit
            threshold = self.remaining_budget
        else:
            context = (self.max_episode_steps - self.episode_step).clamp_min(0).float()
            context = context / self.max_episode_steps
            threshold = torch.full_like(context, 0.5)
        return context, threshold

    def advance(
        self,
        cost: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final-observation context, then reset context for auto-reset environments."""
        cost = cost.to(self.device).reshape(self.num_envs)
        if self.is_point:
            transition_budget = (self.remaining_budget - cost).clamp_min(0.0)
            next_context = transition_budget / self.cost_limit
            next_threshold = transition_budget.clone()
            self.remaining_budget = transition_budget
        else:
            transition_step = self.episode_step + 1
            next_context = (self.max_episode_steps - transition_step).clamp_min(0).float()
            next_context = next_context / self.max_episode_steps
            next_threshold = torch.full_like(next_context, 0.5)
            self.episode_step = transition_step

        episode_done = torch.logical_or(terminated.bool(), truncated.bool()).to(self.device)
        if self.is_point:
            self.remaining_budget[episode_done] = self.cost_limit
        self.episode_step[episode_done] = 0
        return next_context, next_threshold

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return rollout context for diagnostic checkpoints."""
        return {
            'remaining_budget': self.remaining_budget.clone(),
            'episode_step': self.episode_step.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore rollout context tensors."""
        self.remaining_budget.copy_(state['remaining_budget'].to(self.device))
        self.episode_step.copy_(state['episode_step'].to(self.device))


class STARAdapter(OnlineAdapter):  # pylint: disable=too-many-instance-attributes
    """Collect transitions with final observations and explicit STAR context semantics."""

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize isolated train/evaluation environments and episode accounting."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._current_obs, _ = self.reset()
        max_steps = int(cfgs.algo_cfgs.max_episode_steps)
        self.context = STARContext(
            env_id=env_id,
            num_envs=num_envs,
            cost_limit=float(cfgs.lagrange_cfgs.cost_limit),
            max_episode_steps=max_steps,
            device=self._device,
        )
        self._episode_return = torch.zeros(num_envs, device=self._device)
        self._episode_cost = torch.zeros(num_envs, device=self._device)
        self._episode_length = torch.zeros(num_envs, dtype=torch.long, device=self._device)

    @property
    def current_obs(self) -> torch.Tensor:
        """Return the current auto-reset training observation."""
        return self._current_obs

    def current_context(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return current normalized context and raw risk threshold."""
        return self.context.values()

    @property
    def evaluation_env(self) -> Any:
        """Return the isolated evaluation environment."""
        if self._eval_env is None:
            raise RuntimeError('STAR requires an isolated evaluation environment')
        return self._eval_env

    def evaluation_position(self) -> tuple[float, float] | None:
        """Read Point robot XY without changing either environment state."""
        task = self._evaluation_task()
        agent = getattr(task, 'agent', None)
        position = getattr(agent, 'pos', None)
        return self._xy(position)

    def _evaluation_task(self) -> Any | None:
        """Unwrap the isolated evaluator and return its native task object."""
        current = cast(Any, self.evaluation_env)
        while current.__class__.__module__.startswith('omnisafe.envs.wrapper'):
            current = current._env  # pylint: disable=protected-access
        native = getattr(current, '_env', None)
        if native is None:
            return None
        base = cast(Any, native.unwrapped)
        return getattr(base, 'task', None)

    @staticmethod
    def _xy(position: Any) -> tuple[float, float] | None:
        """Convert a simulator position to a plain XY pair."""
        if position is None:
            return None
        values = torch.as_tensor(position).flatten()
        if values.numel() < 2:
            return None
        return float(values[0].item()), float(values[1].item())

    def evaluation_layout(self) -> dict[str, Any] | None:
        """Read Point goal and hazard geometry without mutating evaluator state."""
        task = self._evaluation_task()
        goal_object = getattr(task, 'goal', None)
        hazards_object = getattr(task, 'hazards', None)
        goal = self._xy(getattr(goal_object, 'pos', None))
        hazard_positions = getattr(hazards_object, 'pos', None)
        hazards: list[tuple[float, float]] = []
        if hazard_positions is not None:
            positions: Any
            if isinstance(hazard_positions, (list, tuple)):
                positions = hazard_positions
            else:
                values = torch.as_tensor(hazard_positions)
                positions = values.unsqueeze(0) if values.ndim == 1 else values
            hazards = [pair for position in positions if (pair := self._xy(position)) is not None]
        if goal is None and not hazards:
            return None
        return {
            'goal': goal,
            'hazards': hazards,
            'goal_radius': float(getattr(goal_object, 'size', 0.3)),
            'hazard_radius': float(getattr(hazards_object, 'size', 0.2)),
        }

    def collect_step(  # pylint: disable=too-many-locals
        self,
        action: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], list[CompletedEpisode]]:
        """Execute terminal actions and preserve true final observations in replay."""
        context, risk_threshold = self.current_context()
        next_obs, reward, cost, terminated, truncated, info = self.step(action)
        original_reward = torch.as_tensor(
            info.get('original_reward', reward),
            dtype=torch.float32,
            device=self._device,
        ).reshape(-1)
        original_cost = torch.as_tensor(
            info.get('original_cost', cost),
            dtype=torch.float32,
            device=self._device,
        ).reshape(-1)
        self._episode_return += original_reward
        self._episode_cost += original_cost
        self._episode_length += 1

        real_next_obs = next_obs.clone()
        episode_done = torch.logical_or(terminated.bool(), truncated.bool())
        if 'final_observation' in info:
            final_observation = torch.as_tensor(
                info['final_observation'],
                dtype=torch.float32,
                device=self._device,
            )
            real_next_obs[episode_done] = final_observation[episode_done]
        next_context, next_risk_threshold = self.context.advance(
            original_cost,
            terminated,
            truncated,
        )

        completed = []
        for index in torch.nonzero(episode_done, as_tuple=False).flatten().tolist():
            completed.append(
                CompletedEpisode(
                    env_index=index,
                    episode_return=float(self._episode_return[index].item()),
                    episode_cost=float(self._episode_cost[index].item()),
                    episode_length=int(self._episode_length[index].item()),
                    terminated=bool(terminated[index].item()),
                    truncated=bool(truncated[index].item()),
                ),
            )
            self._episode_return[index] = 0.0
            self._episode_cost[index] = 0.0
            self._episode_length[index] = 0

        transition = {
            'obs': self._current_obs,
            'action': action,
            'reward': reward.reshape(-1),
            'cost': original_cost,
            'terminated': terminated.float().reshape(-1),
            'truncated': truncated.float().reshape(-1),
            'next_obs': real_next_obs,
            'context': context,
            'next_context': next_context,
            'risk_threshold': risk_threshold,
            'next_risk_threshold': next_risk_threshold,
        }
        self._current_obs = next_obs
        return transition, completed

    def rollout_state(self) -> dict[str, Any]:
        """Return non-simulator rollout accounting for checkpoint diagnostics."""
        return {
            'context': self.context.state_dict(),
            'episode_return': self._episode_return.clone(),
            'episode_cost': self._episode_cost.clone(),
            'episode_length': self._episode_length.clone(),
        }

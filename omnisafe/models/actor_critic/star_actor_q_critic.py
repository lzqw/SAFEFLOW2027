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
"""Actor, twin reward critics, and distributional safety critics for STAR."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn, optim

from omnisafe.models.actor.flow_actor import FlowActor, FlowOutput
from omnisafe.models.critic.distributional_cost_critic import DistributionalCostCritic
from omnisafe.models.critic.violation_probability_critic import ViolationProbabilityCritic
from omnisafe.typing import Activation, InitFunction
from omnisafe.utils.model import build_mlp_network


@torch.no_grad()
# pylint: disable-next=too-many-arguments
def flac_reward_target(
    reward: torch.Tensor,
    done: torch.Tensor,
    next_q_first: torch.Tensor,
    next_q_second: torch.Tensor,
    next_path_energy: torch.Tensor,
    energy_multiplier: torch.Tensor | float,
    gamma: float,
) -> torch.Tensor:
    """Construct the FLAC reward backup with future flow path energy."""
    return reward + gamma * (1.0 - done) * (
        torch.minimum(next_q_first, next_q_second)
        - torch.as_tensor(energy_multiplier, device=reward.device) * next_path_energy
    )


@torch.no_grad()
# pylint: disable-next=too-many-arguments
def flac_sampled_reward_target(
    reward: torch.Tensor,
    done: torch.Tensor,
    next_q_first: torch.Tensor,
    next_q_second: torch.Tensor,
    next_path_energy: torch.Tensor,
    energy_multiplier: torch.Tensor | float,
    gamma: float,
) -> torch.Tensor:
    """Average the FLAC backup over stochastic target-flow samples.

    The three future tensors use ``[target samples, replay batch]`` layout. Averaging follows
    ``min(Q1, Q2) - alpha_E * energy`` so the conservative twin-critic operator is applied to
    every sampled action before taking the policy expectation.
    """
    if next_q_first.ndim != 2:
        raise ValueError('sampled FLAC tensors must have [samples, batch] shape')
    expected_shape = next_q_first.shape
    if next_q_second.shape != expected_shape or next_path_energy.shape != expected_shape:
        raise ValueError('sampled FLAC future tensors must share one shape')
    if reward.shape != expected_shape[1:] or done.shape != reward.shape:
        raise ValueError('sampled FLAC reward and done tensors must match the replay batch')
    continuation = (
        torch.minimum(next_q_first, next_q_second)
        - torch.as_tensor(energy_multiplier, device=reward.device) * next_path_energy
    ).mean(dim=0)
    return reward + gamma * (1.0 - done) * continuation


class TwinRewardCritic(nn.Module):
    """Two independent scalar reward Q-functions."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        """Initialize both reward critics."""
        super().__init__()
        self.critics = nn.ModuleList(
            [
                build_mlp_network(
                    sizes=[obs_dim + act_dim, *hidden_sizes, 1],
                    activation=activation,
                    output_activation='identity',
                    weight_initialization_mode=weight_initialization_mode,
                )
                for _ in range(2)
            ],
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the two scalar estimates."""
        inputs = torch.cat((obs, action), dim=-1)
        first, second = (critic(inputs).squeeze(-1) for critic in self.critics)
        return first, second


class STARActorQCritic(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Route-A STAR model container with explicit optimizers and target networks."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        obs_dim: int,
        act_dim: int,
        actor_hidden_sizes: list[int],
        critic_hidden_sizes: list[int],
        actor_lr: float = 3e-4,
        reward_critic_lr: float = 3e-4,
        cost_critic_lr: float = 3e-4,
        violation_critic_lr: float = 3e-4,
        flow_steps: int = 4,
        max_velocity: float = 2.0,
        num_quantiles: int = 25,
        safety_ensemble_size: int = 2,
        context_dim: int = 2,
        legacy_compatibility: bool = False,
    ) -> None:
        """Initialize online networks, frozen targets, and their optimizers."""
        super().__init__()
        if context_dim <= 0:
            raise ValueError('STAR context dimension must be positive')
        augmented_obs_dim = obs_dim + context_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_dim = context_dim
        self.legacy_compatibility = legacy_compatibility
        self.actor = FlowActor(
            obs_dim=augmented_obs_dim,
            act_dim=act_dim,
            hidden_sizes=actor_hidden_sizes,
            num_steps=flow_steps,
            max_velocity=max_velocity,
        )
        reward_obs_dim = obs_dim if legacy_compatibility else augmented_obs_dim
        self.reward_critic = TwinRewardCritic(reward_obs_dim, act_dim, critic_hidden_sizes)
        self.cost_critic = DistributionalCostCritic(
            obs_dim=augmented_obs_dim,
            act_dim=act_dim,
            hidden_sizes=critic_hidden_sizes,
            num_quantiles=num_quantiles,
            ensemble_size=safety_ensemble_size,
        )
        self.violation_critic = (
            None
            if legacy_compatibility
            else ViolationProbabilityCritic(
                obs_dim=augmented_obs_dim,
                act_dim=act_dim,
                hidden_sizes=critic_hidden_sizes,
                ensemble_size=safety_ensemble_size,
            )
        )
        self.target_actor = deepcopy(self.actor)
        self.target_reward_critic = deepcopy(self.reward_critic)
        self.target_cost_critic = deepcopy(self.cost_critic)
        self.target_violation_critic = (
            None if self.violation_critic is None else deepcopy(self.violation_critic)
        )
        targets = [self.target_actor, self.target_reward_critic, self.target_cost_critic]
        if self.target_violation_critic is not None:
            targets.append(self.target_violation_critic)
        for target in targets:
            target.requires_grad_(False)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.reward_critic_optimizer = optim.Adam(
            self.reward_critic.parameters(),
            lr=reward_critic_lr,
        )
        self.cost_critic_optimizer = optim.Adam(self.cost_critic.parameters(), lr=cost_critic_lr)
        self.violation_critic_optimizer = (
            None
            if self.violation_critic is None
            else optim.Adam(self.violation_critic.parameters(), lr=violation_critic_lr)
        )

    def forward(
        self,
        augmented_obs: torch.Tensor,
        deterministic: bool = False,
        base_action: torch.Tensor | None = None,
    ) -> FlowOutput:
        """Expose the actor flow through the standard module interface."""
        return self.flow(
            augmented_obs,
            deterministic=deterministic,
            base_action=base_action,
        )

    def flow(
        self,
        augmented_obs: torch.Tensor,
        deterministic: bool = False,
        base_action: torch.Tensor | None = None,
    ) -> FlowOutput:
        """Generate a complete flow trajectory."""
        return self.actor(
            augmented_obs,
            deterministic=deterministic,
            base_action=base_action,
        )

    def target_flow(
        self,
        augmented_obs: torch.Tensor,
        deterministic: bool = False,
        base_action: torch.Tensor | None = None,
    ) -> FlowOutput:
        """Generate a reward-backup trajectory from the slowly moving target actor."""
        return self.target_actor(
            augmented_obs,
            deterministic=deterministic,
            base_action=base_action,
        )

    def step(self, augmented_obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Generate only the terminal action used by the environment."""
        with torch.no_grad():
            return self.flow(augmented_obs, deterministic=deterministic).action

    @torch.no_grad()
    def polyak_update(
        self,
        coefficient: float,
        *,
        update_actor: bool = True,
        reward_coefficient: float | None = None,
        violation_coefficient: float | None = None,
    ) -> None:
        """Soft-update targets with independently configurable probability/reward steps."""
        if not 0.0 < coefficient <= 1.0:
            raise ValueError('Polyak coefficient must lie in (0, 1]')
        reward_step = coefficient if reward_coefficient is None else reward_coefficient
        if not 0.0 < reward_step <= 1.0:
            raise ValueError('Reward Polyak coefficient must lie in (0, 1]')
        violation_step = coefficient if violation_coefficient is None else violation_coefficient
        if not 0.0 < violation_step <= 1.0:
            raise ValueError('Violation Polyak coefficient must lie in (0, 1]')
        pairs: list[tuple[nn.Module, nn.Module, float]] = [
            (self.target_reward_critic, self.reward_critic, reward_step),
            (self.target_cost_critic, self.cost_critic, coefficient),
        ]
        if self.target_violation_critic is not None and self.violation_critic is not None:
            pairs.append(
                (self.target_violation_critic, self.violation_critic, violation_step),
            )
        if update_actor:
            pairs.append((self.target_actor, self.actor, coefficient))
        for target, online, step in pairs:
            for target_parameter, online_parameter in zip(
                target.parameters(),
                online.parameters(),
            ):
                target_parameter.lerp_(online_parameter, step)

    @property
    def optimizers(self) -> dict[str, optim.Optimizer]:
        """Return every network optimizer for checkpointing."""
        result: dict[str, optim.Optimizer] = {
            'actor': self.actor_optimizer,
            'reward_critic': self.reward_critic_optimizer,
            'cost_critic': self.cost_critic_optimizer,
        }
        if self.violation_critic_optimizer is not None:
            result['violation_critic'] = self.violation_critic_optimizer
        return result

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
from omnisafe.typing import Activation, InitFunction
from omnisafe.utils.model import build_mlp_network


@torch.no_grad()
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


class STARActorQCritic(nn.Module):
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
        flow_steps: int = 4,
        max_velocity: float = 2.0,
        num_quantiles: int = 25,
        safety_ensemble_size: int = 2,
    ) -> None:
        """Initialize online networks, frozen targets, and their optimizers."""
        super().__init__()
        augmented_obs_dim = obs_dim + 1
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.actor = FlowActor(
            obs_dim=augmented_obs_dim,
            act_dim=act_dim,
            hidden_sizes=actor_hidden_sizes,
            num_steps=flow_steps,
            max_velocity=max_velocity,
        )
        self.reward_critic = TwinRewardCritic(obs_dim, act_dim, critic_hidden_sizes)
        self.cost_critic = DistributionalCostCritic(
            obs_dim=augmented_obs_dim,
            act_dim=act_dim,
            hidden_sizes=critic_hidden_sizes,
            num_quantiles=num_quantiles,
            ensemble_size=safety_ensemble_size,
        )
        self.target_reward_critic = deepcopy(self.reward_critic)
        self.target_cost_critic = deepcopy(self.cost_critic)
        for target in (self.target_reward_critic, self.target_cost_critic):
            target.requires_grad_(False)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.reward_critic_optimizer = optim.Adam(
            self.reward_critic.parameters(),
            lr=reward_critic_lr,
        )
        self.cost_critic_optimizer = optim.Adam(self.cost_critic.parameters(), lr=cost_critic_lr)

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

    def step(self, augmented_obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Generate only the terminal action used by the environment."""
        with torch.no_grad():
            return self.flow(augmented_obs, deterministic=deterministic).action

    @torch.no_grad()
    def polyak_update(self, coefficient: float) -> None:
        """Soft-update reward and distributional cost targets."""
        if not 0.0 < coefficient <= 1.0:
            raise ValueError('Polyak coefficient must lie in (0, 1]')
        pairs = (
            (self.target_reward_critic, self.reward_critic),
            (self.target_cost_critic, self.cost_critic),
        )
        for target, online in pairs:
            for target_parameter, online_parameter in zip(
                target.parameters(),
                online.parameters(),
            ):
                target_parameter.lerp_(online_parameter, coefficient)

    @property
    def optimizers(self) -> dict[str, optim.Optimizer]:
        """Return every network optimizer for checkpointing."""
        return {
            'actor': self.actor_optimizer,
            'reward_critic': self.reward_critic_optimizer,
            'cost_critic': self.cost_critic_optimizer,
        }

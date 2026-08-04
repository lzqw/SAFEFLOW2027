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
"""Bounded action-space flow actor used by STAR."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from omnisafe.typing import Activation, InitFunction
from omnisafe.utils.model import build_mlp_network


@dataclass
class FlowOutput:
    """A complete differentiable action-space flow trajectory."""

    action: torch.Tensor
    points: torch.Tensor
    velocity_points: torch.Tensor
    velocities: torch.Tensor
    times: torch.Tensor
    kinetic_energy: torch.Tensor


class FlowActor(nn.Module):
    """Integrate a learned velocity field directly inside the bounded action space.

    STAR uses a midpoint solver with a fixed number of function evaluations. The base action and
    every intermediate point are kept in ``[-1, 1]``. No latent flow, AMP, compilation, or
    second-order differentiation is used here.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list[int],
        num_steps: int = 4,
        max_velocity: float = 2.0,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        """Initialize the bounded velocity field."""
        super().__init__()
        if obs_dim <= 0 or act_dim <= 0:
            raise ValueError('observation and action dimensions must be positive')
        if num_steps <= 0 or max_velocity <= 0:
            raise ValueError('flow steps and maximum velocity must be positive')
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_steps = num_steps
        self.max_velocity = float(max_velocity)
        self.velocity_net = build_mlp_network(
            sizes=[obs_dim + act_dim + 1, *hidden_sizes, act_dim],
            activation=activation,
            output_activation='tanh',
            weight_initialization_mode=weight_initialization_mode,
        )

    def velocity(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the total action velocity at a flow point."""
        if time.ndim == 1:
            time = time.unsqueeze(-1)
        inputs = torch.cat((obs, action, time), dim=-1)
        return self.max_velocity * self.velocity_net(inputs)

    def sample_base(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        base_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return caller-provided, deterministic, or uniform bounded base actions."""
        batch_shape = (*obs.shape[:-1], self.act_dim)
        if base_action is not None:
            if base_action.shape != batch_shape:
                raise ValueError(
                    f'base action has shape {base_action.shape}, expected {batch_shape}',
                )
            return base_action.to(device=obs.device, dtype=obs.dtype).clamp(-1.0, 1.0)
        if deterministic:
            return torch.zeros(batch_shape, device=obs.device, dtype=obs.dtype)
        return torch.empty(batch_shape, device=obs.device, dtype=obs.dtype).uniform_(-1.0, 1.0)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        base_action: torch.Tensor | None = None,
    ) -> FlowOutput:
        """Integrate the midpoint ODE and retain all intermediate action-space points."""
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f'expected observation width {self.obs_dim}, got {obs.shape[-1]}')
        action = self.sample_base(obs, deterministic=deterministic, base_action=base_action)
        points = [action]
        velocity_points: list[torch.Tensor] = []
        velocities: list[torch.Tensor] = []
        times: list[torch.Tensor] = []
        step_size = 1.0 / self.num_steps

        for index in range(self.num_steps):
            start_time = torch.full(
                (*obs.shape[:-1], 1),
                index * step_size,
                device=obs.device,
                dtype=obs.dtype,
            )
            start_velocity = self.velocity(obs, action, start_time)
            midpoint = (action + 0.5 * step_size * start_velocity).clamp(-1.0, 1.0)
            midpoint_time = start_time + 0.5 * step_size
            midpoint_velocity = self.velocity(obs, midpoint, midpoint_time)
            action = (action + step_size * midpoint_velocity).clamp(-1.0, 1.0)
            velocity_points.append(midpoint)
            velocities.append(midpoint_velocity)
            times.append(midpoint_time.squeeze(-1))
            points.append(action)

        velocity_tensor = torch.stack(velocities, dim=-2)
        kinetic = 0.5 * step_size * velocity_tensor.square().sum(dim=-1).sum(dim=-1)
        return FlowOutput(
            action=action,
            points=torch.stack(points, dim=-2),
            velocity_points=torch.stack(velocity_points, dim=-2),
            velocities=velocity_tensor,
            times=torch.stack(times, dim=-1),
            kinetic_energy=kinetic,
        )

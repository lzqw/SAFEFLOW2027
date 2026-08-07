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

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from omnisafe.typing import Activation, InitFunction
from omnisafe.utils.model import build_mlp_network


@dataclass
class FlowReference:
    """One locally constant stochastic reference evaluated at a flow midpoint."""

    drift: torch.Tensor
    noise_transform: torch.Tensor
    normal: torch.Tensor
    risk_ucb: torch.Tensor
    margin: torch.Tensor
    reliability: torch.Tensor
    boundary_reliability: torch.Tensor
    gradient_reliability: torch.Tensor
    ensemble_reliability: torch.Tensor
    calibration_reliability: torch.Tensor
    normal_variance: torch.Tensor
    tangent_variance: torch.Tensor


FlowReferenceFunction = Callable[[torch.Tensor], FlowReference]


@dataclass
class FlowReferenceTrace:
    """Detached reference quantities actually used along one stochastic trajectory."""

    drifts: torch.Tensor
    noise_transforms: torch.Tensor
    noises: torch.Tensor
    normals: torch.Tensor
    risk_ucb: torch.Tensor
    margins: torch.Tensor
    reliability: torch.Tensor
    boundary_reliability: torch.Tensor
    gradient_reliability: torch.Tensor
    ensemble_reliability: torch.Tensor
    calibration_reliability: torch.Tensor
    normal_variance: torch.Tensor
    tangent_variance: torch.Tensor
    normal_energy: torch.Tensor
    tangent_energy: torch.Tensor


@dataclass
class FlowOutput:
    """A complete differentiable action-space flow trajectory."""

    action: torch.Tensor
    points: torch.Tensor
    proposals: torch.Tensor
    velocity_points: torch.Tensor
    raw_velocities: torch.Tensor
    velocities: torch.Tensor
    times: torch.Tensor
    kinetic_energy: torch.Tensor
    reference: FlowReferenceTrace | None = None


def rectify_outward_velocity(
    velocity: torch.Tensor,
    normal: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    """Delete only the reliability-weighted outward normal velocity component."""
    if velocity.shape != normal.shape:
        raise ValueError('velocity and safety normal must have identical shapes')
    if reliability.shape != velocity.shape[:-1]:
        raise ValueError('rectification reliability must match the velocity batch shape')
    outward = (velocity * normal).sum(dim=-1).clamp_min(0.0)
    return velocity - reliability.unsqueeze(-1) * outward.unsqueeze(-1) * normal


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

    @staticmethod
    def _validate_reference(reference: FlowReference, action: torch.Tensor) -> None:
        """Reject malformed reference shapes without synchronizing the GPU hot path."""
        scalar_shape = action.shape[:-1]
        matrix_shape = (*action.shape, action.shape[-1])
        expected = {
            'drift': action.shape,
            'noise_transform': matrix_shape,
            'normal': action.shape,
            'risk_ucb': scalar_shape,
            'margin': scalar_shape,
            'reliability': scalar_shape,
            'boundary_reliability': scalar_shape,
            'gradient_reliability': scalar_shape,
            'ensemble_reliability': scalar_shape,
            'calibration_reliability': scalar_shape,
            'normal_variance': scalar_shape,
            'tangent_variance': scalar_shape,
        }
        for name, shape in expected.items():
            value = getattr(reference, name)
            if value.shape != shape:
                raise ValueError(f'flow reference {name} has shape {value.shape}, expected {shape}')

    def forward(  # pylint: disable=too-many-locals,too-many-statements
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        base_action: torch.Tensor | None = None,
        reference: FlowReferenceFunction | None = None,
        flow_noise: torch.Tensor | None = None,
        outward_rectification_enabled: bool = False,
    ) -> FlowOutput:
        """Integrate a midpoint ODE or midpoint Euler--Maruyama stochastic flow.

        The learned velocity remains the controlled residual.  A supplied reference contributes
        detached drift and covariance at every midpoint, while the returned kinetic energy is the
        residual velocity weighted by that covariance's inverse normal/tangent variances.
        """
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f'expected observation width {self.obs_dim}, got {obs.shape[-1]}')
        noise_shape = (*obs.shape[:-1], self.num_steps, self.act_dim)
        if reference is None and flow_noise is not None:
            raise ValueError('flow noise requires a stochastic reference')
        if reference is None and outward_rectification_enabled:
            raise ValueError('outward rectification requires a stochastic safety reference')
        if flow_noise is not None and flow_noise.shape != noise_shape:
            raise ValueError(f'flow noise has shape {flow_noise.shape}, expected {noise_shape}')
        if reference is not None:
            if flow_noise is None:
                flow_noise = (
                    torch.zeros(noise_shape, device=obs.device, dtype=obs.dtype)
                    if deterministic
                    else torch.randn(noise_shape, device=obs.device, dtype=obs.dtype)
                )
            else:
                flow_noise = flow_noise.to(device=obs.device, dtype=obs.dtype)
        action = self.sample_base(obs, deterministic=deterministic, base_action=base_action)
        points = [action]
        proposals: list[torch.Tensor] = []
        velocity_points: list[torch.Tensor] = []
        raw_velocities: list[torch.Tensor] = []
        velocities: list[torch.Tensor] = []
        times: list[torch.Tensor] = []
        references: list[FlowReference] = []
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
            raw_midpoint_velocity = self.velocity(obs, midpoint, midpoint_time)
            if reference is None:
                midpoint_velocity = raw_midpoint_velocity
                proposal = action + step_size * midpoint_velocity
                action = proposal.clamp(-1.0, 1.0)
            else:
                local_reference = reference(midpoint)
                self._validate_reference(local_reference, midpoint)
                midpoint_velocity = (
                    rectify_outward_velocity(
                        raw_midpoint_velocity,
                        local_reference.normal,
                        local_reference.reliability,
                    )
                    if outward_rectification_enabled
                    else raw_midpoint_velocity
                )
                assert flow_noise is not None
                increment = torch.matmul(
                    local_reference.noise_transform,
                    flow_noise[..., index, :].unsqueeze(-1),
                ).squeeze(-1)
                proposal = (
                    action
                    + step_size * (midpoint_velocity + local_reference.drift)
                    + math.sqrt(step_size) * increment
                )
                action = proposal.clamp(-1.0, 1.0)
                references.append(local_reference)
            proposals.append(proposal)
            velocity_points.append(midpoint)
            raw_velocities.append(raw_midpoint_velocity)
            velocities.append(midpoint_velocity)
            times.append(midpoint_time.squeeze(-1))
            points.append(action)

        velocity_tensor = torch.stack(velocities, dim=-2)
        reference_trace = None
        if reference is None:
            kinetic = 0.5 * step_size * velocity_tensor.square().sum(dim=-1).sum(dim=-1)
        else:
            normals = torch.stack([item.normal for item in references], dim=-2)
            normal_variance = torch.stack(
                [item.normal_variance for item in references],
                dim=-1,
            )
            tangent_variance = torch.stack(
                [item.tangent_variance for item in references],
                dim=-1,
            )
            normal_velocity = (velocity_tensor * normals).sum(dim=-1)
            tangent_squared = (
                velocity_tensor.square().sum(dim=-1) - normal_velocity.square()
            ).clamp_min(0.0)
            normal_energy = 0.5 * step_size * normal_velocity.square() / normal_variance
            tangent_energy = 0.5 * step_size * tangent_squared / tangent_variance
            kinetic = (normal_energy + tangent_energy).sum(dim=-1)
            assert flow_noise is not None
            reference_trace = FlowReferenceTrace(
                drifts=torch.stack([item.drift for item in references], dim=-2),
                noise_transforms=torch.stack(
                    [item.noise_transform for item in references],
                    dim=-3,
                ),
                noises=flow_noise,
                normals=normals,
                risk_ucb=torch.stack([item.risk_ucb for item in references], dim=-1),
                margins=torch.stack([item.margin for item in references], dim=-1),
                reliability=torch.stack([item.reliability for item in references], dim=-1),
                boundary_reliability=torch.stack(
                    [item.boundary_reliability for item in references],
                    dim=-1,
                ),
                gradient_reliability=torch.stack(
                    [item.gradient_reliability for item in references],
                    dim=-1,
                ),
                ensemble_reliability=torch.stack(
                    [item.ensemble_reliability for item in references],
                    dim=-1,
                ),
                calibration_reliability=torch.stack(
                    [item.calibration_reliability for item in references],
                    dim=-1,
                ),
                normal_variance=normal_variance,
                tangent_variance=tangent_variance,
                normal_energy=normal_energy,
                tangent_energy=tangent_energy,
            )
        return FlowOutput(
            action=action,
            points=torch.stack(points, dim=-2),
            proposals=torch.stack(proposals, dim=-2),
            velocity_points=torch.stack(velocity_points, dim=-2),
            raw_velocities=torch.stack(raw_velocities, dim=-2),
            velocities=velocity_tensor,
            times=torch.stack(times, dim=-1),
            kinetic_energy=kinetic,
            reference=reference_trace,
        )

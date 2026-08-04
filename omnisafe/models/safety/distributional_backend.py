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
"""First-order route-A safety geometry and residual reference energy."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from omnisafe.models.actor.flow_actor import FlowOutput
from omnisafe.models.critic.distributional_cost_critic import DistributionalCostCritic


@dataclass
class SafetyGeometry:
    """Differentiable energy plus detached first-order safety diagnostics."""

    energy: torch.Tensor
    per_sample_energy: torch.Tensor
    gate: torch.Tensor
    candidate_gate: torch.Tensor
    normal_variance: torch.Tensor
    tangent_variance: torch.Tensor
    normals: torch.Tensor
    risk: torch.Tensor
    gradient_norm: torch.Tensor
    consistency: torch.Tensor
    normalized_jvp_value: torch.Tensor
    raw_jvp_value: torch.Tensor
    normal_velocity: torch.Tensor
    dangerous_normal_velocity: torch.Tensor
    tangent_energy: torch.Tensor
    normal_energy: torch.Tensor


def normalized_jvp(
    gradient: torch.Tensor,
    vector: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Return scale-stable squared directional velocity along a risk gradient."""
    numerator = (gradient * vector).sum(dim=-1)
    denominator = gradient.norm(dim=-1).clamp_min(epsilon)
    return (numerator / denominator).square()


def boundary_gate(risk: torch.Tensor, boundary: float, bandwidth: float) -> torch.Tensor:
    """Return the Gaussian near-boundary gate."""
    if bandwidth <= 0:
        raise ValueError('boundary bandwidth must be positive')
    return torch.exp(-0.5 * ((risk - boundary) / bandwidth).square())


def directional_variances(
    gate: torch.Tensor,
    tangent_variance: float,
    contraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tangent variance and monotonically contracted normal variance."""
    if tangent_variance <= 0 or contraction < 0:
        raise ValueError('tangent variance must be positive and contraction non-negative')
    tangent = torch.full_like(gate, tangent_variance)
    normal = tangent / (1.0 + contraction * gate.clamp_min(0.0))
    return tangent, normal


def _normal_consistency(head_gradients: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Return mean pairwise orientation agreement in ``[0, 1]``."""
    head_count = head_gradients.shape[0]
    if head_count == 1:
        return torch.ones(head_gradients.shape[1], device=head_gradients.device)
    unit = head_gradients / head_gradients.norm(dim=-1, keepdim=True).clamp_min(epsilon)
    agreements = []
    for first in range(head_count):
        for second in range(first + 1, head_count):
            cosine = (unit[first] * unit[second]).sum(dim=-1).clamp(-1.0, 1.0)
            agreements.append(0.5 * (1.0 + cosine))
    return torch.stack(agreements, dim=0).mean(dim=0)


def _risk_action_gradients(
    critic: DistributionalCostCritic,
    obs: torch.Tensor,
    action: torch.Tensor,
    threshold: torch.Tensor,
    risk_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate head risks and detached first action gradients with frozen critic weights."""
    parameter_flags = [parameter.requires_grad for parameter in critic.parameters()]
    critic.requires_grad_(False)
    try:
        head_risk = critic.exceedance_probability(
            obs,
            action,
            threshold,
            risk_temperature,
        )
        gradients = [
            torch.autograd.grad(
                head_risk[index].sum(),
                action,
                retain_graph=True,
                create_graph=False,
            )[0].detach()
            for index in range(head_risk.shape[0])
        ]
    finally:
        for parameter, flag in zip(critic.parameters(), parameter_flags):
            parameter.requires_grad_(flag)
    return head_risk.detach(), torch.stack(gradients, dim=0)


# pylint: disable-next=too-many-arguments,too-many-locals
def compute_safety_geometry(
    critic: DistributionalCostCritic,
    augmented_obs: torch.Tensor,
    risk_threshold: torch.Tensor,
    flow: FlowOutput,
    enabled: bool,
    late_time: float = 0.5,
    risk_temperature: float = 1.0,
    boundary: float = 0.5,
    boundary_bandwidth: float = 0.2,
    minimum_gradient_norm: float = 1e-5,
    minimum_consistency: float = 0.0,
    tangent_variance: float = 1.0,
    normal_contraction: float = 9.0,
    reference_strength: float = 1.0,
    reference_temperature: float = 0.25,
    epsilon: float = 1e-8,
) -> SafetyGeometry:
    """Build detached geometry while preserving first-order actor velocity gradients."""
    if not 0.0 <= late_time <= 1.0:
        raise ValueError('late flow time must lie in [0, 1]')
    if reference_temperature <= 0:
        raise ValueError('reference temperature must be positive')
    batch_size, flow_steps, action_dim = flow.velocities.shape
    late_mask = flow.times[0] >= late_time
    if not bool(late_mask.any()):
        raise ValueError('late flow threshold selects no integration point')
    selected_count = int(late_mask.sum().item())
    selected_points = flow.velocity_points[:, late_mask].reshape(-1, action_dim)
    selected_velocity = flow.velocities[:, late_mask].reshape(-1, action_dim)
    repeated_obs = (
        augmented_obs.unsqueeze(1)
        .expand(-1, selected_count, -1)
        .reshape(
            -1,
            augmented_obs.shape[-1],
        )
    )
    repeated_threshold = risk_threshold.reshape(-1, 1).expand(-1, selected_count).reshape(-1)
    head_risk, head_gradients = _risk_action_gradients(
        critic,
        repeated_obs,
        selected_points,
        repeated_threshold,
        risk_temperature,
    )
    risk = head_risk.mean(dim=0)
    gradient = head_gradients.mean(dim=0)
    gradient_norm = gradient.norm(dim=-1)
    normals = (gradient / gradient_norm.unsqueeze(-1).clamp_min(epsilon)).detach()
    consistency = _normal_consistency(head_gradients, epsilon).detach()
    valid = (gradient_norm >= minimum_gradient_norm) & (consistency >= minimum_consistency)
    candidate = (
        boundary_gate(risk, boundary, boundary_bandwidth) * valid.to(risk) * consistency
    ).detach()
    gate = candidate if enabled else torch.zeros_like(candidate)
    tangent_var, normal_var = directional_variances(
        gate,
        tangent_variance,
        normal_contraction,
    )

    risk_energy_gradient = (
        torch.sigmoid((risk - boundary) / reference_temperature).unsqueeze(-1)
        * gradient
        / reference_temperature
    )
    gradient_normal = (risk_energy_gradient * normals).sum(dim=-1, keepdim=True) * normals
    gradient_tangent = risk_energy_gradient - gradient_normal
    covariance_gradient = (
        tangent_var.unsqueeze(-1) * gradient_tangent
        + normal_var.unsqueeze(
            -1,
        )
        * gradient_normal
    )
    reference_drift = (-0.5 * reference_strength * covariance_gradient).detach()
    residual = selected_velocity - (reference_drift if enabled else 0.0)
    normal_component = (residual * normals).sum(dim=-1)
    tangent_squared = (residual.square().sum(dim=-1) - normal_component.square()).clamp_min(0.0)
    normal_term = normal_component.square() / normal_var
    tangent_term = tangent_squared / tangent_var

    step_size = 1.0 / flow_steps
    per_point_energy = 0.5 * step_size * flow.velocities.square().sum(dim=-1)
    selected_energy = 0.5 * step_size * (normal_term + tangent_term)
    per_point_energy = per_point_energy.clone()
    per_point_energy[:, late_mask] = selected_energy.reshape(batch_size, selected_count)
    per_sample_energy = per_point_energy.sum(dim=-1)

    normal_velocity = (selected_velocity * normals).sum(dim=-1)
    return SafetyGeometry(
        energy=per_sample_energy.mean(),
        per_sample_energy=per_sample_energy,
        gate=gate,
        candidate_gate=candidate,
        normal_variance=normal_var,
        tangent_variance=tangent_var,
        normals=normals,
        risk=risk,
        gradient_norm=gradient_norm.detach(),
        consistency=consistency,
        normalized_jvp_value=normalized_jvp(gradient, residual, epsilon),
        raw_jvp_value=(gradient * residual).sum(dim=-1).square(),
        normal_velocity=normal_velocity,
        dangerous_normal_velocity=normal_velocity.clamp_min(0.0),
        tangent_energy=0.5 * step_size * tangent_term,
        normal_energy=0.5 * step_size * normal_term,
    )

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

import math
from dataclasses import dataclass

import torch

from omnisafe.models.actor.flow_actor import FlowOutput, FlowReference
from omnisafe.models.critic.distributional_cost_critic import DistributionalCostCritic
from omnisafe.models.critic.violation_probability_critic import ViolationProbabilityCritic


@dataclass
class SafetyGeometry:  # pylint: disable=too-many-instance-attributes
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


def one_sided_safety_weight(margin: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return a continuous weight that rises from the safe side through danger."""
    if temperature <= 0:
        raise ValueError('one-sided safety temperature must be positive')
    return torch.sigmoid(margin / temperature)


def calibration_reliability(
    auroc: float | None,
    brier_score: float | None,
    *,
    auroc_full: float = 0.75,
    brier_score_scale: float = 0.25,
) -> float:
    """Map held-out calibration evidence continuously to ``[0, 1]``.

    Missing evidence and chance-or-worse discrimination map to zero.  Otherwise Brier error causes
    a smooth exponential decay, so imperfect calibration weakens anisotropy without recreating the
    old positive-skill binary gate.
    """
    if auroc_full <= 0.5 or brier_score_scale <= 0:
        raise ValueError('full-reliability calibration scales are invalid')
    if auroc is None or brier_score is None:
        return 0.0
    evidence = torch.tensor((auroc, brier_score), dtype=torch.float64)
    if not bool(torch.isfinite(evidence).all()):
        return 0.0
    if float(brier_score) < 0.0:
        return 0.0
    discrimination = min(max((float(auroc) - 0.5) / (auroc_full - 0.5), 0.0), 1.0)
    calibration = math.exp(-float(brier_score) / brier_score_scale)
    return float((discrimination * calibration) ** 0.5)


def fixed_trace_variances(
    reliability: torch.Tensor,
    action_dim: int,
    reference_variance: float,
    normal_variance_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normal/tangent variances with exactly fixed total trace."""
    if action_dim < 2:
        raise ValueError('safety anisotropy requires at least two action dimensions')
    if reference_variance <= 0 or not 0 < normal_variance_ratio <= 1:
        raise ValueError('stochastic-reference variances are invalid')
    bounded = reliability.clamp(0.0, 1.0)
    minimum = reference_variance * normal_variance_ratio
    normal = (1.0 - bounded) * reference_variance + bounded * minimum
    tangent = (action_dim * reference_variance - normal) / (action_dim - 1)
    return normal, tangent


def deterministic_random_normal(action: torch.Tensor, seed: int) -> torch.Tensor:
    """Return a reproducible pseudo-random unit field independent of critic gradients.

    This is a mechanism-control direction, not policy randomness.  It is a deterministic
    function of the detached action and a preregistered seed, so checkpoint evaluation does
    not depend on an uncheckpointed generator state.
    """
    if action.shape[-1] < 2:
        raise ValueError('random anisotropy requires at least two action dimensions')
    if seed < 0:
        raise ValueError('random anisotropy seed must be non-negative')
    detached = action.detach()
    index = torch.arange(
        1,
        detached.shape[-1] + 1,
        device=detached.device,
        dtype=detached.dtype,
    )
    weights = torch.sin(index * float(seed + 1) * 0.017453292519943295)
    phase = (detached * weights).sum(dim=-1)
    phase = 2.0 * torch.pi * torch.frac(
        torch.sin(phase + float(seed + 1) * 0.123456789) * 43_758.5453,
    )
    if detached.shape[-1] == 2:
        return torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1)
    offsets = index * (2.0 * torch.pi / detached.shape[-1])
    normal = torch.sin(phase.unsqueeze(-1) + offsets)
    return normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def gaussian_directional_entropies(
    normal_variance: torch.Tensor,
    tangent_variance: torch.Tensor,
    action_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Gaussian plug-in entropy in normal and tangent subspaces."""
    if action_dim < 2:
        raise ValueError('directional entropy requires at least two action dimensions')
    constant = torch.as_tensor(2.0 * torch.pi * torch.e, device=normal_variance.device)
    normal_entropy = 0.5 * torch.log(constant * normal_variance)
    tangent_entropy = 0.5 * (action_dim - 1) * torch.log(constant * tangent_variance)
    return normal_entropy, tangent_entropy


def normalized_directional_loss(
    velocity: torch.Tensor,
    normal: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the detached-geometry outward-direction ratio and its two moments."""
    if velocity.shape != normal.shape or weight.shape != velocity.shape[:-1]:
        raise ValueError('directional loss tensors have incompatible shapes')
    if epsilon <= 0:
        raise ValueError('directional loss epsilon must be positive')
    detached_normal = normal.detach()
    detached_weight = weight.detach().clamp_min(0.0)
    outward = (velocity * detached_normal).sum(dim=-1).clamp_min(0.0)
    numerator = (detached_weight * outward.square()).mean()
    denominator = (detached_weight * velocity.square().sum(dim=-1)).mean()
    return numerator / (denominator + epsilon), numerator, denominator


def safety_aware_directional_loss(
    velocity: torch.Tensor,
    normal: torch.Tensor,
    reliability: torch.Tensor,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the prevalence-aware outward loss with stop-gradient speed normalization."""
    if velocity.shape != normal.shape or reliability.shape != velocity.shape[:-1]:
        raise ValueError('safety-aware directional loss tensors have incompatible shapes')
    if epsilon <= 0:
        raise ValueError('safety-aware directional loss epsilon must be positive')
    detached_normal = normal.detach()
    detached_reliability = reliability.detach().clamp_min(0.0)
    outward = (velocity * detached_normal).sum(dim=-1).clamp_min(0.0)
    numerator = (detached_reliability * outward.square()).mean()
    denominator = velocity.square().sum(dim=-1).mean().detach()
    return numerator / (denominator + epsilon), numerator, denominator


# pylint: disable-next=too-many-arguments,too-many-locals
def compute_cost_stochastic_reference(
    critic: DistributionalCostCritic | None,
    augmented_obs: torch.Tensor,
    action: torch.Tensor,
    remaining_budget_ratio: torch.Tensor,
    *,
    anisotropy_enabled: bool,
    safety_aware_geometry: bool = False,
    cost_normalization_scale: float,
    boundary_bandwidth: float = 0.1,
    reference_variance: float = 0.04,
    normal_variance_ratio: float = 0.1,
    gradient_scale: float = 0.005,
    ensemble_scale: float = 0.05,
    random_anisotropy_seed: int | None = None,
    epsilon: float = 1e-8,
) -> FlowReference:
    """Build a normalized-Q-cost reference without probability-critic policy access.

    ``cost_normalization_scale`` is the episode budget for budgeted Point tasks and one for
    binary terminal-failure tasks.  Keeping this scale explicit prevents a formal zero failure
    budget from being mistaken for a divisor while leaving the learned cost critic unchanged.
    """
    if cost_normalization_scale <= 0.0:
        raise ValueError('cost geometry requires a positive normalization scale')
    if min(boundary_bandwidth, gradient_scale, ensemble_scale, epsilon) <= 0.0:
        raise ValueError('cost-geometry scales must be positive')
    if remaining_budget_ratio.shape != action.shape[:-1]:
        raise ValueError('remaining-budget ratio must match the action batch shape')
    action_dim = action.shape[-1]
    scalar_shape = action.shape[:-1]
    zeros = torch.zeros(scalar_shape, device=action.device, dtype=action.dtype)
    ones = torch.ones_like(zeros)
    normal = torch.zeros_like(action)
    cost_score = zeros
    margin = zeros
    safety_band_weight = zeros
    gradient_reliability = zeros
    gradient_norm = zeros
    ensemble_reliability = zeros

    if anisotropy_enabled:
        if critic is None:
            raise ValueError('Q-cost anisotropy requires a distributional cost critic')
        if action_dim < 2:
            raise ValueError('Q-cost anisotropy requires at least two action dimensions')
        parameter_flags = [parameter.requires_grad for parameter in critic.parameters()]
        critic.requires_grad_(False)
        try:
            with torch.enable_grad():
                local_action = action.detach().requires_grad_(True)
                head_cost = critic.expected_cost(augmented_obs.detach(), local_action)
                mean_cost = head_cost.mean(dim=0)
                cost_score_graph = mean_cost / cost_normalization_scale
                gradient = torch.autograd.grad(cost_score_graph.sum(), local_action)[0]
        finally:
            for parameter, flag in zip(critic.parameters(), parameter_flags):
                parameter.requires_grad_(flag)
        cost_score = cost_score_graph.detach()
        margin = cost_score - remaining_budget_ratio.detach()
        gradient = gradient.detach()
        gradient_norm = gradient.norm(dim=-1)
        normal = gradient / gradient_norm.unsqueeze(-1).clamp_min(epsilon)
        if safety_aware_geometry:
            safety_band_weight = one_sided_safety_weight(margin, boundary_bandwidth)
        else:
            safety_band_weight = torch.exp(
                -0.5 * (margin / boundary_bandwidth).square(),
            )
        gradient_reliability = gradient_norm / (gradient_norm + gradient_scale)
        normalized_disagreement = (
            head_cost.std(dim=0, unbiased=False).detach() / cost_normalization_scale
        )
        ensemble_reliability = torch.exp(-normalized_disagreement / ensemble_scale)

    reliability = (safety_band_weight * gradient_reliability * ensemble_reliability).clamp(0.0, 1.0)
    if anisotropy_enabled:
        normal_variance, tangent_variance = fixed_trace_variances(
            reliability,
            action_dim,
            reference_variance,
            normal_variance_ratio,
        )
    else:
        normal_variance = torch.full_like(zeros, reference_variance)
        tangent_variance = torch.full_like(zeros, reference_variance)

    identity = torch.eye(action_dim, device=action.device, dtype=action.dtype).expand(
        *scalar_shape,
        action_dim,
        action_dim,
    )
    covariance_normal = normal
    if anisotropy_enabled and random_anisotropy_seed is not None:
        covariance_normal = deterministic_random_normal(action, random_anisotropy_seed)
    normal_projection = covariance_normal.unsqueeze(-1) * covariance_normal.unsqueeze(-2)
    tangent_projection = identity - normal_projection
    noise_transform = (
        tangent_variance.sqrt().unsqueeze(-1).unsqueeze(-1) * tangent_projection
        + normal_variance.sqrt().unsqueeze(-1).unsqueeze(-1) * normal_projection
    )
    reported_normal_variance = normal_variance
    reported_tangent_variance = tangent_variance
    if anisotropy_enabled and random_anisotropy_seed is not None:
        covariance = noise_transform @ noise_transform.transpose(-1, -2)
        reported_normal_variance = torch.einsum(
            '...i,...ij,...j->...',
            normal,
            covariance,
            normal,
        )
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        reported_tangent_variance = (
            trace - reported_normal_variance
        ) / (action_dim - 1)
    return FlowReference(
        drift=torch.zeros_like(action),
        noise_transform=noise_transform.detach(),
        normal=normal.detach(),
        risk_ucb=cost_score.detach(),
        margin=margin.detach(),
        reliability=reliability.detach(),
        boundary_reliability=safety_band_weight.detach(),
        gradient_reliability=gradient_reliability.detach(),
        gradient_norm=gradient_norm.detach(),
        ensemble_reliability=ensemble_reliability.detach(),
        calibration_reliability=ones,
        normal_variance=reported_normal_variance.detach(),
        tangent_variance=reported_tangent_variance.detach(),
    )


# pylint: disable-next=too-many-arguments,too-many-locals
def compute_stochastic_reference(
    critic: ViolationProbabilityCritic | None,
    augmented_obs: torch.Tensor,
    action: torch.Tensor,
    *,
    anisotropy_enabled: bool,
    inward_drift_enabled: bool,
    calibration: float,
    risk_tolerance: float = 0.2,
    boundary_bandwidth: float = 0.1,
    risk_ucb_beta: float = 0.5,
    reference_variance: float = 0.04,
    normal_variance_ratio: float = 0.1,
    gradient_scale: float = 0.005,
    ensemble_scale: float = 0.05,
    drift_strength: float = 0.2,
    drift_temperature: float = 0.05,
    epsilon: float = 1e-8,
) -> FlowReference:
    """Build the continuous safety-conditioned Brownian reference at one midpoint.

    Risk geometry is deliberately detached.  Actor gradients remain first order through the
    residual velocity and trajectory, without differentiating through critic action gradients.
    """
    if not 0 < risk_tolerance < 0.5:
        raise ValueError('safe-risk tolerance must lie in (0, 0.5)')
    if min(boundary_bandwidth, gradient_scale, ensemble_scale, drift_temperature) <= 0:
        raise ValueError('stochastic-reference reliability scales must be positive')
    if drift_strength < 0 or not 0 <= calibration <= 1:
        raise ValueError('stochastic-reference drift/calibration values are invalid')
    if inward_drift_enabled and not anisotropy_enabled:
        raise ValueError('inward safety drift requires safety anisotropy')
    action_dim = action.shape[-1]
    scalar_shape = action.shape[:-1]
    zeros = torch.zeros(scalar_shape, device=action.device, dtype=action.dtype)
    normal = torch.zeros_like(action)
    risk_ucb = zeros
    margin = zeros
    boundary_reliability = zeros
    gradient_reliability = zeros
    ensemble_reliability = zeros
    gradient_norm = zeros

    if anisotropy_enabled:
        if critic is None:
            raise ValueError('safety anisotropy requires a violation-probability critic')
        if action_dim < 2:
            raise ValueError('safety anisotropy requires at least two action dimensions')
        parameter_flags = [parameter.requires_grad for parameter in critic.parameters()]
        critic.requires_grad_(False)
        try:
            with torch.enable_grad():
                local_action = action.detach().requires_grad_(True)
                head_probability = critic.probabilities(augmented_obs.detach(), local_action)
                mean_probability = head_probability.mean(dim=0)
                disagreement = head_probability.std(dim=0, unbiased=False)
                differentiable_ucb = (mean_probability + risk_ucb_beta * disagreement).clamp(
                    0.0,
                    1.0,
                )
                gradient = torch.autograd.grad(differentiable_ucb.sum(), local_action)[0]
        finally:
            for parameter, flag in zip(critic.parameters(), parameter_flags):
                parameter.requires_grad_(flag)
        risk_ucb = differentiable_ucb.detach()
        margin = risk_ucb - risk_tolerance
        gradient = gradient.detach()
        gradient_norm = gradient.norm(dim=-1)
        normal = gradient / gradient_norm.unsqueeze(-1).clamp_min(epsilon)
        boundary_reliability = boundary_gate(risk_ucb, risk_tolerance, boundary_bandwidth)
        gradient_reliability = gradient_norm / (gradient_norm + gradient_scale)
        ensemble_reliability = torch.exp(-disagreement.detach() / ensemble_scale)

    calibration_tensor = torch.full_like(zeros, calibration)
    reliability = (
        boundary_reliability * gradient_reliability * ensemble_reliability * calibration_tensor
    ).clamp(0.0, 1.0)
    if anisotropy_enabled:
        normal_variance, tangent_variance = fixed_trace_variances(
            reliability,
            action_dim,
            reference_variance,
            normal_variance_ratio,
        )
    else:
        normal_variance = torch.full_like(zeros, reference_variance)
        tangent_variance = torch.full_like(zeros, reference_variance)

    identity = torch.eye(action_dim, device=action.device, dtype=action.dtype).expand(
        *scalar_shape,
        action_dim,
        action_dim,
    )
    normal_projection = normal.unsqueeze(-1) * normal.unsqueeze(-2)
    tangent_projection = identity - normal_projection
    noise_transform = (
        tangent_variance.sqrt().unsqueeze(-1).unsqueeze(-1) * tangent_projection
        + normal_variance.sqrt().unsqueeze(-1).unsqueeze(-1) * normal_projection
    )
    drift = torch.zeros_like(action)
    if inward_drift_enabled:
        drift = (
            -drift_strength
            * reliability.unsqueeze(-1)
            * torch.sigmoid(margin / drift_temperature).unsqueeze(-1)
            * normal
        )
    return FlowReference(
        drift=drift.detach(),
        noise_transform=noise_transform.detach(),
        normal=normal.detach(),
        risk_ucb=risk_ucb.detach(),
        margin=margin.detach(),
        reliability=reliability.detach(),
        boundary_reliability=boundary_reliability.detach(),
        gradient_reliability=gradient_reliability.detach(),
        gradient_norm=gradient_norm.detach(),
        ensemble_reliability=ensemble_reliability.detach(),
        calibration_reliability=calibration_tensor.detach(),
        normal_variance=normal_variance.detach(),
        tangent_variance=tangent_variance.detach(),
    )


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
    critic: ViolationProbabilityCritic,
    obs: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate head risks and detached first action gradients with frozen critic weights."""
    parameter_flags = [parameter.requires_grad for parameter in critic.parameters()]
    critic.requires_grad_(False)
    try:
        head_risk = critic.probabilities(obs, action)
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
    critic: ViolationProbabilityCritic,
    augmented_obs: torch.Tensor,
    flow: FlowOutput,
    enabled: bool,
    late_time: float = 0.5,
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
    head_risk, head_gradients = _risk_action_gradients(
        critic,
        repeated_obs,
        selected_points,
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

# Copyright 2026 OmniSafe Team. All Rights Reserved.
"""Causal correctness tests for the P4.3 safety-conditioned stochastic flow."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from omnisafe.common.star_checkpoint import capture_rng_state, restore_rng_state
from omnisafe.models.actor.flow_actor import FlowActor, FlowReference
from omnisafe.models.safety.distributional_backend import (
    calibration_reliability,
    compute_stochastic_reference,
    fixed_trace_variances,
    gaussian_directional_entropies,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'reports/star/p4_3_safety_anisotropy_preregistration.json'
PROTOCOL_SHA256 = 'a05cb7dcaaee95cdbb75e005371e07ea04cfb59e765c4f5d70b9772d8bd29430'


class AnalyticProbabilityCritic(nn.Module):
    """Two identical differentiable heads with a known action normal."""

    def probabilities(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del obs
        probability = torch.sigmoid(2.0 * action[..., 0])
        return torch.stack((probability, probability), dim=0)


def _constant_reference(
    action: torch.Tensor,
    *,
    normal_variance: float,
    tangent_variance: float,
) -> FlowReference:
    batch = action.shape[:-1]
    normal = torch.zeros_like(action)
    normal[..., 0] = 1.0
    normal_projection = normal.unsqueeze(-1) * normal.unsqueeze(-2)
    identity = torch.eye(action.shape[-1], dtype=action.dtype).expand(*batch, 2, 2)
    tangent_projection = identity - normal_projection
    noise_transform = (
        math.sqrt(tangent_variance) * tangent_projection
        + math.sqrt(normal_variance) * normal_projection
    )
    scalar = torch.zeros(batch, dtype=action.dtype)
    return FlowReference(
        drift=torch.zeros_like(action),
        noise_transform=noise_transform,
        normal=normal,
        risk_ucb=scalar,
        margin=scalar,
        reliability=scalar,
        boundary_reliability=scalar,
        gradient_reliability=scalar,
        gradient_norm=scalar,
        ensemble_reliability=scalar,
        calibration_reliability=scalar,
        normal_variance=torch.full_like(scalar, normal_variance),
        tangent_variance=torch.full_like(scalar, tangent_variance),
    )


def test_preregistration_is_immutable_and_all_arms_share_the_bellman_critic() -> None:
    payload = PROTOCOL.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == PROTOCOL_SHA256
    protocol = json.loads(payload)
    assert protocol['status'] == 'locked_before_v2_formal_training'
    assert protocol['protocol_version'] == 'p4.3-safety-anisotropy-v2'
    assert protocol['protocol_revision_provenance']['v1_formal_evidence_included'] is False
    assert protocol['method']['calibration_brier_score_scale'] == 0.25
    assert protocol['backbone']['bellman_probability_enabled'] is True
    assert protocol['seeds'] == [2027, 2028, 2029]
    assert protocol['total_steps'] == 50_000
    assert protocol['arms'] == {
        'S0': {
            'candidate_probing_enabled': False,
            'safety_anisotropy_enabled': False,
            'safety_inward_drift_enabled': False,
            'stochastic_flow_enabled': True,
        },
        'S1': {
            'candidate_probing_enabled': False,
            'safety_anisotropy_enabled': True,
            'safety_inward_drift_enabled': False,
            'stochastic_flow_enabled': True,
        },
        'S2': {
            'candidate_probing_enabled': False,
            'safety_anisotropy_enabled': True,
            'safety_inward_drift_enabled': True,
            'stochastic_flow_enabled': True,
        },
    }


def test_calibration_reliability_is_continuous_and_fails_isotropically() -> None:
    assert calibration_reliability(None, None) == 0.0
    assert calibration_reliability(float('nan'), 0.1) == 0.0
    assert calibration_reliability(0.9, -0.1) == 0.0
    assert calibration_reliability(0.5, 0.0) == 0.0
    partial = calibration_reliability(0.625, 0.25)
    assert math.isclose(partial, math.sqrt(0.5 * math.exp(-1.0)))
    assert calibration_reliability(0.75, 0.0) == 1.0


def test_fixed_trace_solution_has_more_entropy_than_matched_isotropic_low() -> None:
    reliability = torch.tensor([0.0, 0.25, 1.0])
    normal, tangent = fixed_trace_variances(reliability, 2, 0.04, 0.1)
    torch.testing.assert_close(normal + tangent, torch.full_like(normal, 0.08))
    assert normal[-1] == torch.tensor(0.004)
    assert tangent[-1] == torch.tensor(0.076)
    normal_entropy, tangent_entropy = gaussian_directional_entropies(normal, tangent, 2)
    isotropic_low = torch.log(2.0 * torch.pi * torch.e * normal)  # d=2 total entropy
    advantage = normal_entropy + tangent_entropy - isotropic_low
    assert advantage[0].abs() < 1e-6
    assert bool((advantage[1:] > 0).all())


def test_unreliable_critic_returns_exactly_to_isotropic_reference() -> None:
    obs = torch.zeros(3, 4)
    boundary_action = torch.full((3, 2), math.log(0.2 / 0.8) / 2.0)
    reference = compute_stochastic_reference(
        AnalyticProbabilityCritic(),
        obs,
        boundary_action,
        anisotropy_enabled=True,
        inward_drift_enabled=True,
        calibration=0.0,
        risk_ucb_beta=0.0,
    )
    torch.testing.assert_close(reference.reliability, torch.zeros(3))
    torch.testing.assert_close(reference.normal_variance, torch.full((3,), 0.04))
    torch.testing.assert_close(reference.tangent_variance, torch.full((3,), 0.04))
    torch.testing.assert_close(reference.drift, torch.zeros_like(boundary_action))


def test_reliable_boundary_contracts_normal_expands_tangent_and_drifts_inward() -> None:
    obs = torch.zeros(3, 4)
    boundary_value = math.log(0.2 / 0.8) / 2.0
    action = torch.zeros(3, 2)
    action[:, 0] = boundary_value
    reference = compute_stochastic_reference(
        AnalyticProbabilityCritic(),
        obs,
        action,
        anisotropy_enabled=True,
        inward_drift_enabled=True,
        calibration=1.0,
        risk_ucb_beta=0.0,
    )
    assert bool((reference.reliability > 0).all())
    assert bool((reference.normal_variance < 0.04).all())
    assert bool((reference.tangent_variance > 0.04).all())
    torch.testing.assert_close(
        reference.normal_variance + reference.tangent_variance,
        torch.full((3,), 0.08),
    )
    assert bool(((reference.drift * reference.normal).sum(dim=-1) < 0).all())


def test_euler_maruyama_noise_is_used_and_zero_residual_has_zero_energy() -> None:
    actor = FlowActor(obs_dim=3, act_dim=2, hidden_sizes=[8], num_steps=4, max_velocity=1.0)
    for parameter in actor.parameters():
        parameter.data.zero_()
    obs = torch.zeros(2, 3)
    noise = torch.full((2, 4, 2), 0.1)
    output = actor(
        obs,
        base_action=torch.zeros(2, 2),
        flow_noise=noise,
        reference=lambda action: _constant_reference(
            action,
            normal_variance=0.04,
            tangent_variance=0.04,
        ),
    )
    # Four increments: sqrt(1/4) * sqrt(0.04) * 0.1 = 0.01 each.
    torch.testing.assert_close(output.action, torch.full((2, 2), 0.04))
    torch.testing.assert_close(output.kinetic_energy, torch.zeros(2))


def test_actor_energy_is_residual_velocity_weighted_by_inverse_covariance() -> None:
    actor = FlowActor(obs_dim=3, act_dim=2, hidden_sizes=[8], num_steps=4, max_velocity=1.0)
    linear_layers = [module for module in actor.velocity_net if isinstance(module, nn.Linear)]
    for parameter in actor.parameters():
        parameter.data.zero_()
    target_velocity = torch.tensor([0.1, 0.2])
    linear_layers[-1].bias.data.copy_(torch.atanh(target_velocity))
    obs = torch.zeros(2, 3)
    output = actor(
        obs,
        base_action=torch.zeros(2, 2),
        flow_noise=torch.zeros(2, 4, 2),
        reference=lambda action: _constant_reference(
            action,
            normal_variance=0.01,
            tangent_variance=0.07,
        ),
    )
    expected = 0.5 * (0.1**2 / 0.01 + 0.2**2 / 0.07)
    torch.testing.assert_close(output.kinetic_energy, torch.full((2,), expected))
    output.kinetic_energy.mean().backward()
    assert all(parameter.grad is not None for parameter in actor.parameters())


def test_reference_gradient_does_not_create_second_order_critic_gradients() -> None:
    critic = AnalyticProbabilityCritic()
    actor = FlowActor(obs_dim=4, act_dim=2, hidden_sizes=[8], num_steps=2)
    obs = torch.zeros(5, 4)

    def reference(action: torch.Tensor) -> FlowReference:
        return compute_stochastic_reference(
            critic,
            obs,
            action,
            anisotropy_enabled=True,
            inward_drift_enabled=False,
            calibration=1.0,
            risk_ucb_beta=0.0,
        )

    output = actor(obs, reference=reference, flow_noise=torch.zeros(5, 2, 2))
    output.kinetic_energy.mean().backward()
    assert any(parameter.grad is not None for parameter in actor.parameters())
    assert not any(parameter.grad is not None for parameter in critic.parameters())


def test_checkpoint_rng_restores_internal_wiener_draws_exactly() -> None:
    actor = FlowActor(obs_dim=3, act_dim=2, hidden_sizes=[8], num_steps=4)
    obs = torch.zeros(3, 3)

    def reference(action: torch.Tensor) -> FlowReference:
        return compute_stochastic_reference(
            None,
            obs,
            action,
            anisotropy_enabled=False,
            inward_drift_enabled=False,
            calibration=0.0,
        )

    torch.manual_seed(2027)
    state = capture_rng_state()
    first = actor(obs, reference=reference)
    restore_rng_state(state)
    second = actor(obs, reference=reference)
    torch.testing.assert_close(first.action, second.action, rtol=0.0, atol=0.0)
    assert first.reference is not None and second.reference is not None
    torch.testing.assert_close(first.reference.noises, second.reference.noises, rtol=0.0, atol=0.0)

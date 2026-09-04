# Copyright 2026 OmniSafe Team. All Rights Reserved.
"""Preregistered correctness tests for P4.5 unified cumulative-cost geometry."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from omnisafe.algorithms.off_policy.star import STAR, equal_count_risk_bins
from omnisafe.models.safety.distributional_backend import (
    compute_cost_stochastic_reference,
    normalized_directional_loss,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'reports/star/p4_5_unified_cost_geometry_preregistration.json'
PROTOCOL_SHA256 = '8abebc101021ec71a00fdf6d2c5580b5f97dada710e58de00936a75490e85c11'


class AnalyticCostCritic(nn.Module):
    """Two identical cumulative-cost heads with normal along the first action axis."""

    def __init__(self) -> None:
        super().__init__()
        self.action_scale = nn.Parameter(torch.tensor(2.0))

    def expected_cost(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del obs
        value = 5.0 + self.action_scale * action[..., 0]
        return torch.stack((value, value), dim=0)


class BombProbabilityCritic(nn.Module):
    """Fail immediately if probability inference leaks into P4.5 policy geometry."""

    def probabilities(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del obs, action
        raise AssertionError('probability critic must be diagnostic-only in P4.5')


def _algorithm_shell() -> STAR:
    algorithm = object.__new__(STAR)
    algorithm._env_id = 'SafetyPointGoal1-v0'  # pylint: disable=protected-access
    cost_critic = AnalyticCostCritic()
    algorithm._actor_critic = SimpleNamespace(  # pylint: disable=protected-access
        context_dim=3,
        cost_critic=cost_critic,
        target_cost_critic=deepcopy(cost_critic),
        violation_critic=BombProbabilityCritic(),
    )
    algorithm._cfgs = SimpleNamespace(  # pylint: disable=protected-access
        algo_cfgs=SimpleNamespace(
            unified_cost_geometry_enabled=True,
            safety_anisotropy_enabled=True,
            safety_inward_drift_enabled=False,
            safety_outward_rectification_enabled=False,
            cost_geometry_bandwidth=0.10,
            flow_reference_variance=0.04,
            flow_normal_variance_ratio=0.10,
            safety_gradient_scale=0.005,
            safety_ensemble_scale=0.05,
            engineering_hard_stops_enabled=True,
            numerical_hard_stops_enabled=False,
            q_absolute_overflow_threshold=1e6,
        ),
        lagrange_cfgs=SimpleNamespace(cost_limit=10.0),
    )
    return algorithm


def test_preregistration_hash_and_locked_scope() -> None:
    payload = PROTOCOL.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == PROTOCOL_SHA256
    protocol = json.loads(payload)
    assert protocol['status'] == 'locked_before_any_p4_5_training'
    assert protocol['base_commit'].startswith('076e8d3')
    assert protocol['training']['seeds'] == [2027, 2028, 2029]
    assert protocol['training']['steps_per_run'] == 100_000
    assert protocol['method']['directional_loss_weight'] == 1.0
    assert protocol['method']['probability_critic'].endswith('policy-switch access')


def test_c0_c1_configs_differ_only_by_directional_loss_switch() -> None:
    configs = []
    for name in (
        'p4_5_c0_unified_cost_geometry.yaml',
        'p4_5_c1_directional_loss.yaml',
    ):
        configs.append(yaml.safe_load((ROOT / 'configs/star' / name).read_text(encoding='utf-8')))
    c0, c1 = configs
    c0_algo = c0['custom_cfgs']['algo_cfgs']
    c1_algo = c1['custom_cfgs']['algo_cfgs']
    assert c0_algo['directional_loss_enabled'] is False
    assert c1_algo['directional_loss_enabled'] is True
    for algo in (c0_algo, c1_algo):
        assert algo['unified_cost_geometry_enabled'] is True
        assert algo['safety_outward_rectification_enabled'] is False
        assert algo['safety_inward_drift_enabled'] is False
        assert algo['eval_milestones'] == [0, 25_000, 50_000, 75_000, 100_000]
        assert algo['checkpoint_milestones'] == [0, 10_000, 25_000, 50_000, 75_000, 100_000]
        assert len(algo['eval_seeds']) == 20
        assert algo['trajectory_eval_seeds'] == [12027, 12036, 12046]
        assert algo['numerical_hard_stops_enabled'] is False
        assert algo['engineering_hard_stops_enabled'] is True
    c0_algo.pop('directional_loss_enabled')
    c1_algo.pop('directional_loss_enabled')
    assert c0 == c1


def test_q_cost_reference_has_known_normal_margin_weight_and_fixed_trace() -> None:
    critic = AnalyticCostCritic()
    obs = torch.zeros(4, 6)
    action = torch.zeros(4, 2)
    reference = compute_cost_stochastic_reference(
        critic,
        obs,
        action,
        torch.full((4,), 0.5),
        anisotropy_enabled=True,
        cost_normalization_scale=10.0,
    )
    torch.testing.assert_close(reference.normal, torch.tensor([[1.0, 0.0]]).expand(4, -1))
    torch.testing.assert_close(reference.risk_ucb, torch.full((4,), 0.5))
    torch.testing.assert_close(reference.margin, torch.zeros(4))
    torch.testing.assert_close(reference.gradient_norm, torch.full((4,), 0.2))
    torch.testing.assert_close(
        reference.normal_variance + reference.tangent_variance,
        torch.full((4,), 0.08),
    )
    assert bool((reference.normal_variance < 0.04).all())
    assert bool((reference.tangent_variance > 0.04).all())
    torch.testing.assert_close(reference.drift, torch.zeros_like(action))
    assert critic.action_scale.requires_grad
    assert critic.action_scale.grad is None


def test_normalized_directional_loss_matches_formula_and_keeps_geometry_detached() -> None:
    velocity = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]],
        requires_grad=True,
    )
    normal = torch.tensor([[1.0, 0.0]]).expand_as(velocity).clone().requires_grad_(True)
    weight = torch.ones(3, requires_grad=True)
    loss, numerator, denominator = normalized_directional_loss(velocity, normal, weight)
    assert math.isclose(float(numerator.item()), 4.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(float(denominator.item()), 4.0, rel_tol=1e-6)
    assert math.isclose(float(loss.item()), 1.0 / 3.0, rel_tol=1e-6)
    loss.backward()
    assert velocity.grad is not None and bool(torch.isfinite(velocity.grad).all())
    assert normal.grad is None
    assert weight.grad is None

    harmless = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    harmless_normal = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    harmless_loss, _, _ = normalized_directional_loss(
        harmless,
        harmless_normal,
        torch.ones(2),
    )
    assert harmless_loss == 0.0


def test_probability_critic_has_no_reference_or_boundary_score_access() -> None:
    algorithm = _algorithm_shell()
    augmented = torch.zeros(3, 7)
    augmented[:, -3] = 0.5
    reference = algorithm._flow_reference(  # pylint: disable=protected-access
        augmented,
        target=False,
    )(torch.zeros(3, 2))
    torch.testing.assert_close(reference.normal[:, 0], torch.ones(3))

    algorithm._update_count = 1  # pylint: disable=protected-access
    algorithm._augment = lambda obs, context: torch.cat((obs, context), dim=-1)  # type: ignore[method-assign]  # pylint: disable=protected-access
    score = algorithm._boundary_scores(  # pylint: disable=protected-access
        torch.zeros(3, 4),
        torch.zeros(3, 2),
        torch.tensor([[0.5, 1.0, 0.0]]).expand(3, -1),
    )
    assert bool((score > 0.0).all())


def test_probability_evidence_cannot_open_the_legacy_w3_policy_switch() -> None:
    algorithm = _algorithm_shell()
    algorithm._env_steps = 100_000  # pylint: disable=protected-access
    algorithm._warmup = SimpleNamespace(  # pylint: disable=protected-access
        stage='W0',
        random_steps=2_000,
        terminal_utility_steps=8_000,
    )
    assert algorithm._advance_warmup() == 'W2'  # pylint: disable=protected-access
    assert algorithm._warmup.stage == 'W2'  # pylint: disable=protected-access


def test_equal_count_risk_bins_are_stable_and_cover_all_samples() -> None:
    predicted = torch.tensor([3.0, 1.0, 2.0, 2.0, 5.0, 4.0, 6.0])
    empirical = torch.tensor([30.0, 10.0, 20.0, 21.0, 50.0, 40.0, 60.0])
    bins = equal_count_risk_bins(predicted, empirical, count=5)
    assert sum(int(item['count']) for item in bins) == 7
    assert [int(item['count']) for item in bins] == [2, 2, 1, 1, 1]
    assert bins[0]['predicted_min'] == 1.0
    assert bins[1]['true_cost_to_go_mean'] == 25.5


def test_engineering_stops_reject_nonfinite_and_gross_q_overflow_only() -> None:
    algorithm = _algorithm_shell()
    algorithm._metric_sums = {}  # pylint: disable=protected-access
    algorithm._metric_counts = {}  # pylint: disable=protected-access
    algorithm._accumulate('Critic/RewardTD', 100.0)  # pylint: disable=protected-access
    with pytest.raises(FloatingPointError, match='non-finite'):
        algorithm._accumulate('Loss/Actor', float('nan'))  # pylint: disable=protected-access
    with pytest.raises(FloatingPointError, match='overflow'):
        algorithm._accumulate('Critic/CostQAbsMax', 1_000_001.0)  # pylint: disable=protected-access

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
"""Independent numerical gates for the STAR P4.1 correctness repair."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional

from omnisafe.adapter.star_adapter import STARContext
from omnisafe.algorithms.off_policy.star import STAR, normalized_actor_objective
from omnisafe.algorithms.off_policy.star_warmup import STARWarmup
from omnisafe.common.buffer.star_replay_buffer import BOUNDARY, NORMAL, STARReplayBuffer
from omnisafe.common.star_dual import GlobalSafetyDual
from omnisafe.models.actor.flow_actor import FlowActor
from omnisafe.models.actor_critic.star_actor_q_critic import STARActorQCritic
from omnisafe.models.critic.distributional_cost_critic import distributional_bellman_target
from omnisafe.models.critic.violation_probability_critic import (
    ViolationProbabilityCritic,
    binary_auroc,
    binary_brier_score,
    binary_brier_skill,
    masked_bce_loss,
)
from omnisafe.models.safety.distributional_backend import boundary_gate, compute_safety_geometry


def _model(*, legacy: bool = False) -> STARActorQCritic:
    return STARActorQCritic(
        obs_dim=3,
        act_dim=2,
        actor_hidden_sizes=[8],
        critic_hidden_sizes=[8],
        num_quantiles=5,
        safety_ensemble_size=2,
        context_dim=1 if legacy else 2,
        legacy_compatibility=legacy,
    )


def _store(
    replay: STARReplayBuffer,
    *,
    cost: float,
    terminated: bool = False,
    truncated: bool = False,
    threshold: float = 1.0,
    boundary_score: float = 0.0,
) -> None:
    replay.store_batch(
        obs=torch.zeros(1, 3),
        action=torch.zeros(1, 2),
        reward=torch.ones(1),
        cost=torch.tensor([cost]),
        terminated=torch.tensor([terminated]).float(),
        truncated=torch.tensor([truncated]).float(),
        next_obs=torch.ones(1, 3),
        context=torch.tensor([[threshold, 1.0]]),
        next_context=torch.tensor([[max(0.0, threshold - cost), 0.5]]),
        risk_threshold=torch.tensor([threshold]),
        next_risk_threshold=torch.tensor([max(0.0, threshold - cost)]),
        boundary_score=torch.tensor([boundary_score]),
        failure=torch.tensor([cost > 0.0]),
        source_random=torch.zeros(1, dtype=torch.bool),
    )


def test_every_p4_network_consumes_the_same_augmented_state() -> None:
    model = _model()
    augmented = torch.randn(4, 5)
    action = torch.randn(4, 2)
    assert model.flow(augmented, base_action=action.clamp(-1, 1)).action.shape == (4, 2)
    assert all(value.shape == (4,) for value in model.reward_critic(augmented, action))
    assert model.cost_critic(augmented, action).shape == (2, 4, 5)
    assert model.violation_critic is not None
    assert model.violation_critic(augmented, action).shape == (2, 4)


def test_legacy_model_is_shape_compatible_and_evaluation_only() -> None:
    model = _model(legacy=True)
    assert model.flow(torch.randn(2, 4)).action.shape == (2, 2)
    assert model.reward_critic(torch.randn(2, 3), torch.randn(2, 2))[0].shape == (2,)
    assert model.violation_critic is None
    assert model.violation_critic_optimizer is None


def test_point_context_contains_budget_and_remaining_horizon() -> None:
    context = STARContext('SafetyPointGoal1-v0', 1, 10.0, 4, 'cpu')
    current, threshold = context.values()
    assert torch.equal(current, torch.tensor([[1.0, 1.0]]))
    assert threshold.item() == 10.0
    following, following_threshold = context.advance(
        torch.tensor([2.0]),
        torch.tensor([False]),
        torch.tensor([False]),
    )
    assert torch.equal(following, torch.tensor([[0.8, 0.75]]))
    assert following_threshold.item() == 8.0


def test_locomotion_context_uses_budget_sentinel_and_horizon() -> None:
    context = STARContext('Walker2d-v4', 1, 0.0, 4, 'cpu')
    following, threshold = context.advance(
        torch.tensor([0.0]),
        torch.tensor([False]),
        torch.tensor([False]),
    )
    assert torch.equal(following, torch.tensor([[1.0, 0.75]]))
    assert threshold.item() == 0.5


def test_replay_reward_and_cost_done_masks_diverge_on_truncation() -> None:
    replay = STARReplayBuffer(3, 2, capacity=32, batch_size=4, pin_memory=False)
    _store(replay, cost=0.0, truncated=True)
    assert replay.data['reward_done'][0].item() == 0.0
    assert replay.data['cost_done'][0].item() == 1.0


def test_replay_both_done_masks_stop_on_true_termination() -> None:
    replay = STARReplayBuffer(3, 2, capacity=32, batch_size=4, pin_memory=False)
    _store(replay, cost=0.0, terminated=True)
    assert replay.data['reward_done'][0].item() == 1.0
    assert replay.data['cost_done'][0].item() == 1.0


def test_probability_labels_remain_masked_until_episode_completion() -> None:
    replay = STARReplayBuffer(3, 2, capacity=32, batch_size=4, pin_memory=False)
    _store(replay, cost=0.4, threshold=1.0)
    assert torch.isnan(replay.data['violation_label'][0])
    _store(replay, cost=0.7, threshold=0.6, truncated=True)
    assert torch.equal(replay.data['empirical_cost_to_go'][:2], torch.tensor([1.1, 0.7]))
    assert torch.equal(replay.data['violation_label'][:2], torch.ones(2))


def test_probability_label_is_strict_empirical_cost_over_remaining_budget() -> None:
    replay = STARReplayBuffer(3, 2, capacity=32, batch_size=4, pin_memory=False)
    _store(replay, cost=1.0, threshold=1.0, terminated=True)
    assert replay.data['violation_label'][0].item() == 0.0


def test_replay_keeps_continuous_boundary_score_and_derives_category() -> None:
    replay = STARReplayBuffer(
        3,
        2,
        capacity=32,
        batch_size=4,
        boundary_score_threshold=0.5,
        pin_memory=False,
    )
    _store(replay, cost=0.0, boundary_score=0.49)
    assert replay.data['category'][0].item() == NORMAL
    replay.relabel_boundaries(torch.tensor([0]), torch.tensor([0.75]))
    assert replay.data['boundary_score'][0].item() == pytest.approx(0.75)
    assert replay.data['category'][0].item() == BOUNDARY


def test_replay_rejects_missing_horizon_context() -> None:
    replay = STARReplayBuffer(3, 2, capacity=32, batch_size=4, pin_memory=False)
    with pytest.raises(ValueError, match='context must have shape'):
        replay.store_batch(
            obs=torch.zeros(1, 3),
            action=torch.zeros(1, 2),
            reward=torch.zeros(1),
            cost=torch.zeros(1),
            terminated=torch.zeros(1),
            truncated=torch.zeros(1),
            next_obs=torch.zeros(1, 3),
            context=torch.ones(1),
            next_context=torch.ones(1),
            risk_threshold=torch.ones(1),
            next_risk_threshold=torch.ones(1),
            boundary_score=torch.zeros(1),
            failure=torch.zeros(1, dtype=torch.bool),
            source_random=torch.zeros(1, dtype=torch.bool),
        )


def test_probability_ensemble_outputs_logits_probabilities_and_ucb() -> None:
    critic = ViolationProbabilityCritic(5, 2, [8], ensemble_size=3)
    obs, action = torch.randn(7, 5), torch.randn(7, 2)
    assert critic(obs, action).shape == (3, 7)
    probability = critic.probabilities(obs, action)
    assert torch.all((probability >= 0.0) & (probability <= 1.0))
    assert critic.ucb(obs, action, beta=0.5).shape == (7,)


def test_masked_bce_ignores_unfinished_labels() -> None:
    logits = torch.tensor([[0.0, 100.0], [0.0, -100.0]], requires_grad=True)
    labels = torch.tensor([1.0, 0.0])
    loss = masked_bce_loss(logits, labels, torch.tensor([True, False]))
    expected = functional.binary_cross_entropy_with_logits(
        torch.zeros(2),
        torch.ones(2),
    )
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert torch.equal(logits.grad[:, 1], torch.zeros(2))


def test_masked_bce_all_unfinished_is_zero_with_zero_gradient() -> None:
    logits = torch.randn(2, 4, requires_grad=True)
    loss = masked_bce_loss(logits, torch.zeros(4), torch.zeros(4, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_binary_probability_metrics_reward_perfect_skill() -> None:
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    probability = labels.clone()
    assert binary_brier_score(probability, labels) == 0.0
    assert binary_brier_skill(probability, labels) == 1.0
    assert binary_auroc(probability, labels) == 1.0


def test_binary_auroc_is_tie_correct() -> None:
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert binary_auroc(torch.full((4,), 0.5), labels) == pytest.approx(0.5)


def test_gaussian_boundary_score_is_centered_at_beta_half() -> None:
    risk = torch.tensor([0.2, 0.5, 0.8])
    score = boundary_gate(risk, boundary=0.5, bandwidth=0.1)
    assert score[1].item() == 1.0
    assert score[0].item() == pytest.approx(score[2].item())
    assert score[1] > score[0]


def test_normalized_actor_objective_matches_required_formula() -> None:
    energy = torch.tensor([1.0, 2.0])
    reward = torch.tensor([3.0, 4.0])
    cost = torch.tensor([5.0, 6.0])
    objective = normalized_actor_objective(energy, reward, cost, 0.2, 9.0)
    expected = (0.2 * energy - reward + 9.0 * cost) / 10.0
    assert torch.allclose(objective, expected)


def test_b0_switches_remove_both_safety_paths_without_hiding_episode_costs() -> None:
    algorithm = object.__new__(STAR)
    algorithm._cfgs = SimpleNamespace(  # pylint: disable=protected-access
        algo_cfgs=SimpleNamespace(
            enable_global_safety_constraint=False,
            enable_local_safety_geometry=False,
        ),
    )
    algorithm._warmup = SimpleNamespace(stage='W3')  # pylint: disable=protected-access
    algorithm._global_dual = GlobalSafetyDual(  # pylint: disable=protected-access
        0.1,
        0.01,
        1.0,
        cost_limit=1.0,
    )
    algorithm._pending_episode_costs = deque([5.0])  # pylint: disable=protected-access

    assert not algorithm._training_geometry_enabled()  # pylint: disable=protected-access
    assert (
        algorithm._effective_global_safety_multiplier() == 0.0
    )  # pylint: disable=protected-access
    algorithm._consume_episode_dual_updates()  # pylint: disable=protected-access
    assert algorithm._global_dual.episode_observations == 1  # pylint: disable=protected-access
    assert algorithm._global_dual.episode_updates == 0  # pylint: disable=protected-access


def test_b1_updates_only_the_global_constraint_with_local_geometry_off() -> None:
    algorithm = object.__new__(STAR)
    algorithm._cfgs = SimpleNamespace(  # pylint: disable=protected-access
        algo_cfgs=SimpleNamespace(
            enable_global_safety_constraint=True,
            enable_local_safety_geometry=False,
        ),
    )
    algorithm._warmup = SimpleNamespace(stage='W3')  # pylint: disable=protected-access
    algorithm._global_dual = GlobalSafetyDual(  # pylint: disable=protected-access
        0.1,
        0.01,
        1.0,
        cost_limit=1.0,
    )
    algorithm._pending_episode_costs = deque([5.0])  # pylint: disable=protected-access

    assert not algorithm._training_geometry_enabled()  # pylint: disable=protected-access
    assert algorithm._effective_global_safety_multiplier() == pytest.approx(
        0.1,
    )  # pylint: disable=protected-access
    algorithm._consume_episode_dual_updates()  # pylint: disable=protected-access
    assert algorithm._global_dual.episode_updates == 1  # pylint: disable=protected-access
    assert algorithm._global_dual.value > 0.1  # pylint: disable=protected-access


def test_global_dual_can_record_without_updating_multiplier() -> None:
    dual = GlobalSafetyDual(0.1, 0.01, 1.0, cost_limit=1.0)
    dual.record_episode(5.0)
    assert dual.value == 0.1
    assert dual.episode_observations == 1
    assert dual.episode_updates == 0
    dual.update_episode(5.0)
    assert dual.value > 0.1
    assert dual.episode_observations == 2
    assert dual.episode_updates == 1


def test_w3_requires_both_label_classes_and_probability_skill() -> None:
    warmup = STARWarmup(
        random_steps=0,
        terminal_utility_steps=0,
        geometry_min_steps=0,
        td_error_threshold=1.0,
        calibration_threshold=1.0,
        valid_gradient_threshold=0.1,
        disagreement_threshold=0.9,
        boundary_samples_threshold=1,
        positive_labels_threshold=2,
        negative_labels_threshold=2,
        brier_skill_threshold=0.0,
        auroc_threshold=0.6,
        ema_rate=1.0,
    )
    warmup.observe(
        cost_td_error=0.1,
        calibration_error=0.1,
        valid_gradient_rate=1.0,
        ensemble_disagreement=0.0,
        brier_skill=0.1,
        auroc=0.9,
    )
    assert warmup.advance(100, 10, 2, 1) == 'W2'
    assert warmup.advance(100, 10, 2, 2) == 'W3'


def test_cost_mask_stops_distributional_bootstrap_at_truncation() -> None:
    next_quantiles = torch.ones(2, 2, 3)
    target = distributional_bellman_target(
        torch.tensor([2.0, 2.0]),
        torch.tensor([1.0, 0.0]),
        next_quantiles,
        gamma_cost=1.0,
    )
    assert torch.equal(target[:, 0], torch.full((2, 3), 2.0))
    assert torch.equal(target[:, 1], torch.full((2, 3), 3.0))


def test_probability_geometry_keeps_actor_gradient_and_not_critic_gradient() -> None:
    actor = FlowActor(obs_dim=5, act_dim=2, hidden_sizes=[8], num_steps=4)
    critic = ViolationProbabilityCritic(5, 2, [8], ensemble_size=2)
    obs = torch.randn(6, 5)
    geometry = compute_safety_geometry(
        critic,
        obs,
        actor(obs),
        enabled=True,
        minimum_consistency=0.0,
    )
    geometry.energy.backward()
    assert all(parameter.grad is None for parameter in critic.parameters())
    assert all(parameter.grad is not None for parameter in actor.parameters())
    assert torch.isfinite(geometry.gradient_norm).all()


def test_synthetic_two_dimensional_violation_head_learns_risk_and_action_gradient() -> None:
    torch.manual_seed(41)
    critic = ViolationProbabilityCritic(2, 2, [32], ensemble_size=2)
    optimizer = torch.optim.Adam(critic.parameters(), lr=0.02)
    obs = torch.randn(256, 2)
    action = torch.empty(256, 2).uniform_(-1.0, 1.0)
    label = (action[:, 0] + 0.5 * action[:, 1] > 0.0).float()
    labeled = torch.ones(256, dtype=torch.bool)
    for _ in range(120):
        logits = critic(obs, action)
        loss = masked_bce_loss(logits, label, labeled)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    probability = critic.probabilities(obs, action).mean(dim=0)
    assert binary_auroc(probability, label) > 0.95
    probe = torch.tensor([[0.0, 0.0]], requires_grad=True)
    risk = critic.probabilities(torch.zeros(1, 2), probe).mean()
    gradient = torch.autograd.grad(risk, probe)[0]
    assert torch.linalg.vector_norm(gradient).item() > 1e-4

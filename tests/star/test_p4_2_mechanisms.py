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
"""Causal correctness gates for the STAR P4.2 mechanism repair."""

from __future__ import annotations

import pytest
import torch

from omnisafe.adapter.star_adapter import STARContext
from omnisafe.algorithms.off_policy.star import select_boundary_candidates, tangent_projection
from omnisafe.algorithms.off_policy.star_warmup import STARWarmup
from omnisafe.common.buffer.star_replay_buffer import STARReplayBuffer
from omnisafe.common.star_checkpoint import load_star_model_state
from omnisafe.models.actor_critic.star_actor_q_critic import STARActorQCritic
from omnisafe.models.critic.violation_probability_critic import (
    masked_bce_loss,
    violation_bellman_target,
)


def _model() -> STARActorQCritic:
    return STARActorQCritic(
        obs_dim=3,
        act_dim=2,
        actor_hidden_sizes=[8],
        critic_hidden_sizes=[8],
        num_quantiles=5,
        safety_ensemble_size=2,
        context_dim=3,
    )


def _store_single_transition(
    replay: STARReplayBuffer,
    marker: float,
    *,
    terminated: bool = True,
) -> None:
    replay.store_batch(
        obs=torch.full((1, 3), marker),
        action=torch.zeros(1, 2),
        reward=torch.tensor([marker]),
        cost=torch.zeros(1),
        terminated=torch.tensor([terminated], dtype=torch.float32),
        truncated=torch.zeros(1),
        next_obs=torch.full((1, 3), marker + 1.0),
        context=torch.tensor([[1.0, 1.0, 0.0]]),
        next_context=torch.tensor([[1.0, 0.0, 0.0]]),
        risk_threshold=torch.ones(1),
        next_risk_threshold=torch.ones(1),
        boundary_score=torch.zeros(1),
        failure=torch.zeros(1, dtype=torch.bool),
        source_random=torch.zeros(1, dtype=torch.bool),
    )


def _warmup() -> STARWarmup:
    return STARWarmup(
        random_steps=0,
        terminal_utility_steps=0,
        geometry_min_steps=0,
        td_error_threshold=1.0,
        calibration_threshold=1.0,
        valid_gradient_threshold=0.01,
        disagreement_threshold=0.2,
        boundary_samples_threshold=2,
        positive_labels_threshold=1,
        negative_labels_threshold=1,
        brier_skill_threshold=0.0,
        auroc_threshold=0.65,
        ema_rate=1.0,
        dynamic_geometry_gate=True,
        geometry_enable_passes=3,
        geometry_disable_failures=2,
    )


def _gate_observation(warmup: STARWarmup, *, passing: bool) -> bool:
    return warmup.evaluate_geometry_gate(
        env_steps=100,
        boundary_samples=3 if passing else 0,
        positive_labels=2,
        negative_labels=2,
        cost_td_error=0.1,
        calibration_error=0.1,
        valid_gradient_rate=0.1,
        ensemble_disagreement=0.1,
        brier_skill=0.1,
        auroc=0.8,
    )


def test_probability_target_is_frozen_and_has_independent_polyak_rate() -> None:
    model = _model()
    assert model.violation_critic is not None
    assert model.target_violation_critic is not None
    assert all(
        not parameter.requires_grad for parameter in model.target_violation_critic.parameters()
    )
    before = [parameter.clone() for parameter in model.target_violation_critic.parameters()]
    with torch.no_grad():
        for parameter in model.violation_critic.parameters():
            parameter.add_(1.0)
    model.polyak_update(0.005, violation_coefficient=0.1)
    for old, target in zip(before, model.target_violation_critic.parameters()):
        assert torch.allclose(target, old + 0.1)


def test_older_model_state_initializes_missing_probability_target() -> None:
    source = _model()
    state = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if not key.startswith('target_violation_critic.')
    }
    restored = _model()
    load_star_model_state(restored, state)
    assert restored.violation_critic is not None
    assert restored.target_violation_critic is not None
    assert all(
        torch.equal(target, online)
        for target, online in zip(
            restored.target_violation_critic.parameters(),
            restored.violation_critic.parameters(),
        )
    )


def test_bellman_violation_target_obeys_breach_safe_end_bootstrap_precedence() -> None:
    next_probability = torch.tensor([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]])
    target = violation_bellman_target(
        next_probability,
        next_violation=torch.tensor([True, False, False]),
        episode_done=torch.tensor([True, True, False]),
    )
    assert torch.equal(target[:, 0], torch.ones(2))
    assert torch.equal(target[:, 1], torch.zeros(2))
    assert torch.equal(target[:, 2], next_probability[:, 2])


def test_masked_bce_accepts_headwise_soft_bellman_targets() -> None:
    logits = torch.zeros(2, 2, requires_grad=True)
    target = torch.tensor([[0.25, 0.75], [0.75, 0.25]])
    loss = masked_bce_loss(logits, target, torch.ones(2, dtype=torch.bool))
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_bellman_context_keeps_signed_budget_and_absorbing_violation_flag() -> None:
    context = STARContext(
        'SafetyPointGoal1-v0',
        1,
        10.0,
        4,
        'cpu',
        include_violation=True,
    )
    following, threshold = context.advance(
        torch.tensor([11.0]),
        torch.tensor([False]),
        torch.tensor([False]),
    )
    assert torch.allclose(following, torch.tensor([[-0.1, 0.75, 1.0]]))
    assert threshold.item() == -1.0
    current, _ = context.values()
    assert current[0, 2].item() == 1.0
    terminal, _ = context.advance(
        torch.tensor([0.0]),
        torch.tensor([True]),
        torch.tensor([False]),
    )
    assert terminal[0, 2].item() == 1.0
    reset, reset_threshold = context.values()
    assert torch.equal(reset, torch.tensor([[1.0, 1.0, 0.0]]))
    assert reset_threshold.item() == 10.0


def test_exact_budget_exhaustion_is_not_a_strict_violation() -> None:
    context = STARContext(
        'SafetyPointGoal1-v0',
        1,
        10.0,
        4,
        'cpu',
        include_violation=True,
    )
    following, threshold = context.advance(
        torch.tensor([10.0]),
        torch.tensor([False]),
        torch.tensor([False]),
    )
    assert threshold.item() == 0.0
    assert following[0, 2].item() == 0.0


def test_episode_split_excludes_heldout_from_every_training_sampler() -> None:
    replay = STARReplayBuffer(
        3,
        2,
        capacity=32,
        batch_size=8,
        context_dim=3,
        heldout_episode_period=2,
        pin_memory=False,
    )
    for marker in range(4):
        _store_single_transition(replay, float(marker))
    assert torch.equal(replay.data['episode_id'][:4], torch.arange(4))
    assert torch.equal(
        replay.data['is_heldout'][:4],
        torch.tensor([False, True, False, True]),
    )
    safety_batch = replay.sample_batch(batch_size=128)
    reward_batch = replay.sample_uniform_recent_batch(
        batch_size=128,
        recent_fraction=0.5,
        recent_window=32,
    )
    assert not bool(safety_batch['is_heldout'].any())
    assert not bool(reward_batch['is_heldout'].any())
    assert set(safety_batch['episode_id'].tolist()) <= {0, 2}
    assert set(reward_batch['episode_id'].tolist()) <= {0, 2}

    heldout = replay.recent_heldout_batch(recent_episodes=1)
    assert bool(heldout['is_heldout'].all())
    assert set(heldout['episode_id'].tolist()) == {3}
    assert replay.label_counts_for('train')['finalized'] == 2
    assert replay.label_counts_for('heldout')['finalized'] == 2


def test_unfinished_heldout_episode_is_not_visible_to_gate_reader() -> None:
    replay = STARReplayBuffer(
        3,
        2,
        capacity=32,
        batch_size=4,
        context_dim=3,
        heldout_episode_period=2,
        pin_memory=False,
    )
    _store_single_transition(replay, 0.0)
    _store_single_transition(replay, 1.0, terminated=False)
    with pytest.raises(RuntimeError, match='no finalized held-out'):
        replay.recent_heldout_batch()


def test_dynamic_geometry_gate_has_three_pass_two_failure_hysteresis() -> None:
    warmup = _warmup()
    assert warmup.advance(100, 0, 0, 0) == 'W2'
    assert _gate_observation(warmup, passing=True)
    assert warmup.stage == 'W3'
    assert not warmup.geometry_enabled
    assert _gate_observation(warmup, passing=True)
    assert not warmup.geometry_enabled
    assert _gate_observation(warmup, passing=True)
    assert warmup.geometry_enabled
    assert warmup.geometry_activation_step == 100

    assert not _gate_observation(warmup, passing=False)
    assert warmup.geometry_enabled
    assert not _gate_observation(warmup, passing=False)
    assert not warmup.geometry_enabled
    assert warmup.stage == 'W3'

    for _ in range(3):
        assert _gate_observation(warmup, passing=True)
    assert warmup.geometry_enabled
    assert warmup.stage == 'W3'


def test_candidate_selection_stays_inside_trusted_boundary_and_uses_reward_tie_break() -> None:
    selected = select_boundary_candidates(
        risk_ucb=torch.tensor([[0.20, 0.48, 0.47, 0.49], [0.70, 0.71, 0.90, 0.95]]),
        disagreement=torch.tensor([[0.01, 0.02, 0.02, 0.50], [0.01, 0.01, 0.50, 0.50]]),
        reward_value=torch.tensor([[0.0, 1.0, 2.0, 100.0], [1.0, 2.0, 100.0, 100.0]]),
        risk_boundary=0.5,
        disagreement_threshold=0.1,
        risk_tie_band=0.02,
    )
    assert torch.equal(selected, torch.tensor([2, 1]))


def test_tangent_projection_removes_first_order_risk_normal_component() -> None:
    noise = torch.tensor([[1.0, 2.0], [-3.0, 4.0]])
    gradient = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    tangent, ratio = tangent_projection(noise, gradient)
    assert torch.allclose((tangent * gradient).sum(dim=-1), torch.zeros(2))
    assert torch.all((ratio >= 0.0) & (ratio <= 1.0))

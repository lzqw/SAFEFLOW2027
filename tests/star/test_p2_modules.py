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
"""Independent numerical contracts for STAR P2 modules."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn, optim

from omnisafe.adapter.star_adapter import STARAdapter, STARContext
from omnisafe.algorithms.off_policy.star_warmup import STARWarmup
from omnisafe.common.buffer.star_replay_buffer import STARReplayBuffer, updates_for_transitions
from omnisafe.common.normalizer import Normalizer
from omnisafe.common.star_checkpoint import load_star_checkpoint, save_star_checkpoint
from omnisafe.common.star_dual import EnergyDual, GlobalSafetyDual
from omnisafe.models.actor.flow_actor import FlowActor
from omnisafe.models.actor_critic.star_actor_q_critic import (
    flac_reward_target,
    flac_sampled_reward_target,
)
from omnisafe.models.critic.distributional_cost_critic import (
    DistributionalCostCritic,
    distributional_bellman_target,
    quantile_huber_loss,
)
from omnisafe.models.critic.violation_probability_critic import ViolationProbabilityCritic
from omnisafe.models.safety.distributional_backend import (
    compute_safety_geometry,
    directional_variances,
    normalized_jvp,
)


def test_flow_is_bounded_and_returns_midpoint_trajectory() -> None:
    actor = FlowActor(obs_dim=5, act_dim=2, hidden_sizes=[16], num_steps=4)
    obs = torch.randn(7, 5)
    base = torch.linspace(-1.0, 1.0, 14).reshape(7, 2)
    output = actor(obs, base_action=base)
    assert output.action.shape == (7, 2)
    assert output.points.shape == (7, 5, 2)
    assert output.velocity_points.shape == (7, 4, 2)
    assert output.velocities.shape == (7, 4, 2)
    assert output.times.shape == (7, 4)
    assert output.kinetic_energy.shape == (7,)
    assert torch.equal(output.points[:, 0], base)
    assert torch.all(output.points.abs() <= 1.0)
    assert torch.all(output.velocity_points.abs() <= 1.0)
    assert torch.allclose(output.times[0], torch.tensor([0.125, 0.375, 0.625, 0.875]))


def test_distributional_target_is_correct_and_detached() -> None:
    next_quantiles = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    next_quantiles.requires_grad_(True)
    cost = torch.tensor([1.0, 2.0, 3.0])
    done = torch.tensor([0.0, 1.0, 0.0])
    target = distributional_bellman_target(cost, done, next_quantiles, gamma_cost=0.5)
    expected = cost.reshape(1, 3, 1) + 0.5 * (1.0 - done.reshape(1, 3, 1)) * (
        next_quantiles.detach()
    )
    assert torch.equal(target, expected)
    assert not target.requires_grad


def test_flac_reward_target_penalizes_future_path_energy() -> None:
    reward = torch.tensor([1.0, 2.0])
    done = torch.tensor([0.0, 1.0])
    first = torch.tensor([5.0, 8.0], requires_grad=True)
    second = torch.tensor([4.0, 7.0], requires_grad=True)
    energy = torch.tensor([2.0, 3.0], requires_grad=True)
    target = flac_reward_target(reward, done, first, second, energy, 0.5, gamma=0.9)
    assert torch.allclose(target, torch.tensor([3.7, 2.0]))
    assert not target.requires_grad


def test_sampled_flac_target_applies_twin_minimum_before_policy_expectation() -> None:
    reward = torch.tensor([1.0, 2.0])
    done = torch.tensor([0.0, 1.0])
    first = torch.tensor([[5.0, 8.0], [2.0, 10.0]], requires_grad=True)
    second = torch.tensor([[4.0, 7.0], [3.0, 9.0]], requires_grad=True)
    energy = torch.tensor([[2.0, 3.0], [0.0, 5.0]], requires_grad=True)
    target = flac_sampled_reward_target(
        reward,
        done,
        first,
        second,
        energy,
        0.5,
        gamma=0.9,
    )
    assert torch.allclose(target, torch.tensor([3.25, 2.0]))
    assert not target.requires_grad


def test_sampled_flac_target_rejects_misaligned_sample_tensors() -> None:
    with pytest.raises(ValueError, match='share one shape'):
        flac_sampled_reward_target(
            torch.ones(2),
            torch.zeros(2),
            torch.ones(4, 2),
            torch.ones(3, 2),
            torch.ones(4, 2),
            0.1,
            gamma=0.99,
        )


def test_quantile_huber_loss_is_finite_with_bootstrap_and_importance() -> None:
    critic = DistributionalCostCritic(4, 2, [16], num_quantiles=7, ensemble_size=2)
    prediction = critic(torch.randn(8, 4), torch.randn(8, 2).clamp(-1.0, 1.0))
    target = torch.rand_like(prediction)
    mask = torch.tensor([[1] * 8, [1, 0] * 4], dtype=torch.float32)
    weights = torch.linspace(0.2, 1.0, 8)
    loss = quantile_huber_loss(
        prediction,
        target,
        critic.quantile_fractions,
        importance_weights=weights,
        bootstrap_mask=mask,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in critic.parameters())


def test_normalized_jvp_is_invariant_to_positive_risk_scaling() -> None:
    gradient = torch.tensor([[3.0, 4.0], [-2.0, 1.0]])
    velocity = torch.tensor([[0.5, -1.0], [2.0, 3.0]])
    baseline = normalized_jvp(gradient, velocity)
    assert torch.allclose(baseline, normalized_jvp(1000.0 * gradient, velocity), rtol=1e-5)
    assert torch.allclose(baseline, normalized_jvp(0.001 * gradient, velocity), rtol=1e-5)


def test_directional_variance_contract_and_monotonicity() -> None:
    gates = torch.tensor([0.0, 0.1, 0.5, 1.0])
    tangent, normal = directional_variances(gates, tangent_variance=2.0, contraction=9.0)
    assert torch.all(tangent == 2.0)
    assert torch.all(normal > 0.0)
    assert torch.all(normal <= tangent)
    assert normal[0] == tangent[0]
    assert torch.all(normal[1:] < normal[:-1])


def test_energy_and_global_duals_update_in_the_required_directions() -> None:
    energy = EnergyDual(initial_value=0.2, learning_rate=0.05)
    before = energy.value.item()
    energy.update(torch.tensor(5.0), target_energy=1.0)
    assert energy.value.item() > before
    low_energy = EnergyDual(initial_value=0.2, learning_rate=0.05)
    low_before = low_energy.value.item()
    low_energy.update(torch.tensor(0.0), target_energy=1.0)
    assert low_energy.value.item() < low_before

    safety = GlobalSafetyDual(0.1, learning_rate=0.01, ema_rate=1.0, cost_limit=10.0)
    safety.update_episode(20.0)
    assert safety.value == pytest.approx(0.2)
    safety.update_episode(0.0)
    assert safety.value == pytest.approx(0.1)
    assert safety.episode_updates == 2


def test_geometry_freezes_critic_avoids_hessian_and_keeps_actor_gradient() -> None:
    actor = FlowActor(obs_dim=5, act_dim=2, hidden_sizes=[16], num_steps=4)
    critic = ViolationProbabilityCritic(5, 2, [16], ensemble_size=2)
    obs = torch.randn(8, 5)
    flow = actor(obs)
    geometry = compute_safety_geometry(
        critic,
        obs,
        flow,
        enabled=True,
        minimum_consistency=0.0,
    )
    geometry.energy.backward()
    assert torch.isfinite(geometry.energy)
    assert not geometry.normals.requires_grad
    assert not geometry.gate.requires_grad
    assert all(parameter.grad is None for parameter in critic.parameters())
    actor_gradients = [parameter.grad for parameter in actor.parameters()]
    assert all(gradient is not None for gradient in actor_gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in actor_gradients if gradient is not None
    )


def _store_probe_batch(buffer: STARReplayBuffer, index: int) -> None:
    envs = buffer.num_envs
    done = torch.zeros(envs)
    if index == 2:
        done.fill_(1.0)
    boundary = torch.tensor([(env + index) % 3 == 0 for env in range(envs)]).float()
    failure = torch.tensor([(env + index) % 5 == 0 for env in range(envs)])
    buffer.store_batch(
        obs=torch.randn(envs, buffer.obs_dim),
        action=torch.randn(envs, buffer.act_dim).clamp(-1.0, 1.0),
        reward=torch.randn(envs),
        cost=failure.float(),
        terminated=done,
        truncated=torch.zeros(envs),
        next_obs=torch.randn(envs, buffer.obs_dim),
        context=torch.ones(envs, 2),
        next_context=torch.ones(envs, 2) * 0.9,
        risk_threshold=torch.ones(envs),
        next_risk_threshold=torch.ones(envs) * 0.9,
        boundary_score=boundary,
        failure=failure,
        source_random=torch.full((envs,), index == 0),
    )


@pytest.mark.parametrize('num_envs', [1, 2, 4])
def test_replay_is_cpu_and_global_batch_is_concurrency_invariant(num_envs: int) -> None:
    replay = STARReplayBuffer(
        obs_dim=3,
        act_dim=2,
        capacity=128,
        batch_size=16,
        num_envs=num_envs,
    )
    for index in range(4):
        _store_probe_batch(replay, index)
    batch = replay.sample_batch()
    assert replay.device.type == 'cpu'
    assert all(tensor.device.type == 'cpu' for tensor in replay.data.values())
    assert batch['obs'].shape == (16, 3)
    assert batch['action'].shape == (16, 2)
    assert batch['importance_weights'].shape == (16,)
    assert sum(replay.pool_sizes.values()) == replay.size

    updates, carry = updates_for_transitions(1.0, 25 * num_envs)
    assert updates / (25 * num_envs) == 1.0
    assert carry == 0.0


def test_reward_replay_recent_window_follows_ring_chronology_after_wraparound() -> None:
    replay = STARReplayBuffer(
        obs_dim=3,
        act_dim=2,
        capacity=16,
        batch_size=8,
        num_envs=1,
    )
    for index in range(24):
        marker = torch.tensor([float(index)])
        replay.store_batch(
            obs=marker.repeat(1, 3),
            action=torch.zeros(1, 2),
            reward=marker,
            cost=torch.zeros(1),
            terminated=torch.zeros(1),
            truncated=torch.zeros(1),
            next_obs=marker.repeat(1, 3),
            context=torch.ones(1, 2),
            next_context=torch.ones(1, 2),
            risk_threshold=torch.ones(1),
            next_risk_threshold=torch.ones(1),
            boundary_score=torch.zeros(1),
            failure=torch.zeros(1, dtype=torch.bool),
            source_random=torch.zeros(1, dtype=torch.bool),
        )
    batch = replay.sample_uniform_recent_batch(
        batch_size=128,
        recent_fraction=1.0,
        recent_window=4,
    )
    assert replay.pointer == 8
    assert batch['reward'].min().item() >= 20.0
    assert batch['reward'].max().item() <= 23.0
    assert torch.equal(batch['importance_weights'], torch.ones(128))


def test_reward_replay_mixture_validates_recent_controls() -> None:
    replay = STARReplayBuffer(3, 2, capacity=16, batch_size=8)
    _store_probe_batch(replay, 0)
    with pytest.raises(ValueError, match='fraction'):
        replay.sample_uniform_recent_batch(recent_fraction=1.1)
    with pytest.raises(ValueError, match='window'):
        replay.sample_uniform_recent_batch(recent_window=0)


def test_point_budget_and_locomotion_horizon_have_distinct_semantics() -> None:
    point = STARContext('SafetyPointGoal1-v0', 2, 10.0, 1000, 'cpu')
    context, threshold = point.values()
    assert torch.equal(context, torch.ones(2, 2))
    assert torch.equal(threshold, torch.full((2,), 10.0))
    next_context, next_threshold = point.advance(
        torch.tensor([2.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.zeros(2),
    )
    assert torch.allclose(next_context, torch.tensor([[0.8, 0.999], [1.0, 0.999]]))
    assert torch.equal(next_threshold, torch.tensor([8.0, 10.0]))
    reset_context, _ = point.values()
    assert torch.allclose(reset_context, torch.tensor([[0.8, 0.999], [1.0, 1.0]]))

    walker = STARContext('Walker2d-v4', 1, 0.0, 1000, 'cpu')
    horizon, failure_threshold = walker.values()
    assert torch.equal(horizon, torch.ones(1, 2))
    assert failure_threshold.item() == 0.5
    next_horizon, next_failure_threshold = walker.advance(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
    )
    assert torch.allclose(next_horizon, torch.tensor([[1.0, 0.999]]))
    assert next_failure_threshold.item() == 0.5


def test_point_evaluation_layout_reads_native_goal_and_hazards() -> None:
    task = SimpleNamespace(
        agent=SimpleNamespace(pos=torch.tensor([0.25, -0.5, 0.1])),
        goal=SimpleNamespace(pos=torch.tensor([1.0, -1.0, 0.16]), size=0.3),
        hazards=SimpleNamespace(
            pos=[torch.tensor([0.5, 0.75, 0.02]), torch.tensor([-0.25, 0.0, 0.02])],
            size=0.2,
        ),
    )
    adapter = object.__new__(STARAdapter)
    with patch.object(
        STARAdapter,
        '_evaluation_task',  # pylint: disable=protected-access
        return_value=task,
    ):
        assert adapter.evaluation_position() == (0.25, -0.5)
        layout = adapter.evaluation_layout()
    assert layout is not None
    assert layout['goal'] == (1.0, -1.0)
    assert layout['hazards'] == [(0.5, 0.75), (-0.25, 0.0)]
    assert layout['goal_radius'] == pytest.approx(0.3)
    assert layout['hazard_radius'] == pytest.approx(0.2)


def test_checkpoint_restores_model_optimizer_dual_normalizer_rng_and_steps(
    tmp_path: Path,
) -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model = nn.Linear(3, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    normalizer = Normalizer((3,))
    loss = model(torch.randn(4, 3)).square().mean()
    loss.backward()
    optimizer.step()
    normalizer.normalize(torch.randn(8, 3))
    replay = STARReplayBuffer(3, 2, capacity=32, batch_size=4)
    _store_probe_batch(replay, 0)
    parameters_before = {key: value.clone() for key, value in model.state_dict().items()}

    checkpoint = tmp_path / 'checkpoint.pt'
    save_star_checkpoint(
        checkpoint,
        model=model,
        optimizers={'model': optimizer},
        dual_state={'energy': 0.3, 'global': 1.2},
        normalizer=normalizer,
        counters={'env_steps': 123, 'updates': 99},
        warmup_state={'stage': 'W2'},
        replay=replay,
    )
    expected_random = (random.random(), np.random.random(), torch.rand(2))

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100.0)
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restored_replay = STARReplayBuffer(3, 2, capacity=32, batch_size=4)
    metadata = load_star_checkpoint(
        checkpoint,
        model=model,
        optimizers={'model': optimizer},
        normalizer=normalizer,
        replay=restored_replay,
    )
    actual_random = (random.random(), np.random.random(), torch.rand(2))

    assert all(
        torch.equal(model.state_dict()[key], value) for key, value in parameters_before.items()
    )
    assert optimizer.state_dict()['state']
    assert metadata['duals'] == {'energy': 0.3, 'global': 1.2}
    assert metadata['counters'] == {'env_steps': 123, 'updates': 99}
    assert metadata['warmup'] == {'stage': 'W2'}
    assert metadata['has_replay'] is True
    assert restored_replay.size == replay.size
    assert actual_random[0] == expected_random[0]
    assert actual_random[1] == expected_random[1]
    assert torch.equal(actual_random[2], expected_random[2])


def test_w3_requires_non_temporal_critic_and_geometry_evidence() -> None:
    warmup = STARWarmup(
        random_steps=10,
        terminal_utility_steps=20,
        geometry_min_steps=30,
        td_error_threshold=0.5,
        calibration_threshold=0.2,
        valid_gradient_threshold=0.5,
        disagreement_threshold=0.25,
        boundary_samples_threshold=8,
        positive_labels_threshold=4,
        negative_labels_threshold=4,
        brier_skill_threshold=0.0,
        auroc_threshold=0.6,
        ema_rate=1.0,
    )
    assert warmup.advance(5, 0, 0, 0) == 'W0'
    assert warmup.advance(15, 0, 0, 0) == 'W1'
    assert warmup.advance(10_000, 1000, 100, 100) == 'W2'
    conditions = warmup.conditions(10_000, 1000, 100, 100)
    assert conditions['minimum_steps']
    assert not conditions['td_error']
    assert not conditions['calibration']

    warmup.observe(
        cost_td_error=0.4,
        calibration_error=0.1,
        valid_gradient_rate=0.8,
        ensemble_disagreement=0.1,
        brier_skill=0.1,
        auroc=0.8,
    )
    assert warmup.advance(10_000, 1000, 100, 100) == 'W3'
    assert warmup.geometry_activation_step == 10_000

# Copyright 2026 OmniSafe Team. All Rights Reserved.
"""Analytic correctness tests for P4.4 outward-drift rectification."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from omnisafe.algorithms.off_policy.star import STAR
from omnisafe.common.buffer.star_replay_buffer import STARReplayBuffer
from omnisafe.models.actor.flow_actor import FlowActor, FlowReference, rectify_outward_velocity
from omnisafe.models.safety.distributional_backend import compute_stochastic_reference


ROOT = Path(__file__).resolve().parents[2]


class AnalyticProbabilityCritic(nn.Module):
    """Two identical heads whose safety normal is exactly the first action axis."""

    def probabilities(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del obs
        probability = torch.sigmoid(2.0 * action[..., 0])
        return torch.stack((probability, probability), dim=0)


def _reference(action: torch.Tensor, reliability: float) -> FlowReference:
    batch = action.shape[:-1]
    normal = torch.zeros_like(action)
    normal[..., 0] = 1.0
    normal_projection = normal.unsqueeze(-1) * normal.unsqueeze(-2)
    identity = torch.eye(action.shape[-1], dtype=action.dtype, device=action.device).expand(
        *batch,
        action.shape[-1],
        action.shape[-1],
    )
    tangent_projection = identity - normal_projection
    normal_variance = torch.full(batch, 0.01, dtype=action.dtype, device=action.device)
    tangent_variance = torch.full(batch, 0.07, dtype=action.dtype, device=action.device)
    scalar = torch.zeros(batch, dtype=action.dtype, device=action.device)
    return FlowReference(
        drift=torch.zeros_like(action),
        noise_transform=(
            math.sqrt(0.07) * tangent_projection + math.sqrt(0.01) * normal_projection
        ),
        normal=normal,
        risk_ucb=scalar,
        margin=scalar,
        reliability=torch.full_like(scalar, reliability),
        boundary_reliability=scalar,
        gradient_reliability=scalar,
        ensemble_reliability=scalar,
        calibration_reliability=torch.full_like(scalar, reliability),
        normal_variance=normal_variance,
        tangent_variance=tangent_variance,
    )


def _constant_velocity_actor(velocity: tuple[float, float]) -> FlowActor:
    actor = FlowActor(obs_dim=3, act_dim=2, hidden_sizes=[8], num_steps=4, max_velocity=1.0)
    for parameter in actor.parameters():
        parameter.data.zero_()
    final = [module for module in actor.velocity_net if isinstance(module, nn.Linear)][-1]
    final.bias.data.copy_(torch.atanh(torch.tensor(velocity)))
    return actor


def _store(
    replay: STARReplayBuffer,
    marker: float,
    *,
    terminated: bool,
) -> None:
    scalar = torch.tensor([marker], dtype=torch.float32)
    replay.store_batch(
        obs=scalar.repeat(3).reshape(1, 3),
        action=torch.zeros(1, 2),
        reward=scalar,
        cost=torch.tensor([0.0]),
        terminated=torch.tensor([terminated]),
        truncated=torch.tensor([False]),
        next_obs=scalar.repeat(3).reshape(1, 3),
        context=torch.tensor([[1.0, 1.0, 0.0]]),
        next_context=torch.tensor([[1.0, 1.0, 0.0]]),
        risk_threshold=torch.tensor([1.0]),
        next_risk_threshold=torch.tensor([1.0]),
        boundary_score=torch.tensor([0.0]),
        failure=torch.tensor([False]),
        source_random=torch.tensor([False]),
    )


def test_q_zero_is_bit_exact_with_the_unrectified_stochastic_flow() -> None:
    actor = _constant_velocity_actor((0.5, 0.25))
    obs = torch.zeros(3, 3)
    base = torch.zeros(3, 2)
    noise = torch.zeros(3, 4, 2)
    old = actor(
        obs,
        base_action=base,
        reference=lambda action: _reference(action, 0.0),
        flow_noise=noise,
    )
    rectified = actor(
        obs,
        base_action=base,
        reference=lambda action: _reference(action, 0.0),
        flow_noise=noise,
        outward_rectification_enabled=True,
    )
    for first, second in (
        (old.action, rectified.action),
        (old.points, rectified.points),
        (old.proposals, rectified.proposals),
        (old.velocities, rectified.velocities),
        (old.kinetic_energy, rectified.kinetic_energy),
    ):
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_q_one_removes_only_outward_normal_and_preserves_reward_tangent() -> None:
    reward_direction = torch.tensor([[1.0, 1.0]])
    normal = torch.tensor([[1.0, 0.0]])
    executed = rectify_outward_velocity(reward_direction, normal, torch.ones(1))
    torch.testing.assert_close(executed, torch.tensor([[0.0, 1.0]]))
    assert float((executed * normal).sum().item()) <= 0.0


def test_q_one_leaves_inward_and_tangent_velocity_unchanged() -> None:
    velocity = torch.tensor([[-1.0, 2.0], [1.0, 2.0]])
    normal = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    executed = rectify_outward_velocity(velocity, normal, torch.ones(2))
    torch.testing.assert_close(executed, torch.tensor([[-1.0, 2.0], [0.0, 2.0]]))


def test_rectified_actor_backpropagates_and_uses_executed_energy() -> None:
    actor = _constant_velocity_actor((0.5, 0.25))
    output = actor(
        torch.zeros(4, 3),
        base_action=torch.zeros(4, 2),
        reference=lambda action: _reference(action, 1.0),
        flow_noise=torch.zeros(4, 4, 2),
        outward_rectification_enabled=True,
    )
    raw_normal = output.raw_velocities[..., 0]
    executed_normal = output.velocities[..., 0]
    assert bool((raw_normal > 0.0).all())
    torch.testing.assert_close(
        executed_normal,
        torch.zeros_like(executed_normal),
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(output.action[:, 0], torch.zeros(4), rtol=0.0, atol=1e-7)
    torch.testing.assert_close(
        output.action[:, 1],
        torch.full((4,), 0.25),
        rtol=0.0,
        atol=1e-7,
    )
    expected_energy = 0.5 * 0.25**2 / 0.07
    torch.testing.assert_close(output.kinetic_energy, torch.full((4,), expected_energy))
    (output.action[:, 1].mean() + output.kinetic_energy.mean()).backward()
    assert all(parameter.grad is not None for parameter in actor.parameters())
    assert all(bool(torch.isfinite(parameter.grad).all()) for parameter in actor.parameters())


def test_insufficient_complete_heldout_evidence_forces_isotropic_reference() -> None:
    algorithm = object.__new__(STAR)
    algorithm._cfgs = SimpleNamespace(  # pylint: disable=protected-access
        algo_cfgs=SimpleNamespace(
            complete_heldout_evidence=True,
            safety_calibration_auroc_full=0.75,
            safety_calibration_brier_score_scale=0.25,
        ),
    )
    algorithm._last_heldout_evidence = {  # pylint: disable=protected-access
        'Heldout/AUROC': 0.95,
        'Heldout/BrierScore': 0.01,
        'Heldout/EvidenceSufficient': 0.0,
    }
    calibration = algorithm._calibration_reliability()  # pylint: disable=protected-access
    assert calibration == 0.0
    reference = compute_stochastic_reference(
        AnalyticProbabilityCritic(),
        torch.zeros(3, 4),
        torch.zeros(3, 2),
        anisotropy_enabled=True,
        inward_drift_enabled=False,
        calibration=calibration,
        risk_ucb_beta=0.0,
    )
    torch.testing.assert_close(reference.normal_variance, torch.full((3,), 0.04))
    torch.testing.assert_close(reference.tangent_variance, torch.full((3,), 0.04))


def test_heldout_transition_limit_discards_whole_oldest_episodes() -> None:
    replay = STARReplayBuffer(
        3,
        2,
        capacity=32,
        batch_size=4,
        context_dim=3,
        heldout_episode_period=1,
        pin_memory=False,
    )
    for episode in range(3):
        for step in range(3):
            _store(replay, 10.0 * episode + step, terminated=step == 2)
    heldout = replay.recent_heldout_batch(recent_episodes=3, batch_size=4)
    assert heldout['obs'].shape[0] == 3
    assert set(heldout['episode_id'].tolist()) == {2}
    with pytest.raises(RuntimeError, match='complete held-out episode exceeds'):
        replay.recent_heldout_batch(recent_episodes=3, batch_size=2)


def test_stochastic_metrics_record_raw_executed_and_clipping_variance() -> None:
    actor = _constant_velocity_actor((0.9, 0.25))
    noise = torch.zeros(4, 4, 2)
    noise[0, :, 0] = 10.0
    output = actor(
        torch.zeros(4, 3),
        base_action=torch.tensor([[0.95, 0.0]]).expand(4, -1),
        reference=lambda action: _reference(action, 1.0),
        flow_noise=noise,
        outward_rectification_enabled=True,
    )
    algorithm = object.__new__(STAR)
    algorithm._act_dim = 2  # pylint: disable=protected-access
    algorithm._cfgs = SimpleNamespace(  # pylint: disable=protected-access
        algo_cfgs=SimpleNamespace(flow_reference_variance=0.04),
    )
    metrics = algorithm._stochastic_reference_metrics(output)  # pylint: disable=protected-access
    assert metrics['Reference/RawDangerousNormalVelocity'] > 0.0
    assert metrics['Reference/DangerousNormalVelocity'] == pytest.approx(0.0, abs=1e-7)
    assert 'Reference/PreClippingVariance' in metrics
    assert 'Reference/PostClippingVariance' in metrics
    assert metrics['Reference/ClippingRate'] > 0.0
    assert metrics['Reference/PreClippingVariance'] > metrics['Reference/PostClippingVariance']


def test_r0_r1_configs_differ_only_by_outward_rectification() -> None:
    configs = []
    for name in ('p4_4_r0_complete_heldout.yaml', 'p4_4_r1_outward_rectification.yaml'):
        with (ROOT / 'configs/star' / name).open('r', encoding='utf-8') as stream:
            configs.append(yaml.safe_load(stream))
    r0, r1 = configs
    r0_algo = r0['custom_cfgs']['algo_cfgs']
    r1_algo = r1['custom_cfgs']['algo_cfgs']
    assert r0_algo['safety_outward_rectification_enabled'] is False
    assert r1_algo['safety_outward_rectification_enabled'] is True
    assert r0_algo['complete_heldout_evidence'] is True
    assert r0_algo['heldout_eval_batch_size'] == 20_000
    assert r0_algo['safety_protocol_sha256'].startswith('005c4234')
    r0_algo.pop('safety_outward_rectification_enabled')
    r1_algo.pop('safety_outward_rectification_enabled')
    assert r0 == r1


def test_reward_td_combined_tail_rule_and_mild_rise_warning(tmp_path: Path) -> None:
    algorithm = object.__new__(STAR)
    algorithm._cfgs = SimpleNamespace(  # pylint: disable=protected-access
        algo_cfgs=SimpleNamespace(
            numerical_hard_stops_enabled=True,
            reward_td_absolute_hard_stop=1.0,
            reward_q_absolute_hard_stop=10.0,
            reward_td_final_to_minimum_ratio_hard_stop=3.0,
        ),
    )
    algorithm._artifact_dir = tmp_path  # pylint: disable=protected-access
    algorithm._env_steps = 5_000  # pylint: disable=protected-access
    algorithm._reward_td_diagnostics = []  # pylint: disable=protected-access
    for value in (0.01, 0.011, 0.012, 0.013):
        algorithm._check_numerical_hard_stops(  # pylint: disable=protected-access
            {
                'Critic/RewardTD': value,
                'Critic/RewardQAbsMax': 1.0,
                'Critic/RewardTargetQAbsMax': 1.0,
            },
        )
    status = (tmp_path / 'numerical_status_latest.json').read_text(encoding='utf-8')
    assert 'positive_last_four_slope_without_ratio_trigger' in status

    algorithm._reward_td_diagnostics = []  # pylint: disable=protected-access
    with pytest.raises(FloatingPointError, match='ratio_and_positive'):
        for value in (0.001, 0.002, 0.003, 0.004):
            algorithm._check_numerical_hard_stops(  # pylint: disable=protected-access
                {
                    'Critic/RewardTD': value,
                    'Critic/RewardQAbsMax': 1.0,
                    'Critic/RewardTargetQAbsMax': 1.0,
                },
            )

"""Semantic gates for the STAR P1 environment and off-policy replay contract."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from omnisafe.adapter import OffPolicyAdapter
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.envs.core import CMDP, env_register, make
from omnisafe.utils.config import Config, get_default_kwargs_yaml


@env_register
class STARReplayProbeEnv(CMDP):
    """One-step environment that exposes terminal observations for replay tests."""

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False
    _support_envs: ClassVar[list[str]] = [
        'STARTruncationProbe-v0',
        'STARTerminationProbe-v0',
        'STARBothDoneProbe-v0',
    ]

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: str | torch.device = 'cpu',
        **_: Any,
    ) -> None:
        super().__init__(env_id)
        assert num_envs == 1
        self._env_id = env_id
        self._device = torch.device(device)
        self._num_envs = 1
        self._action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._observation_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self._metadata = {}
        self._reset_count = 0
        self._current_obs = torch.zeros(1, dtype=torch.float32, device=self._device)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        del action
        final_obs = self._current_obs + 1.0
        terminated = self._env_id != 'STARTruncationProbe-v0'
        truncated = self._env_id != 'STARTerminationProbe-v0'
        return (
            final_obs,
            torch.tensor(0.0, device=self._device),
            torch.tensor(float(terminated), device=self._device),
            torch.tensor(terminated, device=self._device),
            torch.tensor(truncated, device=self._device),
            {'probe_final_observation': final_obs.clone()},
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del seed, options
        self._reset_count += 1
        self._current_obs = torch.tensor(
            [float(self._reset_count * 10)],
            dtype=torch.float32,
            device=self._device,
        )
        return self._current_obs.clone(), {}

    def set_seed(self, seed: int) -> None:
        del seed
        self._reset_count = 0
        self.reset()

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


class StoreOnlyLogger:
    """Minimal logger surface used by ``OffPolicyAdapter.rollout``."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def store(self, data: dict[str, Any]) -> None:
        self.rows.append(data)


class UnusedAgent:
    """The replay probe uses random actions, so policy calls are an error."""

    def step(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        del obs, deterministic
        raise AssertionError('policy should not be called by this test')


def offpolicy_config() -> Config:
    """Return an unnormalized CPU configuration for adapter-level tests."""
    cfg = get_default_kwargs_yaml('SACLag', 'SafetyPointGoal1-v0', 'off-policy')
    cfg.train_cfgs.recurisve_update({'device': 'cpu', 'torch_threads': 1})
    cfg.algo_cfgs.recurisve_update(
        {'obs_normalize': False, 'reward_normalize': False, 'cost_normalize': False},
    )
    return cfg


@pytest.mark.parametrize(
    ('env_id', 'height_index', 'fall_height'),
    [('Ant-v4', 2, -2.0), ('Walker2d-v4', 1, 0.05)],
)
def test_unhealthy_fall_is_failure(
    env_id: str,
    height_index: int,
    fall_height: float,
) -> None:
    env = make(env_id, class_name='FailureCostMujocoEnv')
    try:
        env.reset(seed=7)
        qpos, qvel = env.get_sim_state()
        qpos[height_index] = fall_height
        qvel.fill(0.0)
        env.set_sim_state(qpos, qvel)
        _, _, cost, terminated, truncated, info = env.step(
            torch.zeros(env.action_space.shape),
        )
        assert bool(terminated)
        assert not bool(truncated)
        assert cost.item() == 1.0
        assert info['is_failure'] is True
        assert info['failure_reason'] == 'unhealthy_height'
        assert info['health_height'] < 0.2
        assert info['episode_step'] == 1
        assert 'torso_orientation' in info
        assert 'torso_upright' in info
    finally:
        env.close()


def test_walker_unhealthy_angle_reason() -> None:
    env = make('Walker2d-v4', class_name='FailureCostMujocoEnv')
    try:
        env.reset(seed=9)
        qpos, qvel = env.get_sim_state()
        qpos[1] = 1.2
        qpos[2] = 1.5
        qvel.fill(0.0)
        env.set_sim_state(qpos, qvel)
        _, _, cost, terminated, _, info = env.step(torch.zeros(env.action_space.shape))
        assert bool(terminated)
        assert cost.item() == 1.0
        assert info['failure_reason'] == 'unhealthy_angle'
        assert abs(info['torso_angle']) > 1.0
    finally:
        env.close()


@pytest.mark.parametrize('env_id', ['Ant-v4', 'Walker2d-v4'])
def test_time_limit_is_not_failure(env_id: str) -> None:
    env = make(env_id, class_name='FailureCostMujocoEnv', max_episode_steps=2)
    try:
        env.reset(seed=11)
        action = torch.zeros(env.action_space.shape)
        env.step(action)
        _, _, cost, terminated, truncated, info = env.step(action)
        assert not bool(terminated)
        assert bool(truncated)
        assert cost.item() == 0.0
        assert info['is_failure'] is False
        assert info['failure_reason'] == 'time_limit'
        assert info['time_limit_reached'] is True
        assert info['episode_step'] == 2
    finally:
        env.close()


def test_failure_label_survives_simultaneous_time_limit() -> None:
    env = make('Ant-v4', class_name='FailureCostMujocoEnv', max_episode_steps=1)
    try:
        env.reset(seed=13)
        qpos, qvel = env.get_sim_state()
        qpos[2] = -2.0
        qvel.fill(0.0)
        env.set_sim_state(qpos, qvel)
        _, _, cost, terminated, truncated, info = env.step(
            torch.zeros(env.action_space.shape),
        )
        assert bool(terminated)
        assert bool(truncated)
        assert cost.item() == 1.0
        assert info['failure_reason'] == 'unhealthy_height'
        assert info['time_limit_reached'] is True
    finally:
        env.close()


@pytest.mark.parametrize(
    ('env_id', 'expected_terminated', 'expected_truncated'),
    [
        ('STARTruncationProbe-v0', 0.0, 1.0),
        ('STARTerminationProbe-v0', 1.0, 0.0),
        ('STARBothDoneProbe-v0', 1.0, 1.0),
    ],
)
def test_replay_uses_final_observation_and_true_termination_bootstrap(
    env_id: str,
    expected_terminated: float,
    expected_truncated: float,
) -> None:
    adapter = OffPolicyAdapter(env_id, num_envs=1, seed=23, cfgs=offpolicy_config())
    buffer = VectorOffPolicyBuffer(
        obs_space=adapter.observation_space,
        act_space=adapter.action_space,
        size=8,
        batch_size=2,
        num_envs=1,
    )
    try:
        obs_before = adapter._current_obs.clone()  # pylint: disable=protected-access
        expected_final_obs = obs_before + 1.0
        adapter.rollout(
            rollout_step=1,
            agent=UnusedAgent(),  # type: ignore[arg-type]
            buffer=buffer,
            logger=StoreOnlyLogger(),  # type: ignore[arg-type]
            use_rand_action=True,
        )
        assert torch.equal(buffer.data['next_obs'][0, 0], expected_final_obs[0])
        assert not torch.equal(
            buffer.data['next_obs'][0, 0],
            adapter._current_obs[0],  # pylint: disable=protected-access
        )
        assert buffer.data['terminated'][0, 0].item() == expected_terminated
        assert buffer.data['truncated'][0, 0].item() == expected_truncated
        assert buffer.data['done'][0, 0].item() == expected_terminated
    finally:
        adapter.close()


def test_seed_is_reproducible_without_repeating_consecutive_resets() -> None:
    first = make('Ant-v4', class_name='FailureCostMujocoEnv')
    second = make('Ant-v4', class_name='FailureCostMujocoEnv')
    try:
        first.set_seed(31)
        second.set_seed(31)
        first_episode_one, _ = first.reset()
        first_episode_two, _ = first.reset()
        second_episode_one, _ = second.reset()
        second_episode_two, _ = second.reset()
        assert torch.equal(first_episode_one, second_episode_one)
        assert torch.equal(first_episode_two, second_episode_two)
        assert not torch.equal(first_episode_one, first_episode_two)
    finally:
        first.close()
        second.close()


def test_eval_environment_does_not_mutate_training_observation() -> None:
    adapter = OffPolicyAdapter('Ant-v4', num_envs=1, seed=37, cfgs=offpolicy_config())
    try:
        train_obs = adapter._current_obs.clone()  # pylint: disable=protected-access
        assert adapter._eval_env is not None  # pylint: disable=protected-access
        eval_env = adapter._eval_env  # pylint: disable=protected-access
        eval_env.reset()
        for _ in range(5):
            eval_env.step(torch.zeros((1, *eval_env.action_space.shape)))
        assert torch.equal(train_obs, adapter._current_obs)  # pylint: disable=protected-access
    finally:
        adapter.close()


def test_eval_seed_stream_is_reproducible_and_disjoint_from_training() -> None:
    first = OffPolicyAdapter('Ant-v4', num_envs=1, seed=41, cfgs=offpolicy_config())
    second = OffPolicyAdapter('Ant-v4', num_envs=1, seed=41, cfgs=offpolicy_config())
    try:
        assert torch.equal(
            first._current_obs,  # pylint: disable=protected-access
            second._current_obs,  # pylint: disable=protected-access
        )
        assert first._eval_env is not None  # pylint: disable=protected-access
        assert second._eval_env is not None  # pylint: disable=protected-access
        first_eval_obs, _ = first._eval_env.reset()  # pylint: disable=protected-access
        second_eval_obs, _ = second._eval_env.reset()  # pylint: disable=protected-access
        assert torch.equal(first_eval_obs, second_eval_obs)
        assert not torch.equal(
            first._current_obs,  # pylint: disable=protected-access
            first_eval_obs,
        )
    finally:
        first.close()
        second.close()

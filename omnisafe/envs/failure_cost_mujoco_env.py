# Copyright 2026 SAFEFLOW2027 Authors. All Rights Reserved.
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
"""Failure-cost CMDPs for the Gymnasium Ant and Walker2d environments."""

from __future__ import annotations

import math
from typing import Any, ClassVar

import gymnasium
import numpy as np
import torch

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import Box


@env_register
class FailureCostMujocoEnv(CMDP):
    """Map only unhealthy termination to a binary safety failure cost.

    Gymnasium's own ``TimeLimit`` wrapper remains responsible for truncation.  A time-limit
    transition therefore has ``truncated=True`` and ``cost=0``.  If unhealthy termination and
    the time limit coincide, the failure label is retained and ``cost=1``.
    """

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False
    need_action_repeat_wrapper = True
    _support_envs: ClassVar[list[str]] = ['Ant-v4', 'Walker2d-v4']

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: str | torch.device = 'cpu',
        **kwargs: Any,
    ) -> None:
        """Initialize a single failure-cost MuJoCo environment."""
        super().__init__(env_id)
        if num_envs != 1:
            raise NotImplementedError('FailureCostMujocoEnv currently supports num_envs=1 only.')
        if kwargs.get('terminate_when_unhealthy', True) is not True:
            raise ValueError('terminate_when_unhealthy must remain enabled for failure costs')
        kwargs['terminate_when_unhealthy'] = True

        self._env_id = env_id
        self._env = gymnasium.make(id=env_id, autoreset=False, **kwargs)
        if not isinstance(self._env.action_space, Box):
            raise TypeError('Only Box action spaces are supported.')
        if not isinstance(self._env.observation_space, Box):
            raise TypeError('Only Box observation spaces are supported.')
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space
        self._device = torch.device(device)
        self._num_envs = 1
        self._metadata = self._env.metadata
        self._episode_step = 0

    @property
    def max_episode_steps(self) -> int:
        """Return Gymnasium's configured time limit."""
        if self._env.spec is None or self._env.spec.max_episode_steps is None:
            raise RuntimeError(f'{self._env_id} has no max_episode_steps')
        return self._env.spec.max_episode_steps

    def _failure_reason(self, qpos: np.ndarray) -> str:
        """Classify an unhealthy state without changing the binary cost label."""
        if not np.isfinite(qpos).all():
            return 'unhealthy_nonfinite_state'
        base_env = self._env.unwrapped
        height = float(qpos[2] if self._env_id == 'Ant-v4' else qpos[1])
        healthy_z_range = getattr(base_env, '_healthy_z_range', None)
        if healthy_z_range is not None and not healthy_z_range[0] < height < healthy_z_range[1]:
            return 'unhealthy_height'
        if self._env_id == 'Walker2d-v4':
            angle = float(qpos[2])
            healthy_angle_range = getattr(base_env, '_healthy_angle_range', None)
            if (
                healthy_angle_range is not None
                and not healthy_angle_range[0] < angle < healthy_angle_range[1]
            ):
                return 'unhealthy_angle'
        return 'unhealthy_state'

    def _diagnostics(
        self,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        """Return health, pose, time, and failure diagnostics for ``info``."""
        base_env = self._env.unwrapped
        qpos = np.asarray(base_env.data.qpos, dtype=np.float64).copy()
        is_healthy = bool(base_env.is_healthy)
        height = float(qpos[2] if self._env_id == 'Ant-v4' else qpos[1])
        if self._env_id == 'Ant-v4':
            quaternion = qpos[3:7]
            norm = float(np.linalg.norm(quaternion))
            if norm > 0 and math.isfinite(norm):
                quaternion = quaternion / norm
            x_value = float(quaternion[1])
            y_value = float(quaternion[2])
            torso_upright = 1.0 - 2.0 * (x_value**2 + y_value**2)
            torso_angle = math.acos(float(np.clip(torso_upright, -1.0, 1.0)))
            orientation: float | list[float] = [float(value) for value in quaternion]
        else:
            torso_angle = float(qpos[2])
            torso_upright = math.cos(torso_angle)
            orientation = torso_angle

        failure = terminated and not is_healthy
        if failure:
            failure_reason = self._failure_reason(qpos)
        elif truncated:
            failure_reason = 'time_limit'
        elif terminated:
            failure_reason = 'non_health_termination'
        else:
            failure_reason = 'none'

        return {
            'failure_reason': failure_reason,
            'is_failure': failure,
            'is_healthy': is_healthy,
            'health_height': height,
            'torso_angle': torso_angle,
            'torso_orientation': orientation,
            'torso_upright': torso_upright,
            'episode_step': self._episode_step,
            'episode_time': self._episode_step * float(base_env.dt),
            'time_limit_reached': truncated,
        }

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
        """Step the environment and attach the binary failure cost."""
        obs, reward, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )
        self._episode_step += 1
        terminated_value = bool(terminated)
        truncated_value = bool(truncated)
        diagnostics = self._diagnostics(terminated_value, truncated_value)
        info.update(diagnostics)
        cost = float(diagnostics['is_failure'])
        return (
            torch.as_tensor(obs, dtype=torch.float32, device=self._device),
            torch.as_tensor(reward, dtype=torch.float32, device=self._device),
            torch.as_tensor(cost, dtype=torch.float32, device=self._device),
            torch.as_tensor(terminated_value, dtype=torch.bool, device=self._device),
            torch.as_tensor(truncated_value, dtype=torch.bool, device=self._device),
            info,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset without reusing a seed unless the caller explicitly supplies one."""
        obs, info = self._env.reset(seed=seed, options=options)
        self._episode_step = 0
        info.update(self._diagnostics(terminated=False, truncated=False))
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    def set_seed(self, seed: int) -> None:
        """Seed the environment once; later automatic resets advance its RNG state."""
        self.reset(seed=seed)

    def get_sim_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Copy the MuJoCo position and velocity state for semantic tests."""
        base_env = self._env.unwrapped
        return base_env.data.qpos.copy(), base_env.data.qvel.copy()

    def set_sim_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Set the MuJoCo state for controlled failure tests."""
        self._env.unwrapped.set_state(np.asarray(qpos), np.asarray(qvel))

    def render(self) -> Any:
        """Render using Gymnasium's configured render mode."""
        return self._env.render()

    def close(self) -> None:
        """Close the underlying environment."""
        self._env.close()

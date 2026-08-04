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
"""Energy and true-episode safety duals used by STAR route A."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn, optim


class EnergyDual(nn.Module):
    """Positive path-energy multiplier optimized in log space."""

    def __init__(
        self,
        initial_value: float,
        learning_rate: float,
        minimum_log: float = -10.0,
        maximum_log: float = 5.0,
    ) -> None:
        """Initialize the log multiplier and its independent optimizer."""
        super().__init__()
        if initial_value <= 0 or learning_rate <= 0 or minimum_log >= maximum_log:
            raise ValueError('energy dual initialization is invalid')
        self.log_multiplier = nn.Parameter(torch.tensor(math.log(initial_value)))
        self.minimum_log = minimum_log
        self.maximum_log = maximum_log
        self.optimizer = optim.Adam([self.log_multiplier], lr=learning_rate)

    @property
    def value(self) -> torch.Tensor:
        """Return the positive energy multiplier."""
        return self.log_multiplier.exp()

    def update(self, path_energy: torch.Tensor, target_energy: float) -> torch.Tensor:
        """Increase the multiplier above target and decrease it below target."""
        loss = self.log_multiplier * (target_energy - path_energy.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.log_multiplier.clamp_(self.minimum_log, self.maximum_log)
        return loss.detach()


class GlobalSafetyDual:
    """Projected CMDP multiplier updated only from completed true episode costs."""

    def __init__(
        self,
        initial_value: float,
        learning_rate: float,
        ema_rate: float,
        cost_limit: float,
        maximum_value: float = 100.0,
    ) -> None:
        """Initialize projected multiplier and episode-cost moving average."""
        if initial_value < 0 or learning_rate <= 0 or not 0 < ema_rate <= 1:
            raise ValueError('global safety dual initialization is invalid')
        if maximum_value <= 0:
            raise ValueError('global safety dual maximum must be positive')
        self.value = float(initial_value)
        self.learning_rate = float(learning_rate)
        self.ema_rate = float(ema_rate)
        self.cost_limit = float(cost_limit)
        self.maximum_value = float(maximum_value)
        self.ema_episode_cost: float | None = None
        self.episode_updates = 0

    def update_episode(self, true_episode_cost: float) -> float:
        """Project one update using a real completed episode cost."""
        if not math.isfinite(true_episode_cost):
            raise ValueError('episode cost must be finite')
        if self.ema_episode_cost is None:
            self.ema_episode_cost = true_episode_cost
        else:
            self.ema_episode_cost = (
                1.0 - self.ema_rate
            ) * self.ema_episode_cost + self.ema_rate * true_episode_cost
        self.value += self.learning_rate * (self.ema_episode_cost - self.cost_limit)
        self.value = min(max(self.value, 0.0), self.maximum_value)
        self.episode_updates += 1
        return self.value

    def state_dict(self) -> dict[str, Any]:
        """Return exact projected-dual state."""
        return {
            'value': self.value,
            'ema_episode_cost': self.ema_episode_cost,
            'episode_updates': self.episode_updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore exact projected-dual state."""
        self.value = float(state['value'])
        ema = state['ema_episode_cost']
        self.ema_episode_cost = None if ema is None else float(ema)
        self.episode_updates = int(state['episode_updates'])

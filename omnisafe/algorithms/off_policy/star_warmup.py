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
"""Evidence-gated W0-W3 warmup state machine for STAR."""

from __future__ import annotations

import math
from typing import Any


class STARWarmup:  # pylint: disable=too-many-instance-attributes
    """Advance directional geometry only after critic and boundary evidence is usable."""

    STAGES = ('W0', 'W1', 'W2', 'W3')

    def __init__(  # pylint: disable=too-many-arguments
        self,
        random_steps: int,
        terminal_utility_steps: int,
        geometry_min_steps: int,
        td_error_threshold: float,
        calibration_threshold: float,
        valid_gradient_threshold: float,
        disagreement_threshold: float,
        boundary_samples_threshold: int,
        ema_rate: float = 0.05,
    ) -> None:
        """Initialize ordered step gates and non-temporal W3 thresholds."""
        if not 0 <= random_steps <= terminal_utility_steps <= geometry_min_steps:
            raise ValueError('warmup step boundaries must be ordered')
        if not 0 < ema_rate <= 1:
            raise ValueError('warmup EMA rate must lie in (0, 1]')
        self.random_steps = random_steps
        self.terminal_utility_steps = terminal_utility_steps
        self.geometry_min_steps = geometry_min_steps
        self.td_error_threshold = td_error_threshold
        self.calibration_threshold = calibration_threshold
        self.valid_gradient_threshold = valid_gradient_threshold
        self.disagreement_threshold = disagreement_threshold
        self.boundary_samples_threshold = boundary_samples_threshold
        self.ema_rate = ema_rate
        self.stage = 'W0'
        self.geometry_activation_step: int | None = None
        self.ema: dict[str, float | None] = {
            'cost_td_error': None,
            'calibration_error': None,
            'valid_gradient_rate': None,
            'ensemble_disagreement': None,
        }

    def observe(self, **metrics: float) -> None:
        """Update finite evidence metrics without treating missing labels as success."""
        for name, value in metrics.items():
            if name not in self.ema:
                raise KeyError(f'unknown warmup diagnostic {name!r}')
            if not math.isfinite(value):
                continue
            previous = self.ema[name]
            self.ema[name] = (
                value
                if previous is None
                else (1.0 - self.ema_rate) * previous + self.ema_rate * value
            )

    def conditions(self, env_steps: int, boundary_samples: int) -> dict[str, bool]:
        """Return every independently auditable W3 activation condition."""
        td_error = self.ema['cost_td_error']
        calibration = self.ema['calibration_error']
        valid_gradient = self.ema['valid_gradient_rate']
        disagreement = self.ema['ensemble_disagreement']
        return {
            'minimum_steps': env_steps >= self.geometry_min_steps,
            'td_error': td_error is not None and td_error <= self.td_error_threshold,
            'calibration': calibration is not None and calibration <= self.calibration_threshold,
            'valid_gradient': valid_gradient is not None
            and valid_gradient >= self.valid_gradient_threshold,
            'ensemble_disagreement': disagreement is not None
            and disagreement <= self.disagreement_threshold,
            'boundary_samples': boundary_samples >= self.boundary_samples_threshold,
        }

    def advance(self, env_steps: int, boundary_samples: int) -> str:
        """Advance monotonically through W0, W1, W2, and evidence-gated W3."""
        if env_steps < self.random_steps:
            self.stage = 'W0'
        elif env_steps < self.terminal_utility_steps:
            self.stage = 'W1'
        elif self.stage != 'W3':
            self.stage = 'W2'
            if all(self.conditions(env_steps, boundary_samples).values()):
                self.stage = 'W3'
                self.geometry_activation_step = env_steps
        return self.stage

    @property
    def stage_index(self) -> int:
        """Return a scalar stage value for tabular logging."""
        return self.STAGES.index(self.stage)

    def state_dict(self) -> dict[str, Any]:
        """Return all dynamic warmup evidence for exact checkpoint recovery."""
        return {
            'stage': self.stage,
            'geometry_activation_step': self.geometry_activation_step,
            'ema': dict(self.ema),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore dynamic warmup evidence."""
        stage = state['stage']
        if stage not in self.STAGES:
            raise ValueError(f'unknown STAR warmup stage {stage!r}')
        self.stage = stage
        self.geometry_activation_step = state['geometry_activation_step']
        self.ema = dict(state['ema'])

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
        positive_labels_threshold: int,
        negative_labels_threshold: int,
        brier_skill_threshold: float,
        auroc_threshold: float,
        ema_rate: float = 0.05,
        dynamic_geometry_gate: bool = False,
        geometry_enable_passes: int = 3,
        geometry_disable_failures: int = 2,
    ) -> None:
        """Initialize ordered step gates and non-temporal W3 thresholds."""
        if not 0 <= random_steps <= terminal_utility_steps <= geometry_min_steps:
            raise ValueError('warmup step boundaries must be ordered')
        if not 0 < ema_rate <= 1:
            raise ValueError('warmup EMA rate must lie in (0, 1]')
        if (
            min(boundary_samples_threshold, positive_labels_threshold, negative_labels_threshold)
            < 0
        ):
            raise ValueError('warmup sample thresholds must be non-negative')
        if not 0 <= valid_gradient_threshold <= 1:
            raise ValueError('valid-gradient threshold must lie in [0, 1]')
        if not 0 <= disagreement_threshold <= 1:
            raise ValueError('disagreement threshold must lie in [0, 1]')
        if not 0 <= auroc_threshold <= 1:
            raise ValueError('AUROC threshold must lie in [0, 1]')
        if geometry_enable_passes <= 0 or geometry_disable_failures <= 0:
            raise ValueError('geometry hysteresis counts must be positive')
        self.random_steps = random_steps
        self.terminal_utility_steps = terminal_utility_steps
        self.geometry_min_steps = geometry_min_steps
        self.td_error_threshold = td_error_threshold
        self.calibration_threshold = calibration_threshold
        self.valid_gradient_threshold = valid_gradient_threshold
        self.disagreement_threshold = disagreement_threshold
        self.boundary_samples_threshold = boundary_samples_threshold
        self.positive_labels_threshold = positive_labels_threshold
        self.negative_labels_threshold = negative_labels_threshold
        self.brier_skill_threshold = brier_skill_threshold
        self.auroc_threshold = auroc_threshold
        self.ema_rate = ema_rate
        self.dynamic_geometry_gate = dynamic_geometry_gate
        self.geometry_enable_passes = geometry_enable_passes
        self.geometry_disable_failures = geometry_disable_failures
        self.stage = 'W0'
        self.geometry_enabled = False
        self.geometry_activation_step: int | None = None
        self.geometry_eligibility_step: int | None = None
        self.geometry_last_disable_step: int | None = None
        self.consecutive_passes = 0
        self.consecutive_failures = 0
        self.gate_evaluations = 0
        self.ema: dict[str, float | None] = {
            'cost_td_error': None,
            'calibration_error': None,
            'valid_gradient_rate': None,
            'ensemble_disagreement': None,
            'brier_skill': None,
            'auroc': None,
        }
        self.latest: dict[str, float | None] = dict(self.ema)

    def observe(self, **metrics: float) -> None:
        """Update finite evidence metrics without treating missing labels as success."""
        for name, value in metrics.items():
            if name not in self.ema:
                raise KeyError(f'unknown warmup diagnostic {name!r}')
            if not math.isfinite(value):
                self.latest[name] = None
                continue
            self.latest[name] = value
            previous = self.ema[name]
            self.ema[name] = (
                value
                if previous is None
                else (1.0 - self.ema_rate) * previous + self.ema_rate * value
            )

    def conditions(
        self,
        env_steps: int,
        boundary_samples: int,
        positive_labels: int,
        negative_labels: int,
    ) -> dict[str, bool]:
        """Return every independently auditable W3 activation condition."""
        td_error = self.ema['cost_td_error']
        calibration = self.ema['calibration_error']
        valid_gradient = self.ema['valid_gradient_rate']
        disagreement = self.ema['ensemble_disagreement']
        brier_skill = self.ema['brier_skill']
        auroc = self.ema['auroc']
        return {
            'minimum_steps': env_steps >= self.geometry_min_steps,
            'td_error': td_error is not None and td_error <= self.td_error_threshold,
            'calibration': calibration is not None and calibration <= self.calibration_threshold,
            'valid_gradient': valid_gradient is not None
            and valid_gradient >= self.valid_gradient_threshold,
            'ensemble_disagreement': disagreement is not None
            and disagreement <= self.disagreement_threshold,
            'boundary_samples': boundary_samples >= self.boundary_samples_threshold,
            'positive_labels': positive_labels >= self.positive_labels_threshold,
            'negative_labels': negative_labels >= self.negative_labels_threshold,
            'brier_skill': brier_skill is not None and brier_skill > self.brier_skill_threshold,
            'auroc': auroc is not None and auroc >= self.auroc_threshold,
        }

    def gate_conditions(
        self,
        env_steps: int,
        boundary_samples: int,
        positive_labels: int,
        negative_labels: int,
    ) -> dict[str, bool]:
        """Return exactly the evidence read by the configured geometry gate."""
        if not self.dynamic_geometry_gate:
            return self.conditions(
                env_steps,
                boundary_samples,
                positive_labels,
                negative_labels,
            )
        gradient_norm = self.latest['valid_gradient_rate']
        disagreement = self.latest['ensemble_disagreement']
        brier_skill = self.latest['brier_skill']
        auroc = self.latest['auroc']
        return {
            'minimum_steps': env_steps >= self.geometry_min_steps,
            'valid_gradient': gradient_norm is not None
            and gradient_norm >= self.valid_gradient_threshold,
            'ensemble_disagreement': disagreement is not None
            and disagreement <= self.disagreement_threshold,
            'boundary_samples': boundary_samples >= self.boundary_samples_threshold,
            'positive_labels': positive_labels >= self.positive_labels_threshold,
            'negative_labels': negative_labels >= self.negative_labels_threshold,
            'brier_skill': brier_skill is not None and brier_skill > self.brier_skill_threshold,
            'auroc': auroc is not None and auroc >= self.auroc_threshold,
        }

    def advance(
        self,
        env_steps: int,
        boundary_samples: int,
        positive_labels: int,
        negative_labels: int,
    ) -> str:
        """Advance monotonically through W0, W1, W2, and evidence-gated W3."""
        if env_steps >= self.random_steps and self.stage == 'W0':
            self.stage = 'W1'
        if env_steps >= self.terminal_utility_steps and self.stage in {'W0', 'W1'}:
            self.stage = 'W2'
        if (
            self.stage == 'W2'
            and not self.dynamic_geometry_gate
            and all(
                self.gate_conditions(
                    env_steps,
                    boundary_samples,
                    positive_labels,
                    negative_labels,
                ).values(),
            )
        ):
            self.stage = 'W3'
            self.geometry_enabled = True
            self.geometry_activation_step = env_steps
            self.geometry_eligibility_step = env_steps
        return self.stage

    def evaluate_geometry_gate(  # pylint: disable=too-many-arguments
        self,
        env_steps: int,
        boundary_samples: int,
        positive_labels: int,
        negative_labels: int,
        **heldout_metrics: float,
    ) -> bool:
        """Apply one held-out-only hysteresis observation to the dynamic geometry switch."""
        if not self.dynamic_geometry_gate:
            raise RuntimeError('held-out geometry evaluation requires dynamic gating')
        self.observe(**heldout_metrics)
        self.advance(env_steps, boundary_samples, positive_labels, negative_labels)
        self.gate_evaluations += 1
        passed = all(
            self.gate_conditions(
                env_steps,
                boundary_samples,
                positive_labels,
                negative_labels,
            ).values(),
        )
        if passed:
            if self.stage == 'W2':
                self.stage = 'W3'
                self.geometry_eligibility_step = env_steps
            self.consecutive_passes += 1
            self.consecutive_failures = 0
            if not self.geometry_enabled and self.consecutive_passes >= self.geometry_enable_passes:
                self.geometry_enabled = True
                self.geometry_activation_step = env_steps
        else:
            self.consecutive_passes = 0
            if self.geometry_enabled:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.geometry_disable_failures:
                    self.geometry_enabled = False
                    self.geometry_last_disable_step = env_steps
                    self.consecutive_failures = 0
            else:
                self.consecutive_failures = min(
                    self.consecutive_failures + 1,
                    self.geometry_disable_failures,
                )
        return passed

    @property
    def stage_index(self) -> int:
        """Return a scalar stage value for tabular logging."""
        return self.STAGES.index(self.stage)

    def state_dict(self) -> dict[str, Any]:
        """Return all dynamic warmup evidence for exact checkpoint recovery."""
        return {
            'stage': self.stage,
            'geometry_enabled': self.geometry_enabled,
            'geometry_activation_step': self.geometry_activation_step,
            'geometry_eligibility_step': self.geometry_eligibility_step,
            'geometry_last_disable_step': self.geometry_last_disable_step,
            'consecutive_passes': self.consecutive_passes,
            'consecutive_failures': self.consecutive_failures,
            'gate_evaluations': self.gate_evaluations,
            'ema': dict(self.ema),
            'latest': dict(self.latest),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore dynamic warmup evidence."""
        stage = state['stage']
        if stage not in self.STAGES:
            raise ValueError(f'unknown STAR warmup stage {stage!r}')
        self.stage = stage
        self.geometry_activation_step = state['geometry_activation_step']
        self.geometry_enabled = bool(
            state.get('geometry_enabled', stage == 'W3' and not self.dynamic_geometry_gate),
        )
        self.geometry_eligibility_step = state.get(
            'geometry_eligibility_step',
            self.geometry_activation_step,
        )
        self.geometry_last_disable_step = state.get('geometry_last_disable_step')
        self.consecutive_passes = int(state.get('consecutive_passes', 0))
        self.consecutive_failures = int(state.get('consecutive_failures', 0))
        self.gate_evaluations = int(state.get('gate_evaluations', 0))
        restored = dict(state['ema'])
        self.ema = {key: restored.get(key) for key in self.ema}
        latest = dict(state.get('latest', {}))
        self.latest = {key: latest.get(key) for key in self.latest}

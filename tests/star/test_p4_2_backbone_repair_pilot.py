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
"""Tests for the frozen P4.2 B0-R1 feasibility hard stops."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.star.summarize_p4_2_backbone_repair_pilot import (
    hard_stop_decision,
    pilot_external_rows,
)


def write_evaluation(path: Path, *, success: float, stall: float) -> None:
    """Write the exact two-row pilot evaluation shape."""
    fields = [
        'training_seed',
        'milestone',
        'evaluation_mode',
        'evaluation_role',
        'episodes',
        'mean_return',
        'goal_met_episode_rate',
        'stall_ratio',
        'mean_step_displacement',
        'mean_start_end_displacement',
        'mean_action_norm',
    ]
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in (0, 50_000):
            writer.writerow(
                {
                    'training_seed': 2028,
                    'milestone': step,
                    'evaluation_mode': 'fixed-random-flow',
                    'evaluation_role': 'primary',
                    'episodes': 20,
                    'mean_return': 1.0,
                    'goal_met_episode_rate': success if step else 0.1,
                    'stall_ratio': stall if step else 0.1,
                    'mean_step_displacement': 0.01,
                    'mean_start_end_displacement': 1.0,
                    'mean_action_norm': 0.5,
                },
            )


def test_pilot_external_rows_require_the_frozen_endpoints(tmp_path: Path) -> None:
    path = tmp_path / 'evaluation.csv'
    write_evaluation(path, success=0.2, stall=0.1)
    rows = pilot_external_rows(path)
    assert set(rows) == {0, 50_000}
    assert rows[50_000]['goal_met_episode_rate'] == 0.2
    assert rows[50_000]['stall_ratio'] == 0.1


def test_any_pilot_hard_stop_keeps_safety_paused() -> None:
    clear = {
        'non_finite_artifact': False,
        'zero_endpoint_success': False,
        'endpoint_stall_at_least_0_5': False,
        'reward_td_monotonic_divergence': False,
    }
    assert hard_stop_decision(clear) == 'launch_remaining_repair_seeds'
    for name in clear:
        triggered = {**clear, name: True}
        assert hard_stop_decision(triggered) == 'continue_backbone_repair_and_pause_safety'

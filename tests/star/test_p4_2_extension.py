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
"""Tests for the P4.2 top-two 100k-by-three extension summary."""

from __future__ import annotations

from scripts.star.summarize_p4_2_extension import (
    SEEDS,
    expected_diagnostic_steps,
    extension_decision,
    reward_td_trend,
)


def diagnostics(values: list[float], *, start: int) -> list[dict[str, str]]:
    """Build a compact reward-TD diagnostic series."""
    return [
        {'TotalEnvSteps': str(step), 'Critic/RewardTD': str(value)}
        for step, value in zip(range(start, 100_001, 5_000), values)
    ]


def test_reward_td_trend_handles_fresh_and_resumed_history() -> None:
    fresh_values = [0.1 + 0.001 * (index % 2) for index in range(20)]
    fresh = reward_td_trend(diagnostics(fresh_values, start=5_000), 'B4', 2027)
    assert fresh['monotonic_divergence'] is False
    assert fresh['includes_screen_history'] is False
    resumed_values = [0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.04, 0.05, 0.06, 0.07]
    resumed = reward_td_trend(
        diagnostics(resumed_values, start=55_000),
        'B4',
        2028,
        screen_historical_minimum=0.02,
    )
    assert expected_diagnostic_steps(2028)[0] == 55_000
    assert resumed['historical_minimum'] == 0.02
    assert resumed['monotonic_divergence'] is True
    assert resumed['includes_screen_history'] is True


def make_arm(
    *,
    success: float,
    stall: float,
    cost: float,
    cvar: float,
    brier: float = 0.2,
    auroc: float = 0.8,
    boundary: float = 200.0,
    dangerous: float = 0.02,
) -> dict:
    """Build synthetic three-seed endpoint evidence."""
    runs = {}
    for seed in SEEDS:
        runs[str(seed)] = {
            'reward_td_trend': {'monotonic_divergence': False},
            'heldout_probability_final': {
                'Heldout/BrierSkill': brier,
                'Heldout/AUROC': auroc,
                'Heldout/BoundarySamples': boundary,
            },
            'candidate_probing_final': {'Probing/DangerousNormalVelocity': dangerous},
        }
    return {
        'runs': runs,
        'pooled_external_100k': {
            'goal_met_episode_rate': success,
            'stall_ratio': stall,
            'mean_cost': cost,
            'cost_cvar_20': cvar,
        },
    }


def test_extension_decision_requires_every_b4_seed_to_pass() -> None:
    arms = {
        'B4': make_arm(success=0.7, stall=0.05, cost=10.0, cvar=20.0),
        'B1': make_arm(success=0.65, stall=0.1, cost=20.0, cvar=40.0),
    }
    screen = {
        'arms': {
            'B4': {'external_evaluation': {'50000': {'goal_met_episode_rate': 0.7}}},
        },
    }
    p4_1 = {'stall_ratio': 0.16, 'dangerous_normal_velocity': 0.1}
    passed = extension_decision(arms, screen, p4_1)
    assert passed['p5_entry_guard_passed'] is True
    arms['B4']['runs']['2029']['heldout_probability_final']['Heldout/BoundarySamples'] = 127
    failed = extension_decision(arms, screen, p4_1)
    assert failed['p5_entry_guard_passed'] is False
    assert failed['criteria']['heldout_boundary_samples_at_least_128_every_seed'] is False

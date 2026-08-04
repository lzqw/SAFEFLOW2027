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
"""Numerical contracts for compact STAR validation artifact tooling."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from scripts.star.evaluate_pointgoal_checkpoints import json_safe, public_row

from omnisafe.algorithms.off_policy.star import _spearman
from omnisafe.utils.star_validation import (
    average_ranks,
    calibration_rows,
    checkpoint_reconstruction_config,
    deterministic_indices,
    spearman,
    threshold_rows,
)


def test_checkpoint_reconstruction_prefers_saved_config_and_infers_legacy_shapes() -> None:
    saved = {
        'extra': {
            'resolved_config': {
                'algo_cfgs': {'boundary_low': 0.07},
                'model_cfgs': {'actor': {'hidden_sizes': [11, 13]}},
            },
        },
    }
    config = checkpoint_reconstruction_config(saved)
    assert config['algo_cfgs']['boundary_low'] == pytest.approx(0.07)
    assert config['model_cfgs']['actor']['hidden_sizes'] == [11, 13]

    legacy = {
        'model': {
            'actor.velocity_net.0.weight': torch.zeros(16, 7),
            'actor.velocity_net.2.weight': torch.zeros(16, 16),
            'actor.velocity_net.4.weight': torch.zeros(2, 16),
            'reward_critic.critics.0.0.weight': torch.zeros(32, 8),
            'reward_critic.critics.0.2.weight': torch.zeros(32, 32),
            'reward_critic.critics.0.4.weight': torch.zeros(1, 32),
            'cost_critic.heads.0.0.weight': torch.zeros(32, 9),
            'cost_critic.heads.0.2.weight': torch.zeros(32, 32),
            'cost_critic.heads.0.4.weight': torch.zeros(5, 32),
            'cost_critic.heads.1.0.weight': torch.zeros(32, 9),
            'cost_critic.heads.1.2.weight': torch.zeros(32, 32),
            'cost_critic.heads.1.4.weight': torch.zeros(5, 32),
        },
    }
    inferred = checkpoint_reconstruction_config(legacy)
    assert inferred['model_cfgs']['actor']['hidden_sizes'] == [16, 16]
    assert inferred['model_cfgs']['critic']['hidden_sizes'] == [32, 32]
    assert inferred['model_cfgs']['critic']['num_quantiles'] == 5
    assert inferred['model_cfgs']['critic']['ensemble_size'] == 2


def test_tie_correct_ranks_and_spearman() -> None:
    values = torch.tensor([1.0, 1.0, 3.0, 5.0])
    assert torch.equal(average_ranks(values), torch.tensor([0.5, 0.5, 2.0, 3.0]))
    assert spearman(values, values) == pytest.approx(1.0)
    assert spearman(values, -values) == pytest.approx(-1.0)
    assert _spearman(values, values) == pytest.approx(1.0)
    assert math.isnan(_spearman(values, torch.ones_like(values)))


def test_checkpoint_evaluation_serializes_degenerate_spearman_as_null() -> None:
    payload = {'extra': {'seed': 2027}, 'counters': {'env_steps': 50_000}}
    evaluation = {
        'episodes': 5,
        'mean_return': 1.0,
        'mean_cost': 0.0,
        'cost_cvar_20': 0.0,
        'violation_ratio': 0.0,
        'calibration_error': 0.1,
        'spearman': float('nan'),
        'predicted_risk_mean': 0.1,
        'empirical_risk_mean': 0.0,
    }
    row = public_row(payload, Path('milestone_50000.pt'), evaluation)
    assert row['spearman'] is None
    assert json_safe({'spearman': float('nan')}) == {'spearman': None}


def test_calibration_rows_are_finite_or_empty() -> None:
    rows = calibration_rows(
        torch.tensor([0.05, 0.15, 0.85, 0.95]),
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
        bins=4,
    )
    assert rows[0]['count'] == 2
    assert rows[0]['predicted_risk_mean'] == pytest.approx(0.1)
    assert rows[0]['empirical_risk_mean'] == 0.0
    assert rows[1]['count'] == 0
    assert rows[1]['predicted_risk_mean'] is None
    assert rows[3]['count'] == 2
    assert rows[3]['absolute_gap'] == pytest.approx(0.1)


def test_deterministic_replay_subset_and_boundary_sensitivity() -> None:
    assert torch.equal(deterministic_indices(10, 4), torch.tensor([0, 3, 6, 9]))
    audit = {
        'risk_ucb': torch.tensor([0.01, 0.15, 0.35, 0.48, 0.8]),
        'source_random': torch.tensor([False, True, False, False, False]),
        'is_failure': torch.tensor([False, False, True, False, True]),
    }
    rows = threshold_rows(audit, current_low=0.1, current_safe=0.4, boundary=0.5)
    current = next(row for row in rows if row['is_current'])
    assert current['sample_count'] == 2
    assert current['sample_fraction'] == pytest.approx(0.4)
    assert current['source_random_fraction'] == pytest.approx(0.5)
    assert current['immediate_violation_fraction'] == pytest.approx(0.5)

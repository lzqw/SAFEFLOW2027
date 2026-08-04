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

import pytest
import torch

from omnisafe.utils.star_validation import (
    average_ranks,
    calibration_rows,
    deterministic_indices,
    spearman,
    threshold_rows,
)


def test_tie_correct_ranks_and_spearman() -> None:
    values = torch.tensor([1.0, 1.0, 3.0, 5.0])
    assert torch.equal(average_ranks(values), torch.tensor([0.5, 0.5, 2.0, 3.0]))
    assert spearman(values, values) == pytest.approx(1.0)
    assert spearman(values, -values) == pytest.approx(-1.0)


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

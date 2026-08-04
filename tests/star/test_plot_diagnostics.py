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
"""Tests for corrected fixed-device STAR diagnostic plotting inputs."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.star.plot_star_diagnostics import checkpoint_evaluation_rows


def test_checkpoint_evaluations_group_and_sort_by_training_seed(tmp_path: Path) -> None:
    """Plotting consumes the same ordered tie-correct table as strict P4 summary."""
    path = tmp_path / 'checkpoint-evaluation.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=['training_seed', 'milestone', 'mean_return', 'spearman'],
        )
        writer.writeheader()
        writer.writerows(
            [
                {'training_seed': 2027, 'milestone': 300_000, 'mean_return': 1.0, 'spearman': ''},
                {'training_seed': 2028, 'milestone': 0, 'mean_return': -1.0, 'spearman': 0.2},
                {'training_seed': 2027, 'milestone': 0, 'mean_return': -2.0, 'spearman': 0.1},
            ],
        )
    grouped = checkpoint_evaluation_rows(path)
    assert sorted(grouped) == [2027, 2028]
    assert [int(row['milestone']) for row in grouped[2027]] == [0, 300_000]
    assert grouped[2027][-1]['spearman'] == ''

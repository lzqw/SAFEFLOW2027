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
"""Tests for the unattended PointGoal P4 postprocessing launcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.star.run_pointgoal_p4_postprocess import MILESTONES, discover_run


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one compact launcher fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def make_completed_run(tmp_path: Path) -> tuple[Path, Path]:
    """Create one structurally complete 300k run without checkpoint payload contents."""
    parent = tmp_path / 'parent'
    run_dir = parent / 'runs/STAR-{SafetyPointGoal1-v0}/seed-2027-test'
    artifacts = run_dir / 'artifacts'
    receipt = tmp_path / 'receipt.json'
    write_json(
        artifacts / 'training_summary.json',
        {
            'seed': 2027,
            'status': 'completed',
            'env_steps': 300_000,
            'updates': 298_000,
        },
    )
    write_json(
        artifacts / 'run_manifest.json',
        {'seed': 2027, 'eval_milestones': list(MILESTONES)},
    )
    write_json(
        receipt,
        {'seed': 2027, 'status': 'passed', 'steps': 300_000, 'updates': 298_000},
    )
    checkpoints = run_dir / 'checkpoints'
    checkpoints.mkdir(parents=True)
    for milestone in MILESTONES:
        (checkpoints / f'milestone_{milestone}.pt').write_bytes(b'fixture')
    (checkpoints / 'final_full.pt').write_bytes(b'fixture')
    return parent, receipt


def test_discover_run_validates_exact_endpoints_and_checkpoints(tmp_path: Path) -> None:
    """A complete run resolves to the expected seed and ordered milestone files."""
    parent, receipt = make_completed_run(tmp_path)
    run = discover_run(parent, receipt)
    assert run['seed'] == 2027
    assert [path.name for path in run['checkpoints']] == [
        f'milestone_{milestone}.pt' for milestone in MILESTONES
    ]


def test_discover_run_rejects_missing_full_replay_checkpoint(tmp_path: Path) -> None:
    """Postprocessing cannot start without the final replay-bearing checkpoint."""
    parent, receipt = make_completed_run(tmp_path)
    (parent / 'runs/STAR-{SafetyPointGoal1-v0}/seed-2027-test/checkpoints/final_full.pt').unlink()
    with pytest.raises(FileNotFoundError, match='final_full'):
        discover_run(parent, receipt)

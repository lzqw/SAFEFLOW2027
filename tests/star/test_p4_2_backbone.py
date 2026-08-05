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
"""Tests for the preregistered P4.2 backbone decision."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.star.summarize_p4_2_backbone import backbone_decision, variant_summary


def write_json(path: Path, payload: dict) -> None:
    """Write one small fixture object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def make_arm(tmp_path: Path, name: str, *, final_stall: float) -> tuple[list[Path], list[Path]]:
    """Create three complete synthetic run directories and receipts."""
    global_enabled = name == 'B1'
    run_dirs, receipts = [], []
    for seed in (2027, 2028, 2029):
        run_dir = tmp_path / name / f'run-{seed}'
        run_dirs.append(run_dir)
        write_json(
            run_dir / 'artifacts/training_summary.json',
            {
                'status': 'completed',
                'seed': seed,
                'env_steps': 50_000,
                'updates': 48_000,
                'training_seconds': 10.0,
                'global_safety_constraint_enabled': global_enabled,
                'local_safety_geometry_enabled': False,
                'training_geometry_enabled': False,
            },
        )
        write_json(
            run_dir / 'artifacts/run_manifest.json',
            {
                'seed': seed,
                'eval_milestones': [0, 50_000],
                'p4_2_backbone_ablation': {
                    'reward_flow_enabled': True,
                    'global_safety_constraint_enabled': global_enabled,
                    'local_safety_geometry_enabled': False,
                },
            },
        )
        with (run_dir / 'artifacts/diagnostics.csv').open(
            'w',
            encoding='utf-8',
            newline='',
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=['TotalEnvSteps', 'Critic/RewardTD'])
            writer.writeheader()
            values = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.10, 0.12, 0.11]
            for index, value in enumerate(values, start=1):
                writer.writerow({'TotalEnvSteps': index * 5000, 'Critic/RewardTD': value})
        for filename in ('milestone_0.pt', 'milestone_50000.pt', 'latest_full.pt', 'final_full.pt'):
            checkpoint = run_dir / 'checkpoints' / filename
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b'checkpoint')
        receipt = tmp_path / name / f'receipt-{seed}.json'
        receipts.append(receipt)
        write_json(
            receipt,
            {
                'status': 'passed',
                'seed': seed,
                'steps': 50_000,
                'updates': 48_000,
                'estimated_cost_cny': 0.1,
            },
        )
    evaluation = tmp_path / name / 'evaluation.csv'
    fields = [
        'training_seed',
        'milestone',
        'evaluation_mode',
        'episodes',
        'mean_return',
        'goal_met_episode_rate',
        'stall_ratio',
        'mean_start_end_displacement',
        'mean_cost',
        'goal_met_total',
        'brier_skill',
        'auroc',
    ]
    with evaluation.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for seed in (2027, 2028, 2029):
            writer.writerow(
                {
                    'training_seed': seed,
                    'milestone': 0,
                    'evaluation_mode': 'fixed-random-flow',
                    'episodes': 20,
                    'mean_return': 0.0,
                    'goal_met_episode_rate': 0.1,
                    'stall_ratio': 0.1,
                    'mean_start_end_displacement': 1.0,
                    'mean_cost': 5.0,
                    'goal_met_total': 2,
                    'brier_skill': '',
                    'auroc': '',
                },
            )
            writer.writerow(
                {
                    'training_seed': seed,
                    'milestone': 50_000,
                    'evaluation_mode': 'fixed-random-flow',
                    'episodes': 20,
                    'mean_return': 1.0,
                    'goal_met_episode_rate': 0.2,
                    'stall_ratio': final_stall,
                    'mean_start_end_displacement': 1.1,
                    'mean_cost': 6.0,
                    'goal_met_total': 4,
                    'brier_skill': -0.1,
                    'auroc': 0.7,
                },
            )
    return run_dirs, receipts


def test_backbone_decision_localizes_a_b1_functional_collapse(tmp_path: Path) -> None:
    b0_runs, b0_receipts = make_arm(tmp_path, 'B0', final_stall=0.1)
    b1_runs, b1_receipts = make_arm(tmp_path, 'B1', final_stall=0.2)
    b0 = variant_summary('B0', b0_runs, b0_receipts, tmp_path / 'B0/evaluation.csv')
    b1 = variant_summary('B1', b1_runs, b1_receipts, tmp_path / 'B1/evaluation.csv')

    assert b0['functional_gate_passed'] is True
    assert b1['criteria']['stall_does_not_rise'] is False
    assert backbone_decision(b0, b1) == 'diagnose_global_cost_dual_conservative_collapse'


def test_backbone_failure_takes_priority_over_b1() -> None:
    failed = {'functional_gate_passed': False}
    passed = {'functional_gate_passed': True}
    assert backbone_decision(failed, passed) == 'repair_backbone_and_pause_safety'
    assert backbone_decision(passed, passed) == 'proceed_to_bellman_probability_critic'

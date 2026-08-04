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
"""Tests for the compact PointGoal P4 result aggregator."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scripts.star.summarize_pointgoal_p4 import EVALUATION_METRICS, aggregate_evaluations, summarize


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one strict fixture object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=False), encoding='utf-8')


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one fixture CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_seed(tmp_path: Path, seed: int) -> tuple[Path, Path, Path]:
    """Create one complete synthetic P4 seed artifact set."""
    run_dir = tmp_path / f'run-{seed}'
    artifact_dir = run_dir / 'artifacts'
    audit_dir = tmp_path / f'audit-{seed}'
    receipt = tmp_path / f'receipt-{seed}.json'
    write_json(artifact_dir / 'run_manifest.json', {'seed': seed})
    write_json(
        artifact_dir / 'training_summary.json',
        {
            'seed': seed,
            'status': 'completed',
            'env_steps': 300_000,
            'updates': 298_000,
            'training_seconds': 100.0,
        },
    )
    write_json(
        artifact_dir / 'warmup_status.json',
        {
            'stage': 'W3',
            'geometry_activation_step': 20_000,
            'boundary_samples': 256,
            'ema': {
                'calibration_error': 0.1,
                'cost_td_error': 0.02,
                'valid_gradient_rate': 0.8,
                'ensemble_disagreement': 0.2,
                'brier_skill': 0.2,
                'auroc': 0.8,
            },
            'conditions': {
                'minimum_steps': True,
                'td_error': True,
                'calibration': True,
                'valid_gradient': True,
                'ensemble_disagreement': True,
                'boundary_samples': True,
                'positive_labels': True,
                'negative_labels': True,
                'brier_skill': True,
                'auroc': True,
            },
        },
    )
    write_json(
        artifact_dir / 'counterfactual_step_300000.json',
        {
            'available': True,
            'gate_boundary_mean': 0.6,
            'gate_interior_mean': 0.02,
            'isotropic_vn': 1.0,
            'directional_vn': 0.5,
            'isotropic_vt': 1.0,
            'directional_vt': 1.0,
            'vt_preservation_ratio': 1.0,
        },
    )
    write_csv(
        artifact_dir / 'diagnostics.csv',
        [
            {
                'Train/WarmupStage': 2,
                'Geometry/DangerousNormalVelocity': 0.2,
                'Loss/Actor': 1.0,
                'Loss/CostCritic': 0.2,
                'Loss/RewardCritic': 0.3,
                'Dual/Energy': 0.1,
                'Dual/GlobalSafety': 0.01,
            },
            {
                'Train/WarmupStage': 3,
                'Geometry/DangerousNormalVelocity': 0.1,
                'Loss/Actor': 0.5,
                'Loss/CostCritic': 0.1,
                'Loss/RewardCritic': 0.2,
                'Dual/Energy': 0.2,
                'Dual/GlobalSafety': 1.0,
            },
        ],
    )
    write_csv(
        artifact_dir / 'evaluation.csv',
        [
            {
                'milestone': 0,
                'episodes': 5,
                'mean_return': -1.0,
                'mean_cost': 20.0,
                'cost_cvar_20': 50.0,
                'violation_ratio': 0.02,
                'calibration_error': 0.3,
                'brier_score': 0.3,
                'brier_skill': -0.2,
                'auroc': 0.55,
                'spearman': 0.1,
                'goals_per_episode_mean': 0.0,
                'stall_ratio': 1.0,
                'mean_start_end_displacement': 0.0,
                'mean_action_norm': 0.0,
            },
            {
                'milestone': 300_000,
                'episodes': 5,
                'mean_return': 1.0,
                'mean_cost': 5.0,
                'cost_cvar_20': 10.0,
                'violation_ratio': 0.005,
                'calibration_error': 0.1,
                'brier_score': 0.1,
                'brier_skill': 0.2,
                'auroc': 0.8,
                'spearman': 0.4,
                'goals_per_episode_mean': 1.0,
                'stall_ratio': 0.0,
                'mean_start_end_displacement': 1.0,
                'mean_action_norm': 0.5,
            },
        ],
    )
    write_json(
        audit_dir / 'replay_risk_summary.json',
        {
            'seed': seed,
            'calibration_ece': 0.1,
            'calibration_mae': 0.1,
            'predicted_cost_empirical_cost_spearman': 0.4,
            'brier_skill': 0.2,
            'auroc': 0.8,
            'positive_labels': 128,
            'negative_labels': 128,
            'resident_boundary_source_random_fraction': 0.0,
            'resident_boundary_immediate_violation_fraction': 0.0,
        },
    )
    write_json(receipt, {'seed': seed, 'estimated_cost_cny': 0.5})
    return run_dir, audit_dir, receipt


def test_aggregate_evaluations_reports_sample_standard_deviation() -> None:
    """Aggregate rows retain milestone-level means and sample deviations."""
    rows = [
        {'milestone': 0, **dict.fromkeys(EVALUATION_METRICS, 1.0)},
        {'milestone': 0, **dict.fromkeys(EVALUATION_METRICS, 3.0)},
    ]
    rows[1]['spearman'] = None
    summary = aggregate_evaluations(rows)
    assert summary[0]['mean_return_mean'] == 2.0
    assert summary[0]['mean_return_std'] == 2**0.5
    assert summary[0]['spearman_count'] == 1
    assert summary[0]['spearman_mean'] == 1.0


def test_summarize_requires_every_seed_to_pass(tmp_path: Path) -> None:
    """The verdict is strict across all three preregistered seeds and criteria."""
    fixtures = [make_seed(tmp_path, seed) for seed in (2027, 2028, 2029)]
    runs = [item[0] for item in fixtures]
    audits = [item[1] for item in fixtures]
    receipts = [item[2] for item in fixtures]
    verdict = summarize(runs, audits, receipts, tmp_path / 'passing')
    assert verdict['status'] == 'passed'
    assert (tmp_path / 'passing/pointgoal_evaluation_summary.csv').exists()

    diagnostics = runs[-1] / 'artifacts/diagnostics.csv'
    with diagnostics.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    rows[-1]['Geometry/DangerousNormalVelocity'] = '0.3'
    write_csv(diagnostics, rows)
    verdict = summarize(runs, audits, receipts, tmp_path / 'failing')
    assert verdict['status'] == 'failed'
    assert not verdict['seed_pass']['2029']
    assert not verdict['criteria_by_seed']['2029']['dangerous_normal_velocity_reduced']


def test_summarize_uses_tie_correct_checkpoint_evaluations(tmp_path: Path) -> None:
    """Offline fixed-device rows drive both tables and strict endpoint criteria."""
    fixtures = [make_seed(tmp_path, seed) for seed in (2027, 2028, 2029)]
    runs = [item[0] for item in fixtures]
    audits = [item[1] for item in fixtures]
    receipts = [item[2] for item in fixtures]
    for run_dir in runs:
        online = run_dir / 'artifacts/evaluation.csv'
        with online.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        rows[-1]['mean_return'] = '-2.0'
        rows[-1]['mean_cost'] = '30.0'
        rows[-1]['cost_cvar_20'] = '60.0'
        write_csv(online, rows)

    checkpoint_rows = []
    for seed in (2027, 2028, 2029):
        checkpoint_rows.extend(
            [
                {
                    'training_seed': seed,
                    'milestone': 0,
                    'episodes': 5,
                    **dict.fromkeys(EVALUATION_METRICS, 1.0),
                },
                {
                    'training_seed': seed,
                    'milestone': 300_000,
                    'episodes': 5,
                    **dict.fromkeys(EVALUATION_METRICS, 0.5),
                    'mean_return': 2.0,
                    'spearman': '',
                    'goals_per_episode_mean': 1.0,
                    'stall_ratio': 0.0,
                },
            ],
        )
    checkpoint_csv = tmp_path / 'checkpoint-evaluations.csv'
    write_csv(checkpoint_csv, checkpoint_rows)
    verdict = summarize(
        runs,
        audits,
        receipts,
        tmp_path / 'offline',
        checkpoint_evaluation_csv=checkpoint_csv,
    )
    assert verdict['status'] == 'passed'
    assert verdict['criteria_by_seed']['2027']['mean_goal_met_positive']
    with (tmp_path / 'offline/pointgoal_evaluation_summary.csv').open(
        encoding='utf-8',
        newline='',
    ) as stream:
        summary = list(csv.DictReader(stream))
    assert summary[-1]['spearman_count'] == '0'
    assert summary[-1]['spearman_mean'] == ''


def test_undefined_probability_auroc_fails_without_crashing(tmp_path: Path) -> None:
    """A degenerate probability statistic is evidence failure, not a serializer crash."""
    fixtures = [make_seed(tmp_path, seed) for seed in (2027, 2028, 2029)]
    audit = fixtures[-1][1] / 'replay_risk_summary.json'
    payload = json.loads(audit.read_text(encoding='utf-8'))
    payload['auroc'] = None
    write_json(audit, payload)
    verdict = summarize(
        [item[0] for item in fixtures],
        [item[1] for item in fixtures],
        [item[2] for item in fixtures],
        tmp_path / 'undefined',
    )
    assert verdict['status'] == 'failed'
    assert not verdict['criteria_by_seed']['2029']['probability_auroc_above_065']


def test_w2_seed_without_activation_step_fails_without_crashing(tmp_path: Path) -> None:
    """A seed that remains in W2 yields a strict failure and a blank activation step."""
    fixtures = [make_seed(tmp_path, seed) for seed in (2027, 2028, 2029)]
    warmup = fixtures[1][0] / 'artifacts/warmup_status.json'
    payload = json.loads(warmup.read_text(encoding='utf-8'))
    payload['stage'] = 'W2'
    payload['geometry_activation_step'] = None
    payload['conditions']['boundary_samples'] = False
    write_json(warmup, payload)
    output_dir = tmp_path / 'w2'
    verdict = summarize(
        [item[0] for item in fixtures],
        [item[1] for item in fixtures],
        [item[2] for item in fixtures],
        output_dir,
    )
    assert verdict['status'] == 'failed'
    assert not verdict['criteria_by_seed']['2028']['final_warmup_evidence_passed']
    with (output_dir / 'pointgoal_mechanism_summary.csv').open(
        encoding='utf-8',
        newline='',
    ) as stream:
        rows = {int(row['seed']): row for row in csv.DictReader(stream)}
    assert rows[2028]['geometry_activation_step'] == ''

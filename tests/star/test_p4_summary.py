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
            },
            'conditions': {
                'minimum_steps': True,
                'td_error': True,
                'calibration': True,
                'valid_gradient': True,
                'ensemble_disagreement': True,
                'boundary_samples': True,
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
                'spearman': 0.1,
            },
            {
                'milestone': 300_000,
                'episodes': 5,
                'mean_return': 1.0,
                'mean_cost': 5.0,
                'cost_cvar_20': 10.0,
                'violation_ratio': 0.005,
                'calibration_error': 0.1,
                'spearman': 0.4,
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
            'resident_boundary_source_random_fraction': 0.0,
            'resident_boundary_immediate_violation_fraction': 0.0,
        },
    )
    write_json(receipt, {'seed': seed, 'estimated_cost_cny': 0.5})
    return run_dir, audit_dir, receipt


def test_aggregate_evaluations_reports_sample_standard_deviation() -> None:
    """Aggregate rows retain milestone-level means and sample deviations."""
    rows = [
        {'milestone': 0, **{metric: 1.0 for metric in EVALUATION_METRICS}},
        {'milestone': 0, **{metric: 3.0 for metric in EVALUATION_METRICS}},
    ]
    summary = aggregate_evaluations(rows)
    assert summary[0]['mean_return_mean'] == 2.0
    assert summary[0]['mean_return_std'] == 2**0.5


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

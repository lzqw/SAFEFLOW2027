"""Tests for the final four-method PointGoal evidence merger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.star.finalize_pointgoal_four_method_comparison import (
    BASE_SEEDS,
    EPISODES_PER_BASE_SEED,
    TRAINING_SEEDS,
    aggregate_method,
    load_prior_result,
    markdown_report,
    raw_summary,
)


def episode_records(method: str, training_seed: int) -> list[dict[str, Any]]:
    """Build one exact 5x20 schedule with simple deterministic metrics."""
    return [
        {
            'method': method,
            'algorithm': method,
            'training_seed': training_seed,
            'base_seed': base_seed,
            'episode_index': episode_index,
            'return': float(training_seed - 2027 + episode_index / 100),
            'cost': float(episode_index),
            'success': int(episode_index % 2 == 0),
            'length': 10,
            'goal_met_count': int(episode_index % 2 == 0),
            'violation_steps': episode_index,
            'mean_action_norm': 0.5,
        }
        for base_seed in BASE_SEEDS
        for episode_index in range(EPISODES_PER_BASE_SEED)
    ]


def test_prior_result_is_recomputed_from_exact_raw_schedule(tmp_path: Path) -> None:
    records = episode_records('SACLag', 2027)
    summary = {**raw_summary(records), 'episodes': 100}
    path = tmp_path / 'saclag.json'
    path.write_text(
        json.dumps(
            {
                'status': 'complete',
                'algorithm': 'SACLag',
                'environment': 'SafetyPointGoal1-v0',
                'training_seed': 2027,
                'training_steps': 200000,
                'training_cumulative_cost': 123.0,
                'evaluation_role': 'primary',
                'evaluation_mode': 'fixed-native-stochastic-policy-stream',
                'base_seeds': list(BASE_SEEDS),
                'episodes_per_base_seed': 20,
                'summary': summary,
                'episode_records': records,
            },
        ),
        encoding='utf-8',
    )

    row, normalized = load_prior_result(path)

    assert len(normalized) == 100
    assert row['mean_cost'] == pytest.approx(9.5)
    assert row['cost_cvar_20'] == pytest.approx(17.5)
    assert row['training_cumulative_cost'] == 123.0


def test_aggregate_uses_three_seed_variation_and_pooled_cvar() -> None:
    all_records = [
        record
        for training_seed in TRAINING_SEEDS
        for record in episode_records('SACLag', training_seed)
    ]
    seed_rows = []
    for training_seed in TRAINING_SEEDS:
        records = [row for row in all_records if row['training_seed'] == training_seed]
        seed_rows.append(
            {
                'method': 'SACLag',
                'training_seed': training_seed,
                **raw_summary(records),
                'training_cumulative_cost': float(training_seed),
                'high_risk_vn_vt': None,
                'high_risk_dangerous_outward_velocity': None,
            },
        )

    aggregate = aggregate_method('SACLag', seed_rows, all_records)

    assert aggregate['training_seeds'] == 3
    assert aggregate['pooled_evaluation_episodes'] == 300
    assert aggregate['seed_std_mean_return'] == pytest.approx(1.0)
    assert aggregate['cost_cvar_20'] == pytest.approx(17.5)
    assert aggregate['high_risk_vn_vt'] is None


def test_report_keeps_aggregate_claim_boundary() -> None:
    rows = []
    for index, method in enumerate(
        ('C0 / isotropic FLAC', 'SACLag', 'SHAPO-PIDLag', 'P4.7 / Ours'),
    ):
        rows.append(
            {
                'method': method,
                'mean_return': 1.0 + index,
                'success_rate': 0.2 + 0.1 * index,
                'mean_cost': 10.0 - index,
                'cost_cvar_20': 20.0 - index,
                'training_cumulative_cost_mean': 100.0 - index,
                'high_risk_vn_vt': 0.9 if index in (0, 3) else None,
            },
        )

    report = markdown_report(rows)

    assert 'independent statistical unit' in report
    assert 'not an every-seed Pareto claim' in report
    assert 'native stochastic action distributions' in report

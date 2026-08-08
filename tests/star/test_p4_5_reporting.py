# Copyright 2026 OmniSafe Team. All Rights Reserved.
"""Tests for the locked P4.5 three-class summary and seven figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.star.plot_p4_5_unified_cost_geometry import (
    plot_learning_curves,
    plot_trajectories,
)
from scripts.star.summarize_p4_5_unified_cost_geometry import _classification


def _fake_runs(
    *,
    c1_return: float = 0.95,
    c1_success: float = 0.85,
    improve_seeds: tuple[int, ...] = (2027, 2028, 2029),
) -> dict:
    runs = {}
    for arm in ('C0', 'C1'):
        for seed in (2027, 2028, 2029):
            improved = arm == 'C1' and seed in improve_seeds
            mean_cost = 8.0 if improved else 10.0 if arm == 'C0' else 11.0
            cvar = 12.0 if improved else 15.0 if arm == 'C0' else 16.0
            danger = 0.08 if improved else 0.10 if arm == 'C0' else 0.11
            mean_return = 1.0 if arm == 'C0' else c1_return
            success = 0.90 if arm == 'C0' else c1_success
            evaluation = {
                'returns': [mean_return] * 20,
                'costs': [mean_cost] * 16 + [cvar] * 4,
                'goal_counts': [1] * round(success * 20) + [0] * (20 - round(success * 20)),
                'direction_geometry': {'dangerous_normal_velocity': danger},
                'mean_path_length': 5.0,
                'stall_ratio': 0.1,
            }
            runs[(arm, seed)] = {
                'checkpoint_rows': [
                    {
                        'mean_cost': mean_cost,
                        'cost_cvar_20': cvar,
                        'dangerous_normal_velocity': danger,
                    },
                ],
                'evaluations': {100_000: evaluation},
            }
    return runs


def test_three_classification_labels_follow_the_preregistered_rules() -> None:
    effective = _classification(_fake_runs())
    assert effective['label'] == 'effective'
    assert effective['safety_majority'] is True
    assert effective['performance_maintained'] is True

    performance_cost = _classification(_fake_runs(c1_return=0.70, c1_success=0.60))
    assert performance_cost['label'] == 'safety_benefit_with_performance_cost'
    assert performance_cost['safety_majority'] is True
    assert performance_cost['performance_maintained'] is False

    unstable = _classification(_fake_runs(improve_seeds=(2027,)))
    assert unstable['label'] == 'no_stable_effect'
    assert unstable['safety_majority'] is False


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_all_preregistered_plots_render_from_compact_evidence(tmp_path: Path) -> None:
    summary = tmp_path / 'summary'
    plots = tmp_path / 'plots'
    checkpoint_rows = []
    diagnostic_rows = []
    trajectory_manifest = []
    for arm_index, arm in enumerate(('C0', 'C1')):
        for seed in (2027, 2028, 2029):
            for milestone in (0, 25_000, 50_000, 75_000, 100_000):
                checkpoint_rows.append(
                    {
                        'arm': arm,
                        'training_seed': seed,
                        'milestone': milestone,
                        'mean_return': milestone / 100_000 + arm_index * 0.01,
                        'success_rate': milestone / 100_000,
                        'mean_cost': 10.0 - milestone / 20_000,
                        'cost_cvar_20': 15.0 - milestone / 20_000,
                        'training_cumulative_cost': milestone / 1000,
                        'dangerous_normal_velocity': 0.1 - milestone / 2_000_000,
                        'normal_to_tangent_variance_ratio': 0.5,
                    },
                )
            for step in range(5_000, 100_001, 5_000):
                diagnostic_rows.append(
                    {
                        'arm': arm,
                        'training_seed': seed,
                        'TotalEnvSteps': step,
                        'Critic/RewardTD': 0.01,
                        'Critic/CostTD': 0.02,
                        'Critic/RewardQAbsMax': 2.0,
                        'Critic/RewardTargetQAbsMax': 2.1,
                        'Critic/CostQAbsMax': 3.0,
                        'Critic/CostTargetQAbsMax': 3.1,
                    },
                )
            trajectory_path = tmp_path / f'{arm}-{seed}-trajectory.csv'
            trajectory_rows = []
            for evaluation_seed in (12_027, 12_036, 12_046):
                for step in range(6):
                    trajectory_rows.append(
                        {
                            'seed': evaluation_seed,
                            'step': step,
                            'x': step * 0.1,
                            'y': arm_index * 0.05,
                            'goal_x': 1.0,
                            'goal_y': 1.0,
                            'hazards_json': '[[0.5,0.5]]' if step == 0 else '',
                            'hazard_radius': 0.2,
                            'cost': int(step == 3),
                            'action_0': 0.1,
                            'action_1': 0.0,
                            'cost_normal_0': 1.0,
                            'cost_normal_1': 0.0,
                            'cost_tangent_0': 0.0,
                            'cost_tangent_1': 1.0,
                        },
                    )
            _write_csv(trajectory_path, trajectory_rows)
            trajectory_manifest.append(
                {
                    'arm': arm,
                    'training_seed': seed,
                    'milestone': 100_000,
                    'path': trajectory_path,
                },
            )
    _write_csv(summary / 'p4_5_checkpoint_metrics.csv', checkpoint_rows)
    _write_csv(summary / 'p4_5_diagnostics_combined.csv', diagnostic_rows)
    _write_csv(summary / 'p4_5_trajectory_manifest.csv', trajectory_manifest)
    outputs = [
        *plot_learning_curves(summary, plots),
        *plot_trajectories(summary, plots),
    ]
    assert len(outputs) == 7
    assert all(path.stat().st_size > 0 for path in outputs)
    assert json.loads(json.dumps([str(path) for path in outputs]))

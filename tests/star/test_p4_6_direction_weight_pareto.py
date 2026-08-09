# Copyright 2026 OmniSafe Team. All Rights Reserved.
"""Tests for the preregistered P4.6 direction-weight Pareto sweep."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from scripts.star.plot_p4_6_direction_weight_pareto import (
    plot_final_pareto,
    plot_learning_curves,
    plot_trajectories,
)
from scripts.star.summarize_p4_6_direction_weight_pareto import (
    ARM_WEIGHTS,
    _load_baseline,
    classify_tradeoff,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'reports/star/p4_6_direction_weight_pareto_preregistration.json'
PROTOCOL_SHA256 = 'c473da2ed99b31cde673a1cd038ec0477053b23ac63a130376ebccc4271ad93e'


def test_protocol_was_sealed_before_training_and_binds_the_full_sweep() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert payload['status'] == 'locked_before_any_p4_6_training'
    assert payload['analysis']['final_comparison'] == [0.0, 0.25, 0.5, 1.0]
    assert payload['training']['new_run_count'] == 6
    assert payload['training']['total_environment_steps_per_run'] == 100_000
    assert payload['training']['early_scientific_stop'] is False
    assert payload['excluded'] == [
        'critic_changes',
        'safety_band_changes',
        'covariance_changes',
        'execution_changes',
        'probability_critic_policy_control',
        'p4_4_execution_rectification',
        'new_weight_values_beyond_0.25_and_0.50',
        'rerunning_C0_or_C1',
        '30k_pilot_or_early_scientific_gate',
        'training_beyond_100000_steps',
        'new_environment',
        'four_environment_experiment',
        'method_paper_or_reference_paper_edits',
    ]


def _config(name: str) -> dict:
    return yaml.safe_load((ROOT / 'configs/star' / name).read_text(encoding='utf-8'))


def test_candidate_configs_change_only_protocol_metadata_and_weight() -> None:
    p45 = _config('p4_5_c1_directional_loss.yaml')
    c025 = _config('p4_6_c025_directional_loss.yaml')
    c050 = _config('p4_6_c050_directional_loss.yaml')
    assert c025['custom_cfgs']['algo_cfgs']['directional_loss_weight'] == 0.25
    assert c050['custom_cfgs']['algo_cfgs']['directional_loss_weight'] == 0.50
    for payload in (p45, c025, c050):
        algo = payload['custom_cfgs']['algo_cfgs']
        assert algo['directional_loss_enabled'] is True
        algo.pop('directional_loss_weight')
        algo.pop('safety_protocol_version')
        algo.pop('safety_protocol_sha256')
    assert c025 == c050 == p45


def test_finalizer_exports_source_root_before_running_audits() -> None:
    script = (
        ROOT / 'scripts/star/finalize_p4_6_direction_weight_pareto.sh'
    ).read_text(encoding='utf-8')
    export = 'export PYTHONPATH="$SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"'
    analyzer = '"$PYTHON_BIN" "$SOURCE_DIR/scripts/star/analyze_p4_6_cost_geometry.py"'
    assert export in script
    assert analyzer in script
    assert script.index(export) < script.index(analyzer)


def _flags(weight: float) -> dict:
    return {
        'unified_cost_geometry_enabled': True,
        'stochastic_flow_enabled': True,
        'safety_anisotropy_enabled': True,
        'safety_inward_drift_enabled': False,
        'safety_outward_rectification_enabled': False,
        'candidate_probing_enabled': False,
        'dynamic_geometry_gate': False,
        'directional_loss_enabled': True,
        'directional_loss_weight': weight,
    }


def test_audit_identity_accepts_only_the_two_locked_weights() -> None:
    from scripts.star.analyze_p4_6_cost_geometry import arm_from_flags

    assert arm_from_flags(_flags(0.25)) == 'C025'
    assert arm_from_flags(_flags(0.50)) == 'C050'
    with pytest.raises(ValueError, match='unregistered'):
        arm_from_flags(_flags(0.75))
    bad = _flags(0.25)
    bad['safety_anisotropy_enabled'] = False
    with pytest.raises(ValueError, match='method flags'):
        arm_from_flags(bad)


def _decision_evidence(
    *,
    c025_return: float = 0.66,
    c025_cost: float = 8.0,
    c025_cvar: float = 12.0,
    c025_path_2028: float = 4.0,
    c025_stall_2029: float = 0.10,
    c050_return: float = 0.64,
    c050_cost: float = 7.5,
    c050_cvar: float = 11.5,
) -> tuple[dict, dict]:
    rows = {}
    for arm in ARM_WEIGHTS:
        for seed in (2027, 2028, 2029):
            if arm == 'C0':
                row = {
                    'mean_cost': 10.0,
                    'cost_cvar_20': 15.0,
                    'mean_path_length': 3.0,
                    'stall_ratio': 0.05,
                }
            elif arm == 'C025':
                row = {
                    'mean_cost': c025_cost,
                    'cost_cvar_20': c025_cvar,
                    'mean_path_length': c025_path_2028 if seed == 2028 else 3.0,
                    'stall_ratio': c025_stall_2029 if seed == 2029 else 0.05,
                }
            elif arm == 'C050':
                row = {
                    'mean_cost': c050_cost,
                    'cost_cvar_20': c050_cvar,
                    'mean_path_length': 3.5 if seed == 2028 else 3.0,
                    'stall_ratio': 0.10 if seed == 2029 else 0.05,
                }
            else:
                row = {
                    'mean_cost': 6.0,
                    'cost_cvar_20': 9.0,
                    'mean_path_length': 5.0,
                    'stall_ratio': 0.20,
                }
            rows[(arm, seed)] = row
    pooled = {
        'C0': {'mean_return': 0.70, 'success_rate': 0.30, 'mean_cost': 10.0, 'cost_cvar_20': 15.0},
        'C025': {
            'mean_return': c025_return,
            'success_rate': 0.25,
            'mean_cost': c025_cost,
            'cost_cvar_20': c025_cvar,
        },
        'C050': {
            'mean_return': c050_return,
            'success_rate': 0.24,
            'mean_cost': c050_cost,
            'cost_cvar_20': c050_cvar,
        },
        'C1': {'mean_return': 0.05, 'success_rate': 0.20, 'mean_cost': 6.0, 'cost_cvar_20': 9.0},
    }
    return rows, pooled


def test_locked_selection_gates_and_tie_breaks() -> None:
    rows, pooled = _decision_evidence()
    decision = classify_tradeoff(rows, pooled)
    assert decision['label'] == 'select_C025'
    assert decision['tie_break'] == 'lower_intervention_c025'

    rows, pooled = _decision_evidence(c050_cost=6.5, c050_cvar=10.0)
    decision = classify_tradeoff(rows, pooled)
    assert decision['label'] == 'select_C050'
    assert decision['tie_break'] == 'c050_extra_safety'

    rows, pooled = _decision_evidence(c025_return=0.40, c050_return=0.40)
    assert classify_tradeoff(rows, pooled)['label'] == 'partial_tradeoff_only'

    rows, pooled = _decision_evidence(
        c025_cost=10.0,
        c025_cvar=15.0,
        c050_cost=10.0,
        c050_cvar=15.0,
    )
    assert classify_tradeoff(rows, pooled)['label'] == 'no_stable_tradeoff'


def test_sealed_p4_5_baseline_is_reusable_without_checkpoints() -> None:
    root = ROOT / 'reports/star/p4_5_unified_cost_geometry_artifacts/final'
    baseline = _load_baseline(root)
    assert len(baseline['checkpoint_rows']) == 30
    assert len(baseline['final_rows']) == 6
    assert set(baseline['evaluations']) == {
        (arm, seed) for arm in ('C0', 'C1') for seed in (2027, 2028, 2029)
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_all_eight_pareto_plots_render(tmp_path: Path) -> None:
    summary = tmp_path / 'summary'
    plots = tmp_path / 'plots'
    checkpoints = []
    diagnostics = []
    final_rows = []
    manifest = []
    pooled = {}
    for arm_index, arm in enumerate(ARM_WEIGHTS):
        for seed in (2027, 2028, 2029):
            for milestone in (0, 25_000, 50_000, 75_000, 100_000):
                row = {
                    'arm': arm,
                    'training_seed': seed,
                    'milestone': milestone,
                    'mean_return': milestone / 100_000 - arm_index * 0.02,
                    'success_rate': milestone / 100_000,
                    'mean_cost': 12.0 - arm_index - milestone / 50_000,
                    'cost_cvar_20': 18.0 - arm_index - milestone / 50_000,
                    'training_cumulative_cost': milestone / 1000,
                    'dangerous_normal_velocity': 0.2 - arm_index * 0.01,
                    'normal_to_tangent_variance_ratio': 0.9,
                    'mean_path_length': 3.0,
                    'stall_ratio': 0.05,
                    'action_clipping_rate': 0.02,
                }
                checkpoints.append(row)
                if milestone == 100_000:
                    final_rows.append(row)
            for step in range(5_000, 100_001, 5_000):
                diagnostics.append(
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
            trajectory = tmp_path / f'{arm}-{seed}.csv'
            trajectory_rows = []
            for evaluation_seed in (12_027, 12_036, 12_046):
                for step in range(6):
                    trajectory_rows.append(
                        {
                            'seed': evaluation_seed,
                            'step': step,
                            'x': step * 0.1,
                            'y': arm_index * 0.03,
                            'goal_x': 1.0,
                            'goal_y': 1.0,
                            'hazards_json': '[[0.5,0.5]]',
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
            _write_csv(trajectory, trajectory_rows)
            manifest.append({'arm': arm, 'training_seed': seed, 'path': trajectory})
        pooled[arm] = {
            'mean_return': 1.0 - arm_index * 0.02,
            'success_rate': 0.9,
            'mean_cost': 10.0 - arm_index,
            'cost_cvar_20': 15.0 - arm_index,
        }
    _write_csv(summary / 'p4_6_checkpoint_metrics.csv', checkpoints)
    _write_csv(summary / 'p4_6_diagnostics_combined.csv', diagnostics)
    _write_csv(summary / 'p4_6_final_seed_metrics.csv', final_rows)
    _write_csv(summary / 'p4_6_trajectory_manifest.csv', manifest)
    (summary / 'p4_6_final_summary.json').write_text(
        json.dumps({'decision': {'pooled': pooled}}),
        encoding='utf-8',
    )
    outputs = [
        *plot_learning_curves(summary, plots),
        plot_final_pareto(summary, plots),
        *plot_trajectories(summary, plots),
    ]
    assert len(outputs) == 8
    assert all(path.stat().st_size > 0 for path in outputs)

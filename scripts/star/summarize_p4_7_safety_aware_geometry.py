#!/usr/bin/env python3
"""Validate P4.7 evidence, report it scientifically, and run the locked machine audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

import yaml


ARMS = ('C0', 'C1', 'P47')
SEEDS = (2027, 2028, 2029)
FINAL_STEPS = {'C0': 200_000, 'C1': 100_000, 'P47': 200_000}
INTERMEDIATE_STEPS = {
    'C0': (50_000, 100_000, 150_000, 200_000),
    'C1': (50_000, 100_000),
    'P47': (50_000, 100_000, 150_000, 200_000),
}
PROTOCOL_SHA256 = '3310d6f229d579a877cfaa98d3df2a44b25eac9998184623d1723b82d37a03ed'
FORMAL_BASE_SEEDS = (22_027, 22_028, 22_029, 22_030, 22_031)


def json_safe(value: Any) -> Any:
    """Replace non-finite floats with JSON null recursively."""
    if isinstance(value, dict):
        return {key: json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [json_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_text(path: Path, text: str) -> None:
    """Write text without exposing a partial result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict, sorted JSON atomically."""
    atomic_text(
        path,
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + '\n',
    )


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a consistently headed CSV atomically."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(rows))
    os.replace(temporary, path)


def sha256_path(path: Path) -> str:
    """Return a streaming SHA-256 for one artifact."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tail_mean(values: list[float], fraction: float = 0.2) -> float:
    """Return empirical upper-tail CVaR."""
    if not values:
        return 0.0
    count = max(1, math.ceil(len(values) * fraction))
    return statistics.fmean(sorted(values, reverse=True)[:count])


def _single(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f'expected one {pattern!r} below {root}, found {len(matches)}')
    return matches[0]


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return payload


def _evaluation_path(
    run_root: Path,
    tier: str,
    arm: str,
    seed: int,
    step: int,
) -> Path:
    return run_root / 'evaluation' / tier / arm / f'evaluation_seed_{seed}_step_{step}.json'


def load_evaluation(
    run_root: Path,
    tier: str,
    arm: str,
    seed: int,
    step: int,
) -> dict[str, Any]:
    """Load one evaluator artifact and enforce the locked stream shape."""
    path = _evaluation_path(run_root, tier, arm, seed, step)
    payload = _json(path)
    expected_per_seed = 20 if tier == 'formal' else 4
    expected_episodes = len(FORMAL_BASE_SEEDS) * expected_per_seed
    expected = {
        'training_seed': seed,
        'milestone': step,
        'episodes': expected_episodes,
        'episodes_per_base_seed': expected_per_seed,
        'evaluation_mode': 'fixed-random-flow',
        'evaluation_role': 'primary',
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if tuple(payload.get('base_seeds', ())) != FORMAL_BASE_SEEDS:
        mismatches['base_seeds'] = (payload.get('base_seeds'), list(FORMAL_BASE_SEEDS))
    if mismatches:
        raise ValueError(f'{path} violates the P4.7 evaluation seal: {mismatches}')
    if payload.get('center_flow_diagnostic') is not None:
        raise ValueError(f'{path} unexpectedly contains a center-flow selection path')
    return payload


def _receipt(run_root: Path, p45_root: Path, arm: str, seed: int) -> dict[str, Any]:
    root = p45_root / f'{arm}-{seed}' if arm == 'C1' else run_root / 'runs' / f'{arm}-{seed}'
    payload = _json(root / 'receipt.json')
    if payload.get('status') != 'passed' or payload.get('steps') != FINAL_STEPS[arm]:
        raise ValueError(f'{arm}-{seed} has no valid completed training receipt')
    if payload.get('source_dirty') is not False:
        raise ValueError(f'{arm}-{seed} was not trained from a clean source tree')
    return payload


def _diagnostics_path(run_root: Path, p45_root: Path, arm: str, seed: int, step: int) -> Path:
    if arm == 'C1' or (arm == 'C0' and step <= 100_000):
        root = p45_root / f'{arm}-{seed}'
    else:
        root = run_root / 'runs' / f'{arm}-{seed}'
    return _single(root, 'training/**/artifacts/diagnostics.csv')


def cumulative_cost_at(
    run_root: Path,
    p45_root: Path,
    arm: str,
    seed: int,
    step: int,
) -> float:
    """Read cumulative online cost at one exact logging milestone."""
    path = _diagnostics_path(run_root, p45_root, arm, seed, step)
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if int(float(row.get('TotalEnvSteps', -1))) == step]
    if len(matches) != 1:
        raise ValueError(f'{path} has {len(matches)} rows for step {step}')
    return float(matches[0]['Train/CumulativeCost'])


def evaluation_row(
    arm: str,
    seed: int,
    step: int,
    evaluation: dict[str, Any],
    cumulative_training_cost: float,
) -> dict[str, Any]:
    """Flatten the primary metrics and top-risk geometry for CSV reporting."""
    geometry = evaluation.get('direction_geometry_high_risk_top_20_percent') or {}
    return {
        'arm': arm,
        'training_seed': seed,
        'milestone': step,
        'episodes': int(evaluation['episodes']),
        'mean_return': float(evaluation['mean_return']),
        'success_rate': float(evaluation['goal_met_episode_rate']),
        'mean_cost': float(evaluation['mean_cost']),
        'cost_cvar_20': float(evaluation['cost_cvar_20']),
        'training_cumulative_cost': cumulative_training_cost,
        'violation_ratio': float(evaluation['violation_ratio']),
        'mean_path_length': float(evaluation['mean_path_length']),
        'stall_ratio': float(evaluation['stall_ratio']),
        'mean_action_norm': float(evaluation['mean_action_norm']),
        'high_risk_population_samples': int(geometry.get('population_samples', 0)),
        'high_risk_selected_samples': int(geometry.get('selected_samples', 0)),
        'high_risk_minimum_margin': float(
            geometry.get('minimum_selected_margin', float('nan')),
        ),
        'high_risk_normal_to_tangent_variance_ratio': float(
            geometry.get('normal_to_tangent_variance_ratio', float('nan')),
        ),
        'high_risk_dangerous_outward_velocity': float(
            geometry.get('dangerous_outward_velocity', float('nan')),
        ),
        'high_risk_outward_velocity_ratio': float(
            geometry.get('outward_velocity_ratio', float('nan')),
        ),
        'high_risk_reliability': float(geometry.get('reliability', float('nan'))),
        'high_risk_safety_weight': float(geometry.get('safety_weight', float('nan'))),
    }


def pooled_formal(
    formal: dict[str, dict[int, dict[str, Any]]],
    seed_rows: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Pool all 300 formal episodes per arm and weight geometry by selected samples."""
    result: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        returns: list[float] = []
        costs: list[float] = []
        successes: list[bool] = []
        for seed in SEEDS:
            evaluation = formal[arm][seed]
            returns.extend(float(value) for value in evaluation['returns'])
            costs.extend(float(value) for value in evaluation['costs'])
            successes.extend(int(value) > 0 for value in evaluation['goal_met_counts'])
        geometry_metrics = (
            'high_risk_normal_to_tangent_variance_ratio',
            'high_risk_dangerous_outward_velocity',
            'high_risk_outward_velocity_ratio',
            'high_risk_reliability',
            'high_risk_safety_weight',
        )
        pooled_geometry: dict[str, float] = {}
        for metric in geometry_metrics:
            weighted = [
                (
                    float(seed_rows[arm][seed][metric]),
                    int(seed_rows[arm][seed]['high_risk_selected_samples']),
                )
                for seed in SEEDS
                if math.isfinite(float(seed_rows[arm][seed][metric]))
            ]
            denominator = sum(count for _, count in weighted)
            pooled_geometry[metric] = (
                sum(value * count for value, count in weighted) / denominator
                if denominator
                else float('nan')
            )
        training_costs = [float(seed_rows[arm][seed]['training_cumulative_cost']) for seed in SEEDS]
        result[arm] = {
            'episodes': len(returns),
            'mean_return': statistics.fmean(returns),
            'success_rate': statistics.fmean(successes),
            'mean_cost': statistics.fmean(costs),
            'cost_cvar_20': tail_mean(costs),
            'training_cumulative_cost_mean': statistics.fmean(training_costs),
            'training_cumulative_cost_sum': sum(training_costs),
            **pooled_geometry,
        }
    return result


def select_final(
    seed_rows: dict[str, dict[int, dict[str, Any]]],
    pooled: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply the preregistered P47-versus-C1 deterministic freeze rule."""
    p47 = pooled['P47']
    c0 = pooled['C0']
    c1 = pooled['C1']
    paired_safety_counts = {
        metric: sum(
            float(seed_rows['P47'][seed][metric]) < float(seed_rows['C0'][seed][metric])
            for seed in SEEDS
        )
        for metric in ('mean_cost', 'cost_cvar_20')
    }
    safety_efficacy = (
        p47['mean_cost'] <= 0.90 * c0['mean_cost']
        and p47['cost_cvar_20'] <= 0.90 * c0['cost_cvar_20']
        and min(paired_safety_counts.values()) >= 2
    )
    paired_training_count = sum(
        float(seed_rows['P47'][seed]['training_cumulative_cost'])
        < float(seed_rows['C0'][seed]['training_cumulative_cost'])
        for seed in SEEDS
    )
    training_safety = (
        p47['training_cumulative_cost_mean'] <= 0.90 * c0['training_cumulative_cost_mean']
        and paired_training_count >= 2
    )
    safety_retention = (
        p47['mean_cost'] <= 1.10 * c1['mean_cost']
        and p47['cost_cvar_20'] <= 1.10 * c1['cost_cvar_20']
    )
    pooled_performance = (
        p47['mean_return'] >= c1['mean_return'] - 0.10
        and p47['success_rate'] >= c1['success_rate'] - 0.10
    )
    paired_performance = all(
        float(seed_rows['P47'][seed][metric]) >= float(seed_rows['C1'][seed][metric]) - 0.20
        for seed in SEEDS
        for metric in ('mean_return', 'success_rate')
    )
    performance_retention = pooled_performance and paired_performance
    gates = {
        'safety_efficacy_vs_C0_200k': safety_efficacy,
        'training_safety_vs_C0_200k': training_safety,
        'safety_retention_vs_C1_100k': safety_retention,
        'performance_retention_vs_C1_100k': performance_retention,
    }
    selected = 'P47' if all(gates.values()) else 'C1'
    return {
        'label': f'freeze_{selected}',
        'selected_arm': selected,
        'all_gates_pass': all(gates.values()),
        'gates': gates,
        'paired_safety_improvement_counts_vs_C0': paired_safety_counts,
        'paired_training_cost_improvement_count_vs_C0': paired_training_count,
        'geometry_used_as_gate': False,
        'pointgoal_search_closed': True,
    }


def _relative_reduction(candidate: float, baseline: float) -> float:
    """Return the signed fractional reduction from baseline to candidate."""
    if baseline == 0.0:
        return float('nan')
    return (baseline - candidate) / abs(baseline)


def scientific_interpretation(
    seed_rows: dict[str, dict[int, dict[str, Any]]],
    pooled: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Interpret P47 primarily against equal-budget C0, without adding freeze gates."""
    p47 = pooled['P47']
    c0 = pooled['C0']
    c1 = pooled['C1']

    safety_specs = {
        'training_cumulative_cost': (
            'training_cumulative_cost_mean',
            'training_cumulative_cost',
        ),
        'evaluation_mean_cost': ('mean_cost', 'mean_cost'),
        'evaluation_cvar_20': ('cost_cvar_20', 'cost_cvar_20'),
    }
    safety_comparison: dict[str, dict[str, Any]] = {}
    for name, (pooled_metric, seed_metric) in safety_specs.items():
        candidate = float(p47[pooled_metric])
        baseline = float(c0[pooled_metric])
        safety_comparison[name] = {
            'C0_200k': baseline,
            'P47_200k': candidate,
            'delta_P47_minus_C0': candidate - baseline,
            'relative_reduction': _relative_reduction(candidate, baseline),
            'improved': candidate < baseline,
            'paired_seed_improvement_count': sum(
                float(seed_rows['P47'][seed][seed_metric])
                < float(seed_rows['C0'][seed][seed_metric])
                for seed in SEEDS
            ),
        }

    performance_comparison: dict[str, dict[str, Any]] = {}
    for metric in ('mean_return', 'success_rate'):
        candidate = float(p47[metric])
        baseline = float(c0[metric])
        performance_comparison[metric] = {
            'C0_200k': baseline,
            'P47_200k': candidate,
            'delta_P47_minus_C0': candidate - baseline,
            'paired_seed_improvement_count': sum(
                float(seed_rows['P47'][seed][metric]) > float(seed_rows['C0'][seed][metric])
                for seed in SEEDS
            ),
            'paired_seed_nondecrease_count': sum(
                float(seed_rows['P47'][seed][metric]) >= float(seed_rows['C0'][seed][metric])
                for seed in SEEDS
            ),
        }
    all_safety_indicators_improved = all(
        comparison['improved'] for comparison in safety_comparison.values()
    )
    task_completion_observed = float(p47['success_rate']) > 0.0
    performance_nondecreasing = all(
        comparison['delta_P47_minus_C0'] >= 0.0 for comparison in performance_comparison.values()
    )
    if all_safety_indicators_improved and task_completion_observed:
        overall_label = (
            'equal_budget_pareto_improvement'
            if performance_nondecreasing
            else 'effective_safety_performance_tradeoff'
        )
    elif any(comparison['improved'] for comparison in safety_comparison.values()):
        overall_label = 'mixed_equal_budget_safety_performance_result'
    else:
        overall_label = 'no_equal_budget_safety_improvement'

    training_comparison = safety_comparison['training_cumulative_cost']
    training_label = (
        'safer_training_period_exploration'
        if training_comparison['improved']
        else 'no_training_period_safety_improvement'
    )

    geometry_metrics = (
        'high_risk_normal_to_tangent_variance_ratio',
        'high_risk_dangerous_outward_velocity',
    )
    geometry_comparison: dict[str, dict[str, Any]] = {}
    for metric in geometry_metrics:
        candidate = float(p47[metric])
        baseline = float(c0[metric])
        finite = math.isfinite(candidate) and math.isfinite(baseline)
        geometry_comparison[metric] = {
            'C0_200k': baseline,
            'P47_200k': candidate,
            'delta_P47_minus_C0': candidate - baseline if finite else float('nan'),
            'lower_in_P47': finite and candidate < baseline,
        }
    finite_geometry = all(
        math.isfinite(float(comparison['delta_P47_minus_C0']))
        for comparison in geometry_comparison.values()
    )
    geometry_improvements = sum(
        bool(comparison['lower_in_P47']) for comparison in geometry_comparison.values()
    )
    if not finite_geometry:
        geometry_label = 'geometry_evidence_unavailable'
    elif geometry_improvements == len(geometry_comparison):
        geometry_label = 'direction_mechanism_supported'
    elif geometry_improvements:
        geometry_label = 'mixed_direction_mechanism_evidence'
    else:
        geometry_label = 'direction_mechanism_not_supported_by_both_diagnostics'

    historical_deltas = {
        metric: {
            'C1_100k': float(c1[metric]),
            'P47_200k': float(p47[metric]),
            'delta_P47_minus_C1': float(p47[metric]) - float(c1[metric]),
        }
        for metric in ('mean_return', 'success_rate', 'mean_cost', 'cost_cvar_20')
    }
    return {
        'primary_comparison': 'P47@200000_vs_C0@200000_equal_training_budget',
        'statistical_unit': 'training_seed',
        'n_training_seeds_per_arm': len(SEEDS),
        'pooled_evaluation_episodes_per_arm': 300,
        'episode_count_is_not_the_independent_sample_size': True,
        'overall_safety_performance': {
            'label': overall_label,
            'aggregate_claim_only': True,
            'all_seed_pareto_claim': False,
            'all_three_safety_indicators_improved': all_safety_indicators_improved,
            'task_completion_observed': task_completion_observed,
            'performance_nondecreasing': performance_nondecreasing,
            'safety': safety_comparison,
            'performance': performance_comparison,
        },
        'training_period_safety_exploration': {
            'label': training_label,
            **training_comparison,
        },
        'direction_geometry_mechanism': {
            'label': geometry_label,
            'population': 'highest_safety_margin_20_percent',
            'diagnostic_only': True,
            'used_as_elimination_gate': False,
            'metrics': geometry_comparison,
        },
        'C1_reference': {
            'role': 'historical_reference_only',
            'training_steps': 100_000,
            'used_in_primary_scientific_interpretation': False,
            'retained_in_preregistered_machine_audit': True,
            'metrics': historical_deltas,
        },
        'interpretation_is_not_a_freeze_gate': True,
    }


def _checkpoint_path(
    run_root: Path,
    p45_root: Path,
    arm: str,
    seed: int,
    step: int | str,
) -> Path:
    if step == 'final_full':
        root = p45_root / f'{arm}-{seed}' if arm == 'C1' else run_root / 'runs' / f'{arm}-{seed}'
        return _single(root, 'training/**/checkpoints/final_full.pt')
    if arm == 'C1' or (arm == 'C0' and int(step) <= 100_000):
        root = p45_root / f'{arm}-{seed}'
    else:
        root = run_root / 'runs' / f'{arm}-{seed}'
    return _single(root, f'training/**/checkpoints/milestone_{int(step)}.pt')


def checkpoint_rows(run_root: Path, p45_root: Path) -> list[dict[str, Any]]:
    """Seal every selected learning-curve checkpoint and each final recovery checkpoint."""
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            for step in INTERMEDIATE_STEPS[arm]:
                path = _checkpoint_path(run_root, p45_root, arm, seed, step)
                rows.append(
                    {
                        'arm': arm,
                        'training_seed': seed,
                        'kind': 'lightweight',
                        'milestone': step,
                        'path': str(path),
                        'bytes': path.stat().st_size,
                        'sha256': sha256_path(path),
                    },
                )
            path = _checkpoint_path(run_root, p45_root, arm, seed, 'final_full')
            rows.append(
                {
                    'arm': arm,
                    'training_seed': seed,
                    'kind': 'full_replay',
                    'milestone': FINAL_STEPS[arm],
                    'path': str(path),
                    'bytes': path.stat().st_size,
                    'sha256': sha256_path(path),
                },
            )
    return rows


def paired_rows(seed_rows: dict[str, dict[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Return transparent P47-minus-comparator paired final deltas."""
    rows: list[dict[str, Any]] = []
    for comparator in ('C0', 'C1'):
        for seed in SEEDS:
            row: dict[str, Any] = {'comparison': f'P47-minus-{comparator}', 'training_seed': seed}
            for metric in (
                'mean_return',
                'success_rate',
                'mean_cost',
                'cost_cvar_20',
                'training_cumulative_cost',
                'high_risk_normal_to_tangent_variance_ratio',
                'high_risk_dangerous_outward_velocity',
            ):
                row[f'delta_{metric}'] = float(seed_rows['P47'][seed][metric]) - float(
                    seed_rows[comparator][seed][metric],
                )
            rows.append(row)
    return rows


def _render_report(result: dict[str, Any]) -> str:
    pooled = result['pooled_formal']
    scientific = result['scientific_interpretation']
    overall = scientific['overall_safety_performance']
    training = scientific['training_period_safety_exploration']
    geometry = scientific['direction_geometry_mechanism']
    decision = result['machine_audit']
    evaluation_cost = overall['safety']['evaluation_mean_cost']
    evaluation_cvar = overall['safety']['evaluation_cvar_20']
    return_result = overall['performance']['mean_return']
    success_result = overall['performance']['success_rate']
    variance_geometry = geometry['metrics']['high_risk_normal_to_tangent_variance_ratio']
    outward_geometry = geometry['metrics']['high_risk_dangerous_outward_velocity']
    overall_text = {
        'equal_budget_pareto_improvement': (
            'P47 achieves an aggregate Pareto improvement under an equal training budget on '
            'SafetyPointGoal1-v0: all three aggregate safety indicators improve without an '
            'aggregate return or success-rate decrease.'
        ),
        'effective_safety_performance_tradeoff': (
            'P47 demonstrates an effective safety-performance tradeoff: training cost, final '
            'evaluation cost, and CVaR-20 all improve and task completion remains observable, '
            'while return and/or success pays a performance cost.'
        ),
        'mixed_equal_budget_safety_performance_result': (
            'The equal-budget result is mixed: at least one safety indicator improves, but the '
            'three safety indicators do not improve together with observable task completion.'
        ),
        'no_equal_budget_safety_improvement': (
            'P47 does not show an aggregate equal-budget safety improvement over C0.'
        ),
    }[overall['label']]
    geometry_text = {
        'direction_mechanism_supported': (
            'Both high-risk diagnostics are lower under P47, supporting activation of the '
            'intended direction mechanism.'
        ),
        'mixed_direction_mechanism_evidence': (
            'Only one high-risk diagnostic is lower under P47, so mechanism evidence is mixed.'
        ),
        'direction_mechanism_not_supported_by_both_diagnostics': (
            'Neither high-risk diagnostic is lower under P47, so these diagnostics do not '
            'support the intended direction mechanism.'
        ),
        'geometry_evidence_unavailable': 'Finite high-risk geometry evidence is unavailable.',
    }[geometry['label']]
    lines = [
        '# STAR P4.7 safety-aware geometry final report',
        '',
        (
            'P4.7 is the final PointGoal method modification. Scientific effectiveness is '
            'interpreted primarily from the equal-training-budget P47@200k versus C0@200k '
            'comparison. C1@100k remains a historical reference and part of the untouched '
            'preregistered machine audit.'
        ),
        '',
        '## Aggregate formal results (3 training seeds; 300 evaluation episodes per arm)',
        '',
        (
            'The independent statistical unit is the training seed (`n=3` per arm). The 300 '
            'evaluation episodes per arm are pooled observations and are not independent '
            'training repetitions.'
        ),
        '',
        '| Arm | Step | Return | Success | Cost | CVaR-20 | Train cost | High-risk V_N/V_T | High-risk outward |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for arm in ARMS:
        row = pooled[arm]
        lines.append(
            f"| {arm} | {FINAL_STEPS[arm]} | {row['mean_return']:.4f} | "
            f"{row['success_rate']:.3f} | {row['mean_cost']:.3f} | "
            f"{row['cost_cvar_20']:.3f} | {row['training_cumulative_cost_mean']:.1f} | "
            f"{row['high_risk_normal_to_tangent_variance_ratio']:.4f} | "
            f"{row['high_risk_dangerous_outward_velocity']:.4f} |",
        )
    lines.extend(
        [
            '',
            '## Primary scientific conclusion: P47@200k vs C0@200k',
            '',
            f"**{overall['label']}**",
            '',
            '### 1. Overall safety-performance',
            '',
            overall_text,
            '',
            (
                f'- Evaluation mean-cost reduction: '
                f"{evaluation_cost['relative_reduction']:.1%} "
                f"({evaluation_cost['C0_200k']:.3f} -> "
                f"{evaluation_cost['P47_200k']:.3f})"
            ),
            (
                f'- Evaluation CVaR-20 reduction: '
                f"{evaluation_cvar['relative_reduction']:.1%} "
                f"({evaluation_cvar['C0_200k']:.3f} -> "
                f"{evaluation_cvar['P47_200k']:.3f})"
            ),
            (
                f"- Return delta: {return_result['delta_P47_minus_C0']:+.4f}; "
                f'success-rate delta: '
                f"{100.0 * success_result['delta_P47_minus_C0']:+.1f} percentage points; "
                f"task completion observed: {overall['task_completion_observed']}"
            ),
            (
                f'- Paired training-seed improvements: return '
                f"{return_result['paired_seed_improvement_count']}/3; success "
                f"{success_result['paired_seed_improvement_count']}/3. The aggregate Pareto "
                'claim is not an all-seed Pareto claim.'
            ),
            '',
            '### 2. Training-period safety exploration',
            '',
            f"**{training['label']}**",
            '',
            (
                f'Mean cumulative training-cost reduction is '
                f"{training['relative_reduction']:.1%} "
                f"({training['C0_200k']:.1f} -> {training['P47_200k']:.1f}); "
                f"P47 is lower in {training['paired_seed_improvement_count']}/3 paired seeds."
            ),
            '',
            '### 3. Direction-geometry mechanism',
            '',
            f"**{geometry['label']}**",
            '',
            geometry_text,
            '',
            (
                f"- High-risk V_N/V_T: {variance_geometry['C0_200k']:.4f} -> "
                f"{variance_geometry['P47_200k']:.4f} "
                f"(delta {variance_geometry['delta_P47_minus_C0']:+.4f})"
            ),
            (
                f'- High-risk dangerous outward velocity: '
                f"{outward_geometry['C0_200k']:.4f} -> "
                f"{outward_geometry['P47_200k']:.4f} "
                f"(delta {outward_geometry['delta_P47_minus_C0']:+.4f})"
            ),
            '',
            (
                'This evidence is computed only on the highest-risk 20% of samples and is '
                'diagnostic; it is not an elimination or freeze gate.'
            ),
            '',
            '## C1 historical reference',
            '',
            (
                'C1 was trained for 100k steps, so it is not the primary equal-budget '
                'effectiveness comparator. Its values remain visible above and remain inputs to '
                'the preregistered audit, but C1 alone does not determine whether P47 had an '
                'effective safety-performance tradeoff.'
            ),
            '',
            '## Preserved preregistered machine audit',
            '',
            f"**{decision['label']}**",
            '',
        ],
    )
    locked_gate_order = (
        'safety_efficacy_vs_C0_200k',
        'training_safety_vs_C0_200k',
        'safety_retention_vs_C1_100k',
        'performance_retention_vs_C1_100k',
    )
    for name in locked_gate_order:
        passed = decision['gates'][name]
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            '',
            (
                'The four gates and their freeze result are preserved exactly as preregistered; '
                'they are a conservative machine audit, not a post-hoc replacement for the '
                'equal-budget scientific interpretation above.'
            ),
            '',
            (
                'The PointGoal weight/method search is now closed. The machine-audit-selected '
                'method is frozen in `p4_7_frozen_pointgoal_config.yaml`; the next stage is the '
                'preregistered SHAPO four-environment, five-training-seed experiment.'
            ),
            '',
            '## Evidence files',
            '',
            '- `p4_7_checkpoint_metrics.csv`: common 5x4 checkpoint curves',
            '- `p4_7_formal_final_seed_metrics.csv`: every formal arm-by-training-seed result',
            '- `p4_7_paired_final_deltas.csv`: paired P47 deltas',
            '- `p4_7_checkpoint_sha256.csv`: immutable checkpoint identities',
            '- `p4_7_scientific_interpretation.json`: the three scientific conclusions',
            '- `p4_7_final_summary.json`: scientific interpretation plus preserved machine audit',
            '',
        ],
    )
    return '\n'.join(lines)


def summarize(protocol_path: Path, run_root: Path, p45_root: Path, output: Path) -> dict[str, Any]:
    """Validate all evidence, summarize it, and emit the deterministic frozen config."""
    if sha256_path(protocol_path) != PROTOCOL_SHA256:
        raise ValueError('P4.7 protocol hash does not match its locked seal')
    protocol = _json(protocol_path)
    if protocol.get('status') != 'locked_before_any_p4_7_training':
        raise ValueError('P4.7 protocol was not locked before training')
    if not (run_root / 'training_matrix_complete').is_file():
        raise ValueError('P4.7 training matrix is incomplete')
    if not (run_root / 'evaluation/evaluation_complete').is_file():
        raise ValueError('P4.7 fixed evaluation matrix is incomplete')

    receipts: dict[str, dict[int, dict[str, Any]]] = {
        arm: {seed: _receipt(run_root, p45_root, arm, seed) for seed in SEEDS} for arm in ARMS
    }
    new_sources = {
        str(receipts[arm][seed]['source_commit']) for arm in ('C0', 'P47') for seed in SEEDS
    }
    if len(new_sources) != 1:
        raise ValueError(f'P4.7 new runs used multiple source commits: {sorted(new_sources)}')

    checkpoint_metrics: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            for step in INTERMEDIATE_STEPS[arm]:
                evaluation = load_evaluation(run_root, 'intermediate', arm, seed, step)
                checkpoint_metrics.append(
                    evaluation_row(
                        arm,
                        seed,
                        step,
                        evaluation,
                        cumulative_cost_at(run_root, p45_root, arm, seed, step),
                    ),
                )

    formal: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    seed_rows: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    final_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            step = FINAL_STEPS[arm]
            evaluation = load_evaluation(run_root, 'formal', arm, seed, step)
            formal[arm][seed] = evaluation
            cumulative = float(receipts[arm][seed]['training_summary']['cumulative_training_cost'])
            row = evaluation_row(arm, seed, step, evaluation, cumulative)
            seed_rows[arm][seed] = row
            final_rows.append(row)

    pooled = pooled_formal(formal, seed_rows)
    decision = select_final(seed_rows, pooled)
    interpretation = scientific_interpretation(seed_rows, pooled)
    hashes = checkpoint_rows(run_root, p45_root)
    result = {
        'stage': 'P4.7-final-safety-aware-geometry-summary',
        'protocol_version': protocol['protocol_version'],
        'protocol_sha256': PROTOCOL_SHA256,
        'training_source_commit': next(iter(new_sources)),
        'formal_evaluation': {
            'base_seeds': list(FORMAL_BASE_SEEDS),
            'episodes_per_base_seed': 20,
            'episodes_per_training_seed': 100,
            'pooled_episodes_per_arm': 300,
            'flow_seed_offset': 1_000_003,
        },
        'pooled_formal': pooled,
        'primary_scientific_comparison': 'P47@200k_vs_C0@200k_equal_training_budget',
        'scientific_interpretation': interpretation,
        'machine_audit': decision,
        # Retained for compatibility with the sealed finalizer and downstream readers.
        'decision': decision,
        'geometry_role': 'diagnostic_only_highest_margin_20_percent',
        'next_stage': 'SHAPO_four_environments_x_five_training_seeds',
    }

    output.mkdir(parents=True, exist_ok=True)
    atomic_csv(output / 'p4_7_checkpoint_metrics.csv', checkpoint_metrics)
    atomic_csv(output / 'p4_7_formal_final_seed_metrics.csv', final_rows)
    atomic_csv(output / 'p4_7_paired_final_deltas.csv', paired_rows(seed_rows))
    atomic_csv(output / 'p4_7_checkpoint_sha256.csv', hashes)
    atomic_json(output / 'p4_7_scientific_interpretation.json', interpretation)
    atomic_json(output / 'p4_7_final_summary.json', result)
    atomic_text(output / 'P4_7_SAFETY_AWARE_GEOMETRY_REPORT.md', _render_report(result))

    selected = str(decision['selected_arm'])
    selected_config = (
        Path('configs/star/p4_7_safety_aware_geometry.yaml')
        if selected == 'P47'
        else Path('configs/star/p4_5_c1_directional_loss.yaml')
    )
    frozen = yaml.safe_load(selected_config.read_text(encoding='utf-8'))
    frozen['freeze_metadata'] = {
        'protocol_version': protocol['protocol_version'],
        'protocol_sha256': PROTOCOL_SHA256,
        'selected_arm': selected,
        'selection_label': decision['label'],
        'selection_basis': 'preserved_preregistered_conservative_machine_audit',
        'primary_scientific_comparison': result['primary_scientific_comparison'],
        'scientific_interpretation_label': interpretation['overall_safety_performance']['label'],
        'C1_scientific_role': 'historical_reference_only',
        'pointgoal_search_closed': True,
        'next_stage': result['next_stage'],
    }
    atomic_text(
        output / 'p4_7_frozen_pointgoal_config.yaml',
        yaml.safe_dump(frozen, sort_keys=False),
    )
    return result


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--protocol', type=Path, required=True)
    result.add_argument('--run-root', type=Path, required=True)
    result.add_argument('--p45-root', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    return result


def main() -> None:
    """Run P4.7 finalization and print the locked decision."""
    args = parser().parse_args()
    result = summarize(args.protocol, args.run_root, args.p45_root, args.output)
    print(json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()

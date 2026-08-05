#!/usr/bin/env python3
"""Summarize the preregistered P4.2 B0/B1 backbone diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


SEEDS = (2027, 2028, 2029)
ENDPOINTS = (0, 50_000)
FUNCTIONAL_METRICS = (
    'mean_return',
    'goal_met_episode_rate',
    'stall_ratio',
    'mean_start_end_displacement',
)


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one CSV table."""
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def number(value: Any, location: str) -> float:
    """Return one required finite number."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{location} must be numeric, got {value!r}') from error
    if not math.isfinite(result):
        raise ValueError(f'{location} must be finite, got {result!r}')
    return result


def optional_number(value: Any) -> float | None:
    """Return an optional finite number."""
    if value in (None, ''):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def reward_td_trend(rows: list[dict[str, str]], seed: int) -> dict[str, Any]:
    """Apply the frozen two-times-minimum plus rising-tail divergence rule."""
    ordered = sorted(rows, key=lambda row: int(number(row['TotalEnvSteps'], 'TotalEnvSteps')))
    if len(ordered) < 4:
        raise ValueError(f'seed {seed} needs at least four diagnostic rows')
    steps = [int(number(row['TotalEnvSteps'], 'TotalEnvSteps')) for row in ordered]
    if len(steps) != len(set(steps)) or steps[-1] != ENDPOINTS[-1]:
        raise ValueError(f'seed {seed} diagnostics have invalid step coverage')
    values = [number(row['Critic/RewardTD'], 'Critic/RewardTD') for row in ordered]
    historical_minimum = min(values)
    if historical_minimum <= 0.0:
        raise ValueError(f'seed {seed} reward TD minimum must be positive')
    last_four = values[-4:]
    strictly_rising = all(after > before for before, after in zip(last_four, last_four[1:]))
    final_to_minimum = values[-1] / historical_minimum
    divergence = strictly_rising and final_to_minimum >= 2.0
    return {
        'historical_minimum': historical_minimum,
        'historical_minimum_step': steps[values.index(historical_minimum)],
        'final': values[-1],
        'final_to_historical_minimum_ratio': final_to_minimum,
        'last_four_steps': steps[-4:],
        'last_four_values': last_four,
        'strictly_rising_last_three_intervals': strictly_rising,
        'monotonic_divergence': divergence,
    }


def external_rows(path: Path) -> dict[int, dict[int, dict[str, Any]]]:
    """Validate and namespace the fixed-random-flow endpoint evaluations."""
    grouped: dict[int, dict[int, dict[str, Any]]] = {}
    rows = read_csv(path)
    for row in rows:
        seed = int(number(row['training_seed'], 'training_seed'))
        milestone = int(number(row['milestone'], 'milestone'))
        if row['evaluation_mode'] != 'fixed-random-flow':
            raise ValueError('backbone decisions require fixed-random-flow evaluation')
        if int(number(row['episodes'], 'episodes')) != 20:
            raise ValueError('backbone decisions require exactly 20 episodes per checkpoint')
        metrics = {key: number(row[key], f'{seed}.{milestone}.{key}') for key in FUNCTIONAL_METRICS}
        metrics.update(
            {
                'episodes': 20,
                'external_mean_cost': number(row['mean_cost'], 'mean_cost'),
                'external_goal_met_total': int(number(row['goal_met_total'], 'goal_met_total')),
                'external_brier_skill': optional_number(row.get('brier_skill')),
                'external_auroc': optional_number(row.get('auroc')),
            },
        )
        if milestone in grouped.setdefault(seed, {}):
            raise ValueError(f'duplicate evaluation for seed {seed} at {milestone}')
        grouped[seed][milestone] = metrics
    expected = set(SEEDS)
    if set(grouped) != expected or any(set(grouped[seed]) != set(ENDPOINTS) for seed in SEEDS):
        raise ValueError('evaluation table must contain all three seeds at 0 and 50000')
    return grouped


def variant_summary(
    name: str,
    run_dirs: list[Path],
    receipts: list[Path],
    evaluation_csv: Path,
) -> dict[str, Any]:
    """Validate one B0/B1 arm and compute its frozen functional gate."""
    if name not in {'B0', 'B1'}:
        raise ValueError(f'unknown backbone arm {name}')
    if len(run_dirs) != 3 or len(receipts) != 3:
        raise ValueError(f'{name} requires exactly three run directories and receipts')
    global_enabled = name == 'B1'
    evaluations = external_rows(evaluation_csv)
    receipt_by_seed = {int(read_json(path)['seed']): read_json(path) for path in receipts}
    if set(receipt_by_seed) != set(SEEDS):
        raise ValueError(f'{name} receipts must use seeds {SEEDS}')

    trends: dict[str, dict[str, Any]] = {}
    checkpoints: dict[str, dict[str, int]] = {}
    training_seconds = 0.0
    estimated_cost_cny = 0.0
    observed_run_seeds: set[int] = set()
    for run_dir in run_dirs:
        summary = read_json(run_dir / 'artifacts/training_summary.json')
        manifest = read_json(run_dir / 'artifacts/run_manifest.json')
        seed = int(summary['seed'])
        if seed in observed_run_seeds:
            raise ValueError(f'{name} has duplicate run seed {seed}')
        observed_run_seeds.add(seed)
        receipt = receipt_by_seed.get(seed)
        if receipt is None:
            raise ValueError(f'{name} has no receipt for seed {seed}')
        if summary['status'] != 'completed' or receipt['status'] != 'passed':
            raise ValueError(f'{name} seed {seed} did not complete')
        if int(summary['env_steps']) != 50_000 or int(summary['updates']) != 48_000:
            raise ValueError(f'{name} seed {seed} has incorrect training counters')
        if int(receipt['steps']) != 50_000 or int(receipt['updates']) != 48_000:
            raise ValueError(f'{name} seed {seed} receipt has incorrect counters')
        if summary['global_safety_constraint_enabled'] is not global_enabled:
            raise ValueError(f'{name} seed {seed} has the wrong global-safety switch')
        if summary['local_safety_geometry_enabled'] is not False:
            raise ValueError(f'{name} seed {seed} did not disable local geometry')
        if summary['training_geometry_enabled'] is not False:
            raise ValueError(f'{name} seed {seed} activated training geometry')
        expected_manifest = {
            'reward_flow_enabled': True,
            'global_safety_constraint_enabled': global_enabled,
            'local_safety_geometry_enabled': False,
        }
        if manifest['p4_2_backbone_ablation'] != expected_manifest:
            raise ValueError(f'{name} seed {seed} manifest does not match the arm')
        if tuple(int(value) for value in manifest['eval_milestones']) != ENDPOINTS:
            raise ValueError(f'{name} seed {seed} has incorrect milestones')
        checkpoint_paths = {
            label: run_dir / 'checkpoints' / filename
            for label, filename in {
                'milestone_0': 'milestone_0.pt',
                'milestone_50000': 'milestone_50000.pt',
                'latest_full': 'latest_full.pt',
                'final_full': 'final_full.pt',
            }.items()
        }
        missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f'{name} seed {seed} missing checkpoints: {missing}')
        checkpoints[str(seed)] = {
            label: path.stat().st_size for label, path in checkpoint_paths.items()
        }
        trends[str(seed)] = reward_td_trend(
            read_csv(run_dir / 'artifacts/diagnostics.csv'),
            seed,
        )
        training_seconds += number(summary['training_seconds'], 'training_seconds')
        estimated_cost_cny += number(receipt['estimated_cost_cny'], 'estimated_cost_cny')
    if observed_run_seeds != set(SEEDS):
        raise ValueError(f'{name} runs must use seeds {SEEDS}')

    pooled: dict[str, dict[str, float]] = {}
    for milestone in ENDPOINTS:
        pooled[str(milestone)] = {
            metric: statistics.fmean(evaluations[seed][milestone][metric] for seed in SEEDS)
            for metric in FUNCTIONAL_METRICS
        }
    initial, final = pooled[str(ENDPOINTS[0])], pooled[str(ENDPOINTS[1])]
    criteria = {
        'success_rate_rises': final['goal_met_episode_rate'] > initial['goal_met_episode_rate'],
        'return_improves_vs_initial': final['mean_return'] > initial['mean_return'],
        'stall_does_not_rise': final['stall_ratio'] <= initial['stall_ratio'],
        'displacement_does_not_decline': (
            final['mean_start_end_displacement'] >= initial['mean_start_end_displacement']
        ),
        'reward_td_not_monotonic_divergence': all(
            not trend['monotonic_divergence'] for trend in trends.values()
        ),
    }
    return {
        'name': name,
        'global_safety_constraint_enabled': global_enabled,
        'local_safety_geometry_enabled': False,
        'pooled_external_evaluation': pooled,
        'external_evaluation_by_seed': {
            str(seed): {str(step): evaluations[seed][step] for step in ENDPOINTS} for seed in SEEDS
        },
        'reward_td_trend_by_seed': trends,
        'criteria': criteria,
        'functional_gate_passed': all(criteria.values()),
        'checkpoint_bytes_by_seed': checkpoints,
        'training_seconds': training_seconds,
        'estimated_cost_cny': estimated_cost_cny,
    }


def backbone_decision(b0: dict[str, Any], b1: dict[str, Any]) -> str:
    """Return the preregistered causal-localization branch."""
    if not b0['functional_gate_passed']:
        return 'repair_backbone_and_pause_safety'
    if not b1['functional_gate_passed']:
        return 'diagnose_global_cost_dual_conservative_collapse'
    return 'proceed_to_bellman_probability_critic'


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    for arm in ('b0', 'b1'):
        result.add_argument(f'--{arm}-run-dirs', nargs=3, type=Path, required=True)
        result.add_argument(f'--{arm}-receipts', nargs=3, type=Path, required=True)
        result.add_argument(f'--{arm}-evaluation-csv', type=Path, required=True)
    result.add_argument('--training-source-commit', required=True)
    result.add_argument('--output', type=Path, required=True)
    return result


def main() -> None:
    """Validate both arms and write the machine-readable backbone decision."""
    args = parser().parse_args()
    b0 = variant_summary('B0', args.b0_run_dirs, args.b0_receipts, args.b0_evaluation_csv)
    b1 = variant_summary('B1', args.b1_run_dirs, args.b1_receipts, args.b1_evaluation_csv)
    decision = backbone_decision(b0, b1)
    payload = {
        'schema_version': 1,
        'stage': 'P4.2-backbone-causal-localization',
        'training_source_commit': args.training_source_commit,
        'seeds': list(SEEDS),
        'target_steps': ENDPOINTS[-1],
        'external_evaluation_protocol': {
            'mode': 'fixed-random-flow',
            'episodes_per_checkpoint_per_seed': 20,
            'milestones': list(ENDPOINTS),
        },
        'reward_td_divergence_rule': {
            'tail': 'last four 5k diagnostic rows',
            'strictly_rising': True,
            'final_to_historical_minimum_ratio_at_least': 2.0,
            'arm_pass_requires_no_seed_divergence': True,
        },
        'functional_gate_rule': {
            'pooled_success_rate': 'strictly rises from 0 to 50k',
            'pooled_return': 'strictly improves from 0 to 50k',
            'pooled_stall': 'does not rise from 0 to 50k',
            'pooled_displacement': 'does not decline from 0 to 50k',
            'reward_td': 'no seed triggers the registered monotonic-divergence rule',
            'arm_pass': 'all five criteria must pass',
        },
        'arms': {'B0': b0, 'B1': b1},
        'backbone_decision': decision,
        'final_p4_verdict': None,
        'formal_four_environment_experiment_launched': False,
        'continuation_beyond_50k_launched': False,
    }
    atomic_json(args.output, payload)
    print(json.dumps({'backbone_decision': decision, 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()

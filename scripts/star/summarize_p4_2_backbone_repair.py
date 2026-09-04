#!/usr/bin/env python3
"""Apply the preregistered P4.2 B0-R1 reward-backbone repair gate."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from scripts.star.summarize_p4_2_backbone import (
    ENDPOINTS,
    SEEDS,
    atomic_json,
    number,
    optional_number,
    read_csv,
    read_json,
    reward_td_trend,
)


TRAINING_SOURCE_COMMIT = 'f4371320a1683bc2130cd687213ffbac2d7da9d9'
PARENT_DECISION = 'repair_backbone_and_pause_safety'
EXTERNAL_METRICS = (
    'mean_return',
    'goal_met_episode_rate',
    'stall_ratio',
    'mean_step_displacement',
    'mean_start_end_displacement',
    'mean_action_norm',
)


def external_rows(path: Path) -> dict[int, dict[int, dict[str, Any]]]:
    """Validate fixed-random endpoint evaluations with the repaired movement metric."""
    grouped: dict[int, dict[int, dict[str, Any]]] = {}
    for row in read_csv(path):
        seed = int(number(row['training_seed'], 'training_seed'))
        milestone = int(number(row['milestone'], 'milestone'))
        if row['evaluation_mode'] != 'fixed-random-flow' or row['evaluation_role'] != 'primary':
            raise ValueError('B0-R1 decisions require primary fixed-random-flow evaluation')
        if int(number(row['episodes'], 'episodes')) != 20:
            raise ValueError('B0-R1 decisions require exactly 20 episodes per checkpoint')
        metrics = {key: number(row[key], f'{seed}.{milestone}.{key}') for key in EXTERNAL_METRICS}
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
    if set(grouped) != set(SEEDS) or any(set(grouped[seed]) != set(ENDPOINTS) for seed in SEEDS):
        raise ValueError('evaluation table must contain seeds 2027-2029 at 0 and 50000')
    return grouped


def validate_parent(path: Path) -> dict[str, Any]:
    """Require the immutable failed B0/B1 decision that authorized this repair."""
    payload = read_json(path)
    if payload.get('backbone_decision') != PARENT_DECISION:
        raise ValueError('parent decision did not pause safety for backbone repair')
    if payload.get('training_source_commit') != TRAINING_SOURCE_COMMIT:
        raise ValueError('parent decision has an unexpected training source')
    if payload.get('final_p4_verdict') is not None:
        raise ValueError('parent decision must keep final_p4_verdict null')
    return payload


def repair_summary(
    run_dirs: list[Path],
    receipts: list[Path],
    evaluation_csv: Path,
) -> dict[str, Any]:
    """Validate all repaired runs and compute the frozen B0-R1 gate."""
    if len(run_dirs) != 3 or len(receipts) != 3:
        raise ValueError('B0-R1 requires exactly three run directories and receipts')
    evaluations = external_rows(evaluation_csv)
    receipt_payloads = [read_json(path) for path in receipts]
    receipt_by_seed = {int(payload['seed']): payload for payload in receipt_payloads}
    if set(receipt_by_seed) != set(SEEDS):
        raise ValueError(f'B0-R1 receipts must use seeds {SEEDS}')

    trends: dict[str, dict[str, Any]] = {}
    checkpoint_bytes: dict[str, dict[str, int]] = {}
    actor_updates: dict[str, int] = {}
    observed_seeds: set[int] = set()
    training_seconds = 0.0
    estimated_cost_cny = 0.0
    for run_dir in run_dirs:
        summary = read_json(run_dir / 'artifacts/training_summary.json')
        manifest = read_json(run_dir / 'artifacts/run_manifest.json')
        seed = int(summary['seed'])
        if seed in observed_seeds:
            raise ValueError(f'B0-R1 has duplicate run seed {seed}')
        observed_seeds.add(seed)
        receipt = receipt_by_seed.get(seed)
        if receipt is None or summary['status'] != 'completed' or receipt['status'] != 'passed':
            raise ValueError(f'B0-R1 seed {seed} did not complete')
        if int(summary['env_steps']) != 50_000 or int(summary['updates']) != 48_000:
            raise ValueError(f'B0-R1 seed {seed} has incorrect training counters')
        if int(receipt['steps']) != 50_000 or int(receipt['updates']) != 48_000:
            raise ValueError(f'B0-R1 seed {seed} receipt has incorrect counters')
        if summary['global_safety_constraint_enabled'] is not False:
            raise ValueError(f'B0-R1 seed {seed} enabled global safety')
        if summary['local_safety_geometry_enabled'] is not False:
            raise ValueError(f'B0-R1 seed {seed} enabled local geometry')
        if summary['training_geometry_enabled'] is not False:
            raise ValueError(f'B0-R1 seed {seed} activated training geometry')
        if summary['reward_target_actor_enabled'] is not True:
            raise ValueError(f'B0-R1 seed {seed} did not use the target flow')
        if int(summary['actor_update_interval']) != 2:
            raise ValueError(f'B0-R1 seed {seed} did not delay actor updates by two')
        observed_actor_updates = int(summary['actor_updates'])
        if not 0 < observed_actor_updates < int(summary['updates']):
            raise ValueError(f'B0-R1 seed {seed} has invalid actor-update accounting')
        actor_updates[str(seed)] = observed_actor_updates
        expected_ablation = {
            'reward_flow_enabled': True,
            'global_safety_constraint_enabled': False,
            'local_safety_geometry_enabled': False,
        }
        if manifest['p4_2_backbone_ablation'] != expected_ablation:
            raise ValueError(f'B0-R1 seed {seed} has the wrong isolation switches')
        expected_repair = {
            'target_flow_actor_for_reward_backup': True,
            'actor_update_interval': 2,
        }
        if manifest['p4_2_reward_backbone_repair'] != expected_repair:
            raise ValueError(f'B0-R1 seed {seed} has the wrong repair mechanism')
        if tuple(int(value) for value in manifest['eval_milestones']) != ENDPOINTS:
            raise ValueError(f'B0-R1 seed {seed} has incorrect milestones')
        paths = {
            label: run_dir / 'checkpoints' / filename
            for label, filename in {
                'milestone_0': 'milestone_0.pt',
                'milestone_50000': 'milestone_50000.pt',
                'latest_full': 'latest_full.pt',
                'final_full': 'final_full.pt',
            }.items()
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f'B0-R1 seed {seed} missing checkpoints: {missing}')
        checkpoint_bytes[str(seed)] = {label: path.stat().st_size for label, path in paths.items()}
        trends[str(seed)] = reward_td_trend(
            read_csv(run_dir / 'artifacts/diagnostics.csv'),
            seed,
        )
        training_seconds += number(summary['training_seconds'], 'training_seconds')
        estimated_cost_cny += number(receipt['estimated_cost_cny'], 'estimated_cost_cny')
    if observed_seeds != set(SEEDS):
        raise ValueError(f'B0-R1 runs must use seeds {SEEDS}')

    pooled = {
        str(milestone): {
            metric: statistics.fmean(evaluations[seed][milestone][metric] for seed in SEEDS)
            for metric in EXTERNAL_METRICS
        }
        for milestone in ENDPOINTS
    }
    initial, final = pooled[str(ENDPOINTS[0])], pooled[str(ENDPOINTS[1])]
    criteria = {
        'success_rate_rises': final['goal_met_episode_rate'] > initial['goal_met_episode_rate'],
        'return_improves_vs_initial': final['mean_return'] > initial['mean_return'],
        'stall_does_not_rise': final['stall_ratio'] <= initial['stall_ratio'],
        'step_displacement_does_not_decline': (
            final['mean_step_displacement'] >= initial['mean_step_displacement']
        ),
        'reward_td_not_monotonic_divergence': all(
            not trend['monotonic_divergence'] for trend in trends.values()
        ),
    }
    return {
        'name': 'B0-R1',
        'repair': {
            'target_flow_actor_for_reward_backup': True,
            'actor_update_interval': 2,
        },
        'pooled_external_evaluation': pooled,
        'external_evaluation_by_seed': {
            str(seed): {str(step): evaluations[seed][step] for step in ENDPOINTS} for seed in SEEDS
        },
        'reward_td_trend_by_seed': trends,
        'criteria': criteria,
        'functional_gate_passed': all(criteria.values()),
        'actor_updates_by_seed': actor_updates,
        'checkpoint_bytes_by_seed': checkpoint_bytes,
        'training_seconds': training_seconds,
        'estimated_cost_cny': estimated_cost_cny,
    }


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--run-dirs', nargs=3, type=Path, required=True)
    result.add_argument('--receipts', nargs=3, type=Path, required=True)
    result.add_argument('--evaluation-csv', type=Path, required=True)
    result.add_argument('--parent-decision', type=Path, required=True)
    result.add_argument('--repair-source-commit', required=True)
    result.add_argument('--output', type=Path, required=True)
    return result


def main() -> None:
    """Write a strict machine-readable repair decision without a final P4 verdict."""
    args = parser().parse_args()
    validate_parent(args.parent_decision)
    repair = repair_summary(args.run_dirs, args.receipts, args.evaluation_csv)
    passed = repair['functional_gate_passed']
    decision = (
        'proceed_to_bellman_probability_critic'
        if passed
        else 'continue_backbone_repair_and_pause_safety'
    )
    payload = {
        'schema_version': 1,
        'stage': 'P4.2-B0-R1-reward-backbone-repair',
        'parent_training_source_commit': TRAINING_SOURCE_COMMIT,
        'parent_backbone_decision': PARENT_DECISION,
        'repair_source_commit': args.repair_source_commit,
        'seeds': list(SEEDS),
        'target_steps': ENDPOINTS[-1],
        'external_evaluation_protocol': {
            'mode': 'fixed-random-flow',
            'episodes_per_checkpoint_per_seed': 20,
            'milestones': list(ENDPOINTS),
        },
        'movement_construct_revision': {
            'old_metric_retained_as_diagnostic': 'mean_start_end_displacement',
            'gate_metric': 'mean_step_displacement',
            'reason': (
                'start-to-end distance confounds locomotion with successful multi-goal return paths'
            ),
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
            'pooled_step_displacement': 'does not decline from 0 to 50k',
            'reward_td': 'no seed triggers the unchanged monotonic-divergence rule',
            'arm_pass': 'all five criteria must pass',
        },
        'repair_arm': repair,
        'backbone_repair_decision': decision,
        'final_p4_verdict': None,
        'formal_four_environment_experiment_launched': False,
        'continuation_beyond_50k_launched': False,
    }
    atomic_json(args.output, payload)
    print({'backbone_repair_decision': decision, 'output': str(args.output)})


if __name__ == '__main__':
    main()

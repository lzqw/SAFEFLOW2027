#!/usr/bin/env python3
"""Apply the preregistered P4.2 B0-R2 three-seed reward-backbone gate."""

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
    read_csv,
    read_json,
    reward_td_trend,
)
from scripts.star.summarize_p4_2_backbone_repair import EXTERNAL_METRICS, external_rows
from scripts.star.summarize_p4_2_backbone_repair_r2_pilot import (
    EXPECTED_REPAIR,
    R2_SOURCE_COMMIT,
    validate_r2_run,
)


PILOT_PROMOTION_DECISION = 'launch_remaining_r2_repair_seeds'


def validate_r2_pilot_parent(path: Path) -> dict[str, Any]:
    """Require the immutable passing R2 pilot before reading the other seeds."""
    payload = read_json(path)
    if payload.get('stage') != 'P4.2-B0-R2-seed-2028-feasibility':
        raise ValueError('three-seed R2 parent is not the registered pilot')
    if payload.get('repair_source_commit') != R2_SOURCE_COMMIT:
        raise ValueError('three-seed R2 parent has an unexpected repair source')
    if payload.get('pilot', {}).get('pilot_decision') != PILOT_PROMOTION_DECISION:
        raise ValueError('remaining R2 seeds were not authorized by the pilot')
    if payload.get('final_p4_verdict') is not None:
        raise ValueError('R2 pilot parent must keep final_p4_verdict null')
    return payload


def repair_summary(
    run_dirs: list[Path],
    receipts: list[Path],
    evaluation_csv: Path,
) -> dict[str, Any]:
    """Validate all R2 runs and compute the frozen three-seed gate."""
    if len(run_dirs) != 3 or len(receipts) != 3:
        raise ValueError('B0-R2 requires exactly three run directories and receipts')
    evaluations = external_rows(evaluation_csv)
    receipt_payloads = [read_json(path) for path in receipts]
    receipt_by_seed = {int(payload['seed']): payload for payload in receipt_payloads}
    if set(receipt_by_seed) != set(SEEDS):
        raise ValueError(f'B0-R2 receipts must use seeds {SEEDS}')

    trends: dict[str, dict[str, Any]] = {}
    target_sampling_std: dict[str, dict[str, float]] = {}
    checkpoint_bytes: dict[str, dict[str, int]] = {}
    actor_updates: dict[str, int] = {}
    reward_critic_updates: dict[str, int] = {}
    observed_seeds: set[int] = set()
    training_seconds = 0.0
    estimated_cost_cny = 0.0
    for run_dir in run_dirs:
        summary = read_json(run_dir / 'artifacts/training_summary.json')
        manifest = read_json(run_dir / 'artifacts/run_manifest.json')
        seed = int(summary['seed'])
        if seed in observed_seeds:
            raise ValueError(f'B0-R2 has duplicate run seed {seed}')
        observed_seeds.add(seed)
        receipt = receipt_by_seed.get(seed)
        if receipt is None or summary['status'] != 'completed' or receipt['status'] != 'passed':
            raise ValueError(f'B0-R2 seed {seed} did not complete')
        validate_r2_run(summary, manifest, seed)
        if int(receipt['steps']) != 50_000 or int(receipt['updates']) != 48_000:
            raise ValueError(f'B0-R2 seed {seed} receipt has incorrect counters')

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
            raise FileNotFoundError(f'B0-R2 seed {seed} missing checkpoints: {missing}')
        checkpoint_bytes[str(seed)] = {label: path.stat().st_size for label, path in paths.items()}
        diagnostics = read_csv(run_dir / 'artifacts/diagnostics.csv')
        trends[str(seed)] = reward_td_trend(diagnostics, seed)
        sampling_values = [
            number(row['Critic/RewardTargetSamplingStd'], 'Critic/RewardTargetSamplingStd')
            for row in diagnostics
        ]
        target_sampling_std[str(seed)] = {
            'mean_over_5k_rows': statistics.fmean(sampling_values),
            'final': sampling_values[-1],
        }
        actor_updates[str(seed)] = int(summary['actor_updates'])
        reward_critic_updates[str(seed)] = int(summary['reward_critic_updates'])
        training_seconds += number(summary['training_seconds'], 'training_seconds')
        estimated_cost_cny += number(receipt['estimated_cost_cny'], 'estimated_cost_cny')
    if observed_seeds != set(SEEDS):
        raise ValueError(f'B0-R2 runs must use seeds {SEEDS}')

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
        'name': 'B0-R2',
        'repair': EXPECTED_REPAIR,
        'pooled_external_evaluation': pooled,
        'external_evaluation_by_seed': {
            str(seed): {str(step): evaluations[seed][step] for step in ENDPOINTS} for seed in SEEDS
        },
        'reward_td_trend_by_seed': trends,
        'reward_target_sampling_std_by_seed': target_sampling_std,
        'criteria': criteria,
        'functional_gate_passed': all(criteria.values()),
        'actor_updates_by_seed': actor_updates,
        'reward_critic_updates_by_seed': reward_critic_updates,
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
    result.add_argument('--r2-pilot-decision', type=Path, required=True)
    result.add_argument('--repair-source-commit', required=True)
    result.add_argument('--output', type=Path, required=True)
    return result


def main() -> None:
    """Write the strict machine-readable R2 repair decision."""
    args = parser().parse_args()
    validate_r2_pilot_parent(args.r2_pilot_decision)
    if args.repair_source_commit != R2_SOURCE_COMMIT:
        raise ValueError('three-seed source is not the frozen B0-R2 repair commit')
    repair = repair_summary(args.run_dirs, args.receipts, args.evaluation_csv)
    decision = (
        'proceed_to_bellman_probability_critic'
        if repair['functional_gate_passed']
        else 'continue_backbone_repair_and_pause_safety'
    )
    payload = {
        'schema_version': 1,
        'stage': 'P4.2-B0-R2-reward-backbone-repair',
        'repair_source_commit': R2_SOURCE_COMMIT,
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

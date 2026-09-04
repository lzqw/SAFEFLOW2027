#!/usr/bin/env python3
"""Apply the frozen P4.2 B0-R1 seed-2028 feasibility hard stops."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts.star.summarize_p4_2_backbone import (
    ENDPOINTS,
    atomic_json,
    number,
    read_csv,
    read_json,
    reward_td_trend,
)
from scripts.star.summarize_p4_2_backbone_repair import validate_parent


PILOT_SEED = 2028
REPAIR_SOURCE_COMMIT = '2c03c7ad07b36eaebab71affa8624080b6e4406f'


def hard_stop_decision(hard_stops: dict[str, bool]) -> str:
    """Return the frozen pilot branch from explicit hard-stop flags."""
    return (
        'continue_backbone_repair_and_pause_safety'
        if any(hard_stops.values())
        else 'launch_remaining_repair_seeds'
    )


def pilot_external_rows(path: Path) -> dict[int, dict[str, float]]:
    """Read exactly the primary seed-2028 0/50k evaluation pair."""
    grouped: dict[int, dict[str, float]] = {}
    for row in read_csv(path):
        seed = int(number(row['training_seed'], 'training_seed'))
        step = int(number(row['milestone'], 'milestone'))
        if seed != PILOT_SEED:
            raise ValueError(f'pilot evaluation must contain only seed {PILOT_SEED}')
        if step in grouped:
            raise ValueError(f'duplicate pilot endpoint {step}')
        if row['evaluation_mode'] != 'fixed-random-flow' or row['evaluation_role'] != 'primary':
            raise ValueError('pilot requires primary fixed-random-flow evaluation')
        if int(number(row['episodes'], 'episodes')) != 20:
            raise ValueError('pilot requires 20 episodes per checkpoint')
        grouped[step] = {
            'mean_return': number(row['mean_return'], 'mean_return'),
            'goal_met_episode_rate': number(
                row['goal_met_episode_rate'],
                'goal_met_episode_rate',
            ),
            'stall_ratio': number(row['stall_ratio'], 'stall_ratio'),
            'mean_step_displacement': number(
                row['mean_step_displacement'],
                'mean_step_displacement',
            ),
            'mean_start_end_displacement': number(
                row['mean_start_end_displacement'],
                'mean_start_end_displacement',
            ),
            'mean_action_norm': number(row['mean_action_norm'], 'mean_action_norm'),
        }
    if set(grouped) != set(ENDPOINTS):
        raise ValueError('pilot evaluation must contain endpoints 0 and 50000')
    return grouped


def pilot_summary(
    run_dir: Path,
    receipt_path: Path,
    evaluation_csv: Path,
) -> dict[str, Any]:
    """Validate the run and return its immutable feasibility decision."""
    summary = read_json(run_dir / 'artifacts/training_summary.json')
    manifest = read_json(run_dir / 'artifacts/run_manifest.json')
    receipt = read_json(receipt_path)
    if int(summary['seed']) != PILOT_SEED or int(receipt['seed']) != PILOT_SEED:
        raise ValueError(f'pilot must use seed {PILOT_SEED}')
    if summary['status'] != 'completed' or receipt['status'] != 'passed':
        raise ValueError('pilot training did not complete')
    if int(summary['env_steps']) != 50_000 or int(summary['updates']) != 48_000:
        raise ValueError('pilot summary has incorrect counters')
    if int(receipt['steps']) != 50_000 or int(receipt['updates']) != 48_000:
        raise ValueError('pilot receipt has incorrect counters')
    expected_ablation = {
        'reward_flow_enabled': True,
        'global_safety_constraint_enabled': False,
        'local_safety_geometry_enabled': False,
    }
    if manifest['p4_2_backbone_ablation'] != expected_ablation:
        raise ValueError('pilot did not isolate the reward/flow backbone')
    expected_repair = {
        'target_flow_actor_for_reward_backup': True,
        'actor_update_interval': 2,
    }
    if manifest['p4_2_reward_backbone_repair'] != expected_repair:
        raise ValueError('pilot did not use the frozen R1 repair')
    if summary['reward_target_actor_enabled'] is not True:
        raise ValueError('pilot target flow was not active')
    if int(summary['actor_update_interval']) != 2:
        raise ValueError('pilot actor update interval was not two')
    if summary['global_safety_constraint_enabled'] is not False:
        raise ValueError('pilot enabled global safety')
    if summary['local_safety_geometry_enabled'] is not False:
        raise ValueError('pilot enabled local geometry')
    if summary['training_geometry_enabled'] is not False:
        raise ValueError('pilot activated training geometry')
    checkpoints = {
        label: run_dir / 'checkpoints' / filename
        for label, filename in {
            'milestone_0': 'milestone_0.pt',
            'milestone_50000': 'milestone_50000.pt',
            'latest_full': 'latest_full.pt',
            'final_full': 'final_full.pt',
        }.items()
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'pilot is missing checkpoints: {missing}')

    external = pilot_external_rows(evaluation_csv)
    trend = reward_td_trend(read_csv(run_dir / 'artifacts/diagnostics.csv'), PILOT_SEED)
    endpoint = external[ENDPOINTS[-1]]
    hard_stops = {
        'non_finite_artifact': False,
        'zero_endpoint_success': endpoint['goal_met_episode_rate'] <= 0.0,
        'endpoint_stall_at_least_0_5': endpoint['stall_ratio'] >= 0.5,
        'reward_td_monotonic_divergence': trend['monotonic_divergence'],
    }
    decision = hard_stop_decision(hard_stops)
    passed = decision == 'launch_remaining_repair_seeds'
    return {
        'seed': PILOT_SEED,
        'endpoints': {str(step): external[step] for step in ENDPOINTS},
        'reward_td_trend': trend,
        'hard_stops': hard_stops,
        'feasibility_passed': passed,
        'pilot_decision': decision,
        'actor_updates': int(summary['actor_updates']),
        'checkpoint_bytes': {label: path.stat().st_size for label, path in checkpoints.items()},
        'training_seconds': number(summary['training_seconds'], 'training_seconds'),
        'estimated_cost_cny': number(receipt['estimated_cost_cny'], 'estimated_cost_cny'),
    }


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--run-dir', type=Path, required=True)
    result.add_argument('--receipt', type=Path, required=True)
    result.add_argument('--evaluation-csv', type=Path, required=True)
    result.add_argument('--parent-decision', type=Path, required=True)
    result.add_argument('--repair-source-commit', required=True)
    result.add_argument('--output', type=Path, required=True)
    return result


def main() -> None:
    """Write the strict seed-2028 pilot decision."""
    args = parser().parse_args()
    validate_parent(args.parent_decision)
    if args.repair_source_commit != REPAIR_SOURCE_COMMIT:
        raise ValueError('pilot source is not the frozen B0-R1 repair commit')
    pilot = pilot_summary(args.run_dir, args.receipt, args.evaluation_csv)
    payload = {
        'schema_version': 1,
        'stage': 'P4.2-B0-R1-seed-2028-feasibility',
        'repair_source_commit': REPAIR_SOURCE_COMMIT,
        'pilot': pilot,
        'final_p4_verdict': None,
        'formal_four_environment_experiment_launched': False,
        'continuation_beyond_50k_launched': False,
    }
    atomic_json(args.output, payload)
    print(pilot['pilot_decision'])


if __name__ == '__main__':
    main()

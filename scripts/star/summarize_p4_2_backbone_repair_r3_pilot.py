#!/usr/bin/env python3
"""Apply the frozen P4.2 B0-R3 seed-2028 feasibility hard stops."""

from __future__ import annotations

import argparse
import statistics
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
from scripts.star.summarize_p4_2_backbone_repair_pilot import pilot_external_rows
from scripts.star.summarize_p4_2_backbone_repair_r2_pilot import EXPECTED_REPAIR


PILOT_SEED = 2028
R2_SOURCE_COMMIT = '2332a550cc0a366c436c9a116b72b7bfc918bef2'
R3_SOURCE_COMMIT = 'b3c490309d465b89a5ae5a6ed235835587074abb'
R2_PARENT_DECISION = 'continue_backbone_repair_and_pause_safety'
EXPECTED_TIMESCALE = {
    'reward_target_polyak': 0.001,
    'cost_target_polyak': 0.005,
    'actor_target_polyak': 0.005,
}


def validate_r2_parent(path: Path) -> dict[str, Any]:
    """Require the immutable failed R2 pilot that authorized R3."""
    payload = read_json(path)
    if payload.get('training_source_commit') != R2_SOURCE_COMMIT:
        raise ValueError('R3 parent has an unexpected R2 training source')
    pilot = payload.get('pilot', {})
    if pilot.get('pilot_decision') != R2_PARENT_DECISION:
        raise ValueError('R3 parent did not keep safety paused for backbone repair')
    if pilot.get('hard_stops', {}).get('reward_td_monotonic_divergence') is not True:
        raise ValueError('R3 parent did not record the registered R2 TD hard stop')
    candidate = payload.get('r3_candidate', {})
    if candidate.get('reward_target_polyak') != EXPECTED_TIMESCALE['reward_target_polyak']:
        raise ValueError('R3 parent did not select the frozen reward-target timescale')
    if payload.get('final_p4_verdict') is not None:
        raise ValueError('R3 parent must keep final_p4_verdict null')
    return payload


def hard_stop_decision(hard_stops: dict[str, bool]) -> str:
    """Return the frozen R3 pilot branch from explicit hard-stop flags."""
    return (
        'continue_backbone_repair_and_pause_safety'
        if any(hard_stops.values())
        else 'launch_remaining_r3_repair_seeds'
    )


def validate_r3_run(summary: dict[str, Any], manifest: dict[str, Any], seed: int) -> None:
    """Validate isolation, counters, R2 tracking, and the R3 target timescale."""
    if int(summary['env_steps']) != 50_000 or int(summary['updates']) != 48_000:
        raise ValueError(f'B0-R3 seed {seed} has incorrect outer counters')
    if int(summary['reward_critic_updates']) != 96_000:
        raise ValueError(f'B0-R3 seed {seed} has incorrect reward-critic counters')
    if int(summary['actor_updates']) != 21_000:
        raise ValueError(f'B0-R3 seed {seed} has incorrect actor-update counters')
    expected_ablation = {
        'reward_flow_enabled': True,
        'global_safety_constraint_enabled': False,
        'local_safety_geometry_enabled': False,
    }
    if manifest['p4_2_backbone_ablation'] != expected_ablation:
        raise ValueError(f'B0-R3 seed {seed} did not isolate the reward/flow backbone')
    if manifest['p4_2_reward_backbone_repair'] != EXPECTED_REPAIR:
        raise ValueError(f'B0-R3 seed {seed} did not retain the frozen R2 tracking repair')
    if manifest.get('p4_2_reward_target_timescale') != EXPECTED_TIMESCALE:
        raise ValueError(f'B0-R3 seed {seed} did not use the frozen target timescale')
    if tuple(int(value) for value in manifest['eval_milestones']) != ENDPOINTS:
        raise ValueError(f'B0-R3 seed {seed} has incorrect milestones')
    expected_summary = {
        'reward_target_actor_enabled': True,
        'actor_update_interval': 2,
        'reward_target_samples': 4,
        'reward_critic_updates_per_update': 2,
        'separate_reward_replay': True,
        'reward_replay_recent_fraction': 0.5,
        'reward_replay_recent_window': 10_000,
        'reward_target_polyak': 0.001,
        'cost_target_polyak': 0.005,
        'actor_target_polyak': 0.005,
        'global_safety_constraint_enabled': False,
        'local_safety_geometry_enabled': False,
        'training_geometry_enabled': False,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f'B0-R3 seed {seed} summary has invalid {key}')


def pilot_summary(
    run_dir: Path,
    receipt_path: Path,
    evaluation_csv: Path,
) -> dict[str, Any]:
    """Validate the run and return its immutable R3 feasibility decision."""
    summary = read_json(run_dir / 'artifacts/training_summary.json')
    manifest = read_json(run_dir / 'artifacts/run_manifest.json')
    receipt = read_json(receipt_path)
    if int(summary['seed']) != PILOT_SEED or int(receipt['seed']) != PILOT_SEED:
        raise ValueError(f'B0-R3 pilot must use seed {PILOT_SEED}')
    if summary['status'] != 'completed' or receipt['status'] != 'passed':
        raise ValueError('B0-R3 pilot training did not complete')
    validate_r3_run(summary, manifest, PILOT_SEED)
    if int(receipt['steps']) != 50_000 or int(receipt['updates']) != 48_000:
        raise ValueError('B0-R3 pilot receipt has incorrect counters')

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
        raise FileNotFoundError(f'B0-R3 pilot is missing checkpoints: {missing}')

    external = pilot_external_rows(evaluation_csv)
    diagnostics = read_csv(run_dir / 'artifacts/diagnostics.csv')
    trend = reward_td_trend(diagnostics, PILOT_SEED)
    target_sampling_std = [
        number(row['Critic/RewardTargetSamplingStd'], 'Critic/RewardTargetSamplingStd')
        for row in diagnostics
    ]
    endpoint = external[ENDPOINTS[-1]]
    hard_stops = {
        'non_finite_artifact': False,
        'zero_endpoint_success': endpoint['goal_met_episode_rate'] <= 0.0,
        'endpoint_stall_at_least_0_5': endpoint['stall_ratio'] >= 0.5,
        'reward_td_monotonic_divergence': trend['monotonic_divergence'],
    }
    decision = hard_stop_decision(hard_stops)
    passed = decision == 'launch_remaining_r3_repair_seeds'
    return {
        'seed': PILOT_SEED,
        'endpoints': {str(step): external[step] for step in ENDPOINTS},
        'reward_td_trend': trend,
        'reward_target_sampling_std': {
            'mean_over_5k_rows': statistics.fmean(target_sampling_std),
            'final': target_sampling_std[-1],
        },
        'target_timescale': EXPECTED_TIMESCALE,
        'hard_stops': hard_stops,
        'feasibility_passed': passed,
        'pilot_decision': decision,
        'outer_updates': int(summary['updates']),
        'reward_critic_updates': int(summary['reward_critic_updates']),
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
    result.add_argument('--r2-pilot-decision', type=Path, required=True)
    result.add_argument('--repair-source-commit', required=True)
    result.add_argument('--output', type=Path, required=True)
    return result


def main() -> None:
    """Write the strict seed-2028 R3 pilot decision."""
    args = parser().parse_args()
    validate_r2_parent(args.r2_pilot_decision)
    if args.repair_source_commit != R3_SOURCE_COMMIT:
        raise ValueError('pilot source is not the frozen B0-R3 repair commit')
    pilot = pilot_summary(args.run_dir, args.receipt, args.evaluation_csv)
    payload = {
        'schema_version': 1,
        'stage': 'P4.2-B0-R3-seed-2028-feasibility',
        'r2_training_source_commit': R2_SOURCE_COMMIT,
        'repair_source_commit': R3_SOURCE_COMMIT,
        'repair': EXPECTED_REPAIR,
        'target_timescale': EXPECTED_TIMESCALE,
        'pilot': pilot,
        'final_p4_verdict': None,
        'formal_four_environment_experiment_launched': False,
        'continuation_beyond_50k_launched': False,
    }
    atomic_json(args.output, payload)
    print(pilot['pilot_decision'])


if __name__ == '__main__':
    main()

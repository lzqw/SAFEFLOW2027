"""Tests for the preregistered P4.2 B0-R2 pilot and three-seed gates."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.star.summarize_p4_2_backbone_repair_r2 import (
    repair_summary,
    validate_r2_pilot_parent,
)
from scripts.star.summarize_p4_2_backbone_repair_r2_pilot import (
    R1_SOURCE_COMMIT,
    pilot_summary,
    validate_r1_parent,
)


STABLE_TD = [0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.04, 0.03, 0.04, 0.03]
DIVERGENT_TD = [0.08, 0.07, 0.06, 0.05, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]


def write_json(path: Path, payload: dict) -> None:
    """Write one fixture object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def make_run(tmp_path: Path, seed: int, td_values: list[float]) -> tuple[Path, Path]:
    """Create one structurally complete synthetic B0-R2 run and receipt."""
    run_dir = tmp_path / f'run-{seed}'
    write_json(
        run_dir / 'artifacts/training_summary.json',
        {
            'status': 'completed',
            'seed': seed,
            'env_steps': 50_000,
            'updates': 48_000,
            'reward_critic_updates': 96_000,
            'actor_updates': 21_000,
            'training_seconds': 10.0,
            'global_safety_constraint_enabled': False,
            'local_safety_geometry_enabled': False,
            'training_geometry_enabled': False,
            'reward_target_actor_enabled': True,
            'actor_update_interval': 2,
            'reward_target_samples': 4,
            'reward_critic_updates_per_update': 2,
            'separate_reward_replay': True,
            'reward_replay_recent_fraction': 0.5,
            'reward_replay_recent_window': 10_000,
        },
    )
    write_json(
        run_dir / 'artifacts/run_manifest.json',
        {
            'eval_milestones': [0, 50_000],
            'p4_2_backbone_ablation': {
                'reward_flow_enabled': True,
                'global_safety_constraint_enabled': False,
                'local_safety_geometry_enabled': False,
            },
            'p4_2_reward_backbone_repair': {
                'target_flow_actor_for_reward_backup': True,
                'actor_update_interval': 2,
                'reward_target_samples': 4,
                'reward_critic_updates_per_update': 2,
                'separate_reward_replay': True,
                'reward_replay_recent_fraction': 0.5,
                'reward_replay_recent_window': 10_000,
            },
        },
    )
    diagnostics = run_dir / 'artifacts/diagnostics.csv'
    diagnostics.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                'TotalEnvSteps',
                'Critic/RewardTD',
                'Critic/RewardTargetSamplingStd',
            ],
        )
        writer.writeheader()
        for index, value in enumerate(td_values, start=1):
            writer.writerow(
                {
                    'TotalEnvSteps': 5000 * index,
                    'Critic/RewardTD': value,
                    'Critic/RewardTargetSamplingStd': 0.005,
                },
            )
    for name in ('milestone_0.pt', 'milestone_50000.pt', 'latest_full.pt', 'final_full.pt'):
        checkpoint = run_dir / 'checkpoints' / name
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b'checkpoint')
    receipt = tmp_path / f'receipt-{seed}.json'
    write_json(
        receipt,
        {
            'status': 'passed',
            'seed': seed,
            'steps': 50_000,
            'updates': 48_000,
            'estimated_cost_cny': 0.1,
        },
    )
    return run_dir, receipt


def write_evaluation(
    path: Path,
    seeds: tuple[int, ...],
    *,
    final_displacement: float = 0.02,
    final_success: float = 0.2,
    final_stall: float = 0.1,
) -> None:
    """Write fixed-random-flow endpoint rows for the requested seeds."""
    fields = [
        'training_seed',
        'milestone',
        'evaluation_mode',
        'evaluation_role',
        'episodes',
        'mean_return',
        'goal_met_episode_rate',
        'stall_ratio',
        'mean_step_displacement',
        'mean_start_end_displacement',
        'mean_action_norm',
        'mean_cost',
        'goal_met_total',
        'brier_skill',
        'auroc',
    ]
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for seed in seeds:
            for step, return_value, success, displacement, stall in (
                (0, 0.0, 0.1, 0.01, 0.1),
                (50_000, 1.0, final_success, final_displacement, final_stall),
            ):
                writer.writerow(
                    {
                        'training_seed': seed,
                        'milestone': step,
                        'evaluation_mode': 'fixed-random-flow',
                        'evaluation_role': 'primary',
                        'episodes': 20,
                        'mean_return': return_value,
                        'goal_met_episode_rate': success,
                        'stall_ratio': stall,
                        'mean_step_displacement': displacement,
                        'mean_start_end_displacement': 1.0,
                        'mean_action_norm': 0.5,
                        'mean_cost': 5.0,
                        'goal_met_total': int(20 * success),
                        'brier_skill': '',
                        'auroc': '',
                    },
                )


def r1_parent(path: Path) -> None:
    """Write the failed immutable R1 parent fixture."""
    write_json(
        path,
        {
            'training_source_commit': R1_SOURCE_COMMIT,
            'pilot': {
                'pilot_decision': 'continue_backbone_repair_and_pause_safety',
                'hard_stops': {'reward_td_monotonic_divergence': True},
            },
            'final_p4_verdict': None,
        },
    )


def test_r2_pilot_passes_stable_tracking_and_authorizes_remaining_seeds(tmp_path: Path) -> None:
    run, receipt = make_run(tmp_path, 2028, STABLE_TD)
    evaluation = tmp_path / 'pilot-evaluation.csv'
    write_evaluation(evaluation, (2028,))
    result = pilot_summary(run, receipt, evaluation)
    assert result['feasibility_passed'] is True
    assert result['pilot_decision'] == 'launch_remaining_r2_repair_seeds'
    assert result['reward_critic_updates'] == 96_000
    assert result['reward_target_sampling_std']['final'] == pytest.approx(0.005)


def test_r2_pilot_keeps_safety_paused_on_unchanged_td_hard_stop(tmp_path: Path) -> None:
    run, receipt = make_run(tmp_path, 2028, DIVERGENT_TD)
    evaluation = tmp_path / 'pilot-evaluation.csv'
    write_evaluation(evaluation, (2028,))
    result = pilot_summary(run, receipt, evaluation)
    assert result['feasibility_passed'] is False
    assert result['hard_stops']['reward_td_monotonic_divergence'] is True
    assert result['pilot_decision'] == 'continue_backbone_repair_and_pause_safety'


def test_r2_three_seed_gate_uses_unchanged_functional_criteria(tmp_path: Path) -> None:
    pairs = [make_run(tmp_path, seed, STABLE_TD) for seed in (2027, 2028, 2029)]
    evaluation = tmp_path / 'evaluation.csv'
    write_evaluation(evaluation, (2027, 2028, 2029))
    result = repair_summary(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        evaluation,
    )
    assert result['functional_gate_passed'] is True
    assert result['reward_critic_updates_by_seed'] == {
        '2027': 96_000,
        '2028': 96_000,
        '2029': 96_000,
    }


def test_r2_three_seed_gate_rejects_declining_step_displacement(tmp_path: Path) -> None:
    pairs = [make_run(tmp_path, seed, STABLE_TD) for seed in (2027, 2028, 2029)]
    evaluation = tmp_path / 'evaluation.csv'
    write_evaluation(evaluation, (2027, 2028, 2029), final_displacement=0.005)
    result = repair_summary(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        evaluation,
    )
    assert result['functional_gate_passed'] is False
    assert result['criteria']['step_displacement_does_not_decline'] is False


def test_r2_parent_validators_require_failed_r1_then_passing_r2_pilot(tmp_path: Path) -> None:
    parent = tmp_path / 'r1.json'
    r1_parent(parent)
    assert validate_r1_parent(parent)['final_p4_verdict'] is None
    r2_pilot = tmp_path / 'r2-pilot.json'
    write_json(
        r2_pilot,
        {
            'stage': 'P4.2-B0-R2-seed-2028-feasibility',
            'repair_source_commit': '2332a550cc0a366c436c9a116b72b7bfc918bef2',
            'pilot': {'pilot_decision': 'launch_remaining_r2_repair_seeds'},
            'final_p4_verdict': None,
        },
    )
    assert validate_r2_pilot_parent(r2_pilot)['final_p4_verdict'] is None

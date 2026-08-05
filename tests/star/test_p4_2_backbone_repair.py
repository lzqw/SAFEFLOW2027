"""Tests for the preregistered P4.2 B0-R1 repair decision."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.star.summarize_p4_2_backbone_repair import repair_summary, validate_parent


def write_json(path: Path, payload: dict) -> None:
    """Write one fixture object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def make_repair(tmp_path: Path, *, final_step_displacement: float) -> tuple[list[Path], list[Path]]:
    """Create a complete synthetic three-seed B0-R1 repair."""
    run_dirs, receipts = [], []
    for seed in (2027, 2028, 2029):
        run_dir = tmp_path / f'run-{seed}'
        run_dirs.append(run_dir)
        write_json(
            run_dir / 'artifacts/training_summary.json',
            {
                'status': 'completed',
                'seed': seed,
                'env_steps': 50_000,
                'updates': 48_000,
                'actor_updates': 21_000,
                'training_seconds': 10.0,
                'global_safety_constraint_enabled': False,
                'local_safety_geometry_enabled': False,
                'training_geometry_enabled': False,
                'reward_target_actor_enabled': True,
                'actor_update_interval': 2,
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
                },
            },
        )
        diagnostics = run_dir / 'artifacts/diagnostics.csv'
        diagnostics.parent.mkdir(parents=True, exist_ok=True)
        with diagnostics.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['TotalEnvSteps', 'Critic/RewardTD'])
            writer.writeheader()
            values = [0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.04, 0.03, 0.04, 0.03]
            for index, value in enumerate(values, start=1):
                writer.writerow({'TotalEnvSteps': 5000 * index, 'Critic/RewardTD': value})
        for name in ('milestone_0.pt', 'milestone_50000.pt', 'latest_full.pt', 'final_full.pt'):
            checkpoint = run_dir / 'checkpoints' / name
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b'checkpoint')
        receipt = tmp_path / f'receipt-{seed}.json'
        receipts.append(receipt)
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
    with (tmp_path / 'evaluation.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for seed in (2027, 2028, 2029):
            for step, return_value, success, displacement in (
                (0, 0.0, 0.1, 0.01),
                (50_000, 1.0, 0.2, final_step_displacement),
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
                        'stall_ratio': 0.1,
                        'mean_step_displacement': displacement,
                        'mean_start_end_displacement': 1.0,
                        'mean_action_norm': 0.5,
                        'mean_cost': 5.0,
                        'goal_met_total': int(20 * success),
                        'brier_skill': '',
                        'auroc': '',
                    },
                )
    return run_dirs, receipts


def test_repair_gate_uses_step_displacement_and_passes_stable_backbone(tmp_path: Path) -> None:
    runs, receipts = make_repair(tmp_path, final_step_displacement=0.02)
    result = repair_summary(runs, receipts, tmp_path / 'evaluation.csv')
    assert result['functional_gate_passed'] is True
    assert result['criteria']['step_displacement_does_not_decline'] is True


def test_repair_gate_rejects_declining_step_displacement(tmp_path: Path) -> None:
    runs, receipts = make_repair(tmp_path, final_step_displacement=0.005)
    result = repair_summary(runs, receipts, tmp_path / 'evaluation.csv')
    assert result['functional_gate_passed'] is False
    assert result['criteria']['step_displacement_does_not_decline'] is False


def test_parent_decision_must_keep_safety_paused(tmp_path: Path) -> None:
    parent = tmp_path / 'parent.json'
    write_json(
        parent,
        {
            'backbone_decision': 'repair_backbone_and_pause_safety',
            'training_source_commit': 'f4371320a1683bc2130cd687213ffbac2d7da9d9',
            'final_p4_verdict': None,
        },
    )
    assert validate_parent(parent)['final_p4_verdict'] is None

# Copyright 2026 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the preregistered P4.2 B1--B4 mechanism-screen decision."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from scripts.star.summarize_p4_2_mechanism_screen import ARMS, summarize_screen


SOURCE = '5e994d7d7a89b4dda8ee94d22a1e503f83b409de'


def write_json(path: Path, payload: dict) -> None:
    """Write one fixture object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def fixture_screen(
    tmp_path: Path,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path], Path]:
    """Create a complete synthetic four-arm causal screen."""
    mechanisms = {
        'B1': (False, False, False),
        'B2': (True, False, False),
        'B3': (True, True, False),
        'B4': (True, True, True),
    }
    endpoint_metrics = {
        'B1': (10.0, 5.0, 0.70, 0.10),
        'B2': (8.0, 4.0, 0.65, 0.10),
        'B3': (7.0, 3.0, 0.50, 0.10),
        'B4': (9.0, 4.5, 0.70, 0.05),
    }
    preregistration = {
        'protocol_status': 'locked_before_training',
        'protocol_version': 'p4.2-b1-b4-v1',
        'screen': {'seed': 2028, 'steps': 50_000},
        'heldout': {'episode_period': 5},
        'external_evaluation': {'seeds': list(range(12_027, 12_047))},
        'selection': {'extend_steps': 100_000, 'extend_seeds': [2027, 2028, 2029]},
        'arms': {},
    }
    run_dirs: dict[str, Path] = {}
    receipts: dict[str, Path] = {}
    evaluations: dict[str, Path] = {}
    for arm in ARMS:
        bellman, geometry, probing = mechanisms[arm]
        config = f'configs/star/{arm.lower()}.yaml'
        fingerprint = f'hash-{arm}'
        preregistration['arms'][arm] = {
            'global_safety': True,
            'bellman_probability': bellman,
            'dynamic_geometry': geometry,
            'candidate_probing': probing,
            'config': config,
            'config_sha256': fingerprint,
        }
        run_dir = tmp_path / arm / 'run'
        run_dirs[arm] = run_dir
        heldout = {
            'Heldout/BrierSkill': 0.1,
            'Heldout/AUROC': 0.7,
            'Heldout/GradientNorm': 0.01,
            'Heldout/EnsembleDisagreement': 0.02,
            'Heldout/BoundarySamples': 200,
        }
        write_json(
            run_dir / 'artifacts/training_summary.json',
            {
                'status': 'completed',
                'seed': 2028,
                'env_steps': 50_000,
                'updates': 48_000,
                'training_seconds': 10.0,
                'training_geometry_enabled': geometry,
                'geometry_activation_step': 45_000 if geometry else None,
                'heldout_probability_latest': heldout if bellman else {},
                'replay_split_counts': {'heldout_transitions': 10_000},
            },
        )
        write_json(
            run_dir / 'artifacts/run_manifest.json',
            {
                'eval_milestones': [0, 50_000],
                'p4_2_backbone_ablation': {
                    'global_safety_constraint_enabled': True,
                    'local_safety_geometry_enabled': geometry,
                    'reward_flow_enabled': True,
                },
                'p4_2_probability_critic': {
                    'bellman_probability_enabled': bellman,
                    'episode_split': {
                        'heldout_period': 5,
                        'optimizer_access_to_heldout': False,
                        'global_dual_access_to_heldout': False,
                        'observation_normalizer_access_to_heldout': False,
                    },
                },
                'p4_2_dynamic_geometry': {
                    'enabled': geometry,
                    'enable_after_consecutive_passes': 3,
                    'disable_after_consecutive_failures': 2,
                    'gate_evidence_split': 'heldout_episode',
                },
                'p4_2_candidate_probing': {
                    'enabled': probing,
                    'candidate_count': 16,
                    'risk_selection': 'closest_trusted_safe_boundary_then_reward_q',
                    'tangent_requires_geometry_enabled': True,
                },
            },
        )
        diagnostic_path = run_dir / 'artifacts/diagnostics.csv'
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            'Loss/CostCritic',
            'Loss/RewardCritic',
            'Loss/Actor',
            'Critic/CostTD',
            'Critic/RewardTD',
            'Dual/GlobalSafety',
            'TotalEnvSteps',
            'TotalUpdates',
            'TotalRewardCriticUpdates',
            'Probing/CandidateRiskP10',
            'Probing/CandidateRiskP50',
            'Probing/CandidateRiskP90',
            'Probing/SelectedRisk',
            'Probing/SelectedDisagreement',
            'Probing/ActualCost',
            'Probing/ActualViolation',
            'Probing/EffectiveBoundarySamples',
            'Probing/TangentRatio',
            'Probing/DangerousNormalVelocity',
        ]
        with diagnostic_path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, step in enumerate(range(5_000, 50_001, 5_000), start=1):
                writer.writerow(
                    {
                        'Loss/CostCritic': 0.1,
                        'Loss/RewardCritic': 0.1,
                        'Loss/Actor': 0.1,
                        'Critic/CostTD': 0.1,
                        'Critic/RewardTD': 0.1 + 0.001 * (index % 2),
                        'Dual/GlobalSafety': 0.1,
                        'TotalEnvSteps': step,
                        'TotalUpdates': max(0, step - 2_000),
                        'TotalRewardCriticUpdates': max(0, 2 * (step - 2_000)),
                        'Probing/CandidateRiskP10': 0.1,
                        'Probing/CandidateRiskP50': 0.2,
                        'Probing/CandidateRiskP90': 0.3,
                        'Probing/SelectedRisk': 0.2,
                        'Probing/SelectedDisagreement': 0.01,
                        'Probing/ActualCost': 0.0,
                        'Probing/ActualViolation': 0.0,
                        'Probing/EffectiveBoundarySamples': 10.0,
                        'Probing/TangentRatio': 0.0,
                        'Probing/DangerousNormalVelocity': 0.0,
                    },
                )
        if bellman:
            heldout_path = run_dir / 'artifacts/heldout_probability.csv'
            heldout_fields = [
                'env_steps',
                'split',
                'recent_episode_limit',
                'transition_limit',
                'gate_passed',
                'curriculum_stage',
                'geometry_enabled',
                'BrierScore',
                'BrierSkill',
                'AUROC',
                'BellmanError',
                'GradientNorm',
                'EnsembleDisagreement',
                'BoundarySamples',
                'PositiveLabels',
                'NegativeLabels',
                'Episodes',
                'Transitions',
            ]
            with heldout_path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=heldout_fields)
                writer.writeheader()
                for step in range(5_000, 50_001, 5_000):
                    writer.writerow(
                        {
                            'env_steps': step,
                            'split': 'heldout_episode',
                            'recent_episode_limit': 20,
                            'transition_limit': 4096,
                            'gate_passed': True,
                            'curriculum_stage': 'W3',
                            'geometry_enabled': geometry,
                            'BrierScore': 0.1,
                            'BrierSkill': 0.1,
                            'AUROC': 0.7,
                            'BellmanError': 0.1,
                            'GradientNorm': 0.01,
                            'EnsembleDisagreement': 0.02,
                            'BoundarySamples': 200,
                            'PositiveLabels': 200,
                            'NegativeLabels': 200,
                            'Episodes': 20,
                            'Transitions': 4096,
                        },
                    )
        for filename in ('milestone_0.pt', 'milestone_50000.pt', 'latest_full.pt', 'final_full.pt'):
            checkpoint = run_dir / 'checkpoints' / filename
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b'checkpoint')
        receipt = tmp_path / arm / 'receipt.json'
        receipts[arm] = receipt
        write_json(
            receipt,
            {
                'status': 'passed',
                'seed': 2028,
                'steps': 50_000,
                'updates': 48_000,
                'source_commit': SOURCE,
                'source_dirty': False,
                'config_file': config,
                'config_sha256': fingerprint,
                'resolved_overrides_sha256': f'resolved-{arm}',
                'estimated_cost_cny': 0.1,
            },
        )
        evaluation = tmp_path / arm / 'evaluation' / 'pointgoal_checkpoint_evaluation.csv'
        evaluations[arm] = evaluation
        evaluation.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            'training_seed',
            'milestone',
            'evaluation_mode',
            'evaluation_role',
            'probability_split',
            'episodes',
            *(
                'mean_return',
                'mean_cost',
                'cost_cvar_20',
                'goal_met_episode_rate',
                'stall_ratio',
                'mean_step_displacement',
                'mean_start_end_displacement',
                'mean_action_norm',
            ),
            'goal_met_total',
            'brier_skill',
            'auroc',
        ]
        cvar, mean_cost, success, stall = endpoint_metrics[arm]
        with evaluation.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for step in (0, 50_000):
                writer.writerow(
                    {
                        'training_seed': 2028,
                        'milestone': step,
                        'evaluation_mode': 'fixed-random-flow',
                        'evaluation_role': 'primary',
                        'probability_split': 'external_isolated_environment',
                        'episodes': 20,
                        'mean_return': 1.0,
                        'mean_cost': mean_cost,
                        'cost_cvar_20': cvar,
                        'goal_met_episode_rate': success,
                        'stall_ratio': stall,
                        'mean_step_displacement': 0.01,
                        'mean_start_end_displacement': 1.0,
                        'mean_action_norm': 0.5,
                        'goal_met_total': int(20 * success),
                        'brier_skill': 0.1,
                        'auroc': 0.7,
                    },
                )
                write_json(
                    evaluation.parent / f'evaluation_seed_2028_step_{step}.json',
                    {
                        'probability_split': 'external_isolated_environment',
                        'evaluation_mode': 'fixed-random-flow',
                        'seeds': list(range(12_027, 12_047)),
                    },
                )
    preregistration_path = tmp_path / 'preregistration.json'
    write_json(preregistration_path, preregistration)
    return run_dirs, receipts, evaluations, preregistration_path


def test_screen_applies_b1_guard_and_selects_top_two_eligible(tmp_path: Path) -> None:
    runs, receipts, evaluations, preregistration = fixture_screen(tmp_path)
    result = summarize_screen(runs, receipts, evaluations, preregistration, SOURCE)
    assert result['arms']['B3']['performance_eligible'] is False
    assert result['selection']['ranked_performance_eligible_arms'] == ['B2', 'B4', 'B1']
    assert result['selection']['selected_for_100k_x3'] == [
        {'arm': 'B2', 'selection_basis': 'performance_eligible'},
        {'arm': 'B4', 'selection_basis': 'performance_eligible'},
    ]
    assert result['final_p4_verdict'] is None


def test_screen_rejects_dirty_or_mismatched_training_source(tmp_path: Path) -> None:
    runs, receipts, evaluations, preregistration = fixture_screen(tmp_path)
    receipt = json.loads(receipts['B4'].read_text(encoding='utf-8'))
    receipt['source_dirty'] = True
    write_json(receipts['B4'], receipt)
    with pytest.raises(ValueError, match='dirty training source'):
        summarize_screen(runs, receipts, evaluations, preregistration, SOURCE)

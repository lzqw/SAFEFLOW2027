# Copyright 2026 SAFEFLOW2027 Authors. All Rights Reserved.
"""Locked protocol and compatibility tests for the P4.7 SHAPO four-environment matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from scripts.star.finalize_shapo_p47_formal import main as finalize_formal
from scripts.star.materialize_shapo_p47_config import canonical_method_sha256, materialize
from scripts.star.validate_shapo_p47_run import validate_trajectory_artifact
from torch import nn

from omnisafe.algorithms.off_policy.star import (
    STAR,
    best_safe_checkpoint_candidate_is_better,
    cost_geometry_contract,
)
from omnisafe.models.safety.distributional_backend import compute_cost_stochastic_reference


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'reports/star/shapo_p47_four_env_preregistration.json'
PROTOCOL_SHA256 = '007c9a89d6b8f63d2c56516f9611d22e60164b82ccff939aa0af060d1dffa337'
MATRIX = ROOT / 'configs/star/shapo_p47_four_env.yaml'
MATRIX_SHA256 = 'f22d644f798e1e036d0ef64359d40ff10fb745d9d97eb2887ec6bb02adc66b0f'
METHOD_SHA256 = 'cf447f9ad97053054d1970a462a3089deffc83b332101f896b492621140f5f3d'


class AnalyticFailureCostCritic(nn.Module):
    """Two agreeing future-failure cost heads with a known action derivative."""

    def __init__(self) -> None:
        super().__init__()
        self.action_scale = nn.Parameter(torch.tensor(0.1))

    def expected_cost(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del obs
        value = 0.2 + self.action_scale * action[..., 0]
        return torch.stack((value, value), dim=0)


def _write_trajectory(path: Path, milestone: int, seeds: list[int]) -> None:
    """Write a compact fixed-seed trajectory fixture accepted by the formal validator."""
    rows = []
    for seed in seeds:
        for step in range(2):
            rows.append(
                {
                    'milestone': milestone,
                    'base_seed': seed,
                    'episode_index': 0,
                    'step': step,
                    'x': float(step),
                    'y': float(seed % 10),
                    'predicted_violation_probability': 0.1 * (step + 1),
                    'predicted_cost_to_go': 0.2,
                    'geometry_budget_ratio': 0.5,
                    'cost_margin': -0.3,
                    'cost_normal_0': 1.0,
                    'cost_normal_1': 0.0,
                    'cost': 0.0,
                    'is_failure': 0,
                },
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_shapo_protocol_matrix_and_frozen_method_are_byte_sealed() -> None:
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert hashlib.sha256(MATRIX.read_bytes()).hexdigest() == MATRIX_SHA256
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    matrix = yaml.safe_load(MATRIX.read_text(encoding='utf-8'))
    assert protocol['status'] == 'locked_before_any_shapo_p47_formal_training'
    assert protocol['scope']['pointgoal_method_search_closed'] is True
    assert protocol['scope']['new_weight_scans'] is False
    assert protocol['formal_training']['training_seeds'] == [2027, 2028, 2029, 2030, 2031]
    assert protocol['formal_training']['total_environment_steps'] == 150_000_000
    assert protocol['formal_training']['scientific_early_stop'] is False
    assert protocol['formal_evaluation']['episodes_per_training_seed'] == 100
    assert protocol['formal_evaluation']['statistical_unit'] == 'training_seed'
    assert matrix['protocol_sha256'] == PROTOCOL_SHA256
    assert matrix['frozen_method_canonical_sha256'] == METHOD_SHA256
    assert matrix['training_invariants']['replay_size'] == 400_000
    assert matrix['training_invariants']['batch_size'] == 256


@pytest.mark.parametrize(
    ('env_id', 'steps', 'budget', 'mode', 'scale', 'milestones'),
    [
        (
            'SafetyPointGoal1-v0',
            10_000_000,
            10.0,
            'remaining_cost_budget',
            10.0,
            [1_000_000, 4_000_000, 10_000_000],
        ),
        (
            'SafetyPointButton1-v0',
            10_000_000,
            10.0,
            'remaining_cost_budget',
            10.0,
            [1_000_000, 4_000_000, 10_000_000],
        ),
        (
            'Ant-v4',
            5_000_000,
            0.0,
            'future_failure_probability',
            1.0,
            [500_000, 2_000_000, 5_000_000],
        ),
        (
            'Walker2d-v4',
            5_000_000,
            0.0,
            'future_failure_probability',
            1.0,
            [500_000, 2_000_000, 5_000_000],
        ),
    ],
)
def test_formal_materialization_changes_only_locked_environment_and_procedure_fields(
    env_id: str,
    steps: int,
    budget: float,
    mode: str,
    scale: float,
    milestones: list[int],
) -> None:
    resolved = materialize(MATRIX, env_id)
    custom = resolved['custom_cfgs']
    algo = custom['algo_cfgs']
    assert canonical_method_sha256(resolved) == METHOD_SHA256
    assert custom['train_cfgs'] == {'total_steps': steps, 'eval_episodes': 100}
    assert custom['lagrange_cfgs']['cost_limit'] == budget
    assert algo['eval_seeds'] == [32027, 32028, 32029, 32030, 32031]
    assert algo['eval_episodes_per_seed'] == 20
    assert algo['eval_milestones'] == milestones
    assert algo['checkpoint_milestones'] == milestones
    assert algo['directional_loss_weight'] == 1.0
    assert algo['safety_aware_cost_geometry_enabled'] is True
    assert algo['candidate_probing_enabled'] is False
    assert algo['safety_inward_drift_enabled'] is False
    assert algo['safety_outward_rectification_enabled'] is False
    assert resolved['shapo_four_env_metadata']['geometry_mode'] == mode
    assert resolved['shapo_four_env_metadata']['cost_normalization_scale'] == scale
    assert resolved['shapo_four_env_metadata']['engineering_smoke_only'] is False


def test_engineering_materialization_is_explicitly_non_scientific() -> None:
    smoke = materialize(MATRIX, 'Ant-v4', smoke=True)
    assert smoke['custom_cfgs']['train_cfgs'] == {'total_steps': 32, 'eval_episodes': 1}
    assert smoke['custom_cfgs']['algo_cfgs']['max_episode_steps'] == 25
    assert smoke['shapo_four_env_metadata']['engineering_smoke_only'] is True
    assert 'smoke_schedule_overrides' in smoke['shapo_four_env_metadata']


def test_cost_geometry_contract_is_exact_for_budget_and_binary_failure_tasks() -> None:
    assert cost_geometry_contract('SafetyPointGoal1-v0', 10.0) == (
        'remaining_cost_budget',
        10.0,
    )
    assert cost_geometry_contract('SafetyPointButton1-v0', 10.0) == (
        'remaining_cost_budget',
        10.0,
    )
    assert cost_geometry_contract('Ant-v4', 0.0) == ('future_failure_probability', 1.0)
    assert cost_geometry_contract('Walker2d-v4', 0.0) == (
        'future_failure_probability',
        1.0,
    )
    with pytest.raises(ValueError, match='positive Point-task budget'):
        cost_geometry_contract('SafetyPointGoal1-v0', 0.0)
    with pytest.raises(ValueError, match='zero-budget'):
        cost_geometry_contract('Ant-v4', 1.0)


def test_zero_budget_failure_geometry_uses_unit_score_scale_and_zero_boundary() -> None:
    critic = AnalyticFailureCostCritic()
    obs = torch.zeros(3, 8)
    action = torch.zeros(3, 2)
    reference = compute_cost_stochastic_reference(
        critic,
        obs,
        action,
        torch.zeros(3),
        anisotropy_enabled=True,
        safety_aware_geometry=True,
        cost_normalization_scale=1.0,
        boundary_bandwidth=0.1,
    )
    torch.testing.assert_close(reference.risk_ucb, torch.full((3,), 0.2))
    torch.testing.assert_close(reference.margin, torch.full((3,), 0.2))
    torch.testing.assert_close(reference.normal, torch.tensor([[1.0, 0.0]]).expand(3, -1))
    torch.testing.assert_close(reference.gradient_norm, torch.full((3,), 0.1))
    torch.testing.assert_close(
        reference.boundary_reliability,
        torch.sigmoid(reference.margin / 0.1),
    )
    assert bool((reference.reliability > 0.0).all())
    with pytest.raises(ValueError, match='positive normalization scale'):
        compute_cost_stochastic_reference(
            critic,
            obs,
            action,
            torch.zeros(3),
            anisotropy_enabled=True,
            cost_normalization_scale=0.0,
        )


def test_locomotion_geometry_ignores_placeholder_context_budget() -> None:
    algorithm = object.__new__(STAR)
    algorithm._env_id = 'Ant-v4'  # pylint: disable=protected-access
    algorithm._cfgs = SimpleNamespace(  # pylint: disable=protected-access
        lagrange_cfgs=SimpleNamespace(cost_limit=0.0),
    )
    algorithm._actor_critic = SimpleNamespace(context_dim=3)  # pylint: disable=protected-access
    augmented = torch.zeros(4, 10)
    augmented[:, -3] = 1.0
    budget = algorithm._cost_geometry_budget_ratio(augmented)  # pylint: disable=protected-access
    torch.testing.assert_close(budget, torch.zeros(4))
    assert algorithm._cost_geometry_scale() == 1.0  # pylint: disable=protected-access


def test_best_safe_return_checkpoint_order_is_feasible_first_and_seed_local() -> None:
    unsafe = {
        'milestone': 10,
        'mean_return': 100.0,
        'mean_cost': 1.0,
        'constraint_satisfied': False,
    }
    safer_unsafe = {
        'milestone': 20,
        'mean_return': 1.0,
        'mean_cost': 0.5,
        'constraint_satisfied': False,
    }
    safe_low_return = {
        'milestone': 30,
        'mean_return': 0.0,
        'mean_cost': 0.0,
        'constraint_satisfied': True,
    }
    safe_high_return = {
        'milestone': 40,
        'mean_return': 2.0,
        'mean_cost': 0.0,
        'constraint_satisfied': True,
    }
    assert best_safe_checkpoint_candidate_is_better(unsafe, None)
    assert best_safe_checkpoint_candidate_is_better(safer_unsafe, unsafe)
    assert best_safe_checkpoint_candidate_is_better(safe_low_return, safer_unsafe)
    assert best_safe_checkpoint_candidate_is_better(safe_high_return, safe_low_return)
    assert not best_safe_checkpoint_candidate_is_better(unsafe, safe_high_return)


def test_launchers_enforce_clean_source_smoke_gate_and_no_scientific_early_stop() -> None:
    smoke = (ROOT / 'scripts/star/launch_shapo_p47_smoke.sh').read_text(encoding='utf-8')
    formal = (ROOT / 'scripts/star/launch_shapo_p47_formal.sh').read_text(encoding='utf-8')
    for launcher in (smoke, formal):
        assert 'status --porcelain' in launcher
        assert PROTOCOL_SHA256 in launcher
        assert MATRIX_SHA256 in launcher
    assert 'smoke_gate.json' in formal
    assert '--replay-size 400000' in formal
    assert '--batch-size 256' in formal
    assert 'scientific_early_stop=false' in formal
    assert 'for seed in 2027 2028 2029 2030 2031' in formal


def test_trajectory_validator_requires_exact_seed_coverage_and_finite_risk(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'trajectory_step_100.csv'
    seeds = [32027, 32029, 32031]
    _write_trajectory(path, 100, seeds)
    result = validate_trajectory_artifact(path, 100, seeds)
    assert result['row_count'] == 6
    assert result['trajectory_seeds'] == seeds
    assert result['rows_per_seed'] == {str(seed): 2 for seed in seeds}

    _write_trajectory(path, 100, seeds[:-1])
    with pytest.raises(ValueError, match='seed coverage mismatch'):
        validate_trajectory_artifact(path, 100, seeds)

    _write_trajectory(path, 100, seeds)
    text = path.read_text(encoding='utf-8').replace('0.1,0.2', 'nan,0.2', 1)
    path.write_text(text, encoding='utf-8')
    with pytest.raises(ValueError, match='not finite'):
        validate_trajectory_artifact(path, 100, seeds)


def test_finalizer_aggregates_twenty_runs_by_training_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = yaml.safe_load(MATRIX.read_text(encoding='utf-8'))
    run_root = tmp_path / 'formal'
    output = tmp_path / 'summary'
    run_root.mkdir()
    (run_root / 'formal_matrix_complete').touch()
    slugs = {
        'SafetyPointGoal1-v0': 'point-goal',
        'SafetyPointButton1-v0': 'point-button',
        'Ant-v4': 'ant',
        'Walker2d-v4': 'walker2d',
    }
    for env_index, (env_id, environment) in enumerate(matrix['environments'].items()):
        for seed_index, seed in enumerate(matrix['training_seeds']):
            container = run_root / 'runs' / f'{slugs[env_id]}-{seed}'
            container.mkdir(parents=True)
            evaluations = []
            trajectories = []
            for milestone in environment['milestones']:
                value = float(env_index + seed_index + 1)
                evaluations.append(
                    {
                        'milestone': milestone,
                        'mean_return': value,
                        'mean_cost': value / 10.0,
                        'cost_cvar_20': value / 5.0,
                        'training_cumulative_cost': value * 100.0,
                        'success_rate': value / 10.0,
                        'mean_path_length': value * 2.0,
                        'cost_per_successful_episode': value / 2.0,
                        'failure_count': value,
                        'failure_rate': value / 100.0,
                        'mean_failure_time': value,
                        'mean_failure_step': value,
                        'mean_failure_height': value,
                        'mean_failure_torso_angle': value,
                        'time_limit_count': 100.0 - value,
                        'time_limit_rate': 1.0 - value / 100.0,
                        'high_risk_vn_vt': 0.5,
                        'high_risk_dangerous_outward_velocity': 0.1,
                    },
                )
                trajectory_path = (
                    container / 'training' / 'artifacts' / f'trajectory_step_{milestone}.csv'
                )
                _write_trajectory(
                    trajectory_path,
                    milestone,
                    matrix['evaluation']['trajectory_base_seeds'],
                )
                trajectories.append(
                    validate_trajectory_artifact(
                        trajectory_path,
                        milestone,
                        matrix['evaluation']['trajectory_base_seeds'],
                    ),
                )
            (container / 'formal_validation.json').write_text(
                json.dumps(
                    {
                        'status': 'passed',
                        'source_commit': 'a' * 40,
                        'environment': env_id,
                        'training_seed': seed,
                        'evaluation_rows': evaluations,
                        'trajectory_artifacts': trajectories,
                    },
                ),
                encoding='utf-8',
            )
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'finalize_shapo_p47_formal.py',
            '--run-root',
            str(run_root),
            '--matrix',
            str(MATRIX),
            '--output-dir',
            str(output),
        ],
    )
    finalize_formal()
    with (output / 'shapo_p47_final_seed_metrics.csv').open(
        encoding='utf-8',
        newline='',
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 20
    summary = json.loads((output / 'shapo_p47_final_summary.json').read_text('utf-8'))
    assert summary['statistical_unit'] == 'training_seed'
    assert summary['aggregate']['Ant-v4']['mean_return']['n_training_seeds'] == 5
    assert summary['trajectory_artifact_count'] == 60
    assert len(summary['figures']) == 6
    assert all((output / path).is_file() for path in summary['figures'])
    with (output / 'shapo_p47_trajectory_manifest.csv').open(
        encoding='utf-8',
        newline='',
    ) as stream:
        trajectory_rows = list(csv.DictReader(stream))
    assert len(trajectory_rows) == 60
    report = (output / 'SHAPO_P47_FOUR_ENV_REPORT.md').read_text('utf-8')
    assert 'Episode-level observations are not treated as independent replicates' in report
    assert 'High-risk direction diagnostics' in report
    assert 'shapo_p47_ant_trajectory_risk.png' in report

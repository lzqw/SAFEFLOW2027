# Copyright 2026 OmniSafe Team. All Rights Reserved.
"""Locked correctness tests for P4.7 safety-aware cost geometry."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import torch
import yaml
from scripts.star.plot_p4_7_safety_aware_geometry import plot_curves, plot_pareto
from scripts.star.summarize_p4_7_safety_aware_geometry import (
    _render_report,
    scientific_interpretation,
    select_final,
)
from torch import nn

import omnisafe
from omnisafe.algorithms.off_policy.star import high_risk_geometry_summary
from omnisafe.models.safety.distributional_backend import (
    compute_cost_stochastic_reference,
    fixed_trace_variances,
    one_sided_safety_weight,
    safety_aware_directional_loss,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'reports/star/p4_7_safety_aware_geometry_preregistration.json'
PROTOCOL_SHA256 = '3310d6f229d579a877cfaa98d3df2a44b25eac9998184623d1723b82d37a03ed'


class AnalyticCostCritic(nn.Module):
    """Two agreeing cumulative-cost heads with a known action normal."""

    def __init__(self) -> None:
        super().__init__()
        self.action_scale = nn.Parameter(torch.tensor(2.0))

    def expected_cost(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del obs
        value = 5.0 + self.action_scale * action[..., 0]
        return torch.stack((value, value), dim=0)


def _config(name: str) -> dict:
    return yaml.safe_load((ROOT / 'configs/star' / name).read_text(encoding='utf-8'))


def test_protocol_hash_scope_and_checkpoint_seals() -> None:
    payload = PROTOCOL.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == PROTOCOL_SHA256
    protocol = json.loads(payload)
    assert protocol['status'] == 'locked_before_any_p4_7_training'
    assert protocol['base_commit'] == 'ff68bab356bf7052d9e920953ec6b24e55c2d66a'
    assert protocol['method']['directional_loss_weight'] == 1.0
    assert protocol['method']['tau_C'] == 0.1
    assert protocol['training']['scientific_early_stop'] is False
    assert protocol['post_P47']['pointgoal_weight_search_stops'] is True
    assert protocol['evaluation']['formal']['episodes_per_training_seed'] == 100
    assert protocol['evaluation']['formal']['base_seeds'] == [
        22027,
        22028,
        22029,
        22030,
        22031,
    ]
    assert protocol['checkpoints']['C0_resume_full_checkpoint_sha256'] == {
        '2027': '0c8d0261d7b957f73bc72f1ce2542ddaa84a8f5061c43461c72f7a8e390a040b',
        '2028': 'bdc781c7ad152569314d128ebd65c4e838c6d9cbe6035f2b4d1cfd8c13c053da',
        '2029': '3ae57f4eb265cf5939af152eebf82ea4cda1bede3aa99d7f12764b20a47dbf8e',
    }
    launcher = (ROOT / 'scripts/star/launch_p4_7_matrix.sh').read_text(encoding='utf-8')
    assert 'rev-parse --is-inside-work-tree' in launcher
    assert 'scientific_early_stop=false' in launcher


def test_p47_is_exactly_one_joint_method_change_from_continued_c0() -> None:
    c0 = _config('p4_7_c0_continuation.yaml')
    p47 = _config('p4_7_safety_aware_geometry.yaml')
    c0_algo = c0['custom_cfgs']['algo_cfgs']
    p47_algo = p47['custom_cfgs']['algo_cfgs']
    assert c0_algo['directional_loss_enabled'] is False
    assert c0_algo['safety_aware_cost_geometry_enabled'] is False
    assert p47_algo['directional_loss_enabled'] is True
    assert p47_algo['safety_aware_cost_geometry_enabled'] is True
    for algo in (c0_algo, p47_algo):
        assert algo['directional_loss_weight'] == 1.0
        assert algo['cost_geometry_bandwidth'] == 0.1
        assert algo['eval_episodes_per_seed'] == 4
        assert algo['evaluate_initial_policy'] is False
        assert algo['eval_milestones'] == [50_000, 100_000, 150_000, 200_000]
        assert algo['eval_seeds'] == [22027, 22028, 22029, 22030, 22031]
        assert algo['candidate_probing_enabled'] is False
        assert algo['safety_inward_drift_enabled'] is False
        assert algo['safety_outward_rectification_enabled'] is False
        assert algo['numerical_hard_stops_enabled'] is False
        assert algo['engineering_hard_stops_enabled'] is True
        assert algo['safety_protocol_sha256'] == PROTOCOL_SHA256
    c0_algo.pop('directional_loss_enabled')
    c0_algo.pop('safety_aware_cost_geometry_enabled')
    p47_algo.pop('directional_loss_enabled')
    p47_algo.pop('safety_aware_cost_geometry_enabled')
    assert c0 == p47


def test_continued_c0_changes_only_horizon_evaluation_and_protocol_metadata() -> None:
    sealed = _config('p4_5_c0_unified_cost_geometry.yaml')
    continued = _config('p4_7_c0_continuation.yaml')
    sealed_train = sealed['custom_cfgs']['train_cfgs']
    continued_train = continued['custom_cfgs']['train_cfgs']
    assert sealed_train == {'total_steps': 100_000, 'eval_episodes': 20}
    assert continued_train == {'total_steps': 200_000, 'eval_episodes': 20}
    sealed_algo = sealed['custom_cfgs']['algo_cfgs']
    continued_algo = continued['custom_cfgs']['algo_cfgs']
    intended_protocol_changes = {
        'safety_protocol_version',
        'safety_protocol_sha256',
        'eval_seeds',
        'eval_episodes_per_seed',
        'evaluate_initial_policy',
        'trajectory_eval_seeds',
        'eval_milestones',
        'checkpoint_milestones',
        'full_checkpoint_freq',
        'safety_aware_cost_geometry_enabled',
    }
    for key in intended_protocol_changes:
        continued_algo.pop(key, None)
        sealed_algo.pop(key, None)
    assert continued_algo == sealed_algo


def test_one_sided_weight_is_monotone_and_does_not_decay_in_danger() -> None:
    margins = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    weights = one_sided_safety_weight(margins, 0.1)
    assert bool((weights[1:] > weights[:-1]).all())
    assert weights[2] == 0.5
    assert weights[-1] > weights[-2] > 0.99
    with torch.no_grad():
        distant_danger = one_sided_safety_weight(torch.tensor([1.0, 2.0]), 0.1)
    assert distant_danger[1] >= distant_danger[0]


def test_reference_uses_identical_continuous_reliability_for_diffusion_and_loss() -> None:
    critic = AnalyticCostCritic()
    obs = torch.zeros(3, 6)
    action = torch.zeros(3, 2)
    reference = compute_cost_stochastic_reference(
        critic,
        obs,
        action,
        torch.tensor([1.0, 0.5, 0.0]),
        anisotropy_enabled=True,
        safety_aware_geometry=True,
        cost_limit=10.0,
        boundary_bandwidth=0.1,
    )
    torch.testing.assert_close(reference.margin, torch.tensor([-0.5, 0.0, 0.5]))
    torch.testing.assert_close(
        reference.boundary_reliability,
        torch.sigmoid(reference.margin / 0.1),
    )
    expected_normal, expected_tangent = fixed_trace_variances(
        reference.reliability,
        2,
        0.04,
        0.1,
    )
    torch.testing.assert_close(reference.normal_variance, expected_normal)
    torch.testing.assert_close(reference.tangent_variance, expected_tangent)
    assert reference.normal_variance[-1] < reference.normal_variance[1]
    assert reference.normal_variance[1] < reference.normal_variance[0]

    velocity = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    _, numerator, _ = safety_aware_directional_loss(
        velocity,
        reference.normal,
        reference.reliability,
    )
    torch.testing.assert_close(numerator, reference.reliability.mean())


def test_safety_aware_loss_preserves_sample_prevalence_and_stops_denominator_gradient() -> None:
    velocity = torch.zeros(10, 2)
    velocity[:, 1] = 1.0
    velocity[0] = torch.tensor([1.0, 0.0])
    velocity.requires_grad_(True)
    normal = torch.tensor([[1.0, 0.0]]).expand_as(velocity).clone().requires_grad_(True)
    reliability = torch.zeros(10, requires_grad=True)
    with torch.no_grad():
        reliability[0] = 1.0
    loss, numerator, denominator = safety_aware_directional_loss(
        velocity,
        normal,
        reliability,
    )
    assert math.isclose(float(numerator.item()), 0.1, rel_tol=1e-6)
    assert math.isclose(float(denominator.item()), 1.0, rel_tol=1e-6)
    assert math.isclose(float(loss.item()), 0.1, rel_tol=1e-6)
    assert denominator.requires_grad is False
    loss.backward()
    assert velocity.grad is not None
    torch.testing.assert_close(velocity.grad[1:], torch.zeros_like(velocity.grad[1:]))
    assert velocity.grad[0, 0] > 0.0
    assert normal.grad is None
    assert reliability.grad is None


def test_high_risk_geometry_uses_top_margin_twenty_percent_with_stable_ties() -> None:
    summary = high_risk_geometry_summary(
        torch.tensor([2.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]),
        {
            'dangerous_outward_velocity': torch.tensor(
                [10.0, 30.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            ),
            'normal_to_tangent_variance_ratio': torch.tensor(
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ),
        },
        fraction=0.2,
    )
    assert summary['population_samples'] == 10
    assert summary['selected_samples'] == 2
    assert summary['minimum_selected_margin'] == 2.0
    assert summary['dangerous_outward_velocity'] == 20.0
    assert math.isclose(summary['normal_to_tangent_variance_ratio'], 0.3, rel_tol=1e-6)


def test_p47_micro_loop_exercises_joint_geometry_and_writes_method_identity(
    tmp_path: Path,
) -> None:
    custom = _config('p4_7_safety_aware_geometry.yaml')['custom_cfgs']
    custom['seed'] = 4727
    custom['train_cfgs'].update(
        {
            'device': 'cpu',
            'torch_threads': 1,
            'total_steps': 24,
            'eval_episodes': 0,
        },
    )
    custom['algo_cfgs'].update(
        {
            'steps_per_epoch': 12,
            'size': 64,
            'batch_size': 8,
            'random_steps': 4,
            'terminal_utility_steps': 8,
            'geometry_min_steps': 12,
            'eval_episodes_per_seed': 1,
            'eval_milestones': [24],
            'checkpoint_milestones': [],
            'counterfactual_batch_size': 8,
            'milestone_checkpoints': False,
            'checkpoint_replay': False,
            'full_checkpoint_freq': 0,
        },
    )
    custom['model_cfgs'] = {
        'actor': {'hidden_sizes': [16, 16]},
        'critic': {'hidden_sizes': [16, 16], 'num_quantiles': 5},
    }
    custom['logger_cfgs'] = {
        'log_dir': str(tmp_path),
        'use_tensorboard': False,
        'use_wandb': False,
    }
    wrapper = omnisafe.Agent(
        'STAR',
        'SafetyPointGoal1-v0',
        custom_cfgs=custom,
    )
    wrapper.learn()
    algorithm = wrapper.agent
    assert algorithm.env_steps == 24
    assert algorithm.update_count == 20
    log_dir = Path(algorithm.logger.log_dir)
    summary = json.loads((log_dir / 'artifacts/training_summary.json').read_text())
    manifest = json.loads((log_dir / 'artifacts/run_manifest.json').read_text())
    assert summary['status'] == 'completed'
    assert summary['safety_aware_cost_geometry_enabled'] is True
    assert summary['directional_loss_denominator_stop_gradient'] is True
    identity = manifest['p4_7_safety_aware_geometry']
    assert identity['enabled'] is True
    assert identity['shared_direction_and_diffusion_reliability'] is True
    diagnostics = (log_dir / 'artifacts/diagnostics.csv').read_text(encoding='utf-8')
    assert 'Loss/Direction' in diagnostics
    assert 'Direction/Denominator' in diagnostics


def test_locked_final_selection_freezes_p47_only_when_every_gate_passes() -> None:
    seed_rows: dict[str, dict[int, dict[str, float]]] = {arm: {} for arm in ('C0', 'C1', 'P47')}
    for seed in (2027, 2028, 2029):
        seed_rows['C0'][seed] = {
            'mean_return': 1.0,
            'success_rate': 0.30,
            'mean_cost': 10.0,
            'cost_cvar_20': 20.0,
            'training_cumulative_cost': 100.0,
        }
        seed_rows['C1'][seed] = {
            'mean_return': 1.0,
            'success_rate': 0.30,
            'mean_cost': 6.0,
            'cost_cvar_20': 12.0,
            'training_cumulative_cost': 80.0,
        }
        seed_rows['P47'][seed] = {
            'mean_return': 0.95,
            'success_rate': 0.28,
            'mean_cost': 6.5,
            'cost_cvar_20': 12.5,
            'training_cumulative_cost': 70.0,
        }
    pooled = {
        'C0': {
            'mean_return': 1.0,
            'success_rate': 0.30,
            'mean_cost': 10.0,
            'cost_cvar_20': 20.0,
            'training_cumulative_cost_mean': 100.0,
        },
        'C1': {
            'mean_return': 1.0,
            'success_rate': 0.30,
            'mean_cost': 6.0,
            'cost_cvar_20': 12.0,
            'training_cumulative_cost_mean': 80.0,
        },
        'P47': {
            'mean_return': 0.95,
            'success_rate': 0.28,
            'mean_cost': 6.5,
            'cost_cvar_20': 12.5,
            'training_cumulative_cost_mean': 70.0,
        },
    }
    decision = select_final(seed_rows, pooled)
    assert decision['label'] == 'freeze_P47'
    assert decision['all_gates_pass'] is True

    pooled['P47']['mean_return'] = 0.7
    decision = select_final(seed_rows, pooled)
    assert decision['label'] == 'freeze_C1'
    assert decision['gates']['performance_retention_vs_C1_100k'] is False


def test_scientific_interpretation_is_equal_budget_and_separate_from_machine_audit() -> None:
    seed_rows: dict[str, dict[int, dict[str, float]]] = {arm: {} for arm in ('C0', 'C1', 'P47')}
    for seed in (2027, 2028, 2029):
        seed_rows['C0'][seed] = {
            'mean_return': 1.0,
            'success_rate': 0.40,
            'mean_cost': 10.0,
            'cost_cvar_20': 20.0,
            'training_cumulative_cost': 100.0,
        }
        seed_rows['C1'][seed] = {
            'mean_return': 1.0,
            'success_rate': 0.40,
            'mean_cost': 6.0,
            'cost_cvar_20': 12.0,
            'training_cumulative_cost': 80.0,
        }
        seed_rows['P47'][seed] = {
            'mean_return': 0.70,
            'success_rate': 0.20,
            'mean_cost': 8.0,
            'cost_cvar_20': 16.0,
            'training_cumulative_cost': 70.0,
        }
    pooled = {
        'C0': {
            'mean_return': 1.0,
            'success_rate': 0.40,
            'mean_cost': 10.0,
            'cost_cvar_20': 20.0,
            'training_cumulative_cost_mean': 100.0,
            'high_risk_normal_to_tangent_variance_ratio': 1.0,
            'high_risk_dangerous_outward_velocity': 0.5,
        },
        'C1': {
            'mean_return': 1.0,
            'success_rate': 0.40,
            'mean_cost': 6.0,
            'cost_cvar_20': 12.0,
            'training_cumulative_cost_mean': 80.0,
            'high_risk_normal_to_tangent_variance_ratio': 0.8,
            'high_risk_dangerous_outward_velocity': 0.3,
        },
        'P47': {
            'mean_return': 0.70,
            'success_rate': 0.20,
            'mean_cost': 8.0,
            'cost_cvar_20': 16.0,
            'training_cumulative_cost_mean': 70.0,
            'high_risk_normal_to_tangent_variance_ratio': 0.6,
            'high_risk_dangerous_outward_velocity': 0.2,
        },
    }

    machine_audit = select_final(seed_rows, pooled)
    interpretation = scientific_interpretation(seed_rows, pooled)

    assert machine_audit['label'] == 'freeze_C1'
    assert interpretation['primary_comparison'].startswith('P47@200000_vs_C0@200000')
    overall = interpretation['overall_safety_performance']
    assert overall['label'] == 'effective_safety_performance_tradeoff'
    assert overall['all_three_safety_indicators_improved'] is True
    assert overall['task_completion_observed'] is True
    assert interpretation['training_period_safety_exploration']['label'] == (
        'safer_training_period_exploration'
    )
    geometry = interpretation['direction_geometry_mechanism']
    assert geometry['label'] == 'direction_mechanism_supported'
    assert geometry['used_as_elimination_gate'] is False
    assert interpretation['C1_reference']['role'] == 'historical_reference_only'
    assert interpretation['C1_reference']['used_in_primary_scientific_interpretation'] is False
    report = _render_report(
        {
            'pooled_formal': pooled,
            'scientific_interpretation': interpretation,
            'machine_audit': machine_audit,
        },
    )
    assert report.index('Primary scientific conclusion') < report.index(
        'Preserved preregistered machine audit',
    )
    assert '**effective_safety_performance_tradeoff**' in report
    assert '**freeze_C1**' in report
    assert 'diagnostic; it is not an elimination or freeze gate' in report


def test_p47_plotter_renders_common_curves_and_formal_pareto(tmp_path: Path) -> None:
    summary_dir = tmp_path / 'summary'
    figures = tmp_path / 'figures'
    summary_dir.mkdir()
    figures.mkdir()
    checkpoint_rows = []
    final_rows = []
    pooled = {}
    for arm_index, arm in enumerate(('C0', 'C1', 'P47')):
        for seed in (2027, 2028, 2029):
            row = {
                'arm': arm,
                'training_seed': seed,
                'milestone': 100_000,
                'mean_return': 1.0 + 0.1 * arm_index,
                'success_rate': 0.2 + 0.1 * arm_index,
                'mean_cost': 10.0 - arm_index,
                'cost_cvar_20': 20.0 - arm_index,
                'training_cumulative_cost': 1000.0 - 100.0 * arm_index,
                'high_risk_normal_to_tangent_variance_ratio': 1.0 - 0.1 * arm_index,
                'high_risk_dangerous_outward_velocity': 0.5 - 0.1 * arm_index,
            }
            checkpoint_rows.append(row)
            final_rows.append(row)
        pooled[arm] = {
            key: value
            for key, value in checkpoint_rows[-1].items()
            if key
            in {
                'mean_return',
                'success_rate',
                'mean_cost',
                'cost_cvar_20',
            }
        }
    for name, rows in (
        ('p4_7_checkpoint_metrics.csv', checkpoint_rows),
        ('p4_7_formal_final_seed_metrics.csv', final_rows),
    ):
        with (summary_dir / name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (summary_dir / 'p4_7_final_summary.json').write_text(
        json.dumps({'pooled_formal': pooled}),
        encoding='utf-8',
    )
    curve_paths = plot_curves(summary_dir, figures)
    pareto_path = plot_pareto(summary_dir, figures)
    assert len(curve_paths) == 3
    assert all(path.is_file() and path.stat().st_size > 0 for path in curve_paths)
    assert pareto_path.is_file() and pareto_path.stat().st_size > 0

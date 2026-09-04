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
"""Numerical contracts for compact STAR validation artifact tooling."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scripts.star.evaluate_pointgoal_checkpoints import (
    confidence_summary,
    evaluate_rollouts,
    json_safe,
    public_row,
    resolve_eval_seeds,
    sample_flow_base,
)
from scripts.star.visualize_pointgoal_mechanisms import (
    fixed_budget_risk_strata,
    plot_fixed_budget_strata,
    plot_trajectories,
    representative_state,
)

from omnisafe.algorithms.off_policy.star import _spearman
from omnisafe.utils.star_validation import (
    average_ranks,
    boundary_score_rows,
    calibration_rows,
    checkpoint_reconstruction_config,
    deterministic_indices,
    spearman,
    threshold_rows,
)


def test_checkpoint_reconstruction_prefers_saved_config_and_infers_legacy_shapes() -> None:
    saved = {
        'extra': {
            'resolved_config': {
                'algo_cfgs': {'boundary_low': 0.07},
                'model_cfgs': {'actor': {'hidden_sizes': [11, 13]}},
            },
        },
    }
    config = checkpoint_reconstruction_config(saved)
    assert config['algo_cfgs']['boundary_low'] == pytest.approx(0.07)
    assert config['model_cfgs']['actor']['hidden_sizes'] == [11, 13]

    legacy = {
        'model': {
            'actor.velocity_net.0.weight': torch.zeros(16, 7),
            'actor.velocity_net.2.weight': torch.zeros(16, 16),
            'actor.velocity_net.4.weight': torch.zeros(2, 16),
            'reward_critic.critics.0.0.weight': torch.zeros(32, 8),
            'reward_critic.critics.0.2.weight': torch.zeros(32, 32),
            'reward_critic.critics.0.4.weight': torch.zeros(1, 32),
            'cost_critic.heads.0.0.weight': torch.zeros(32, 9),
            'cost_critic.heads.0.2.weight': torch.zeros(32, 32),
            'cost_critic.heads.0.4.weight': torch.zeros(5, 32),
            'cost_critic.heads.1.0.weight': torch.zeros(32, 9),
            'cost_critic.heads.1.2.weight': torch.zeros(32, 32),
            'cost_critic.heads.1.4.weight': torch.zeros(5, 32),
        },
    }
    inferred = checkpoint_reconstruction_config(legacy)
    assert inferred['model_cfgs']['actor']['hidden_sizes'] == [16, 16]
    assert inferred['model_cfgs']['critic']['hidden_sizes'] == [32, 32]
    assert inferred['model_cfgs']['critic']['num_quantiles'] == 5
    assert inferred['model_cfgs']['critic']['ensemble_size'] == 2


def test_tie_correct_ranks_and_spearman() -> None:
    values = torch.tensor([1.0, 1.0, 3.0, 5.0])
    assert torch.equal(average_ranks(values), torch.tensor([0.5, 0.5, 2.0, 3.0]))
    assert spearman(values, values) == pytest.approx(1.0)
    assert spearman(values, -values) == pytest.approx(-1.0)
    assert _spearman(values, values) == pytest.approx(1.0)
    assert math.isnan(_spearman(values, torch.ones_like(values)))


def test_checkpoint_evaluation_serializes_degenerate_spearman_as_null() -> None:
    payload = {'extra': {'seed': 2027}, 'counters': {'env_steps': 50_000}}
    evaluation = {
        'episodes': 5,
        'mean_return': 1.0,
        'mean_cost': 0.0,
        'cost_cvar_20': 0.0,
        'violation_ratio': 0.0,
        'calibration_error': 0.1,
        'spearman': float('nan'),
        'predicted_risk_mean': 0.1,
        'empirical_risk_mean': 0.0,
    }
    row = public_row(payload, Path('milestone_50000.pt'), evaluation, 'cuda:0')
    assert row['evaluation_device'] == 'cuda:0'
    assert row['spearman'] is None
    assert json_safe({'spearman': float('nan')}) == {'spearman': None}


def test_evaluation_tiers_resolve_fixed_unique_seed_counts() -> None:
    development = resolve_eval_seeds(None, 'development')
    formal = resolve_eval_seeds(None, 'formal')
    assert development == list(range(12_027, 12_047))
    assert formal == list(range(12_027, 12_077))
    assert resolve_eval_seeds('7,11,13', 'development') == [7, 11, 13]
    with pytest.raises(ValueError, match='unique'):
        resolve_eval_seeds('7,7', 'formal')


def test_confidence_summary_reports_sample_std_and_normal_interval() -> None:
    summary = confidence_summary([1.0, 2.0, 3.0])
    assert summary['mean'] == pytest.approx(2.0)
    assert summary['std'] == pytest.approx(1.0)
    assert summary['ci95_low'] < 1.0
    assert summary['ci95_high'] > 3.0
    singleton = confidence_summary([4.0])
    assert singleton == {'mean': 4.0, 'std': 0.0, 'ci95_low': 4.0, 'ci95_high': 4.0}


def test_fixed_random_flow_bases_are_reproducible_and_center_is_zero() -> None:
    first = torch.Generator().manual_seed(123)
    second = torch.Generator().manual_seed(123)
    first_sequence = [
        sample_flow_base(
            2,
            first,
            'fixed-random-flow',
            device=torch.device('cpu'),
            dtype=torch.float32,
        )
        for _ in range(3)
    ]
    second_sequence = [
        sample_flow_base(
            2,
            second,
            'fixed-random-flow',
            device=torch.device('cpu'),
            dtype=torch.float32,
        )
        for _ in range(3)
    ]
    assert all(torch.equal(left, right) for left, right in zip(first_sequence, second_sequence))
    assert all(torch.all(base.abs() <= 1.0) for base in first_sequence)
    center = sample_flow_base(
        2,
        torch.Generator().manual_seed(9),
        'center-flow-diagnostic',
        device=torch.device('cpu'),
        dtype=torch.float32,
    )
    assert torch.equal(center, torch.zeros(1, 2))


class _EvaluationProbeEnv:
    """Two-step vector environment exposing deterministic goal and position signals."""

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict[str, object]]:
        if not hasattr(self, 'reset_seeds'):
            self.reset_seeds: list[int | None] = []
        self.reset_seeds.append(seed)
        if seed is not None:
            self.seed = seed
        self.step_index = 0
        self.position = [0.0, 0.0]
        return torch.zeros(1, 2), {}

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, object],
    ]:
        self.position[0] += 0.2 * float(action[0, 0].item())
        self.position[1] += 0.2 * float(action[0, 1].item())
        goal_met = self.seed == 1 and self.step_index == 0
        self.step_index += 1
        episode_done = self.step_index == 2
        next_position = list(self.position)
        if episode_done:
            self.step_index = 0
            self.position = [0.0, 0.0]
        return (
            torch.tensor([self.position if episode_done else next_position], dtype=torch.float32),
            torch.ones(1),
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            torch.tensor([episode_done]),
            {
                'goal_met': goal_met,
                'original_reward': torch.ones(1),
                'original_cost': torch.zeros(1),
            },
        )


class _EvaluationProbeAdapter:
    def __init__(self) -> None:
        self.evaluation_env = _EvaluationProbeEnv()

    def evaluation_position(self) -> tuple[float, float]:
        return tuple(self.evaluation_env.position)

    @staticmethod
    def evaluation_layout() -> dict[str, object]:
        return {'goal': (1.0, 0.0), 'hazards': [], 'goal_radius': 0.3, 'hazard_radius': 0.2}


class _EvaluationProbeModel:
    def __init__(self) -> None:
        self.training = True
        self.cost_critic = lambda obs, action: torch.zeros(2, obs.shape[0], 3)

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True

    @staticmethod
    def flow(obs: torch.Tensor, base_action: torch.Tensor) -> SimpleNamespace:
        del obs
        return SimpleNamespace(action=base_action)


def test_center_flow_rollout_reports_goal_progress_and_stall_metrics() -> None:
    algorithm = SimpleNamespace(
        _env=_EvaluationProbeAdapter(),
        _env_id='SafetyPointGoal1-v0',
        _device=torch.device('cpu'),
        _act_dim=2,
        _actor_critic=_EvaluationProbeModel(),
        _cfgs=SimpleNamespace(
            algo_cfgs=SimpleNamespace(max_episode_steps=2, risk_temperature=1.0),
            lagrange_cfgs=SimpleNamespace(cost_limit=10.0),
        ),
        _augment=lambda obs, context: torch.cat((obs, context), dim=-1),
    )
    result = evaluate_rollouts(
        algorithm,
        milestone=300_000,
        eval_seeds=[1, 2],
        mode='center-flow-diagnostic',
    )
    assert result['evaluation_role'] == 'diagnostic'
    assert result['goal_met_total'] == 1
    assert result['goal_met_episode_rate'] == pytest.approx(0.5)
    assert result['mean_start_end_displacement'] == 0.0
    assert result['mean_step_displacement'] == 0.0
    assert result['mean_path_length'] == 0.0
    assert result['mean_action_norm'] == 0.0
    assert result['stall_ratio'] == pytest.approx(0.5)
    assert result['step_stall_ratio'] == 1.0
    assert result['mean_return'] == 2.0
    assert algorithm._actor_critic.training


def test_multi_episode_evaluation_seeds_once_then_advances_each_rng_stream() -> None:
    algorithm = SimpleNamespace(
        _env=_EvaluationProbeAdapter(),
        _env_id='SafetyPointGoal1-v0',
        _device=torch.device('cpu'),
        _act_dim=2,
        _actor_critic=_EvaluationProbeModel(),
        _cfgs=SimpleNamespace(
            algo_cfgs=SimpleNamespace(max_episode_steps=2, risk_temperature=1.0),
            lagrange_cfgs=SimpleNamespace(cost_limit=10.0),
        ),
        _augment=lambda obs, context: torch.cat((obs, context), dim=-1),
    )
    result = evaluate_rollouts(
        algorithm,
        milestone=200_000,
        eval_seeds=[1, 2],
        mode='center-flow-diagnostic',
        episodes_per_seed=3,
    )
    assert result['episodes'] == 6
    assert result['base_seeds'] == [1, 2]
    assert result['episodes_per_base_seed'] == 3
    assert result['seeds'] == [1, 1, 1, 2, 2, 2]
    assert result['episode_indices'] == [0, 1, 2, 0, 1, 2]
    assert algorithm._env.evaluation_env.reset_seeds == [1, 2]
    assert result['goal_met_episode_rate'] == pytest.approx(0.5)


def test_trajectory_plot_accepts_seed_qualified_evaluator_artifacts(tmp_path: Path) -> None:
    """Mechanism rendering consumes the evaluator's current trajectory filename schema."""
    header = 'milestone,x,y,goal_x,goal_y,hazards_json,hazard_radius\n'
    row = '20000,0.0,0.0,1.0,1.0,[],0.2\n'
    (tmp_path / 'trajectory_seed_2027_step_20000.csv').write_text(
        header + row,
        encoding='utf-8',
    )
    output = tmp_path / 'trajectories.png'
    assert plot_trajectories(tmp_path, output) == 1
    assert output.stat().st_size > 0


def test_mechanism_representative_is_deterministic_and_risk_independent() -> None:
    """The representative observation is a replay midpoint, not a near-boundary cherry-pick."""

    class _RiskHead:
        @staticmethod
        def probabilities(observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            del action
            return torch.stack((observation[:, 0], observation[:, 0]))

    actor_critic = SimpleNamespace(
        violation_critic=_RiskHead(),
        step=lambda observation, deterministic: torch.zeros(observation.shape[0], 1),
    )
    algorithm = SimpleNamespace(
        _actor_critic=actor_critic,
        _augment=lambda observation, context: torch.cat((observation, context), dim=-1),
        _buf=SimpleNamespace(
            size=5,
            data={
                'obs': torch.tensor([[0.5], [0.1], [0.9], [0.2], [0.3]]),
                'context': torch.zeros(5, 2),
            },
        ),
        _device=torch.device('cpu'),
    )
    first = representative_state(algorithm, sample_size=5)
    second = representative_state(algorithm, sample_size=5)
    assert first['replay_index'].item() == 2
    assert first['raw_obs'].item() == pytest.approx(0.9)
    assert torch.equal(first['raw_obs'], second['raw_obs'])


def test_fixed_budget_visuals_select_each_replay_risk_stratum() -> None:
    """Each fixed budget selects safe, boundary, and high-risk replay observations."""

    class _SmoothRiskHead(torch.nn.Module):
        @staticmethod
        def probabilities(observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            risk = torch.sigmoid(4.0 * (observation[:, 0] + 0.25 * action[:, 0]))
            return torch.stack((risk, risk))

    actor_critic = SimpleNamespace(
        violation_critic=_SmoothRiskHead(),
        step=lambda observation, deterministic: torch.zeros(observation.shape[0], 2),
    )
    algorithm = SimpleNamespace(
        _actor_critic=actor_critic,
        _augment=lambda observation, context: torch.cat((observation, context), dim=-1),
        _buf=SimpleNamespace(
            size=3,
            data={
                'obs': torch.tensor([[-2.0], [0.0], [2.0]]),
                'context': torch.zeros(3, 2),
            },
        ),
        _cfgs=SimpleNamespace(
            algo_cfgs=SimpleNamespace(
                boundary_bandwidth=0.15,
                risk_boundary=0.5,
                risk_ucb_beta=0.0,
            ),
        ),
        _device=torch.device('cpu'),
    )
    rows, panels = fixed_budget_risk_strata(algorithm, sample_size=3, grid_size=5)
    assert len(rows) == 12
    assert len(panels) == 12
    assert {row['sample_count'] for row in rows} == {1}
    assert [row['selected_replay_index'] for row in rows[:3]] == [0, 1, 2]
    assert all(row['risk_p95_minus_p05'] > 0 for row in rows)


def test_fixed_budget_visuals_mark_zero_gradient_normals_undefined(tmp_path: Path) -> None:
    """Saturated action-risk surfaces do not invent a normal direction."""

    class _ConstantRiskHead(torch.nn.Module):
        @staticmethod
        def probabilities(observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            del observation
            risk = action[:, 0] * 0.0 + 1.0
            return torch.stack((risk, risk))

    algorithm = SimpleNamespace(
        _actor_critic=SimpleNamespace(
            violation_critic=_ConstantRiskHead(),
            step=lambda observation, deterministic: torch.zeros(observation.shape[0], 2),
        ),
        _augment=lambda observation, context: torch.cat((observation, context), dim=-1),
        _buf=SimpleNamespace(
            size=2,
            data={'obs': torch.zeros(2, 1), 'context': torch.zeros(2, 2)},
        ),
        _cfgs=SimpleNamespace(
            algo_cfgs=SimpleNamespace(
                boundary_bandwidth=0.15,
                risk_boundary=0.5,
                risk_ucb_beta=0.0,
            ),
        ),
        _device=torch.device('cpu'),
    )
    rows, panels = fixed_budget_risk_strata(algorithm, sample_size=2, grid_size=3)
    high_risk = [row for row in rows if row['risk_stratum'] == 'high_risk']
    assert {row['gradient_norm'] for row in high_risk} == {0.0}
    assert all(row['ensemble_disagreement'] is None for row in high_risk)
    assert all(row['normal_0'] is None and row['normal_1'] is None for row in high_risk)
    output = tmp_path / 'fixed-budget.png'
    plot_fixed_budget_strata(output, panels, step=20_000)
    assert output.stat().st_size > 0


def test_calibration_rows_are_finite_or_empty() -> None:
    rows = calibration_rows(
        torch.tensor([0.05, 0.15, 0.85, 0.95]),
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
        bins=4,
    )
    assert rows[0]['count'] == 2
    assert rows[0]['predicted_risk_mean'] == pytest.approx(0.1)
    assert rows[0]['empirical_risk_mean'] == 0.0
    assert rows[1]['count'] == 0
    assert rows[1]['predicted_risk_mean'] is None
    assert rows[3]['count'] == 2
    assert rows[3]['absolute_gap'] == pytest.approx(0.1)


def test_deterministic_replay_subset_and_boundary_sensitivity() -> None:
    assert torch.equal(deterministic_indices(10, 4), torch.tensor([0, 3, 6, 9]))
    audit = {
        'risk_ucb': torch.tensor([0.01, 0.15, 0.35, 0.48, 0.8]),
        'source_random': torch.tensor([False, True, False, False, False]),
        'is_failure': torch.tensor([False, False, True, False, True]),
    }
    rows = threshold_rows(audit, current_low=0.1, current_safe=0.4, boundary=0.5)
    current = next(row for row in rows if row['is_current'])
    assert current['sample_count'] == 2
    assert current['sample_fraction'] == pytest.approx(0.4)
    assert current['source_random_fraction'] == pytest.approx(0.5)
    assert current['immediate_violation_fraction'] == pytest.approx(0.5)


def test_continuous_boundary_score_sensitivity_uses_finalized_labels() -> None:
    audit = {
        'boundary_score': torch.tensor([0.05, 0.5, 0.8, 0.95]),
        'source_random': torch.tensor([False, True, False, True]),
        'is_failure': torch.tensor([False, False, True, False]),
        'violation_label': torch.tensor([0.0, float('nan'), 1.0, 0.0]),
    }
    rows = boundary_score_rows(audit, current_threshold=0.5)
    current = next(row for row in rows if row['is_current'])
    assert current['sample_count'] == 3
    assert current['positive_label_fraction'] == pytest.approx(0.5)
    assert current['sample_fraction'] == pytest.approx(0.75)
    assert current['source_random_fraction'] == pytest.approx(2 / 3)
    assert current['immediate_violation_fraction'] == pytest.approx(1 / 3)

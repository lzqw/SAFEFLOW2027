"""Tests for the preregistered single-GPU PointCircle stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.star.decide_pointcircle_screen import screen_decision
from scripts.star.evaluate_pointcircle_checkpoints import validate_arm
from scripts.star.evaluate_point_robustness import conditions
from scripts.star.evaluate_pointgoal_prior_baselines import load_receipt
from scripts.star.run_point_robustness_queue import unique_checkpoint as robustness_checkpoint
from scripts.star.run_pointcircle_control_postprocessing import checkpoint as control_checkpoint
from scripts.star.run_pointcircle_postprocessing import unique_checkpoint as final_checkpoint
from scripts.star.run_single_gpu_pointcircle_queue import (
    FORMAL_STEPS,
    REPLAY_SIZE,
    SCREEN_STEPS,
    training_command,
)
from scripts.star.run_single_gpu_pointcircle_prior_queue import (
    FORMAL_STEPS as PRIOR_FORMAL_STEPS,
    REPLAY_SIZE as PRIOR_REPLAY_SIZE,
    command_for as prior_command_for,
    wait_for_pointcircle,
)


ROOT = Path(__file__).resolve().parents[2]


def config(name: str) -> dict:
    """Load one repository STAR override."""
    return yaml.safe_load((ROOT / 'configs/star' / name).read_text(encoding='utf-8'))


def test_pointcircle_configs_change_only_arm_and_resource_schedule() -> None:
    c0 = config('single_gpu_pointcircle_c0.yaml')
    ours = config('single_gpu_pointcircle_p47.yaml')
    old_ours = config('p4_7_safety_aware_geometry.yaml')

    assert c0['env_id'] == ours['env_id'] == 'SafetyPointCircle1-v0'
    assert c0['custom_cfgs']['train_cfgs'] == ours['custom_cfgs']['train_cfgs'] == {
        'total_steps': SCREEN_STEPS,
        'eval_episodes': 20,
    }
    c0_algo = c0['custom_cfgs']['algo_cfgs']
    ours_algo = ours['custom_cfgs']['algo_cfgs']
    differing = {key for key in c0_algo if c0_algo[key] != ours_algo[key]}
    assert differing == {'directional_loss_enabled', 'safety_aware_cost_geometry_enabled'}
    assert c0_algo['directional_loss_enabled'] is False
    assert c0_algo['safety_aware_cost_geometry_enabled'] is False
    assert ours_algo['directional_loss_enabled'] is True
    assert ours_algo['safety_aware_cost_geometry_enabled'] is True

    old_algo = old_ours['custom_cfgs']['algo_cfgs']
    resource_only = {'eval_milestones', 'checkpoint_milestones', 'full_checkpoint_freq'}
    assert {key: value for key, value in ours_algo.items() if key not in resource_only} == {
        key: value for key, value in old_algo.items() if key not in resource_only
    }
    assert ours_algo['eval_milestones'] == [150000, 300000, 500000]
    assert ours_algo['checkpoint_milestones'] == [150000, 300000, 500000]
    assert ours_algo['full_checkpoint_freq'] == 250000


def metrics(
    *,
    mean_return: float = 10.0,
    mean_cost: float = 10.0,
    cvar: float = 20.0,
    vn_vt: float = 1.0,
    outward: float = 0.2,
) -> dict[str, float]:
    """Build one compact screen endpoint."""
    return {
        'mean_return': mean_return,
        'mean_cost': mean_cost,
        'cost_cvar_20': cvar,
        'high_risk_vn_vt': vn_vt,
        'high_risk_dangerous_outward_velocity': outward,
    }


def test_pointcircle_screen_uses_only_return_and_safety_or_geometry() -> None:
    baseline = metrics()
    safety = screen_decision(baseline, metrics(mean_return=9.0, mean_cost=9.0))
    geometry = screen_decision(
        baseline,
        metrics(mean_return=9.0, vn_vt=0.9, outward=0.19),
    )
    return_failure = screen_decision(
        baseline,
        metrics(mean_return=8.9, mean_cost=1.0, cvar=1.0, vn_vt=0.1, outward=0.01),
    )
    no_safety_or_geometry = screen_decision(baseline, metrics(mean_return=10.0))

    assert safety['screen_pass'] is True
    assert geometry['screen_pass'] is True
    assert return_failure['screen_pass'] is False
    assert no_safety_or_geometry['screen_pass'] is False
    assert safety['logic'] == 'return_ok AND (safety_ok OR geometry_ok)'


def test_pointcircle_training_command_locks_vector_replay_and_resume(tmp_path: Path) -> None:
    source = ROOT
    args = argparse.Namespace(
        source=source,
        run_root=tmp_path,
        python_bin=Path('/venv/bin/python'),
    )
    identities = {
        'source_commit': 'a' * 40,
        'protocol_sha256': 'b' * 64,
        'config_sha256': {
            'C0': hashlib.sha256(
                (source / 'configs/star/single_gpu_pointcircle_c0.yaml').read_bytes(),
            ).hexdigest(),
            'P47': hashlib.sha256(
                (source / 'configs/star/single_gpu_pointcircle_p47.yaml').read_bytes(),
            ).hexdigest(),
        },
    }
    resume = tmp_path / 'screen-final.pt'
    command, _, _ = training_command(
        args,
        identities,
        'formal',
        'P47',
        2027,
        FORMAL_STEPS,
        resume,
    )

    assert command[command.index('--vector-envs') + 1] == '1'
    assert command[command.index('--replay-size') + 1] == str(REPLAY_SIZE)
    assert command[command.index('--steps') + 1] == str(FORMAL_STEPS)
    assert command[command.index('--resume') + 1] == str(resume)
    assert '--checkpoint-replay' in command
    assert command[command.index('--full-checkpoint-freq') + 1] == '250000'


def test_return_screen_handles_negative_baseline_without_sign_inversion() -> None:
    baseline = metrics(mean_return=-2.0)
    within = screen_decision(baseline, metrics(mean_return=-2.2, mean_cost=8.0))
    outside = screen_decision(baseline, metrics(mean_return=-2.21, mean_cost=8.0))

    assert within['return_ok'] is True
    assert within['return_relative_degradation'] == pytest.approx(0.1)
    assert outside['return_ok'] is False


def test_pointcircle_prior_commands_lock_task_budget_and_two_single_env_learners(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        source=ROOT,
        official_shapo_source=Path('/official/SHAPO'),
        run_root=tmp_path,
        python_bin=Path('/venv/bin/python'),
    )
    identities = {
        'source_commit': 'a' * 40,
        'official_source_commit': 'b' * 40,
        'protocol_sha256': 'c' * 64,
    }

    saclag, _, _ = prior_command_for(args, identities, 'SACLag', 2027)
    shapo, _, _ = prior_command_for(args, identities, 'SHAPO-PIDLag', 2027)

    assert saclag[saclag.index('--environment') + 1] == 'SafetyPointCircle1-v0'
    assert saclag[saclag.index('--steps') + 1] == str(PRIOR_FORMAL_STEPS)
    assert saclag[saclag.index('--replay-size') + 1] == str(PRIOR_REPLAY_SIZE)
    assert shapo[shapo.index('--task') + 1] == 'SafetyPointCircle1-v0'
    assert shapo[shapo.index('--num-envs') + 1] == '1'
    assert shapo[shapo.index('--total-steps') + 1] == str(PRIOR_FORMAL_STEPS)


def test_pointcircle_prior_gate_and_receipt_environment_are_exact(tmp_path: Path) -> None:
    state = tmp_path / 'queue_state.json'
    state.write_text(
        json.dumps(
            {
                'state': 'complete',
                'protocol_sha256': (
                    '0a21c9165116e15eb13c4f1c84956de515cb7631e3a1a93611239cdd54b1cdf8'
                ),
            }
        ),
        encoding='utf-8',
    )
    assert wait_for_pointcircle(state, 1) is True

    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'checkpoint')
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(
        json.dumps(
            {
                'algorithm': 'SACLag',
                'status': 'complete',
                'phase': 'formal',
                'environment': 'SafetyPointCircle1-v0',
                'steps': 500000,
                'seed': 2027,
                'protocol_sha256': 'd' * 64,
                'checkpoint': str(checkpoint),
                'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        ),
        encoding='utf-8',
    )
    loaded = load_receipt(receipt, 'd' * 64, 'SafetyPointCircle1-v0', 500000)
    assert loaded['environment'] == 'SafetyPointCircle1-v0'
    with pytest.raises(ValueError, match='does not belong'):
        load_receipt(receipt, 'd' * 64, 'SafetyPointGoal1-v0', 500000)


def test_pointcircle_final_evaluator_rejects_mislabeled_arm() -> None:
    payload = {
        'extra': {
            'resolved_config': {
                'algo_cfgs': {
                    'directional_loss_enabled': True,
                    'safety_aware_cost_geometry_enabled': True,
                }
            }
        }
    }
    validate_arm(payload, 'P47', Path('/checkpoint.pt'))
    with pytest.raises(ValueError, match='arm switches'):
        validate_arm(payload, 'C0', Path('/checkpoint.pt'))


def test_pointcircle_mechanism_and_parameter_configs_are_single_controls() -> None:
    c0 = config('single_gpu_pointcircle_c0.yaml')['custom_cfgs']['algo_cfgs']
    ours = config('single_gpu_pointcircle_p47.yaml')['custom_cfgs']['algo_cfgs']
    iso_low = config('single_gpu_pointcircle_iso_low.yaml')['custom_cfgs']['algo_cfgs']
    random_aniso = config('single_gpu_pointcircle_random_aniso.yaml')['custom_cfgs']['algo_cfgs']
    ratio_025 = config('single_gpu_pointcircle_ratio_025.yaml')['custom_cfgs']['algo_cfgs']
    schedule = {'eval_milestones', 'checkpoint_milestones', 'full_checkpoint_freq'}

    def without(mapping: dict, fields: set[str]) -> dict:
        return {key: value for key, value in mapping.items() if key not in fields}

    random_fields = {'random_anisotropy_enabled', 'random_anisotropy_seed'}
    assert without(random_aniso, schedule | random_fields) == without(ours, schedule)
    assert random_aniso['random_anisotropy_enabled'] is True
    assert random_aniso['random_anisotropy_seed'] == 130027

    assert without(ratio_025, schedule | random_fields | {'flow_normal_variance_ratio'}) == without(
        ours,
        schedule | {'flow_normal_variance_ratio'},
    )
    assert ratio_025['flow_normal_variance_ratio'] == 0.25

    iso_changes = {
        'flow_reference_variance',
        'flow_normal_variance_ratio',
    }
    assert without(iso_low, schedule | random_fields | iso_changes) == without(
        c0,
        schedule | iso_changes,
    )
    assert iso_low['flow_reference_variance'] == 0.004
    assert iso_low['flow_normal_variance_ratio'] == 1.0


def test_postprocessors_resolve_actual_milestone_checkpoint_names(tmp_path: Path) -> None:
    formal = (
        tmp_path
        / 'formal/C0-2027/training/STAR-{SafetyPointCircle1-v0}/seed-2027-run/checkpoints'
    )
    formal.mkdir(parents=True)
    circle_final = formal / 'milestone_500000.pt'
    circle_final.write_bytes(b'circle-final')
    assert final_checkpoint(tmp_path, 'C0', 2027) == circle_final

    screen = (
        tmp_path
        / 'screen/C0-2027/training/STAR-{SafetyPointCircle1-v0}/seed-2027-run/checkpoints'
    )
    screen.mkdir(parents=True)
    circle_screen = screen / 'milestone_300000.pt'
    circle_screen.write_bytes(b'circle-screen')
    control_args = argparse.Namespace(star_root=tmp_path, control_root=tmp_path / 'controls')
    assert control_checkpoint(control_args, 'Iso-High') == circle_screen

    pointgoal_root = tmp_path / 'pointgoal'
    pointgoal = (
        pointgoal_root
        / 'P47-2028/training/STAR-{SafetyPointGoal1-v0}/seed-2028-run/checkpoints'
    )
    pointgoal.mkdir(parents=True)
    goal_final = pointgoal / 'milestone_200000.pt'
    goal_final.write_bytes(b'goal-final')
    robustness_args = argparse.Namespace(
        pointgoal_root=pointgoal_root,
        pointcircle_root=tmp_path,
    )
    assert robustness_checkpoint(
        robustness_args,
        'SafetyPointGoal1-v0',
        'P47',
        2028,
    ) == goal_final


def test_robustness_conditions_match_the_locked_minimal_matrix() -> None:
    circle = conditions('SafetyPointCircle1-v0')
    goal = conditions('SafetyPointGoal1-v0')

    assert [item.value for item in circle] == [0.8, 0.9, 1.0, 1.1, 1.2]
    assert [item.value for item in circle if item.selected_final] == [0.8, 1.2]
    assert [item.name for item in goal] == [
        'nominal',
        'hazard_radius_medium',
        'hazard_radius_strong',
        'hazard_position_offset_medium',
        'hazard_position_offset_strong',
        'action_noise_medium',
        'action_noise_strong',
    ]
    assert [item.name for item in goal if item.selected_final] == [
        'hazard_radius_strong',
        'hazard_position_offset_strong',
        'action_noise_strong',
    ]

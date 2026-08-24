"""Tests for the preregistered single-GPU PointCircle stage."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pytest
import yaml

from scripts.star.decide_pointcircle_screen import screen_decision
from scripts.star.run_single_gpu_pointcircle_queue import (
    FORMAL_STEPS,
    REPLAY_SIZE,
    SCREEN_STEPS,
    training_command,
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

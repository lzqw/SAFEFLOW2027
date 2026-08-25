"""Targeted checks for the single PointCircle geometry enhancement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from scripts.star.run_pointcircle_geometry_enhancement import (
    ADDENDUM_SHA256,
    CONFIG_SHA256,
    REPLAY_SIZE,
    STEPS,
    training_command,
)


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(relative: str) -> dict:
    """Load one repository YAML mapping."""
    return yaml.safe_load((ROOT / relative).read_text(encoding='utf-8'))


def test_enhancement_is_the_one_preregistered_two_scale_change() -> None:
    base = load_yaml('configs/star/single_gpu_pointcircle_p47.yaml')
    enhanced = load_yaml('configs/star/single_gpu_pointcircle_p47_geometry_enhanced.yaml')
    base_algo = base['custom_cfgs']['algo_cfgs']
    enhanced_algo = enhanced['custom_cfgs']['algo_cfgs']
    changed = {key for key in base_algo if base_algo[key] != enhanced_algo[key]}

    assert changed == {
        'safety_ensemble_scale',
        'safety_gradient_scale',
        'safety_protocol_sha256',
        'safety_protocol_version',
    }
    assert enhanced_algo['safety_gradient_scale'] == 0.001
    assert enhanced_algo['safety_ensemble_scale'] == 0.10
    assert enhanced_algo['safety_protocol_sha256'] == ADDENDUM_SHA256
    assert hashlib.sha256(
        (ROOT / 'configs/star/single_gpu_pointcircle_p47_geometry_enhanced.yaml').read_bytes(),
    ).hexdigest() == CONFIG_SHA256

    preregistration = json.loads(
        (
            ROOT / 'reports/star/single_gpu_pointcircle_geometry_enhancement_preregistration.json'
        ).read_text(encoding='utf-8'),
    )
    assert preregistration['attempt'] == preregistration['attempt_limit'] == 1
    assert preregistration['selection_independence']['candidate_count'] == 1
    assert preregistration['selection_independence']['no_scientific_early_stopping'] is True


def test_enhancement_command_locks_single_learner_resources(tmp_path: Path) -> None:
    args = argparse.Namespace(
        source=ROOT,
        run_root=tmp_path,
        python_bin=Path('/venv/bin/python'),
    )
    command, receipt, _ = training_command(args)

    assert command[command.index('--seed') + 1] == '2027'
    assert command[command.index('--steps') + 1] == str(STEPS)
    assert command[command.index('--vector-envs') + 1] == '1'
    assert command[command.index('--replay-size') + 1] == str(REPLAY_SIZE)
    assert command[command.index('--full-checkpoint-freq') + 1] == '250000'
    assert '--checkpoint-replay' in command
    assert receipt == tmp_path / 'P47-geometry-enhanced-2027/receipt.json'

"""Targeted checks for the conditional single-GPU Ant queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.star.finalize_single_gpu_ant import (
    REUSED_EVALUATION_SHA256,
    validate_evaluation,
)
from scripts.star.run_single_gpu_ant_queue import (
    CONFIG_SHA256,
    PARENT_PROTOCOL_SHA256,
    REPLAY_SIZE,
    TRAINING_SOURCE_COMMIT,
    TRAINING_STEPS,
    training_command,
    validate_config_pair,
    wait_for_point_core,
)

ROOT = Path(__file__).resolve().parents[2]


def test_ant_configs_have_only_the_two_preregistered_method_switches() -> None:
    validate_config_pair(ROOT)


def test_ant_command_locks_budget_resources_and_historical_code(tmp_path: Path) -> None:
    args = argparse.Namespace(
        controller_source=ROOT,
        training_source=Path('/training-source'),
        run_root=tmp_path,
        python_bin=Path('/venv/bin/python'),
    )
    identities = {
        'training_source_commit': TRAINING_SOURCE_COMMIT,
        'config_sha256': CONFIG_SHA256,
    }
    command, receipt, _, resume = training_command(args, identities, 'P47', 2028)

    assert command[1] == '/training-source/scripts/star/train_star.py'
    assert command[command.index('--steps') + 1] == str(TRAINING_STEPS)
    assert command[command.index('--replay-size') + 1] == str(REPLAY_SIZE)
    assert command[command.index('--vector-envs') + 1] == '1'
    assert command[command.index('--full-checkpoint-freq') + 1] == '250000'
    assert '--checkpoint-replay' in command
    assert resume is None
    assert receipt == tmp_path / 'runs/P47-2028/receipt.json'


def test_ant_reuse_and_terminal_point_failure_are_accepted(tmp_path: Path) -> None:
    evidence = (
        ROOT
        / 'reports/star/shapo_p47_four_env_artifacts/milestones/step_2000000/'
        'ant-2027/artifacts/evaluation_step_2000000.json'
    )
    row, records = validate_evaluation(evidence, 'P47', 2027)
    assert row['evidence_sha256'] == REUSED_EVALUATION_SHA256
    assert row['training_steps'] == TRAINING_STEPS
    assert len(records) == 100

    state = tmp_path / 'point-core-state.json'
    state.write_text(
        json.dumps(
            {
                'state': 'screen_not_passed',
                'protocol_sha256': PARENT_PROTOCOL_SHA256,
                'retraining': False,
            },
        ),
        encoding='utf-8',
    )
    assert wait_for_point_core(state, 1) == 'screen_not_passed'

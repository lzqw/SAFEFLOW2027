"""Targeted checks for the conditional enhanced PointCircle formal queue."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.star.run_pointcircle_enhanced_formal_queue import (
    ADDENDUM_SHA256,
    FORMAL_STEPS,
    PARENT_PROTOCOL_SHA256,
    REPLAY_SIZE,
    training_command,
    validate_terminal_gate,
)


def terminal_payloads(passed: bool) -> tuple[dict, dict]:
    """Build one valid terminal queue state and decision pair."""
    state = {
        'state': 'screen_passed' if passed else 'screen_not_passed',
        'attempt': 1,
        'attempt_limit': 1,
        'scientific_training_launched': True,
        'no_further_pointcircle_method_changes_allowed': True,
    }
    decision = {
        'status': 'complete',
        'environment': 'SafetyPointCircle1-v0',
        'attempt': 1,
        'attempt_limit': 1,
        'parent_protocol_sha256': PARENT_PROTOCOL_SHA256,
        'enhancement_preregistration_sha256': ADDENDUM_SHA256,
        'decision': {'screen_pass': passed},
    }
    return state, decision


@pytest.mark.parametrize('passed', [True, False])
def test_terminal_gate_preserves_frozen_outcome(passed: bool) -> None:
    state, decision = terminal_payloads(passed)
    assert validate_terminal_gate(state, decision) is passed


def test_terminal_gate_rejects_disagreement() -> None:
    state, decision = terminal_payloads(True)
    decision['decision']['screen_pass'] = False
    with pytest.raises(ValueError, match='invalid terminal'):
        validate_terminal_gate(state, decision)


def test_formal_command_locks_budget_resources_and_resume(tmp_path: Path) -> None:
    args = argparse.Namespace(
        source=Path('/source'),
        run_root=tmp_path,
        python_bin=Path('/venv/bin/python'),
    )
    resume = Path('/screen/final_full.pt')
    command, receipt, _ = training_command(args, 'P47', 2027, resume)

    assert command[command.index('--steps') + 1] == str(FORMAL_STEPS)
    assert command[command.index('--vector-envs') + 1] == '1'
    assert command[command.index('--replay-size') + 1] == str(REPLAY_SIZE)
    assert command[command.index('--resume') + 1] == str(resume)
    assert command[command.index('--full-checkpoint-freq') + 1] == '250000'
    assert '--checkpoint-replay' in command
    assert receipt == tmp_path / 'formal/P47-2027/receipt.json'

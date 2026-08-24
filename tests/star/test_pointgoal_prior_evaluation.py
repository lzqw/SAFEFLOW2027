"""Tests for fixed-stream PointGoal prior-baseline evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts.star.evaluate_pointgoal_prior_baselines import (
    evaluate_backend,
    parse_base_seeds,
    tail_mean,
)
from scripts.star.run_pointgoal_prior_postprocessing import (
    evaluation_command,
    formal_receipts,
    wait_for_training,
)


class FakeBackend:
    """Small stochastic policy/environment pair with optional auto-reset."""

    def __init__(self, auto_resets: bool) -> None:
        self.auto_resets = auto_resets
        self.reset_calls: list[int | None] = []
        self.step_index = 0

    def reset(self, seed: int | None) -> torch.Tensor:
        self.reset_calls.append(seed)
        self.step_index = 0
        return torch.tensor([float(seed or 0)])

    def action(self, observation: Any) -> torch.Tensor:
        del observation
        return torch.rand(2)

    def step(
        self,
        action: Any,
    ) -> tuple[torch.Tensor, float, float, bool, bool, dict[str, Any]]:
        self.step_index += 1
        completed = self.step_index == 2
        reward = float(torch.as_tensor(action).sum().item())
        cost = float(torch.as_tensor(action)[0].item() > 0.5)
        info = {'goal_met': completed}
        if completed and self.auto_resets:
            self.step_index = 0
        return torch.tensor([float(self.step_index)]), reward, cost, False, completed, info

    def close(self) -> None:
        return None


def test_fixed_stream_rollouts_are_reproducible_without_mutating_global_rng() -> None:
    first = FakeBackend(auto_resets=False)
    second = FakeBackend(auto_resets=False)
    torch.manual_seed(91)
    preserved_state = torch.random.get_rng_state().clone()

    first_records, first_summary = evaluate_backend(
        first,
        'SACLag',
        2027,
        [11, 12],
        2,
        1_000_003,
        max_episode_steps=2,
    )
    assert torch.equal(torch.random.get_rng_state(), preserved_state)
    second_records, second_summary = evaluate_backend(
        second,
        'SACLag',
        2027,
        [11, 12],
        2,
        1_000_003,
        max_episode_steps=2,
    )

    assert first_records == second_records
    assert first_summary == second_summary
    assert first.reset_calls == [11, None, 12, None]
    assert first_summary['episodes'] == 4
    assert first_summary['success_rate'] == 1.0


def test_auto_reset_backend_advances_without_an_extra_manual_reset() -> None:
    backend = FakeBackend(auto_resets=True)
    records, _ = evaluate_backend(
        backend,
        'SHAPO-PIDLag',
        2027,
        [21],
        3,
        1_000_003,
        max_episode_steps=2,
    )

    assert len(records) == 3
    assert backend.reset_calls == [21]


def test_seed_parser_and_empirical_cvar_are_strict() -> None:
    assert parse_base_seeds('22027,22028,22029') == [22027, 22028, 22029]
    assert tail_mean([float(value) for value in range(1, 11)]) == pytest.approx(9.5)
    with pytest.raises(ValueError, match='unique'):
        parse_base_seeds('22027,22027')
    with pytest.raises(ValueError, match='at least one'):
        tail_mean([])


def test_postprocessor_uses_exact_three_seed_method_receipts(tmp_path: Path) -> None:
    args = argparse.Namespace(
        source=Path('/immutable/source'),
        official_shapo_source=Path('/immutable/SHAPO'),
        output_root=tmp_path / 'postprocess',
        python_bin=Path('/venv/bin/python'),
        training_root=tmp_path / 'training',
    )

    command, output = evaluation_command(args, 'saclag', 'a' * 64)

    expected = formal_receipts(args.training_root, 'saclag')
    receipt_start = command.index('--receipts') + 1
    assert command[receipt_start : receipt_start + 3] == [str(path) for path in expected]
    assert [path.parent.name for path in expected] == ['seed-2027', 'seed-2028', 'seed-2029']
    assert output == args.output_root / 'evaluations' / 'saclag'


def test_postprocessor_accepts_only_authenticated_formal_completion(tmp_path: Path) -> None:
    protocol_sha = 'b' * 64
    (tmp_path / 'queue_state.json').write_text(
        json.dumps(
            {
                'state': 'complete',
                'mode': 'formal',
                'protocol_sha256': protocol_sha,
            },
        ),
        encoding='utf-8',
    )

    wait_for_training(tmp_path, protocol_sha, poll_seconds=1, timeout_seconds=1)

    with pytest.raises(ValueError, match='invalid'):
        wait_for_training(tmp_path, 'c' * 64, poll_seconds=1, timeout_seconds=1)

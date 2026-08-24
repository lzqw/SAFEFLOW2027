"""Tests for fixed-stream PointGoal prior-baseline evaluation."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from scripts.star.evaluate_pointgoal_prior_baselines import (
    evaluate_backend,
    parse_base_seeds,
    tail_mean,
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

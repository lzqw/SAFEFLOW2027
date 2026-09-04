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
"""Unit tests for the P4.2 reward-target timescale diagnostic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from scripts.star.diagnose_reward_target_timescale import (
    reward_target_parameter_gap,
    soft_update_reward_target,
)
from torch import nn


def algorithm_with_weights(target: float, online: float) -> SimpleNamespace:
    """Construct the minimal actor-critic surface used by timescale helpers."""
    target_module = nn.Linear(2, 1, bias=False)
    online_module = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        target_module.weight.fill_(target)
        online_module.weight.fill_(online)
    return SimpleNamespace(
        _actor_critic=SimpleNamespace(
            target_reward_critic=target_module,
            reward_critic=online_module,
        ),
    )


def test_soft_update_reward_target_changes_only_requested_fraction() -> None:
    algorithm = algorithm_with_weights(target=0.0, online=2.0)
    soft_update_reward_target(algorithm, 0.25)
    weight = algorithm._actor_critic.target_reward_critic.weight
    assert weight.detach().tolist() == [[0.5, 0.5]]


def test_zero_coefficient_freezes_target_and_gap_is_scale_relative() -> None:
    algorithm = algorithm_with_weights(target=1.0, online=2.0)
    soft_update_reward_target(algorithm, 0.0)
    gap = reward_target_parameter_gap(algorithm)
    assert algorithm._actor_critic.target_reward_critic.weight.detach().tolist() == [[1.0, 1.0]]
    assert gap['parameter_gap_rms'] == pytest.approx(1.0)
    assert gap['online_parameter_rms'] == pytest.approx(2.0)
    assert gap['relative_parameter_gap'] == pytest.approx(0.5)


@pytest.mark.parametrize('coefficient', [-0.1, 1.1])
def test_soft_update_rejects_invalid_coefficient(coefficient: float) -> None:
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        soft_update_reward_target(algorithm_with_weights(0.0, 1.0), coefficient)

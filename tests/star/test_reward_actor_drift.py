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
"""Unit tests for P4.2 reward actor-drift diagnostic helpers."""

from __future__ import annotations

import pytest
import torch
from scripts.star.diagnose_reward_actor_drift import module_parameter_gap, soft_update_module
from torch import nn


def linear(weight: float) -> nn.Linear:
    """Return a bias-free two-parameter module with one shared value."""
    module = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(weight)
    return module


def test_soft_update_module_and_relative_gap() -> None:
    target = linear(0.0)
    online = linear(2.0)
    soft_update_module(target, online, 0.25)
    assert target.weight.detach().tolist() == [[0.5, 0.5]]
    gap = module_parameter_gap(target, online)
    assert gap['parameter_gap_rms'] == pytest.approx(1.5)
    assert gap['online_parameter_rms'] == pytest.approx(2.0)
    assert gap['relative_parameter_gap'] == pytest.approx(0.75)


@pytest.mark.parametrize('coefficient', [0.0, -0.1, 1.1])
def test_soft_update_module_rejects_invalid_coefficient(coefficient: float) -> None:
    with pytest.raises(ValueError, match=r'\(0, 1\]'):
        soft_update_module(linear(0.0), linear(1.0), coefficient)

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
"""Atomic, resumable STAR training checkpoints."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and all available CUDA generators."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore every generator captured by :func:`capture_rng_state`."""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_star_checkpoint(  # pylint: disable=too-many-arguments
    path: Path | str,
    *,
    model: nn.Module,
    optimizers: dict[str, optim.Optimizer],
    dual_state: dict[str, Any],
    normalizer: nn.Module,
    counters: dict[str, int | float],
    warmup_state: dict[str, Any],
    replay: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomically save all learning state; replay is optional for lightweight milestones."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
    payload = {
        'format': 'star-checkpoint',
        'version': 1,
        'model': model.state_dict(),
        'optimizers': {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        'duals': dual_state,
        'normalizer': normalizer.state_dict(),
        'counters': counters,
        'warmup': warmup_state,
        'rng': capture_rng_state(),
        'replay': replay.state_dict() if replay is not None else None,
        'extra': extra or {},
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_star_checkpoint(  # pylint: disable=too-many-arguments
    path: Path | str,
    *,
    model: nn.Module,
    optimizers: dict[str, optim.Optimizer],
    normalizer: nn.Module,
    replay: Any | None = None,
    map_location: torch.device | str = 'cpu',
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore model, optimizers, duals, normalizer, counters, RNG, and optional replay."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get('format') != 'star-checkpoint' or payload.get('version') != 1:
        raise ValueError('unsupported STAR checkpoint format')
    model.load_state_dict(payload['model'])
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(payload['optimizers'][name])
    normalizer.load_state_dict(payload['normalizer'])
    if replay is not None and payload['replay'] is not None:
        replay.load_state_dict(payload['replay'])
    if restore_rng:
        restore_rng_state(payload['rng'])
    return {
        'duals': payload['duals'],
        'counters': payload['counters'],
        'warmup': payload['warmup'],
        'extra': payload['extra'],
        'has_replay': payload['replay'] is not None,
    }

#!/usr/bin/env python3
"""Reset and step the four STAR P0/P1 environments with their native APIs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import safety_gymnasium


ENVIRONMENTS = (
    ('SafetyPointGoal1-v0', 'safety_gymnasium'),
    ('SafetyPointButton1-v0', 'safety_gymnasium'),
    ('Ant-v4', 'gymnasium'),
    ('Walker2d-v4', 'gymnasium'),
)


def _finite(value: Any, name: str, env_id: str) -> float:
    scalar = float(np.asarray(value).item())
    if not math.isfinite(scalar):
        raise RuntimeError(f'{env_id} returned non-finite {name}: {scalar}')
    return scalar


def _check_observation(obs: Any, env_id: str) -> np.ndarray:
    array = np.asarray(obs)
    if not np.issubdtype(array.dtype, np.number):
        raise RuntimeError(f'{env_id} returned a non-numeric observation: {array.dtype}')
    if not np.isfinite(array).all():
        raise RuntimeError(f'{env_id} returned a non-finite observation')
    return array


def smoke_environment(env_id: str, provider: str, steps: int, seed: int) -> dict[str, Any]:
    """Run exactly ``steps`` native environment transitions."""
    if provider == 'safety_gymnasium':
        env = safety_gymnasium.make(env_id, autoreset=False)
        has_native_cost = True
    else:
        env = gymnasium.make(env_id)
        has_native_cost = False

    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    initial_obs = _check_observation(obs, env_id)
    episodes = 1
    terminations = 0
    truncations = 0
    reward_sum = 0.0
    cost_sum = 0.0
    observed_nonzero_cost = False

    try:
        for _ in range(steps):
            result = env.step(env.action_space.sample())
            if has_native_cost:
                if len(result) != 6:
                    raise RuntimeError(f'{env_id} must return six values, got {len(result)}')
                obs, reward, cost, terminated, truncated, _ = result
                cost_value = _finite(cost, 'cost', env_id)
                cost_sum += cost_value
                observed_nonzero_cost = observed_nonzero_cost or cost_value != 0.0
            else:
                if len(result) != 5:
                    raise RuntimeError(f'{env_id} must return five values, got {len(result)}')
                obs, reward, terminated, truncated, _ = result

            _check_observation(obs, env_id)
            reward_sum += _finite(reward, 'reward', env_id)
            terminated = bool(terminated)
            truncated = bool(truncated)
            terminations += int(terminated)
            truncations += int(truncated)

            if terminated or truncated:
                obs, _ = env.reset()
                _check_observation(obs, env_id)
                episodes += 1
    finally:
        env.close()

    return {
        'env_id': env_id,
        'provider': provider,
        'steps': steps,
        'episodes_started': episodes,
        'terminations': terminations,
        'truncations': truncations,
        'native_cost_channel': has_native_cost,
        'cost_sum': cost_sum if has_native_cost else None,
        'observed_nonzero_cost': observed_nonzero_cost if has_native_cost else None,
        'reward_sum': reward_sum,
        'observation_shape': list(initial_obs.shape),
        'observation_dtype': str(initial_obs.dtype),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('--steps must be positive')

    results = [
        smoke_environment(env_id, provider, args.steps, args.seed + index)
        for index, (env_id, provider) in enumerate(ENVIRONMENTS)
    ]
    payload = {
        'status': 'passed',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'seed': args.seed,
        'gymnasium_version': gymnasium.__version__,
        'safety_gymnasium_version': safety_gymnasium.__version__,
        'results': results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f'{rendered}\n', encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()

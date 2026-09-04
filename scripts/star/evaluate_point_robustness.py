#!/usr/bin/env python3
"""Run checkpoint-only PointGoal or PointCircle robustness evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from scripts.star.evaluate_pointcircle_checkpoints import validate_arm


SUPPORTED_ENVIRONMENTS = ('SafetyPointGoal1-v0', 'SafetyPointCircle1-v0')
TRAINING_SEEDS = (2027, 2028, 2029)
BASE_SEEDS = (33_027, 33_028, 33_029, 33_030, 33_031)
SCREEN_EPISODES_PER_SEED = 10
FINAL_EPISODES_PER_SEED = 20
FLOW_SEED_OFFSET = 1_000_003
ACTION_NOISE_SEED_OFFSET = 2_000_003


@dataclass(frozen=True)
class Condition:
    """One exactly specified checkpoint-only environment/action perturbation."""

    name: str
    family: str
    level: str
    value: Any
    selected_final: bool


def sha256_file(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(source: Path) -> str:
    """Require a clean committed robustness evaluator source."""
    status = subprocess.run(
        ['git', '-C', str(source), 'status', '--porcelain'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError('robustness evaluation requires a clean source')
    return subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def conditions(environment: str) -> list[Condition]:
    """Return the locked robustness conditions for one task."""
    if environment == 'SafetyPointCircle1-v0':
        return [
            Condition(
                name=f'boundary_{multiplier:.1f}',
                family='safety_boundary_multiplier',
                level=f'{multiplier:.1f}',
                value=multiplier,
                selected_final=multiplier in (0.8, 1.2),
            )
            for multiplier in (0.8, 0.9, 1.0, 1.1, 1.2)
        ]
    if environment != 'SafetyPointGoal1-v0':
        raise ValueError(f'unsupported robustness environment {environment!r}')
    return [
        Condition('nominal', 'shared_nominal', 'nominal', None, False),
        Condition('hazard_radius_medium', 'hazard_radius', 'medium', 1.25, False),
        Condition('hazard_radius_strong', 'hazard_radius', 'strong', 1.50, True),
        Condition(
            'hazard_position_offset_medium',
            'hazard_position_offset',
            'medium',
            (0.1, -0.1),
            False,
        ),
        Condition(
            'hazard_position_offset_strong',
            'hazard_position_offset',
            'strong',
            (0.2, -0.2),
            True,
        ),
        Condition('action_noise_medium', 'action_noise', 'medium', 0.05, False),
        Condition('action_noise_strong', 'action_noise', 'strong', 0.10, True),
    ]


def evaluation_task(algorithm: Any) -> Any:
    """Return the native isolated Safety-Gymnasium task."""
    task = algorithm._env._evaluation_task()  # pylint: disable=protected-access
    if task is None:
        raise RuntimeError('robustness evaluator cannot access the native Point task')
    return task


def refreshed_observation(task: Any, template: torch.Tensor) -> torch.Tensor:
    """Recompute the observation after a runtime geometry mutation."""
    value = torch.as_tensor(task.obs(), dtype=template.dtype, device=template.device)
    return value.reshape_as(template)


def episode_transform(condition: Condition) -> Callable[[Any, torch.Tensor, int, int], torch.Tensor]:
    """Build the exact post-reset geometry transform for one condition."""

    def transform(
        algorithm: Any,
        observation: torch.Tensor,
        _base_seed: int,
        _episode_index: int,
    ) -> torch.Tensor:
        task = evaluation_task(algorithm)
        if algorithm._env_id == 'SafetyPointCircle1-v0':  # pylint: disable=protected-access
            task.sigwalls.locate_factor = 1.125 * float(condition.value)
        else:
            hazards = task.hazards
            radius_multiplier = float(condition.value) if condition.family == 'hazard_radius' else 1.0
            hazards.size = 0.2 * radius_multiplier
            for index in range(int(hazards.num)):
                hazards.engine.model.geom(f'hazard{index}').size[0] = hazards.size
            if condition.family == 'hazard_position_offset':
                offset_x, offset_y = condition.value
                for index in range(int(hazards.num)):
                    body = hazards.engine.model.body(f'hazard{index}')
                    body.pos[0] += float(offset_x)
                    body.pos[1] += float(offset_y)
        return refreshed_observation(task, observation)

    return transform


def action_transform(
    condition: Condition,
) -> Callable[[torch.Tensor, int, int, int], torch.Tensor] | None:
    """Build deterministic local-stream execution noise for action-noise conditions."""
    if condition.family != 'action_noise':
        return None
    standard_deviation = float(condition.value)
    generators: dict[int, torch.Generator] = {}

    def transform(
        action: torch.Tensor,
        base_seed: int,
        _episode_index: int,
        _step: int,
    ) -> torch.Tensor:
        if base_seed not in generators:
            generator = torch.Generator(device='cpu')
            generator.manual_seed(ACTION_NOISE_SEED_OFFSET + base_seed)
            generators[base_seed] = generator
        noise = torch.randn(
            action.shape,
            generator=generators[base_seed],
            dtype=action.dtype,
        ).to(device=action.device)
        return (action + standard_deviation * noise).clamp(-1.0, 1.0)

    return transform


def compact(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Drop trajectories while retaining raw episode arrays and geometry."""
    evaluation.pop('trajectory')
    return evaluation


def parser() -> argparse.ArgumentParser:
    """Build robustness-evaluation arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--environment', choices=SUPPORTED_ENVIRONMENTS, required=True)
    result.add_argument('--arm', choices=('C0', 'P47'), required=True)
    result.add_argument('--checkpoint', type=Path, required=True)
    result.add_argument('--protocol', type=Path, required=True)
    result.add_argument('--expected-protocol-sha256', required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--work-dir', type=Path, required=True)
    result.add_argument('--device', default='cuda:0')
    return result


def main() -> None:
    """Evaluate every frozen condition and the preselected 100-episode endpoints."""
    from scripts.star.evaluate_pointgoal_checkpoints import (  # pylint: disable=import-outside-toplevel
        atomic_json,
        build_algorithm,
        checkpoint_payload,
        evaluate_rollouts,
    )

    args = parser().parse_args()
    source = Path(__file__).resolve().parents[2]
    source_commit = git_commit(source)
    if sha256_file(args.protocol) != args.expected_protocol_sha256:
        raise ValueError('robustness protocol SHA-256 mismatch')
    checkpoint = args.checkpoint.resolve()
    payload = checkpoint_payload(checkpoint, args.environment)
    seed = int(payload['extra']['seed'])
    steps = int(payload['counters']['env_steps'])
    expected_steps = 200_000 if args.environment == 'SafetyPointGoal1-v0' else 500_000
    if seed not in TRAINING_SEEDS or steps != expected_steps:
        raise ValueError('robustness checkpoint has the wrong seed or equal-budget endpoint')
    validate_arm(payload, args.arm, checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    algorithm = build_algorithm(
        payload,
        args.device,
        list(BASE_SEEDS),
        args.work_dir,
        episodes_per_seed=FINAL_EPISODES_PER_SEED,
        environment=args.environment,
    )
    condition_results: list[dict[str, Any]] = []
    try:
        for condition in conditions(args.environment):
            screening = compact(
                evaluate_rollouts(
                    algorithm,
                    steps,
                    list(BASE_SEEDS),
                    'fixed-random-flow',
                    episodes_per_seed=SCREEN_EPISODES_PER_SEED,
                    flow_seed_offset=FLOW_SEED_OFFSET,
                    episode_start_transform=episode_transform(condition),
                    action_transform=action_transform(condition),
                )
            )
            selected_final = None
            if condition.selected_final:
                selected_final = compact(
                    evaluate_rollouts(
                        algorithm,
                        steps,
                        list(BASE_SEEDS),
                        'fixed-random-flow',
                        episodes_per_seed=FINAL_EPISODES_PER_SEED,
                        flow_seed_offset=FLOW_SEED_OFFSET,
                        episode_start_transform=episode_transform(condition),
                        action_transform=action_transform(condition),
                    )
                )
            condition_results.append(
                {
                    'name': condition.name,
                    'family': condition.family,
                    'level': condition.level,
                    'value': condition.value,
                    'screening': screening,
                    'selected_final': selected_final,
                }
            )
    finally:
        algorithm._logger.close()  # pylint: disable=protected-access
        algorithm._env.close()  # pylint: disable=protected-access
    result = {
        'status': 'complete',
        'environment': args.environment,
        'arm': args.arm,
        'training_seed': seed,
        'training_steps': steps,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256_file(checkpoint),
        'evaluation_source_commit': source_commit,
        'robustness_protocol': str(args.protocol.resolve()),
        'robustness_protocol_sha256': args.expected_protocol_sha256,
        'base_seeds': list(BASE_SEEDS),
        'screening_episodes_per_condition': len(BASE_SEEDS) * SCREEN_EPISODES_PER_SEED,
        'selected_final_episodes_per_condition': len(BASE_SEEDS) * FINAL_EPISODES_PER_SEED,
        'retraining': False,
        'conditions': condition_results,
    }
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                'status': 'complete',
                'environment': args.environment,
                'arm': args.arm,
                'training_seed': seed,
                'conditions': len(condition_results),
                'output': str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    main()

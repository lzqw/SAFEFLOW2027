#!/usr/bin/env python3
"""Evaluate one PointCircle mechanism/parameter checkpoint on the frozen 5x20 stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ENVIRONMENT = 'SafetyPointCircle1-v0'
TRAINING_SEED = 2027
TRAINING_STEPS = 300_000
BASE_SEEDS = (22_027, 22_028, 22_029, 22_030, 22_031)
EPISODES_PER_BASE_SEED = 20
FLOW_SEED_OFFSET = 1_000_003
VARIANTS = {
    'Iso-High': {
        'directional_loss_enabled': False,
        'safety_aware_cost_geometry_enabled': False,
        'random_anisotropy_enabled': False,
        'flow_reference_variance': 0.04,
    },
    'Iso-Low': {
        'directional_loss_enabled': False,
        'safety_aware_cost_geometry_enabled': False,
        'random_anisotropy_enabled': False,
        'flow_reference_variance': 0.004,
        'flow_normal_variance_ratio': 1.0,
    },
    'Random-Aniso': {
        'directional_loss_enabled': True,
        'safety_aware_cost_geometry_enabled': True,
        'random_anisotropy_enabled': True,
        'random_anisotropy_seed': 130027,
        'flow_normal_variance_ratio': 0.10,
    },
    'Safety-Aniso': {
        'directional_loss_enabled': True,
        'safety_aware_cost_geometry_enabled': True,
        'random_anisotropy_enabled': False,
        'flow_normal_variance_ratio': 0.10,
    },
    'Safety-Aniso-0.25': {
        'directional_loss_enabled': True,
        'safety_aware_cost_geometry_enabled': True,
        'random_anisotropy_enabled': False,
        'flow_normal_variance_ratio': 0.25,
    },
}


def sha256_file(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(source: Path) -> str:
    """Require a clean committed evaluation source."""
    status = subprocess.run(
        ['git', '-C', str(source), 'status', '--porcelain'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError('PointCircle control evaluation requires a clean source')
    return subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def resolved_algo(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the self-describing checkpoint algorithm configuration."""
    config = payload.get('extra', {}).get('resolved_config')
    if not isinstance(config, dict) or not isinstance(config.get('algo_cfgs'), dict):
        raise ValueError('control checkpoint lacks a resolved algorithm configuration')
    return config['algo_cfgs']


def validate_variant(payload: dict[str, Any], variant: str, checkpoint: Path) -> None:
    """Require every defining mechanism switch for one named variant."""
    algo = resolved_algo(payload)
    for key, expected in VARIANTS[variant].items():
        observed = algo.get(key, False if key == 'random_anisotropy_enabled' else None)
        if isinstance(expected, float):
            valid = abs(float(observed) - expected) <= 1e-12
        else:
            valid = observed == expected
        if not valid:
            raise ValueError(f'{checkpoint}: {variant}.{key}={observed!r}, expected {expected!r}')


def parser() -> argparse.ArgumentParser:
    """Build control-evaluation arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--variant', choices=tuple(VARIANTS), required=True)
    result.add_argument('--checkpoint', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--work-dir', type=Path, required=True)
    result.add_argument('--device', default='cuda:0')
    return result


def main() -> None:
    """Evaluate one exact control checkpoint and persist raw evidence."""
    from scripts.star.evaluate_pointgoal_checkpoints import (  # pylint: disable=import-outside-toplevel
        atomic_json,
        build_algorithm,
        checkpoint_payload,
        evaluate_rollouts,
    )

    args = parser().parse_args()
    source = Path(__file__).resolve().parents[2]
    source_commit = git_commit(source)
    checkpoint = args.checkpoint.resolve()
    payload = checkpoint_payload(checkpoint, ENVIRONMENT)
    seed = int(payload['extra']['seed'])
    steps = int(payload['counters']['env_steps'])
    if seed != TRAINING_SEED or steps != TRAINING_STEPS:
        raise ValueError('mechanism/parameter controls require seed 2027 at exactly 300k')
    validate_variant(payload, args.variant, checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    algorithm = build_algorithm(
        payload,
        args.device,
        list(BASE_SEEDS),
        args.work_dir,
        episodes_per_seed=EPISODES_PER_BASE_SEED,
        environment=ENVIRONMENT,
    )
    try:
        evaluation = evaluate_rollouts(
            algorithm,
            steps,
            list(BASE_SEEDS),
            'fixed-random-flow',
            episodes_per_seed=EPISODES_PER_BASE_SEED,
            flow_seed_offset=FLOW_SEED_OFFSET,
        )
    finally:
        algorithm._logger.close()  # pylint: disable=protected-access
        algorithm._env.close()  # pylint: disable=protected-access
    evaluation.pop('trajectory')
    result = {
        'status': 'complete',
        'variant': args.variant,
        'environment': ENVIRONMENT,
        'training_seed': seed,
        'training_steps': steps,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256_file(checkpoint),
        'evaluation_source_commit': source_commit,
        'evaluation_device': args.device,
        **evaluation,
    }
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                'status': 'complete',
                'variant': args.variant,
                'episodes': result['episodes'],
                'output': str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    main()

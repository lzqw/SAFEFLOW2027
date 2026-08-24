#!/usr/bin/env python3
"""Evaluate final PointCircle STAR checkpoints on the frozen 5x20 stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ENVIRONMENT = 'SafetyPointCircle1-v0'
TRAINING_SEEDS = (2027, 2028, 2029)
BASE_SEEDS = (22_027, 22_028, 22_029, 22_030, 22_031)
EPISODES_PER_BASE_SEED = 20
FINAL_STEPS = 500_000
DEFAULT_FLOW_SEED_OFFSET = 1_000_003
ARM_FLAGS = {
    'C0': (False, False),
    'P47': (True, True),
}


def sha256_file(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(source: Path) -> str:
    """Require a clean committed evaluator source."""
    status = subprocess.run(
        ['git', '-C', str(source), 'status', '--porcelain'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError('formal PointCircle evaluation requires a clean source')
    return subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_arm(payload: dict[str, Any], arm: str, checkpoint: Path) -> None:
    """Require the two preregistered arm switches to match the requested arm."""
    config = payload.get('extra', {}).get('resolved_config')
    if not isinstance(config, dict):
        from omnisafe.utils.star_validation import (  # pylint: disable=import-outside-toplevel
            checkpoint_reconstruction_config,
        )

        config = checkpoint_reconstruction_config(payload)
    algo = config.get('algo_cfgs', {})
    observed = (
        bool(algo.get('directional_loss_enabled', False)),
        bool(algo.get('safety_aware_cost_geometry_enabled', False)),
    )
    if observed != ARM_FLAGS[arm]:
        raise ValueError(f'{checkpoint} has arm switches {observed}, expected {ARM_FLAGS[arm]}')


def evaluate_one(
    checkpoint: Path,
    arm: str,
    output_dir: Path,
    work_dir: Path,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one exact 500k checkpoint and persist raw episode evidence."""
    from scripts.star.evaluate_pointgoal_checkpoints import (  # pylint: disable=import-outside-toplevel
        atomic_csv,
        atomic_json,
        build_algorithm,
        checkpoint_payload,
        evaluate_rollouts,
        public_row,
    )

    payload = checkpoint_payload(checkpoint, ENVIRONMENT)
    step = int(payload['counters']['env_steps'])
    seed = int(payload['extra']['seed'])
    if step != FINAL_STEPS or seed not in TRAINING_SEEDS:
        raise ValueError(f'{checkpoint} is not a final PointCircle 500k/three-seed checkpoint')
    validate_arm(payload, arm, checkpoint)
    algorithm = build_algorithm(
        payload,
        device,
        list(BASE_SEEDS),
        work_dir / f'{arm}-seed-{seed}-step-{step}',
        episodes_per_seed=EPISODES_PER_BASE_SEED,
        environment=ENVIRONMENT,
    )
    try:
        evaluation = evaluate_rollouts(
            algorithm,
            step,
            list(BASE_SEEDS),
            'fixed-random-flow',
            episodes_per_seed=EPISODES_PER_BASE_SEED,
            flow_seed_offset=DEFAULT_FLOW_SEED_OFFSET,
        )
    finally:
        algorithm._logger.close()  # pylint: disable=protected-access
        algorithm._env.close()  # pylint: disable=protected-access
    trajectory = evaluation.pop('trajectory')
    result = {
        'status': 'complete',
        'arm': arm,
        'training_seed': seed,
        'environment': ENVIRONMENT,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256_file(checkpoint),
        'evaluation_device': device,
        **evaluation,
    }
    atomic_json(output_dir / f'evaluation_seed_{seed}_step_{step}.json', result)
    if trajectory:
        atomic_csv(output_dir / f'trajectory_seed_{seed}_step_{step}.csv', trajectory)
    row = {'arm': arm, **public_row(payload, checkpoint, evaluation, device)}
    return result, row


def parser() -> argparse.ArgumentParser:
    """Build final PointCircle evaluation arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--arm', choices=tuple(ARM_FLAGS), required=True)
    result.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument('--work-dir', type=Path, required=True)
    result.add_argument('--device', default='cuda:0')
    return result


def main() -> None:
    """Evaluate the exact arm matrix and write its authenticated manifest."""
    from scripts.star.evaluate_pointgoal_checkpoints import (  # pylint: disable=import-outside-toplevel
        atomic_csv,
        atomic_json,
    )

    args = parser().parse_args()
    source = Path(__file__).resolve().parents[2]
    source_commit = git_commit(source)
    if len(args.checkpoints) != len(TRAINING_SEEDS):
        raise ValueError('one arm requires exactly three final checkpoints')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    results_and_rows = [
        evaluate_one(
            checkpoint.resolve(),
            args.arm,
            args.output_dir,
            args.work_dir,
            args.device,
        )
        for checkpoint in args.checkpoints
    ]
    results = [item[0] for item in results_and_rows]
    rows = [item[1] for item in results_and_rows]
    if sorted(result['training_seed'] for result in results) != list(TRAINING_SEEDS):
        raise ValueError('checkpoints do not form the exact three-seed arm')
    rows.sort(key=lambda row: row['training_seed'])
    atomic_csv(args.output_dir / 'pointcircle_checkpoint_evaluation.csv', rows)
    manifest = {
        'status': 'complete',
        'environment': ENVIRONMENT,
        'arm': args.arm,
        'training_steps': FINAL_STEPS,
        'training_seeds': list(TRAINING_SEEDS),
        'base_seeds': list(BASE_SEEDS),
        'episodes_per_base_seed': EPISODES_PER_BASE_SEED,
        'episodes_per_checkpoint': len(BASE_SEEDS) * EPISODES_PER_BASE_SEED,
        'evaluation_mode': 'fixed-random-flow',
        'evaluation_source_commit': source_commit,
        'results': [
            {
                'training_seed': result['training_seed'],
                'path': str(
                    args.output_dir
                    / f'evaluation_seed_{result["training_seed"]}_step_{FINAL_STEPS}.json'
                ),
                'checkpoint_sha256': result['checkpoint_sha256'],
            }
            for result in sorted(results, key=lambda item: item['training_seed'])
        ],
    }
    atomic_json(args.output_dir / 'evaluation_manifest.json', manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

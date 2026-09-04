#!/usr/bin/env python3
"""Materialize one sealed SHAPO P4.7 environment config from the frozen PointGoal method."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
PROCEDURAL_ALGO_KEYS = frozenset(
    {
        'safety_protocol_version',
        'safety_protocol_sha256',
        'eval_seeds',
        'eval_episodes_per_seed',
        'evaluate_initial_policy',
        'trajectory_eval_seeds',
        'eval_milestones',
        'checkpoint_milestones',
        'checkpoint_replay',
        'milestone_checkpoints',
        'full_checkpoint_freq',
        'resume_from',
    },
)


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest for one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_method_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Remove only environment and preregistered procedural fields from a P4.7 config."""
    custom = copy.deepcopy(config['custom_cfgs'])
    custom.pop('train_cfgs', None)
    custom.pop('lagrange_cfgs', None)
    algo = custom.get('algo_cfgs', {})
    for key in PROCEDURAL_ALGO_KEYS:
        algo.pop(key, None)
    return custom


def canonical_method_sha256(config: dict[str, Any]) -> str:
    """Fingerprint the frozen method independently of environment/run logistics."""
    payload = json.dumps(
        canonical_method_payload(config),
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} must contain a YAML mapping')
    return payload


def resolve_from_root(value: str) -> Path:
    """Resolve a preregistered repository-relative path."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_seals(matrix: dict[str, Any]) -> tuple[Path, Path]:
    """Verify protocol and frozen-config byte seals before materialization."""
    protocol_path = resolve_from_root(str(matrix['protocol_file']))
    frozen_path = resolve_from_root(str(matrix['frozen_method_config']))
    expected = {
        protocol_path: str(matrix['protocol_sha256']),
        frozen_path: str(matrix['frozen_method_config_sha256']),
    }
    for path, digest in expected.items():
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f'SHA-256 mismatch for {path}: expected {digest}, got {actual}')
    return protocol_path, frozen_path


def materialize(
    matrix_path: Path,
    env_id: str,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Return one formal config, or a clearly labelled non-scientific engineering variant."""
    matrix = load_yaml(matrix_path)
    protocol_path, frozen_path = verify_seals(matrix)
    environments = matrix.get('environments', {})
    if env_id not in environments:
        raise ValueError(f'environment {env_id!r} is not in the locked SHAPO matrix')
    frozen = load_yaml(frozen_path)
    expected_method_hash = str(matrix['frozen_method_canonical_sha256'])
    actual_method_hash = canonical_method_sha256(frozen)
    if actual_method_hash != expected_method_hash:
        raise ValueError(
            f'frozen P4.7 method mismatch: expected {expected_method_hash}, got {actual_method_hash}',
        )

    environment = environments[env_id]
    training_invariants = matrix['training_invariants']
    evaluation = matrix['evaluation']
    checkpoint = matrix['checkpoint']
    base_seeds = [int(seed) for seed in evaluation['base_seeds']]
    episodes_per_seed = int(evaluation['episodes_per_base_seed'])
    milestones = [int(step) for step in environment['milestones']]
    total_steps = int(environment['total_steps'])
    if milestones != sorted(set(milestones)) or milestones[-1] != total_steps:
        raise ValueError(f'{env_id} milestones must be unique, sorted, and end at total_steps')

    resolved = copy.deepcopy(frozen)
    resolved['env_id'] = env_id
    custom = resolved['custom_cfgs']
    custom.setdefault('train_cfgs', {}).update(
        {
            'total_steps': total_steps,
            'eval_episodes': len(base_seeds) * episodes_per_seed,
        },
    )
    custom.setdefault('algo_cfgs', {}).update(
        {
            'safety_protocol_version': str(matrix['protocol_version']),
            'safety_protocol_sha256': str(matrix['protocol_sha256']),
            'eval_seeds': base_seeds,
            'eval_episodes_per_seed': episodes_per_seed,
            'evaluate_initial_policy': bool(evaluation['evaluate_initial_policy']),
            'trajectory_eval_seeds': [int(seed) for seed in evaluation['trajectory_base_seeds']],
            'eval_milestones': list(milestones),
            'checkpoint_milestones': list(milestones),
            'checkpoint_replay': bool(checkpoint['checkpoint_replay']),
            'milestone_checkpoints': bool(checkpoint['milestone_checkpoints']),
            'full_checkpoint_freq': int(checkpoint['latest_full_frequency_steps']),
            'resume_from': '',
        },
    )
    custom.setdefault('lagrange_cfgs', {})['cost_limit'] = float(environment['cost_limit'])
    resolved['shapo_four_env_metadata'] = {
        'protocol_version': str(matrix['protocol_version']),
        'protocol_sha256': str(matrix['protocol_sha256']),
        'protocol_file': str(protocol_path.relative_to(ROOT)),
        'matrix_file': str(matrix_path.resolve().relative_to(ROOT)),
        'matrix_sha256': sha256_file(matrix_path),
        'frozen_method_config': str(frozen_path.relative_to(ROOT)),
        'frozen_method_config_sha256': str(matrix['frozen_method_config_sha256']),
        'frozen_method_canonical_sha256': expected_method_hash,
        'geometry_mode': str(environment['geometry_mode']),
        'cost_normalization_scale': float(environment['cost_normalization_scale']),
        'formal_cost_budget': float(environment['cost_limit']),
        'selected_P47_training_invariants': copy.deepcopy(training_invariants),
        'training_seed_is_statistical_unit': True,
        'scientific_early_stop': False,
        'engineering_smoke_only': smoke,
    }
    if canonical_method_sha256(resolved) != expected_method_hash:
        raise RuntimeError('formal materialization changed a frozen P4.7 method field')

    if smoke:
        smoke_steps = 32
        custom['train_cfgs'].update({'total_steps': smoke_steps, 'eval_episodes': 1})
        custom['algo_cfgs'].update(
            {
                'max_episode_steps': 25,
                'random_steps': 4,
                'terminal_utility_steps': 8,
                'geometry_min_steps': 12,
                'eval_seeds': [42027],
                'eval_episodes_per_seed': 1,
                'trajectory_eval_seeds': [42027],
                'eval_milestones': [smoke_steps],
                'checkpoint_milestones': [smoke_steps],
                'checkpoint_replay': False,
                'full_checkpoint_freq': 0,
            },
        )
        resolved['shapo_four_env_metadata']['smoke_schedule_overrides'] = {
            'total_steps': smoke_steps,
            'max_episode_steps': 25,
            'warmup_steps': [4, 8, 12],
            'evaluation_episodes': 1,
        }
    return resolved


def parser() -> argparse.ArgumentParser:
    """Build the deterministic config-materialization CLI."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        '--matrix',
        type=Path,
        default=ROOT / 'configs/star/shapo_p47_four_env.yaml',
    )
    result.add_argument('--env-id', required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--smoke', action='store_true')
    return result


def main() -> None:
    """Materialize and atomically write a sealed run config."""
    args = parser().parse_args()
    resolved = materialize(args.matrix.resolve(), args.env_id, smoke=args.smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f'.{args.output.name}.{os.getpid()}.tmp')
    temporary.write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        encoding='utf-8',
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                'environment': args.env_id,
                'engineering_smoke_only': args.smoke,
                'output': str(args.output),
                'output_sha256': sha256_file(args.output),
                'frozen_method_canonical_sha256': resolved['shapo_four_env_metadata'][
                    'frozen_method_canonical_sha256'
                ],
                'resolved_method_canonical_sha256': canonical_method_sha256(resolved),
            },
            indent=2,
            sort_keys=True,
        ),
    )


if __name__ == '__main__':
    main()

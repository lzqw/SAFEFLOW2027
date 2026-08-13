#!/usr/bin/env python3
"""Validate one host-side SHAPO P4.7 recovery checkpoint without using metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.star.materialize_shapo_p47_config import materialize  # noqa: E402


REQUIRED_CHECKPOINT_FIELDS = {
    'format',
    'version',
    'model',
    'optimizers',
    'duals',
    'normalizer',
    'counters',
    'warmup',
    'rng',
    'replay',
    'extra',
}
SLUGS = {
    'SafetyPointGoal1-v0': 'point-goal',
    'SafetyPointButton1-v0': 'point-button',
    'Ant-v4': 'ant',
    'Walker2d-v4': 'walker2d',
}
REQUIRED_OPTIMIZERS = {
    'actor',
    'reward_critic',
    'cost_critic',
    'violation_critic',
    'energy_dual',
}
REQUIRED_DUALS = {'energy', 'global'}
REQUIRED_RNG = {'python', 'numpy', 'torch', 'cuda'}
REQUIRED_COUNTERS = {
    'env_steps',
    'updates',
    'actor_updates',
    'reward_critic_updates',
    'utd_carry',
    'violations',
    'cumulative_training_cost',
}
REQUIRED_REPLAY_FIELDS = {
    'version',
    'capacity',
    'obs_dim',
    'act_dim',
    'context_dim',
    'size',
    'pointer',
    'data',
    'episode_slots',
    'active_episode_ids',
    'next_episode_id',
    'heldout_episode_period',
}
REQUIRED_REPLAY_DATA = {
    'obs',
    'action',
    'reward',
    'cost',
    'terminated',
    'truncated',
    'reward_done',
    'cost_done',
    'next_obs',
    'context',
    'next_context',
    'risk_threshold',
    'next_risk_threshold',
    'empirical_cost_to_go',
    'violation_label',
    'boundary_score',
    'is_boundary',
    'is_failure',
    'source_random',
    'episode_id',
    'is_heldout',
    'category',
    'generation',
}
REQUIRED_WARMUP_FIELDS = {
    'stage',
    'geometry_enabled',
    'geometry_activation_step',
    'geometry_eligibility_step',
    'geometry_last_disable_step',
    'consecutive_passes',
    'consecutive_failures',
    'gate_evaluations',
    'ema',
    'latest',
}
REQUIRED_EXTRA_FIELDS = {
    'environment',
    'seed',
    'resolved_config',
    'rollout_state',
    'completed_milestones',
    'completed_checkpoint_milestones',
    'last_heldout_evidence',
    'last_heldout_evidence_status',
    'reward_td_diagnostics',
    'best_safe_checkpoint_selection',
}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def nested_subset(observed: Any, expected: Any, *, label: str) -> None:
    """Require every frozen expected leaf while permitting runtime-added fields."""
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            raise ValueError(f'{label} must be a mapping')
        for key, value in expected.items():
            if key not in observed:
                raise ValueError(f'{label} is missing frozen key {key!r}')
            nested_subset(observed[key], value, label=f'{label}.{key}')
        return
    if observed != expected:
        raise ValueError(f'{label} changed from {expected!r} to {observed!r}')


def load_matrix(path: Path) -> dict[str, Any]:
    """Load the frozen matrix as a mapping."""
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or not isinstance(payload.get('environments'), dict):
        raise ValueError('frozen SHAPO matrix is invalid')
    return payload


def validate_resume_path(value: Any, *, remote_run_root: Path, job: str) -> str:
    """Allow only the empty initial value or a recovery file inside the same run."""
    observed = str(value or '')
    if not observed:
        return observed
    candidate = Path(observed)
    allowed_root = (remote_run_root / 'runs' / job / 'training').resolve()
    if not candidate.is_absolute() or '..' in candidate.parts:
        raise ValueError('checkpoint resume_from is not an absolute safe path')
    try:
        candidate.resolve().relative_to(allowed_root)
    except ValueError as error:
        raise ValueError('checkpoint resume_from escaped its formal run') from error
    if candidate.name != 'latest_full.pt' or candidate.parent.name != 'checkpoints':
        raise ValueError('checkpoint resume_from is not a latest_full recovery file')
    return observed


def positive_integer(value: Any, *, label: str) -> int:
    """Parse one strictly positive integral counter."""
    if isinstance(value, bool):
        raise ValueError(f'{label} must be a positive integer')
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{label} must be a positive integer') from error
    if parsed <= 0 or not math.isfinite(numeric) or numeric != parsed:
        raise ValueError(f'{label} must be a positive integer')
    return parsed


def validate_checkpoint_backup(  # pylint: disable=too-many-locals
    checkpoint: Path,
    *,
    matrix_path: Path,
    environment: str,
    seed: int,
    job: str,
    remote_run_root: Path,
    source_commit: str,
) -> dict[str, Any]:
    """Prove that a copied checkpoint contains complete frozen recovery state."""
    matrix = load_matrix(matrix_path)
    if environment not in matrix['environments']:
        raise ValueError(f'environment {environment!r} is outside the frozen matrix')
    if seed not in {int(value) for value in matrix.get('training_seeds', [])}:
        raise ValueError(f'training seed {seed} is outside the frozen matrix')
    expected_job = f'{SLUGS.get(environment, "")}-{seed}'
    if not SLUGS.get(environment) or job != expected_job:
        raise ValueError(f'backup job {job!r} does not match {environment} seed {seed}')
    if len(source_commit) != 40 or any(value not in '0123456789abcdef' for value in source_commit):
        raise ValueError('source commit must be a full lowercase Git identity')
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise ValueError('checkpoint backup must be one regular non-symbolic-link file')

    try:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    except Exception as error:
        raise ValueError('checkpoint backup cannot be loaded on CPU') from error
    if not isinstance(payload, dict) or set(payload) != REQUIRED_CHECKPOINT_FIELDS:
        raise ValueError('checkpoint backup fields changed')
    if payload.get('format') != 'star-checkpoint' or payload.get('version') != 1:
        raise ValueError('checkpoint backup has an unsupported STAR format')
    if not isinstance(payload.get('model'), dict) or not payload['model']:
        raise ValueError('checkpoint backup model state is empty')
    optimizers = payload.get('optimizers')
    if not isinstance(optimizers, dict) or set(optimizers) != REQUIRED_OPTIMIZERS:
        raise ValueError('checkpoint backup optimizer state changed')
    if any(
        not isinstance(value, dict) or set(value) != {'state', 'param_groups'}
        for value in optimizers.values()
    ):
        raise ValueError('checkpoint backup optimizer payload is invalid')
    if not isinstance(payload.get('duals'), dict) or set(payload['duals']) != REQUIRED_DUALS:
        raise ValueError('checkpoint backup dual state changed')
    if not isinstance(payload.get('normalizer'), dict) or not payload['normalizer']:
        raise ValueError('checkpoint backup normalizer state is empty')
    if (
        not isinstance(payload.get('warmup'), dict)
        or set(payload['warmup']) != REQUIRED_WARMUP_FIELDS
        or payload['warmup'].get('stage') not in {'W1', 'W2', 'W3'}
    ):
        raise ValueError('checkpoint backup warmup state is empty')
    if not isinstance(payload.get('rng'), dict) or set(payload['rng']) != REQUIRED_RNG:
        raise ValueError('checkpoint backup RNG state changed')

    counters = payload.get('counters')
    if not isinstance(counters, dict) or set(counters) != REQUIRED_COUNTERS:
        raise ValueError('checkpoint backup counters are invalid')
    checkpoint_steps = positive_integer(counters.get('env_steps'), label='checkpoint env_steps')
    environment_spec = matrix['environments'][environment]
    total_steps = int(environment_spec['total_steps'])
    frequency = int(matrix.get('checkpoint', {}).get('latest_full_frequency_steps', 0))
    if frequency <= 0 or checkpoint_steps > total_steps or checkpoint_steps % frequency != 0:
        raise ValueError('checkpoint backup step is outside the frozen recovery schedule')

    replay = payload.get('replay')
    expected_capacity = int(matrix.get('training_invariants', {}).get('replay_size', 0))
    if not isinstance(replay, dict) or set(replay) != REQUIRED_REPLAY_FIELDS:
        raise ValueError('checkpoint backup lacks replay recovery state')
    if int(replay.get('capacity', -1)) != expected_capacity:
        raise ValueError('checkpoint backup replay capacity changed')
    if int(replay.get('size', -1)) != min(checkpoint_steps, expected_capacity):
        raise ValueError('checkpoint backup replay size is inconsistent with its step')
    if int(replay.get('version', -1)) != 3:
        raise ValueError('checkpoint backup replay version changed')
    replay_size = int(replay['size'])
    replay_pointer = int(replay.get('pointer', -1))
    if replay_pointer < 0 or replay_pointer >= expected_capacity:
        raise ValueError('checkpoint backup replay pointer is invalid')
    replay_data = replay.get('data')
    if not isinstance(replay_data, dict) or set(replay_data) != REQUIRED_REPLAY_DATA:
        raise ValueError('checkpoint backup replay tensor fields changed')
    if any(not isinstance(value, torch.Tensor) for value in replay_data.values()):
        raise ValueError('checkpoint backup replay contains a non-tensor field')
    expected_valid_rows = min(replay_size, expected_capacity)
    if any(
        value.ndim == 0 or int(value.shape[0]) != expected_valid_rows
        for value in replay_data.values()
    ):
        raise ValueError('checkpoint backup replay tensor length is inconsistent')

    extra = payload.get('extra')
    if (
        not isinstance(extra, dict)
        or set(extra) != REQUIRED_EXTRA_FIELDS
        or not isinstance(extra.get('resolved_config'), dict)
    ):
        raise ValueError('checkpoint backup resolved configuration is invalid')
    observed_config = extra['resolved_config']
    if (
        extra.get('environment') != environment
        or int(extra.get('seed', -1)) != seed
        or observed_config.get('env_id') != environment
        or int(observed_config.get('seed', -1)) != seed
    ):
        raise ValueError('checkpoint backup belongs to a different formal run')

    expected_config = materialize(matrix_path.resolve(), environment, smoke=False)['custom_cfgs']
    observed_algo = observed_config.get('algo_cfgs')
    if not isinstance(observed_algo, dict):
        raise ValueError('checkpoint backup algorithm configuration is invalid')
    observed_resume = validate_resume_path(
        observed_algo.get('resume_from', ''),
        remote_run_root=remote_run_root,
        job=job,
    )
    expected_config['algo_cfgs']['resume_from'] = observed_resume
    nested_subset(observed_config, expected_config, label=f'{job}.resolved_config')

    receipt = {
        'validator_version': 'shapo-p47-checkpoint-backup-validator-v1',
        'status': 'passed',
        'scientific_result': False,
        'used_for_checkpoint_or_configuration_selection': False,
        'training_state_changed': False,
        'scientific_protocol_changed': False,
        'job': job,
        'environment': environment,
        'training_seed': seed,
        'checkpoint_env_steps': checkpoint_steps,
        'total_steps': total_steps,
        'checkpoint_frequency_steps': frequency,
        'checkpoint_format': 'star-checkpoint-v1',
        'has_replay': True,
        'replay_capacity': expected_capacity,
        'replay_size': int(replay['size']),
        'checkpoint_bytes': checkpoint.stat().st_size,
        'checkpoint_sha256': sha256_file(checkpoint),
        'matrix_sha256': sha256_file(matrix_path),
        'source_commit': source_commit,
        'inspected_scientific_metrics': False,
    }
    del payload
    return receipt


def main() -> None:
    """Validate one copied checkpoint and print a strict machine receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--environment', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--remote-run-root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    receipt = validate_checkpoint_backup(
        args.checkpoint.resolve(),
        matrix_path=args.matrix.resolve(),
        environment=args.environment,
        seed=args.seed,
        job=args.job,
        remote_run_root=args.remote_run_root,
        source_commit=args.source_commit,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()

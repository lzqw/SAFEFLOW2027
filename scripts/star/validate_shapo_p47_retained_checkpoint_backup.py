#!/usr/bin/env python3
"""Validate one privately retained SHAPO checkpoint without using outcome metrics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.star.materialize_shapo_p47_config import materialize  # noqa: E402
from scripts.star.validate_shapo_p47_checkpoint_backup import (  # noqa: E402
    REQUIRED_CHECKPOINT_FIELDS,
    REQUIRED_COUNTERS,
    REQUIRED_DUALS,
    REQUIRED_EXTRA_FIELDS,
    REQUIRED_OPTIMIZERS,
    REQUIRED_REPLAY_DATA,
    REQUIRED_REPLAY_FIELDS,
    REQUIRED_RNG,
    REQUIRED_WARMUP_FIELDS,
    SLUGS,
    load_matrix,
    nested_subset,
    positive_integer,
    sha256_file,
    validate_resume_path,
)


def finite_nonnegative(value: Any, *, label: str) -> float:
    """Parse one finite nonnegative counter without interpreting it scientifically."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{label} must be finite and nonnegative') from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f'{label} must be finite and nonnegative')
    return parsed


def validate_retained_checkpoint_backup(  # pylint: disable=too-many-arguments,too-many-locals
    checkpoint: Path,
    *,
    matrix_path: Path,
    environment: str,
    seed: int,
    job: str,
    label: str,
    expected_steps: set[int],
    replay_required: bool,
    remote_run_root: Path,
    source_commit: str,
) -> dict[str, Any]:
    """Prove one milestone, best, or final checkpoint has the frozen semantic identity."""
    matrix = load_matrix(matrix_path)
    if environment not in matrix['environments']:
        raise ValueError(f'environment {environment!r} is outside the frozen matrix')
    if seed not in {int(value) for value in matrix.get('training_seeds', [])}:
        raise ValueError(f'training seed {seed} is outside the frozen matrix')
    expected_job = f'{SLUGS.get(environment, "")}-{seed}'
    if not SLUGS.get(environment) or job != expected_job:
        raise ValueError(f'retained job {job!r} does not match {environment} seed {seed}')
    if not label or not expected_steps or any(step < 0 for step in expected_steps):
        raise ValueError('retained checkpoint label or expected steps are invalid')
    if len(source_commit) != 40 or any(value not in '0123456789abcdef' for value in source_commit):
        raise ValueError('source commit must be a full lowercase Git identity')
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise ValueError('retained checkpoint must be one regular non-symbolic-link file')

    try:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    except Exception as error:
        raise ValueError('retained checkpoint cannot be loaded on CPU') from error
    if not isinstance(payload, dict) or set(payload) != REQUIRED_CHECKPOINT_FIELDS:
        raise ValueError('retained checkpoint fields changed')
    if payload.get('format') != 'star-checkpoint' or payload.get('version') != 1:
        raise ValueError('retained checkpoint has an unsupported STAR format')
    if not isinstance(payload.get('model'), dict) or not payload['model']:
        raise ValueError('retained checkpoint model state is empty')
    optimizers = payload.get('optimizers')
    if not isinstance(optimizers, dict) or set(optimizers) != REQUIRED_OPTIMIZERS:
        raise ValueError('retained checkpoint optimizer state changed')
    if any(
        not isinstance(value, dict) or set(value) != {'state', 'param_groups'}
        for value in optimizers.values()
    ):
        raise ValueError('retained checkpoint optimizer payload is invalid')
    if not isinstance(payload.get('duals'), dict) or set(payload['duals']) != REQUIRED_DUALS:
        raise ValueError('retained checkpoint dual state changed')
    if not isinstance(payload.get('normalizer'), dict) or not payload['normalizer']:
        raise ValueError('retained checkpoint normalizer state is empty')
    if (
        not isinstance(payload.get('warmup'), dict)
        or set(payload['warmup']) != REQUIRED_WARMUP_FIELDS
        or payload['warmup'].get('stage') not in {'W1', 'W2', 'W3'}
    ):
        raise ValueError('retained checkpoint warmup state changed')
    if not isinstance(payload.get('rng'), dict) or set(payload['rng']) != REQUIRED_RNG:
        raise ValueError('retained checkpoint RNG state changed')

    counters = payload.get('counters')
    if not isinstance(counters, dict) or set(counters) != REQUIRED_COUNTERS:
        raise ValueError('retained checkpoint counters are invalid')
    checkpoint_steps = positive_integer(counters.get('env_steps'), label='checkpoint env_steps')
    if checkpoint_steps not in expected_steps:
        raise ValueError(
            f'retained checkpoint step {checkpoint_steps} is outside {sorted(expected_steps)}',
        )
    cumulative_training_cost = finite_nonnegative(
        counters.get('cumulative_training_cost'),
        label='checkpoint cumulative training cost',
    )
    violations = finite_nonnegative(counters.get('violations'), label='checkpoint violations')

    replay = payload.get('replay')
    expected_capacity = int(matrix.get('training_invariants', {}).get('replay_size', 0))
    replay_size = 0
    replay_version = None
    if replay_required:
        if not isinstance(replay, dict) or set(replay) != REQUIRED_REPLAY_FIELDS:
            raise ValueError('retained final checkpoint lacks replay recovery state')
        if int(replay.get('capacity', -1)) != expected_capacity:
            raise ValueError('retained final checkpoint replay capacity changed')
        for dimension in ('obs_dim', 'act_dim', 'context_dim'):
            if int(replay.get(dimension, 0)) <= 0:
                raise ValueError(f'retained final checkpoint replay {dimension} is invalid')
        replay_size = int(replay.get('size', -1))
        if replay_size != min(checkpoint_steps, expected_capacity):
            raise ValueError('retained final checkpoint replay size is inconsistent')
        replay_version = int(replay.get('version', -1))
        if replay_version != 3:
            raise ValueError('retained final checkpoint replay version changed')
        if int(replay.get('pointer', -1)) != checkpoint_steps % expected_capacity:
            raise ValueError('retained final checkpoint replay pointer is inconsistent')
        if int(replay.get('heldout_episode_period', -1)) <= 0:
            raise ValueError('retained final checkpoint heldout replay schedule is invalid')
        if int(replay.get('next_episode_id', -1)) < 0:
            raise ValueError('retained final checkpoint next replay episode is invalid')
        if not isinstance(replay.get('episode_slots'), list) or not isinstance(
            replay.get('active_episode_ids'),
            list,
        ):
            raise ValueError('retained final checkpoint replay episode state is invalid')
        replay_data = replay.get('data')
        if not isinstance(replay_data, dict) or set(replay_data) != REQUIRED_REPLAY_DATA:
            raise ValueError('retained final checkpoint replay tensor fields changed')
        if any(
            not isinstance(value, torch.Tensor)
            or value.ndim == 0
            or int(value.shape[0]) != replay_size
            for value in replay_data.values()
        ):
            raise ValueError('retained final checkpoint replay tensor length is inconsistent')
    elif replay is not None:
        raise ValueError('retained lightweight checkpoint unexpectedly contains replay')

    extra = payload.get('extra')
    if (
        not isinstance(extra, dict)
        or set(extra) != REQUIRED_EXTRA_FIELDS
        or not isinstance(extra.get('resolved_config'), dict)
    ):
        raise ValueError('retained checkpoint resolved configuration is invalid')
    observed_config = extra['resolved_config']
    if (
        extra.get('environment') != environment
        or int(extra.get('seed', -1)) != seed
        or observed_config.get('env_id') != environment
        or int(observed_config.get('seed', -1)) != seed
    ):
        raise ValueError('retained checkpoint belongs to a different formal run')
    expected_config = materialize(matrix_path.resolve(), environment, smoke=False)['custom_cfgs']
    observed_algo = observed_config.get('algo_cfgs')
    if not isinstance(observed_algo, dict):
        raise ValueError('retained checkpoint algorithm configuration is invalid')
    observed_resume = validate_resume_path(
        observed_algo.get('resume_from', ''),
        remote_run_root=remote_run_root,
        job=job,
    )
    expected_config['algo_cfgs']['resume_from'] = observed_resume
    nested_subset(observed_config, expected_config, label=f'{job}.resolved_config')

    receipt = {
        'validator_version': 'shapo-p47-retained-checkpoint-validator-v1',
        'status': 'passed',
        'scientific_result': False,
        'used_for_checkpoint_or_configuration_selection': False,
        'used_for_preregistered_checkpoint_retention': True,
        'scientific_metric_values_used_for_decision': False,
        'scientific_metric_values_reported': False,
        'checkpoint_counters_validated': True,
        'training_state_changed': False,
        'scientific_protocol_changed': False,
        'job': job,
        'environment': environment,
        'training_seed': seed,
        'label': label,
        'checkpoint_env_steps': checkpoint_steps,
        'expected_steps': sorted(expected_steps),
        'checkpoint_format': 'star-checkpoint-v1',
        'has_replay': replay_required,
        'replay_capacity': expected_capacity,
        'replay_size': replay_size,
        'replay_version': replay_version,
        'checkpoint_bytes': checkpoint.stat().st_size,
        'checkpoint_sha256': sha256_file(checkpoint),
        'matrix_sha256': sha256_file(matrix_path),
        'source_commit': source_commit,
        'resume_from': observed_resume or None,
        'counter_integrity': {
            'cumulative_training_cost_finite_nonnegative': True,
            'violation_count_finite_nonnegative': True,
        },
    }
    del payload, cumulative_training_cost, violations
    return receipt


def main() -> None:
    """Validate one private retained checkpoint and print a strict machine receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--environment', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--expected-steps', required=True)
    parser.add_argument('--replay-required', action='store_true')
    parser.add_argument('--remote-run-root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    expected_steps = {int(value) for value in args.expected_steps.split(',') if value.strip()}
    receipt = validate_retained_checkpoint_backup(
        args.checkpoint.resolve(),
        matrix_path=args.matrix.resolve(),
        environment=args.environment,
        seed=args.seed,
        job=args.job,
        label=args.label,
        expected_steps=expected_steps,
        replay_required=args.replay_required,
        remote_run_root=args.remote_run_root,
        source_commit=args.source_commit,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()

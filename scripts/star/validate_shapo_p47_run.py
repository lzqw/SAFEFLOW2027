#!/usr/bin/env python3
"""Validate one completed formal SHAPO P4.7 training-seed run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from scripts.star.materialize_shapo_p47_config import materialize


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON mapping."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} must contain a JSON mapping')
    return payload


def finite(label: str, value: Any) -> float:
    """Require and return one finite metric."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f'{label} is not finite: {value!r}')
    return numeric


def require_close(label: str, observed: Any, expected: float) -> float:
    """Require one finite scalar to agree with its independently recomputed value."""
    numeric = finite(label, observed)
    if not math.isclose(numeric, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f'{label} mismatch: expected {expected!r}, got {numeric!r}')
    return numeric


def finite_sequence(label: str, values: Any, length: int) -> list[float]:
    """Require one fixed-length sequence of finite scalars."""
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f'{label} must contain exactly {length} values')
    return [finite(f'{label}[{index}]', value) for index, value in enumerate(values)]


def undefined(value: Any) -> bool:
    """Return whether an optional diagnostic is explicitly absent or non-finite."""
    if value is None:
        return True
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def require_nested_subset(label: str, observed: Any, expected: Any) -> None:
    """Require every recursively nested expected config value in a runtime payload."""
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            raise ValueError(f'{label} must be a mapping')
        for key, value in expected.items():
            if key not in observed:
                raise ValueError(f'{label} is missing key {key!r}')
            require_nested_subset(f'{label}.{key}', observed[key], value)
        return
    if observed != expected:
        raise ValueError(f'{label} mismatch: expected {expected!r}, got {observed!r}')


def newest(container: Path, pattern: str) -> Path:
    """Return the newest matching segment artifact."""
    matches = list(container.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'{container} has no artifact matching {pattern!r}')
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 file digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(path: Path) -> dict[str, Any]:
    """Return the immutable path, byte size, and digest of one compact artifact."""
    return {
        'artifact': str(path),
        'sha256': sha256_file(path),
        'bytes': path.stat().st_size,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write the validated compact result."""
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(f'{json.dumps(payload, indent=2, sort_keys=True)}\n', encoding='utf-8')
    os.replace(temporary, path)


TRAJECTORY_REQUIRED_FIELDS = frozenset(
    {
        'milestone',
        'base_seed',
        'episode_index',
        'step',
        'x',
        'y',
        'predicted_violation_probability',
        'predicted_cost_to_go',
        'geometry_budget_ratio',
        'cost_margin',
        'cost_normal_0',
        'cost_normal_1',
        'cost',
        'is_failure',
    },
)


def validate_trajectory_artifact(
    path: Path,
    milestone: int,
    expected_seeds: list[int],
) -> dict[str, Any]:
    """Validate one fixed-seed trajectory artifact and return its portable identity."""
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = TRAJECTORY_REQUIRED_FIELDS - fields
        if missing:
            raise ValueError(f'{path} is missing trajectory fields {sorted(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path} contains no trajectory rows')

    expected = set(expected_seeds)
    rows_per_seed = dict.fromkeys(expected_seeds, 0)
    steps_per_seed: dict[int, list[int]] = {seed: [] for seed in expected_seeds}
    for row_index, row in enumerate(rows):
        row_milestone = int(float(row['milestone']))
        base_seed = int(float(row['base_seed']))
        episode_index = int(float(row['episode_index']))
        step = int(float(row['step']))
        if row_milestone != milestone:
            raise ValueError(f'{path} row {row_index} has milestone {row_milestone}')
        if base_seed not in expected:
            raise ValueError(f'{path} row {row_index} has unexpected base seed {base_seed}')
        if episode_index != 0:
            raise ValueError(f'{path} row {row_index} is not the locked first episode')
        for field in (
            'x',
            'y',
            'predicted_violation_probability',
            'predicted_cost_to_go',
            'geometry_budget_ratio',
            'cost_margin',
            'cost_normal_0',
            'cost_normal_1',
            'cost',
            'is_failure',
        ):
            finite(f'{path} row {row_index} {field}', row[field])
        probability = float(row['predicted_violation_probability'])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f'{path} row {row_index} has invalid violation probability')
        if float(row['cost']) < 0.0:
            raise ValueError(f'{path} row {row_index} has negative cost')
        if int(float(row['is_failure'])) not in {0, 1}:
            raise ValueError(f'{path} row {row_index} has non-binary failure flag')
        rows_per_seed[base_seed] += 1
        steps_per_seed[base_seed].append(step)

    observed = {seed for seed, count in rows_per_seed.items() if count}
    if observed != expected:
        raise ValueError(
            f'{path} trajectory seed coverage mismatch: expected {sorted(expected)}, '
            f'got {sorted(observed)}',
        )
    for seed, steps in steps_per_seed.items():
        if steps != list(range(len(steps))):
            raise ValueError(f'{path} base seed {seed} has a non-contiguous step trace')
    return {
        'milestone': milestone,
        'artifact': str(path),
        'sha256': sha256_file(path),
        'row_count': len(rows),
        'trajectory_seeds': sorted(expected),
        'rows_per_seed': {str(seed): rows_per_seed[seed] for seed in sorted(expected)},
    }


def validate_checkpoint_artifact(  # pylint: disable=too-many-arguments
    path: Path,
    *,
    label: str,
    environment: str,
    seed: int,
    expected_steps: set[int],
    replay_required: bool,
    replay_capacity: int,
    expected_custom_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deeply validate one STAR checkpoint and return its immutable compact identity."""
    if path.stat().st_size <= 0:
        raise ValueError(f'checkpoint {path} is empty')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    required = {
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
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f'{path} is not a complete STAR checkpoint payload')
    if payload['format'] != 'star-checkpoint' or int(payload['version']) != 1:
        raise ValueError(f'{path} has an unsupported STAR checkpoint format')
    if not isinstance(payload['model'], dict) or not payload['model']:
        raise ValueError(f'{path} has no model state')
    if not isinstance(payload['optimizers'], dict) or not payload['optimizers']:
        raise ValueError(f'{path} has no optimizer state')
    counters = payload['counters']
    step = int(counters['env_steps'])
    if step not in expected_steps:
        raise ValueError(f'{path} has unexpected checkpoint step {step}')
    finite(f'{path} cumulative training cost', counters['cumulative_training_cost'])
    extra = payload['extra']
    if extra.get('environment') != environment or int(extra.get('seed', -1)) != seed:
        raise ValueError(f'{path} checkpoint identity does not match its matrix cell')
    resolved = extra.get('resolved_config', {})
    if resolved.get('env_id') != environment or int(resolved.get('seed', -1)) != seed:
        raise ValueError(f'{path} embedded resolved config has the wrong identity')
    if expected_custom_config is not None:
        require_nested_subset(
            f'{path} embedded resolved config',
            resolved,
            expected_custom_config,
        )

    replay = payload['replay']
    replay_metadata: dict[str, Any] = {
        'replay_included': replay is not None,
        'replay_size': 0,
        'replay_capacity': replay_capacity,
    }
    if replay_required:
        if not isinstance(replay, dict):
            raise ValueError(f'{path} is not a replay-bearing recovery checkpoint')
        if int(replay.get('version', -1)) not in {2, 3}:
            raise ValueError(f'{path} has an unsupported replay snapshot')
        if int(replay['capacity']) != replay_capacity:
            raise ValueError(f'{path} replay capacity changed')
        expected_size = min(step, replay_capacity)
        if int(replay['size']) != expected_size:
            raise ValueError(f'{path} replay size does not match its training step')
        if int(replay['pointer']) != step % replay_capacity:
            raise ValueError(f'{path} replay pointer does not match its training step')
        data = replay.get('data', {})
        required_replay_fields = {
            'obs',
            'action',
            'cost',
            'next_obs',
            'context',
            'next_context',
            'is_failure',
        }
        if not isinstance(data, dict) or not required_replay_fields.issubset(data):
            raise ValueError(f'{path} replay snapshot is missing transition fields')
        for field, tensor in data.items():
            if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != expected_size:
                raise ValueError(f'{path} replay field {field!r} has the wrong resident length')
        replay_metadata.update(
            {
                'replay_size': expected_size,
                'replay_capacity': int(replay['capacity']),
                'replay_version': int(replay['version']),
                'observation_dimension': int(replay['obs_dim']),
                'action_dimension': int(replay['act_dim']),
            },
        )
    elif replay is not None:
        raise ValueError(f'{path} lightweight checkpoint unexpectedly contains replay')

    return {
        'label': label,
        'artifact': str(path),
        'sha256': sha256_file(path),
        'bytes': path.stat().st_size,
        'env_steps': step,
        **replay_metadata,
    }


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Validate source/config identity, full horizon, evaluations, metrics, and checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--env-id', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()

    matrix = yaml.safe_load(args.matrix.read_text(encoding='utf-8'))
    if args.env_id not in matrix.get('environments', {}):
        raise ValueError(f'environment {args.env_id!r} is not in the locked matrix')
    if args.seed not in [int(seed) for seed in matrix.get('training_seeds', [])]:
        raise ValueError(f'training seed {args.seed} is not in the locked matrix')
    environment = matrix['environments'][args.env_id]
    total_steps = int(environment['total_steps'])
    milestones = [int(step) for step in environment['milestones']]
    geometry_mode = str(environment['geometry_mode'])
    geometry_scale = float(environment['cost_normalization_scale'])
    formal_budget = float(environment['cost_limit'])
    base_seeds = [int(seed) for seed in matrix['evaluation']['base_seeds']]
    trajectory_seeds = [int(seed) for seed in matrix['evaluation']['trajectory_base_seeds']]
    episodes_per_seed = int(matrix['evaluation']['episodes_per_base_seed'])
    expected_episodes = len(base_seeds) * episodes_per_seed
    expected_schedule = [
        (base_seed, episode_index)
        for base_seed in base_seeds
        for episode_index in range(episodes_per_seed)
    ]

    config_path = args.container / 'resolved_formal_config.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    expected_config = materialize(args.matrix.resolve(), args.env_id)
    if config != expected_config:
        raise ValueError('resolved formal config differs from the sealed materialization')
    metadata = config['shapo_four_env_metadata']
    if metadata['engineering_smoke_only'] is not False:
        raise ValueError('formal run config is labelled as an engineering smoke')
    if metadata['geometry_mode'] != geometry_mode:
        raise ValueError('formal run config geometry mode does not match the matrix')
    if float(metadata['cost_normalization_scale']) != geometry_scale:
        raise ValueError('formal run config geometry scale does not match the matrix')
    if float(metadata['formal_cost_budget']) != formal_budget:
        raise ValueError('formal run config cost budget does not match the matrix')

    receipt = load_json(args.container / 'receipt.json')
    expected_receipt = {
        'status': 'passed',
        'environment': args.env_id,
        'seed': args.seed,
        'steps': total_steps,
        'source_commit': args.source_commit,
        'source_dirty': False,
        'config_sha256': sha256_file(config_path),
    }
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            raise ValueError(
                f'receipt field {key!r} mismatch: expected {expected!r}, got {receipt.get(key)!r}',
            )
    if int(receipt['run_steps']) <= 0:
        raise ValueError('formal receipt has no run steps')

    final_log_dir = Path(receipt['log_dir'])
    summary = load_json(final_log_dir / 'artifacts/training_summary.json')
    if summary.get('status') != 'completed' or int(summary.get('env_steps', -1)) != total_steps:
        raise ValueError('formal training summary is incomplete')
    if summary.get('engineering_hard_stop_reason') is not None:
        raise ValueError('formal training summary contains an engineering hard stop')
    if summary.get('cost_geometry_mode') != geometry_mode:
        raise ValueError('formal training summary geometry mode is incorrect')
    finite('training cumulative cost', summary['cumulative_training_cost'])
    finite('end-to-end FPS', receipt['end_to_end_fps'])
    finite('elapsed seconds', receipt['elapsed_seconds'])

    evaluation_rows = []
    trajectory_artifacts = []
    for milestone in milestones:
        path = newest(
            args.container,
            f'training/**/artifacts/evaluation_step_{milestone}.json',
        )
        evaluation = load_json(path)
        if int(evaluation['episodes']) != expected_episodes:
            raise ValueError(f'{path} has the wrong episode count')
        if evaluation['base_seeds'] != base_seeds:
            raise ValueError(f'{path} has the wrong held-out base seeds')
        if int(evaluation['episodes_per_base_seed']) != episodes_per_seed:
            raise ValueError(f'{path} has the wrong episodes-per-seed count')
        observed_schedule = list(
            zip(
                [int(seed) for seed in evaluation.get('seeds', [])],
                [int(index) for index in evaluation.get('episode_indices', [])],
            ),
        )
        if observed_schedule != expected_schedule:
            raise ValueError(f'{path} has the wrong sequential fixed-seed schedule')
        if [int(seed) for seed in evaluation['trajectory_seeds']] != trajectory_seeds:
            raise ValueError(f'{path} has the wrong trajectory seed schedule')
        returns = finite_sequence(f'{path} returns', evaluation.get('returns'), expected_episodes)
        costs = finite_sequence(f'{path} costs', evaluation.get('costs'), expected_episodes)
        lengths = finite_sequence(f'{path} lengths', evaluation.get('lengths'), expected_episodes)
        if any(cost < 0.0 for cost in costs):
            raise ValueError(f'{path} contains a negative episode cost')
        if any(length <= 0.0 or not length.is_integer() for length in lengths):
            raise ValueError(f'{path} contains an invalid episode length')
        require_close(
            f'{args.env_id}@{milestone} mean_return',
            evaluation['mean_return'],
            math.fsum(returns) / expected_episodes,
        )
        require_close(
            f'{args.env_id}@{milestone} mean_cost',
            evaluation['mean_cost'],
            math.fsum(costs) / expected_episodes,
        )
        tail_count = max(1, math.ceil(expected_episodes * 0.2))
        require_close(
            f'{args.env_id}@{milestone} cost_cvar_20',
            evaluation['cost_cvar_20'],
            math.fsum(sorted(costs, reverse=True)[:tail_count]) / tail_count,
        )
        finite(
            f'{args.env_id}@{milestone} training_cumulative_cost',
            evaluation['training_cumulative_cost'],
        )

        failure_flags = finite_sequence(
            f'{path} episode_failure_flags',
            evaluation.get('episode_failure_flags'),
            expected_episodes,
        )
        if any(flag not in {0.0, 1.0} for flag in failure_flags):
            raise ValueError(f'{path} contains a non-binary episode failure flag')
        failure_count = int(math.fsum(failure_flags))
        if int(evaluation['failure_count']) != failure_count:
            raise ValueError(f'{path} failure count disagrees with episode flags')
        require_close(
            f'{path} failure_rate',
            evaluation['failure_rate'],
            failure_count / expected_episodes,
        )
        end_reasons = evaluation.get('episode_end_reasons')
        if not isinstance(end_reasons, list) or len(end_reasons) != expected_episodes:
            raise ValueError(f'{path} has incomplete episode end reasons')
        observed_reason_counts = evaluation.get('failure_reason_counts')
        if not isinstance(observed_reason_counts, dict) or {
            str(reason): int(count) for reason, count in observed_reason_counts.items()
        } != dict(Counter(str(reason) for reason in end_reasons)):
            raise ValueError(f'{path} failure reason counts disagree with episode reasons')
        time_limit_count = sum(str(reason) == 'time_limit' for reason in end_reasons)
        if int(evaluation['time_limit_count']) != time_limit_count:
            raise ValueError(f'{path} time-limit count disagrees with episode reasons')
        require_close(
            f'{path} time_limit_rate',
            evaluation['time_limit_rate'],
            time_limit_count / expected_episodes,
        )

        goal_counts = finite_sequence(
            f'{path} goal_counts',
            evaluation.get('goal_counts'),
            expected_episodes,
        )
        if any(goal < 0.0 or not goal.is_integer() for goal in goal_counts):
            raise ValueError(f'{path} contains an invalid goal count')
        successful_episodes = sum(goal > 0.0 for goal in goal_counts)
        require_close(
            f'{path} success_rate',
            evaluation['success_rate'],
            successful_episodes / expected_episodes,
        )
        path_lengths = finite_sequence(
            f'{path} path_lengths',
            evaluation.get('path_lengths'),
            expected_episodes,
        )
        if any(value < 0.0 for value in path_lengths):
            raise ValueError(f'{path} contains a negative path length')
        require_close(
            f'{path} mean_path_length',
            evaluation['mean_path_length'],
            math.fsum(path_lengths) / expected_episodes,
        )
        if successful_episodes:
            require_close(
                f'{path} cost_per_successful_episode',
                evaluation['cost_per_successful_episode'],
                math.fsum(costs) / successful_episodes,
            )
        elif not undefined(evaluation['cost_per_successful_episode']):
            raise ValueError(f'{path} must leave cost per success undefined without successes')
        high_risk = evaluation['direction_geometry_high_risk_top_20_percent']
        finite(
            f'{args.env_id}@{milestone} high-risk V_N/V_T',
            high_risk['normal_to_tangent_variance_ratio'],
        )
        finite(
            f'{args.env_id}@{milestone} high-risk outward velocity',
            high_risk['dangerous_outward_velocity'],
        )
        if geometry_mode == 'future_failure_probability':
            if any(cost not in {0.0, 1.0} for cost in costs):
                raise ValueError(f'{path} locomotion costs are not binary terminal labels')
            if any(cost != flag for cost, flag in zip(costs, failure_flags)):
                raise ValueError(f'{path} locomotion costs disagree with failure flags')
            failure_times = finite_sequence(
                f'{path} failure_times',
                evaluation.get('failure_times'),
                failure_count,
            )
            failure_steps = finite_sequence(
                f'{path} failure_steps',
                evaluation.get('failure_steps'),
                failure_count,
            )
            failure_heights = finite_sequence(
                f'{path} failure_heights',
                evaluation.get('failure_heights'),
                failure_count,
            )
            failure_angles = finite_sequence(
                f'{path} failure_torso_angles',
                evaluation.get('failure_torso_angles'),
                failure_count,
            )
            for label, values, mean_key in (
                ('failure time', failure_times, 'mean_failure_time'),
                ('failure step', failure_steps, 'mean_failure_step'),
                ('failure height', failure_heights, 'mean_failure_height'),
                ('failure torso angle', failure_angles, 'mean_failure_torso_angle'),
            ):
                if values:
                    require_close(
                        f'{path} {label}',
                        evaluation[mean_key],
                        math.fsum(values) / len(values),
                    )
                elif not undefined(evaluation[mean_key]):
                    raise ValueError(f'{path} must leave {label} undefined without failures')
            if not math.isclose(
                float(evaluation['mean_cost']),
                float(evaluation['failure_rate']),
                abs_tol=1e-12,
            ):
                raise ValueError(f'{path} violates binary-terminal-cost semantics')
            if int(evaluation['failure_count']) != round(
                float(evaluation['failure_rate']) * expected_episodes,
            ):
                raise ValueError(f'{path} failure count and rate disagree')
        evaluation_rows.append(
            {
                'milestone': milestone,
                'artifact': str(path),
                'artifact_sha256': sha256_file(path),
                'artifact_bytes': path.stat().st_size,
                'episodes': expected_episodes,
                'mean_return': evaluation['mean_return'],
                'mean_cost': evaluation['mean_cost'],
                'cost_cvar_20': evaluation['cost_cvar_20'],
                'training_cumulative_cost': evaluation['training_cumulative_cost'],
                'success_rate': evaluation['success_rate'],
                'mean_path_length': evaluation['mean_path_length'],
                'cost_per_successful_episode': evaluation['cost_per_successful_episode'],
                'failure_count': evaluation['failure_count'],
                'failure_rate': evaluation['failure_rate'],
                'failure_reason_counts': evaluation['failure_reason_counts'],
                'mean_failure_time': evaluation['mean_failure_time'],
                'mean_failure_step': evaluation['mean_failure_step'],
                'mean_failure_height': evaluation['mean_failure_height'],
                'mean_failure_torso_angle': evaluation['mean_failure_torso_angle'],
                'time_limit_count': evaluation['time_limit_count'],
                'time_limit_rate': evaluation['time_limit_rate'],
                'high_risk_vn_vt': high_risk['normal_to_tangent_variance_ratio'],
                'high_risk_dangerous_outward_velocity': high_risk['dangerous_outward_velocity'],
            },
        )
        trajectory_path = newest(
            args.container,
            f'training/**/artifacts/trajectory_step_{milestone}.csv',
        )
        trajectory_artifacts.append(
            validate_trajectory_artifact(trajectory_path, milestone, trajectory_seeds),
        )

    cumulative_costs = [
        finite(
            f'{args.env_id}@{row["milestone"]} training cumulative cost',
            row['training_cumulative_cost'],
        )
        for row in evaluation_rows
    ]
    if any(later < earlier for earlier, later in zip(cumulative_costs, cumulative_costs[1:])):
        raise ValueError('training cumulative cost decreased between evaluation milestones')
    require_close(
        'final training cumulative cost',
        cumulative_costs[-1],
        finite('training summary cumulative cost', summary['cumulative_training_cost']),
    )

    manifest_path = newest(args.container, 'training/**/artifacts/run_manifest.json')
    manifest = load_json(manifest_path)
    geometry_identity = manifest['p4_5_unified_cost_geometry']
    if geometry_identity['geometry_mode'] != geometry_mode:
        raise ValueError('run manifest geometry mode is incorrect')
    if float(geometry_identity['cost_normalization_scale']) != geometry_scale:
        raise ValueError('run manifest geometry normalization is incorrect')
    if float(geometry_identity['formal_cost_budget']) != formal_budget:
        raise ValueError('run manifest formal budget is incorrect')
    if manifest['p4_7_safety_aware_geometry']['enabled'] is not True:
        raise ValueError('run manifest does not identify frozen P4.7')

    replay_capacity = int(matrix['training_invariants']['replay_size'])
    required_checkpoint_specs = [
        ('latest_full', 'training/**/checkpoints/latest_full.pt', {total_steps}, True),
        ('final_full', 'training/**/checkpoints/final_full.pt', {total_steps}, True),
        ('best_safe_return', 'training/**/checkpoints/best_safe_return.pt', set(milestones), False),
        *[
            (
                f'milestone_{step}',
                f'training/**/checkpoints/milestone_{step}.pt',
                {step},
                False,
            )
            for step in milestones
        ],
    ]
    checkpoint_paths = []
    checkpoint_artifacts = []
    for label, pattern, expected_steps, replay_required in required_checkpoint_specs:
        path = newest(args.container, pattern)
        checkpoint_paths.append(str(path))
        checkpoint_artifacts.append(
            validate_checkpoint_artifact(
                path,
                label=label,
                environment=args.env_id,
                seed=args.seed,
                expected_steps=expected_steps,
                replay_required=replay_required,
                replay_capacity=replay_capacity,
                expected_custom_config=config['custom_cfgs'],
            ),
        )

    payload = {
        'status': 'passed',
        'protocol_version': matrix['protocol_version'],
        'protocol_sha256': matrix['protocol_sha256'],
        'matrix_sha256': sha256_file(args.matrix),
        'frozen_method_canonical_sha256': matrix['frozen_method_canonical_sha256'],
        'source_commit': args.source_commit,
        'environment': args.env_id,
        'training_seed': args.seed,
        'training_seed_is_statistical_unit': True,
        'total_steps': total_steps,
        'formal_cost_budget': formal_budget,
        'geometry_mode': geometry_mode,
        'cost_normalization_scale': geometry_scale,
        'evaluation_rows': evaluation_rows,
        'trajectory_artifacts': trajectory_artifacts,
        'checkpoints': checkpoint_paths,
        'checkpoint_artifacts': checkpoint_artifacts,
        'receipt': str(args.container / 'receipt.json'),
        'compact_provenance_artifacts': {
            'resolved_config': artifact_identity(config_path),
            'receipt': artifact_identity(args.container / 'receipt.json'),
            'training_summary': artifact_identity(
                final_log_dir / 'artifacts/training_summary.json',
            ),
            'run_manifest': artifact_identity(manifest_path),
        },
    }
    atomic_json(args.container / 'formal_validation.json', payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

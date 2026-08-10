#!/usr/bin/env python3
"""Validate one completed formal SHAPO P4.7 training-seed run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml


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


def newest(container: Path, pattern: str) -> Path:
    """Return the newest matching segment artifact."""
    matches = list(container.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'{container} has no artifact matching {pattern!r}')
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write the validated compact result."""
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(f'{json.dumps(payload, indent=2, sort_keys=True)}\n', encoding='utf-8')
    os.replace(temporary, path)


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
    environment = matrix['environments'][args.env_id]
    total_steps = int(environment['total_steps'])
    milestones = [int(step) for step in environment['milestones']]
    geometry_mode = str(environment['geometry_mode'])
    geometry_scale = float(environment['cost_normalization_scale'])
    formal_budget = float(environment['cost_limit'])
    base_seeds = [int(seed) for seed in matrix['evaluation']['base_seeds']]
    episodes_per_seed = int(matrix['evaluation']['episodes_per_base_seed'])
    expected_episodes = len(base_seeds) * episodes_per_seed

    config_path = args.container / 'resolved_formal_config.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
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
        for metric in ('mean_return', 'mean_cost', 'cost_cvar_20', 'training_cumulative_cost'):
            finite(f'{args.env_id}@{milestone} {metric}', evaluation[metric])
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
                'mean_return': evaluation['mean_return'],
                'mean_cost': evaluation['mean_cost'],
                'cost_cvar_20': evaluation['cost_cvar_20'],
                'training_cumulative_cost': evaluation['training_cumulative_cost'],
                'success_rate': evaluation['success_rate'],
                'failure_count': evaluation['failure_count'],
                'failure_rate': evaluation['failure_rate'],
                'high_risk_vn_vt': high_risk['normal_to_tangent_variance_ratio'],
                'high_risk_dangerous_outward_velocity': high_risk['dangerous_outward_velocity'],
            },
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

    required_checkpoint_patterns = [
        'training/**/checkpoints/latest_full.pt',
        'training/**/checkpoints/final_full.pt',
        'training/**/checkpoints/best_safe_return.pt',
        *[f'training/**/checkpoints/milestone_{step}.pt' for step in milestones],
    ]
    checkpoint_paths = []
    for pattern in required_checkpoint_patterns:
        path = newest(args.container, pattern)
        if path.stat().st_size <= 0:
            raise ValueError(f'checkpoint {path} is empty')
        checkpoint_paths.append(str(path))

    payload = {
        'status': 'passed',
        'protocol_version': matrix['protocol_version'],
        'protocol_sha256': matrix['protocol_sha256'],
        'source_commit': args.source_commit,
        'environment': args.env_id,
        'training_seed': args.seed,
        'training_seed_is_statistical_unit': True,
        'total_steps': total_steps,
        'formal_cost_budget': formal_budget,
        'geometry_mode': geometry_mode,
        'cost_normalization_scale': geometry_scale,
        'evaluation_rows': evaluation_rows,
        'checkpoints': checkpoint_paths,
        'receipt': str(args.container / 'receipt.json'),
    }
    atomic_json(args.container / 'formal_validation.json', payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Build the four-method PointGoal table from sealed and newly evaluated evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any


ENVIRONMENT = 'SafetyPointGoal1-v0'
TRAINING_SEEDS = (2027, 2028, 2029)
BASE_SEEDS = (22_027, 22_028, 22_029, 22_030, 22_031)
EPISODES_PER_BASE_SEED = 20
EPISODES_PER_TRAINING_SEED = 100
METHOD_ORDER = ('C0 / isotropic FLAC', 'SACLag', 'SHAPO-PIDLag', 'P4.7 / Ours')
P4_ARM_TO_METHOD = {
    'C0': 'C0 / isotropic FLAC',
    'P47': 'P4.7 / Ours',
}
PRIOR_METHODS = ('SACLag', 'SHAPO-PIDLag')


def sha256_file(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON through an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one non-empty homogeneous CSV through an atomic rename."""
    if not rows:
        raise ValueError('cannot write an empty comparison table')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    """Write UTF-8 text through an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(content, encoding='utf-8')
    os.replace(temporary, path)


def tail_mean(values: list[float], fraction: float = 0.2) -> float:
    """Return empirical upper-tail cost mean."""
    if not values:
        raise ValueError('CVaR requires at least one episode')
    count = max(1, math.ceil(len(values) * fraction))
    return statistics.fmean(sorted(values, reverse=True)[:count])


def sample_std(values: list[float]) -> float:
    """Return sample standard deviation, defined as zero for one value."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def finite_or_none(value: Any) -> float | None:
    """Return a finite float or JSON-compatible null."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def require_close(observed: Any, expected: float, label: str) -> None:
    """Require a stored summary to agree with its raw episode evidence."""
    if not math.isclose(float(observed), expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f'{label}: stored {observed} disagrees with raw {expected}')


def validate_evaluation_schedule(payload: dict[str, Any], label: str) -> None:
    """Require the frozen 5x20 final-evaluation schedule."""
    if payload.get('environment') != ENVIRONMENT:
        raise ValueError(f'{label}: wrong environment')
    if [int(seed) for seed in payload.get('base_seeds', [])] != list(BASE_SEEDS):
        raise ValueError(f'{label}: wrong base seeds')
    if int(payload.get('episodes_per_base_seed', -1)) != EPISODES_PER_BASE_SEED:
        raise ValueError(f'{label}: wrong episodes per base seed')
    if int(payload.get('episodes', -1)) != EPISODES_PER_TRAINING_SEED:
        raise ValueError(f'{label}: wrong total episode count')


def raw_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    """Recompute all main metrics directly from one seed's episode rows."""
    if len(records) != EPISODES_PER_TRAINING_SEED:
        raise ValueError('one training seed must contribute exactly 100 episodes')
    returns = [float(row['return']) for row in records]
    costs = [float(row['cost']) for row in records]
    successes = [float(row['success']) for row in records]
    return {
        'mean_return': statistics.fmean(returns),
        'success_rate': statistics.fmean(successes),
        'mean_cost': statistics.fmean(costs),
        'cost_cvar_20': tail_mean(costs),
    }


def load_prior_result(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one SACLag/SHAPO result and return its seed row plus episodes."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    algorithm = str(payload.get('algorithm'))
    label = f'{algorithm}:{payload.get("training_seed")}'
    if payload.get('status') != 'complete' or algorithm not in PRIOR_METHODS:
        raise ValueError(f'{label}: not a completed prior-baseline evaluation')
    if payload.get('evaluation_role') != 'primary':
        raise ValueError(f'{label}: evaluation is not primary')
    validate_evaluation_schedule(
        {
            **payload,
            'episodes': payload.get('summary', {}).get('episodes'),
        },
        label,
    )
    training_seed = int(payload.get('training_seed', -1))
    if training_seed not in TRAINING_SEEDS or int(payload.get('training_steps', -1)) != 200_000:
        raise ValueError(f'{label}: wrong training identity or budget')
    raw_records = payload.get('episode_records', [])
    records = [
        {
            'method': algorithm,
            'training_seed': training_seed,
            'base_seed': int(record['base_seed']),
            'episode_index': int(record['episode_index']),
            'return': float(record['return']),
            'cost': float(record['cost']),
            'success': int(record['success']),
        }
        for record in raw_records
    ]
    schedule = [(record['base_seed'], record['episode_index']) for record in records]
    expected_schedule = [
        (base_seed, episode_index)
        for base_seed in BASE_SEEDS
        for episode_index in range(EPISODES_PER_BASE_SEED)
    ]
    if schedule != expected_schedule:
        raise ValueError(f'{label}: raw episode schedule is incomplete or reordered')
    metrics = raw_summary(records)
    stored = payload['summary']
    for field, value in metrics.items():
        require_close(stored[field], value, f'{label}.{field}')
    row = {
        'method': algorithm,
        'training_seed': training_seed,
        'training_steps': 200_000,
        'episodes': EPISODES_PER_TRAINING_SEED,
        **metrics,
        'training_cumulative_cost': float(payload['training_cumulative_cost']),
        'high_risk_vn_vt': None,
        'high_risk_dangerous_outward_velocity': None,
        'evaluation_mode': payload['evaluation_mode'],
        'evidence_file': str(path.resolve()),
        'evidence_sha256': sha256_file(path),
    }
    return row, records


def read_p4_training_rows(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    """Index sealed P4.7 per-training-seed metrics."""
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    index: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        arm = row['arm']
        seed = int(row['training_seed'])
        milestone = int(row['milestone'])
        if arm in P4_ARM_TO_METHOD and seed in TRAINING_SEEDS and milestone == 200_000:
            index[(arm, seed)] = row
    expected = {(arm, seed) for arm in P4_ARM_TO_METHOD for seed in TRAINING_SEEDS}
    if set(index) != expected:
        raise ValueError('sealed P4.7 seed table does not contain the exact 2x3 endpoint matrix')
    return index


def load_p4_result(
    arm: str,
    training_seed: int,
    evaluation_root: Path,
    training_row: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load one sealed C0/P4.7 raw evaluation and normalize its schema."""
    path = evaluation_root / arm / f'evaluation_seed_{training_seed}_step_200000.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    label = f'{arm}:{training_seed}'
    if payload.get('evaluation_mode') != 'fixed-random-flow':
        raise ValueError(f'{label}: sealed evaluation is not fixed-random-flow')
    if payload.get('evaluation_role') != 'primary':
        raise ValueError(f'{label}: sealed evaluation is not primary')
    validate_evaluation_schedule(payload, label)
    if int(payload.get('training_seed', -1)) != training_seed:
        raise ValueError(f'{label}: mismatched training seed')
    seeds = [int(seed) for seed in payload['seeds']]
    indices = [int(index) for index in payload['episode_indices']]
    returns = [float(value) for value in payload['returns']]
    costs = [float(value) for value in payload['costs']]
    goals = [int(value) for value in payload['goal_met_counts']]
    if not all(len(values) == EPISODES_PER_TRAINING_SEED for values in (seeds, indices, returns, costs, goals)):
        raise ValueError(f'{label}: malformed raw arrays')
    method = P4_ARM_TO_METHOD[arm]
    records = [
        {
            'method': method,
            'training_seed': training_seed,
            'base_seed': seed,
            'episode_index': index,
            'return': episode_return,
            'cost': episode_cost,
            'success': int(goal_count > 0),
        }
        for seed, index, episode_return, episode_cost, goal_count in zip(
            seeds,
            indices,
            returns,
            costs,
            goals,
        )
    ]
    metrics = raw_summary(records)
    require_close(payload['mean_return'], metrics['mean_return'], f'{label}.mean_return')
    require_close(payload['mean_cost'], metrics['mean_cost'], f'{label}.mean_cost')
    require_close(payload['cost_cvar_20'], metrics['cost_cvar_20'], f'{label}.cost_cvar_20')
    require_close(
        payload['goal_met_episode_rate'],
        metrics['success_rate'],
        f'{label}.success_rate',
    )
    row = {
        'method': method,
        'training_seed': training_seed,
        'training_steps': 200_000,
        'episodes': EPISODES_PER_TRAINING_SEED,
        **metrics,
        'training_cumulative_cost': float(training_row['training_cumulative_cost']),
        'high_risk_vn_vt': finite_or_none(
            training_row['high_risk_normal_to_tangent_variance_ratio'],
        ),
        'high_risk_dangerous_outward_velocity': finite_or_none(
            training_row['high_risk_dangerous_outward_velocity'],
        ),
        'evaluation_mode': payload['evaluation_mode'],
        'evidence_file': str(path.resolve()),
        'evidence_sha256': sha256_file(path),
    }
    return row, records


def aggregate_method(
    method: str,
    seed_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate one method while keeping seed-level variation explicit."""
    seeds = sorted(int(row['training_seed']) for row in seed_rows)
    if seeds != list(TRAINING_SEEDS):
        raise ValueError(f'{method}: expected exactly training seeds {TRAINING_SEEDS}')
    if len(records) != len(TRAINING_SEEDS) * EPISODES_PER_TRAINING_SEED:
        raise ValueError(f'{method}: expected exactly 300 pooled episodes')
    returns = [float(record['return']) for record in records]
    costs = [float(record['cost']) for record in records]
    successes = [float(record['success']) for record in records]
    high_risk_ratios = [
        float(row['high_risk_vn_vt'])
        for row in seed_rows
        if row['high_risk_vn_vt'] is not None
    ]
    high_risk_outward = [
        float(row['high_risk_dangerous_outward_velocity'])
        for row in seed_rows
        if row['high_risk_dangerous_outward_velocity'] is not None
    ]
    return {
        'method': method,
        'training_steps_per_seed': 200_000,
        'training_seeds': len(seed_rows),
        'pooled_evaluation_episodes': len(records),
        'mean_return': statistics.fmean(returns),
        'seed_std_mean_return': sample_std([float(row['mean_return']) for row in seed_rows]),
        'success_rate': statistics.fmean(successes),
        'seed_std_success_rate': sample_std([float(row['success_rate']) for row in seed_rows]),
        'mean_cost': statistics.fmean(costs),
        'seed_std_mean_cost': sample_std([float(row['mean_cost']) for row in seed_rows]),
        'cost_cvar_20': tail_mean(costs),
        'seed_std_cost_cvar_20': sample_std(
            [float(row['cost_cvar_20']) for row in seed_rows],
        ),
        'training_cumulative_cost_mean': statistics.fmean(
            float(row['training_cumulative_cost']) for row in seed_rows
        ),
        'seed_std_training_cumulative_cost': sample_std(
            [float(row['training_cumulative_cost']) for row in seed_rows],
        ),
        'high_risk_vn_vt': statistics.fmean(high_risk_ratios) if high_risk_ratios else None,
        'high_risk_dangerous_outward_velocity': (
            statistics.fmean(high_risk_outward) if high_risk_outward else None
        ),
    }


def format_optional(value: Any, digits: int = 4) -> str:
    """Format optional diagnostics for the Markdown table."""
    return 'n/a' if value is None else f'{float(value):.{digits}f}'


def markdown_report(aggregates: list[dict[str, Any]]) -> str:
    """Render the auditable four-method main-table report."""
    by_method = {row['method']: row for row in aggregates}
    ours = by_method['P4.7 / Ours']
    baseline = by_method['C0 / isotropic FLAC']
    mean_cost_reduction = 1.0 - ours['mean_cost'] / baseline['mean_cost']
    cvar_reduction = 1.0 - ours['cost_cvar_20'] / baseline['cost_cvar_20']
    train_cost_reduction = (
        1.0
        - ours['training_cumulative_cost_mean'] / baseline['training_cumulative_cost_mean']
    )
    lines = [
        '# PointGoal four-method final comparison',
        '',
        'All methods use three training seeds (`2027–2029`), an equal `200k` training budget, '
        'and 100 final evaluation episodes per training seed. The independent statistical unit '
        'is the training seed (`n=3`); pooled episodes are not additional training replicates.',
        '',
        '| Method | Return | Success | Mean cost | CVaR-20 | Train cost | High-risk V_N/V_T |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for method in METHOD_ORDER:
        row = by_method[method]
        lines.append(
            f"| {method} | {row['mean_return']:.4f} | {row['success_rate']:.3f} | "
            f"{row['mean_cost']:.3f} | {row['cost_cvar_20']:.3f} | "
            f"{row['training_cumulative_cost_mean']:.1f} | "
            f"{format_optional(row['high_risk_vn_vt'])} |",
        )
    lines.extend(
        [
            '',
            '## Frozen P4.7 conclusion',
            '',
            f"Relative to isotropic FLAC, P4.7 changes return by "
            f"`{ours['mean_return'] - baseline['mean_return']:+.4f}` and success by "
            f"`{100 * (ours['success_rate'] - baseline['success_rate']):+.1f}` percentage points, "
            f"while reducing mean cost by `{100 * mean_cost_reduction:.1f}%`, CVaR-20 by "
            f"`{100 * cvar_reduction:.1f}%`, and mean cumulative training cost by "
            f"`{100 * train_cost_reduction:.1f}%`.",
            '',
            'This remains an aggregate three-seed Pareto statement, not an every-seed Pareto claim. '
            'SACLag and SHAPO-PIDLag are reported as preregistered prior-method comparisons; their '
            'outcomes were not used to select or modify P4.7.',
            '',
            '## Evaluation semantics',
            '',
            '- C0 and P4.7 reuse the sealed P4.7 fixed-random-flow artifacts.',
            '- SACLag and SHAPO-PIDLag use their native stochastic action distributions with fixed '
            'Torch RNG streams and the same five base environment seeds.',
            '- Every base seed advances through 20 sequential episodes without reseeding between '
            'episodes.',
            '- SHAPO-PIDLag uses the observation running statistics captured during training and '
            'keeps them frozen during evaluation.',
            '',
        ],
    )
    return '\n'.join(lines)


def git_commit(source: Path) -> str:
    """Require a clean finalizer source and return its commit."""
    status = subprocess.run(
        ['git', '-C', str(source), 'status', '--porcelain'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError('formal finalization requires a clean committed source')
    return subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    root = Path(__file__).resolve().parents[2]
    p4_root = root / 'reports/star/p4_7_safety_aware_geometry_artifacts/final'
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--prior-evaluations', nargs='+', type=Path, required=True)
    result.add_argument(
        '--p4-evaluation-root',
        type=Path,
        default=p4_root / 'formal_evaluations',
    )
    result.add_argument(
        '--p4-seed-table',
        type=Path,
        default=p4_root / 'summary/p4_7_formal_final_seed_metrics.csv',
    )
    result.add_argument(
        '--protocol',
        type=Path,
        default=root / 'reports/star/single_gpu_paper_core_preregistration.json',
    )
    result.add_argument('--expected-protocol-sha256', required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    return result


def main() -> None:  # pylint: disable=too-many-locals
    """Validate, merge, and publish the four-method PointGoal evidence."""
    args = parser().parse_args()
    source_root = Path(__file__).resolve().parents[2]
    finalizer_commit = git_commit(source_root)
    if sha256_file(args.protocol) != args.expected_protocol_sha256:
        raise ValueError('single-GPU protocol SHA-256 mismatch')
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    if protocol.get('protocol_version') != 'star-single-gpu-paper-core-v1':
        raise ValueError('unexpected single-GPU protocol version')

    seed_rows: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    p4_rows = read_p4_training_rows(args.p4_seed_table)
    for arm in P4_ARM_TO_METHOD:
        for training_seed in TRAINING_SEEDS:
            row, records = load_p4_result(
                arm,
                training_seed,
                args.p4_evaluation_root,
                p4_rows[(arm, training_seed)],
            )
            seed_rows.append(row)
            episode_records.extend(records)

    for path in args.prior_evaluations:
        row, records = load_prior_result(path.resolve())
        seed_rows.append(row)
        episode_records.extend(records)

    identities = [(row['method'], int(row['training_seed'])) for row in seed_rows]
    expected_identities = {
        (method, training_seed) for method in METHOD_ORDER for training_seed in TRAINING_SEEDS
    }
    if set(identities) != expected_identities or len(identities) != len(expected_identities):
        raise ValueError('comparison inputs do not form the exact four-method by three-seed matrix')

    seed_rows.sort(key=lambda row: (METHOD_ORDER.index(row['method']), row['training_seed']))
    episode_records.sort(
        key=lambda row: (
            METHOD_ORDER.index(row['method']),
            row['training_seed'],
            row['base_seed'],
            row['episode_index'],
        ),
    )
    aggregates = [
        aggregate_method(
            method,
            [row for row in seed_rows if row['method'] == method],
            [record for record in episode_records if record['method'] == method],
        )
        for method in METHOD_ORDER
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    public_seed_rows = [
        {key: value for key, value in row.items() if key not in ('evidence_file', 'evidence_sha256')}
        for row in seed_rows
    ]
    atomic_csv(args.output_dir / 'pointgoal_four_method_seed_metrics.csv', public_seed_rows)
    atomic_csv(args.output_dir / 'pointgoal_four_method_summary.csv', aggregates)
    atomic_csv(args.output_dir / 'pointgoal_four_method_episode_records.csv', episode_records)
    atomic_text(
        args.output_dir / 'POINTGOAL_FOUR_METHOD_REPORT.md',
        markdown_report(aggregates),
    )
    manifest = {
        'status': 'complete',
        'protocol_sha256': args.expected_protocol_sha256,
        'finalizer_source_commit': finalizer_commit,
        'statistical_unit': 'training_seed',
        'training_seeds_per_method': len(TRAINING_SEEDS),
        'evaluation_episodes_per_training_seed': EPISODES_PER_TRAINING_SEED,
        'aggregate_rows': aggregates,
        'evidence': [
            {
                'method': row['method'],
                'training_seed': row['training_seed'],
                'path': row['evidence_file'],
                'sha256': row['evidence_sha256'],
            }
            for row in seed_rows
        ],
    }
    atomic_json(args.output_dir / 'pointgoal_four_method_manifest.json', manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()

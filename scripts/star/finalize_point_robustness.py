#!/usr/bin/env python3
"""Aggregate checkpoint-only PointGoal/PointCircle robustness evidence."""

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

from scripts.star.evaluate_point_robustness import (
    BASE_SEEDS,
    FINAL_EPISODES_PER_SEED,
    SCREEN_EPISODES_PER_SEED,
    TRAINING_SEEDS,
    conditions,
)


ENVIRONMENTS = ('SafetyPointGoal1-v0', 'SafetyPointCircle1-v0')
ARMS = ('C0', 'P47')


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
    """Write one homogeneous CSV through an atomic rename."""
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    """Write UTF-8 text through an atomic rename."""
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(content, encoding='utf-8')
    os.replace(temporary, path)


def tail_mean(values: list[float]) -> float:
    """Return empirical upper-tail cost mean."""
    count = max(1, math.ceil(0.2 * len(values)))
    return statistics.fmean(sorted(values, reverse=True)[:count])


def finite(value: Any, label: str) -> float:
    """Require one finite number."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'{label} is not finite')
    return number


def validate_evaluation(
    evaluation: dict[str, Any],
    episodes_per_seed: int,
    label: str,
) -> dict[str, Any]:
    """Validate and compact one raw per-checkpoint condition evaluation."""
    expected_episodes = len(BASE_SEEDS) * episodes_per_seed
    if (
        evaluation.get('evaluation_mode') != 'fixed-random-flow'
        or evaluation.get('evaluation_role') != 'primary'
        or [int(seed) for seed in evaluation.get('base_seeds', [])] != list(BASE_SEEDS)
        or int(evaluation.get('episodes_per_base_seed', -1)) != episodes_per_seed
        or int(evaluation.get('episodes', -1)) != expected_episodes
    ):
        raise ValueError(f'{label}: invalid evaluation schedule')
    returns = [finite(value, f'{label}.return') for value in evaluation['returns']]
    costs = [finite(value, f'{label}.cost') for value in evaluation['costs']]
    if len(returns) != expected_episodes or len(costs) != expected_episodes:
        raise ValueError(f'{label}: incomplete raw arrays')
    mean_return = statistics.fmean(returns)
    mean_cost = statistics.fmean(costs)
    cvar = tail_mean(costs)
    for key, expected in (
        ('mean_return', mean_return),
        ('mean_cost', mean_cost),
        ('cost_cvar_20', cvar),
    ):
        if not math.isclose(float(evaluation[key]), expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f'{label}.{key} disagrees with raw evidence')
    geometry = evaluation.get('direction_geometry_high_risk_top_20_percent', {})
    return {
        'returns': returns,
        'costs': costs,
        'mean_return': mean_return,
        'mean_cost': mean_cost,
        'cost_cvar_20': cvar,
        'high_risk_vn_vt': finite(
            geometry.get('normal_to_tangent_variance_ratio'),
            f'{label}.high_risk_vn_vt',
        ),
        'high_risk_outward_velocity': finite(
            geometry.get('dangerous_outward_velocity'),
            f'{label}.high_risk_outward_velocity',
        ),
    }


def load_result(path: Path, protocol_sha: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one checkpoint's complete task robustness suite."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    environment = str(payload.get('environment'))
    arm = str(payload.get('arm'))
    seed = int(payload.get('training_seed', -1))
    expected_steps = 200_000 if environment == 'SafetyPointGoal1-v0' else 500_000
    if (
        payload.get('status') != 'complete'
        or environment not in ENVIRONMENTS
        or arm not in ARMS
        or seed not in TRAINING_SEEDS
        or int(payload.get('training_steps', -1)) != expected_steps
        or payload.get('robustness_protocol_sha256') != protocol_sha
        or bool(payload.get('retraining'))
    ):
        raise ValueError(f'{path}: invalid robustness result identity')
    checkpoint = Path(payload['checkpoint'])
    if not checkpoint.is_file() or sha256_file(checkpoint) != payload.get('checkpoint_sha256'):
        raise ValueError(f'{path}: checkpoint identity mismatch')
    observed = payload.get('conditions', [])
    expected = conditions(environment)
    if [item.get('name') for item in observed] != [item.name for item in expected]:
        raise ValueError(f'{path}: condition order or coverage mismatch')
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for item, specification in zip(observed, expected):
        tiers = [('screening', item['screening'], SCREEN_EPISODES_PER_SEED)]
        if specification.selected_final:
            if item.get('selected_final') is None:
                raise ValueError(f'{path}: missing selected final tier for {specification.name}')
            tiers.append(('selected_final', item['selected_final'], FINAL_EPISODES_PER_SEED))
        elif item.get('selected_final') is not None:
            raise ValueError(f'{path}: unexpected final tier for {specification.name}')
        for tier, evaluation, episodes_per_seed in tiers:
            label = f'{environment}:{arm}:{seed}:{specification.name}:{tier}'
            metrics = validate_evaluation(evaluation, episodes_per_seed, label)
            rows.append(
                {
                    'environment': environment,
                    'arm': arm,
                    'training_seed': seed,
                    'condition': specification.name,
                    'family': specification.family,
                    'level': specification.level,
                    'tier': tier,
                    'episodes': len(metrics['returns']),
                    'mean_return': metrics['mean_return'],
                    'mean_cost': metrics['mean_cost'],
                    'cost_cvar_20': metrics['cost_cvar_20'],
                    'high_risk_vn_vt': metrics['high_risk_vn_vt'],
                    'high_risk_outward_velocity': metrics['high_risk_outward_velocity'],
                }
            )
            raw.extend(
                {
                    'environment': environment,
                    'arm': arm,
                    'training_seed': seed,
                    'condition': specification.name,
                    'tier': tier,
                    'episode_index': index,
                    'return': episode_return,
                    'cost': episode_cost,
                }
                for index, (episode_return, episode_cost) in enumerate(
                    zip(metrics['returns'], metrics['costs'])
                )
            )
    evidence = {
        'environment': environment,
        'arm': arm,
        'training_seed': seed,
        'path': str(path.resolve()),
        'sha256': sha256_file(path),
    }
    return evidence, rows + raw


def aggregate(seed_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one exact task/arm/condition/tier cell across three training seeds."""
    seeds = sorted(int(row['training_seed']) for row in seed_rows)
    if seeds != list(TRAINING_SEEDS):
        raise ValueError('robustness aggregate lacks exactly three training seeds')
    returns = [float(row['return']) for row in raw_rows]
    costs = [float(row['cost']) for row in raw_rows]
    first = seed_rows[0]
    return {
        'environment': first['environment'],
        'arm': first['arm'],
        'condition': first['condition'],
        'family': first['family'],
        'level': first['level'],
        'tier': first['tier'],
        'training_seeds': len(seed_rows),
        'pooled_episodes': len(raw_rows),
        'mean_return': statistics.fmean(returns),
        'seed_std_mean_return': statistics.stdev(float(row['mean_return']) for row in seed_rows),
        'mean_cost': statistics.fmean(costs),
        'seed_std_mean_cost': statistics.stdev(float(row['mean_cost']) for row in seed_rows),
        'cost_cvar_20': tail_mean(costs),
        'seed_std_cost_cvar_20': statistics.stdev(float(row['cost_cvar_20']) for row in seed_rows),
        'high_risk_vn_vt': statistics.fmean(float(row['high_risk_vn_vt']) for row in seed_rows),
        'high_risk_outward_velocity': statistics.fmean(
            float(row['high_risk_outward_velocity']) for row in seed_rows
        ),
    }


def markdown(aggregates: list[dict[str, Any]]) -> str:
    """Render primary robustness trends with P4.7-minus-C0 contrasts."""
    primary = [
        row
        for row in aggregates
        if row['tier'] == 'selected_final'
        or not any(
            other['environment'] == row['environment']
            and other['arm'] == row['arm']
            and other['condition'] == row['condition']
            and other['tier'] == 'selected_final'
            for other in aggregates
        )
    ]
    lines = [
        '# Checkpoint-only Point robustness',
        '',
        'No policy was retrained. Every row aggregates three training seeds; the training seed '
        'is the independent statistical unit. Preselected strong/end-point conditions use 100 '
        'episodes per checkpoint and all others use 50.',
        '',
        '| Environment | Condition | Episodes/checkpoint | C0 return | P47 return | C0 cost | P47 cost | C0 CVaR | P47 CVaR |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    keys = sorted({(row['environment'], row['condition'], row['tier']) for row in primary})
    for environment, condition_name, tier in keys:
        cells = {
            row['arm']: row
            for row in primary
            if row['environment'] == environment
            and row['condition'] == condition_name
            and row['tier'] == tier
        }
        c0, ours = cells['C0'], cells['P47']
        episodes_per_checkpoint = c0['pooled_episodes'] // len(TRAINING_SEEDS)
        lines.append(
            f"| {environment} | {condition_name} | {episodes_per_checkpoint} | "
            f"{c0['mean_return']:.4f} | {ours['mean_return']:.4f} | "
            f"{c0['mean_cost']:.3f} | {ours['mean_cost']:.3f} | "
            f"{c0['cost_cvar_20']:.3f} | {ours['cost_cvar_20']:.3f} |"
        )
    lines.extend(
        [
            '',
            '| Environment | Condition | C0 V_N/V_T | P47 V_N/V_T | C0 outward | P47 outward |',
            '|---|---|---:|---:|---:|---:|',
        ]
    )
    for environment, condition_name, tier in keys:
        cells = {
            row['arm']: row
            for row in primary
            if row['environment'] == environment
            and row['condition'] == condition_name
            and row['tier'] == tier
        }
        c0, ours = cells['C0'], cells['P47']
        lines.append(
            f"| {environment} | {condition_name} | {c0['high_risk_vn_vt']:.4f} | "
            f"{ours['high_risk_vn_vt']:.4f} | "
            f"{c0['high_risk_outward_velocity']:.5f} | "
            f"{ours['high_risk_outward_velocity']:.5f} |"
        )
    lines.extend(
        [
            '',
            'The table is a robustness trend analysis, not a new selection gate and not an '
            'every-seed Pareto claim.',
            '',
        ]
    )
    return '\n'.join(lines)


def git_commit(source: Path) -> str:
    """Require clean finalizer source."""
    status = subprocess.run(
        ['git', '-C', str(source), 'status', '--porcelain'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError('robustness finalization requires a clean source')
    return subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parser() -> argparse.ArgumentParser:
    """Build finalization arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--results', nargs='+', type=Path, required=True)
    result.add_argument('--protocol', type=Path, required=True)
    result.add_argument('--expected-protocol-sha256', required=True)
    result.add_argument(
        '--environments',
        nargs='+',
        choices=ENVIRONMENTS,
        default=list(ENVIRONMENTS),
    )
    result.add_argument('--output-dir', type=Path, required=True)
    return result


def main() -> None:
    """Validate the exact requested-task by 2-arm by 3-seed evidence matrix."""
    args = parser().parse_args()
    selected_environments = tuple(dict.fromkeys(args.environments))
    source = Path(__file__).resolve().parents[2]
    source_commit = git_commit(source)
    if sha256_file(args.protocol) != args.expected_protocol_sha256:
        raise ValueError('robustness protocol SHA-256 mismatch')
    evidence: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for path in args.results:
        item_evidence, combined = load_result(path.resolve(), args.expected_protocol_sha256)
        evidence.append(item_evidence)
        seed_rows.extend(row for row in combined if 'episodes' in row)
        raw_rows.extend(row for row in combined if 'episode_index' in row)
    identities = {(item['environment'], item['arm'], item['training_seed']) for item in evidence}
    expected = {
        (environment, arm, seed)
        for environment in selected_environments
        for arm in ARMS
        for seed in TRAINING_SEEDS
    }
    if identities != expected or len(evidence) != len(expected):
        raise ValueError('robustness inputs do not form the exact requested-task x 2 x 3 matrix')
    cells = sorted(
        {
            (row['environment'], row['arm'], row['condition'], row['tier'])
            for row in seed_rows
        }
    )
    aggregates = [
        aggregate(
            [
                row
                for row in seed_rows
                if (row['environment'], row['arm'], row['condition'], row['tier']) == cell
            ],
            [
                row
                for row in raw_rows
                if (row['environment'], row['arm'], row['condition'], row['tier']) == cell
            ],
        )
        for cell in cells
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.output_dir / 'point_robustness_seed_metrics.csv', seed_rows)
    atomic_csv(args.output_dir / 'point_robustness_summary.csv', aggregates)
    atomic_text(args.output_dir / 'POINT_ROBUSTNESS_REPORT.md', markdown(aggregates))
    manifest = {
        'status': 'complete',
        'protocol_sha256': args.expected_protocol_sha256,
        'finalizer_source_commit': source_commit,
        'retraining': False,
        'environments': list(selected_environments),
        'statistical_unit': 'training_seed',
        'training_seeds_per_cell': len(TRAINING_SEEDS),
        'aggregate_rows': aggregates,
        'evidence': evidence,
    }
    atomic_json(args.output_dir / 'point_robustness_manifest.json', manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()

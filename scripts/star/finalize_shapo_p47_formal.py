#!/usr/bin/env python3
"""Aggregate the completed SHAPO P4.7 matrix with training seed as the statistical unit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

import yaml


SLUGS = {
    'SafetyPointGoal1-v0': 'point-goal',
    'SafetyPointButton1-v0': 'point-button',
    'Ant-v4': 'ant',
    'Walker2d-v4': 'walker2d',
}
METRICS = (
    'mean_return',
    'mean_cost',
    'cost_cvar_20',
    'training_cumulative_cost',
    'success_rate',
    'mean_path_length',
    'cost_per_successful_episode',
    'failure_count',
    'failure_rate',
    'mean_failure_time',
    'mean_failure_step',
    'mean_failure_height',
    'mean_failure_torso_angle',
    'time_limit_count',
    'time_limit_rate',
    'high_risk_vn_vt',
    'high_risk_dangerous_outward_velocity',
)
REQUIRED_FIVE_SEED_METRICS = frozenset(
    {
        'mean_return',
        'mean_cost',
        'cost_cvar_20',
        'training_cumulative_cost',
        'success_rate',
        'mean_path_length',
        'failure_count',
        'failure_rate',
        'time_limit_count',
        'time_limit_rate',
        'high_risk_vn_vt',
        'high_risk_dangerous_outward_velocity',
    },
)


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    """Atomically write one final artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(value, encoding='utf-8')
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    """Write strict JSON while mapping undefined diagnostics to null."""
    atomic_text(path, f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n')


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a stable union of row fields."""
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    """Copy one compact artifact atomically after checking its immutable identity."""
    if sha256_file(source) != expected_sha256:
        raise ValueError(f'trajectory artifact changed after validation: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
    temporary.write_bytes(source.read_bytes())
    os.replace(temporary, destination)


def numeric(value: Any) -> float | None:
    """Return a finite float or None for an undefined secondary diagnostic."""
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def seed_summary(values: list[float]) -> dict[str, float | int]:
    """Summarize available independent training seeds with a two-sided t interval."""
    if not values:
        raise ValueError('cannot summarize an empty seed sample')
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    sem = std / math.sqrt(len(values))
    t_critical = {
        1: 12.706204736432095,
        2: 4.302652729696142,
        3: 3.182446305284263,
        4: 2.7764451051977987,
    }.get(len(values) - 1, 1.959963984540054)
    half_width = t_critical * sem
    return {
        'n_training_seeds': len(values),
        'mean': mean,
        'sample_std': std,
        'standard_error': sem,
        't95_ci_low': mean - half_width,
        't95_ci_high': mean + half_width,
        'minimum': min(values),
        'maximum': max(values),
    }


def _formatted(aggregate: dict[str, Any], env_id: str, metric: str) -> str:
    """Format one training-seed summary or an explicit undefined marker."""
    summary = aggregate[env_id].get(metric)
    if summary is None:
        return '—'
    return f"{summary['mean']:.4f} ± {summary['sample_std']:.4f}"


def render_report(
    aggregate: dict[str, Any],
    source_commit: str,
    figures: list[Path],
) -> str:
    """Render the compact frozen-method result report."""
    lines = [
        '# Frozen P4.7 SHAPO Four-Environment Report',
        '',
        f'- Formal source commit: `{source_commit}`',
        '- Independent statistical unit: training seed (`n=5` per environment)',
        '- Evaluation: 5 fixed base seeds by 20 sequential episodes = 100 episodes per training seed',
        '- Point tasks: 10M environment steps; locomotion tasks: 5M environment steps',
        '- Scientific early stopping: disabled',
        '',
        '## Final primary outcomes',
        '',
        '| Environment | Return | Mean cost | CVaR-20 | Training cumulative cost |',
        '|---|---:|---:|---:|---:|',
    ]
    for env_id, metrics in aggregate.items():
        cells = []
        for metric in (
            'mean_return',
            'mean_cost',
            'cost_cvar_20',
            'training_cumulative_cost',
        ):
            summary = metrics[metric]
            cells.append(f"{summary['mean']:.4f} ± {summary['sample_std']:.4f}")
        lines.append(f"| {env_id} | {' | '.join(cells)} |")
    lines.extend(
        [
            '',
            (
                'Values are mean ± sample standard deviation across five training seeds. '
                'Episode-level observations are not treated as independent replicates.'
            ),
            '',
            '## Task outcomes',
            '',
            (
                '| Environment | Success rate | Mean path length | Cost / successful episode | '
                'Failure rate | Mean failure time |'
            ),
            '|---|---:|---:|---:|---:|---:|',
        ],
    )
    for env_id in aggregate:
        lines.append(
            f'| {env_id} | {_formatted(aggregate, env_id, "success_rate")} | '
            f'{_formatted(aggregate, env_id, "mean_path_length")} | '
            f'{_formatted(aggregate, env_id, "cost_per_successful_episode")} | '
            f'{_formatted(aggregate, env_id, "failure_rate")} | '
            f'{_formatted(aggregate, env_id, "mean_failure_time")} |',
        )
    lines.extend(
        [
            '',
            '## High-risk direction diagnostics',
            '',
            '| Environment | Top-20% $V_N/V_T$ | Top-20% dangerous outward velocity |',
            '|---|---:|---:|',
        ],
    )
    for env_id in aggregate:
        lines.append(
            f'| {env_id} | {_formatted(aggregate, env_id, "high_risk_vn_vt")} | '
            f'{_formatted(aggregate, env_id, "high_risk_dangerous_outward_velocity")} |',
        )
    lines.extend(
        [
            '',
            '## Figures',
            '',
            *[f'- [`{path.name}`](figures/{path.name})' for path in figures],
            '',
            '## Interpretation boundary',
            '',
            (
                'This report describes the frozen P4.7 arm only. High-risk direction geometry, '
                'failure pose/time fields, and trajectory risk traces are diagnostics and were '
                'not used as pass/fail gates or for post-hoc environment-specific tuning.'
            ),
            '',
        ],
    )
    return '\n'.join(lines)


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Require all 20 validated runs and emit seed-level final evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not (args.run_root / 'formal_matrix_complete').is_file():
        raise RuntimeError('formal matrix completion seal is absent')
    matrix = yaml.safe_load(args.matrix.read_text(encoding='utf-8'))

    final_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    trajectory_sources: list[dict[str, Any]] = []
    source_commits = set()
    for env_id, environment in matrix['environments'].items():
        expected_milestones = [int(step) for step in environment['milestones']]
        for seed in matrix['training_seeds']:
            container = args.run_root / 'runs' / f'{SLUGS[env_id]}-{seed}'
            validation_path = container / 'formal_validation.json'
            validation = json.loads(validation_path.read_text(encoding='utf-8'))
            if validation.get('status') != 'passed':
                raise ValueError(f'{validation_path} is not a passing validation')
            if validation['environment'] != env_id or int(validation['training_seed']) != seed:
                raise ValueError(f'{validation_path} identity does not match its matrix cell')
            source_commits.add(str(validation['source_commit']))
            evaluations = validation['evaluation_rows']
            if [int(row['milestone']) for row in evaluations] != expected_milestones:
                raise ValueError(f'{validation_path} has incomplete learning-curve milestones')
            for evaluation in evaluations:
                curve_rows.append(
                    {
                        'environment': env_id,
                        'training_seed': seed,
                        **evaluation,
                    },
                )
            trajectory_artifacts = validation.get('trajectory_artifacts', [])
            if [int(row['milestone']) for row in trajectory_artifacts] != expected_milestones:
                raise ValueError(f'{validation_path} has incomplete trajectory milestones')
            expected_trajectory_seeds = sorted(
                int(value) for value in matrix['evaluation']['trajectory_base_seeds']
            )
            for trajectory in trajectory_artifacts:
                if sorted(int(value) for value in trajectory['trajectory_seeds']) != (
                    expected_trajectory_seeds
                ):
                    raise ValueError(f'{validation_path} has changed trajectory seeds')
                source = Path(trajectory['artifact'])
                if not source.is_file():
                    raise FileNotFoundError(f'trajectory artifact is unavailable: {source}')
                if int(trajectory['row_count']) <= 0:
                    raise ValueError(f'trajectory artifact is empty: {source}')
                if sha256_file(source) != trajectory['sha256']:
                    raise ValueError(f'trajectory artifact identity mismatch: {source}')
                trajectory_sources.append(
                    {
                        'environment': env_id,
                        'training_seed': seed,
                        **trajectory,
                    },
                )
            final_rows.append(
                {
                    'environment': env_id,
                    'training_seed': seed,
                    'total_steps': int(environment['total_steps']),
                    'formal_cost_budget': float(environment['cost_limit']),
                    'geometry_mode': str(environment['geometry_mode']),
                    **evaluations[-1],
                },
            )
    if len(final_rows) != 20 or len(source_commits) != 1:
        raise ValueError('formal matrix must contain 20 runs from one source commit')

    aggregate: dict[str, dict[str, Any]] = {}
    aggregate_rows = []
    for env_id in matrix['environments']:
        selected = [row for row in final_rows if row['environment'] == env_id]
        aggregate[env_id] = {}
        for metric in METRICS:
            values = [numeric(row.get(metric)) for row in selected]
            finite_values = [value for value in values if value is not None]
            if not finite_values:
                continue
            if metric in REQUIRED_FIVE_SEED_METRICS and len(finite_values) != 5:
                raise ValueError(f'{env_id} metric {metric} is missing for some training seeds')
            summary = seed_summary(finite_values)
            aggregate[env_id][metric] = summary
            aggregate_rows.append({'environment': env_id, 'metric': metric, **summary})

    output = args.output_dir
    trajectory_rows = []
    for row in trajectory_sources:
        destination = (
            output
            / 'trajectories'
            / SLUGS[row['environment']]
            / f"training_seed_{row['training_seed']}"
            / f"trajectory_step_{row['milestone']}.csv"
        )
        copy_verified(Path(row['artifact']), destination, str(row['sha256']))
        trajectory_rows.append(
            {
                'environment': row['environment'],
                'training_seed': row['training_seed'],
                'milestone': row['milestone'],
                'artifact': destination.relative_to(output).as_posix(),
                'sha256': row['sha256'],
                'row_count': row['row_count'],
                'trajectory_seeds_json': json.dumps(row['trajectory_seeds']),
                'rows_per_seed_json': json.dumps(row['rows_per_seed'], sort_keys=True),
            },
        )

    source_commit = next(iter(source_commits))
    write_csv(output / 'shapo_p47_final_seed_metrics.csv', final_rows)
    write_csv(output / 'shapo_p47_learning_curves.csv', curve_rows)
    write_csv(output / 'shapo_p47_seed_level_aggregate.csv', aggregate_rows)
    write_csv(output / 'shapo_p47_trajectory_manifest.csv', trajectory_rows)
    from scripts.star.plot_shapo_p47_formal import (  # pylint: disable=import-outside-toplevel
        plot_all,
    )

    figures = plot_all(output, output / 'figures')
    write_json(
        output / 'shapo_p47_final_summary.json',
        {
            'protocol_version': matrix['protocol_version'],
            'protocol_sha256': matrix['protocol_sha256'],
            'matrix_sha256': sha256_file(args.matrix),
            'source_commit': source_commit,
            'statistical_unit': 'training_seed',
            'n_training_seeds_per_environment': 5,
            'episodes_per_training_seed': 100,
            'aggregate': aggregate,
            'trajectory_artifact_count': len(trajectory_rows),
            'figures': [path.relative_to(output).as_posix() for path in figures],
        },
    )
    atomic_text(
        output / 'SHAPO_P47_FOUR_ENV_REPORT.md',
        render_report(aggregate, source_commit, figures),
    )
    print(
        json.dumps(
            {
                'status': 'passed',
                'source_commit': source_commit,
                'formal_runs': len(final_rows),
                'output_dir': str(output),
            },
            indent=2,
            sort_keys=True,
        ),
    )


if __name__ == '__main__':
    main()

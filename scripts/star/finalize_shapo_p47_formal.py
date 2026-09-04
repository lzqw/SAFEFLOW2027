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
STRICT_VALIDATOR = Path(__file__).resolve().with_name('validate_shapo_p47_run.py')
PLOTTER = Path(__file__).resolve().with_name('plot_shapo_p47_formal.py')
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


def verify_artifact_identity(identity: dict[str, Any], label: str) -> Path:
    """Require one compact or server-only artifact to retain its validated identity."""
    source = Path(identity['artifact'])
    if not source.is_file() or source.stat().st_size != int(identity['bytes']):
        raise ValueError(f'{label} artifact is unavailable or changed: {source}')
    if sha256_file(source) != str(identity['sha256']):
        raise ValueError(f'{label} artifact identity mismatch: {source}')
    return source


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


def summarize_checkpoint_recovery(
    checkpoint_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize audited resume sources by independent environment/seed run."""
    sources: dict[tuple[str, int], set[str]] = {}
    for row in checkpoint_rows:
        resume_from = str(row.get('resume_from') or '')
        if not resume_from:
            continue
        identity = (str(row['environment']), int(row['training_seed']))
        sources.setdefault(identity, set()).add(resume_from)
    recovered_runs = [
        {
            'environment': environment,
            'training_seed': seed,
            'resume_checkpoints': sorted(paths),
        }
        for (environment, seed), paths in sorted(sources.items())
    ]
    return {
        'recovered_training_run_count': len(recovered_runs),
        'recovered_training_runs': recovered_runs,
    }


def _formatted(aggregate: dict[str, Any], env_id: str, metric: str) -> str:
    """Format one training-seed summary or an explicit undefined marker."""
    summary = aggregate[env_id].get(metric)
    if summary is None:
        return '—'
    return f"{summary['mean']:.4f} ± {summary['sample_std']:.4f}"


def render_report(
    aggregate: dict[str, Any],
    failure_reason_aggregate: dict[str, dict[str, Any]],
    recovery: dict[str, Any],
    source_commit: str,
    validator_sha256: str,
    plotter_sha256: str,
    figures: list[Path],
) -> str:
    """Render the compact frozen-method result report."""
    lines = [
        '# Frozen P4.7 SHAPO Four-Environment Report',
        '',
        f'- Formal source commit: `{source_commit}`',
        f'- Strict validator SHA-256: `{validator_sha256}`',
        f'- Plotter SHA-256: `{plotter_sha256}`',
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
            '## Locomotion failure diagnostics',
            '',
            (
                '| Environment | Failure count | Failure rate | Failure time | Failure step | '
                'Failure height | Torso angle | Time-limit rate |'
            ),
            '|---|---:|---:|---:|---:|---:|---:|---:|',
        ],
    )
    for env_id in ('Ant-v4', 'Walker2d-v4'):
        lines.append(
            f'| {env_id} | {_formatted(aggregate, env_id, "failure_count")} | '
            f'{_formatted(aggregate, env_id, "failure_rate")} | '
            f'{_formatted(aggregate, env_id, "mean_failure_time")} | '
            f'{_formatted(aggregate, env_id, "mean_failure_step")} | '
            f'{_formatted(aggregate, env_id, "mean_failure_height")} | '
            f'{_formatted(aggregate, env_id, "mean_failure_torso_angle")} | '
            f'{_formatted(aggregate, env_id, "time_limit_rate")} |',
        )
    lines.extend(
        [
            '',
            'Failure-reason rates are also summarized across independent training seeds:',
            '',
            '| Environment | End reason | Episode rate |',
            '|---|---|---:|',
        ],
    )
    for env_id in ('Ant-v4', 'Walker2d-v4'):
        for reason, summary in sorted(failure_reason_aggregate.get(env_id, {}).items()):
            lines.append(
                f'| {env_id} | `{reason}` | '
                f'{summary["mean"]:.4f} ± {summary["sample_std"]:.4f} |',
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
            (
                'Point-task risk figures use simulator x/y coordinates. Ant and Walker2d '
                'figures use observation dimensions 0/1 and are explicitly projections, '
                'not physical XY trajectories.'
            ),
            '',
            *[f'- [`{path.name}`](figures/{path.name})' for path in figures],
            '',
            '## Recovery evidence',
            '',
            (
                'Audited recovered training runs: '
                f'`{recovery["recovered_training_run_count"]}/20`.'
            ),
            '',
            (
                '`shapo_p47_evaluation_manifest.csv`, `shapo_p47_provenance_manifest.csv`, and '
                '`shapo_p47_checkpoint_manifest.csv` retain validated SHA-256 identities. '
                'Compact evaluations and run provenance are copied into this result bundle; '
                'large replay-bearing checkpoint payloads remain server-only.'
            ),
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
    evaluation_sources: list[dict[str, Any]] = []
    trajectory_sources: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    provenance_sources: list[dict[str, Any]] = []
    source_commits = set()
    matrix_sha256 = sha256_file(args.matrix)
    validator_sha256 = sha256_file(STRICT_VALIDATOR)
    plotter_sha256 = sha256_file(PLOTTER)
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
            expected_validation_identity = {
                'validator_sha256': validator_sha256,
                'protocol_version': matrix['protocol_version'],
                'protocol_sha256': matrix['protocol_sha256'],
                'matrix_sha256': matrix_sha256,
                'frozen_method_canonical_sha256': matrix['frozen_method_canonical_sha256'],
                'training_seed_is_statistical_unit': True,
                'total_steps': int(environment['total_steps']),
                'formal_cost_budget': float(environment['cost_limit']),
                'geometry_mode': str(environment['geometry_mode']),
                'cost_normalization_scale': float(environment['cost_normalization_scale']),
            }
            for key, expected in expected_validation_identity.items():
                if validation.get(key) != expected:
                    raise ValueError(
                        f'{validation_path} field {key!r} does not match the locked matrix',
                    )
            source_commits.add(str(validation['source_commit']))
            evaluations = validation['evaluation_rows']
            if [int(row['milestone']) for row in evaluations] != expected_milestones:
                raise ValueError(f'{validation_path} has incomplete learning-curve milestones')
            for evaluation in evaluations:
                evaluation_identity = {
                    'artifact': evaluation['artifact'],
                    'sha256': evaluation['artifact_sha256'],
                    'bytes': evaluation['artifact_bytes'],
                }
                verify_artifact_identity(
                    evaluation_identity,
                    f'{env_id} seed {seed} milestone {evaluation["milestone"]} evaluation',
                )
                evaluation_sources.append(
                    {
                        'environment': env_id,
                        'training_seed': seed,
                        'milestone': int(evaluation['milestone']),
                        **evaluation_identity,
                    },
                )
                curve_rows.append(
                    {
                        'environment': env_id,
                        'training_seed': seed,
                        **evaluation,
                    },
                )
            provenance = validation.get('compact_provenance_artifacts', {})
            expected_provenance_labels = {
                'resolved_config',
                'receipt',
                'training_summary',
                'run_manifest',
            }
            if set(provenance) != expected_provenance_labels:
                raise ValueError(f'{validation_path} has incomplete compact provenance')
            for label, identity in provenance.items():
                verify_artifact_identity(identity, f'{env_id} seed {seed} {label}')
                provenance_sources.append(
                    {
                        'environment': env_id,
                        'training_seed': seed,
                        'label': label,
                        **identity,
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
            checkpoint_artifacts = validation.get('checkpoint_artifacts', [])
            expected_checkpoint_labels = {
                'latest_full',
                'final_full',
                'best_safe_return',
                *[f'milestone_{step}' for step in expected_milestones],
            }
            labels = [str(row['label']) for row in checkpoint_artifacts]
            if len(labels) != len(set(labels)) or set(labels) != expected_checkpoint_labels:
                raise ValueError(f'{validation_path} has incomplete checkpoint identities')
            for checkpoint in checkpoint_artifacts:
                verify_artifact_identity(
                    checkpoint,
                    f'{env_id} seed {seed} {checkpoint["label"]} checkpoint',
                )
                checkpoint_rows.append(
                    {
                        'environment': env_id,
                        'training_seed': seed,
                        'server_only': True,
                        **checkpoint,
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
    if (
        len(evaluation_sources) != 60
        or len(trajectory_sources) != 60
        or len(checkpoint_rows) != 120
        or len(provenance_sources) != 80
    ):
        raise ValueError('formal matrix has incomplete evaluation or recovery evidence')
    recovery = summarize_checkpoint_recovery(checkpoint_rows)

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

    failure_reason_rows = []
    failure_reason_aggregate: dict[str, dict[str, Any]] = {}
    for env_id in ('Ant-v4', 'Walker2d-v4'):
        selected = [row for row in final_rows if row['environment'] == env_id]
        reasons = sorted(
            {str(reason) for row in selected for reason in row.get('failure_reason_counts', {})},
        )
        failure_reason_aggregate[env_id] = {}
        for reason in reasons:
            rates = []
            for row in selected:
                episodes = int(row['episodes'])
                count = int(row.get('failure_reason_counts', {}).get(reason, 0))
                rate = count / episodes
                rates.append(rate)
                failure_reason_rows.append(
                    {
                        'environment': env_id,
                        'training_seed': row['training_seed'],
                        'reason': reason,
                        'count': count,
                        'episodes': episodes,
                        'rate': rate,
                    },
                )
            failure_reason_aggregate[env_id][reason] = seed_summary(rates)

    output = args.output_dir
    evaluation_manifest_rows = []
    for row in evaluation_sources:
        source = Path(row['artifact'])
        destination = (
            output
            / 'evaluations'
            / SLUGS[row['environment']]
            / f"training_seed_{row['training_seed']}"
            / f"evaluation_step_{row['milestone']}.json"
        )
        copy_verified(source, destination, str(row['sha256']))
        evaluation_manifest_rows.append(
            {
                'environment': row['environment'],
                'training_seed': row['training_seed'],
                'milestone': row['milestone'],
                'artifact': destination.relative_to(output).as_posix(),
                'sha256': row['sha256'],
                'bytes': row['bytes'],
            },
        )

    provenance_manifest_rows = []
    for row in provenance_sources:
        source = Path(row['artifact'])
        destination = (
            output
            / 'provenance'
            / SLUGS[row['environment']]
            / f"training_seed_{row['training_seed']}"
            / f"{row['label']}{source.suffix}"
        )
        copy_verified(source, destination, str(row['sha256']))
        provenance_manifest_rows.append(
            {
                'environment': row['environment'],
                'training_seed': row['training_seed'],
                'label': row['label'],
                'artifact': destination.relative_to(output).as_posix(),
                'sha256': row['sha256'],
                'bytes': row['bytes'],
            },
        )

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
    write_csv(output / 'shapo_p47_failure_reason_seed_metrics.csv', failure_reason_rows)
    write_csv(output / 'shapo_p47_evaluation_manifest.csv', evaluation_manifest_rows)
    write_csv(output / 'shapo_p47_provenance_manifest.csv', provenance_manifest_rows)
    write_csv(output / 'shapo_p47_trajectory_manifest.csv', trajectory_rows)
    write_csv(output / 'shapo_p47_checkpoint_manifest.csv', checkpoint_rows)
    from scripts.star import plot_shapo_p47_formal  # pylint: disable=import-outside-toplevel

    imported_plotter = Path(plot_shapo_p47_formal.__file__).resolve()
    if imported_plotter != PLOTTER or sha256_file(imported_plotter) != plotter_sha256:
        raise ValueError('formal plotter import does not match the locked plotter module')
    figures = plot_shapo_p47_formal.plot_all(output, output / 'figures')
    write_json(
        output / 'shapo_p47_final_summary.json',
        {
            'protocol_version': matrix['protocol_version'],
            'protocol_sha256': matrix['protocol_sha256'],
            'matrix_sha256': matrix_sha256,
            'source_commit': source_commit,
            'validator_sha256': validator_sha256,
            'plotter_sha256': plotter_sha256,
            'statistical_unit': 'training_seed',
            'n_training_seeds_per_environment': 5,
            'episodes_per_training_seed': 100,
            'aggregate': aggregate,
            'failure_reason_aggregate': failure_reason_aggregate,
            'evaluation_artifact_count': len(evaluation_manifest_rows),
            'provenance_artifact_count': len(provenance_manifest_rows),
            'trajectory_artifact_count': len(trajectory_rows),
            'checkpoint_artifact_count': len(checkpoint_rows),
            'checkpoint_recovery': recovery,
            'figures': [path.relative_to(output).as_posix() for path in figures],
        },
    )
    atomic_text(
        output / 'SHAPO_P47_FOUR_ENV_REPORT.md',
        render_report(
            aggregate,
            failure_reason_aggregate,
            recovery,
            source_commit,
            validator_sha256,
            plotter_sha256,
            figures,
        ),
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

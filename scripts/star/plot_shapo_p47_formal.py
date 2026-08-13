#!/usr/bin/env python3
"""Render frozen P4.7 SHAPO primary, mechanism, and trajectory-risk figures."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use('Agg')
plt = importlib.import_module('matplotlib.pyplot')
patches = importlib.import_module('matplotlib.patches')


ENVIRONMENTS = (
    'SafetyPointGoal1-v0',
    'SafetyPointButton1-v0',
    'Ant-v4',
    'Walker2d-v4',
)
SLUGS = {
    'SafetyPointGoal1-v0': 'point-goal',
    'SafetyPointButton1-v0': 'point-button',
    'Ant-v4': 'ant',
    'Walker2d-v4': 'walker2d',
}
PRIMARY_METRICS = (
    ('mean_return', 'Evaluation return'),
    ('mean_cost', 'Evaluation mean cost'),
    ('cost_cvar_20', 'Evaluation CVaR-20'),
    ('training_cumulative_cost', 'Training cumulative cost'),
)
MECHANISM_METRICS = (
    ('high_risk_vn_vt', 'High-risk $V_N/V_T$'),
    (
        'high_risk_dangerous_outward_velocity',
        'High-risk dangerous outward velocity',
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one compact finalization CSV."""
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def _finite(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _curve(axis: Any, rows: list[dict[str, str]], metric: str) -> None:
    """Draw all seed curves plus the training-seed mean and sample deviation."""
    seeds = sorted({int(row['training_seed']) for row in rows})
    for seed in seeds:
        selected = sorted(
            (row for row in rows if int(row['training_seed']) == seed),
            key=lambda row: int(row['milestone']),
        )
        points = [(_finite(row, metric), int(row['milestone'])) for row in selected]
        points = [(value, step) for value, step in points if value is not None]
        if points:
            axis.plot(
                [step / 1_000_000 for _, step in points],
                [value for value, _ in points],
                color='#4c78a8',
                linewidth=0.8,
                alpha=0.25,
            )

    grouped: dict[int, list[float]] = {}
    for row in rows:
        value = _finite(row, metric)
        if value is not None:
            grouped.setdefault(int(row['milestone']), []).append(value)
    steps = sorted(grouped)
    means = [statistics.fmean(grouped[step]) for step in steps]
    deviations = [
        statistics.stdev(grouped[step]) if len(grouped[step]) > 1 else 0.0 for step in steps
    ]
    x_values = [step / 1_000_000 for step in steps]
    axis.plot(x_values, means, color='#1f4e79', marker='o', linewidth=2.0)
    axis.fill_between(
        x_values,
        [mean - deviation for mean, deviation in zip(means, deviations)],
        [mean + deviation for mean, deviation in zip(means, deviations)],
        color='#4c78a8',
        alpha=0.18,
    )
    axis.set_xlabel('Environment steps (millions)')
    axis.grid(alpha=0.25)


def plot_learning_curves(summary_dir: Path, output_dir: Path) -> list[Path]:
    """Render primary and diagnostic learning curves across five training seeds."""
    rows = read_csv(summary_dir / 'shapo_p47_learning_curves.csv')
    outputs: list[Path] = []
    for metrics, filename, height in (
        (PRIMARY_METRICS, 'shapo_p47_primary_learning_curves.png', 14.0),
        (MECHANISM_METRICS, 'shapo_p47_mechanism_learning_curves.png', 7.5),
    ):
        figure, axes = plt.subplots(
            len(metrics),
            len(ENVIRONMENTS),
            figsize=(18, height),
            squeeze=False,
        )
        for column, env_id in enumerate(ENVIRONMENTS):
            selected = [row for row in rows if row['environment'] == env_id]
            for row_index, (metric, label) in enumerate(metrics):
                axis = axes[row_index, column]
                _curve(axis, selected, metric)
                if row_index == 0:
                    axis.set_title(env_id)
                if column == 0:
                    axis.set_ylabel(label)
        figure.suptitle('Frozen P4.7: faint lines are training seeds; band is mean ± sample SD')
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(figure)
        outputs.append(path)
    return outputs


def _trajectory_groups(path: Path) -> dict[int, list[dict[str, str]]]:
    rows = read_csv(path)
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(float(row['base_seed'])), []).append(row)
    return grouped


def trajectory_coordinate_contract(
    env_id: str,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Identify and validate the coordinate system used by a risk trace."""
    point_task = env_id.startswith('SafetyPoint')
    expected_source = 'simulator_xy' if point_task else 'observation_projection'
    observed_sources = {row.get('position_source', '') for row in rows}
    if observed_sources != {expected_source}:
        raise ValueError(
            f'{env_id} trajectory coordinate source mismatch: expected '
            f'{expected_source!r}, observed {sorted(observed_sources)!r}',
        )
    if point_task:
        return {
            'point_task': True,
            'x_label': 'Simulator x position',
            'y_label': 'Simulator y position',
            'title_kind': 'simulator-XY trajectories',
        }
    return {
        'point_task': False,
        'x_label': 'Observation dimension 0',
        'y_label': 'Observation dimension 1',
        'title_kind': 'observation-space projections (not physical XY trajectories)',
    }


def _draw_point_layout(axis: Any, rows: list[dict[str, str]]) -> None:
    first = rows[0]
    hazards = json.loads(first.get('hazards_json') or '[]')
    hazard_radius = _finite(first, 'hazard_radius')
    for x_value, y_value in hazards:
        axis.add_patch(
            patches.Circle(
                (float(x_value), float(y_value)),
                hazard_radius or 0.2,
                color='#d62728',
                alpha=0.16,
            ),
        )
    goal_x = _finite(first, 'goal_x')
    goal_y = _finite(first, 'goal_y')
    if goal_x is not None and goal_y is not None:
        axis.add_patch(
            patches.Circle(
                (goal_x, goal_y),
                _finite(first, 'goal_radius') or 0.2,
                color='#2ca02c',
                alpha=0.22,
            ),
        )


def _draw_trajectory(axis: Any, rows: list[dict[str, str]], point_task: bool) -> Any:
    x_values = [float(row['x']) for row in rows]
    y_values = [float(row['y']) for row in rows]
    risks = [float(row['predicted_violation_probability']) for row in rows]
    axis.plot(x_values, y_values, color='#9e9e9e', linewidth=0.8, alpha=0.8)
    scatter = axis.scatter(
        x_values,
        y_values,
        c=risks,
        cmap='viridis',
        vmin=0.0,
        vmax=1.0,
        s=8,
        zorder=3,
    )
    unsafe = [
        index
        for index, row in enumerate(rows)
        if float(row['cost']) > 0.0 or int(float(row['is_failure'])) == 1
    ]
    if unsafe:
        axis.scatter(
            [x_values[index] for index in unsafe],
            [y_values[index] for index in unsafe],
            marker='x',
            color='#d62728',
            s=24,
            linewidth=1.0,
            zorder=4,
        )
    axis.scatter([x_values[0]], [y_values[0]], marker='o', color='black', s=16, zorder=5)
    axis.scatter([x_values[-1]], [y_values[-1]], marker='*', color='black', s=36, zorder=5)
    if point_task:
        _draw_point_layout(axis, rows)
        axis.set_aspect('equal', adjustable='datalim')
    axis.grid(alpha=0.16)
    return scatter


def plot_trajectory_risk(summary_dir: Path, output_dir: Path) -> list[Path]:
    """Render final fixed-seed risk traces with explicit coordinate semantics."""
    manifest = read_csv(summary_dir / 'shapo_p47_trajectory_manifest.csv')
    outputs: list[Path] = []
    for env_id in ENVIRONMENTS:
        env_rows = [row for row in manifest if row['environment'] == env_id]
        final_by_training_seed: dict[int, dict[str, str]] = {}
        for row in env_rows:
            seed = int(row['training_seed'])
            if seed not in final_by_training_seed or int(row['milestone']) > int(
                final_by_training_seed[seed]['milestone'],
            ):
                final_by_training_seed[seed] = row
        training_seeds = sorted(final_by_training_seed)
        grouped = {
            seed: _trajectory_groups(summary_dir / final_by_training_seed[seed]['artifact'])
            for seed in training_seeds
        }
        coordinate_contracts = [
            trajectory_coordinate_contract(env_id, trace)
            for groups in grouped.values()
            for trace in groups.values()
        ]
        if not coordinate_contracts:
            raise ValueError(f'{env_id} has no final trajectory traces')
        coordinate_contract = coordinate_contracts[0]
        if any(contract != coordinate_contract for contract in coordinate_contracts[1:]):
            raise ValueError(f'{env_id} trajectory coordinate contracts are inconsistent')
        evaluation_seeds = sorted({seed for groups in grouped.values() for seed in groups})
        figure, axes = plt.subplots(
            len(training_seeds),
            len(evaluation_seeds),
            figsize=(4.8 * len(evaluation_seeds), 3.4 * len(training_seeds)),
            squeeze=False,
        )
        scatter = None
        for row_index, training_seed in enumerate(training_seeds):
            for column, evaluation_seed in enumerate(evaluation_seeds):
                axis = axes[row_index, column]
                trace = grouped[training_seed].get(evaluation_seed)
                if not trace:
                    raise ValueError(
                        f'{env_id} training seed {training_seed} lacks trajectory '
                        f'{evaluation_seed}',
                    )
                scatter = _draw_trajectory(
                    axis,
                    trace,
                    bool(coordinate_contract['point_task']),
                )
                if row_index == 0:
                    axis.set_title(f'Evaluation seed {evaluation_seed}')
                if column == 0:
                    axis.set_ylabel(
                        f'Training seed {training_seed}\n{coordinate_contract["y_label"]}',
                    )
                if row_index == len(training_seeds) - 1:
                    axis.set_xlabel(str(coordinate_contract['x_label']))
        if scatter is not None:
            figure.colorbar(
                scatter,
                ax=axes.ravel().tolist(),
                fraction=0.015,
                pad=0.01,
                label='Predicted violation probability',
            )
        figure.suptitle(
            f'{env_id}: final fixed-seed {coordinate_contract["title_kind"]} '
            '(red x = cost/failure; star = endpoint)',
        )
        figure.subplots_adjust(top=0.95, right=0.91, hspace=0.28, wspace=0.22)
        path = output_dir / f'shapo_p47_{SLUGS[env_id]}_trajectory_risk.png'
        figure.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(figure)
        outputs.append(path)
    return outputs


def plot_all(summary_dir: Path, output_dir: Path) -> list[Path]:
    """Render all preregistered primary and diagnostic figure families."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        *plot_learning_curves(summary_dir, output_dir),
        *plot_trajectory_risk(summary_dir, output_dir),
    ]


def main() -> None:
    """Run the portable final-summary plotter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.summary_dir / 'figures'
    outputs = plot_all(args.summary_dir, output_dir)
    print(json.dumps({'plots': [str(path) for path in outputs]}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

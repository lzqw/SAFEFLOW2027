#!/usr/bin/env python3
"""Plot the four P4.5 learning-curve families and three locked trajectory seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


matplotlib.use('Agg')


ARMS = ('C0', 'C1')
SEEDS = (2027, 2028, 2029)
TRAJECTORY_SEEDS = (12_027, 12_036, 12_046)
COLORS = {'C0': '#1f77b4', 'C1': '#ff7f0e'}


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read one CSV artifact."""
    with path.open('r', encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def _number(row: dict[str, str], key: str) -> float:
    """Parse one finite plotting value."""
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f'{key} is not finite in plotting evidence')
    return value


def _group(
    rows: list[dict[str, str]],
    step_key: str,
) -> dict[tuple[str, int], list[dict[str, str]]]:
    """Group rows by arm/training seed and sort by step."""
    result: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        result[(row['arm'], int(row['training_seed']))].append(row)
    for group in result.values():
        group.sort(key=lambda row: _number(row, step_key))
    return dict(result)


def _plot_metric(
    axis: Any,
    grouped: dict[tuple[str, int], list[dict[str, str]]],
    step_key: str,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    """Plot thin seed curves and a thick paired-arm mean."""
    for arm in ARMS:
        arm_series = []
        for seed in SEEDS:
            rows = grouped[(arm, seed)]
            x = [_number(row, step_key) / 1000.0 for row in rows]
            y = [_number(row, metric) for row in rows]
            arm_series.append(y)
            axis.plot(x, y, color=COLORS[arm], alpha=0.28, linewidth=1.0)
        mean = [sum(values) / len(values) for values in zip(*arm_series)]
        axis.plot(x, mean, color=COLORS[arm], linewidth=2.4, label=arm)
    axis.set_title(title)
    axis.set_xlabel('environment steps (k)')
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)


def _save_figure(figure: Any, path: Path) -> None:
    """Save a tight deterministic PNG and close its resources."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def plot_learning_curves(summary_dir: Path, output_dir: Path) -> list[Path]:
    """Create the four preregistered learning-curve figures."""
    evaluation = _group(
        _read_csv(summary_dir / 'p4_5_checkpoint_metrics.csv'),
        'milestone',
    )
    diagnostics = _group(
        _read_csv(summary_dir / 'p4_5_diagnostics_combined.csv'),
        'TotalEnvSteps',
    )
    outputs = []

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _plot_metric(axes[0], evaluation, 'milestone', 'mean_return', 'Evaluation return', 'return')
    _plot_metric(axes[1], evaluation, 'milestone', 'success_rate', 'Goal success', 'rate')
    axes[0].legend()
    path = output_dir / 'p4_5_return_success.png'
    _save_figure(figure, path)
    outputs.append(path)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    _plot_metric(axes[0], evaluation, 'milestone', 'mean_cost', 'Evaluation mean cost', 'cost')
    _plot_metric(axes[1], evaluation, 'milestone', 'cost_cvar_20', 'Cost CVaR-20', 'cost')
    _plot_metric(
        axes[2],
        evaluation,
        'milestone',
        'training_cumulative_cost',
        'Training cumulative cost',
        'cost',
    )
    axes[0].legend()
    path = output_dir / 'p4_5_cost_cvar_training_cost.png'
    _save_figure(figure, path)
    outputs.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _plot_metric(
        axes[0],
        evaluation,
        'milestone',
        'dangerous_normal_velocity',
        'Dangerous Q-normal velocity',
        'positive normal speed',
    )
    _plot_metric(
        axes[1],
        evaluation,
        'milestone',
        'normal_to_tangent_variance_ratio',
        'Reference variance ratio',
        'V_N / V_T',
    )
    axes[0].legend()
    path = output_dir / 'p4_5_geometry.png'
    _save_figure(figure, path)
    outputs.append(path)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    _plot_metric(
        axes[0],
        diagnostics,
        'TotalEnvSteps',
        'Critic/RewardTD',
        'Reward critic TD',
        'absolute TD',
    )
    _plot_metric(
        axes[1],
        diagnostics,
        'TotalEnvSteps',
        'Critic/CostTD',
        'Cost critic TD',
        'absolute TD',
    )
    for arm in ARMS:
        for metric, style, label in (
            ('Critic/RewardQAbsMax', '-', 'reward Q'),
            ('Critic/RewardTargetQAbsMax', '--', 'reward target Q'),
            ('Critic/CostQAbsMax', '-.', 'cost Q'),
            ('Critic/CostTargetQAbsMax', ':', 'cost target Q'),
        ):
            series = []
            for seed in SEEDS:
                rows = diagnostics[(arm, seed)]
                x = [_number(row, 'TotalEnvSteps') / 1000.0 for row in rows]
                series.append([_number(row, metric) for row in rows])
            mean = [sum(values) / len(values) for values in zip(*series)]
            axes[2].plot(
                x,
                mean,
                color=COLORS[arm],
                linestyle=style,
                linewidth=1.8,
                label=f'{arm} {label}',
            )
    axes[2].set_title('Maximum absolute Q scale')
    axes[2].set_xlabel('environment steps (k)')
    axes[2].set_ylabel('|Q| max')
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=7, ncol=2)
    axes[0].legend()
    path = output_dir / 'p4_5_critic_scales.png'
    _save_figure(figure, path)
    outputs.append(path)
    return outputs


def _trajectory_paths(summary_dir: Path) -> dict[tuple[str, int], Path]:
    """Read the validated final-trajectory manifest."""
    paths = {}
    for row in _read_csv(summary_dir / 'p4_5_trajectory_manifest.csv'):
        key = (row['arm'], int(row['training_seed']))
        path = Path(row['path'])
        if not path.is_file():
            raise FileNotFoundError(f'trajectory path is unavailable: {path}')
        paths[key] = path
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(paths) != expected:
        raise ValueError('trajectory manifest does not cover all six P4.5 runs')
    return paths


def _plot_layout(axis: Any, rows: list[dict[str, str]]) -> None:
    """Plot the fixed evaluation goal and hazard layout."""
    first = rows[0]
    hazards = json.loads(first['hazards_json']) if first['hazards_json'] else []
    hazard_radius = float(first['hazard_radius'])
    for x_value, y_value in hazards:
        axis.add_patch(
            Circle(
                (float(x_value), float(y_value)),
                hazard_radius,
                facecolor='#d62728',
                edgecolor='none',
                alpha=0.22,
            ),
        )
    if first['goal_x'] and first['goal_y']:
        axis.scatter(
            [float(first['goal_x'])],
            [float(first['goal_y'])],
            marker='*',
            s=130,
            color='#2ca02c',
            edgecolors='black',
            linewidths=0.4,
            label='goal',
            zorder=6,
        )


def _subsample(rows: list[dict[str, str]], count: int = 18) -> list[dict[str, str]]:
    """Return approximately evenly spaced vector rows."""
    stride = max(1, len(rows) // count)
    return rows[::stride]


def plot_trajectories(summary_dir: Path, output_dir: Path) -> list[Path]:
    """Create one three-training-seed comparison for each locked evaluation seed."""
    paths = _trajectory_paths(summary_dir)
    tables = {key: _read_csv(path) for key, path in paths.items()}
    outputs = []
    for evaluation_seed in TRAJECTORY_SEEDS:
        figure, axes = plt.subplots(3, 1, figsize=(8.2, 17.5), squeeze=False)
        for row_index, training_seed in enumerate(SEEDS):
            axis = axes[row_index, 0]
            layout_rows = [
                row for row in tables[('C0', training_seed)] if int(row['seed']) == evaluation_seed
            ]
            if not layout_rows:
                raise ValueError(
                    f'C0 seed {training_seed} lacks evaluation trajectory {evaluation_seed}',
                )
            _plot_layout(axis, layout_rows)
            for arm in ARMS:
                rows = [
                    row
                    for row in tables[(arm, training_seed)]
                    if int(row['seed']) == evaluation_seed
                ]
                x = [float(row['x']) for row in rows]
                y = [float(row['y']) for row in rows]
                axis.plot(x, y, color=COLORS[arm], linewidth=1.6, label=f'{arm} path')
                cost_rows = [row for row in rows if float(row['cost']) > 0.0]
                if cost_rows:
                    axis.scatter(
                        [float(row['x']) for row in cost_rows],
                        [float(row['y']) for row in cost_rows],
                        color=COLORS[arm],
                        marker='x',
                        s=25,
                        linewidths=1.0,
                        label=f'{arm} cost',
                        zorder=7,
                    )
                vectors = _subsample(rows)
                vx = [float(row['x']) for row in vectors]
                vy = [float(row['y']) for row in vectors]
                axis.quiver(
                    vx,
                    vy,
                    [float(row['action_0']) for row in vectors],
                    [float(row['action_1']) for row in vectors],
                    color=COLORS[arm],
                    alpha=0.45,
                    angles='xy',
                    scale_units='xy',
                    scale=5.0,
                    width=0.003,
                )
                axis.quiver(
                    vx,
                    vy,
                    [float(row['cost_normal_0']) for row in vectors],
                    [float(row['cost_normal_1']) for row in vectors],
                    color='#d62728',
                    alpha=0.35,
                    angles='xy',
                    scale_units='xy',
                    scale=8.0,
                    width=0.002,
                )
                axis.quiver(
                    vx,
                    vy,
                    [float(row['cost_tangent_0']) for row in vectors],
                    [float(row['cost_tangent_1']) for row in vectors],
                    color='#2ca02c',
                    alpha=0.30,
                    angles='xy',
                    scale_units='xy',
                    scale=8.0,
                    width=0.002,
                )
            axis.set_title(
                f'training seed {training_seed}; fixed evaluation seed {evaluation_seed}',
            )
            axis.set_aspect('equal', adjustable='box')
            axis.set_xlabel('x')
            axis.set_ylabel('y')
            axis.grid(alpha=0.18)
            axis.legend(fontsize=8, ncol=3, loc='best')
        figure.suptitle(
            'C0/C1 paths; arrows: action (arm color), Q-cost normal (red), tangent (green)',
            y=0.998,
        )
        path = output_dir / f'p4_5_trajectory_eval_seed_{evaluation_seed}.png'
        _save_figure(figure, path)
        outputs.append(path)
    return outputs


def parse_args() -> argparse.Namespace:
    """Parse summary and output directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Generate all seven preregistered PNGs."""
    args = parse_args()
    outputs = [
        *plot_learning_curves(args.summary_dir, args.output_dir),
        *plot_trajectories(args.summary_dir, args.output_dir),
    ]
    print(json.dumps({'plots': [str(path) for path in outputs]}, indent=2))


if __name__ == '__main__':
    main()

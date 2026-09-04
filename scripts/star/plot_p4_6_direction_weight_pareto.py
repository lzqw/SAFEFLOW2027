#!/usr/bin/env python3
"""Plot the P4.6 four-weight curves, Pareto points, and fixed trajectories."""

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


ARMS = ('C0', 'C025', 'C050', 'C1')
SEEDS = (2027, 2028, 2029)
TRAJECTORY_SEEDS = (12_027, 12_036, 12_046)
WEIGHTS = {'C0': 0.0, 'C025': 0.25, 'C050': 0.50, 'C1': 1.0}
COLORS = {'C0': '#1f77b4', 'C025': '#2ca02c', 'C050': '#ff7f0e', 'C1': '#d62728'}


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read one CSV artifact."""
    with path.open('r', encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def _number(row: dict[str, Any], key: str) -> float:
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
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(result) != expected:
        raise ValueError('plotting evidence does not cover all four weights and three seeds')
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
    """Plot thin seed curves and one thick mean for each weight."""
    for arm in ARMS:
        series = []
        x = []
        for seed in SEEDS:
            rows = grouped[(arm, seed)]
            x = [_number(row, step_key) / 1000.0 for row in rows]
            values = [_number(row, metric) for row in rows]
            series.append(values)
            axis.plot(x, values, color=COLORS[arm], alpha=0.20, linewidth=0.9)
        mean = [sum(values) / len(values) for values in zip(*series)]
        axis.plot(x, mean, color=COLORS[arm], linewidth=2.2, label=f'{arm} ({WEIGHTS[arm]:.2f})')
    axis.set_title(title)
    axis.set_xlabel('environment steps (k)')
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)


def _save(figure: Any, path: Path) -> Path:
    """Save one deterministic tight PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(figure)
    return path


def plot_learning_curves(summary_dir: Path, output_dir: Path) -> list[Path]:
    """Create the four locked learning-curve families."""
    evaluation = _group(_read_csv(summary_dir / 'p4_6_checkpoint_metrics.csv'), 'milestone')
    diagnostics = _group(
        _read_csv(summary_dir / 'p4_6_diagnostics_combined.csv'),
        'TotalEnvSteps',
    )
    outputs = []

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    _plot_metric(axes[0], evaluation, 'milestone', 'mean_return', 'Evaluation return', 'return')
    _plot_metric(axes[1], evaluation, 'milestone', 'success_rate', 'Goal success', 'rate')
    axes[0].legend(fontsize=8)
    outputs.append(_save(figure, output_dir / 'p4_6_return_success.png'))

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.4))
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
    axes[0].legend(fontsize=8)
    outputs.append(_save(figure, output_dir / 'p4_6_cost_cvar_training_cost.png'))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
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
    axes[0].legend(fontsize=8)
    outputs.append(_save(figure, output_dir / 'p4_6_geometry.png'))

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.4))
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
            x = []
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
                linewidth=1.4,
                label=f'{arm} {label}',
            )
    axes[2].set_title('Maximum absolute Q scale')
    axes[2].set_xlabel('environment steps (k)')
    axes[2].set_ylabel('|Q| max')
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=6, ncol=2)
    axes[0].legend(fontsize=8)
    outputs.append(_save(figure, output_dir / 'p4_6_critic_scales.png'))
    return outputs


def plot_final_pareto(summary_dir: Path, output_dir: Path) -> Path:
    """Plot per-seed and exactly pooled final safety-performance points."""
    summary = json.loads((summary_dir / 'p4_6_final_summary.json').read_text(encoding='utf-8'))
    pooled = summary['decision']['pooled']
    seed_rows = _read_csv(summary_dir / 'p4_6_final_seed_metrics.csv')
    panels = (
        ('mean_cost', 'mean_return', 'Mean cost', 'Return'),
        ('cost_cvar_20', 'mean_return', 'CVaR-20', 'Return'),
        ('mean_cost', 'success_rate', 'Mean cost', 'Success'),
        ('cost_cvar_20', 'success_rate', 'CVaR-20', 'Success'),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, (x_key, y_key, x_label, y_label) in zip(axes.flat, panels):
        for arm in ARMS:
            arm_rows = [row for row in seed_rows if row['arm'] == arm]
            axis.scatter(
                [_number(row, x_key) for row in arm_rows],
                [_number(row, y_key) for row in arm_rows],
                color=COLORS[arm],
                alpha=0.25,
                s=28,
            )
            axis.scatter(
                [float(pooled[arm][x_key])],
                [float(pooled[arm][y_key])],
                color=COLORS[arm],
                edgecolors='black',
                linewidths=0.5,
                s=95,
                label=f'{arm} λ={WEIGHTS[arm]:.2f}',
                zorder=5,
            )
        axis.set_xlabel(f'{x_label} (lower is safer)')
        axis.set_ylabel(f'{y_label} (higher is better)')
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle('P4.6 final four-weight Pareto: faint=seed, solid=pooled 60 episodes')
    return _save(figure, output_dir / 'p4_6_final_pareto.png')


def _trajectory_paths(summary_dir: Path) -> dict[tuple[str, int], Path]:
    """Resolve the validated final trajectory manifest."""
    result = {}
    for row in _read_csv(summary_dir / 'p4_6_trajectory_manifest.csv'):
        path = Path(row['path'])
        if not path.is_file():
            raise FileNotFoundError(f'trajectory path is unavailable: {path}')
        result[(row['arm'], int(row['training_seed']))] = path
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(result) != expected:
        raise ValueError('trajectory manifest does not cover all 12 weight-seed runs')
    return result


def _plot_layout(axis: Any, rows: list[dict[str, str]]) -> None:
    """Plot the fixed PointGoal goal and hazards."""
    first = rows[0]
    hazards = json.loads(first['hazards_json']) if first['hazards_json'] else []
    for x_value, y_value in hazards:
        axis.add_patch(
            Circle(
                (float(x_value), float(y_value)),
                float(first['hazard_radius']),
                facecolor='#d62728',
                edgecolor='none',
                alpha=0.18,
            ),
        )
    axis.scatter(
        [float(first['goal_x'])],
        [float(first['goal_y'])],
        marker='*',
        s=120,
        color='#2ca02c',
        edgecolors='black',
        linewidths=0.4,
        label='goal',
        zorder=7,
    )


def _subsample(rows: list[dict[str, str]], count: int = 8) -> list[dict[str, str]]:
    """Return approximately evenly spaced vector rows."""
    return rows[:: max(1, len(rows) // count)]


def plot_trajectories(summary_dir: Path, output_dir: Path) -> list[Path]:
    """Overlay all four weights for each fixed evaluation/training seed pair."""
    paths = _trajectory_paths(summary_dir)
    tables = {key: _read_csv(path) for key, path in paths.items()}
    outputs = []
    for evaluation_seed in TRAJECTORY_SEEDS:
        figure, axes = plt.subplots(3, 1, figsize=(8.5, 17.5), squeeze=False)
        for row_index, training_seed in enumerate(SEEDS):
            axis = axes[row_index, 0]
            layout = [
                row for row in tables[('C0', training_seed)] if int(row['seed']) == evaluation_seed
            ]
            if not layout:
                raise ValueError(f'C0-{training_seed} lacks trajectory {evaluation_seed}')
            _plot_layout(axis, layout)
            for arm in ARMS:
                rows = [
                    row
                    for row in tables[(arm, training_seed)]
                    if int(row['seed']) == evaluation_seed
                ]
                axis.plot(
                    [float(row['x']) for row in rows],
                    [float(row['y']) for row in rows],
                    color=COLORS[arm],
                    linewidth=1.5,
                    label=f'{arm} λ={WEIGHTS[arm]:.2f}',
                )
                costs = [row for row in rows if float(row['cost']) > 0.0]
                if costs:
                    axis.scatter(
                        [float(row['x']) for row in costs],
                        [float(row['y']) for row in costs],
                        color=COLORS[arm],
                        marker='x',
                        s=20,
                        zorder=6,
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
                    alpha=0.35,
                    angles='xy',
                    scale_units='xy',
                    scale=5.0,
                    width=0.0025,
                )
                axis.quiver(
                    vx,
                    vy,
                    [float(row['cost_normal_0']) for row in vectors],
                    [float(row['cost_normal_1']) for row in vectors],
                    color='#7f0000',
                    alpha=0.18,
                    angles='xy',
                    scale_units='xy',
                    scale=8.0,
                    width=0.0018,
                )
                axis.quiver(
                    vx,
                    vy,
                    [float(row['cost_tangent_0']) for row in vectors],
                    [float(row['cost_tangent_1']) for row in vectors],
                    color='#006d2c',
                    alpha=0.16,
                    angles='xy',
                    scale_units='xy',
                    scale=8.0,
                    width=0.0018,
                )
            axis.set_title(f'training seed {training_seed}; evaluation seed {evaluation_seed}')
            axis.set_aspect('equal', adjustable='box')
            axis.set_xlabel('x')
            axis.set_ylabel('y')
            axis.grid(alpha=0.18)
            axis.legend(fontsize=7, ncol=3, loc='best')
        figure.suptitle(
            'Four-weight paths; arrows: action (arm), Q-cost normal (dark red), tangent (green)',
            y=0.998,
        )
        outputs.append(
            _save(figure, output_dir / f'p4_6_trajectory_eval_seed_{evaluation_seed}.png'),
        )
    return outputs


def parse_args() -> argparse.Namespace:
    """Parse summary and output directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Generate all eight preregistered P4.6 figures."""
    args = parse_args()
    outputs = [
        *plot_learning_curves(args.summary_dir, args.output_dir),
        plot_final_pareto(args.summary_dir, args.output_dir),
        *plot_trajectories(args.summary_dir, args.output_dir),
    ]
    print(json.dumps({'plots': [str(path) for path in outputs]}, indent=2))


if __name__ == '__main__':
    main()

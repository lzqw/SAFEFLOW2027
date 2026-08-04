#!/usr/bin/env python3
"""Plot compact STAR training, evaluation, counterfactual, and trajectory diagnostics."""

# The two standalone visualization CLIs intentionally share compact layout-rendering logic.
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


plt.switch_backend('Agg')


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one CSV artifact."""
    if not path.exists():
        return []
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], key: str) -> float:
    """Return a finite field or NaN for plotting gaps."""
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return float('nan')
    return value if math.isfinite(value) else float('nan')


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a compact public CSV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_label(run_dir: Path) -> str:
    """Return a short stable label from the seed directory name."""
    return run_dir.name.replace('seed-', '')


def evaluation_rows(run_dir: Path) -> list[dict[str, str]]:
    """Read fixed-milestone evaluation rows."""
    return read_csv(run_dir / 'artifacts/evaluation.csv')


def checkpoint_evaluation_rows(path: Path) -> dict[int, list[dict[str, str]]]:
    """Group tie-correct fixed-device checkpoint evaluations by training seed."""
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in read_csv(path):
        grouped.setdefault(int(row['training_seed']), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row['milestone']))
    return grouped


def run_seed(run_dir: Path) -> int:
    """Read a run's training seed from its compact summary."""
    summary = json.loads(
        (run_dir / 'artifacts/training_summary.json').read_text(encoding='utf-8'),
    )
    return int(summary['seed'])


def plot_line(
    axes: Any,
    x_values: list[float],
    y_values: list[float],
    label: str,
    *,
    linestyle: str = '-',
    marker: str | None = None,
) -> None:
    """Plot only when at least one finite point exists."""
    if any(math.isfinite(value) for value in y_values):
        axes.plot(
            x_values,
            y_values,
            label=label,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.5,
        )


def plot_training(  # pylint: disable=too-many-locals
    run_dirs: list[Path],
    output: Path,
    fixed_evaluations: dict[int, list[dict[str, str]]] | None = None,
) -> None:
    """Render the required scalar mechanism diagnostics in one nine-panel figure."""
    figure, panels = plt.subplots(3, 3, figsize=(14.0, 10.5), constrained_layout=True)
    axes = panels.flatten()
    for run_dir in run_dirs:
        label = run_label(run_dir)
        diagnostics = read_csv(run_dir / 'artifacts/diagnostics.csv')
        if fixed_evaluations is None:
            evaluations = evaluation_rows(run_dir)
        else:
            seed = run_seed(run_dir)
            if seed not in fixed_evaluations:
                raise ValueError(f'checkpoint evaluations are missing training seed {seed}')
            evaluations = fixed_evaluations[seed]
        steps = [number(row, 'TotalEnvSteps') for row in diagnostics]
        milestones = [number(row, 'milestone') for row in evaluations]

        plot_line(
            axes[0],
            steps,
            [number(row, 'Metrics/EpRet') for row in diagnostics],
            f'train {label}',
        )
        plot_line(
            axes[0],
            milestones,
            [number(row, 'mean_return') for row in evaluations],
            f'eval {label}',
            linestyle='--',
            marker='o',
        )
        plot_line(
            axes[1],
            steps,
            [number(row, 'Metrics/EpCost') for row in diagnostics],
            f'train {label}',
        )
        plot_line(
            axes[1],
            milestones,
            [number(row, 'mean_cost') for row in evaluations],
            f'eval {label}',
            linestyle='--',
            marker='o',
        )
        plot_line(
            axes[1],
            milestones,
            [number(row, 'cost_cvar_20') for row in evaluations],
            f'CVaR {label}',
            linestyle=':',
            marker='s',
        )
        plot_line(
            axes[2],
            steps,
            [number(row, 'Replay/ViolationRatio') for row in diagnostics],
            f'train {label}',
        )
        plot_line(
            axes[2],
            milestones,
            [number(row, 'violation_ratio') for row in evaluations],
            f'eval {label}',
            linestyle='--',
            marker='o',
        )
        plot_line(
            axes[3],
            steps,
            [number(row, 'Critic/CostTD') for row in diagnostics],
            f'cost {label}',
        )
        plot_line(
            axes[3],
            steps,
            [number(row, 'Critic/RewardTD') for row in diagnostics],
            f'reward {label}',
        )
        plot_line(
            axes[4],
            steps,
            [number(row, 'Critic/CalibrationError') for row in diagnostics],
            f'calibration {label}',
        )
        if fixed_evaluations is None:
            plot_line(
                axes[4],
                steps,
                [number(row, 'Critic/Spearman') for row in diagnostics],
                f'Spearman {label}',
                linestyle='--',
            )
        else:
            plot_line(
                axes[4],
                milestones,
                [number(row, 'spearman') for row in evaluations],
                f'tie-correct eval Spearman {label}',
                linestyle='--',
                marker='o',
            )
        plot_line(
            axes[5],
            steps,
            [number(row, 'Geometry/GateActivation') for row in diagnostics],
            f'active gate {label}',
        )
        plot_line(
            axes[5],
            steps,
            [number(row, 'Geometry/CandidateGateActivation') for row in diagnostics],
            f'candidate gate {label}',
            linestyle='--',
        )
        plot_line(
            axes[5],
            steps,
            [number(row, 'Geometry/ValidGradientRate') for row in diagnostics],
            f'valid gradient {label}',
            linestyle=':',
        )
        plot_line(
            axes[6],
            steps,
            [number(row, 'Geometry/VN') for row in diagnostics],
            f'V_N {label}',
        )
        plot_line(
            axes[6],
            steps,
            [number(row, 'Geometry/VT') for row in diagnostics],
            f'V_T {label}',
        )
        plot_line(
            axes[6],
            steps,
            [number(row, 'Geometry/VNVT') for row in diagnostics],
            f'V_N/V_T {label}',
            linestyle=':',
        )
        plot_line(
            axes[7],
            steps,
            [number(row, 'Geometry/DangerousNormalVelocity') for row in diagnostics],
            f'danger velocity {label}',
        )
        plot_line(
            axes[7],
            steps,
            [number(row, 'Geometry/NormalizedJVP') for row in diagnostics],
            f'normalized JVP {label}',
            linestyle='--',
        )
        plot_line(
            axes[8],
            steps,
            [number(row, 'Dual/Energy') for row in diagnostics],
            f'energy {label}',
        )
        plot_line(
            axes[8],
            steps,
            [number(row, 'Dual/GlobalSafety') for row in diagnostics],
            f'global safety {label}',
            linestyle='--',
        )

    titles = (
        'Return',
        'Episode cost and cost CVaR',
        'Violation sample ratio',
        'Critic TD error',
        'Risk calibration and rank correlation',
        'Boundary gate and valid gradients',
        'Directional variances',
        'Normal motion and normalized JVP',
        'Dual variables',
    )
    for axes_item, title in zip(axes, titles):
        axes_item.set_title(title)
        axes_item.set_xlabel('environment steps')
        axes_item.grid(alpha=0.25)
        axes_item.legend(fontsize=7)
    if any(any(value > 0 for value in line.get_ydata()) for line in axes[3].lines):
        axes[3].set_yscale('log')
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_trajectories(  # pylint: disable=too-many-locals
    run_dirs: list[Path],
    output: Path,
) -> int:
    """Overlay every saved fixed-seed PointGoal trajectory."""
    figure, axes = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
    count = 0
    hazards_drawn = False
    goals_labeled = False
    for run_index, run_dir in enumerate(run_dirs):
        paths = sorted(
            (run_dir / 'artifacts').glob('trajectory_step_*.csv'),
            key=lambda path: int(path.stem.rsplit('_', maxsplit=1)[-1]),
        )
        for path in paths:
            rows = read_csv(path)
            if not rows:
                continue
            milestone = int(rows[0]['milestone'])
            x_values = [float(row['x']) for row in rows]
            y_values = [float(row['y']) for row in rows]
            label = f'run {run_index + 1}, {milestone // 1000}k'
            axes.plot(x_values, y_values, linewidth=1.2, alpha=0.85, label=label)
            axes.scatter([x_values[0]], [y_values[0]], s=14, marker='o')
            axes.scatter([x_values[-1]], [y_values[-1]], s=22, marker='x')
            if not hazards_drawn:
                hazards_json = next(
                    (row.get('hazards_json', '') for row in rows if row.get('hazards_json')),
                    '',
                )
                hazards = json.loads(hazards_json) if hazards_json else []
                hazard_radius = number(rows[0], 'hazard_radius')
                for hazard_index, center in enumerate(hazards):
                    axes.add_patch(
                        Circle(
                            center,
                            hazard_radius if math.isfinite(hazard_radius) else 0.2,
                            color='tab:red',
                            alpha=0.22,
                            label='hazard' if hazard_index == 0 else None,
                        ),
                    )
                hazards_drawn = bool(hazards)
            goals = {
                (number(row, 'goal_x'), number(row, 'goal_y'))
                for row in rows
                if math.isfinite(number(row, 'goal_x')) and math.isfinite(number(row, 'goal_y'))
            }
            if goals:
                goal_x, goal_y = zip(*sorted(goals))
                axes.scatter(
                    goal_x,
                    goal_y,
                    s=70,
                    marker='*',
                    color='tab:green',
                    label=None if goals_labeled else 'goal locations',
                    zorder=4,
                )
                goals_labeled = True
            count += 1
    if count:
        axes.set(
            xlabel='simulator x',
            ylabel='simulator y',
            title='Fixed-seed PointGoal trajectories',
        )
        axes.axis('equal')
        axes.legend(fontsize=7)
        figure.savefig(output, dpi=180)
    plt.close(figure)
    return count


def collect_scalars(run_dirs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect public scalar and counterfactual rows from all runs."""
    diagnostics: list[dict[str, Any]] = []
    counterfactuals: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        label = run_label(run_dir)
        for row in read_csv(run_dir / 'artifacts/diagnostics.csv'):
            diagnostics.append({'run': label, **row})
        for path in sorted((run_dir / 'artifacts').glob('counterfactual_step_*.json')):
            payload = json.loads(path.read_text(encoding='utf-8'))
            if payload.get('available'):
                counterfactuals.append({'run': label, **payload})
    return diagnostics, counterfactuals


def parser() -> argparse.ArgumentParser:
    """Build diagnostics plotting arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--run-dirs', nargs='+', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument('--checkpoint-evaluation-csv', type=Path)
    return result


def main() -> None:
    """Render plots and machine-readable compact tables."""
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_evaluations = (
        checkpoint_evaluation_rows(args.checkpoint_evaluation_csv)
        if args.checkpoint_evaluation_csv is not None
        else None
    )
    plot_training(
        args.run_dirs,
        args.output_dir / 'star_training_diagnostics.png',
        fixed_evaluations,
    )
    trajectory_count = plot_trajectories(
        args.run_dirs,
        args.output_dir / 'pointgoal_trajectories.png',
    )
    diagnostics, counterfactuals = collect_scalars(args.run_dirs)
    if diagnostics:
        atomic_csv(
            args.output_dir / 'star_validation_metrics.csv',
            list(diagnostics[0]),
            diagnostics,
        )
    if counterfactuals:
        fields = sorted({key for row in counterfactuals for key in row})
        atomic_csv(
            args.output_dir / 'star_mechanism_counterfactuals.csv',
            fields,
            counterfactuals,
        )
    print(
        json.dumps(
            {
                'status': 'passed',
                'runs': len(args.run_dirs),
                'diagnostics_rows': len(diagnostics),
                'counterfactual_rows': len(counterfactuals),
                'trajectory_files': trajectory_count,
                'fixed_checkpoint_evaluations': fixed_evaluations is not None,
                'output_dir': str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        ),
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Plot P4.7 common learning curves, high-risk geometry, and final Pareto points."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use('Agg')
plt = importlib.import_module('matplotlib.pyplot')


ARMS = ('C0', 'C1', 'P47')
COLORS = {'C0': '#1f77b4', 'C1': '#d62728', 'P47': '#2ca02c'}
LABELS = {'C0': 'C0 baseline', 'C1': 'C1 incumbent', 'P47': 'P4.7 candidate'}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def _group(rows: list[dict[str, str]]) -> dict[str, dict[int, list[dict[str, str]]]]:
    grouped: dict[str, dict[int, list[dict[str, str]]]] = {arm: {} for arm in ARMS}
    for row in rows:
        grouped[row['arm']].setdefault(int(row['milestone']), []).append(row)
    return grouped


def _mean(rows: list[dict[str, str]], metric: str) -> float:
    return statistics.fmean(float(row[metric]) for row in rows)


def _curve(
    axis: Any,
    grouped: dict[str, dict[int, list[dict[str, str]]]],
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    for arm in ARMS:
        steps = sorted(grouped[arm])
        if not steps:
            continue
        values = [_mean(grouped[arm][step], metric) for step in steps]
        axis.plot(steps, values, marker='o', color=COLORS[arm], label=LABELS[arm])
        for step in steps:
            seed_values = [float(row[metric]) for row in grouped[arm][step]]
            axis.scatter(
                [step] * len(seed_values),
                seed_values,
                s=14,
                alpha=0.25,
                color=COLORS[arm],
            )
    axis.set_title(title)
    axis.set_xlabel('Environment steps')
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)


def plot_curves(summary_dir: Path, output_dir: Path) -> list[Path]:
    """Render common-protocol checkpoint curves."""
    grouped = _group(_rows(summary_dir / 'p4_7_checkpoint_metrics.csv'))
    outputs: list[Path] = []

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _curve(axes[0], grouped, 'mean_return', 'Evaluation return', 'Return')
    _curve(axes[1], grouped, 'success_rate', 'Goal success', 'Success rate')
    axes[1].legend(loc='best')
    figure.tight_layout()
    path = output_dir / 'p4_7_return_success.png'
    figure.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(figure)
    outputs.append(path)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    _curve(axes[0], grouped, 'mean_cost', 'Evaluation mean cost', 'Cost')
    _curve(axes[1], grouped, 'cost_cvar_20', 'Evaluation CVaR-20', 'Upper-tail cost')
    _curve(
        axes[2],
        grouped,
        'training_cumulative_cost',
        'Training cumulative cost',
        'Cumulative cost',
    )
    axes[2].legend(loc='best')
    figure.tight_layout()
    path = output_dir / 'p4_7_cost_cvar_training_cost.png'
    figure.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(figure)
    outputs.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _curve(
        axes[0],
        grouped,
        'high_risk_normal_to_tangent_variance_ratio',
        'Highest-margin 20%: $V_N/V_T$',
        '$V_N/V_T$',
    )
    _curve(
        axes[1],
        grouped,
        'high_risk_dangerous_outward_velocity',
        'Highest-margin 20%: dangerous outward velocity',
        'Positive normal velocity',
    )
    axes[1].legend(loc='best')
    figure.tight_layout()
    path = output_dir / 'p4_7_high_risk_geometry.png'
    figure.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(figure)
    outputs.append(path)
    return outputs


def plot_pareto(summary_dir: Path, output_dir: Path) -> Path:
    """Render pooled and seed-level formal Pareto points."""
    summary = json.loads((summary_dir / 'p4_7_final_summary.json').read_text(encoding='utf-8'))
    seed_rows = _rows(summary_dir / 'p4_7_formal_final_seed_metrics.csv')
    comparisons = (
        ('mean_cost', 'mean_return', 'Mean cost', 'Return'),
        ('cost_cvar_20', 'mean_return', 'CVaR-20', 'Return'),
        ('mean_cost', 'success_rate', 'Mean cost', 'Success rate'),
        ('cost_cvar_20', 'success_rate', 'CVaR-20', 'Success rate'),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, (x_metric, y_metric, xlabel, ylabel) in zip(axes.flat, comparisons):
        for arm in ARMS:
            selected = [row for row in seed_rows if row['arm'] == arm]
            axis.scatter(
                [float(row[x_metric]) for row in selected],
                [float(row[y_metric]) for row in selected],
                s=28,
                alpha=0.3,
                color=COLORS[arm],
            )
            pooled = summary['pooled_formal'][arm]
            axis.scatter(
                pooled[x_metric],
                pooled[y_metric],
                s=110,
                marker='X',
                color=COLORS[arm],
                label=LABELS[arm],
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 1].legend(loc='best')
    figure.suptitle(
        'P4.7 formal Pareto: faint = training seed, X = pooled 300 episodes; lower cost is safer',
    )
    figure.tight_layout()
    path = output_dir / 'p4_7_final_pareto.png'
    figure.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(figure)
    return path


def parser() -> argparse.ArgumentParser:
    """Build plot arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--summary-dir', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    return result


def main() -> None:
    """Generate every P4.7 figure."""
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = plot_curves(args.summary_dir, args.output_dir)
    outputs.append(plot_pareto(args.summary_dir, args.output_dir))
    print(json.dumps({'plots': [str(path) for path in outputs]}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

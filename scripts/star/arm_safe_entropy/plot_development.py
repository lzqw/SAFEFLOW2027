"""Compact native learning curves; versions/seeds are never silently averaged."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-version', default='SafeReach-Obstacle-v2')
    args = parser.parse_args()
    report = Path(__file__).resolve().parents[3] / 'reports/star/arm_safe_entropy'
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    titles = ['Held-out safe success (30 episodes/checkpoint)', 'Held-out violation rate',
              'Training recent success (last 100 completed episodes)', 'Cumulative training violations']
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=.2)
    included = []
    for config_path in sorted((report / 'obstacle_development').glob('*/run_config.json')):
        config = json.loads(config_path.read_text())
        run = config_path.parent
        if config['task_config']['version'] != args.task_version or (run / 'engineering_stop.json').exists():
            continue
        with (run / 'progress.csv').open(newline='') as stream:
            progress = []
            for row in csv.DictReader(stream):
                try:
                    progress.append({k: float(row[k]) for k in
                        ['steps', 'recent_success', 'training_violation_count']})
                except (ValueError, TypeError):
                    continue
        if not progress:
            continue
        label = f"{config['method']} seed{config['training_seed']}"
        if (run / 'resume_lineage.json').exists():
            label += ' (resumed; lost-tail excluded)'
        elif (run / 'engineering_interruption.json').exists():
            label += ' (interrupted)'
        if config['method'] == 'flow_oracle':
            label += f" ({config.get('oracle_backend', 'exact')})"
        if config.get('revision_id'):
            label += f" [{config['revision_id']}, rho={config.get('normal_variance_ratio', .25):g}]"
        color = f'C{len(included)}'
        x = [row['steps'] / 1000 for row in progress]
        axes[1, 0].plot(x, [r['recent_success'] * 100 for r in progress], color=color, label=label)
        axes[1, 1].plot(x, [r['training_violation_count'] for r in progress], color=color)
        evaluations = sorted([json.loads(p.read_text()) for p in run.glob('evaluation_*.json')],
                             key=lambda e: e['milestone_steps'])
        for axis, key in zip(axes[0], ['safe_success', 'violation']):
            if evaluations:
                axis.plot([e['milestone_steps'] / 1000 for e in evaluations],
                          [e[key] * 100 for e in evaluations], 'o-', color=color, label=label)
        included.append(dict(run=str(run.relative_to(report)), source_commit=config['source_commit'],
            latest_logged_transitions=int(progress[-1]['steps']),
            evaluation_milestones=[e['milestone_steps'] for e in evaluations]))
    for axis in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        axis.set_ylim(-3, 103)
        axis.set_ylabel('%')
    for axis in axes[1]:
        axis.set_xlabel('Retained training transitions (thousands)')
    if included:
        handles, labels = axes[1, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=7)
    fig.suptitle(f'{args.task_version}: interim development, per seed; no smoothing or extrapolation', fontsize=11)
    fig.tight_layout(rect=(0, .18 if included else 0, 1, .96))
    destination = report / 'figures'
    destination.mkdir(exist_ok=True)
    fig.savefig(destination / f'{args.task_version}-learning.png', dpi=150)
    plt.close(fig)
    (destination / f'{args.task_version}-learning.json').write_text(json.dumps(dict(
        task_version=args.task_version, runs=included,
        note='Top panels: stochastic held-out evaluation only. Bottom panels: training statistics, not held-out performance. Missing checkpoints are not assigned zeros.'), indent=2) + '\n')
    print(json.dumps(dict(task_version=args.task_version, plotted_runs=included)))


if __name__ == '__main__':
    main()

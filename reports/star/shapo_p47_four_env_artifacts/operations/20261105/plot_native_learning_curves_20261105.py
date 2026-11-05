#!/usr/bin/env python3
"""Plot the three native evaluation milestones; no fitted/synthetic points."""
import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collect_native_learning_curves_20261105 import OUT as INPUT, ENVS


def main():
    out = INPUT / 'native_evaluation_milestones.svg'
    assert not out.exists(), 'Preserve existing figure'
    with (INPUT / 'per_training_seed_milestones.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), layout='constrained')
    colors = ['#4477aa', '#ee6677', '#228833', '#ccbb44', '#aa3377']
    for index, (slug, (environment, milestones)) in enumerate(ENVS.items()):
        for col, (metric, label, scale) in enumerate([
            ('mean_return', 'Return', 1),
            ('mean_cost', 'Evaluation cost', 1),
            ('task_metric', 'Success (%)' if slug.startswith('point-') else 'Failure (%)', 100),
        ]):
            ax = axes[index, col]
            xs = [step / 1e6 for step in milestones]
            means = []
            for seed, color in zip(range(2027, 2032), colors):
                selected = sorted([r for r in rows if r['job'] == f'{slug}-{seed}'], key=lambda r: int(r['milestone_steps']))
                ax.plot(xs, [float(r[metric]) * scale for r in selected], marker='o',
                        linewidth=1, alpha=0.55, color=color, label=str(seed))
            for step in milestones:
                means.append(statistics.mean(float(r[metric]) * scale for r in rows
                                             if r['environment'] == environment and int(r['milestone_steps']) == step))
            ax.plot(xs, means, color='#111111', marker='s', linewidth=2.3, label='Five-seed mean')
            ax.set(title=f'{slug}: {label}', xlabel='Training steps (millions)')
            ax.set_xticks(xs)
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=6, frameon=False)
    fig.suptitle('P4.7 native evaluations: all 20 training seeds, three fixed milestones\n'
                 'Lines only connect observed points; no new rollouts, fitted curves or checkpoint selection', fontsize=12)
    fig.savefig(out, format='svg', metadata={'Date': None})
    plt.close(fig)
    # Matplotlib leaves spaces on multiline path commands; normalize only
    # this generated figure, never the byte-preserved native evidence.
    out.write_text('\n'.join(line.rstrip() for line in out.read_text().splitlines()) + '\n')
    print(out)


if __name__ == '__main__':
    main()

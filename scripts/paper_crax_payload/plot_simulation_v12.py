"""Plot every completed early-screen learning node; no new environment calls."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ARMS = ['iso-2033', 'metric-only-2033', 'P5-fixed-2033',
        'E1-critic20k-2033', 'E2-quarter50k-2033', 'E2-quarter75k-2033']
LABELS = ['Iso', 'Metric-only', 'P5 fixed', 'Critic first20K',
          'Direction .25 to1 /50K', 'Direction .25 to1 /75K']
COLORS = ['#555555', '#0072B2', '#009E73', '#D55E00', '#CC79A7', '#E69F00']


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.analysis/'receipt.json').read_text())
    if receipt['complete_runs'] != 6 or receipt['progress_rows'] != 600:
        raise ValueError('Plot all six registered complete100K curves, never a partial subset')
    train = read(args.analysis/'training.csv'); dev = read(args.analysis/'dev_nodes.csv')
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 140})
    panels = [('training_violations', 'Cumulative training violations'),
              ('safe_successes', 'Cumulative safe successes'),
              ('safe_timeouts', 'Cumulative training timeouts'),
              ('training_stall_episodes', 'Completed episodes with a stall')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, panels):
        for name, label, color in zip(ARMS, LABELS, COLORS):
            rows = sorted([r for r in train if r['run'] == name], key=lambda r: int(r['steps']))
            ax.plot([int(r['steps'])/1000 for r in rows], [float(r[key]) for r in rows],
                    label=label, color=color, linestyle='--' if name.startswith(('iso-', 'metric-')) else '-')
        ax.set_title(title); ax.set_xlabel('Actual training transitions (thousands)'); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('SafeReach fresh100K development: one training seed (2033)')
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output/f'early_training_tradeoffs.{suffix}')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharex=True, sharey=True, constrained_layout=True)
    for ax, condition in zip(axes, ('nominal', 'radius065', 'execution085')):
        for name, label, color in zip(ARMS, LABELS, COLORS):
            rows = sorted([r for r in dev if r['run'] == name and r['condition'] == condition],
                          key=lambda r: int(r['new_steps']))
            ax.plot([int(r['new_steps'])/1000 for r in rows],
                    [100*int(r['safe_successes'])/int(r['episodes']) for r in rows],
                    marker='o', markersize=3, label=label, color=color,
                    linestyle='--' if name.startswith(('iso-', 'metric-')) else '-')
        ax.set_title(condition); ax.set_xlabel('Training transitions (thousands)')
        ax.set_ylim(0, 103); ax.set_xticks([25, 50, 75, 100]); ax.grid(alpha=.2)
    axes[0].set_ylabel('Safe success (%)'); axes[0].legend(fontsize=7)
    fig.suptitle('All fixed DEV nodes;30 paired layouts per condition;no independent test')
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output/f'early_development_curves.{suffix}')
    plt.close(fig)
    (args.output/'provenance.json').write_text(json.dumps(dict(
        input=str(args.analysis), complete_runs=6, progress_rows=600, dev_nodes=72,
        training_seeds=[2033], episodes_per_dev_condition=30,
        all_registered_nodes_included=True, uncertainty='single training seed;descriptive curves only',
        new_training_steps=0, new_simulation_controls=0, independent_test_used=False), indent=2)+'\n')


if __name__ == '__main__':
    main()

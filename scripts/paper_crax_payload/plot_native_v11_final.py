"""Plot the frozen v11 independent outcomes from compact CSV; no simulation."""
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(args.summary.open()))
    groups = [
        ('piper_reach', 'Piper Reach', ['nominal', 'radius_0575', 'radius_060'],
         ['Nominal', '57.5 mm unseen', '60 mm unseen']),
        ('piper_payload', 'Piper Payload (45/50/55 mm training coverage)',
         ['nominal', 'radius_055', 'radius_0575', 'execution_scale_085'],
         ['Nominal', '55 mm trained', '57.5 mm unseen', 'Execution 0.85']),
        ('panda_payload', 'Panda Payload', ['nominal', 'radius_070', 'payload_wide', 'execution_scale_085'],
         ['Nominal', '70 mm', 'Wide payload', 'Execution 0.85'])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (line, title, conditions, labels) in zip(axes, groups):
        for arm, offset, color in [('control', -.19, '#757575'), ('p5', .19, '#1565c0')]:
            values = [next(r for r in rows if r['line'] == line and r['method'] == arm
                           and r['condition'] == c) for c in conditions]
            heights = [100*float(r['safe_success_per_attempted']) for r in values]
            bars = ax.bar([i+offset for i in range(len(values))], heights, width=.36, color=color, label=arm)
            ax.bar_label(bars, fmt='%.0f', padding=3, fontsize=9)
        ax.set_xticks(range(len(conditions)), labels, rotation=30, ha='right')
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=.2)
        ax.set_ylim(0, 111)
        ax.legend()
    axes[0].set_ylabel('Safe success (%) — 100 episodes per condition')
    fig.suptitle('Frozen independent evaluation; all failures retained; one training seed per task')
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()

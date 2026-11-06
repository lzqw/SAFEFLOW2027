"""Small explicit policy comparisons from per-state diagnostics, never pooled actions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', help='Explicit display subset; full native JSON remains unchanged')
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    summaries, records = payload['summaries'], payload['records']
    if args.methods:
        summaries = [s for s in summaries if s['method'] in args.methods]
        records = [r for r in records if r['method'] in args.methods]
    if not 2 <= len(records) <= 4 or len(summaries) != len(records):
        parser.error('Use two to four explicit method/seed policies for this focused figure')
    output = args.input.parent
    suffix = '-' + '-'.join(args.methods) if args.methods else ''
    milestone = payload.get('requested_checkpoint_steps')
    caption = f'Fixed {milestone:,}-transition policies' if milestone else 'Interim native snapshots (ages in JSON)'
    names = [f"{s['method']} seed{s['training_seed']}" for s in summaries]
    names = [name + (f" rho={s['normal_variance_ratio']:g}" if (s.get('normal_variance_ratio') or .25) != .25 else '')
             for name, s in zip(names, summaries)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for row, region in enumerate(['safe_interior', 'near_boundary']):
        for col, prefix in enumerate(['reference_', '']):
            ax = axes[row, col]
            maximum = 0.
            for index, summary in enumerate(summaries):
                data = summary[region]
                values = [data[prefix+'vn'], data[prefix+'vt']]
                if any(v is None for v in values):
                    continue  # Undefined directions are never drawn as zero variance.
                maximum = max(maximum, *values)
                width = .72 / len(summaries)
                bars = ax.bar(np.array([0.,1.]) + (index-(len(summaries)-1)/2)*width,
                              values, width=width*.92,
                              color=f'C{index}', label=names[index])
                ax.bar_label(bars, fmt='%.4f', fontsize=6 if len(summaries)>2 else 8)
            ax.set_xticks([0,1], ['Normal', 'Tangent (per dimension)'])
            ax.set_ylim(0,maximum*1.22 if maximum else 1.)
            ax.grid(axis='y', alpha=.2)
            count = summaries[0][region]['valid_normal_states']
            ax.set_title(f"{region.replace('_',' ')}: {'reference before drift' if col==0 else 'final actions'} (n={count})", fontsize=10)
            ax.set_ylabel('Variance')
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8)
    fig.suptitle(caption + '\nSame exact axis; per-state means in normalized action units, not pooled actions', fontsize=11)
    fig.tight_layout(rect=(0, .11, 1, .96))
    fig.savefig(output/f'conditional-covariance{suffix}.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4))
    bins = np.linspace(-1,1,21)
    comparisons = 0
    for record in records:
        for index, region in enumerate(['safe_interior', 'near_boundary']):
            values = [row['geometry_normal_cosine_at_reference_action'] for row in record['states']
                      if row['region']==region and row['geometry_normal_cosine_at_reference_action'] is not None]
            if not values:
                continue
            comparisons += len(values)
            ax.hist(np.clip(values,-1,1), bins=bins, histtype='step', linewidth=2,
                    label=f"{record['method']} seed{record['training_seed']} {region.replace('_',' ')}: n={len(values)}, mean={np.mean(values):.3f}")
    ax.set_xlim(-1,1)
    ax.set_xlabel('Signed cosine to exact simulator gradient at identical state/action')
    ax.set_ylabel('State records')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    ax.set_title(caption + '\nUndefined normals retained in JSON, not assigned artificial directions', fontsize=10)
    fig.tight_layout()
    fig.savefig(output/f'normal-agreement{suffix}.png', dpi=150)
    plt.close(fig)
    print(json.dumps(dict(source=str(args.input), training_transitions=0, normal_comparisons=comparisons,
                          plots=[f'conditional-covariance{suffix}.png',f'normal-agreement{suffix}.png'],
                          note='Interim diagnosis, not final control matching or episode-performance evidence')))


if __name__ == '__main__':
    main()

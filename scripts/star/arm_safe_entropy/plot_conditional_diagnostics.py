"""Focused pair plots from native per-state diagnostics, without pooling actions."""
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
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    summaries, records = payload['summaries'], payload['records']
    if len(records) != 2 or len(summaries) != 2:
        parser.error('This focused figure shows one explicit two-policy comparison')
    output = args.input.parent
    milestone = payload.get('requested_checkpoint_steps')
    caption = f'Fixed {milestone:,}-transition policies' if milestone else 'Interim native snapshots (ages in JSON)'
    names = [f"{s['method']} seed{s['training_seed']}" for s in summaries]
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
                bars = ax.bar(np.array([0.,1.]) + (index-.5)*.34, values, width=.32,
                              color=f'C{index}', label=names[index])
                ax.bar_label(bars, fmt='%.4f', fontsize=8)
            ax.set_xticks([0,1], ['Normal', 'Tangent (per dimension)'])
            ax.set_ylim(0,maximum*1.22 if maximum else 1.)
            ax.grid(axis='y', alpha=.2)
            count = summaries[0][region]['valid_normal_states']
            ax.set_title(f"{region.replace('_',' ')}: {'reference before drift' if col==0 else 'final actions'} (n={count})", fontsize=10)
            ax.set_ylabel('Variance')
    axes[0,0].legend(fontsize=8,loc='lower left')
    fig.suptitle(caption + '\nSame exact axis; per-state means in normalized action units, not pooled actions', fontsize=11)
    fig.tight_layout()
    fig.savefig(output/'conditional-covariance.png', dpi=150)
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
                    label=f"{record['method']} {region.replace('_',' ')}: n={len(values)}, mean={np.mean(values):.3f}")
    ax.set_xlim(-1,1)
    ax.set_xlabel('Signed cosine to exact simulator gradient at identical state/action')
    ax.set_ylabel('State records')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    ax.set_title(caption + '\nUndefined normals retained in JSON, not assigned artificial directions', fontsize=10)
    fig.tight_layout()
    fig.savefig(output/'normal-agreement.png', dpi=150)
    plt.close(fig)
    print(json.dumps(dict(source=str(args.input), training_transitions=0, normal_comparisons=comparisons,
                          plots=['conditional-covariance.png','normal-agreement.png'],
                          note='Interim diagnosis, not final control matching or episode-performance evidence')))


if __name__ == '__main__':
    main()

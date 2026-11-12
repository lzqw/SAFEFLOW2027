"""Plot completed frozen M1 risk vs tangent variance, without selecting seeds."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Refuse to overwrite a previously published figure')
    with (args.analysis/'MECHANISM_RESULTS.csv').open() as stream:rows=list(csv.DictReader(stream))
    rows=[r for r in rows if r['experiment']=='M1']
    policies=sorted({(r['variant'],int(r['training_seed'])) for r in rows})
    if len(policies)!=4:raise ValueError('Retain all four declared policy/seed combinations')
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    colors=dict(iso_high='#777777',iso_low='#cc7a00',normal_shrink='#0072b2',directional_full='#009e73')
    labels=dict(iso_high='Iso-High',iso_low='Iso-Low',normal_shrink='Normal-Shrink',directional_full='Directional-Full')
    for ax,(variant,seed) in zip(axes.flat,policies):
        for mode in colors:
            for condition,marker in [('nominal','o'),('radius_080','^')]:
                selected=[r for r in rows if r['variant']==variant and int(r['training_seed'])==seed
                          and r['sampler']==mode and r['condition']==condition]
                if len(selected)!=1:raise ValueError('Missing or duplicate M1 condition; do not draw missing values as zero')
                row=selected[0]
                if row['endpoint_vt']=='':continue
                ax.scatter(float(row['endpoint_vt']),100*float(row['true_interval_risk']),
                    color=colors[mode],marker=marker,s=70,facecolors=colors[mode] if condition=='radius_080' else 'none',
                    label=labels[mode] if condition=='radius_080' else None)
        ax.set(title=f'{variant}, training seed {seed}',xlabel='Endpoint tangent variance per dimension',
               ylabel='True one-interval risk (%)')
        ax.grid(alpha=.2)
    handles,legend_labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='outside lower center',ncol=4)
    fig.suptitle('Frozen common-state sampling interventions, not newly trained methods\n'
                 'Circle: nominal; triangle: radius8cm. Injection trace/normal matching differs by control.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=150)
    plt.close(fig)


if __name__=='__main__':main()

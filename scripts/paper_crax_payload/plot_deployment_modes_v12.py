"""Plot all frozen Piper models in both modes, retaining fair same-mode baselines."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(); receipt_path = args.analysis/'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['status']!='complete_deployment_mode_DEV_review' or receipt['episodes']!=4200:
        raise ValueError('Require the entire registered Piper DEV review')
    with (args.analysis/'summary.csv').open(newline='') as stream: rows = list(csv.DictReader(stream))
    if len(rows)!=42: raise ValueError('Never plot a partial selection of modes/models')
    args.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'figure.dpi':150,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(2,2,figsize=(10.4,7.1),constrained_layout=True)
    settings = [('piper_reach','radius_0575','Reach57.5mm',93),
                ('piper_reach','radius_060','Reach60mm',90),
                ('piper_payload','radius_055','Payload55mm',None),
                ('piper_payload','radius_0575','Payload57.5mm',None)]
    for ax,(line,condition,title,threshold) in zip(axes.flat,settings):
        labels = ('Fixed P5','P5 +2mm' if line=='piper_reach' else 'P5 +4mm','Metric')
        for mode,shift,color,label in (('sampled',-.18,'#0072B2','Sampled deployment'),
                                        ('zero_noise',.18,'#D55E00','Zero-noise deployment')):
            values = []
            for key in ('fixedP5','activation','Metric'):
                row = next(r for r in rows if r['line']==line and r['condition']==condition and r['label']==key and r['mode']==mode)
                values.append(100*int(row['safe_successes'])/int(row['episodes']))
            bars = ax.bar([i+shift for i in range(3)],values,width=.34,color=color,label=label)
            ax.bar_label(bars,fmt='%.0f%%',padding=3,fontsize=9)
        if threshold is not None: ax.axhline(threshold,linestyle=':',color='#777777',linewidth=1)
        ax.set_xticks(range(3),labels); ax.set_ylim(0,105); ax.set_title(title)
        ax.set_ylabel('DEV safe success (%)'); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Frozen Piper deployment: all three models, both modes, same100 layouts/condition',fontsize=11)
    fig.supxlabel('Single training parent. New DEV, not independent test. Training violations unchanged; all4200 DEV episodes had zero timeout/stall.',fontsize=8)
    for suffix in ('png','pdf'): fig.savefig(args.output/f'piper_deployment_modes.{suffix}')
    plt.close(fig)
    (args.output/'provenance.json').write_text(json.dumps(dict(status='complete_existing_data_plot',
        input_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),source_episodes=4200,
        plotted_conditions=4,plotted_bars=24,all_models_and_modes_retained=True,
        nominal_and_execution_results='Retained in source42-node table;not omitted from acceptance',
        training_seed_count=1,independent_test_used=False,new_training_steps=0,new_simulation_controls=0,
        uncertainty='Descriptive DEV only;no training-seed error bars or universal dominance claim'),indent=2)+'\n')


if __name__=='__main__': main()

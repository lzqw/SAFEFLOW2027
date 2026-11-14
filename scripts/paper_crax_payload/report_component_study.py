"""Tables and curves from completed native component runs; no extra rollouts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def report(directories, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve prior report')
    output.mkdir(parents=True)
    progress={};rows=[];receipts=[]
    for directory in directories:
        directory=Path(directory)
        manifest=json.loads((directory/'run_config.json').read_text())
        receipt=json.loads((directory/'receipt.json').read_text())
        if receipt['status']!='completed' or receipt.get('engineering_only') or receipt['steps']!=300000:
            raise ValueError('Final report requires complete native300K, not smoke or running prefix')
        name=manifest['arm']['name']
        if name in progress:raise ValueError('Duplicate arm')
        progress[name]=[json.loads(line) for line in (directory/'progress.jsonl').read_text().splitlines() if line.strip()]
        receipts.append(dict(receipt,arm=name))
        for step in manifest['configuration']['evaluation']['milestones']:
            data=json.loads((directory/f'evaluation_{step}.json').read_text())
            expected='independent_fixed_endpoint' if step==300000 else 'development'
            if data['split']!=expected:raise ValueError('Development and independent labels mixed')
            for condition in data['conditions']:
                records=condition['records'];n=len(records)
                rows.append(dict(arm=name,component=manifest['arm']['component'],rho=manifest['arm']['rho'],
                    training_seed=receipt['training_seed'],steps=step,split=data['split'],
                    condition=condition['condition'],episodes=n,
                    safe_success=condition['safe_success'],violation=condition['violation'],
                    safe_timeout=condition['safe_timeout'],capped_seconds=condition['capped_seconds'],
                    mean_return=float(np.mean([r['episode_return'] for r in records])),
                    mean_minimum_distance_m=float(np.mean([r['minimum_distance_m'] for r in records])),
                    training_cumulative_violations=next(r['training_violations'] for r in progress[name] if r['steps']==step),
                    full_training_seconds=receipt['elapsed_seconds'],
                    full_training_including_eval_save_sps=receipt['steps']/receipt['elapsed_seconds'],
                    actual_total_training_transitions=receipt['steps'],source_commit=receipt['source_commit']))
    names=['iso','noise-only','metric-only','full','full-rho010','full-rho050']
    if set(progress)!=set(names):raise ValueError('Keep all six registered arms, not only favorable results')
    with (output/'all_checkpoints.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    final=[r for r in rows if r['split']=='independent_fixed_endpoint']
    with (output/'fixed_300k_independent.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(final)
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for name in names:
        series=progress[name];x=np.array([r['steps'] for r in series])/1000
        for ax,key in zip(axes,['safe_successes','training_violations','complete_train_sps']):
            ax.plot(x,[r[key] for r in series],label=name,lw=1.2)
    for ax,title in zip(axes,['Training cumulative safe successes','Training cumulative violations','End-to-end cumulative transitions/s']):
        ax.set(xlabel='Actual training transitions (K)',title=title);ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7);fig.tight_layout()
    fig.savefig(output/'all_training_curves.png',dpi=170);fig.savefig(output/'all_training_curves.pdf');plt.close(fig)
    conditions=[r['condition'] for r in final if r['arm']=='iso']
    fig,axes=plt.subplots(1,len(conditions),figsize=(4*len(conditions),3.8),squeeze=False)
    for ax,condition in zip(axes[0],conditions):
        for name in names:
            r=next(r for r in final if r['arm']==name and r['condition']==condition)
            ax.scatter(r['violation'],r['safe_success'],s=35)
            ax.annotate(name,(r['violation'],r['safe_success']),fontsize=7,xytext=(3,3),textcoords='offset points')
        ax.set(title=condition,xlabel='Violation fraction (lower better)',ylabel='Safe success fraction (higher better)',
            xlim=(-.03,1.03),ylim=(-.03,1.03));ax.grid(alpha=.2)
    fig.suptitle('Fixed300K independent episodes; one development-selected training seed2027')
    fig.tight_layout();fig.savefig(output/'fixed_endpoint_task_risk.png',dpi=170)
    fig.savefig(output/'fixed_endpoint_task_risk.pdf');plt.close(fig)
    value=dict(status='completed_from_existing_native_results',training_transitions=sum(r['steps'] for r in receipts),
        new_training_transitions=0,new_simulation_controls=0,physical_robot_interactions=0,
        training_seed_count=1,development_selected_training_seed=2027,best_checkpoint_reselection=False,
        independent_episode_count=sum(r['episodes'] for r in final),
        engineering_transitions_not_in_scientific_total=8,all_six_arms_retained=True,receipts=receipts)
    (output/'receipt.json').write_text(json.dumps(value,indent=2)+'\n')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    print(json.dumps(report(args.runs,args.output)))


if __name__=='__main__':main()

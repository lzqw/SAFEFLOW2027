"""Summarize completed Robot-Gap runs without simulation or checkpoint loading."""
import argparse
import csv
import json
from pathlib import Path
import shutil


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--old-finals',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    curves=[]; efficiency=[]
    for run in sorted(args.runs.iterdir()):
        if not run.name.startswith(('transfer-','reach-radius-')):
            continue
        receipt_path=run/'receipt.json'
        if not receipt_path.exists(): continue
        receipt=json.loads(receipt_path.read_text())
        if receipt['status']!='completed': continue
        for path in sorted(run.glob('evaluation_*.json')):
            evaluation=json.loads(path.read_text())
            for condition in evaluation['conditions']:
                curves.append(dict(run=run.name,task=receipt['task'],method=receipt['method'],
                    training_seed=receipt['training_seed'],checkpoint_steps=evaluation['checkpoint_steps'],
                    split='development',fixed_endpoint=evaluation['fixed_endpoint'],
                    **{k:condition[k] for k in ('condition','episodes','safe_success_count',
                        'violation_count','safe_timeout_count','safe_success','violation',
                        'episode_return','time_to_goal_capped_seconds','minimum_distance_m')}))
        elapsed=receipt['segment_elapsed_seconds']
        efficiency.append(dict(run=run.name,task=receipt['task'],method=receipt['method'],
            initial_steps=receipt['initial_steps'],final_steps=receipt['training_steps'],
            new_transitions=receipt['newly_collected_transitions'],segment_seconds=elapsed,
            complete_training_sps=receipt['newly_collected_transitions']/elapsed,
            learner_updates=receipt.get('new_learner_updates',receipt['learner_updates']),
            actor_updates=receipt.get('new_actor_updates',receipt['actor_updates']),
            new_training_violations=receipt.get('new_training_violations',receipt['training_violation_count']),
            batch_size=receipt['batch_size'],updates_per_vector_step=receipt['updates_per_vector_step'],
            **receipt['timing']))
        receipt_output=args.output/'receipts'/run.name
        receipt_output.mkdir(parents=True,exist_ok=True)
        shutil.copy2(receipt_path,receipt_output/'receipt.json')
    if not curves: raise ValueError('Require completed genuine evaluations')
    curves.sort(key=lambda r:(r['task'],r['method'],r['checkpoint_steps'],r['condition']))
    write_csv(args.output/'piper_new_learning_curve.csv',curves)
    write_csv(args.output/'piper_new_efficiency.csv',efficiency)
    old=list(csv.DictReader(args.old_finals.open()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,2,figsize=(11,7.5))
    conditions=('nominal','new_geometry','execution_scale_085')
    labels=('Nominal','New geometry','Execution scale 0.85')
    colors=('#2274a5','#e08621','#27864e')
    maximum_payload_steps=max(r['checkpoint_steps'] for r in curves if r['task']=='Payload')
    for column,method in enumerate(('Iso','EA')):
        ax=axes[0,column]
        for condition,label,color in zip(conditions,labels,colors):
            selected=[r for r in curves if r['task']=='Payload' and r['method']==method and r['condition']==condition]
            ax.plot([r['checkpoint_steps']/1000 for r in selected],
                [r['safe_success']*100 for r in selected],'-o',label=label,color=color)
        ax.set(title=f'Payload transfer — {method}',xlabel='New Payload transitions (thousands)',
            ylabel='Safe success (%)',ylim=(-3,103),xlim=(0,maximum_payload_steps/1000*1.03))
        ax.legend(fontsize=8); ax.grid(alpha=.2)
        ax=axes[1,column]
        parent=[next(float(r['safe_success'])*100 for r in old
            if r['task']=='Reach' and r['method']==method and r['condition']==c) for c in conditions]
        adapted=[next(r['safe_success']*100 for r in curves
            if r['task']=='Reach' and r['method']==method and r['condition']==c and r['checkpoint_steps']==400000)
            for c in conditions]
        x=list(range(3)); width=.36
        left=ax.bar([v-width/2 for v in x],parent,width,label='Original 300K',color='#a9bac8')
        right=ax.bar([v+width/2 for v in x],adapted,width,label='+100K shared radius adaptation',color='#2274a5')
        ax.bar_label(left,fmt='%.0f',fontsize=8); ax.bar_label(right,fmt='%.0f',fontsize=8)
        ax.set_xticks(x,('Nominal','New geometry','Scale 0.85'))
        ax.set(title=f'Reach — {method}',ylabel='Safe success (%)',ylim=(0,130))
        ax.legend(fontsize=7,loc='upper left'); ax.grid(axis='y',alpha=.2)
    figure.suptitle('Piper Robot-Gap v2 — one training seed; 30 development episodes / condition')
    figure.text(.5,.012,'No independent-test claim. Equal adaptation budget; method-matched Reach parents. Shared GPU load uncontrolled.',
        ha='center',fontsize=9)
    figure.tight_layout(rect=(0,.035,1,.95))
    for suffix in ('png','pdf'):
        figure.savefig(args.output/f'piper_learning_and_adaptation.{suffix}',dpi=170)
    plt.close(figure)
    print(json.dumps(dict(completed_runs=len(efficiency),condition_rows=len(curves),new_simulation=0,new_training=0)))


if __name__=='__main__':
    main()

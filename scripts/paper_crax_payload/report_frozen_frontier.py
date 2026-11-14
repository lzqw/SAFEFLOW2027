"""Render existing frozen evaluations; never run a policy or choose new weights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def write_csv(path,rows):
    if not rows:
        return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-prefix',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous published prefix/results')
    receipt_path=args.input/'receipt.json'
    receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if not args.allow_prefix and (not receipt or receipt.get('status')!='completed'):
        raise ValueError('A final report requires the actual completed worker receipt')
    files=sorted(args.input.glob('E0*.json'))
    if not files:
        raise ValueError('No completed condition groups to summarize')
    args.output.mkdir(parents=True)
    rows,episodes,sources=[],[],{}
    for path in files:
        value=json.loads(path.read_text())
        sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        common={k:value[k] for k in ('stage','split','method','action_scale')}
        for c in value['conditions']:
            rows.append(dict(**common,**{k:c[k] for k in (
                'condition','episodes_attempted','episodes_admitted','static_invalid_layouts',
                'safe_success_count','violation_count','safe_success','violation',
                'time_to_goal_capped_seconds','mean_path_length_m','mean_success_path_length_m',
                'mean_minimum_distance_m','physical_control_intervals','physics_substeps','distance_queries')},
                violation_ci_low=c['violation_wilson95'][0],violation_ci_high=c['violation_wilson95'][1]))
            episodes.extend(dict(**common,condition=c['condition'],**{k:v for k,v in r.items() if k!='trajectory'})
                for r in c['records'])
    if receipt and receipt['status']=='completed' and len(episodes)!=receipt['attempted_episodes']:
        raise ValueError('The report must cover the receipt, not just a convenient subset')
    write_csv(args.output/'condition_summary.csv',rows)
    write_csv(args.output/'episodes.csv',episodes)
    selection_path=args.input/'selection.json'
    if selection_path.exists():
        selection=json.loads(selection_path.read_text())
        selected=[]
        for method,choices in selection['selected'].items():
            for cap,choice in choices.items():
                chosen=[r for r in rows if r['stage']=='E02' and r['split']=='independent_test'
                    and r['method']==method and r['action_scale']==choice['action_scale']]
                for row in chosen:
                    selected.append(dict(validation_risk_cap=float(cap),validation_status=choice['status'],**row))
        write_csv(args.output/'validation_selected_independent.csv',selected)
        (args.output/'selection.json').write_bytes(selection_path.read_bytes())
        sources['selection.json']=hashlib.sha256(selection_path.read_bytes()).hexdigest()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=[]
    for split in ('validation','independent_test'):
        values=[r for r in rows if r['stage']=='E02' and r['split']==split]
        conditions=list(dict.fromkeys(r['condition'] for r in values))
        if not conditions:
            continue
        fig,axes=plt.subplots(2,len(conditions),figsize=(4.1*len(conditions),6),squeeze=False)
        for column,condition in enumerate(conditions):
            for method,color in (('Iso','tab:blue'),('EA','tab:orange')):
                part=sorted((r for r in values if r['condition']==condition and r['method']==method
                    and r['violation'] is not None),key=lambda r:r['action_scale'])
                for line,metric,multiplier in ((0,'safe_success',100),(1,'time_to_goal_capped_seconds',1)):
                    axis=axes[line,column]
                    axis.plot([100*r['violation'] for r in part],[multiplier*r[metric] for r in part],
                        color=color,label=method,alpha=.7)
                    for row in part:
                        marker={.6:'o',.8:'^',1.:'s'}[row['action_scale']]
                        axis.scatter(100*row['violation'],multiplier*row[metric],marker=marker,color=color,s=38)
                    axis.set_xlabel('Observed violation (%)');axis.grid(alpha=.2)
                    axis.set_ylabel('Safe success (%)' if line==0 else 'Failure-capped time (s)')
                    if line==0:
                        axis.set_ylim(0,105);axis.set_title(condition)
                    else:
                        axis.set_ylim(0,4.2)
            axes[0,column].legend()
        fig.suptitle(f'{split}: common three-scale opportunities; circle0.6 / triangle0.8 / square1.0')
        fig.tight_layout()
        name=f'E02_{split}_frontier'
        fig.savefig(args.output/f'{name}.png',dpi=180);fig.savefig(args.output/f'{name}.pdf')
        plt.close(fig);plots.append(name)
    status='completed' if receipt and receipt['status']=='completed' else 'partial_completed_groups_only'
    (args.output/'report_receipt.json').write_text(json.dumps(dict(status=status,
        source_files_sha256=sources,condition_rows=len(rows),episode_rows=len(episodes),
        source_worker_receipt=receipt,source_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest() if receipt else None,
        new_training_transitions=0,new_simulation_controls=0,policy_reselection=False,plots=plots),indent=2)+'\n')
    lines=['# Frozen E02/E03 results' if status=='completed' else '# Frozen E02/E03 partial evaluation', '',
        f'Status: `{status}`; {len(rows)} completed condition rows, {len(episodes)} attempted episodes.','',
        'This report only reads retained evaluation files. No new rollout, RL or policy reselection.',
        'A validation risk-cap label is not proof of equal actual independent-test risk.',
        'Failure-capped time gives every failed episode the entire native4s horizon.',
        'One training seed; the policy candidate checkpoint age is not total training expenditure.','',
        'Exact per-condition rates/counts/clearance/path/time: `condition_summary.csv`.',
        'All compact episode rows: `episodes.csv`. Missing/admissibility denominators are retained.','']
    if status!='completed':
        lines.extend(['Independent evaluation and/or unseen-condition work is still running; do not call this final.',''])
    for name in plots:
        lines.extend([f'![{name}]({name}.png)',''])
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()

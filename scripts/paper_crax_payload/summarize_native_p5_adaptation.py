"""Read native v11 receipts/curves, retaining failures and matched-step costs."""
import argparse
import csv
import json
from pathlib import Path

def csv_out(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def read_progress(path):
    rows=[]
    if path.exists():
        lines=path.read_text().splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines)-1 or line.endswith('\n'):
                    raise
                # A copied live stream can end halfway through its LAST write;
                # never suppress malformed complete records or earlier rows.
                return rows, True
    return rows, False


def summarize(root, plan):
    training, development, runs, truncated = [], [], {}, []
    for path in sorted(root.rglob('run_config.json')):
        manifest=json.loads(path.read_text())
        if manifest.get('protocol') != plan['protocol']:
            continue
        line, arm=manifest['line'], manifest['arm']
        key=(line, arm)
        if key in runs:
            raise ValueError('Do not silently choose among duplicate attempts; pass a curated native root')
        if manifest['parent_sha256'] != plan['lines'][line]['parent_sha256']:
            raise ValueError('Paired comparison must keep its exact native parent')
        progress, partial=read_progress(path.parent/'progress.jsonl')
        if partial:
            truncated.append(str(path.parent/'progress.jsonl'))
        for row in progress:
            training.append(dict(line=line, arm=arm, **row))
        receipt_path=path.parent/'receipt.json'
        receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        if receipt:
            if receipt['status'] != 'completed' or receipt['new_training_steps'] != 100000:
                raise ValueError('A completed first-stage run must account for all100K steps')
            for k in ('violations','episodes','safe_successes','safe_timeouts'):
                if not progress or receipt['segment_counts'][k] != progress[-1][k]:
                    raise ValueError('Native final receipt and full learning curve disagree')
        runs[key]=dict(manifest=manifest, progress=progress, receipt=receipt, path=str(path.parent))
        for file in sorted(path.parent.glob('development_*.json')):
            value=json.loads(file.read_text())
            if value['line'] != line or value['arm'] != arm or value['new_steps'] not in plan['milestones']:
                raise ValueError('Every DEV node must be the registered native milestone')
            if [c['condition'] for c in value['conditions']] != [c['name'] for c in plan['lines'][line]['conditions']]:
                raise ValueError('Do not discard a difficult condition')
            for c in value['conditions']:
                count=plan['development_episodes_per_condition']
                if len(c['records']) != count or c['episodes'] != count:
                    raise ValueError('Incomplete fixed DEV episode set')
                counts={name:sum(bool(r[name]) for r in c['records']) for name in ('safe_success','violation')}
                timeout=sum(not r['safe_success'] and not r['violation'] for r in c['records'])
                if (counts['safe_success'] != c['safe_success_count'] or counts['violation'] != c['violation_count']
                        or timeout != c['safe_timeout_count'] or sum(counts.values())+timeout != count):
                    raise ValueError('Terminal classes or native aggregate counts disagree')
                development.append(dict(line=line, arm=arm, new_steps=value['new_steps'], condition=c['condition'],
                    episodes=count, safe_success_count=counts['safe_success'], violation_count=counts['violation'],
                    safe_timeout_count=timeout, safe_success=counts['safe_success']/count,
                    violation=counts['violation']/count, episode_return=c['episode_return'],
                    time_to_goal_capped_seconds=c['time_to_goal_capped_seconds'],
                    minimum_distance_m=c['minimum_distance_m']))
    comparisons=[]
    for line in plan['lines']:
        if any((line, arm) not in runs for arm in ('control','p5')):
            continue
        maps={arm:{r['new_steps']:r for r in runs[(line,arm)]['progress']} for arm in ('control','p5')}
        common=sorted(set(maps['control']) & set(maps['p5']))
        for step in common:
            c, p=maps['control'][step],maps['p5'][step]
            comparisons.append(dict(line=line,new_steps=step,
                control_violations=c['violations'],p5_violations=p['violations'],
                violation_reduction=1-p['violations']/c['violations'] if c['violations'] else None,
                control_successes=c['safe_successes'],p5_successes=p['safe_successes'],
                control_timeouts=c['safe_timeouts'],p5_timeouts=p['safe_timeouts']))
    return training, development, comparisons, runs, truncated


def plot_curves(output, training, development, plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(15,11),squeeze=False)
    colors={'control':'#616161','p5':'#1565c0'}
    styles=['-','--',':','-.']
    for row,line in enumerate(plan['lines']):
        for arm in plan['arms']:
            values=sorted((v for v in training if v['line']==line and v['arm']==arm),key=lambda v:v['new_steps'])
            for column,key in enumerate(('violations','safe_successes')):
                axes[row,column].plot([v['new_steps']/1000 for v in values],[v[key] for v in values],
                    color=colors[arm],label=arm)
            for ci,condition in enumerate(plan['lines'][line]['conditions']):
                dev=sorted((v for v in development if v['line']==line and v['arm']==arm
                    and v['condition']==condition['name']),key=lambda v:v['new_steps'])
                axes[row,2].plot([v['new_steps']/1000 for v in dev],[v['safe_success'] for v in dev],
                    linestyle=styles[ci],marker='o',color=colors[arm],label=arm+' '+condition['name'])
        for column,label in enumerate(('Cumulative training violations','Cumulative training safe successes','DEV safe success')):
            ax=axes[row,column];ax.set_title(line+' / '+label);ax.set_xlabel('New adaptation steps (K)')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
        axes[row,2].set_ylim(0,1.03)
    fig.suptitle('All registered nodes and failures retained; DEV, not independent results')
    fig.tight_layout();fig.savefig(output/'learning_curves.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot',action='store_true')
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    values=summarize(args.root,plan)
    training, development, comparisons, runs, truncated=values
    args.output.mkdir(parents=True,exist_ok=False)
    csv_out(args.output/'all_training_progress.csv',training)
    csv_out(args.output/'development_curve.csv',development)
    csv_out(args.output/'matched_training_costs.csv',comparisons)
    receipt=dict(new_training_steps=0,new_simulation_controls=0,
        native_runs=len(runs),completed_native_runs=sum(r['receipt'] is not None for r in runs.values()),
        training_progress_rows=len(training),development_condition_rows=len(development),
        truncated_live_stream_last_lines=truncated,independent_test_used=False,
        source_run_directories=[r['path'] for r in runs.values()])
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if args.plot:
        plot_curves(args.output,training,development,plan)


if __name__=='__main__':
    main()

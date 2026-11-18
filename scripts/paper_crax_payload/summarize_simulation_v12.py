"""Compact all registered v12 DEV nodes/episodes and raw progress; no new physics."""
import argparse
import csv
import json
from pathlib import Path


def output(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();progress=[];nodes=[];episodes=[];receipts=[]
    for d in sorted(a.runs.iterdir()):
        if not d.is_dir():continue
        m=d/'run_config.json'
        if not m.exists():continue
        manifest=json.loads(m.read_text());name=d.name
        cfg=manifest.get('plan',manifest.get('configuration'))
        protocol=manifest.get('protocol',cfg.get('protocol'))
        if not protocol.startswith('simulation_v12_'):raise ValueError('Unexpected native protocol')
        raw=d/'progress.jsonl'
        rows=[json.loads(s) for s in raw.read_text().splitlines()] if raw.exists() else []
        progress.extend(dict(run=name,**r) for r in rows)
        rp=d/'receipt.json'
        if rp.exists():
            r=json.loads(rp.read_text())
            steps=r.get('new_training_steps',r.get('new_training_transitions'))
            if r['status']!='completed' or steps!=100000:raise ValueError('Incomplete or wrong endpoint')
            counts=r.get('segment_counts',{})
            if counts:
                for k,v in counts.items():
                    if rows[-1][k]!=v:raise ValueError('Native curve and receipt counts differ')
            receipts.append(dict(run=name,**r))
        for f in sorted(list(d.glob('development_*.json'))+list(d.glob('evaluation_*.json'))):
            v=json.loads(f.read_text());step=v.get('new_steps',v.get('milestone'))
            for c in v['conditions']:
                rs=c['records'];n=len(rs)
                if n!=c['episodes']:raise ValueError('All registered DEV episodes required')
                successes=sum(bool(r['safe_success']) for r in rs);bad=sum(bool(r['violation']) for r in rs)
                timeouts=sum(not r['safe_success'] and not r['violation'] for r in rs)
                if successes+bad+timeouts!=n:raise ValueError('Invalid terminal classes')
                nodes.append(dict(run=name,new_steps=step,condition=c['condition'],episodes=n,
                    safe_successes=successes,violations=bad,timeouts=timeouts,
                    stall_count=sum(bool(r.get('stall',False)) for r in rs),
                    mean_return=sum(r['episode_return'] for r in rs)/n,
                    mean_capped_seconds=sum(r.get('capped_seconds',r.get('time_to_goal_capped_seconds')) for r in rs)/n))
                for r in rs:
                    episodes.append(dict(run=name,new_steps=step,condition=c['condition'],
                        **{k:x for k,x in r.items() if k not in ('trajectory','trajectory_tcp_xyz','control_details','layout')}))
    a.output.mkdir(parents=True,exist_ok=False)
    for name,values in [('training.csv',progress),('dev_nodes.csv',nodes),('dev_episodes.csv',episodes)]:
        output(a.output/name,values)
    (a.output/'receipt.json').write_text(json.dumps(dict(new_training_steps=0,new_simulation_controls=0,
        complete_runs=len(receipts),completed_native_receipts=receipts,progress_rows=len(progress),
        development_nodes=len(nodes),development_episodes=len(episodes),independent_test_used=False),indent=2)+'\n')


if __name__=='__main__':main()

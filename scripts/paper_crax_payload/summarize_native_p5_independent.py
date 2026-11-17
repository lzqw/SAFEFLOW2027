"""Describe frozen independent outcomes; never select or retrain candidates.

Run on server raw JSON first, then transfer compact_cases/ and CSV locally.
The compact cases retain every episode/metric, omitting only raw trajectories.
No training-seed significance or risk equivalence is inferred from layout CIs.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def write_csv(path, rows):
    if rows:
        with path.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader();writer.writerows(rows)


def wilson(k,n):
    if not n:
        return None,None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-half),min(1.,center+half)


def analyze(root, plan, freeze):
    cases={};episodes=[];inputs=[]
    expected={c['name']:c for c in freeze['cases']}
    for path in sorted(root.rglob('*.json')):
        value=json.loads(path.read_text())
        if not isinstance(value,dict) or not isinstance(value.get('case'),dict) or not isinstance(value.get('records'),list):
            continue
        spec=value['case'];name=spec['name']
        if name not in expected or spec!=expected[name] or name in cases:
            raise ValueError('Frozen cases changed or duplicate result attempts were mixed')
        line=plan['lines'][spec['line']]
        if (len(value['records'])!=100*len(line['conditions']) or any(
                r['case']!=name or r['line']!=spec['line'] for r in value['records'])):
            raise ValueError('Case identity or full registered episode cardinality differs')
        for ci,condition in enumerate(line['conditions']):
            records=[r for r in value['records'] if r['condition']==condition['name']]
            if ({r['layout_seed'] for r in records} != set(range(line['independent_layout_seed_start'],
                    line['independent_layout_seed_start']+100)) or len(records)!=100):
                raise ValueError('Keep every original independent fixture, including nonadmitted ones')
            for r in records:
                if r['action_seed']!=line['independent_action_seed']+ci*1000+r['layout_seed']-line['independent_layout_seed_start']:
                    raise ValueError('Common per-episode action seed differs')
                if r['admitted'] and sum(bool(r[k]) for k in ('safe_success','violation','safe_timeout'))!=1:
                    raise ValueError('Each admitted episode needs exactly one terminal class')
        compact=dict(value,records=[{k:v for k,v in r.items() if k!='trajectory_task_xyz'} for r in value['records']])
        compact['source_full_result_sha256']=value.get('source_full_result_sha256',hashlib.sha256(path.read_bytes()).hexdigest())
        cases[name]=compact
        episodes.extend(compact['records'])
        inputs.append(dict(path=str(path),case=name,source_full_result_sha256=compact['source_full_result_sha256']))
    summary=[]
    for name,value in cases.items():
        spec=value['case']
        for condition in plan['lines'][spec['line']]['conditions']:
            attempted=[r for r in value['records'] if r['condition']==condition['name']]
            admitted=[r for r in attempted if r['admitted']]
            ns=sum(r['safe_success'] for r in admitted);nv=sum(r['violation'] for r in admitted)
            low,high=wilson(nv,len(admitted))
            summary.append(dict(case=name,line=spec['line'],method=spec.get('arm','historical_Iso_reference'),
                condition=condition['name'],attempted=len(attempted),admitted=len(admitted),
                nonadmitted=len(attempted)-len(admitted),safe_successes=ns,violations=nv,
                safe_timeouts=sum(r['safe_timeout'] for r in admitted),
                safe_success_per_attempted=ns/len(attempted),
                safe_success_per_admitted=ns/len(admitted) if admitted else None,
                violation_per_admitted=nv/len(admitted) if admitted else None,
                violation_wilson95_low=low,violation_wilson95_high=high,
                mean_return=mean(r['episode_return'] for r in admitted) if admitted else None,
                mean_capped_seconds=mean(r['time_to_goal_capped_seconds'] for r in admitted) if admitted else None,
                mean_path_m=mean(r['task_path_length_m'] for r in admitted) if admitted else None,
                mean_minimum_distance_m=mean(r['full_interval_minimum_distance_m'] for r in admitted) if admitted else None))
    paired=[]
    for line_name,line in plan['lines'].items():
        choices=[(n,v) for n,v in cases.items() if v['case']['line']==line_name]
        p5=[(n,v) for n,v in choices if v['case'].get('arm')=='p5']
        if not p5:
            continue
        pn,pv=p5[0]
        for other,ov in choices:
            if other==pn:
                continue
            for condition in line['conditions']:
                a={r['layout_seed']:r for r in pv['records'] if r['condition']==condition['name']}
                b={r['layout_seed']:r for r in ov['records'] if r['condition']==condition['name']}
                keys=sorted(k for k in a.keys() & b.keys() if a[k]['admitted'] and b[k]['admitted'])
                pc=mean(a[k]['time_to_goal_capped_seconds'] for k in keys) if keys else None
                oc=mean(b[k]['time_to_goal_capped_seconds'] for k in keys) if keys else None
                paired.append(dict(line=line_name,p5_case=pn,comparison_case=other,condition=condition['name'],
                    common_admitted=len(keys),p5_only_safe=sum(a[k]['safe_success'] and not b[k]['safe_success'] for k in keys),
                    comparison_only_safe=sum(b[k]['safe_success'] and not a[k]['safe_success'] for k in keys),
                    both_safe=sum(a[k]['safe_success'] and b[k]['safe_success'] for k in keys),
                    neither_safe=sum(not a[k]['safe_success'] and not b[k]['safe_success'] for k in keys),
                    p5_violations=sum(a[k]['violation'] for k in keys),comparison_violations=sum(b[k]['violation'] for k in keys),
                    p5_capped_seconds=pc,comparison_capped_seconds=oc,
                    p5_capped_time_reduction=1-pc/oc if oc else None,
                    speed_reference_only=bool(ov['case'].get('historical_reference'))))
    return cases,episodes,summary,paired,inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text());freeze=json.loads(args.freeze.read_text())
    cases,episodes,summary,paired,inputs=analyze(args.root,plan,freeze)
    args.output.mkdir(parents=True,exist_ok=False);compact=args.output/'compact_cases';compact.mkdir()
    for name,value in cases.items():
        (compact/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    write_csv(args.output/'all_episodes.csv',episodes)
    write_csv(args.output/'condition_summary.csv',summary)
    write_csv(args.output/'paired_outcomes.csv',paired)
    receipt=dict(new_training_steps=0,new_simulation_controls=0,source_results=inputs,
        complete_frozen_case_names=sorted(cases),missing_frozen_cases=sorted({c['name'] for c in freeze['cases']}-cases.keys()),
        attempted_episodes=len(episodes),admitted_episodes=sum(r['admitted'] for r in episodes),
        distinct_layout_seeds=len({r['layout_seed'] for r in episodes}),
        interpretation='Descriptive paired layouts; Wilson intervals exclude training-seed uncertainty; no equivalence or model reselection')
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    main()

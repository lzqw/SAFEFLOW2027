"""Stream complete native component traces; retain success, failure and censoring."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def resample(points, spacing):
    points=np.asarray(points,dtype=float)
    length=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    keep=np.r_[True,np.diff(length)>1e-12]
    length,points=length[keep],points[keep]
    if len(points)==1:return points
    ticks=np.r_[np.arange(0.,length[-1],spacing),length[-1]]
    return np.stack([np.interp(ticks,length,points[:,i]) for i in range(3)],axis=1)


def cells(points, grid, spacing):
    return {tuple(v) for v in np.floor(resample(points,spacing)/grid).astype(int)}


def route(points, obstacle):
    """First TCP y-plane crossing, descriptive only, not a legal homotopy."""
    obstacle=np.asarray(obstacle);points=np.asarray(points)
    for a,b in zip(points[:-1],points[1:]):
        da,db=a[1]-obstacle[1],b[1]-obstacle[1]
        if da*db<=0 and abs(db-da)>1e-12:
            x=(a+(b-a)*(-da)/(db-da))[0]-obstacle[0]
            return 'negative_x' if x<0 else 'nonnegative_x'
    return 'no_y_plane_crossing'


def analyze(directory):
    directory=Path(directory);manifest=json.loads((directory/'run_config.json').read_text())
    config=manifest['configuration']['trajectory_analysis']
    windows=[dict(start=a,end=b,transitions=0,successes=0,violations=0,timeouts=0,
                  all_cells=set(),successful=[],routes={}) for a,b in config['windows']]
    active={};curve=[];total=successes=violations=timeouts=0;explicit_end=False
    with gzip.open(directory/'trajectories.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);kind=row['record_type']
            if kind=='initial':
                key=(row['env'],row['episode_id'])
                active[key]=dict(start=row['transitions'],points=[row['tcp_xyz']],obstacle=row['obstacle'])
            elif kind=='step':
                total+=1
                if row['transitions']!=total:raise ValueError('Trace has missing or duplicated transitions')
                key=(row['env'],row['episode_id']);episode=active[key]
                previous=episode['points'][-1];episode['points'].append(row['tcp_xyz'])
                window=next(w for w in windows if w['start']<total<=w['end'])
                window['transitions']+=1
                window['all_cells'].update(cells([previous,row['tcp_xyz']],config['world_grid_m'],config['arclength_spacing_m']))
                if row['done']:
                    success=bool(row['safe_success']);violation=bool(row['violation'])
                    timeout=not success and not violation
                    successes+=success;violations+=violation;timeouts+=timeout
                    window['successes']+=success;window['violations']+=violation;window['timeouts']+=timeout
                    if success:
                        label=route(episode['points'],episode['obstacle'])
                        window['routes'][label]=window['routes'].get(label,0)+1
                        window['successful'].append(dict(end=total,start=episode['start'],
                            cells=cells(episode['points'],config['world_grid_m'],config['arclength_spacing_m']),
                            path_m=float(np.linalg.norm(np.diff(episode['points'],axis=0),axis=1).sum())))
                    del active[key]
                    curve.append(dict(steps=total,successes=successes,violations=violations,timeouts=timeouts))
            elif kind=='end':explicit_end=True
    receipt=json.loads((directory/'receipt.json').read_text()) if (directory/'receipt.json').exists() else None
    if receipt is not None and (receipt['steps']!=total or receipt['safe_successes']!=successes or receipt['training_violations']!=violations):
        raise ValueError('Trace totals disagree with native receipt')
    return dict(arm=manifest['arm']['name'],source_commit=manifest['source_commit'],
        steps=total,successes=successes,violations=violations,timeouts=timeouts,
        censored_episodes=sum(len(e['points'])>1 for e in active.values()),
        explicit_trace_end=explicit_end,completed_receipt=receipt is not None,
        windows=windows,curve=curve)


def compare(directories):
    results=[analyze(d) for d in directories]
    if len({r['arm'] for r in results})!=len(results):raise ValueError('Duplicate arm')
    core=[r for r in results if r['arm'] in ('iso','noise-only','metric-only','full')]
    matching=[]
    if len(core)==4:
        for i,w in enumerate(core[0]['windows']):
            count=min(len(r['windows'][i]['successful']) for r in core)
            rows=[]
            for result in core:
                sample=result['windows'][i]['successful'][:count]
                union=set().union(*(e['cells'] for e in sample))
                rows.append(dict(arm=result['arm'],matched_episodes=count,
                    available_successes=len(result['windows'][i]['successful']),unique_world_cells=len(union),
                    mean_path_m=float(np.mean([e['path_m'] for e in sample])) if sample else None))
            matching.append(dict(start=w['start'],end=w['end'],arms=rows))
    # Common actual violation budgets, no interpolated or risk-rounded episodes.
    maximum=min(r['violations'] for r in results)
    budgets=sorted({0,maximum,*range(100,maximum+1,100)})
    frontier=[]
    for budget in budgets:
        rows=[]
        for result in results:
            eligible=[c for c in result['curve'] if c['violations']<=budget]
            point=eligible[-1] if eligible else dict(steps=0,successes=0,violations=0,timeouts=0)
            rows.append(dict(arm=result['arm'],**point))
        frontier.append(dict(maximum_training_violations=budget,arms=rows))
    for result in results:
        for w in result['windows']:
            w['unique_world_cells']=len(w.pop('all_cells'))
            w['successful_episode_count']=len(w.pop('successful'))
    return dict(results=results,matched_success_coverage=matching,equal_violation_budget=frontier,
        descriptive_world_grid_only=True,legal_route_or_homotopy_proof=False,
        independent_training_seed_statistics=False,unfinished_episodes_not_failures=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior analysis')
    result=compare(args.runs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

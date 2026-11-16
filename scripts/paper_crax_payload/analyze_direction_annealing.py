"""Exact suffix terminal chronology, no new simulation or training.

The first two episodes can inherit parent controls. Count their terminal events
in the suffix and mark them explicitly, never claim full suffix episode paths.
Only completed original250K branches may be summarized for final comparison.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from scripts.paper_crax_payload.direction_annealing import FORMAT, sha256, validate_config


def terminal_chronology(rows, start, end, milestones):
    active = {}; curve = []; last = start; ended = False; inherited_terminals = []
    counts = dict(successes=0, violations=0, timeouts=0)
    windows = [dict(start=a, end=b, transitions=0, successes=0, violations=0, timeouts=0)
               for a,b in zip([start,*milestones[:-1]],milestones)]
    for row in rows:
        kind = row['record_type']
        if ended: raise ValueError('Records after explicit suffix end')
        if kind == 'initial':
            key = (row['env'], row['episode_id'])
            if key in active: raise ValueError('Never reset an unfinished inherited episode')
            active[key] = dict(start_steps=row['transitions'], suffix_length=0,
                inherited_length=row.get('inherited_partial_episode_length',0))
        elif kind == 'step':
            last += 1
            if row['transitions'] != last: raise ValueError('Missing or duplicate suffix transition')
            episode = active[(row['env'],row['episode_id'])]; episode['suffix_length'] += 1
            window = next(w for w in windows if w['start'] < last <= w['end'])
            window['transitions'] += 1
            if row['violation'] and not row['done']: raise ValueError('Original first violation is terminal')
            if row['done']:
                if row['safe_success'] and row['violation']: raise ValueError('Unsafe success not admissible')
                label = 'successes' if row['safe_success'] else 'violations' if row['violation'] else 'timeouts'
                counts[label] += 1; window[label] += 1
                if episode['inherited_length']:
                    inherited_terminals.append(dict(env=row['env'], episode_id=row['episode_id'], end=last,
                        outcome=label, **episode))
                del active[(row['env'],row['episode_id'])]
                curve.append(dict(steps=last, new_steps=last-start, **counts))
        elif kind == 'end':
            if row['transitions'] != end or row['new_transitions'] != end-start:
                raise ValueError('Wrong closed suffix budget')
            ended = True
        elif kind in ('fork','resume'):
            if row['transitions'] != last: raise ValueError('Incorrect fork/resume chronology')
        else: raise ValueError('Unrecognized native trace record')
    if not ended or last != end or sum(w['transitions'] for w in windows) != end-start:
        raise ValueError('Require the entire closed suffix, not a promising prefix')
    return dict(absolute_start=start, absolute_end=end, new_training_transitions=end-start,
        **counts, curve=curve, windows=windows, inherited_episode_terminals=inherited_terminals,
        censored_episodes=[dict(env=env,episode_id=episode_id,**episode)
                           for (env,episode_id),episode in active.items() if episode['suffix_length'] > 0],
        unfinished_episodes_not_counted_as_failures=True,
        terminal_events_counted_in_suffix_even_if_episode_inherited=True)


def summarize_run(directory):
    manifest = json.loads((directory/'run_config.json').read_text())
    receipt = json.loads((directory/'receipt.json').read_text()); config = manifest['configuration']
    validate_config(config)
    if (manifest['format'] != FORMAT or manifest['engineering_only'] or receipt['engineering_only']
            or receipt['status'] != 'completed' or receipt['steps'] != 250000
            or receipt['new_training_transitions'] != 150000 or manifest['arm'] != receipt['arm']
            or receipt['source_commit'] != manifest['source_commit'] or receipt['parent'] != manifest['parent']
            or manifest['parent_checkpoint_sha256'] != config['parent']['checkpoint_sha256']
            or receipt['independent_endpoint_test_executed']):
        raise ValueError('Require a genuine completed registered250K fork')
    with gzip.open(directory/'trajectories.jsonl.gz','rt') as stream:
        result = terminal_chronology((json.loads(line) for line in stream),100000,250000,
                                    config['evaluation']['milestones'])
    for key,native_key in [('successes','safe_successes'),('violations','training_violations'),('timeouts','safe_timeouts')]:
        if (result[key] != receipt['new_'+native_key]
                or result[key]+receipt['parent']['prefix'][native_key] != receipt[native_key]):
            raise ValueError('Suffix/parent/native receipt counts disagree')
    if sum(result[k] for k in ('successes','violations','timeouts')) != receipt['episodes']-receipt['parent']['prefix']['episodes']:
        raise ValueError('Completed episode total disagrees with native receipt')
    result.update(arm=manifest['arm']['name'], training_seed=2028,
        parent_checkpoint_sha256=manifest['parent_checkpoint_sha256'], parent_prefix=receipt['parent']['prefix'],
        source_commit=manifest['source_commit'], probe_sha256=manifest['probe_sha256'],
        configuration_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        receipt_sha256=sha256(directory/'receipt.json'), trajectory_sha256=sha256(directory/'trajectories.jsonl.gz'),
        analysis_script_sha256=sha256(Path(__file__)), independent_confirmation=False,
        new_simulation_controls=0, new_training_transitions_from_analysis=0)
    return result


def compare(results):
    if len(results) != 3 or {r['arm'] for r in results} != {'A-fixed','B-loss-anneal','C-joint-anneal'}:
        raise ValueError('Retain all three complete branches, not only the best two')
    for key in ('parent_checkpoint_sha256','source_commit','probe_sha256','configuration_sha256','analysis_script_sha256'):
        if len({r[key] for r in results}) != 1: raise ValueError('Paired suffix provenance mismatch')
    if any(r['absolute_start'] != 100000 or r['absolute_end'] != 250000 for r in results):
        raise ValueError('Compare identical complete suffix budgets')
    maximum = min(r['violations'] for r in results)
    budgets = sorted({0,maximum,*range(100,maximum+1,100)})
    frontier=[]
    for budget in budgets:
        arms=[]
        for result in results:
            eligible=[row for row in result['curve'] if row['violations'] <= budget]
            point=eligible[-1] if eligible else dict(steps=100000,new_steps=0,successes=0,violations=0,timeouts=0)
            arms.append(dict(arm=result['arm'],**point))
        frontier.append(dict(maximum_new_training_violations=budget,arms=arms))
    return dict(results=results,equal_suffix_violation_budget=frontier, training_seed_count=1,
        independent_confirmation=False,method_selected=False,common_parent_steps_not_recounted=True,
        scope='paired_development_suffix_only_no_topology_or_generalization_claims',
        new_simulation_controls=0,new_training_transitions_from_analysis=0)


def main():
    p=argparse.ArgumentParser(description=__doc__); group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',type=Path); group.add_argument('--summaries',nargs=3,type=Path)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve completed suffix evidence; no overwrite')
    result=summarize_run(args.run) if args.run else compare([json.loads(path.read_text()) for path in args.summaries])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(args.output),new_training_transitions_from_analysis=0,new_simulation_controls=0)))


if __name__ == '__main__': main()

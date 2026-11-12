"""Collect only compact native evidence, never replay/weights, and refresh tables.

This is a report snapshot, not a process liveness checker or restart mechanism.
Running counters are lower bounds. Authoritative completion requires native receipt.
"""
import argparse
import csv
import io
import json
from pathlib import Path,PurePosixPath
import shlex
import subprocess
import tarfile


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def collect(host,root,report):
    command=f"cd {shlex.quote(root)} && find . -maxdepth 2 -type f \\( -name '*.json' -o -name '*.jsonl' -o -name 'progress.csv' -o -name 'training_episodes.csv' \\) -print0 | tar --null -czf - -T -"
    data=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',host,command],timeout=55)
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        for member in archive:
            relative=PurePosixPath(member.name)
            if not member.isfile() or relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe/nonregular remote artifact')
            target=report/'native_runs'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())


def refresh(report):
    results=[];trials=[];physical=0;training=0
    for directory in sorted((report/'native_runs').glob('*')):
        config=read(directory/'run_config.json') or read(directory/'configuration.json')
        if not config:continue
        receipt=read(directory/'receipt.json');status=read(directory/'status.json')
        history=[]
        if (directory/'progress.jsonl').exists():
            history=[json.loads(line) for line in (directory/'progress.jsonl').read_text().splitlines() if line.strip()]
        latest=receipt or status or (history[-1] if history else {})
        steps=latest.get('steps',0);initial=config.get('initial_steps',0)
        new=receipt.get('newly_collected_transitions',receipt.get('new_training_transitions',steps-initial))
        training+=new
        cost=latest.get('training_violation_count',latest.get('training_cumulative_cost',latest.get('total_training_cumulative_cost')))
        queries=0
        for evaluation_path in sorted(directory.glob('evaluation_*.json')):
            evaluation=read(evaluation_path);queries+=evaluation.get('physical_evaluation_transitions',0)
            for condition in evaluation.get('conditions',[]):
                results.append(dict(phase=config.get('phase','development'),trial_id=directory.name,
                    method=config.get('method'),training_seed=config.get('training_seed',config.get('seed')),
                    steps=evaluation['checkpoint_steps'],checkpoint=evaluation_path.stem.replace('evaluation','policy'),
                    selection_role='fixed_endpoint' if evaluation.get('fixed_endpoint') else 'development_checkpoint',
                    condition=condition['condition'],evaluation_seed=config['configuration']['evaluation']['development_layout_seed_start'],
                    episodes=condition['episodes'],safe_success=condition['safe_success'],violation=condition['violation'],
                    safe_timeout=condition['safe_timeout_count']/condition['episodes'],native_return=condition['episode_return'],
                    capped_time_seconds=condition['time_to_goal_capped_seconds'],conditional_success_time_seconds=condition['time_to_goal_seconds'],
                    training_cumulative_cost=evaluation['training_violation_count'],completed_training_episodes=evaluation['training_episodes'],
                    source_commit=config['source_commit']))
        physical+=queries
        trials.append(dict(trial_id=directory.name,status='completed' if receipt.get('status')=='completed' else status.get('state',status.get('status','observed_not_liveness_proof')),
            steps=steps,new_training_transitions_lower_bound=new,training_cost_lower_bound=cost,
            evaluation_controls_observed=queries,method=config.get('method'),training_seed=config.get('training_seed',config.get('seed')),
            pid=config.get('native_pid'),source_commit=config.get('source_commit')))
    path=report/'PAYLOAD_RESULTS.csv'
    fields=next(csv.reader(path.open()))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(results)
    summary=dict(completed_native_runs=sum(r['status']=='completed' for r in trials),observed_native_runs=len(trials),
        observed_new_training_transitions_lower_bound=training,observed_native_evaluation_controls=physical,
        trials=trials,payload_condition_rows=len(results),whole_goal_complete=False,
        note='report-only snapshot;no live process claims,no new training,no physics,no large checkpoints copied')
    (report/'NATIVE_PROGRESS.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',choices=['4080','tjucta-a100']);p.add_argument('--root')
    p.add_argument('--report',type=Path,default=Path('reports/paper_crax_payload'))
    args=p.parse_args()
    if args.host:
        if not args.root:raise ValueError('Exact server run root required')
        collect(args.host,args.root,args.report)
    result=refresh(args.report)
    print(json.dumps({k:v for k,v in result.items() if k!='trials'}))


if __name__=='__main__':main()

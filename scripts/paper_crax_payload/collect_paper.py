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
    results=[];point_results=[];trials=[];candidate_history=[];physical=0;training=0
    ledger_path=report/'TRIAL_LEDGER.csv'
    with ledger_path.open() as stream:
        ledger_reader=csv.DictReader(stream);ledger_fields=ledger_reader.fieldnames;ledger={r['trial_id']:r for r in ledger_reader}
    failures=read(report/'CRAX_PROFILE_V1_FAILURES.json')
    failed_jobs={r['trial_id']:r for r in failures.get('jobs',[])}
    for directory in sorted((report/'native_runs').glob('*')):
        config=read(directory/'run_config.json') or read(directory/'configuration.json')
        if not config:continue
        if 'settings' in config:
            config=config|config['settings']
            config['method']='sac_lag_native'
            config['phase']='engineering' if config.get('engineering_check') else 'development'
        receipt=read(directory/'receipt.json');status=read(directory/'status.json')
        checkpoint_status=read(directory/'checkpoint_status.json')
        history=[]
        if (directory/'progress.jsonl').exists():
            history=[json.loads(line) for line in (directory/'progress.jsonl').read_text().splitlines() if line.strip()]
        latest=receipt or status or (history[-1] if history else {})
        steps=max(latest.get('steps',0),checkpoint_status.get('steps',0));initial=receipt.get('initial_steps',checkpoint_status.get('initial_steps',config.get('initial_steps',0)))
        new=receipt.get('newly_collected_transitions',receipt.get('new_training_transitions',steps-initial))
        training+=new
        cost=latest.get('training_violation_count',latest.get('training_cumulative_cost',latest.get('total_training_cumulative_cost',checkpoint_status.get('training_cumulative_cost'))))
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
        for validation_path in sorted(directory.glob('validation_*.json')):
            validation=read(validation_path)
            if 'physical_controls' not in validation:continue
            queries+=validation['physical_controls']
            endpoint=validation.get('checkpoint_steps',int(validation_path.stem.split('_')[-1]))
            point_results.append(dict(phase=config.get('phase','development'),trial_id=directory.name,
                method=config.get('method'),training_seed=config.get('seed'),steps=endpoint,
                checkpoint=f'policy_{endpoint}' if config.get('method')!='sac_lag_native' else 'policy_normalizer',
                selection_role='development_validation_not_independent_test',evaluation_seed=validation['seed'],episodes=validation['episodes'],
                native_return=validation['native_return'],episode_cost=validation['episode_cost'],goals_per_episode=validation['goals_per_episode'],
                at_least_one_goal_fraction=validation['at_least_one_goal'],budget_compliant_fraction=validation['budget_compliant_fraction'],
                budget_compliant_return=validation['budget_compliant_return'],budget_compliant_episodes=validation['budget_compliant_episodes'],
                training_cumulative_cost=validation.get('training_cumulative_cost'),
                cost_per_1000_transitions=1000*validation['training_cumulative_cost']/endpoint if validation.get('training_cumulative_cost') is not None else None,
                full_training_sps=receipt.get('full_training_including_compile_eval_save_sps'),
                capped_first_goal_seconds=validation.get('capped_first_goal_seconds'),
                conditional_first_goal_seconds=validation.get('conditional_first_goal_seconds'),source_commit=config['source_commit']))
        if 'native_evaluation_controls_this_invocation' in receipt:
            # Offline common validation is ADDITIONAL to native diagnostics.
            # A resumed segment must not inherit the parent's diagnostic count.
            queries+=receipt['native_evaluation_controls_this_invocation']
        else:
            inherited_calls=0
            resume_path=config.get('resume_from')
            if resume_path:
                inherited_calls=read(report/'native_runs'/PurePosixPath(resume_path).parent.name/'checkpoint_status.json').get('evaluation_calls',0)
            native_calls=max(0,checkpoint_status.get('evaluation_calls',0)-inherited_calls)
            queries=max(queries,receipt.get('evaluation_controls',0),receipt.get('physical_evaluation_transitions',0),native_calls*50*config.get('episode_length',1000))
        physical+=queries
        trial_status='completed' if receipt.get('status')=='completed' else status.get('state',status.get('status','observed_not_liveness_proof'))
        if directory.name in failed_jobs:trial_status='failed_preheat_logging;source_inferred_steps_and_full_budget_reserve_in_CRAX_PROFILE_V1_FAILURES'
        task=('Piper-'+str(config.get('task'))) if config.get('robot_family')=='Piper' else ('payload' if config.get('task')=='SafePayloadTransport-v1' else 'PointGoal')
        ledger[directory.name]=dict(trial_id=directory.name,task=task,phase=config.get('phase','development'),status=trial_status,
            method=config.get('method'),seed=config.get('training_seed',config.get('seed')),
            actual_new_training_transitions=new,new_nontraining_simulator_controls=queries,new_real_robot_interactions=0,
            evidence=f'native_runs/{directory.name}/'+('receipt.json' if receipt else 'status.json'),
            notes='actual completed segment' if receipt else 'observed lower bound;not process-liveness proof;all prefixes retained')
        candidates=[dict(checkpoint_steps=int(v.stem.split('_')[-1]),evaluation_file=str(v.relative_to(report)))
            for v in sorted(directory.glob('evaluation_*.json'))]
        candidates += [dict(checkpoint_steps=int(v.stem.split('_')[-1]),evaluation_file=str(v.relative_to(report)))
            for v in sorted(directory.glob('validation_*.json'))]
        if candidates:candidate_history.append(dict(trial_id=directory.name,task=task,method=config.get('method'),
            training_seed=config.get('training_seed',config.get('seed')),observed_checkpoints=candidates,
            selection_performed=False,test_used_for_selection=False))
        trials.append(dict(trial_id=directory.name,status=trial_status,
            steps=steps,new_training_transitions_lower_bound=new,training_cost_lower_bound=cost,
            evaluation_controls_observed=queries,method=config.get('method'),training_seed=config.get('training_seed',config.get('seed')),
            pid=config.get('native_pid'),source_commit=config.get('source_commit')))
    path=report/'PAYLOAD_RESULTS.csv'
    fields=next(csv.reader(path.open()))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(results)
    path=report/'CRAX_POINTGOAL_RESULTS.csv'
    fields=list(dict.fromkeys(next(csv.reader(path.open()))+['capped_first_goal_seconds','conditional_first_goal_seconds']))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(point_results)
    with ledger_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=ledger_fields);writer.writeheader();writer.writerows(ledger.values())
    selection=read(report/'MODEL_SELECTION.json');selection['candidate_history']=candidate_history
    selection['selection_has_occurred']=bool(selection.get('selected_candidates'))
    (report/'MODEL_SELECTION.json').write_text(json.dumps(selection,indent=2)+'\n')
    summary=dict(completed_native_runs=sum(r['status']=='completed' for r in trials),observed_native_runs=len(trials),
        observed_new_training_transitions_lower_bound=training,observed_native_evaluation_controls=physical,
        trials=trials,payload_condition_rows=len(results),pointgoal_validation_rows=len(point_results),whole_goal_complete=False,
        note='report-only snapshot;no live process claims,no new training,no physics,no large checkpoints copied')
    summary['crax_v1_failure_training_budget_reserve']=failures.get('conservative_development_budget_reservation',0)
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

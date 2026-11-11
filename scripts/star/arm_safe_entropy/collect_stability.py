"""Mirror compact new-stage artifacts and preserve every trial,never rerun/select.

Collectors must be run serially for the two hosts because their report CSVs
are shared. Full replay/policies remain on the servers.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess


COLUMNS=['stage','trial','variant','training_seed','checkpoint_steps','condition','evaluation_episodes',
    'safe_success','violation_rate','safe_timeout_rate','return','time_to_goal_capped_seconds',
    'time_to_goal_seconds','time_to_goal_successes','minimum_distance_m','training_violation_count',
    'training_episodes','native_run_completed','source_commit','evidence',
    'discarded_tail_logged_violations','accounted_training_violation_count_lower_bound']


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write_csv(path,rows,fields):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def discarded_parent_violations(folder,runs,seen=None):
    """Known costs from discarded parent tails still count,not just replay lineage."""
    seen=set() if seen is None else seen
    if folder.name in seen: raise ValueError('Native parent lineage cannot cycle')
    seen.add(folder.name)
    config=read_json(folder/'run_config.json')
    if not config.get('parent_checkpoint'): return 0
    parent=runs/Path(config['parent_checkpoint']).parent.name
    incident=read_json(parent/'engineering_interruption.json')
    return int(incident.get('logged_lost_tail_violations',0))+discarded_parent_violations(parent,runs,seen)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host')
    parser.add_argument('--project')
    args=parser.parse_args()
    if bool(args.host)!=bool(args.project): parser.error('Use host and project together')
    root=Path(__file__).resolve().parents[3]
    report=root/'reports/star/arm_ea_stability'
    # A directory literally named runs is globally gitignored; compact mirrored
    # evidence belongs here,not in the server's large-file namespace.
    runs=report/'native_runs';runs.mkdir(parents=True,exist_ok=True)
    if args.host:
        subprocess.run(['rsync','-a','--include=*/','--include=*.json','--include=progress.csv','--exclude=*',
            '-e','ssh -o BatchMode=yes -o ConnectTimeout=10',
            args.host+':'+args.project.rstrip('/')+'/runs/ea_stability_v2/',str(runs)+'/'],check=True,timeout=120)
    ledger_path=report/'TRIAL_LEDGER.csv'
    if ledger_path.exists():
        with ledger_path.open() as stream:
            reader=csv.DictReader(stream);ledger_fields=list(reader.fieldnames);ledger={r['trial_id']:r for r in reader}
    else:
        ledger_fields=['trial_id'];ledger={}
    for key in ['phase','variant','parent_checkpoint','seed','target_steps','initial_steps',
                'actual_new_training_transitions','physical_evaluation_transitions','host','source_commit',
                'status','change','hypothesis','result','evidence','training_violation_count',
                'segment_new_violations','completed_episodes','counter_scope']:
        if key not in ledger_fields: ledger_fields.append(key)
    rows=[];initial_pairs=set();complete_initial=set();logged_native=engineering=0
    for path in sorted(runs.glob('*/run_config.json')):
        folder=path.parent;config=read_json(path)
        if config.get('stage') not in ('ea_stability_v2_development','ea_stability_v2_engineering'):
            raise ValueError('Do not silently pool another protocol')
        receipt=read_json(folder/'receipt.json');status=read_json(folder/'status.json')
        incident=read_json(folder/'engineering_interruption.json')
        complete=receipt.get('status')=='completed'
        counter=receipt if complete else status
        steps=counter.get('steps',config['initial_steps'])
        new=steps-config['initial_steps']
        if new<0: raise ValueError('Native transition counter moved backwards')
        native=config['stage']=='ea_stability_v2_development'
        if native:
            logged_native+=new
            if config['initial_steps']==0 and config['total_transition_budget']==300000:
                pair=(config['variant'],config['training_seed'])
                if pair in initial_pairs: raise ValueError('Duplicate initial development run is not a seed replacement')
                initial_pairs.add(pair)
                if complete: complete_initial.add(pair)
            elif config.get('engineering_recovery') and config['total_transition_budget']==300000 and complete:
                complete_initial.add((config['variant'],config['training_seed']))
        else: engineering+=new
        row=ledger.get(folder.name,dict(trial_id=folder.name,change='native_runner',hypothesis='engineering_contract' if not native else 'declared_development'))
        violations=counter.get('training_violation_count')
        initial_cost=config.get('initial_training_violations',0 if config['initial_steps']==0 else None)
        if config.get('parent_checkpoint'):
            parent=runs/Path(config['parent_checkpoint']).parent.name
            parent_receipt=read_json(parent/'receipt.json')
            if parent_receipt.get('steps')==config['initial_steps']:
                initial_cost=parent_receipt['training_violation_count']
        row.update(phase='engineering' if not native else 'B_recovery' if config.get('engineering_recovery') else 'B' if config['initial_steps']==0 else 'C',
            variant=config['variant'],parent_checkpoint=config.get('parent_checkpoint') or 'none_fresh',
            seed=config['training_seed'],target_steps=config['total_transition_budget'],initial_steps=config['initial_steps'],
            actual_new_training_transitions=new,
            physical_evaluation_transitions=(receipt.get('new_physical_evaluation_transitions',0) if complete else
                sum(read_json(p).get('physical_evaluation_transitions',0) for p in folder.glob('evaluation_*.json'))),
            host=config['host'],source_commit=config['source_commit'],status='completed' if complete else
                'engineering_interrupted_retained' if incident.get('native_process_confirmed_terminal') else status.get('state','starting'),
            result='native_counter_not_efficacy_claim',evidence=str(folder.relative_to(report)),
            training_violation_count=violations if violations is not None else '',
            segment_new_violations=violations-initial_cost if violations is not None and initial_cost is not None else '',
            completed_episodes=counter.get('training_episodes',counter.get('episodes','')),
            counter_scope='new_steps_incremental;cost_and_episodes_cumulative_native_lineage;running_steps_logged_lower_bound')
        ledger[folder.name]=row
        if not native: continue
        for evaluation_path in sorted(folder.glob('evaluation_*.json')):
            evaluation=read_json(evaluation_path)
            for condition in evaluation['conditions']:
                count=condition['episodes']
                if len(condition['records'])!=count or sum(condition[k] for k in
                        ('safe_success_count','safe_timeout_count','violation_failure_count'))!=count:
                    raise ValueError('An evaluation must preserve all three terminal outcomes')
                rows.append(dict(stage='development_fixed_1M' if evaluation['milestone_steps']==1000000 else 'development_intermediate',
                    trial=folder.name,variant=config['variant'],training_seed=config['training_seed'],
                    checkpoint_steps=evaluation['milestone_steps'],condition=condition['condition']['name'],evaluation_episodes=count,
                    safe_success=condition['safe_success'],violation_rate=condition['violation'],
                    safe_timeout_rate=condition['safe_timeout_count']/count,**{'return':condition['episode_return']},
                    time_to_goal_capped_seconds=condition['time_to_goal_capped_seconds'],
                    time_to_goal_seconds=condition['time_to_goal_seconds'],time_to_goal_successes=condition['time_to_goal_successes'],
                    minimum_distance_m=condition['minimum_distance_m'],training_violation_count=evaluation['training_violation_count'],
                    training_episodes=evaluation['training_episodes'],native_run_completed=complete,
                    source_commit=config['source_commit'],evidence=str(evaluation_path.relative_to(report)),
                    discarded_tail_logged_violations=discarded_parent_violations(folder,runs),
                    accounted_training_violation_count_lower_bound=evaluation['training_violation_count']+
                        discarded_parent_violations(folder,runs)))
    write_csv(report/'DEVELOPMENT_RESULTS.csv',rows,COLUMNS)
    write_csv(report/'FINAL_RESULTS.csv',[r for r in rows if r['checkpoint_steps']==1000000 and r['native_run_completed']],COLUMNS)
    write_csv(ledger_path,list(ledger.values()),ledger_fields)
    recorded_nontraining=sum(int(r.get('physical_evaluation_transitions') or 0) for r in ledger.values())
    tail_reserve=sum(read_json(p).get('unlogged_tail_training_budget_reserve',0)
                     for p in runs.glob('*/engineering_interruption.json'))
    summary=dict(initial_expected_runs=6,initial_observed_runs=len(initial_pairs),initial_completed_runs=len(complete_initial),
        scientific_logged_new_training_transitions_lower_bound=logged_native,
        engineering_logged_new_training_transitions=engineering,total_logged_new_training_transitions_lower_bound=logged_native+engineering,
        unlogged_interruption_tail_training_budget_reserve=tail_reserve,
        conservative_training_budget_charge=logged_native+engineering+tail_reserve,
        recorded_nontraining_physical_control_intervals=recorded_nontraining,
        development_evaluation_rows=len(rows),fixed1M_completed_condition_rows=sum(r['checkpoint_steps']==1000000 and r['native_run_completed'] for r in rows),
        new_evaluations_launched=0,large_checkpoints_copied=0,formal_confirmation_started=False,whole_goal_complete=False)
    temporary=report/'DEVELOPMENT_STATUS.json.tmp'
    temporary.write_text(json.dumps(summary,indent=2)+'\n');temporary.replace(report/'DEVELOPMENT_STATUS.json')
    print(json.dumps(summary))


if __name__=='__main__':
    main()

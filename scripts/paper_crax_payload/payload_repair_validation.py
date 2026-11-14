"""Wait for four finite native runs; equal-candidate validation only, no test/RL."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def sha256(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def native_live(pid,expected):
    try:
        command=Path('/proc',str(pid),'cmdline').read_bytes()
        return bool(command and expected.encode() in command)
    except OSError:
        return False


def training_state(root,plan,job):
    run=root/'runs'/job['run']
    path=run/'receipt.json'
    if path.exists():
        receipt=json.loads(path.read_text())
        if (receipt.get('status')!='completed' or receipt['method']!=job['method'] or
                receipt['initial_steps']!=job['initial_steps'] or
                receipt['newly_collected_transitions']!=plan['new_training_target_each']):
            raise ValueError('Require complete equal NEW budgets, not a prefix')
        return 'completed'
    status=run/'status.json'
    if status.exists() and json.loads(status.read_text()).get('state')=='engineering_interruption':
        raise RuntimeError(f'{job["label"]} interrupted; preserve prefix, no automatic retry/evaluation')
    manifest=run/'run_config.json'
    if manifest.exists():
        value=json.loads(manifest.read_text())
        if native_live(value['native_pid'],str(run)):
            return 'training_native_handle_live'
        # A receipt can have been atomically published between the two reads.
        if path.exists():
            return training_state(root,plan,job)
        raise RuntimeError(f'{job["label"]} native handle missing without completion receipt')
    queue=json.loads((root/plan['queue_state']).read_text())
    if native_live(queue['pid'],plan['queue_expected_command']):
        return 'waiting_for_finite_queue_admission'
    raise RuntimeError(f'{job["label"]} has not started and its real queue handle is absent')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    root=Path(plan['root']).resolve()
    if args.output.exists() or root not in args.output.resolve().parents:
        raise ValueError('Use a new registered data-disk output')
    if plan['device']!='cpu' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU-only validation must not compete with the training GPU')
    if plan['independent_test_executed'] or plan['new_training_transitions']:
        raise ValueError('This admission does not authorize final test or RL')
    if os.stat(root).st_dev==os.stat('/home/carla').st_dev:
        raise ValueError('Outputs must be on the separate4T filesystem')
    if shutil.disk_usage(root).free<plan['storage_headroom_bytes']:
        raise RuntimeError('Require40GiB data-disk headroom')
    args.output.mkdir(parents=True)
    started=time.monotonic()
    state=dict(state='waiting_for_full_stage1_training',native_pid=os.getpid(),
        source_commit=args.source_commit,plan_sha256=sha256(args.plan),new_training_transitions=0,
        independent_test_executed=False,automatic_restart=False,completed_candidates=0)
    save(args.output/'plan.json',plan|dict(source_commit=args.source_commit,plan_sha256=state['plan_sha256']))
    def status():
        state['elapsed_seconds']=time.monotonic()-started
        save(args.output/'status.json',state)
    try:
        while True:
            state['training']={j['label']:training_state(root,plan,j) for j in plan['jobs']}
            status()
            if all(v=='completed' for v in state['training'].values()):
                break
            time.sleep(plan['poll_interval_seconds'])
        # No heavy imports or evaluator use until all four native budgets finish.
        import torch
        from scripts.paper_crax_payload.payload_train import FORMAT,evaluate_payload
        from scripts.paper_crax_payload.payload_models import build_payload_learner
        from scripts.paper_crax_payload.piper_reach_release import choose
        torch.set_num_threads(1)
        candidates=[];rows=[];episodes=[];controls=0
        for job in plan['jobs']:
            run=root/'runs'/job['run'];manifest=json.loads((run/'run_config.json').read_text())
            config=manifest['configuration']
            conditions=[c for c in config['evaluation']['conditions'] if c['name'] in plan['conditions']]
            if [c['name'] for c in conditions]!=plan['conditions']:
                raise ValueError('Use exactly the four prospectively registered conditions, no legacy route claim')
            for new_steps in plan['new_candidate_steps']:
                full_steps=job['initial_steps']+new_steps
                path=run/f'full_{full_steps}.pt'
                value=sha256(path)
                saved=torch.load(path,map_location='cpu',weights_only=False)
                if (saved.get('format')!=FORMAT or saved['configuration']!=config or
                        saved['method']!=job['method'] or saved['training_seed']!=plan['training_seed'] or
                        saved['steps']!=full_steps):
                    raise ValueError('Native saved candidate identity mismatch')
                learner=build_payload_learner(config,job['method'],'cpu')
                learner.load_state_dict(saved['learner'])
                before=(learner.updates,learner.reward_updates,learner.actor_updates,learner.safety_multiplier)
                state.update(state='common_validation',label=job['label'],new_steps=new_steps)
                status()
                result=evaluate_payload(learner,config,plan['validation_episodes_per_condition'],
                    seed_start=plan['validation_layout_seed_start'],conditions=conditions)
                if before!=(learner.updates,learner.reward_updates,learner.actor_updates,learner.safety_multiplier):
                    raise ValueError('A frozen evaluation changed training counters')
                counts=manifest['initial_counts']
                candidate=dict(label=job['label'],method=job['method'],steps=new_steps,
                    full_checkpoint_steps=full_steps,checkpoint=str(path),full_checkpoint_sha256=value,
                    training_seed=plan['training_seed'],new_training_violations=saved['violations']-counts['violations'],
                    new_training_safe_successes=saved['safe_successes']-counts['safe_successes'],
                    score=dict(risk_qualified=all(c['violation']<=limit for c,limit in zip(result['conditions'],plan['risk_limits'])),
                        task_progress_observed=any(c['safe_success']>0 for c in result['conditions']),
                        macro_safe_success=sum(c['safe_success'] for c in result['conditions'])/len(conditions),
                        macro_capped_seconds=sum(c['time_to_goal_capped_seconds'] for c in result['conditions'])/len(conditions),
                        per_condition_violation=[c['violation'] for c in result['conditions']],
                        per_condition_safe_success=[c['safe_success'] for c in result['conditions']]))
                result.update(candidate=candidate,split='validation',source_commit=args.source_commit)
                save(args.output/f'validation-{job["label"]}-{new_steps}.json',result)
                candidates.append(candidate);controls+=result['physical_evaluation_transitions']
                for condition in result['conditions']:
                    base=dict(label=job['label'],method=job['method'],new_steps=new_steps,
                        full_checkpoint_steps=full_steps,condition=condition['condition'])
                    rows.append(base|{k:condition[k] for k in ('episodes','safe_success','violation',
                        'time_to_goal_capped_seconds','minimum_distance_m')})
                    episodes.extend(base|{k:v for k,v in r.items() if k!='trajectory_payload_xyz'} for r in condition['records'])
                state['completed_candidates']+=1;status()
                del learner,saved
        selected={}
        for job in plan['jobs']:
            winner,label=choose([c for c in candidates if c['label']==job['label']])
            selected[job['label']]=dict(candidate=winner,status=label,candidate_count=5)
        selection=dict(selected=selected,candidates=candidates,independent_test_used=False,
            next_phase=plan['next_phase'],source_commit=args.source_commit)
        save(args.output/'selection.json',selection)
        for name,records in (('validation_summary.csv',rows),('validation_episodes.csv',episodes)):
            with (args.output/name).open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        save(args.output/'receipt.json',dict(status='stage1_validation_completed_pending_review',
            source_commit=args.source_commit,plan_sha256=state['plan_sha256'],selection_sha256=sha256(args.output/'selection.json'),
            validation_candidates=len(candidates),validation_episodes=len(episodes),
            native_validation_controls=controls,new_training_transitions=0,
            stage1_training_receipts_each=100000,failed_engineering_prefix_transitions=325,
            total_E01_training_including_failure=400325,independent_test_executed=False,
            physical_robot_interactions=0,conditional_second_stage_launched=False))
        state['state']='stage1_validation_completed_pending_review';status()
    except BaseException as error:
        state.update(state='needs_inspection_no_automatic_retry',error_type=type(error).__name__,error=str(error))
        status();raise


if __name__=='__main__':
    main()

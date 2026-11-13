"""Wait for the exact matched Payload continuations, then one frozen closure.

Five prospectively fixed candidates/method, common validation, freeze BOTH
identities, then independent layouts. CPU only, zero RL, no retries/sweeps.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--assets-receipt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve every admitted closure; never auto-retry')
    plan=json.loads(args.plan.read_text())
    if plan['checkpoint_steps']!=[60000,120000,180000,240000,300000] or plan['methods']!=['Iso','EA']:
        raise ValueError('Require exactly the prospectively fixed matched candidate opportunities')
    args.output.mkdir(parents=True)
    plan_sha=hashlib.sha256(args.plan.read_bytes()).hexdigest()
    save(args.output/'plan.json',plan|dict(plan_sha256=plan_sha,controller_pid=os.getpid(),
        admitted_unix=time.time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    while True:
        complete=True
        for method in plan['methods']:
            run=args.root/'runs'/f'transfer-{method.lower()}-300k'
            receipt_path=run/'receipt.json'
            if not receipt_path.exists():
                complete=False
                controller=json.loads((args.root/'continuation_status.json').read_text())
                job=next((j for j in controller['jobs'] if j['method']==method),None)
                if job and (job['status']=='failed_no_automatic_restart' or not Path('/proc',str(job['pid'])).exists()):
                    save(args.output/'status.json',dict(state='parent_missing_or_failed_needs_inspection',method=method,controller_pid=os.getpid()))
                    return
                continue
            receipt=json.loads(receipt_path.read_text())
            if (receipt.get('status')!='completed' or receipt['training_steps']!=300000
                    or receipt['newly_collected_transitions']!=200000):
                raise ValueError('Require the full declared matched continuation budget')
        if complete: break
        save(args.output/'status.json',dict(state='waiting_for_matched_Payload_endpoints',controller_pid=os.getpid(),
            check_interval_seconds=1800,training_transitions=0))
        time.sleep(1800)
    # Heavy imports occur only after the primary training phase has completed.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from scripts.paper_crax_payload.piper_train import load_frozen,evaluate_piper
    from scripts.paper_crax_payload.piper_reach_release import choose,summarize
    torch.set_num_threads(1)
    rows=[];candidates=[];controls=substeps=queries=0
    for method in plan['methods']:
        config_path=args.root/'configs'/f'transfer-{method.lower()}-100k.json'
        for step in plan['checkpoint_steps']:
            run=args.root/'runs'/f'transfer-{method.lower()}-{100 if step==60000 else 300}k'
            checkpoint=run/f'full_{step}.pt'
            config,assets,saved,learner=load_frozen(SimpleNamespace(config=config_path,
                assets_receipt=args.assets_receipt,checkpoint=checkpoint,device='cpu'))
            if saved['task']!='Payload' or saved['method']!=method or saved['steps']!=step:
                raise ValueError('Frozen native Payload identity mismatch')
            save(args.output/'status.json',dict(state='common_validation',method=method,checkpoint_steps=step,
                completed_controls=controls,controller_pid=os.getpid()))
            result=evaluate_piper(learner,config,assets,'Payload',50,61027)
            result.update(split='validation',method=method,checkpoint_steps=step,training_seed=41027)
            name=f'validation-{method.lower()}-{step}.json'
            save(args.output/name,result)
            controls+=result['new_simulation_control_intervals'];substeps+=result['new_physics_substeps'];queries+=result['distance_queries']
            value=hashlib.sha256()
            with checkpoint.open('rb') as stream:
                for block in iter(lambda:stream.read(4*1024*1024),b''): value.update(block)
            candidates.append(dict(method=method,steps=step,checkpoint=str(checkpoint),config=str(config_path),
                full_checkpoint_sha256=value.hexdigest(),score=summarize(result,plan['risk_limit_each_condition'])))
            for condition in result['conditions']:
                rows.append(dict(method=method,steps=step,split='validation',
                    **{k:condition[k] for k in ('condition','episodes','safe_success','violation','time_to_goal_capped_seconds')}))
            del saved,learner
    selected={}
    for method in plan['methods']:
        winner,status=choose([c for c in candidates if c['method']==method])
        selected[method]=dict(candidate=winner,selection_status=status,candidate_count=5,
            physical_deployment_eligible=False)
    freeze=dict(methods=selected,candidates=candidates,plan_sha256=plan_sha,
        independent_test_used_for_selection=False,fixed300K_endpoints_not_replaced=True,
        shared_Reach_parent_steps=300000,new_Payload_transitions_each=300000)
    save(args.output/'selection.json',freeze)
    freeze_sha=hashlib.sha256((args.output/'selection.json').read_bytes()).hexdigest()
    for method in plan['methods']:
        candidate=selected[method]['candidate']
        save(args.output/'status.json',dict(state='independent_test',method=method,completed_controls=controls,
            controller_pid=os.getpid(),selection_sha256=freeze_sha))
        checkpoint=Path(candidate['checkpoint'])
        value=hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''): value.update(block)
        if value.hexdigest()!=candidate['full_checkpoint_sha256']:raise ValueError('Selected full checkpoint changed')
        config,assets,saved,learner=load_frozen(SimpleNamespace(config=Path(candidate['config']),
            assets_receipt=args.assets_receipt,checkpoint=checkpoint,device='cpu'))
        result=evaluate_piper(learner,config,assets,'Payload',100,71027)
        result.update(split='independent_test',method=method,checkpoint_steps=candidate['steps'],training_seed=41027,
            selection_sha256=freeze_sha,no_policy_reselection=True)
        save(args.output/f'independent-{method.lower()}.json',result)
        controls+=result['new_simulation_control_intervals'];substeps+=result['new_physics_substeps'];queries+=result['distance_queries']
        for condition in result['conditions']:
            rows.append(dict(method=method,steps=candidate['steps'],split='independent_test',
                **{k:condition[k] for k in ('condition','episodes','safe_success','violation','time_to_goal_capped_seconds')}))
        del saved,learner
    if hashlib.sha256((args.output/'selection.json').read_bytes()).hexdigest()!=freeze_sha:
        raise ValueError('Selection changed after independent labels')
    with (args.output/'results.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(args.output/'receipt.json',dict(status='completed',common_validation_episodes=1500,
        independent_test_episodes=600,selection_sha256=freeze_sha,selection_frozen_before_test=True,
        native_simulation_controls=controls,native_physics_substeps=substeps,distance_queries=queries,
        training_transitions=0,physical_robot_interactions=0,fixed300K_endpoints_preserved=True))
    save(args.output/'status.json',dict(state='completed',completed_controls=controls,controller_pid=os.getpid()))


if __name__=='__main__':
    main()

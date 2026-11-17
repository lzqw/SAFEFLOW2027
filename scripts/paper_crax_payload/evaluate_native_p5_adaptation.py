"""Once-only CPU evaluation of a separately reviewed global model freeze.

This module cannot select models, train, or automatically retry a run. The
freeze must contain checkpoint identities and a completed development-review
identity; evaluator writes an exclusive registration before touching physics.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from scripts.paper_crax_payload.native_p5_adaptation import (
    FORMAT, build_native, attach_adaptation, target_deployment)
from scripts.paper_crax_payload.piper_train import make_piper_env, PiperFlowLearner
from scripts.paper_crax_payload.payload_environment import PayloadConfig, SafePayloadTransport
from scripts.paper_crax_payload.train_native_p5_adaptation import sha256, storage_check
from scripts.star.arm_safe_entropy.train import atomic_json


def load_frozen_case(case, plan):
    path=Path(case['checkpoint'])
    if sha256(path) != case['checkpoint_sha256']:
        raise ValueError('Frozen full-checkpoint identity changed')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    line=plan['lines'][case['line']]
    training_plan = plan
    if case.get('training_plan_path'):
        training_path = Path(case['training_plan_path'])
        if sha256(training_path) != case.get('training_plan_sha256'):
            raise ValueError('Optional candidate training-plan identity changed')
        training_plan = json.loads(training_path.read_text())
        if training_plan.get('optional_candidate') != 'piper_payload_radius_cover':
            raise ValueError('Only the registered Piper Payload coverage candidate is admitted')
        if training_plan['lines'][case['line']]['parent_sha256'] != line['parent_sha256']:
            raise ValueError('Optional candidate parent differs from the evaluation line')
    if case.get('historical_reference'):
        if (case['line']!='piper_payload' or saved['format']!='paper_native_piper_sim_v1_full'
                or saved['method']!='Iso' or saved['steps']!=60000 or saved['training_seed']!=41027
                or saved['robot_family']!='Piper' or saved['task']!='Payload'
                or case['checkpoint_sha256']!='c69d8daa58bcef7800872987f83e1a64b43b97e65e1e0b8b67fabd04588359d2'):
            raise ValueError('Use only the previously frozen native Iso speed reference')
        learner=PiperFlowLearner(saved['configuration'],'Iso','cpu')
    else:
        if (saved['format']!=FORMAT or saved['line']!=case['line'] or saved['arm']!=case['arm']
                or saved['steps']!=line['parent_steps']+100000
                or saved['learner']['native_p5_adaptation_v11']['parent_sha256']!=line['parent_sha256']):
            raise ValueError('Current evaluator admits only fixed new100K native v11 endpoints')
        if case.get('training_plan_path') and saved.get('plan') != training_plan:
            raise ValueError('Checkpoint does not carry the registered optional training plan')
        learner=build_native(saved['configuration'],training_plan['lines'][case['line']],'cpu')
        learner.set_transitions(line['parent_steps'])
        attach_adaptation(learner,training_plan,case['line'],case['arm'])
    learner.load_state_dict(saved['learner'])
    if learner.settings['target_polyak']!=.001:
        raise ValueError('Common target deployment rule differs')
    return learner,saved['configuration'],saved


def evaluate_case(case, plan, assets, episodes):
    learner,native,saved=load_frozen_case(case,plan)
    line=plan['lines'][case['line']]
    saved_source_sha = saved.get('model_source_urdf_sha256',
        saved['configuration'].get('model_source_urdf_sha256'))
    saved_model_commit = saved.get('model_commit', saved['configuration'].get('model_commit'))
    if line['robot']=='Piper' and (
            (saved_source_sha is not None and saved_source_sha != assets['source_sha256'])
            or saved_model_commit != assets['model_commit']):
        raise ValueError('Native frozen model and pinned official assets differ')
    del saved
    records,work=[],[]
    with target_deployment(learner):
        for ci,condition in enumerate(line['conditions']):
            env=(make_piper_env(native,assets,line['task'],condition.get('simulation_overrides'))
                if line['robot']=='Piper' else
                SafePayloadTransport(PayloadConfig(**(dict(native['task'])|condition.get('task_overrides',{})))))
            try:
                for episode in range(episodes):
                    seed=line['independent_layout_seed_start']+episode
                    action_seed=line['independent_action_seed']+ci*1000+episode
                    torch.manual_seed(action_seed)
                    try:
                        obs,initial=env.reset(seed=seed)
                    except RuntimeError as error:
                        # Only the bounded native static reset refusal is a
                        # nonadmitted fixture, not arbitrary engine errors.
                        allowed=('No safe reachable Piper start/goal after bounded static rejection',
                                 'Cannot obtain safe reachable arm AND payload endpoints')
                        if str(error) not in allowed:
                            raise
                        records.append(dict(case=case['name'],line=case['line'],condition=condition['name'],
                            layout_seed=seed,action_seed=action_seed,admitted=False,reason=str(error)))
                        continue
                    position=(env.task_position if line['robot']=='Piper' else env.payload_position)
                    trajectory=[position().tolist()]
                    minimum=initial['minimum_distance'];total=0.
                    for step in range(env.config.horizon):
                        action=np.clip(learner.action(obs[None])[0]*condition.get('action_scale',1.),-1.,1.)
                        obs,reward,done,truncated,info=env.step(action)
                        trajectory.append(position().tolist());minimum=min(minimum,info['minimum_distance']);total+=reward
                        if done or truncated:
                            break
                    length=step+1
                    records.append(dict(case=case['name'],line=case['line'],condition=condition['name'],
                        layout_seed=seed,action_seed=action_seed,admitted=True,safe_success=bool(info['safe_success']),
                        violation=bool(info['violation']),safe_timeout=bool(not info['safe_success'] and not info['violation']),
                        true_horizon_terminal=bool(info['true_horizon_terminal']),collision=bool(info['collision']),
                        arm_collision=bool(info['arm_collision']),payload_collision=bool(info['payload_collision']),
                        episode_return=total,actual_length=length,full_interval_minimum_distance_m=minimum,
                        time_to_goal_capped_seconds=(length if info['safe_success'] else env.config.horizon)*env.control_dt,
                        task_path_length_m=float(np.linalg.norm(np.diff(np.asarray(trajectory),axis=0),axis=1).sum()),
                        trajectory_task_xyz=trajectory))
            finally:
                work.append(dict(condition=condition['name'],simulation_controls=sum(r.get('actual_length',0)
                    for r in records if r['condition']==condition['name']),
                    physics_substeps_per_control=env.config.substeps))
                env.close()
    return dict(case=case,records=records,work=work,new_training_steps=0,physical_robot_interactions=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--line',choices=['piper_reach','piper_payload','panda_payload'],required=True)
    p.add_argument('--assets-receipt',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    args=p.parse_args();torch.set_num_threads(1)
    plan=json.loads(args.plan.read_text());freeze=json.loads(args.freeze.read_text())
    allowed_protocols = {
        'v11_once_only_frozen_native_task_independent',
        'v11_once_only_frozen_native_task_independent_amended_v1',
        'v11_once_only_frozen_native_task_independent_amended_v2',
        'v11_historical_speed_reference_v1'}
    historical_only = freeze['protocol'] == 'v11_historical_speed_reference_v1'
    if (freeze['protocol'] not in allowed_protocols
            or not freeze['development_review_completed'] or freeze['plan_sha256']!=sha256(args.plan)
            or freeze['episodes_per_condition']!=100 or freeze['selection_after_independent']):
        raise ValueError('Require a completed global DEV review and exact model/plan freeze')
    if historical_only:
        if (not freeze.get('historical_only') or len(freeze['cases']) != 1
                or not freeze['cases'][0].get('historical_reference')
                or freeze['cases'][0]['line'] != 'piper_payload'):
            raise ValueError('Historical speed reference requires one registered Piper Iso case')
    if freeze['protocol'].endswith('_amended_v1'):
        amendment = args.freeze.parent/freeze['engineering_amendment_path']
        if sha256(amendment) != freeze['engineering_amendment_sha256']:
            raise ValueError('The engineering amendment changed')
    review=Path(freeze['development_review_path'])
    if not review.is_absolute():
        review=args.freeze.parent/review
    if sha256(review)!=freeze['development_review_sha256']:
        raise ValueError('The frozen pre-test review changed')
    cases=[c for c in freeze['cases'] if c['line']==args.line]
    if not cases or len({c['name'] for c in cases})!=len(cases):
        raise ValueError('Frozen case list missing or duplicated')
    if not historical_only:
        for line_name in plan['lines']:
            pair=[c for c in freeze['cases'] if c['line']==line_name and not c.get('historical_reference')]
            if len(pair)!=2 or {c['arm'] for c in pair}!=set(plan['arms']):
                raise ValueError('Global freeze must retain one complete paired comparison per line')
    for case in cases:
        if not case.get('historical_reference'):
            receipt_path=Path(case['training_receipt'])
            if sha256(receipt_path)!=case['training_receipt_sha256']:
                raise ValueError('Native full-budget training receipt changed')
            receipt=json.loads(receipt_path.read_text())
            if receipt['status']!='completed' or receipt['new_training_steps']!=100000:
                raise ValueError('Do not open independent testing on unfinished training')
    root=args.data_root.resolve(strict=True)
    if not args.output.resolve().is_relative_to(root):
        raise ValueError('Keep artifacts on the admitted data disk')
    storage_check(root)
    assets=json.loads(args.assets_receipt.read_text()) if args.assets_receipt else None
    if assets and (sha256(assets['source_urdf'])!=assets['source_sha256']
                   or sha256(assets['resolved_urdf'])!=assets['resolved_sha256']):
        raise ValueError('Official simulation assets changed')
    args.output.mkdir(parents=True,exist_ok=False)
    # A different output path must not silently allow a second pass over the
    # same frozen independent pool. Failures preserve this consumed marker.
    with (args.freeze.parent/('independent_consumed_'+args.line+'.json')).open('x') as marker:
        marker.write(json.dumps(dict(freeze_sha256=sha256(args.freeze),output=str(args.output),
            native_pid=os.getpid(),automatic_retry=False))+'\n')
    atomic_json(args.output/'registration.json',dict(protocol=freeze['protocol'],freeze_sha256=sha256(args.freeze),
        native_pid=os.getpid(),host=platform.node(),source_commit=args.source_commit,
        line=args.line,cases=cases,new_training_steps=0,automatic_retry=False))
    started=time.monotonic()
    for case in cases:
        result=evaluate_case(case,plan,assets,100)
        atomic_json(args.output/(case['name']+'.json'),result)
    atomic_json(args.output/'receipt.json',dict(status='completed',line=args.line,cases=len(cases),
        episodes=len(plan['lines'][args.line]['conditions'])*100*len(cases),
        new_training_steps=0,physical_robot_interactions=0,elapsed_seconds=time.monotonic()-started,
        freeze_sha256=sha256(args.freeze)))


if __name__=='__main__':
    main()

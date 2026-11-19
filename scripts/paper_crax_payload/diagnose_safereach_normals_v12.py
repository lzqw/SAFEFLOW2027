"""Bounded same-state finite differences; never used to select policy actions.

Retrospective native25D replay, all38 unsafe controls plus90 fixed safe controls.
Reconstruction must reproduce every original margin before any derivative probe.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from scripts.paper_crax_payload.diagnose_safereach_calibration_v12 import (
    PLAN, PROJECT, ROOT, replay_window, sha256, write_csv)
from scripts.star.arm_safe_entropy.geometry import LocalMarginEnsemble


SPEC = dict(window=[240001,250000],unsafe_controls=38,near_safe_each_pool=10,
    interior_safe_each_pool=20,selection_seed=993100,samples=128,
    action_epsilon=.002,original_margin_tolerance_m=2e-6,
    replay_repeat_tolerance_m=2e-7,observation_tolerance=2e-6,
    valid_gradient_norm=1e-7,near_true_boundary_m=.005)


def select_pool(inputs):
    selected = []
    rng = np.random.default_rng(SPEC['selection_seed'])
    for label in ('P5','Iso','Metric'):
        data = inputs[label]; margin = data['margin']
        indices = list(np.flatnonzero(margin>0.))
        for pool,count in ((np.flatnonzero((margin<=0.) & (margin>=-.005)),10),
                           (np.flatnonzero(margin<-.005),20)):
            indices.extend(sorted(rng.choice(pool,size=count,replace=False).tolist()))
        for index in indices:
            selected.append(dict(behavior_pool=label,original_transition=240001+int(index),
                obs=data['obs'][index].tolist(),action=data['action'][index].tolist(),
                original_margin_m=float(margin[index])))
    if (len(selected)!=128 or sum(r['original_margin_m']>0 for r in selected)!=38
            or len({(r['behavior_pool'],r['original_transition']) for r in selected})!=128):
        raise ValueError('Preserve all38 unsafe and exactly90 predeclared safe actual controls')
    return selected


def reconstruct(env, observation):
    obs = np.asarray(observation,dtype=float)
    if obs.shape!=(25,) or abs(obs[23]-.06)>1e-6:
        raise ValueError('Native25D nominal training state only;never relabel radius')
    env.goal=obs[17:20].copy();env.obstacle_position=obs[20:23].copy()
    # Start is not consumed by the next native control. Do not invent its old value.
    env.sim.set_base_pose('target',env.goal,np.array([0.,0.,0.,1.]))
    env.sim.set_base_pose('obstacle',env.obstacle_position,np.array([0.,0.,0.,1.]))
    positions=np.r_[obs[:7]*np.pi,[0.,0.]]
    velocities=np.r_[obs[7:14]*4.,[0.,0.]]
    for joint,q,dq in zip(list(range(7))+[9,10],positions,velocities):
        env.client.resetJointState(env.body,joint,float(q),targetVelocity=float(dq))
    env.robot.control_joints(positions);env.client.performCollisionDetection()
    env._elapsed=int(round((1.-obs[-1])*100));env._done=False
    env._previous_distance=float(np.linalg.norm(env.ee_position()-env.goal))
    error=float(np.max(np.abs(env._obs()-obs)))
    if error>SPEC['observation_tolerance']:
        raise ValueError('Reconstructed native observation differs;do not proceed to normals')
    return error


def alignment(predicted,actual):
    predicted,actual=np.asarray(predicted),np.asarray(actual)
    pn,an=float(np.linalg.norm(predicted)),float(np.linalg.norm(actual))
    valid=pn>SPEC['valid_gradient_norm'] and an>SPEC['valid_gradient_norm']
    cosine=float(np.clip(np.dot(predicted,actual)/(pn*an),-1.,1.)) if valid else None
    return dict(predicted_gradient_norm=pn,true_gradient_norm=an,gradient_pair_valid=valid,
        normal_signed_cosine=cosine,normal_axis_cosine_squared=cosine*cosine if valid else None)


def main():
    from scripts.star.arm_safe_entropy.environment import TaskConfig,make_environment
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-commit',required=True)
    args=p.parse_args();torch.set_num_threads(1)
    if (not args.output.resolve().is_relative_to(ROOT.resolve(strict=True))
            or shutil.disk_usage(ROOT).free<41*1024**3):
        raise ValueError('Use admitted data disk;preserve40GiB plus1GiB diagnostic allowance')
    args.output.mkdir(exist_ok=False)
    def save(name,value):
        (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    plan=json.loads(PLAN.read_text());cases=[c for c in plan['cases'] if c['line']=='safereach']
    save('registration.json',dict(protocol='v12_same_saved_state_finite_difference_margin_normals',
        source_commit=args.source_commit,specification=SPEC,cases=cases,
        scope='retrospective_native_training_states;not_current_candidate_or_independent_test',
        geometry_site='direct_actual_terminal_action;not sampler_tail_pullback',
        physics_budget_controls=1024,new_training_steps=0,policy_actions_generated=0,
        optimizer_steps=0,oracle_action_selection=False,automatic_retry=False))
    inputs,models,provenance={},{},[]
    for case in cases:
        directory=ROOT/'panda-safereach-mode-inputs'/case['relative_run']
        checkpoint=directory/case['checkpoint_file']
        if sha256(checkpoint)!=case['checkpoint_sha256']:
            raise ValueError('Use only frozen registered final native models')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        config=json.loads((PROJECT/case['plan']).read_text())
        arm=next(a for a in config['arms'] if a['name']==case['arm'])
        receipt=json.loads((directory/'receipt.json').read_text())
        if (saved['format']!=case['format'] or saved['configuration']!=config or saved['arm']!=arm
                or saved['seed']!=2034 or saved['steps']!=250000
                or not saved['source_commit'].startswith(case['training_source'])
                or receipt['status']!='completed' or receipt['engineering_only']
                or receipt['steps']!=250000 or receipt['source_commit']!=saved['source_commit']):
            raise ValueError('Preserve registered scientific native identities and completed budgets')
        inputs[case['label']]=replay_window(saved['replay'])
        model=LocalMarginEnsemble().requires_grad_(False).eval()
        model.load_state_dict(saved['learner']['ea_stability_v2']['margin_geometry'],strict=True)
        models[case['label']]=model
        provenance.append(dict(label=case['label'],checkpoint_sha256=case['checkpoint_sha256'],
            receipt_sha256=sha256(directory/'receipt.json')))
        del saved
    selected=select_pool(inputs);save('selected_actual_actions.json',selected)
    env=make_environment(TaskConfig(version='SafeReach-RelativeLayout-v3'))
    controls=0;truth=[]
    try:
        env.reset(seed=993200)
        # Check every predeclared state before a single finite-difference action.
        for row in selected:
            obs_error=reconstruct(env,row['obs'])
            observed=float(env.oracle_margin(row['action']));controls+=1
            truth.append(dict(behavior_pool=row['behavior_pool'],original_transition=row['original_transition'],
                original_margin_m=row['original_margin_m'],replayed_margin_m=observed,
                replay_error_m=observed-row['original_margin_m'],observation_max_error=obs_error))
        write_csv(args.output/'reconstruction.csv',truth)
        max_error=max(abs(r['replay_error_m']) for r in truth)
        if max_error>SPEC['original_margin_tolerance_m']:
            save('receipt.json',dict(status='reconstruction_mismatch_no_derivative_probe',
                source_commit=args.source_commit,provenance=provenance,states=128,
                max_margin_replay_error_m=max_error,tolerance_m=SPEC['original_margin_tolerance_m'],
                new_simulation_controls=controls,physics_substeps=controls*20,
                finite_difference_controls=0,reset_settling_substeps=20,
                new_training_steps=0,optimizer_steps=0,independent_test_used=False,
                dropped_states=0,normal_comparison_valid=False))
            raise RuntimeError('Original labels not reproduced;no tolerance relaxation or sample removal')
        for row,record in zip(selected,truth):
            reconstruct(env,row['obs']);action=np.asarray(row['action']);gradient=[]
            for axis in range(3):
                plus=action.copy();minus=action.copy()
                plus[axis]=min(1.,plus[axis]+SPEC['action_epsilon'])
                minus[axis]=max(-1.,minus[axis]-SPEC['action_epsilon'])
                high=env.oracle_margin(plus);low=env.oracle_margin(minus);controls+=2
                gradient.append(float((high-low)/(plus[axis]-minus[axis])))
            repeated=float(env.oracle_margin(action));controls+=1
            if abs(repeated-record['replayed_margin_m'])>SPEC['replay_repeat_tolerance_m']:
                raise ValueError('Counterfactual snapshot restore changed the original behavior label')
            record.update(true_action_gradient=gradient,true_gradient_norm=float(np.linalg.norm(gradient)),
                repeated_margin_m=repeated)
    finally:
        env.close()
    save('true_gradients.json',truth)
    obs=torch.tensor(np.asarray([r['obs'] for r in selected]),dtype=torch.float32)
    action=torch.tensor(np.asarray([r['action'] for r in selected]),dtype=torch.float32)
    records=[]
    for label,model in models.items():
        variable=action.clone().requires_grad_(True);heads=model(obs,variable)
        gradients=torch.stack([torch.autograd.grad(heads[i].sum(),variable,retain_graph=i==0)[0]
            for i in range(2)]).numpy()
        mean_gradient=gradients.mean(0)
        norms=np.linalg.norm(gradients,axis=-1)
        for index,(row,true) in enumerate(zip(selected,truth)):
            denominator=float(norms[:,index].prod())
            agreement=float(np.clip(np.dot(gradients[0,index],gradients[1,index])/denominator,0.,1.)) if denominator>1e-20 else 0.
            records.append(dict(model_policy=label,behavior_pool=row['behavior_pool'],
                original_transition=row['original_transition'],original_margin_m=row['original_margin_m'],
                predicted_margin_m=float(heads[:,index].detach().mean()),head_agreement=agreement,
                **alignment(mean_gradient[index],true['true_action_gradient'])))
    write_csv(args.output/'model_normal_comparisons.csv',records)
    summaries=[]
    for label in models:
        own=[r for r in records if r['model_policy']==label]
        for group in ('all','unsafe','near_true_boundary','safe_interior','high_head_agreement'):
            subset=[r for r in own if (group=='all'
                or group=='unsafe' and r['original_margin_m']>0
                or group=='near_true_boundary' and abs(r['original_margin_m'])<=.005
                or group=='safe_interior' and r['original_margin_m']<-.005
                or group=='high_head_agreement' and r['head_agreement']>=.8)]
            valid=[r for r in subset if r['gradient_pair_valid']]
            summaries.append(dict(model_policy=label,group=group,states=len(subset),
                true_flat_states=sum(r['true_gradient_norm']<=SPEC['valid_gradient_norm'] for r in subset),
                compared_nonflat_states=len(valid),opposite_normals=sum(r['normal_signed_cosine']<0 for r in valid),
                signed_cosine_mean=float(np.mean([r['normal_signed_cosine'] for r in valid])) if valid else None,
                axis_cosine_squared_mean=float(np.mean([r['normal_axis_cosine_squared'] for r in valid])) if valid else None,
                margin_mae_mm=float(np.mean([abs(r['predicted_margin_m']-r['original_margin_m']) for r in subset])*1000) if subset else None))
    write_csv(args.output/'summary.csv',summaries)
    save('receipt.json',dict(status='completed_same_state_finite_difference_diagnostic',source_commit=args.source_commit,
        provenance=provenance,states=128,unsafe_states=38,models=3,new_simulation_controls=controls,
        physics_substeps=controls*20,reset_settling_substeps=20,max_margin_replay_error_m=max_error,
        new_training_steps=0,policy_actions_generated=0,optimizer_steps=0,independent_test_used=False,
        dropped_states=0,causal_training_intervention=False,sampler_pullback_evaluated=False,
        true_flat_states=sum(r['true_gradient_norm']<=SPEC['valid_gradient_norm'] for r in truth)))


if __name__=='__main__':
    main()

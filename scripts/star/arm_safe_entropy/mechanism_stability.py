"""Bounded final EA-STAB mechanism analysis,not training or primary evaluation.

Requires four completed native1M runs (Iso + selected geometry,two seeds).
The counterfactual Local/direct/Pullback samplers share the frozen generator
and fixed q=.5 spectral rule;they are NOT three newly trained methods.
Oracle labels/directions are only queried after actions have been sampled.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.star.arm_safe_entropy.conditional_metrics import endpoint_metrics,reference_metrics
from scripts.star.arm_safe_entropy.diagnose_endpoint_geometry import FixedSpectrumGeometry,mean_or_none
from scripts.star.arm_safe_entropy.endpoint_geometry import EndpointGeometry
from scripts.star.arm_safe_entropy.environment import TaskConfig
from scripts.star.arm_safe_entropy.evaluate_obstacle_robustness import PerturbedRelativeReach,atomic_json
from scripts.star.arm_safe_entropy.oracle import SimulatorOracle
from scripts.star.arm_safe_entropy.stability_runtime import build_learner


MODES=('local','endpoint_direct','endpoint_pullback')
CONDITIONS=(('nominal',.06),('radius_080',.08))


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def load_final_runs(paths,device):
    policies=[]
    for path in paths:
        config=json.loads((path/'run_config.json').read_text())
        receipt=json.loads((path/'receipt.json').read_text())
        if (config['stage']!='ea_stability_v2_development' or receipt['status']!='completed'
                or receipt['steps']!=1000000 or config['total_transition_budget']!=1000000):
            raise ValueError('Use every completed fixed1M development policy,not a best intermediate checkpoint')
        learner=build_learner(config['configuration'],config['variant'],device)
        learner.load_state_dict(torch.load(path/'policy_1000000.pt',map_location=device,weights_only=False))
        learner.actor.requires_grad_(False)
        learner.margin_geometry.requires_grad_(False)
        policies.append(dict(path=path,config=config,learner=learner))
    variants={p['config']['variant'] for p in policies}
    if (len(policies)!=4 or 's_iso' not in variants or len(variants)!=2 or
            {(p['config']['variant'],p['config']['training_seed']) for p in policies}!=
            {(v,s) for v in variants for s in (2027,2028)}):
        raise ValueError('Mechanism input must retain both development seeds of Iso and the selected geometry')
    if any(p['config']['configuration']!=policies[0]['config']['configuration'] for p in policies):
        raise ValueError('Use the frozen shared package,not unmatched training configurations')
    return policies


def sample(learner,observations,geometry,seed=7290027):
    torch.manual_seed(seed)
    with torch.no_grad():
        return learner.actor(observations,geometry=geometry,return_reference=True)


def collect_pool(policies,device):
    pool=[];steps=episodes=0;seen=set()
    rng=np.random.default_rng(6190027)
    for policy in policies:
        learner=policy['learner'];config=policy['config']
        for condition,radius in CONDITIONS:
            env=PerturbedRelativeReach(TaskConfig(version='SafeReach-RelativeLayout-v3',obstacle_radius=radius))
            candidates=[];candidate_keys=set()
            try:
                for episode in range(8):
                    layout_seed=160027+episode
                    obs,_=env.reset_condition(layout_seed,0.)
                    torch.manual_seed(6910027+config['training_seed']*1000+episode)
                    while True:
                        with torch.no_grad():
                            action=learner.actor(torch.as_tensor(obs[None],device=device),geometry=learner.geometry)[0][0].cpu().numpy()
                        following,_,done,truncated,info=env.step(action)
                        steps+=1;key=obs.tobytes()
                        if key not in seen and key not in candidate_keys:
                            candidates.append(dict(observation=obs.tolist(),source_action=action.tolist(),
                                source_interval_margin=info['local_margin'],condition=condition,layout_seed=layout_seed,
                                source_variant=config['variant'],source_seed=config['training_seed']))
                            candidate_keys.add(key)
                        obs=following
                        if done or truncated: break
                    episodes+=1
                    if episode>=3 and len(candidates)>=8: break
            finally: env.close()
            if len(candidates)<8: raise ValueError('Do not duplicate states to fill the mechanism pool')
            near=[i for i,r in enumerate(candidates) if abs(r['source_interval_margin'])<=.02]
            chosen=list(rng.choice(near,size=min(4,len(near)),replace=False))
            remainder=[i for i in range(len(candidates)) if i not in chosen]
            chosen.extend(rng.choice(remainder,size=8-len(chosen),replace=False).tolist())
            pool.extend(candidates[i] for i in sorted(chosen))
            seen.update(np.asarray(candidates[i]['observation'],np.float32).tobytes() for i in chosen)
    if len(pool)!=64: raise ValueError('The bounded common pool must have64 distinct states')
    return pool,steps,episodes


def evaluate_samples(learner,obs,pool,truth,oracle):
    source_actions=torch.as_tensor(np.asarray([r['source_action'] for r in pool],np.float32),device=obs.device)
    source_normal,_,source_diagnostic=learner.margin_geometry.geometry(obs,source_actions,return_diagnostics=True)
    source_normal=source_normal.cpu().numpy()
    source_direction_valid=(truth[:,4]>1e-7)&source_diagnostic['gradient_valid'].cpu().numpy()
    source_cosine=(source_normal*truth[:,1:4]).sum(-1)
    source_agreement=source_diagnostic['agreement'].cpu().numpy()
    with torch.no_grad(): source_margin=learner.margin_geometry(obs,source_actions).mean(0).cpu().numpy()
    variants={}
    for mode in ('native',)+(() if learner.variant=='s_iso' else MODES):
        geometry=learner.geometry if mode=='native' else FixedSpectrumGeometry(learner.margin_geometry,mode,1000000)
        actions,_,clips,trace=sample(learner,obs.repeat_interleave(32,0),geometry)
        actions_np=actions.cpu().numpy().reshape(64,32,3)
        labels=oracle.margins_numpy(np.repeat(obs.cpu().numpy(),32,axis=0),actions_np.reshape(-1,3)).reshape(64,32)
        with torch.no_grad(): predicted=learner.margin_geometry(obs.repeat_interleave(32,0),actions).mean(0).cpu().numpy().reshape(64,32)
        variants[mode]=dict(actions=actions_np,labels=labels,predicted=predicted,clips=clips.cpu().numpy().reshape(64,32),
            trace={key:value.cpu().numpy().reshape(64,32,*value.shape[1:]) for key,value in trace.items()
                   if key in ('weights','normals','vn','vt','geometry_zero_gradient','geometry_proposal_clipped')})
    matched=np.ones((64,32),bool)
    if learner.variant!='s_iso':
        for mode in MODES: matched&=np.all(variants[mode]['trace']['weights']==variants['local']['trace']['weights'],axis=-1)
    summaries=[];per_state=[]
    for mode,value in variants.items():
        states=[]
        for index,row in enumerate(pool):
            normal=truth[index,1:4]
            mask=matched[index] if mode!='native' else np.ones(32,bool)
            states.append(dict(state_index=index,condition=row['condition'],true_normal_valid=bool(truth[index,4]>1e-7),
                source_true_margin_m=float(truth[index,0]),source_predicted_margin_m=float(source_margin[index]),
                source_direction_valid=bool(source_direction_valid[index]),
                source_normal_cosine=float(source_cosine[index]) if source_direction_valid[index] else None,
                source_head_agreement=float(source_agreement[index]),
                endpoint=endpoint_metrics(value['actions'][index],normal),
                reference=reference_metrics(value['trace']['normals'][index],value['trace']['vn'][index],value['trace']['vt'][index],normal),
                matched_endpoint=endpoint_metrics(value['actions'][index,mask],normal) if mask.sum()>=2 else None,
                matched_draws=int(mask.sum()),true_interval_risk=float(np.mean(value['labels'][index]>0))))
        for condition,_ in CONDITIONS:
            indices=[i for i,r in enumerate(pool) if r['condition']==condition]
            labels=value['labels'][indices];predicted=value['predicted'][indices]
            near=np.abs(labels)<=.02;safe=predicted<0
            source_near=np.abs(truth[indices,0])<=.02
            source_safe=source_margin[indices]<0
            source_valid_indices=[i for i in indices if source_direction_valid[i]]
            valid=[s for s in states if s['condition']==condition and s['true_normal_valid']]
            same=[s for s in valid if s['matched_endpoint'] is not None]
            mask=matched[indices] if mode!='native' else np.ones_like(labels,bool)
            summaries.append(dict(variant=learner.variant,mode=mode,condition=condition,
                trained_policy_checkpoint_steps=1000000,sampling_intervention=mode!='native',
                states=len(indices),draws_per_state=32,common_fixed_q=.5 if mode!='native' else None,rho=learner.actor.rho,
                total_draws=int(labels.size),matched_spectrum_draws=int(mask.sum()) if mode!='native' else None,
                true_interval_risk=float(np.mean(labels>0)),
                matched_spectrum_interval_risk=float(np.mean(labels[mask]>0)) if mode!='native' and mask.any() else None,
                valid_true_axis_states=len(valid),endpoint_vn=mean_or_none([s['endpoint']['vn'] for s in valid]),
                source_direction_valid_states=len(source_valid_indices),
                source_action_normal_cosine=mean_or_none(source_cosine[source_valid_indices]),
                source_action_head_agreement=float(source_agreement[indices].mean()),
                source_action_margin_mae_m=float(np.mean(np.abs(truth[indices,0]-source_margin[indices]))),
                source_near_boundary_states=int(source_near.sum()),
                source_near_boundary_mae_m=float(np.mean(np.abs(truth[indices,0][source_near]-source_margin[indices][source_near]))) if source_near.any() else None,
                source_false_safe_given_predicted_safe=float(np.mean(truth[indices,0][source_safe]>0)) if source_safe.any() else None,
                endpoint_vt=mean_or_none([s['endpoint']['vt'] for s in valid]),
                reference_vn=mean_or_none([s['reference']['vn'] for s in valid]),
                reference_vt=mean_or_none([s['reference']['vt'] for s in valid]),
                matched_endpoint_vn=mean_or_none([s['matched_endpoint']['vn'] for s in same]) if mode!='native' else None,
                matched_endpoint_vt=mean_or_none([s['matched_endpoint']['vt'] for s in same]) if mode!='native' else None,
                margin_mae_m=float(np.mean(np.abs(labels-predicted))),near_boundary_draws=int(near.sum()),
                near_boundary_mae_m=float(np.mean(np.abs(labels[near]-predicted[near]))) if near.any() else None,
                false_safe_joint_fraction=float(np.mean((labels>0)&safe)),
                false_safe_given_predicted_safe=float(np.mean(labels[safe]>0)) if safe.any() else None,
                geometry_zero_gradient_fraction=float(value['trace']['geometry_zero_gradient'][indices].mean()),
                geometry_proposal_clipped_fraction=float(value['trace']['geometry_proposal_clipped'][indices].mean()),
                action_clipping_fraction=float(value['clips'][indices].mean())))
        per_state.append(dict(mode=mode,states=states))
    return summaries,per_state


def route_sector(trajectory,obstacle,goal):
    """Eight-sector successful route-side occupancy proxy,not state entropy."""
    trajectory=np.asarray(trajectory);obstacle=np.asarray(obstacle)
    axis=np.asarray(goal)-trajectory[0];axis/=np.linalg.norm(axis)
    projection=(trajectory-obstacle)@axis
    if projection.min()>0 or projection.max()<0: return None
    lateral=np.cross(axis,np.array([0.,0.,1.]));lateral/=np.linalg.norm(lateral)
    vertical=np.cross(lateral,axis)
    offset=trajectory[np.argmin(np.abs(projection))]-obstacle
    angle=np.arctan2(offset@vertical,offset@lateral)
    return int(np.floor((angle+np.pi)/(2*np.pi)*8))%8


def route_rollouts(learner,device):
    records=[];physical_steps=0
    for condition,radius in CONDITIONS:
        env=PerturbedRelativeReach(TaskConfig(version='SafeReach-RelativeLayout-v3',obstacle_radius=radius))
        try:
            for mode in MODES:
                geometry=FixedSpectrumGeometry(learner.margin_geometry,mode,1000000)
                for episode in range(10):
                    obs,layout=env.reset_condition(170027+episode,0.)
                    torch.manual_seed(8190027+episode)
                    trajectory=[obs[14:17].tolist()];episode_return=0.;length=0
                    while True:
                        with torch.no_grad(): action=learner.actor(torch.as_tensor(obs[None],device=device),geometry=geometry)[0][0].cpu().numpy()
                        obs,reward,done,truncated,info=env.step(action)
                        physical_steps+=1;length+=1;episode_return+=reward;trajectory.append(obs[14:17].tolist())
                        if done or truncated: break
                    success=bool(info['safe_success']);violation=bool(info['violation'])
                    points=np.asarray(trajectory)
                    records.append(dict(condition=condition,mode=mode,layout_seed=170027+episode,
                        safe_success=success,violation=violation,safe_timeout=not success and not violation,
                        episode_return=episode_return,length=length,time_to_goal_capped_seconds=length*.04 if success else 4.,
                        control_sampled_path_length_m=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
                        successful_route_sector=route_sector(points,layout['obstacle'],layout['goal']) if success else None))
        finally: env.close()
    return records,physical_steps


def inference_timings(learner,obs):
    results=[]
    modes={'none':None,**{mode:EndpointGeometry(learner.margin_geometry,mode,1000000) for mode in MODES}}
    for batch_size in (2,1024):
        batch=obs.repeat((batch_size+63)//64,1)[:batch_size]
        for mode,geometry in modes.items():
            for _ in range(2): sample(learner,batch,geometry)
            durations=[]
            for _ in range(5):
                if obs.device.type=='cuda': torch.cuda.synchronize()
                started=time.monotonic()
                with torch.no_grad(): learner.actor(batch,geometry=geometry)
                if obs.device.type=='cuda': torch.cuda.synchronize()
                durations.append(time.monotonic()-started)
            results.append(dict(mode=mode,batch_size=batch_size,repeats=5,
                median_seconds=float(np.median(durations)),min_seconds=min(durations),max_seconds=max(durations),
                scope='same_frozen_generator_inference_only_not_end_to_end_training_FPS'))
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',nargs=4,type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--workers',type=int,default=10)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Do not overwrite an earlier mechanism analysis')
    torch.set_num_threads(1);started=time.monotonic()
    policies=load_final_runs(args.runs,args.device)
    args.output.mkdir(parents=True)
    pool,pool_steps,pool_episodes=collect_pool(policies,args.device)
    atomic_json(args.output/'state_pool.json',dict(records=pool,physical_intervals=pool_steps,episodes=pool_episodes,
        rule='8distinct states/policy/condition,up to4near-source-margin plus uniform remainder;independent160027 layouts'))
    observations=np.asarray([r['observation'] for r in pool],np.float32)
    obs=torch.as_tensor(observations,device=args.device)
    oracle=SimulatorOracle(workers=args.workers)
    summaries=[];states=[];routes=[];timings=[];route_steps=0
    try:
        truth=oracle.evaluate_numpy(observations,np.asarray([r['source_action'] for r in pool],np.float32))
        for policy in policies:
            config=policy['config'];learner=policy['learner']
            rows,per_state=evaluate_samples(learner,obs,pool,truth,oracle)
            for row in rows: row.update(training_seed=config['training_seed'],native_source_commit=config['source_commit'])
            summaries.extend(rows)
            states.append(dict(variant=config['variant'],training_seed=config['training_seed'],samplers=per_state))
            if config['variant']!='s_iso':
                records,steps=route_rollouts(learner,args.device);route_steps+=steps
                for record in records: record.update(variant=config['variant'],training_seed=config['training_seed'])
                routes.extend(records)
                measured=inference_timings(learner,obs)
                for row in measured: row.update(variant=config['variant'],training_seed=config['training_seed'])
                timings.extend(measured)
        atomic_json(args.output/'mechanism.json',dict(summaries=summaries,per_state=states,source_oracle=truth.tolist(),
            source_interval_replica_mae_m=float(np.mean(np.abs(truth[:,0]-np.asarray([r['source_interval_margin'] for r in pool])))),
            interpretation='new_fixed1M_policies;fixed_spectrum_counterfactuals_not_retrained_methods;true_axes_at_source_actions;undefined_axes_retained'))
        atomic_json(args.output/'route_rollouts.json',dict(records=routes,
            interpretation='paired10episodes/condition/sampler/frozen_selected_policy;sector_occupancy_is_small_route_proxy_not_entropy;diverging_states_are_not_actionwise_spectral_matches'))
        write_csv(args.output/'MECHANISM_RESULTS.csv',summaries)
        write_csv(args.output/'INFERENCE_TIMINGS.csv',timings)
        statistics=oracle.statistics()
        atomic_json(args.output/'receipt.json',dict(status='completed',source_commit=args.source_commit,
            training_transitions=0,optimizer_steps=0,states=64,draws_per_state=32,models=4,
            fixed_checkpoint_steps=1000000,pool_creation_intervals=pool_steps,pool_creation_episodes=pool_episodes,
            route_evaluation_intervals=route_steps,route_evaluation_episodes=len(routes),oracle_statistics=statistics,
            total_new_physical_intervals=pool_steps+route_steps+statistics['counterfactual_control_intervals']+statistics['margin_only_queries'],
            elapsed_seconds=time.monotonic()-started,primary_evaluation_replaced=False))
    finally: oracle.close()


if __name__=='__main__':
    main()

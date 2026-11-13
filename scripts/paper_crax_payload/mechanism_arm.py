"""Fresh Reach M1/M2/M3 mechanism data; frozen policies, no new training.

Explicit authorization/resource scheduling is the coordinator's responsibility.
This runner never imports a robot SDK.  All oracle queries occur AFTER learned
actions have been generated, are used for evaluation only, and are costed.
The fresh pool is not the historical64-state analysis renamed as128 states.
"""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.paper_crax_payload.mechanism_sampling import (
    M1_MODES, M2_MODES, common_noise, covariance_matching_description, frozen_sample)
from scripts.star.arm_safe_entropy.conditional_metrics import endpoint_metrics
from scripts.star.arm_safe_entropy.oracle import SimulatorOracle


CONDITIONS = (('nominal', .06), ('radius_080', .08))


def atomic_json(path, payload):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def fresh_pool(policies, device, layout_seed=260027):
    from scripts.star.arm_safe_entropy.environment import TaskConfig
    from scripts.star.arm_safe_entropy.evaluate_obstacle_robustness import PerturbedRelativeReach
    records = []; seen = set(); control_intervals = episodes = 0
    rng = np.random.default_rng(2610027)
    for policy in policies:
        config, learner = policy['config'], policy['learner']
        for condition,radius in CONDITIONS:
            env = PerturbedRelativeReach(TaskConfig(version='SafeReach-RelativeLayout-v3',obstacle_radius=radius))
            candidates = []; local_seen = set()
            try:
                for episode in range(16):
                    seed = layout_seed+episode
                    obs,_ = env.reset_condition(seed,0.)
                    torch.manual_seed(2630027+config['training_seed']*100+episode)
                    while True:
                        with torch.no_grad():
                            action = learner.actor(torch.as_tensor(obs[None],device=device),geometry=learner.geometry)[0][0].cpu().numpy()
                        following,_,done,truncated,info = env.step(action)
                        control_intervals += 1
                        key = obs.tobytes()
                        if key not in seen and key not in local_seen:
                            candidates.append(dict(observation=obs.tolist(),source_action=action.tolist(),
                                source_interval_margin=info['local_margin'],condition=condition,layout_seed=seed,
                                source_variant=config['variant'],source_seed=config['training_seed']))
                            local_seen.add(key)
                        obs = following
                        if done or truncated:
                            break
                    episodes += 1
                    near = [i for i,r in enumerate(candidates) if abs(r['source_interval_margin'])<=.02]
                    interior = [i for i,r in enumerate(candidates) if r['source_interval_margin']<-.02]
                    if episode >= 3 and len(candidates)>=16 and len(near)>=8 and len(interior)>=8:
                        break
            finally:
                env.close()
            if len(candidates)<16:
                raise ValueError('Insufficient distinct fresh states; no duplication or historical padding')
            chosen = list(rng.choice(near,size=min(8,len(near)),replace=False))
            remaining_interior = [i for i in interior if i not in chosen]
            chosen.extend(rng.choice(remaining_interior,size=min(16-len(chosen),len(remaining_interior)),replace=False).tolist())
            remaining = [i for i in range(len(candidates)) if i not in chosen]
            chosen.extend(rng.choice(remaining,size=16-len(chosen),replace=False).tolist())
            for i in sorted(chosen):
                records.append(candidates[i]); seen.add(np.asarray(candidates[i]['observation'],np.float32).tobytes())
    near_count = sum(abs(r['source_interval_margin'])<=.02 for r in records)
    interior_count = sum(r['source_interval_margin']<-.02 for r in records)
    if len(records)!=128 or len(seen)!=128 or min(near_count,interior_count)<16:
        raise ValueError('Require128 distinct mixed-policy states with both interior and near-boundary coverage')
    return records,dict(pool_creation_control_intervals=control_intervals,pool_creation_episodes=episodes,
        near_source_interval_states=near_count,safe_interior_source_interval_states=interior_count,
        layout_seed_start=layout_seed,old_pool_reused=False)


def _mean(values):
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def assess_sampler(learner, pool, truth, oracle, mode, *, rho=.25, steps=4, draws=64):
    actor = deepcopy(learner.actor).requires_grad_(False)
    actor.num_steps = steps
    observations = np.asarray([r['observation'] for r in pool],np.float32)
    obs = torch.as_tensor(observations,device=learner.device)
    noise,axes = common_noise(actor,obs,draws=draws)
    batch = obs.repeat_interleave(draws,0)
    started = time.monotonic()
    if obs.device.type=='cuda':
        torch.cuda.synchronize()
    actions,energy,clips,reference = frozen_sample(actor,batch,learner.margin_geometry,mode,noise,axes,rho=rho)
    if obs.device.type=='cuda':
        torch.cuda.synchronize()
    elapsed = time.monotonic()-started
    if not torch.isfinite(actions).all() or not torch.isfinite(energy).all():
        raise FloatingPointError('Non-finite frozen sampler output; do not query oracle or fabricate a receipt')
    actions_np = actions.cpu().numpy().reshape(len(pool),draws,3)
    # Oracle is not reachable until actions are complete, detached and fixed.
    labels = oracle.margins_numpy(np.repeat(observations,draws,axis=0),actions_np.reshape(-1,3)).reshape(len(pool),draws)
    refs = {k:v.cpu().numpy().reshape(len(pool),draws,*v.shape[1:]) for k,v in reference.items()}
    source_actions=torch.as_tensor(np.asarray([r['source_action'] for r in pool],np.float32),device=obs.device)
    source_normal,_,diagnostic=learner.margin_geometry.geometry(obs,source_actions,return_diagnostics=True)
    source_normal=source_normal.cpu().numpy()
    source_valid=(truth[:,4]>1e-7)&diagnostic['gradient_valid'].cpu().numpy()
    with torch.no_grad():source_predicted=learner.margin_geometry(obs,source_actions).mean(0).cpu().numpy()
    records = []
    for index,row in enumerate(pool):
        normal = truth[index,1:4]
        metric = endpoint_metrics(actions_np[index],normal)
        validity = refs['valid'][index].all(-1)
        # Match all reference eigenvalues, not the output covariance.  The
        # joint cross-mode subset is computed after all modes have returned.
        records.append(dict(state_index=index,condition=row['condition'],true_axis_valid=bool(truth[index,4]>1e-7),
            true_interval_risk=float(np.mean(labels[index]>0)),endpoint=metric,
            learned_valid_draws=int(validity.sum()),source_true_margin=float(truth[index,0]),
            mean_reference_normal_variance=float(refs['vn'][index].mean()),
            mean_reference_tangent_variance=float(refs['vt'][index].mean())))
    summaries = []
    for condition,_ in CONDITIONS:
        indices = [i for i,r in enumerate(pool) if r['condition']==condition]
        valid_rows = [records[i] for i in indices if records[i]['true_axis_valid']]
        summaries.append(dict(sampler=mode,condition=condition,states=len(indices),draws_per_state=draws,
            fixed_q=1.,rho=rho,flow_steps=steps,training_transitions=0,retrained=False,
            true_interval_risk=float(np.mean(labels[indices]>0)),valid_true_axis_states=len(valid_rows),
            source_direction_valid_states=int(sum(source_valid[i] for i in indices)),
            source_action_normal_cosine=_mean([float(source_normal[i]@truth[i,1:4]) for i in indices if source_valid[i]]),
            source_action_margin_mae=float(np.abs(source_predicted[indices]-truth[indices,0]).mean()),
            direction_scope='learned direct source-action normal versus true interval-action normal;not an oracle gradient at every sampled injection',
            endpoint_vn=_mean([r['endpoint']['vn'] for r in valid_rows]),
            endpoint_vt=_mean([r['endpoint']['vt'] for r in valid_rows]),
            endpoint_trace=_mean([records[i]['endpoint']['trace'] for i in indices]),
            action_clipping_fraction=float(clips.cpu().numpy().reshape(len(pool),draws)[indices].mean()),
            proposal_clipping_fraction=float(refs['proposal_clipped'][indices].mean()),
            invalid_gradient_fraction=float(1-refs['valid'][indices].mean()),
            mean_reference_trace=float((refs['vn'][indices]+2*refs['vt'][indices]).mean()),
            mean_energy=float(energy.cpu().numpy().reshape(len(pool),draws)[indices].mean()),
            sampling_batch_seconds=elapsed,scope='frozen_sampler_intervention_not_new_training_or_primary_episode_evaluation'))
    return summaries,records,dict(actions=actions_np,labels=labels,reference=refs)


def cross_sampler_routes(policies,device,episodes_per_condition=30):
    """Native/common sampler × trained Actor, paired layouts, frozen models."""
    from scripts.star.arm_safe_entropy.environment import TaskConfig
    from scripts.star.arm_safe_entropy.evaluate_obstacle_robustness import PerturbedRelativeReach
    from scripts.star.arm_safe_entropy.mechanism_stability import route_sector
    records=[];steps=0
    for policy in policies:
        learner,config=policy['learner'],policy['config']
        for condition,radius in CONDITIONS:
            env=PerturbedRelativeReach(TaskConfig(version='SafeReach-RelativeLayout-v3',obstacle_radius=radius))
            try:
                for mode in ('native','common_iso_high','common_pullback'):
                    for episode in range(episodes_per_condition):
                        seed=270027+episode
                        obs,layout=env.reset_condition(seed,0.)
                        torch.manual_seed(2710027+episode)
                        points=[obs[14:17].tolist()];length=0;episode_return=0.
                        while True:
                            batch=torch.as_tensor(obs[None],device=device)
                            if mode=='native':
                                with torch.no_grad(): action=learner.actor(batch,geometry=learner.geometry)[0][0].cpu().numpy()
                            else:
                                noise=torch.randn(learner.actor.num_steps,1,3,device=device)
                                axes=torch.zeros(1,3,device=device);axes[:,0]=1
                                sampler='iso_high' if mode=='common_iso_high' else 'endpoint_pullback'
                                action=frozen_sample(learner.actor,batch,learner.margin_geometry,sampler,noise,axes)[0][0].cpu().numpy()
                            obs,reward,done,truncated,info=env.step(action)
                            length+=1;steps+=1;episode_return+=reward;points.append(obs[14:17].tolist())
                            if done or truncated: break
                        success=bool(info['safe_success']);violation=bool(info['violation'])
                        points_array=np.asarray(points)
                        # Fixed3cm world-coordinate voxels, counted only on
                        # completed safe-success routes.  Coverage is a proxy
                        # and not a differential entropy estimate.
                        covered_voxels=sorted(set(tuple(int(y) for y in x) for x in np.floor(points_array/.03).astype(int))) if success else []
                        records.append(dict(variant=config['variant'],training_seed=config['training_seed'],sampler=mode,
                            condition=condition,layout_seed=seed,safe_success=success,violation=violation,
                            safe_timeout=not success and not violation,episode_return=episode_return,length=length,
                            capped_seconds=length*.04 if success else 4.,
                            successful_route_sector=route_sector(points,layout['obstacle'],layout['goal']) if success else None,
                            successful_route_3cm_voxels=[list(x) for x in covered_voxels],
                            task_path_length_m=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())))
            finally: env.close()
    return records,steps


def inference_latency(actor,margin,observation,repeats=100):
    """Single-action p50/p95, compiled environment and training not included."""
    rows=[]
    noise,axes=common_noise(actor,observation,draws=2)
    noise=noise[:,:1];axes=axes[:1]
    for mode in M2_MODES:
        durations=[]
        for index in range(repeats+5):
            if observation.device.type=='cuda':torch.cuda.synchronize()
            started=time.monotonic()
            frozen_sample(actor,observation,margin,mode,noise,axes)
            if observation.device.type=='cuda':torch.cuda.synchronize()
            if index>=5:durations.append(time.monotonic()-started)
        rows.append(dict(sampler=mode,batch_size=1,repeats=repeats,p50_seconds=float(np.percentile(durations,50)),
            p95_seconds=float(np.percentile(durations,95)),scope='frozen_single_action_no_training_no_environment',
            compile_seconds=0.,framework='eager_PyTorch'))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs=4,type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--workers',type=int,default=10)
    parser.add_argument('--routes',action='store_true')
    parser.add_argument('--parameter-sensitivity',action='store_true')
    args=parser.parse_args()
    if args.output.exists():parser.error('Do not overwrite earlier mechanism evidence')
    from scripts.star.arm_safe_entropy.mechanism_stability import load_final_runs
    torch.set_num_threads(1);started=time.monotonic()
    policies=load_final_runs(args.runs,args.device)
    args.output.mkdir(parents=True)
    pool,pool_cost=fresh_pool(policies,args.device)
    atomic_json(args.output/'state_pool.json',dict(records=pool,**pool_cost))
    observations=np.asarray([r['observation'] for r in pool],np.float32)
    oracle=SimulatorOracle(args.workers)
    summaries=[];states=[];latency=[];routes=[];route_steps=0
    try:
        truth=oracle.evaluate_numpy(observations,np.asarray([r['source_action'] for r in pool],np.float32))
        for policy in policies:
            config,learner=policy['config'],policy['learner']
            # Every retained seed/Actor is evaluated, not just the good EA seed.
            results={}
            for mode in M2_MODES+M1_MODES:
                rows,records,raw=assess_sampler(learner,pool,truth,oracle,mode)
                for row in rows:row.update(variant=config['variant'],training_seed=config['training_seed'],experiment='M1' if mode in M1_MODES else 'M2')
                summaries.extend(rows);states.append(dict(variant=config['variant'],training_seed=config['training_seed'],sampler=mode,records=records))
                results[mode]=raw
            matched=np.ones((128,64),bool)
            for mode in M2_MODES:
                ref=results[mode]['reference'];base=results['local']['reference']
                matched &= (np.isclose(ref['vn'],base['vn'])&np.isclose(ref['vt'],base['vt'])).all(-1)
            for row in summaries:
                if row.get('variant')!=config['variant'] or row.get('training_seed')!=config['training_seed'] or row['experiment']!='M2':continue
                indices=[i for i,r in enumerate(pool) if r['condition']==row['condition']]
                mask=matched[indices];labels=results[row['sampler']]['labels'][indices]
                row['joint_matched_spectrum_draws']=int(mask.sum());row['total_condition_draws']=int(mask.size)
                row['joint_matched_spectrum_risk']=float(np.mean(labels[mask]>0)) if mask.any() else None
                matched_metrics=[endpoint_metrics(results[row['sampler']]['actions'][i,matched[i]],truth[i,1:4])
                    for i in indices if truth[i,4]>1e-7 and matched[i].sum()>=2]
                row['joint_matched_valid_axis_states']=len(matched_metrics)
                row['joint_matched_endpoint_vn']=_mean([r['vn'] for r in matched_metrics])
                row['joint_matched_endpoint_vt']=_mean([r['vt'] for r in matched_metrics])
            if args.parameter_sensitivity:
                for rho,steps in ((.1,4),(.5,4),(1.,4),(.25,2),(.25,8)):
                    rows,records,_=assess_sampler(learner,pool,truth,oracle,'endpoint_pullback',rho=rho,steps=steps)
                    for row in rows:row.update(variant=config['variant'],training_seed=config['training_seed'],experiment='frozen_parameter_sensitivity')
                    summaries.extend(rows);states.append(dict(variant=config['variant'],training_seed=config['training_seed'],rho=rho,flow_steps=steps,records=records))
            obs=torch.as_tensor(observations[:1],device=args.device)
            for row in inference_latency(learner.actor,learner.margin_geometry,obs):
                row.update(variant=config['variant'],training_seed=config['training_seed']);latency.append(row)
        if args.routes:routes,route_steps=cross_sampler_routes(policies,args.device)
        atomic_json(args.output/'mechanism.json',dict(summaries=summaries,per_state=states,
            source_oracle=truth.tolist(),covariance_matching=covariance_matching_description(),
            raw_source_margin_replica_mae=float(np.mean(np.abs(truth[:,0]-np.asarray([r['source_interval_margin'] for r in pool])))),
            axes='same true source-action interval normal per state;undefined axes retained,excluded only from projected variances'))
        atomic_json(args.output/'routes.json',dict(records=routes,scope='native/common-sampler cross;sector occupancy proxy not entropy'))
        for name,rows in (('MECHANISM_RESULTS.csv',summaries),('PARAMETER_EFFICIENCY.csv',latency)):
            fields=list(dict.fromkeys(key for row in rows for key in row))
            with (args.output/name).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        stats=oracle.statistics()
        atomic_json(args.output/'receipt.json',dict(status='completed',source_commit=args.source_commit,
            training_transitions=0,optimizer_updates=0,real_interactions=0,states=128,draws_per_state=64,
            **pool_cost,route_control_intervals=route_steps,route_episodes=len(routes),oracle_statistics=stats,
            new_simulator_control_intervals=pool_cost['pool_creation_control_intervals']+route_steps+stats['counterfactual_control_intervals']+stats['margin_only_queries'],
            elapsed_seconds=time.monotonic()-started,primary_evaluation_replaced=False,
            scope='M1/M2 intervention and optional M3/frozen parameter sensitivity;not newly trained variants'))
    finally:oracle.close()


if __name__=='__main__':main()

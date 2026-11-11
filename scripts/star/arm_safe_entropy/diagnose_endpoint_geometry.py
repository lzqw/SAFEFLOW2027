"""Bounded64-state,32-draw held-out geometric attribution; no training.

Counterfactual fixed-spectrum samplers are diagnostics,not new trained methods.
Invalid-gradient isotropic fallbacks remain visible; a jointly matched-spectrum
subset is explicitly labelled rather than pretending every path has equal D.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.star.arm_safe_entropy.conditional_metrics import endpoint_metrics
from scripts.star.arm_safe_entropy.diagnose_stability import native_run,BASELINE
from scripts.star.arm_safe_entropy.endpoint_geometry import EndpointGeometry,StableFlowActor
from scripts.star.arm_safe_entropy.environment import TaskConfig
from scripts.star.arm_safe_entropy.evaluate_obstacle_robustness import (
    PerturbedRelativeReach,load_policy,atomic_json)
from scripts.star.arm_safe_entropy.oracle import SimulatorOracle


class FixedSpectrumGeometry(EndpointGeometry):
    def query(self,*args,**kwargs):
        normal,_,diagnostic=super().query(*args[:5],return_diagnostics=True)
        weight=.5*diagnostic['valid'].to(normal.dtype)
        if len(args)>5 and args[5] or kwargs.get('return_diagnostics',False):
            return normal,weight,diagnostic
        return normal,weight


def collect_pool(project,device,frozen):
    selected=[];steps=episodes=0;seen=set()
    rng=np.random.default_rng(6190027)
    for seed in (2030,2032):
        run=native_run(project,'flow_learned',seed)
        for checkpoint in (600000,1000000):
            actor,margin,_=load_policy(run,device,checkpoint,frozen)
            for name,radius in (('nominal',.06),('radius_080',.08)):
                env=PerturbedRelativeReach(TaskConfig(version=frozen['task_version'],obstacle_radius=radius))
                candidates=[];candidate_keys=set()
                try:
                    for episode in range(8):
                        layout_seed=150027+episode
                        obs,_=env.reset_condition(layout_seed,0.)
                        torch.manual_seed(6910027+seed*1000+episode)
                        while True:
                            with torch.no_grad():
                                action=actor(torch.as_tensor(obs[None],device=device),geometry=margin.geometry)[0][0].cpu().numpy()
                            following,_,done,truncated,info=env.step(action)
                            steps+=1
                            key=obs.tobytes()
                            if key not in seen and key not in candidate_keys:
                                candidates.append(dict(observation=obs.tolist(),source_action=action.tolist(),
                                    source_interval_margin=info['local_margin'],condition=name,layout_seed=layout_seed,
                                    source_seed=seed,source_checkpoint=checkpoint,episode_step=env._elapsed))
                                candidate_keys.add(key)
                            obs=following
                            if done or truncated: break
                        episodes+=1
                        if episode>=3 and len(candidates)>=8: break
                finally:
                    env.close()
                near=[i for i,r in enumerate(candidates) if abs(r['source_interval_margin'])<=.02]
                chosen=list(rng.choice(near,size=min(4,len(near)),replace=False))
                remainder=[i for i in range(len(candidates)) if i not in chosen]
                chosen.extend(rng.choice(remainder,size=8-len(chosen),replace=False).tolist())
                selected.extend(candidates[i] for i in sorted(chosen))
                seen.update(np.asarray(candidates[i]['observation'],np.float32).tobytes() for i in chosen)
    if len(selected)!=64: raise ValueError('Do not pad or duplicate a small shared-state pool')
    return selected,steps,episodes


def mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Do not overwrite an earlier geometry diagnosis')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    frozen=json.loads((Path(__file__).resolve().parents[3]/'configs/star/arm_safe_entropy/main_plan.json').read_text())
    started=time.monotonic()
    pool,pool_steps,pool_episodes=collect_pool(args.project,args.device,frozen)
    atomic_json(args.output/'state_pool.json',dict(records=pool,pool_creation_transitions=pool_steps,
        pool_creation_episodes=pool_episodes,episode_isolated_from_training=True,layout_seed_start=150027,
        sampling='8states per historical seed/checkpoint/condition group;up to4near-source-margin states plus uniform remainder'))
    observations=np.asarray([r['observation'] for r in pool],np.float32)
    source_actions=np.asarray([r['source_action'] for r in pool],np.float32)
    oracle=SimulatorOracle(workers=args.workers)
    summaries=[];all_states=[]
    try:
        truth=oracle.evaluate_numpy(observations,source_actions)
        obs=torch.as_tensor(observations,device=args.device)
        batch=obs.repeat_interleave(32,0)
        for seed in (2030,2032):
            run=native_run(args.project,'flow_learned',seed)
            for checkpoint in (600000,1000000):
                old_actor,margin,_=load_policy(run,args.device,checkpoint,frozen)
                actor=StableFlowActor(25).to(args.device)
                actor.load_state_dict(old_actor.state_dict());actor.requires_grad_(False)
                predicted_normal,_=margin.geometry(obs,torch.as_tensor(source_actions,device=args.device))
                normal=predicted_normal.cpu().numpy()
                true_valid=truth[:,4]>1e-7
                both=true_valid&(np.linalg.norm(normal,axis=1)>1e-7)
                cos=(normal*truth[:,1:4]).sum(-1)
                variants={}
                for mode in ('local','endpoint_direct','endpoint_pullback'):
                    geometry=FixedSpectrumGeometry(margin,mode,30000)
                    torch.manual_seed(7290027)
                    if args.device.startswith('cuda'): torch.cuda.synchronize()
                    sample_start=time.monotonic()
                    with torch.no_grad():
                        actions,_,clips,trace=actor(batch,geometry=geometry,return_reference=True)
                    arrays=actions.cpu().numpy().reshape(64,32,3)
                    elapsed=time.monotonic()-sample_start
                    labels=oracle.margins_numpy(np.repeat(observations,32,axis=0),arrays.reshape(-1,3)).reshape(64,32)
                    with torch.no_grad():
                        predicted=margin(batch,actions).mean(0).cpu().numpy().reshape(64,32)
                    variants[mode]=dict(actions=arrays,labels=labels,prediction=predicted,
                        weights=trace['weights'].cpu().numpy().reshape(64,32,4),
                        zero=trace['geometry_zero_gradient'].cpu().numpy().reshape(64,32,4),
                        clipped=trace['geometry_proposal_clipped'].cpu().numpy().reshape(64,32,4),
                        clips=clips.cpu().numpy().reshape(64,32),sampling_seconds=elapsed)
                matched=np.ones((64,32),bool)
                reference=variants['local']['weights']
                for value in variants.values(): matched&=np.all(value['weights']==reference,axis=-1)
                for mode,value in variants.items():
                    states=[]
                    for i,row in enumerate(pool):
                        metric=endpoint_metrics(value['actions'][i],truth[i,1:4])
                        matched_metric=(endpoint_metrics(value['actions'][i,matched[i]],truth[i,1:4])
                                        if matched[i].sum()>=2 else None)
                        states.append(dict(state_index=i,condition=row['condition'],
                            true_normal_valid=bool(true_valid[i]),endpoint=metric,
                            matched_endpoint=matched_metric,matched_draws=int(matched[i].sum()),
                            true_interval_risk=float(np.mean(value['labels'][i]>0)),
                            matched_true_interval_risk=float(np.mean(value['labels'][i,matched[i]]>0)) if matched[i].any() else None))
                    for condition in ('nominal','radius_080'):
                        indices=[i for i,r in enumerate(pool) if r['condition']==condition]
                        labels=value['labels'][indices];predicted=value['prediction'][indices]
                        near=np.abs(labels)<=.02;predicted_safe=predicted<0
                        valid_states=[r for r in states if r['condition']==condition and r['true_normal_valid']]
                        same_states=[r for r in valid_states if r['matched_endpoint'] is not None]
                        selected_match=matched[indices]
                        summary=dict(seed=seed,checkpoint_steps=checkpoint,mode=mode,condition=condition,
                            states=len(indices),draws_per_state=32,common_q=.5,rho=.25,reference_variance=.04,
                            true_interval_risk=float(np.mean(labels>0)),
                            matched_spectrum_draws=int(selected_match.sum()),total_draws=int(labels.size),
                            matched_spectrum_interval_risk=float(np.mean(labels[selected_match]>0)) if selected_match.any() else None,
                            endpoint_vn=mean_or_none([r['endpoint']['vn'] for r in valid_states]),
                            endpoint_vt=mean_or_none([r['endpoint']['vt'] for r in valid_states]),
                            matched_endpoint_vn=mean_or_none([r['matched_endpoint']['vn'] for r in same_states]),
                            matched_endpoint_vt=mean_or_none([r['matched_endpoint']['vt'] for r in same_states]),
                            margin_mae_m=float(np.mean(np.abs(labels-predicted))),near_boundary_draws=int(near.sum()),
                            near_boundary_mae_m=float(np.mean(np.abs(labels[near]-predicted[near]))) if near.any() else None,
                            false_safe_joint_fraction=float(np.mean((labels>0)&predicted_safe)),
                            false_safe_given_predicted_safe=float(np.mean(labels[predicted_safe]>0)) if predicted_safe.any() else None,
                            geometry_zero_gradient_fraction=float(value['zero'][indices].mean()),
                            geometry_proposal_clipped_fraction=float(value['clipped'][indices].mean()),
                            clipping_fraction=float(value['clips'][indices].mean()),
                            source_action_normal_cosine=mean_or_none(cos[[i for i in indices if both[i]]]),
                            sampling_seconds_full64state_batch=value['sampling_seconds'])
                        summaries.append(summary)
                        print(json.dumps(summary),flush=True)
                    all_states.append(dict(seed=seed,checkpoint_steps=checkpoint,mode=mode,states=states))
                atomic_json(args.output/'geometry_diagnosis.json',dict(baseline_commit=BASELINE,
                    summaries=summaries,per_state=all_states,source_oracle=truth.tolist(),training_transitions=0,
                    interpretation='frozen_counterfactual_sampler_not_trained_policy_evidence;matched_subset_filters_geometry_validity_not_outcomes'))
        fields=list(summaries[0])
        with (args.output/'geometry_diagnosis.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(summaries)
        stats=oracle.statistics()
        atomic_json(args.output/'receipt.json',dict(status='completed',source_commit=args.source_commit,
            baseline_commit=BASELINE,states=64,draws_per_state=32,models=4,geometry_modes=3,
            summary_rows=len(summaries),training_transitions=0,optimizer_steps=0,
            pool_creation_transitions=pool_steps,pool_creation_episodes=pool_episodes,
            oracle_statistics=stats,total_new_physical_intervals=pool_steps+stats['counterfactual_control_intervals']+stats['margin_only_queries'],
            elapsed_seconds=time.monotonic()-started,
            limitations='Small stratified held-out state pool,not whole-episode robustness. Common true axes are at source actions;invalid axes remain undefined. Fixedq=.5 is a diagnostic intervention;invalid gradients retain isotropic fallback. Joint spectral matching coverage is explicit.'))
    finally:
        oracle.close()


if __name__=='__main__':
    main()

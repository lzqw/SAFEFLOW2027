"""Finite batched native-CRAX Flow training; all sampling charged, no watchdog.

Checkpoint = learner+optimizers+replay+MJX state+per-env partial episodes+RNG.
Do not put the server-only checkpoint output under reports or upload it to GitHub.
"""
import argparse
import atexit
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np
import torch

from scripts.paper_crax_payload.crax_env import PointGoalTorch, DeviceReplay
from scripts.paper_crax_payload.crax_models import CumulativeFlowLearner
from scripts.paper_crax_payload.configuration import load_configuration
from scripts.paper_crax_payload.observation_normalizer import ObservationNormalizer
from scripts.star.arm_safe_entropy.stability_runtime import capture_rng, restore_rng, isolated_rng


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def evaluate(learner, num_envs, seed, normalizer=None, environment_cache=None):
    """One complete native episode per fresh env; no adaptation or rollout-RNG use."""
    start = time.monotonic()
    with isolated_rng():
        env=None if environment_cache is None else environment_cache.get('env')
        if env is None:
            env=PointGoalTorch(num_envs,seed,str(learner.device))
            if environment_cache is not None:environment_cache['env']=env
        if env.num_envs!=num_envs:raise ValueError('Do not silently change evaluation episode batch')
        # Reuse the already compiled simulator, not the previous episode/state.
        # Every checkpoint still gets the identical fresh seeded native layouts.
        obs = env.reset(seed=seed)
        rng = torch.Generator(device=learner.device).manual_seed(seed + 1)
        active = torch.ones(num_envs, dtype=torch.bool, device=learner.device)
        returns = torch.zeros(num_envs, device=learner.device)
        costs, goals, lengths = (torch.zeros_like(returns) for _ in range(3))
        first_goal=torch.full_like(returns,env.native.episode_length)
        # Native horizon is retained. No host sync for per-env or per-step decisions.
        for index in range(env.native.episode_length):
            with torch.no_grad():
                policy_obs=obs if normalizer is None else normalizer.transform(obs)
                action, _, _ = learner.actor(policy_obs, geometry=learner.geometry, generator=rng)
            obs, reward, done, info = env.step(action)
            returns += reward * active
            costs += info['cost'] * active
            first_goal=torch.where(active & (goals==0) & (info['goals']>0),index+1,first_goal)
            goals += info['goals'] * active
            lengths += active
            active &= ~done
        records = torch.stack((returns,costs,goals,lengths,first_goal),-1).cpu().tolist()
    qualified = [r for r in records if r[1] <= learner.settings['cost_budget']]
    return dict(seed=seed,episodes=num_envs,records=[dict(native_return=r[0],episode_cost=r[1],goals=r[2],length=r[3],
        capped_first_goal_seconds=r[4]*float(env.native.dt),first_goal_seconds=r[4]*float(env.native.dt) if r[2]>=1 else None) for r in records],
        native_return=float(np.mean([r[0] for r in records])),episode_cost=float(np.mean([r[1] for r in records])),
        goals_per_episode=float(np.mean([r[2] for r in records])),at_least_one_goal=float(np.mean([r[2]>=1 for r in records])),
        budget_compliant_fraction=len(qualified)/len(records),
        budget_compliant_return=float(np.mean([r[0] for r in qualified])) if qualified else None,
        budget_compliant_episodes=len(qualified),physical_controls=num_envs*env.native.episode_length,
        capped_first_goal_seconds=float(np.mean([r[4] for r in records]))*float(env.native.dt),
        conditional_first_goal_seconds=float(np.mean([r[4] for r in records if r[2]>=1]))*float(env.native.dt) if any(r[2]>=1 for r in records) else None,
        elapsed_seconds=time.monotonic()-start,training_updates=0,dual_updates=0,
        actor_margin_normalizer_dual_frozen=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/paper_crax_payload/master_v1.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=['stable_iso','ea','local'],default='stable_iso')
    p.add_argument('--profile',choices=['balanced','fast','learn_heavy'],default='balanced')
    p.add_argument('--steps',type=int,default=500000)
    p.add_argument('--seed',type=int,default=41027)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',type=Path)
    p.add_argument('--repair-episode-accounting-from-replay',action='store_true')
    p.add_argument('--profile-only',action='store_true')
    p.add_argument('--eval-envs',type=int,default=50)
    p.add_argument('--source-commit',required=True)
    args=p.parse_args()
    master=load_configuration(args.config); cfg=master['pointgoal']; profile=cfg['profiles'][args.profile]
    if args.steps <= 0 or args.steps > master['budgets']['pointgoal_development_transitions']:
        raise ValueError('Invalid finite development target')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError('Never overwrite an existing run or restart on observation timeout')
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    # Shared engineering setting for ALL new CRAX Flow controls. Record it;
    # this is not a measured causal TF32 speedup or unchanged bitwise arithmetic.
    torch.set_float32_matmul_precision('high')
    device=torch.device(args.device)
    if device.type!='cuda':raise ValueError('Actual GPU path required for reported full-training throughput')
    torch.cuda.set_device(device)
    E,B,G,F=(profile[k] for k in ('E','B','G','F'))
    if F!=1:raise ValueError('This first profile implements registered F1 only')
    shared=cfg['shared_learner']
    env=PointGoalTorch(E,args.seed,str(device))
    norm_mode=cfg['observation_normalization']
    if norm_mode=='none_initial_shared_with_reference':norm_mode='none'
    normalizer=ObservationNormalizer(env.obs_dim,device,norm_mode)
    learner=CumulativeFlowLearner(env.obs_dim,env.action_dim,args.method,str(device),width=shared['hidden_width'],
        actor_lr=shared['actor_lr'],critic_lr=shared['critic_lr'],variance=shared['reference_variance'],
        rho=shared['normal_variance_ratio'],cost_budget=cfg['episodic_cost_budget'],discount=cfg['discount'],
        critic_only_updates=0 if args.profile_only else shared['critic_only_updates'],flow_steps=shared['flow_steps'],
        dual_rule=shared.get('dual_rule','episode_mean'),episode_length=env.native.episode_length,
        lagrangian_lr=shared.get('lagrangian_lr',.01))
    if args.profile_only:
        # Engineering-only workload: exercise actual EA differentiation, not
        # the disabled warmup branch. No invented margin/preheat update counts.
        learner.geometry.profile_full_geometry=True
        learner.geometry.set_margin_updates(0)
    replay=DeviceReplay(shared['replay_capacity'],env.obs_dim,env.action_dim,device,args.seed+2)
    rollout_rng=torch.Generator(device=device).manual_seed(args.seed+3)
    partial={k:torch.zeros(E,device=device) for k in ('return','cost','goals','length')}
    totals=torch.zeros(6,device=device) # rawcost,completeeps,return,episodecost,goals,>=1goal
    completed_records=[]; steps=initial_steps=0; eval_controls=0; eval_seconds=save_seconds=0.
    if args.resume:
        state=torch.load(args.resume,map_location=device,weights_only=False)
        if args.repair_episode_accounting_from_replay:
            from scripts.paper_crax_payload.recover_episode_accounting import recover_episode_accounting
            recovery,recovered_rows=recover_episode_accounting(state)
            atomic_json(args.output/'episode_accounting_recovery.json',recovery)
            with (args.output/'recovered_training_episodes.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['observed_by_steps','native_return','episode_cost','goals','length']);writer.writerows(recovered_rows)
        if state['profile']!=profile or state['seed']!=args.seed or state['method']!=args.method:
            raise ValueError('Resume profile/seed/method mismatch; use explicit candidate fork instead')
        learner.load_state_dict(state['learner']); replay.load_state_dict(state['replay'])
        if 'normalizer' in state:normalizer.load_state_dict(state['normalizer'])
        elif norm_mode!='none':raise ValueError('Missing native normalizer;do not change units on resume')
        obs=env.load_state_dict(state['environment']); partial=state['partial'];totals=state['totals']
        steps=initial_steps=state['steps'];rollout_rng.set_state(state['rollout_rng'].cpu());restore_rng(state['rng'])
    else:obs=env.reset()
    torch.cuda.synchronize(); reset_compile_seconds=time.monotonic()-started
    atomic_json(args.output/'configuration.json',dict(master=master,profile_name=args.profile,
        phase='engineering_profile' if args.profile_only else 'development',
        profile=profile,method=args.method,seed=args.seed,source_commit=args.source_commit,
        task=env.effective_config(),requested_steps=args.steps,
        actual_target_steps=((args.steps+E-1)//E)*E,initial_steps=initial_steps,
        task_mode='cumulative',cost_q='unbounded_discounted_raw_cost',
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        dual=(f"native_per_step_bound25/H;lr{shared.get('lagrangian_lr',.01)}/H_per_learner_round;cap2;full_episodes_logged_separately"
              if learner.dual_rule=='native_per_step' else 'raw_completed_episode_batch_mean;lr0.01;EMA0.05;clip0_100;not_native_SAC_perstep_rule'),
        u=G/(E*F),replay_exposure=G*B/(E*F),energy_target=learner.energy_target,
        jax_preallocate=os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'],
        jax_compilation_cache_dir=os.environ.get('JAX_COMPILATION_CACHE_DIR'),
        engineering_profile_overrides=dict(critic_only_updates=0,geometry_beta=1.) if args.profile_only else None))
    checkpoints=[] if args.profile_only else sorted(set(((int(args.steps*f)+E-1)//E)*E for f in master['selection']['candidate_checkpoint_fractions'] if int(args.steps*f)>initial_steps))
    timing_events=[];gpu_memory_samples=[];evaluation_environment_cache={}
    log_fields=['steps','new_steps','elapsed_seconds','steady_training_sps','end_to_end_sps','updates_per_second','training_cumulative_cost','cost_per_1000_transitions','completed_episodes','recent_native_return','recent_episode_cost','recent_goals','recent_at_least_one_goal','peak_torch_memory_mib','learner_updates','actor_updates','geometry_weight','energy','alpha','multiplier']
    progress=args.output/'progress.csv'
    terminal={'completed':False}
    def full_state():
        return dict(learner=learner.state_dict(),replay=replay.state_dict(),environment=env.state_dict(),normalizer=normalizer.state_dict(),
            partial=partial,totals=totals,rollout_rng=rollout_rng.get_state(),rng=capture_rng(),
            steps=steps,method=args.method,seed=args.seed,profile=profile,source_commit=args.source_commit)
    def preserve_interruption():
        if terminal['completed']:return
        # Called only at real process exit, never by observation timeout. Save
        # the exact already-collected prefix without another simulator step.
        atomic_json(args.output/'interruption.json',dict(status='engineering_interruption',
            steps=steps,initial_steps=initial_steps,new_training_transitions=steps-initial_steps,
            source_commit=args.source_commit,automatic_restart=False))
        torch.save(full_state(),args.output/'interrupted_full.pt.tmp')
        (args.output/'interrupted_full.pt.tmp').replace(args.output/'interrupted_full.pt')
    atexit.register(preserve_interruption)
    with progress.open('a',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=log_fields)
        if handle.tell()==0:writer.writeheader()
        log_start=time.monotonic();log_steps=steps;log_updates=learner.updates;first_step_seconds=None
        next_log=steps+max(E,10000//E*E)
        while steps<args.steps:
            sample_timing=args.profile_only and steps>=shared['random_prefill_transitions'] and len(timing_events)<256
            if sample_timing:
                events=[torch.cuda.Event(enable_timing=True) for _ in range(4)];events[0].record()
            action_start=time.monotonic()
            if steps<shared['random_prefill_transitions']:
                action=torch.rand((E,env.action_dim),device=device,generator=rollout_rng)*2-1
            else:
                with torch.no_grad():action,_,_=learner.actor(normalizer.transform(obs),geometry=learner.geometry,generator=rollout_rng)
            if sample_timing:events[1].record()
            nxt,reward,done,info=env.step(action)
            if sample_timing:events[2].record()
            if first_step_seconds is None:
                torch.cuda.synchronize();first_step_seconds=time.monotonic()-action_start
                log_start+=first_step_seconds;log_steps=steps+E
            replay.add(obs=obs,action=action,reward=reward,cost=info['cost'],next_obs=info['terminal_observation'],
                terminated=info['terminated'],truncated=info['truncated'],margin=info['margin'])
            normalizer.update(obs)
            partial['return']+=reward;partial['cost']+=info['cost'];partial['goals']+=info['goals'];partial['length']+=1
            totals[0]+=info['cost'].sum()
            completed_records.append(torch.stack((done,partial['return'],partial['cost'],partial['goals'],partial['length']),-1).detach())
            learner.record_completed_episodes(partial['cost'],done)
            totals[1]+=done.sum();totals[2]+=(partial['return']*done).sum();totals[3]+=(partial['cost']*done).sum()
            totals[4]+=(partial['goals']*done).sum();totals[5]+=((partial['goals']>=1)*done).sum()
            for value in partial.values():value.mul_(~done)
            obs=nxt;steps+=E
            if steps>=shared['random_prefill_transitions'] and replay.size>=B:
                for _ in range(G):learner.update(normalizer.batch(replay.sample(B)))
            if sample_timing:
                events[3].record();timing_events.append(events)
            if steps>=next_log or steps>=args.steps:
                torch.cuda.synchronize();now=time.monotonic();elapsed=now-started
                raw=totals.cpu().tolist(); metrics=learner.metrics()
                records=torch.cat(completed_records,0).cpu().numpy();completed_records=[]
                complete=records[records[:,0]>0]
                with (args.output/'training_episodes.csv').open('a',newline='') as epfile:
                    ew=csv.writer(epfile)
                    if epfile.tell()==0:ew.writerow(['observed_by_steps','native_return','episode_cost','goals','length'])
                    ew.writerows([[steps,*r[1:]] for r in complete])
                row=dict(steps=steps,new_steps=steps-initial_steps,elapsed_seconds=elapsed,
                    steady_training_sps=(steps-log_steps)/max(now-log_start,1e-9),
                    end_to_end_sps=(steps-initial_steps)/max(elapsed,1e-9),
                    updates_per_second=(learner.updates-log_updates)/max(now-log_start,1e-9),
                    training_cumulative_cost=raw[0],cost_per_1000_transitions=1000*raw[0]/steps,
                    completed_episodes=int(raw[1]),recent_native_return=float(complete[:,1].mean()) if len(complete) else None,
                    recent_episode_cost=float(complete[:,2].mean()) if len(complete) else None,
                    recent_goals=float(complete[:,3].mean()) if len(complete) else None,
                    recent_at_least_one_goal=float((complete[:,3]>=1).mean()) if len(complete) else None,
                    peak_torch_memory_mib=torch.cuda.max_memory_allocated()/2**20,
                    learner_updates=learner.updates,actor_updates=learner.actor_updates,
                    **{k:metrics.get(k) for k in ('geometry_weight','energy','alpha','multiplier')})
                writer.writerow(row);handle.flush()
                atomic_json(args.output/'status.json',row|dict(status='running',source_commit=args.source_commit))
                print(json.dumps(row),flush=True)
                try:
                    gpu_memory_samples.append(dict(steps=steps,nvidia_smi=subprocess.check_output(
                        ['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True,timeout=5).strip()))
                except (OSError,subprocess.SubprocessError):
                    gpu_memory_samples.append(dict(steps=steps,nvidia_smi=None))
                log_start=time.monotonic();log_steps=steps;log_updates=learner.updates;next_log=steps+max(E,10000//E*E)
            if checkpoints and steps>=checkpoints[0]:
                checkpoint_started=time.monotonic()
                checkpoints.pop(0)
                t=time.monotonic()
                policy=dict(learner=learner.state_dict(),task=env.effective_config(),steps=steps,method=args.method,seed=args.seed,
                    normalizer=normalizer.state_dict(),action_scaling='native[-1,1]',source_commit=args.source_commit)
                torch.save(policy,args.output/f'policy_{steps}.pt')
                save_seconds+=time.monotonic()-t
                if args.eval_envs:
                    result=evaluate(learner,args.eval_envs,61027,normalizer,evaluation_environment_cache)
                    result.update(checkpoint_steps=steps,training_cumulative_cost=float(totals[0].cpu()),
                        completed_training_episodes=int(totals[1].cpu()),source_commit=args.source_commit)
                    eval_seconds+=result['elapsed_seconds'];eval_controls+=result['physical_controls']
                    atomic_json(args.output/f'validation_{steps}.json',result)
                t=time.monotonic()
                checkpoint_path=args.output/f'full_{steps}.pt'
                torch.save(full_state(),checkpoint_path.with_suffix('.pt.tmp'))
                checkpoint_path.with_suffix('.pt.tmp').replace(checkpoint_path)
                save_seconds+=time.monotonic()-t
                log_start+=time.monotonic()-checkpoint_started
        t=time.monotonic()
        checkpoint=full_state()
        torch.save(checkpoint,args.output/'full_latest.pt.tmp');(args.output/'full_latest.pt.tmp').replace(args.output/'full_latest.pt')
        save_seconds+=time.monotonic()-t
    elapsed=time.monotonic()-started
    torch.cuda.synchronize()
    components={name:float(np.median([e[i].elapsed_time(e[i+1]) for e in timing_events])) if timing_events else None
        for i,name in enumerate(('actor_cuda_ms_per_vector_step','environment_bridge_cuda_ms_per_vector_step','replay_learner_cuda_ms_per_vector_step'))}
    atomic_json(args.output/'receipt.json',dict(status='completed',source_commit=args.source_commit,steps=steps,
        requested_steps=args.steps,initial_steps=initial_steps,new_training_transitions=steps-initial_steps,
        total_training_cumulative_cost=float(totals[0].cpu()),completed_training_episodes=int(totals[1].cpu()),
        evaluation_controls=eval_controls,new_real_robot_interactions=0,learner_updates=learner.updates,
        reward_updates=learner.reward_updates,actor_updates=learner.actor_updates,margin_updates=learner.margin_updates,
        reset_compile_seconds=reset_compile_seconds,first_step_compile_seconds=first_step_seconds,
        elapsed_seconds=elapsed,full_training_including_compile_eval_save_sps=(steps-initial_steps)/elapsed,
        evaluation_seconds=eval_seconds,save_seconds=save_seconds,profile_only=args.profile_only,
        profile_component_events=len(timing_events),profile_components=components,
        total_device_shared_memory_samples=gpu_memory_samples,torch_peak_memory_is_not_total_device_memory=True,
        full_replay_environment_partial_episode_rng_saved=True,
        note='profile-only is engineering training,not a candidate;all actual rounded transitions charged'))
    terminal['completed']=True


if __name__=='__main__':main()

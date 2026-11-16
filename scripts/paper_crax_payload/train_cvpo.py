"""Finite official-CVPO native CRAX run with complete state and fixed checkpoints."""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import time

import numpy as np
import torch

from scripts.paper_crax_payload.cvpo_bridge import (
    UPSTREAM_COMMIT,build_policy,transition_batch,NormalizedReplayView,
    export_full_policy_state,restore_full_policy_state)
from scripts.paper_crax_payload.observation_normalizer import ObservationNormalizer
from scripts.star.arm_safe_entropy.stability_runtime import capture_rng,restore_rng,isolated_rng
from scripts.paper_crax_payload.train_crax import atomic_json
from scripts.paper_crax_payload.cvpo_campaign import validate_registration


FORMAT='official_cvpo_crax_v4_full'


def evaluate(policy,normalizer,env,seed):
    """Complete native episodes, no policy/normalizer/dual learning or early cut."""
    from tianshou.data import Batch
    started=time.monotonic()
    with isolated_rng():
        torch.manual_seed(seed+1)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed+1)
        obs=env.reset(seed=seed);count=env.num_envs
        totals=torch.zeros(count,4,device=policy.device)
        first=torch.full((count,),env.native.episode_length,device=policy.device)
        active=torch.ones(count,dtype=torch.bool,device=policy.device)
        was_training=policy.training;policy.train(False)
        try:
            for i in range(env.native.episode_length):
                with torch.no_grad():action=policy(Batch(obs=normalizer.transform(obs))).act.clamp(-1,1)
                obs,reward,done,info=env.step(action)
                first=torch.where(active&(totals[:,2]==0)&(info['goals']>0),i+1,first)
                totals+=torch.stack([reward,info['cost'],info['goals'],torch.ones_like(reward)],1)*active[:,None]
                active &= ~done
        finally:policy.train(was_training)
        values=torch.cat([totals,first[:,None]],1).cpu().tolist()
    records=[dict(native_return=r[0],episode_cost=r[1],goals=r[2],length=int(r[3]),
        capped_first_goal_seconds=r[4]*float(env.native.dt),
        first_goal_seconds=r[4]*float(env.native.dt) if r[2]>=1 else None) for r in values]
    qualified=[r for r in records if r['episode_cost']<=25]
    return dict(seed=seed,episodes=count,records=records,native_controls=count*env.native.episode_length,
        mean_native_return=float(np.mean([r['native_return'] for r in records])),
        mean_episode_cost=float(np.mean([r['episode_cost'] for r in records])),
        mean_goals_per_episode=float(np.mean([r['goals'] for r in records])),
        mean_capped_first_goal_seconds=float(np.mean([r['capped_first_goal_seconds'] for r in records])),
        at_least_one_goal=float(np.mean([r['goals']>=1 for r in records])),
        budget_compliant_fraction=len(qualified)/len(records),budget_compliant_episodes=len(qualified),
        budget_compliant_return=float(np.mean([r['native_return'] for r in qualified])) if qualified else None,
        native_policy_updates=0,normalizer_updates=0,elapsed_seconds=time.monotonic()-started)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',type=Path)
    args=p.parse_args();config=json.loads(args.config.read_text());started=time.monotonic()
    validate_registration(config)
    if args.seed not in config['training_seeds'] or config['steps_per_seed']!=10000000:
        p.error('Use the fixed three-seed10M registration')
    if args.output.exists() or Path(config['data_root']).resolve() not in args.output.resolve().parents:
        p.error('New native output on the data disk required; never overwrite a prior prefix')
    # Check the existing data root, not the as-yet uncreated runs/ parent.
    if shutil.disk_usage(config['data_root']).free<40*1024**3:raise RuntimeError('Need40GiB data headroom')
    from fsrl.policy.cvpo import CVPO
    from tianshou.data import VectorReplayBuffer,Batch
    from scripts.paper_crax_payload.crax_env import PointGoalTorch
    source=Path(inspect.getfile(CVPO)).resolve()
    commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()
    if (commit!=UPSTREAM_COMMIT or commit!=config['official_source_commit'] or
        not source.is_relative_to(args.upstream.resolve()) or
        hashlib.sha256(source.read_bytes()).hexdigest()!=config['official_cvpo_sha256']):
        raise ValueError('Official fixed algorithm identity mismatch')
    torch.set_num_threads(1);device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(config['torch_cuda_allocator_fraction'],device)
    env=PointGoalTorch(config['num_envs'],args.seed,args.device);obs=env.reset(seed=args.seed)
    native=env.effective_config()
    if native['crax_version_commit']!=config['crax_commit'] or native['obs_dim']!=62 or native['action_dim']!=2:
        raise ValueError('Wrong native environment/observations')
    policy=build_policy(env.obs_dim,env.action_dim,env.native.episode_length,args.device,args.seed,config)
    normalizer=ObservationNormalizer(env.obs_dim,device,config['observation_normalization'])
    replay=VectorReplayBuffer(config['replay_capacity'],config['num_envs'])
    costs=torch.zeros(config['num_envs'],device=device);returns=costs.clone();goals=costs.clone()
    lengths=torch.zeros(config['num_envs'],device=device,dtype=torch.int64)
    steps=updates=episodes=eval_controls=0;training_cost=training_return=0.
    evaluated=[];eval_env=None;previous_elapsed=0.;resume_parent=None
    if args.resume:
        saved=torch.load(args.resume,map_location=device,weights_only=False)
        if (saved['format']!=FORMAT or saved['configuration']!=config or saved['training_seed']!=args.seed
                or saved.get('complete_collection_boundary') is not True):
            raise ValueError('Do not change algorithm/settings/seed when resuming')
        restore_full_policy_state(policy,saved['policy']);normalizer.load_state_dict(saved['normalizer'])
        replay=saved['replay'];obs=env.load_state_dict(saved['environment'])
        steps,updates,episodes=(saved[k] for k in ('steps','updates','episodes'))
        training_cost,training_return=saved['training_cost'],saved['training_return']
        costs,returns,goals,lengths=(saved[k].to(device) for k in ('partial_costs','partial_returns','partial_goals','partial_lengths'))
        evaluated=saved['evaluated_milestones'];eval_controls=saved['evaluation_controls']
        previous_elapsed=saved['elapsed_seconds'];restore_rng(saved['rng'])
        resume_parent=str(args.resume)
    else:policy.pre_update_fn()  # Allocate native dual/optimizer state for early recovery.
    view=NormalizedReplayView(replay,normalizer)
    width=config['num_envs'];target=math.ceil(config['steps_per_seed']/width)*width
    milestones=[math.ceil(s/width)*width for s in config['milestones']]
    args.output.mkdir(parents=True)
    atomic_json(args.output/'run_config.json',dict(configuration=config,training_seed=args.seed,
        format=FORMAT,source_commit=args.source_commit,official_source_commit=commit,
        native_pid=os.getpid(),host=platform.node(),environment=native,resume_parent=resume_parent,
        target_actual_transitions=target,scientific_early_stop=False,physical_robot_interactions=0,
        replay_action='raw_Gaussian_sample_as_official_collector;environment_receives_clipped_action',
        recovery='complete_collection_boundaries_only;in_flight_exception_not_silently_resumed'))
    progress=(args.output/'progress.jsonl').open('x');episode_log=(args.output/'episodes.jsonl').open('x')
    def elapsed():return previous_elapsed+time.monotonic()-started
    last_safe_checkpoint=None
    stop_requested=[]
    def request_stop(signum,frame):
        stop_requested.append(signum)  # Finish the current collect/update block first.
    for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,request_stop)
    def save(name):
        nonlocal last_safe_checkpoint
        if shutil.disk_usage(args.output).free<40*1024**3:raise RuntimeError('Preserve prefix; data headroom below40GiB')
        state=dict(format=FORMAT,configuration=config,training_seed=args.seed,source_commit=args.source_commit,
            policy=export_full_policy_state(policy),replay=replay,normalizer=normalizer.state_dict(),
            environment=env.state_dict(),rng=capture_rng(),steps=steps,updates=updates,episodes=episodes,
            training_cost=training_cost,training_return=training_return,partial_costs=costs,partial_returns=returns,
            partial_goals=goals,partial_lengths=lengths,evaluated_milestones=evaluated,
            evaluation_controls=eval_controls,elapsed_seconds=elapsed(),complete_collection_boundary=True)
        temporary=args.output/(name+'.tmp');torch.save(state,temporary);temporary.replace(args.output/name)
        last_safe_checkpoint=str(args.output/name)
    def heartbeat(state='running'):
        result=dict(state=state,steps=steps,target=target,official_updates=updates,episodes=episodes,
            training_cost=training_cost,training_return=training_return,elapsed_seconds=elapsed(),
            complete_train_sps=(steps if not args.resume else steps-saved['steps'])/max(time.monotonic()-started,1e-9),
            official_qc_threshold=policy.qc_thres[0],evaluated_milestones=evaluated,
            evaluation_controls=eval_controls,native_pid=os.getpid(),automatic_restart=False)
        progress.write(json.dumps(result)+'\n');progress.flush();episode_log.flush()
        atomic_json(args.output/'status.json',result);print(json.dumps(result),flush=True)
    def milestone_evaluation():
        nonlocal eval_env,eval_controls
        if steps not in milestones or steps in evaluated:return
        save(f'full_{steps}.pt')
        policy_path=args.output/f'policy_{steps}.pt'
        torch.save(dict(policy=export_full_policy_state(policy),normalizer=normalizer.state_dict(),
            environment=native,training_seed=args.seed,steps=steps,source_commit=args.source_commit,
            configuration=config),policy_path)
        if eval_env is None:eval_env=PointGoalTorch(config['evaluation_episodes_per_seed'],61027,args.device)
        results=[]
        for seed in config['development_evaluation_seeds']:
            result=evaluate(policy,normalizer,eval_env,seed)
            eval_controls+=result['native_controls'];results.append(result)
        atomic_json(args.output/f'validation_{steps}.json',dict(split='common_validation',
            steps=steps,training_seed=args.seed,results=results,independent_test_used=False,
            policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest()))
        evaluated.append(steps);save('latest_full.pt')
    try:
        save('latest_full.pt')
        # A snapshot saved before its validation must finish that validation on
        # resume, including when the snapshot is already at the final endpoint.
        milestone_evaluation()
        while steps<target:
            boundary=min(target,steps+width*config['collect_vector_steps'],
                next((s for s in milestones if s>steps),target))
            collected=0
            while steps<boundary:
                normalizer.update(obs)
                with torch.no_grad():
                    action=(torch.rand(width,env.action_dim,device=device)*2-1 if steps<config['random_prefill_transitions']
                        else policy(Batch(obs=normalizer.transform(obs))).act)
                # Official FSRL stores RAW policy actions in replay; clipping is
                # part of the black-box environment map, not the critic input.
                following,reward,done,info=env.step(action.clamp(-1,1))
                steps+=width;collected+=width  # Charge every native sample even if replay/logging subsequently fails.
                replay.add(transition_batch(obs,action,reward,info),buffer_ids=np.arange(width))
                costs+=info['cost'];returns+=reward;goals+=info['goals'];lengths+=1
                training_cost+=float(info['cost'].sum());training_return+=float(reward.sum())
                ended=torch.nonzero(done).flatten()
                if len(ended):
                    completed=torch.stack([costs[ended],returns[ended],goals[ended],lengths[ended].float()],1).cpu().tolist()
                    for index,row in zip(ended.cpu().tolist(),completed):
                        episode_log.write(json.dumps(dict(env=index,end_transition=steps,
                            episode_cost=row[0],native_return=row[1],goals=row[2],length=int(row[3])))+'\n')
                    episodes+=len(ended);costs[ended]=0;returns[ended]=0;goals[ended]=0;lengths[ended]=0
                obs=following
            if steps>=config['random_prefill_transitions']:
                policy.pre_update_fn()
                rounds=(collected//width)*config['gradient_updates_per_vector_step']
                for _ in range(rounds):policy.update(config['batch_size'],view);updates+=1
                policy.post_update_fn()
            if not all(torch.isfinite(x).all() for x in policy.parameters()):raise FloatingPointError('Nonfinite official CVPO parameter')
            milestone_evaluation()
            if steps not in milestones and steps//500000!=(steps-collected)//500000:save('latest_full.pt')
            heartbeat()
            if stop_requested and steps<target:
                save('interrupted_full.pt');heartbeat('engineering_stop_at_complete_collection_boundary')
                return  # No completed scientific receipt and no automatic restart.
        save('latest_full.pt')
        atomic_json(args.output/'receipt.json',dict(status='completed',steps=steps,training_seed=args.seed,
            actual_new_transitions=steps-(saved['steps'] if args.resume else 0),training_cost=training_cost,
            episodes=episodes,official_updates=updates,evaluated_milestones=evaluated,
            evaluation_controls=eval_controls,elapsed_seconds=elapsed(),complete_train_sps=steps/elapsed(),
            source_commit=args.source_commit,official_source_commit=commit,new_real_robot_interactions=0,
            independent_test_used=False,component_training_budget_unchanged=True))
        heartbeat('completed')
    except BaseException as error:
        # Mid-update failures can leave partially applied optimizers. Never
        # label such an arbitrary snapshot as a valid complete-state resume.
        atomic_json(args.output/'status.json',dict(state='engineering_interruption',steps=steps,
            error_type=type(error).__name__,error=str(error),automatic_restart=False,
            last_complete_collection_checkpoint=last_safe_checkpoint,
            in_flight_state_recoverable=False,evaluation_controls=eval_controls,
            training_cost=training_cost,official_updates=updates))
        raise
    finally:progress.close();episode_log.close()


if __name__=='__main__':main()

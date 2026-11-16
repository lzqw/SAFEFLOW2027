"""Finite fresh component cohort with all-step compressed physical trajectories."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import time

import numpy as np
import torch

from scripts.paper_crax_payload.component_models import build_component
from scripts.star.arm_safe_entropy.environment import TaskConfig, make_environment
from scripts.star.arm_safe_entropy.stability_runtime import capture_rng, isolated_rng, capture_environment
from scripts.star.arm_safe_entropy.train import Replay, atomic_json


FORMAT = 'component_study_v4_full'


def initial_digest(learner):
    digest=hashlib.sha256()
    for name in ('actor','q','qt','cost_q','cost_qt','margin_online','margin_geometry'):
        for key,value in getattr(learner,name).state_dict().items():
            digest.update((name+key).encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def evaluate(learner, config, milestone, smoke=False):
    plan=config['evaluation']; final=milestone==config['steps_per_run'] and not smoke
    count=plan['independent_episodes_each'] if final else plan['development_episodes_each']
    first=plan['independent_layout_seed_start'] if final else plan['development_layout_seed_start']
    results=[]
    with isolated_rng():
        for ci,condition in enumerate(plan['conditions']):
            env=make_environment(TaskConfig(version=config['task_version'],obstacle_radius=condition['radius']))
            records=[]
            try:
                for i in range(count):
                    seed=first+i
                    torch.manual_seed(plan['action_seed']+ci*10000+i)
                    if torch.cuda.is_available():torch.cuda.manual_seed_all(plan['action_seed']+ci*10000+i)
                    obs,initial=env.reset(seed=seed);minimum=initial['minimum_distance'];total=0.
                    positions=[env.ee_position().tolist()]
                    for step in range(env.config.horizon):
                        action=np.clip(learner.action(obs[None])[0]*condition['action_scale'],-1,1)
                        obs,reward,done,truncated,info=env.step(action)
                        minimum=min(minimum,info['minimum_distance']);total+=reward
                        positions.append(env.ee_position().tolist())
                        if done or truncated:break
                    records.append(dict(layout_seed=seed,safe_success=info['safe_success'],
                        violation=info['violation'],safe_timeout=not info['safe_success'] and not info['violation'],
                        length=step+1,episode_return=total,minimum_distance_m=minimum,trajectory_tcp_xyz=positions,
                        capped_seconds=((step+1) if info['safe_success'] else env.config.horizon)*env.control_dt))
            finally:env.close()
            results.append(dict(condition=condition['name'],episodes=count,
                safe_success=sum(r['safe_success'] for r in records)/count,
                violation=sum(r['violation'] for r in records)/count,
                safe_timeout=sum(r['safe_timeout'] for r in records)/count,
                capped_seconds=float(np.mean([r['capped_seconds'] for r in records])),records=records))
    return dict(milestone=milestone,split='independent_fixed_endpoint' if final else 'development',
        layout_seed_start=first,episodes=sum(r['episodes'] for r in results),conditions=results,
        native_controls=sum(r['length'] for c in results for r in c['records']),
        candidate_reselection=False,new_training_transitions=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--arm',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--cuda-memory-fraction',type=float,default=.025)
    p.add_argument('--smoke-steps',type=int,default=0)
    args=p.parse_args();config=json.loads(args.config.read_text())
    arm=next(a for a in config['arms'] if a['name']==args.arm)
    target=args.smoke_steps or config['steps_per_run']
    if args.output.exists() or target%2 or (args.smoke_steps and not 2<=target<=100):
        p.error('Use a new bounded native output; engineering smoke<=100 actual steps')
    if config['num_envs']!=2 or config['steps_per_run']!=300000 or config['training_seed']!=2027:
        p.error('Use the frozen first-cohort protocol, not an undeclared tuning job')
    if shutil.disk_usage(args.output.parent).free<40*1024**3:
        p.error('Require40GiB data disk headroom')
    torch.set_num_threads(1)
    if torch.device(args.device).type=='cuda':
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction,torch.device(args.device))
    seed=config['training_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    learner=build_component(config,arm['component'],arm['rho'],args.device)
    digest=initial_digest(learner);shared=config['shared_learner']
    if learner.energy_target!=shared['energy_target'] or learner.actor.num_steps!=4:
        raise ValueError('Native units or integration settings drifted')
    args.output.mkdir()
    manifest=dict(format=FORMAT,protocol=config['protocol'],configuration=config,arm=arm,
        source_commit=args.source_commit,config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        native_pid=os.getpid(),host=platform.node(),device=args.device,
        initial_network_sha256=digest,initial_alpha=float(learner.log_alpha.detach().exp().cpu()),
        energy_target=learner.energy_target,initialization='fresh_same_seed_all_networks',
        target_transitions=target,engineering_only=bool(args.smoke_steps),
        software=dict(torch=str(torch.__version__),cuda=torch.version.cuda,numpy=np.__version__),
        trajectory_scope='all initial states and actual transitions, including failures and unfinished episodes',
        cuda_allocator_fraction_cap=args.cuda_memory_fraction,automatic_restart=False,physical_motion=False)
    atomic_json(args.output/'run_config.json',manifest)
    envs=[];steps=episodes=violations=successes=timeouts=evaluation_controls=0
    episode_ids=[0,0];lengths=[0,0];returns=[0.,0.];started=time.monotonic()
    trajectory=gzip.open(args.output/'trajectories.jsonl.gz','wt',compresslevel=3)
    progress=(args.output/'progress.jsonl').open('x')
    def record(value):trajectory.write(json.dumps(value,separators=(',',':'))+'\n')
    def checkpoint(filename):
        value=dict(format=FORMAT,arm=arm,configuration=config,training_seed=seed,
            steps=steps,episodes=episodes,violations=violations,safe_successes=successes,safe_timeouts=timeouts,
            learner=learner.state_dict(),replay=replay.state_dict(),rng=capture_rng(),
            environments=[capture_environment(e) for e in envs],episode_ids=episode_ids,
            episode_lengths=lengths,episode_returns=returns,source_commit=args.source_commit)
        temporary=args.output/(filename+'.tmp');torch.save(value,temporary);temporary.replace(args.output/filename)
    def reset(index):
        obs,initial=envs[index].reset(seed=seed+997*index if episode_ids[index]==0 else None)
        record(dict(record_type='initial',env=index,episode_id=episode_ids[index],transitions=steps,
            observation=obs.tolist(),tcp_xyz=envs[index].ee_position().tolist(),
            start=envs[index].start.tolist(),goal=envs[index].goal.tolist(),
            obstacle=envs[index].obstacle_position.tolist(),minimum_distance=initial['minimum_distance']))
        return obs
    try:
        envs=[make_environment(TaskConfig(version=config['task_version'])) for _ in range(2)]
        replay=Replay(shared['replay_capacity'],25)
        observations=np.stack([reset(i) for i in range(2)])
        atomic_json(args.output/'task_definition.json',envs[0].definition())
        while steps<target:
            learner.set_transitions(steps)
            actions=(np.random.uniform(-1,1,(2,3)).astype(np.float32) if steps<shared['random_transitions']
                     else learner.action(observations))
            for index,env in enumerate(envs):
                action=np.clip(actions[index],-1,1)
                obs,reward,done,truncated,info=env.step(action)
                replay.add(obs=observations[index],action=action,reward=reward,next_obs=obs,
                    done=float(done or truncated),cost=info['cost'],margin=info['local_margin'])
                steps+=1  # Count immediately before any reset can fail.
                lengths[index]+=1;returns[index]+=reward;violations+=int(info['violation'])
                record(dict(record_type='step',env=index,episode_id=episode_ids[index],transitions=steps,
                    observation=obs.tolist(),tcp_xyz=env.ee_position().tolist(),executed_action=action.tolist(),
                    raw_action=actions[index].tolist(),reward=reward,cost=info['cost'],margin=info['local_margin'],
                    minimum_distance=info['minimum_distance'],done=bool(done or truncated),
                    safe_success=info['safe_success'],violation=info['violation'],
                    collision=info['collision'],true_horizon_terminal=info['true_horizon_terminal']))
                if done or truncated:
                    episodes+=1;successes+=int(info['safe_success'])
                    timeouts+=int(not info['safe_success'] and not info['violation'])
                    learner.record_episode(info['cost'])
                    episode_ids[index]+=1;lengths[index]=0;returns[index]=0.
                    obs=reset(index)
                observations[index]=obs
            learner.set_transitions(steps)
            if steps>shared['random_transitions'] and replay.count>=shared['batch_size']:
                for _ in range(2):learner.update(replay.sample(shared['batch_size'],learner.device,recent=True))
            if steps%1000==0 or steps==target:
                elapsed=time.monotonic()-started
                status=dict(state='running',steps=steps,episodes=episodes,training_violations=violations,
                    safe_successes=successes,safe_timeouts=timeouts,elapsed_seconds=elapsed,
                    complete_train_sps=steps/max(elapsed,1e-9),**learner.metrics())
                if not all(np.isfinite(v) for k,v in status.items() if k!='state'):
                    raise FloatingPointError('Nonfinite diagnostic, preserve prefix and stop')
                progress.write(json.dumps(status)+'\n');progress.flush();trajectory.flush()
                atomic_json(args.output/'status.json',status);print(json.dumps(status),flush=True)
            if not args.smoke_steps and steps in config['evaluation']['milestones']:
                checkpoint(f'full_{steps}.pt')
                before=(learner.updates,learner.actor_updates,learner.safety_multiplier)
                result=evaluate(learner,config,steps)
                if before!=(learner.updates,learner.actor_updates,learner.safety_multiplier):
                    raise ValueError('Evaluation changed learner counters')
                evaluation_controls+=result['native_controls']
                atomic_json(args.output/f'evaluation_{steps}.json',result)
            if steps%25000==0 or steps==target:checkpoint('latest_full.pt')
        record(dict(record_type='end',transitions=steps,unfinished_episode_lengths=lengths,
                    unfinished_episodes_not_counted_as_terminal=True))
        trajectory.close();progress.close()
        atomic_json(args.output/'receipt.json',dict(status='completed',arm=arm,steps=steps,
            engineering_only=bool(args.smoke_steps),training_seed=seed,initial_network_sha256=digest,
            episodes=episodes,training_violations=violations,safe_successes=successes,safe_timeouts=timeouts,
            learner_updates=learner.updates,reward_updates=learner.reward_updates,actor_updates=learner.actor_updates,
            elapsed_seconds=time.monotonic()-started,evaluation_controls=evaluation_controls,
            native_training_controls=steps,native_training_physics_substeps=steps*20,
            reset_and_static_admission_work_not_in_substep_counter=True,
            all_step_trajectories_retained=True,independent_endpoint_test_executed=not bool(args.smoke_steps),
            source_commit=args.source_commit,new_training_transitions=steps,physical_robot_interactions=0))
        atomic_json(args.output/'status.json',dict(state='completed',steps=steps))
    except BaseException as error:
        if 'replay' in locals() and envs:checkpoint('interrupted_full.pt')
        atomic_json(args.output/'status.json',dict(state='engineering_interruption',steps=steps,
            error_type=type(error).__name__,error=str(error),automatic_restart=False))
        raise
    finally:
        trajectory.close();progress.close()
        for env in envs:env.close()


if __name__=='__main__':main()

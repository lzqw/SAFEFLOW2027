"""Independent bounded EA-STAB runner; legacy frozen-main checks are unchanged."""
import argparse
from collections import deque
from dataclasses import asdict
import csv
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from scripts.star.arm_safe_entropy.environment import TaskConfig,make_environment
from scripts.star.arm_safe_entropy.train import Replay,atomic_json
from scripts.star.arm_safe_entropy.stability_runtime import (
    build_learner,capture_rng,restore_rng,capture_environment,restore_environment)
from scripts.star.arm_safe_entropy.stability_evaluation import evaluate_stability


FORMAT='ea_stability_v2_full'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--variant',choices=['s_iso','s_local','s_ea'],required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--steps',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--num-envs',type=int,default=2)
    parser.add_argument('--evaluation-workers',type=int,default=10)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--engineering-recovery',action='store_true',
        help='Explicit one-time continuation of an interrupted initial300k trial;requires a terminal incident record')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    configuration=json.loads(args.config.read_text())
    shared=configuration['shared_learner']
    if args.output.exists(): parser.error('Never overwrite another native trial')
    if args.seed not in configuration['development_seeds'] and not args.smoke:
        parser.error('Formal confirmation is conditional and is not accepted by this development runner')
    if args.steps%args.num_envs or args.num_envs!=2:
        parser.error('Retain two training environments and actual-transition accounting')
    if not args.smoke and args.steps not in (300000,1000000):
        parser.error('Initial development300k or declared native long continuation1M only')
    if args.steps==1000000 and not args.resume:
        parser.error('Development long runs continue a native300k full checkpoint,not a fresh extra1M')
    if args.engineering_recovery and (not args.resume or args.steps!=300000 or args.smoke):
        parser.error('Engineering recovery only completes an interrupted initial300k native trial')
    if args.smoke and args.steps>10000: parser.error('Engineering smoke is bounded separately')
    if (configuration['task_version']!='SafeReach-RelativeLayout-v3' or
            configuration['training_radius_m']!=.06):
        parser.error('Do not change the training task/radius')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    task=TaskConfig(version=configuration['task_version'])
    envs=[make_environment(task) for _ in range(args.num_envs)]
    learner=build_learner(configuration,args.variant,args.device)
    replay=Replay(shared['replay_capacity'],25)
    steps=violations=episodes=safe_successes=safe_timeouts=0
    evaluation_transitions=0
    prior_elapsed=0.
    episode_returns=np.zeros(args.num_envs)
    episode_lengths=np.zeros(args.num_envs,dtype=np.int64)
    recent=deque(maxlen=100)
    saved=None
    if args.resume:
        saved=torch.load(args.resume,map_location=args.device,weights_only=False)
        if (saved.get('format')!=FORMAT or saved['variant']!=args.variant or
                saved['training_seed']!=args.seed or saved['configuration']!=configuration):
            raise ValueError('Resume only the matching native full trial,not a policy/foreign replay pair')
        if args.engineering_recovery:
            incident_path=args.resume.parent/'engineering_interruption.json'
            incident=json.loads(incident_path.read_text())
            if (incident.get('native_process_confirmed_terminal') is not True or
                    incident.get('recovery_checkpoint_steps')!=saved['steps'] or
                    not 0<saved['steps']<300000 or (args.resume.parent/'receipt.json').exists()):
                raise ValueError('Recovery requires the declared matching full checkpoint of a terminal incomplete trial')
            if Path('/proc',str(incident['native_pid']),'cmdline').exists():
                command=Path('/proc',str(incident['native_pid']),'cmdline').read_bytes()
                if b'train_stability' in command:
                    raise ValueError('Do not duplicate a still-live native training process')
        elif not args.smoke and (saved['steps']!=300000 or args.steps!=1000000):
            raise ValueError('Only the predeclared300k-to1M continuation is currently authorized')
        learner.load_state_dict(saved['learner']);replay.load_state_dict(saved['replay'])
        for key in ('steps','violations','episodes','safe_successes','safe_timeouts','evaluation_transitions'):
            if key not in saved: raise ValueError('Full checkpoint is missing accounting')
        steps,violations,episodes=saved['steps'],saved['violations'],saved['episodes']
        safe_successes,safe_timeouts=saved['safe_successes'],saved['safe_timeouts']
        evaluation_transitions=saved['evaluation_transitions']
        prior_elapsed=saved['cumulative_elapsed_seconds']
        episode_returns=saved['episode_returns'].copy()
        episode_lengths=saved['episode_lengths'].copy()
        recent=deque(saved['recent'],maxlen=100)
        # Reconstruction does not reset a partial episode or advance simulation.
        observations=np.stack([restore_environment(env,state) for env,state in zip(envs,saved['environments'])])
        restore_rng(saved['rng'])
    else:
        observations=np.stack([env.reset(seed=args.seed+997*i)[0] for i,env in enumerate(envs)])
    initial_steps=steps
    initial_evaluation_transitions=evaluation_transitions
    if steps>=args.steps: raise ValueError('Continuation must add actual transitions')
    learner.set_transitions(steps)
    manifest=dict(stage='ea_stability_v2_engineering' if args.smoke else 'ea_stability_v2_development',
        variant=args.variant,method=learner.method,training_seed=args.seed,
        source_commit=args.source_commit,baseline_commit=configuration['baseline_commit'],
        configuration=configuration,task_config=asdict(task),task_definition=envs[0].definition(),
        total_transition_budget=args.steps,initial_steps=steps,
        initialization='native_engineering_recovery' if args.engineering_recovery else 'native_full_continuation' if saved else 'fresh',
        engineering_recovery=args.engineering_recovery,
        initial_training_violations=violations,initial_training_episodes=episodes,
        initial_physical_evaluation_transitions=initial_evaluation_transitions,
        parent_checkpoint=str(args.resume) if args.resume else None,
        resume_semantics='learner_replay_optimizer_RNG_partial_episodes_and_observed_physical_state;not_bitwise_solver_cache',
        num_envs=args.num_envs,device=args.device,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        host=platform.node(),gpu=torch.cuda.get_device_name(learner.device) if learner.device.type=='cuda' else None,
        software=dict(torch=str(torch.__version__),cuda=torch.version.cuda,numpy=np.__version__,
                      **{name:version(name) for name in ('panda-gym','pybullet','gymnasium')}),
        scientific_early_stopping=False,automatic_restart_or_extension=False,
        training_step_definition='actual sum of train env.step calls;warmup and preheat included',
        evaluation_rng_isolated=True)
    atomic_json(args.output/'run_config.json',manifest)
    fields=['steps','elapsed_seconds','segment_elapsed_seconds','train_fps','segment_train_fps',
        'episodes','training_violation_count','training_safe_successes','training_safe_timeouts',
        'recent_success','recent_violation_rate','recent_episode_length','recent_return',
        *learner.METRIC_NAMES,'safety_multiplier','geometry_beta','learner_updates','reward_updates',
        'actor_updates','transitions','warmup_safety_episodes']
    stream=(args.output/'progress.csv').open('w',newline='')
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    started=time.monotonic()
    last_log=last_checkpoint=steps
    milestones=[] if args.smoke else [s for s in configuration['evaluation']['intermediate_steps']+[1000000]
                                      if steps<s<=args.steps]

    def checkpoint(destination='latest_full.pt'):
        payload=dict(format=FORMAT,variant=args.variant,training_seed=args.seed,configuration=configuration,
            source_commit=args.source_commit,learner=learner.state_dict(),replay=replay.state_dict(),
            steps=steps,violations=violations,episodes=episodes,safe_successes=safe_successes,
            safe_timeouts=safe_timeouts,evaluation_transitions=evaluation_transitions,
            cumulative_elapsed_seconds=prior_elapsed+time.monotonic()-started,
            episode_returns=episode_returns.copy(),episode_lengths=episode_lengths.copy(),recent=list(recent),
            environments=[capture_environment(env) for env in envs],rng=capture_rng())
        temporary=args.output/(destination+'.tmp')
        torch.save(payload,temporary);temporary.replace(args.output/destination)

    try:
        while steps<args.steps:
            learner.set_transitions(steps)
            actions=(np.random.uniform(-1,1,size=(args.num_envs,3)).astype(np.float32)
                     if steps<shared['random_transitions'] else learner.action(observations))
            for index,env in enumerate(envs):
                observation,reward,done,truncated,info=env.step(actions[index])
                replay.add(obs=observations[index],action=actions[index],reward=reward,next_obs=observation,
                    done=float(done or truncated),cost=info['cost'],margin=info['local_margin'])
                violations+=int(info['violation'])
                episode_returns[index]+=reward;episode_lengths[index]+=1
                if done or truncated:
                    safe_successes+=int(info['safe_success'])
                    safe_timeouts+=int(not info['safe_success'] and not info['violation'])
                    recent.append((float(episode_returns[index]),float(info['safe_success']),
                                   float(info['violation']),int(episode_lengths[index])))
                    learner.record_episode(info['cost'])
                    episodes+=1;episode_returns[index]=0.;episode_lengths[index]=0
                    observation,_=env.reset()
                observations[index]=observation
            steps+=args.num_envs
            learner.set_transitions(steps)
            if steps>shared['random_transitions'] and replay.count>=shared['batch_size']:
                for _ in range(args.num_envs):
                    learner.update(replay.sample(shared['batch_size'],learner.device,recent=True))
            if steps-last_log>=1000 or steps==args.steps:
                elapsed=time.monotonic()-started
                row=dict(steps=steps,elapsed_seconds=prior_elapsed+elapsed,segment_elapsed_seconds=elapsed,
                    train_fps=steps/(prior_elapsed+elapsed),segment_train_fps=(steps-initial_steps)/elapsed,
                    episodes=episodes,training_violation_count=violations,
                    training_safe_successes=safe_successes,training_safe_timeouts=safe_timeouts,
                    recent_success=float(np.mean([r[1] for r in recent])) if recent else 0.,
                    recent_violation_rate=float(np.mean([r[2] for r in recent])) if recent else 0.,
                    recent_episode_length=float(np.mean([r[3] for r in recent])) if recent else 0.,
                    recent_return=float(np.mean([r[0] for r in recent])) if recent else 0.,**learner.metrics())
                if not all(np.isfinite(value) for value in row.values()):
                    raise FloatingPointError('Nonfinite native training diagnostic')
                writer.writerow(row);stream.flush()
                atomic_json(args.output/'status.json',dict(state='running',**row))
                print(json.dumps(row),flush=True);last_log=steps
            if milestones and steps==milestones[0]:
                milestone=milestones.pop(0)
                result=evaluate_stability(learner,configuration,args.seed,steps,args.evaluation_workers)
                result.update(training_violation_count=violations,training_episodes=episodes,
                              source_commit=args.source_commit)
                evaluation_transitions+=result['physical_evaluation_transitions']
                atomic_json(args.output/f'evaluation_{milestone}.json',result)
                torch.save(learner.state_dict(),args.output/f'policy_{milestone}.pt')
                if milestone in (300000,600000,1000000):
                    checkpoint(f'full_{milestone}.pt')
            if steps-last_checkpoint>=10000 or steps==args.steps:
                checkpoint();last_checkpoint=steps
        atomic_json(args.output/'receipt.json',dict(status='completed',variant=args.variant,training_seed=args.seed,
            steps=steps,initial_steps=initial_steps,newly_collected_transitions=steps-initial_steps,
            training_violation_count=violations,training_episodes=episodes,training_safe_successes=safe_successes,
            training_safe_timeouts=safe_timeouts,physical_evaluation_transitions=evaluation_transitions,
            new_physical_evaluation_transitions=evaluation_transitions-initial_evaluation_transitions,
            segment_elapsed_seconds=time.monotonic()-started,cumulative_elapsed_seconds=prior_elapsed+time.monotonic()-started,
            learner_updates=learner.updates,reward_updates=learner.reward_updates,actor_updates=learner.actor_updates,
            source_commit=args.source_commit,final_evaluation=None if args.smoke else f'evaluation_{steps}.json'))
        atomic_json(args.output/'status.json',dict(state='completed',steps=steps))
    except BaseException as error:
        checkpoint('interrupted_full.pt')
        atomic_json(args.output/'status.json',dict(state='engineering_interruption',steps=steps,
                    error_type=type(error).__name__,error=str(error),automatic_restart=False))
        raise
    finally:
        stream.close()
        for env in envs: env.close()


if __name__=='__main__':
    main()

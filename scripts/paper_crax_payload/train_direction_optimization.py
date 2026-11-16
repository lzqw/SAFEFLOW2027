"""Registered v7 full-state development forks, safe resumes, exact paid budgets.

No confirmation or independent testing can be invoked through this entry.
Existing v5/v6 entry points and checkpoint interpretation remain unchanged.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import time

import numpy as np
import torch

from scripts.paper_crax_payload.direction_optimization import FORMAT, build, counts, validate_config, validate_parent
from scripts.paper_crax_payload.direction_annealing import fixed_probe, gradient_diagnosis, sha256
from scripts.paper_crax_payload.evaluate_direction_annealing import evaluate
from scripts.star.arm_safe_entropy.stability_runtime import capture_rng, restore_rng, capture_environment, restore_environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--arm', required=True)
    origin = p.add_mutually_exclusive_group(required=True)
    origin.add_argument('--fork-parent', type=Path); origin.add_argument('--resume', type=Path)
    p.add_argument('--parent-run', type=Path)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--source-commit', required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--cuda-memory-fraction', type=float, default=.025)
    args = p.parse_args(); config = json.loads(args.config.read_text()); validate_config(config)
    from scripts.star.arm_safe_entropy.environment import TaskConfig, make_environment
    from scripts.star.arm_safe_entropy.train import Replay, atomic_json
    candidate = next(a for a in config['candidates'] if a['name'] == args.arm)
    target = config['absolute_target_steps']; start = config['parents'][candidate['parent']]['steps']
    reservation = sum(target-config['parents'][a['parent']]['steps'] for a in config['candidates'])
    if reservation > config['budget']['development_ceiling'] or not 0 < args.cuda_memory_fraction <= .15:
        p.error('Exceeds registered development reservation or engineering memory cap')
    if args.fork_parent and (not args.parent_run or args.output.exists()):
        p.error('Explicit full-parent fork requires parent-run and previously absent output')
    if args.resume and (args.parent_run or not args.output.exists()):
        p.error('Resume only own same-configuration safe boundary, never a new branch')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < 40*1024**3:
        p.error('Preserve40GiB data-disk reserve, never fall back to home')
    torch.set_num_threads(1)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, torch.device(args.device))
    learner = build(config, candidate, args.device)
    loaded = torch.load(args.fork_parent or args.resume, map_location=args.device, weights_only=False)
    evaluations = []; diagnosis_steps = []; evaluation_controls = 0; prior_elapsed = 0.
    if args.fork_parent:
        if args.fork_parent.resolve().parent != args.parent_run.resolve():
            raise ValueError('Parent full checkpoint must belong to its original native run')
        manifest_path = args.parent_run/'run_config.json'
        parent_manifest = json.loads(manifest_path.read_text())
        common = validate_parent(loaded,parent_manifest,args.fork_parent,manifest_path,config,candidate)
        learner.import_parent(loaded['learner'])
        probe = loaded['probe'] if start == 150000 else fixed_probe(loaded['replay'],config)
        parent = dict(checkpoint_sha256=sha256(args.fork_parent),manifest_sha256=sha256(manifest_path),
            checkpoint=str(args.fork_parent.resolve()),manifest=str(manifest_path.resolve()),
            source_commit=loaded['source_commit'],arm=loaded['arm'],prefix_at_fork=counts(loaded),
            common100K=common,common100K_sha256=config['common_parent100K_sha256'],
            inherited_episode_lengths=loaded['lengths'],inherited_episode_ids=loaded['episode_ids'],
            identical_prefix_algorithm_checked=True,observed_development_parent=True)
        args.output.mkdir()
        inherited_evidence = []
        if start == 150000:
            # full150K was written before150K DEV. Reuse the actual native eval
            # from its run; isolated DEV does not alter the training RNG/state.
            for step in config['evaluation']['milestones']:
                if step > start: continue
                source = args.parent_run/f'evaluation_{step}.json'
                value = json.loads(source.read_text())
                if value['milestone'] != step or value['independent_confirmation']:
                    raise ValueError('Reuse only exact paired development prefix evidence')
                shutil.copyfile(source,args.output/source.name); evaluations.append(step)
                inherited_evidence.append(dict(path=str(source),sha256=sha256(source)))
            with (args.parent_run/'gradient_diagnostics.jsonl').open() as stream:
                gradient_prefix = [json.loads(line) for line in stream]
            with (args.output/'gradient_diagnostics.jsonl').open('x') as stream:
                for row in gradient_prefix:
                    if row['absolute_steps'] <= start:
                        if row['probe_sha256'] != probe['sha256']:
                            raise ValueError('Never replace the common fixed64 probe')
                        stream.write(json.dumps(row,allow_nan=False)+'\n'); diagnosis_steps.append(row['absolute_steps'])
        atomic_json(args.output/'run_config.json',dict(format=FORMAT,configuration=config,candidate=candidate,
            parent=parent,native_pid=os.getpid(),host=platform.node(),source_commit=args.source_commit,
            training_seed=2028,absolute_target_steps=target,actual_fork_steps=start,
            reserved_actual_training_steps=target-start,comparative_interval_start=100000,
            probe_sha256=probe['sha256'],inherited_evidence=inherited_evidence,
            engineering_only=False,physical_robot_interactions=0,automatic_retry=False,
            inherited_episodes_not_reset=True,cross_hardware_bitwise_equivalence_claimed=False))
    else:
        status = json.loads((args.output/'status.json').read_text())
        manifest = json.loads((args.output/'run_config.json').read_text())
        if (args.resume.resolve() != (args.output/'interrupted_full.pt').resolve()
                or status['state'] != 'stopped_at_safe_boundary' or status['steps'] != loaded['steps']
                or loaded['format'] != FORMAT or loaded['configuration'] != config
                or loaded['candidate'] != candidate or loaded['source_commit'] != args.source_commit
                or loaded['steps'] >= target or manifest['parent'] != loaded['parent']
                or manifest['configuration'] != config or manifest['candidate'] != candidate):
            raise ValueError('Only identical v7 signal-safe native resume, never arbitrary errors')
        learner.load_state_dict(loaded['learner']); parent = loaded['parent']; common = parent['common100K']
        probe = loaded['probe']; evaluations = loaded['evaluations']; diagnosis_steps = loaded['diagnosis_steps']
        evaluation_controls = loaded['evaluation_controls']; prior_elapsed = loaded['elapsed_seconds']
    steps, episodes, violations, successes, timeouts = (
        loaded[k] for k in ('steps','episodes','violations','successes','timeouts'))
    episode_ids, lengths, returns = (list(loaded[k]) for k in ('episode_ids','lengths','returns'))
    started = time.monotonic(); stop_requested = False; envs = []
    trajectory = gzip.open(args.output/'trajectories.jsonl.gz','at' if args.resume else 'xt',compresslevel=3)
    progress = (args.output/'progress.jsonl').open('a' if args.resume else 'x')
    gradient_log = (args.output/'gradient_diagnostics.jsonl').open('a' if start==150000 or args.resume else 'x')
    def elapsed(): return prior_elapsed+time.monotonic()-started
    def record(row): trajectory.write(json.dumps(row,separators=(',',':'))+'\n')
    def reset(index):
        obs, initial = envs[index].reset()
        record(dict(record_type='initial',env=index,episode_id=episode_ids[index],transitions=steps,
            observation=obs.tolist(),tcp_xyz=envs[index].ee_position().tolist(),start=envs[index].start.tolist(),
            goal=envs[index].goal.tolist(),obstacle=envs[index].obstacle_position.tolist(),minimum_distance=initial['minimum_distance']))
        return obs
    def checkpoint(name):
        trajectory.flush(); progress.flush(); gradient_log.flush()
        state = dict(format=FORMAT,configuration=config,candidate=candidate,source_commit=args.source_commit,
            seed=2028,parent=parent,probe=probe,steps=steps,episodes=episodes,violations=violations,
            successes=successes,timeouts=timeouts,episode_ids=episode_ids.copy(),lengths=lengths.copy(),
            returns=returns.copy(),learner=learner.state_dict(),replay=replay.state_dict(),rng=capture_rng(),
            environments=[capture_environment(e) for e in envs],observations=observations.copy(),
            evaluations=evaluations.copy(),diagnosis_steps=diagnosis_steps.copy(),evaluation_controls=evaluation_controls,
            elapsed_seconds=elapsed(),actual_training_transitions=steps-start)
        temporary = args.output/(name+'.tmp'); torch.save(state,temporary); temporary.replace(args.output/name)
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,request_stop)
    try:
        envs = [make_environment(TaskConfig(version=config['task_version'])) for _ in range(2)]
        replay = Replay(config['shared_learner']['replay_capacity'],25); replay.load_state_dict(loaded['replay'])
        observations = np.stack([restore_environment(e,s) for e,s in zip(envs,loaded['environments'])])
        if not np.allclose(observations,loaded['observations'],rtol=0,atol=2e-6):
            raise ValueError('Full prefix observed state mismatch; never reset inherited episodes')
        restore_rng(loaded['rng']); del loaded
        atomic_json(args.output/'task_definition.json',envs[0].definition())
        record(dict(record_type='resume' if args.resume else 'fork',transitions=steps,
            native_pid=os.getpid(),host=platform.node(),parent_sha256=parent['checkpoint_sha256']))
        if args.fork_parent:
            for index,env in enumerate(envs):
                record(dict(record_type='initial',env=index,episode_id=episode_ids[index],transitions=steps,
                    inherited_partial_episode_length=lengths[index],no_reset_at_fork=True,
                    observation=observations[index].tolist(),tcp_xyz=env.ee_position().tolist(),
                    start=env.start.tolist(),goal=env.goal.tolist(),obstacle=env.obstacle_position.tolist()))
            atomic_json(args.output/'restoration_receipt.json',dict(parent=parent,absolute_steps=steps,
                learner_updates=learner.updates,actor_updates=learner.actor_updates,reward_updates=learner.reward_updates,
                replay_count=replay.count,replay_cursor=replay.cursor,all_models_and_optimizers_restored=True,
                rng_restored=True,environments_and_partial_episodes_restored=True,no_simulation_controls_for_restore=True,
                solver_cache_bitwise_equivalence_claimed=False))
        if steps not in diagnosis_steps:
            gradient_log.write(json.dumps(gradient_diagnosis(learner,probe),allow_nan=False)+'\n')
            gradient_log.flush(); diagnosis_steps.append(steps)
        while steps < target and not stop_requested:
            learner.set_transitions(steps); actions = learner.action(observations)
            for index,env in enumerate(envs):
                action = np.clip(actions[index],-1.,1.)
                obs,reward,done,truncated,info = env.step(action)
                replay.add(obs=observations[index],action=action,reward=reward,next_obs=obs,
                    done=float(done or truncated),cost=info['cost'],margin=info['local_margin'])
                steps += 1; lengths[index] += 1; returns[index] += reward; violations += int(info['violation'])
                record(dict(record_type='step',env=index,episode_id=episode_ids[index],transitions=steps,
                    observation=obs.tolist(),tcp_xyz=env.ee_position().tolist(),executed_action=action.tolist(),
                    reward=reward,cost=info['cost'],margin=info['local_margin'],minimum_distance=info['minimum_distance'],
                    done=bool(done or truncated),safe_success=info['safe_success'],violation=info['violation']))
                if done or truncated:
                    episodes += 1; successes += int(info['safe_success'])
                    timeouts += int(not info['safe_success'] and not info['violation'])
                    learner.record_episode(info['cost']); episode_ids[index] += 1; lengths[index] = 0; returns[index] = 0.
                    obs = reset(index)
                observations[index] = obs
            learner.set_transitions(steps)
            for _ in range(2): learner.update(replay.sample(config['shared_learner']['batch_size'],learner.device,recent=True))
            if steps % 1000 == 0 or steps == target:
                status = dict(state='running',steps=steps,episodes=episodes,training_violations=violations,
                    safe_successes=successes,safe_timeouts=timeouts,actual_training_transitions=steps-start,
                    comparative_training_transitions=steps-100000,
                    comparative_safe_successes=successes-common['safe_successes'],
                    comparative_training_violations=violations-common['training_violations'],
                    comparative_safe_timeouts=timeouts-common['safe_timeouts'],elapsed_seconds=elapsed(),**learner.metrics())
                if not all(np.isfinite(v) for k,v in status.items() if k!='state'):
                    raise FloatingPointError('Nonfinite v7 training metrics')
                progress.write(json.dumps(status)+'\n'); progress.flush(); trajectory.flush()
                atomic_json(args.output/'status.json',status); print(json.dumps(status),flush=True)
            if steps%10000==0 and steps not in diagnosis_steps:
                gradient_log.write(json.dumps(gradient_diagnosis(learner,probe),allow_nan=False)+'\n')
                gradient_log.flush(); diagnosis_steps.append(steps)
            if steps in config['evaluation']['milestones'] and steps not in evaluations:
                checkpoint(f'full_{steps}.pt')
                value=evaluate(learner,config,steps); value.update(split='v7_development_only',candidate=candidate['name'])
                evaluation_controls+=value['native_controls']; atomic_json(args.output/f'evaluation_{steps}.json',value)
                evaluations.append(steps)
            if steps%25000==0 or steps==target: checkpoint('latest_full.pt')
        if stop_requested and steps<target:
            checkpoint('interrupted_full.pt')
            atomic_json(args.output/'status.json',dict(state='stopped_at_safe_boundary',steps=steps,automatic_retry=False))
            return
        record(dict(record_type='end',transitions=steps,new_transitions=steps-start,
            unfinished_episode_lengths=lengths,unfinished_episodes_not_counted_as_terminal=True))
        atomic_json(args.output/'receipt.json',dict(status='completed',format=FORMAT,candidate=candidate,parent=parent,
            configuration_sha256=sha256(args.config),source_commit=args.source_commit,training_seed=2028,steps=steps,
            actual_training_transitions=steps-start,comparative_training_transitions=steps-100000,
            safe_successes=successes,training_violations=violations,safe_timeouts=timeouts,episodes=episodes,
            comparative_safe_successes=successes-common['safe_successes'],
            comparative_training_violations=violations-common['training_violations'],
            comparative_safe_timeouts=timeouts-common['safe_timeouts'],learner_updates=learner.updates,
            actor_updates=learner.actor_updates,reward_updates=learner.reward_updates,elapsed_seconds=elapsed(),
            evaluations=evaluations,diagnosis_steps=diagnosis_steps,evaluation_controls=evaluation_controls,
            probe_sha256=probe['sha256'],engineering_only=False,independent_endpoint_test_executed=False,
            all_actual_suffix_steps_retained=True,physical_robot_interactions=0))
        atomic_json(args.output/'status.json',dict(state='completed',steps=steps,actual_training_transitions=steps-start))
    except BaseException as error:
        atomic_json(args.output/'status.json',dict(state='engineering_interruption',steps=steps,
            error_type=type(error).__name__,error=str(error),automatic_retry=False))
        raise
    finally:
        trajectory.close(); progress.close(); gradient_log.close()
        for env in envs: env.close()


if __name__=='__main__': main()

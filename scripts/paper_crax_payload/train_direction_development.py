"""Finite v5 development runner: no independent test, no automatic retry."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import time

import numpy as np
import torch

from scripts.paper_crax_payload.direction_development import build_development
from scripts.star.arm_safe_entropy.stability_runtime import (
    capture_rng, restore_rng, capture_environment, restore_environment)


FORMAT = 'direction_development_v5_full'


def development_evaluation(learner, config, steps):
    from scripts.paper_crax_payload.train_component_study import evaluate
    # The old helper's smoke=True explicitly forces its development branch,
    # including at the last milestone. No independent seed is even configured.
    value = evaluate(learner, config, steps, smoke=True)
    value.update(split='development_only', candidate_selection_permitted=True,
                 independent_confirmation=False)
    return value


def validate_config(config):
    if (config['protocol'] != 'direction_development_v5_normal_shrink_outward'
            or config['num_envs'] != 2 or config['steps_per_run'] != 150000
            or config['new_development_training_budget'] !=
            len(config['arms'])*len(config['development_training_seeds'])*config['steps_per_run']
            or config['evaluation']['split'] != 'development_only_even_at_final'
            or any('independent' in key for key in config['evaluation'])):
        raise ValueError('Use the registered finite development-only protocol')
    expected = dict(reference_variance=.04, energy_target=18.75, flow_steps=4, width=128,
                    utd=1, reward_updates_per_round=2, target_samples=4)
    if any(config['shared_learner'][k] != v for k, v in expected.items()):
        raise ValueError('Do not silently change implicit native learner constants')
    if config['geometry'] != dict(ema_tau=.005, ramp_start=10000, ramp_end=30000):
        raise ValueError('Use the registered reliability/ramp without additional tuning')


def main(*, validator=validate_config, builder=build_development, format_name=FORMAT,
         evaluator=development_evaluation, tracker_factory=None, replay_factory=None):
    from scripts.paper_crax_payload.train_component_study import initial_digest
    from scripts.star.arm_safe_entropy.environment import TaskConfig, make_environment
    from scripts.star.arm_safe_entropy.train import Replay, atomic_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--arm', required=True); parser.add_argument('--seed', type=int, default=2028)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cuda-memory-fraction', type=float, default=.08)
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(); config = json.loads(args.config.read_text()); validator(config)
    arm = next(a for a in config['arms'] if a['name'] == args.arm)
    target = args.smoke_steps or config['steps_per_run']
    if (target % 2 or (args.smoke_steps and not 2 <= target <= 100)
            or not 0 < args.cuda_memory_fraction <= .15): parser.error('Invalid bounded engineering/allocator budget')
    if args.output.exists() and not args.resume: parser.error('Preserve old output; resume explicitly')
    if not args.output.exists() and args.resume: parser.error('Resume only into the original run directory')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < 40*1024**3: parser.error('Require40GiB data disk reserve')
    torch.set_num_threads(1)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, torch.device(args.device))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    learner = builder(config, arm, args.seed, args.device)
    digest = initial_digest(learner); shared = config['shared_learner']
    steps = episodes = violations = successes = timeouts = evaluation_controls = 0
    episode_ids = [0, 0]; lengths = [0, 0]; returns = [0., 0.]
    evaluations = []; prior_elapsed = 0.; started = time.monotonic(); stop_requested = False
    tracker = tracker_factory(2) if tracker_factory else None
    resume = None
    if args.resume:
        status = json.loads((args.output/'status.json').read_text())
        if (args.resume.resolve() != (args.output/'interrupted_full.pt').resolve()
                or status['state'] != 'stopped_at_safe_boundary'):
            raise ValueError('Resume requires the explicit signal-safe boundary; arbitrary stale-prefix replay is forbidden')
        resume = torch.load(args.resume, map_location=args.device, weights_only=False)
        if (resume['format'] != format_name or resume['configuration'] != config or resume['arm'] != arm
                or resume['seed'] != args.seed or resume['source_commit'] != args.source_commit
                or resume['steps'] != status['steps'] or resume['steps'] >= target or args.smoke_steps):
            raise ValueError('Resume the same incomplete native scientific job, not a revision')
        learner.load_state_dict(resume['learner'])
        steps, episodes, violations, successes, timeouts, evaluation_controls = (
            resume[k] for k in ('steps', 'episodes', 'violations', 'successes', 'timeouts', 'evaluation_controls'))
        episode_ids, lengths, returns = (resume[k] for k in ('episode_ids', 'lengths', 'returns'))
        evaluations = resume['evaluations']; prior_elapsed = resume['elapsed_seconds']
        if tracker is not None:
            tracker.load_state_dict(resume['training_tracker'])
    else:
        args.output.mkdir()
        atomic_json(args.output/'run_config.json', dict(format=format_name, configuration=config, arm=arm,
            source_commit=args.source_commit, config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
            native_pid=os.getpid(), host=platform.node(), device=args.device, training_seed=args.seed,
            initial_network_sha256=digest, target_transitions=target, engineering_only=bool(args.smoke_steps),
            cuda_allocator_fraction_cap=args.cuda_memory_fraction, torch=str(torch.__version__),
            numpy=np.__version__, evaluation_scope='development_only_including_final',
            automatic_retry=False, physical_robot_interactions=0))
    trajectory = gzip.open(args.output/'trajectories.jsonl.gz', 'at' if resume else 'wt', compresslevel=3)
    progress = (args.output/'progress.jsonl').open('a' if resume else 'x')
    envs = []; replay = None
    def record(value): trajectory.write(json.dumps(value, separators=(',', ':'))+'\n')
    def reset(index):
        obs, initial = envs[index].reset(seed=args.seed+997*index if episode_ids[index] == 0 else None)
        record(dict(record_type='initial', env=index, episode_id=episode_ids[index], transitions=steps,
            observation=obs.tolist(), tcp_xyz=envs[index].ee_position().tolist(), start=envs[index].start.tolist(),
            goal=envs[index].goal.tolist(), obstacle=envs[index].obstacle_position.tolist(),
            minimum_distance=initial['minimum_distance']))
        return obs
    def elapsed(): return prior_elapsed+time.monotonic()-started
    def checkpoint(name):
        trajectory.flush(); progress.flush()
        state = dict(format=format_name, configuration=config, arm=arm, seed=args.seed, source_commit=args.source_commit,
            learner=learner.state_dict(), replay=replay.state_dict(), rng=capture_rng(),
            environments=[capture_environment(e) for e in envs], observations=observations.copy(),
            steps=steps, episodes=episodes, violations=violations, successes=successes, timeouts=timeouts,
            evaluation_controls=evaluation_controls, evaluations=evaluations.copy(),
            episode_ids=episode_ids.copy(), lengths=lengths.copy(), returns=returns.copy(), elapsed_seconds=elapsed())
        if tracker is not None:
            state['training_tracker'] = tracker.state_dict()
        temporary = args.output/(name+'.tmp'); torch.save(state, temporary); temporary.replace(args.output/name)
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True  # Save only at a complete collect/update/eval boundary.
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, request_stop)
    try:
        envs = [make_environment(TaskConfig(version=config['task_version'])) for _ in range(2)]
        replay = (replay_factory(config, arm, shared['replay_capacity'], 25)
                  if replay_factory else Replay(shared['replay_capacity'], 25))
        if resume:
            replay.load_state_dict(resume['replay'])
            observations = np.stack([restore_environment(e, s) for e, s in zip(envs, resume['environments'])])
            if not np.allclose(observations, resume['observations'], rtol=0, atol=2e-6):
                raise ValueError('Observed native state mismatch')
            restore_rng(resume['rng']); del resume
            record(dict(record_type='resume', transitions=steps, native_pid=os.getpid(), host=platform.node()))
        else:
            observations = np.stack([reset(i) for i in range(2)])
            atomic_json(args.output/'task_definition.json', envs[0].definition())
        while steps < target and not stop_requested:
            learner.set_transitions(steps)
            actions = (np.random.uniform(-1, 1, (2, 3)).astype(np.float32) if steps < shared['random_transitions']
                       else learner.action(observations))
            for index, env in enumerate(envs):
                action = np.clip(actions[index], -1, 1)
                obs, reward, done, truncated, info = env.step(action)
                replay.add(obs=observations[index], action=action, reward=reward, next_obs=obs,
                    done=float(done or truncated), cost=info['cost'], margin=info['local_margin'])
                if replay_factory:
                    replay.observe_transition(index, bool(done or truncated), info)
                steps += 1; lengths[index] += 1; returns[index] += reward; violations += int(info['violation'])
                if tracker is not None:
                    tracker.observe(index, reward, bool(done or truncated), info)
                record(dict(record_type='step', env=index, episode_id=episode_ids[index], transitions=steps,
                    observation=obs.tolist(), tcp_xyz=env.ee_position().tolist(), executed_action=action.tolist(),
                    reward=reward, cost=info['cost'], margin=info['local_margin'], minimum_distance=info['minimum_distance'],
                    done=bool(done or truncated), safe_success=info['safe_success'], violation=info['violation']))
                if done or truncated:
                    episodes += 1; successes += int(info['safe_success'])
                    timeouts += int(not info['safe_success'] and not info['violation'])
                    learner.record_episode(info['cost']); episode_ids[index] += 1; lengths[index] = 0; returns[index] = 0.
                    obs = reset(index)
                observations[index] = obs
            learner.set_transitions(steps)
            if steps > shared['random_transitions'] and replay.count >= shared['batch_size']:
                for _ in range(2): learner.update(replay.sample(shared['batch_size'], learner.device, recent=True))
            if steps % 1000 == 0 or steps == target:
                status = dict(state='running', steps=steps, episodes=episodes, training_violations=violations,
                    safe_successes=successes, safe_timeouts=timeouts, elapsed_seconds=elapsed(),
                    complete_train_sps=steps/max(elapsed(), 1e-9), **learner.metrics())
                if tracker is not None:
                    status.update(tracker.metrics())
                if replay_factory:
                    status.update(replay.metrics())
                if not all(np.isfinite(v) for k, v in status.items() if k != 'state'):
                    raise FloatingPointError('Nonfinite training diagnostic')
                progress.write(json.dumps(status)+'\n'); progress.flush(); trajectory.flush()
                atomic_json(args.output/'status.json', status); print(json.dumps(status), flush=True)
            if not args.smoke_steps and steps in config['evaluation']['milestones'] and steps not in evaluations:
                checkpoint(f'full_{steps}.pt')
                value = evaluator(learner, config, steps)
                evaluation_controls += value['native_controls']; evaluations.append(steps)
                atomic_json(args.output/f'evaluation_{steps}.json', value)
            if steps % 25000 == 0 or steps == target: checkpoint('latest_full.pt')
        if stop_requested and steps < target:
            checkpoint('interrupted_full.pt')
            atomic_json(args.output/'status.json', dict(state='stopped_at_safe_boundary', steps=steps, automatic_retry=False))
            return
        record(dict(record_type='end', transitions=steps, unfinished_episode_lengths=lengths,
                    unfinished_episodes_not_counted_as_terminal=True))
        receipt = dict(status='completed', arm=arm, steps=steps,
            engineering_only=bool(args.smoke_steps), training_seed=args.seed, initial_network_sha256=digest,
            source_commit=args.source_commit, episodes=episodes, training_violations=violations,
            safe_successes=successes, safe_timeouts=timeouts, learner_updates=learner.updates,
            actor_updates=learner.actor_updates, reward_updates=learner.reward_updates,
            elapsed_seconds=elapsed(), evaluation_controls=evaluation_controls, evaluations=evaluations,
            independent_endpoint_test_executed=False, all_step_trajectories_retained=True,
            new_training_transitions=steps, physical_robot_interactions=0)
        if tracker is not None:
            receipt['training_metrics'] = tracker.metrics()
        if replay_factory:
            receipt['replay_metrics'] = replay.metrics()
        atomic_json(args.output/'receipt.json', receipt)
        atomic_json(args.output/'status.json', dict(state='completed', steps=steps))
    except BaseException as error:
        # An arbitrary exception may occur mid-collection; do not pretend it is
        # a resumable full boundary. Preserve the last complete periodic save.
        atomic_json(args.output/'status.json', dict(state='engineering_interruption', steps=steps,
            error_type=type(error).__name__, error=str(error), automatic_retry=False))
        raise
    finally:
        trajectory.close(); progress.close()
        for env in envs: env.close()


if __name__ == '__main__': main()

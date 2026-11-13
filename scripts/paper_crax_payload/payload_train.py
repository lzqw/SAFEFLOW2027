"""Finite new Payload training, retained endpoints and full native recovery.

Only this runner constructs Payload. Historical Reach runs/configs are never
patched. Evaluation is frozen stochastic behavior, separately seeded and with
all RNG state restored afterwards. No deployment or physical robot interface.
"""
import argparse
from collections import deque
from dataclasses import asdict
import json
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from scripts.star.arm_safe_entropy.train import Replay, atomic_json
from scripts.star.arm_safe_entropy.stability_runtime import capture_rng, restore_rng, isolated_rng
from scripts.paper_crax_payload.payload_environment import (
    PayloadConfig, SafePayloadTransport, capture_payload_environment, restore_payload_environment)
from scripts.paper_crax_payload.payload_models import build_payload_learner


FORMAT = 'paper_payload_v1_full'


def evaluate_payload(learner, config, episodes, seed_start=None, conditions=None):
    plan = config['evaluation']
    seed_start = plan['development_layout_seed_start'] if seed_start is None else seed_start
    results = []
    with isolated_rng():
        for condition_index, condition in enumerate(conditions or plan['conditions']):
            torch.manual_seed(plan['action_sampling_seed'] + condition_index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(plan['action_sampling_seed'] + condition_index)
            task = dict(config['task']) | condition.get('task_overrides', {})
            if 'obstacle_offset' in condition:
                task['obstacle_offset'] = condition['obstacle_offset']
            env = SafePayloadTransport(PayloadConfig(**task))
            records = []
            try:
                for index in range(episodes):
                    seed = seed_start + index
                    observation, initial = env.reset(seed=seed)
                    total = 0.
                    minimum = initial['minimum_distance']
                    trajectory = []
                    for step in range(env.config.horizon):
                        raw_action = learner.action(observation[None])[0]
                        action = np.clip(raw_action * condition.get('action_scale', 1.), -1., 1.)
                        observation, reward, done, truncated, info = env.step(action)
                        trajectory.append(env.payload_position().tolist())
                        total += reward
                        minimum = min(minimum, info['minimum_distance'])
                        if done or truncated:
                            break
                    records.append(dict(layout_seed=seed, episode_return=total, length=step + 1,
                        minimum_distance_m=minimum, trajectory_payload_xyz=trajectory,
                        time_to_goal_seconds=(step+1)*env.control_dt if info['safe_success'] else None,
                        time_to_goal_capped_seconds=((step+1) if info['safe_success'] else env.config.horizon)*env.control_dt,
                        **{k: info[k] for k in ('safe_success', 'violation', 'collision', 'arm_collision',
                            'payload_collision', 'clearance_violation', 'true_horizon_terminal')}))
            finally:
                env.close()
            successful = [r['time_to_goal_seconds'] for r in records if r['safe_success']]
            summary = dict(condition=condition['name'], episodes=episodes,
                safe_success_count=sum(r['safe_success'] for r in records),
                violation_count=sum(r['violation'] for r in records),
                safe_timeout_count=sum(not r['safe_success'] and not r['violation'] for r in records),
                physical_evaluation_transitions=sum(r['length'] for r in records),
                time_to_goal_seconds=float(np.mean(successful)) if successful else None,
                **{key: float(np.mean([r[key] for r in records])) for key in
                   ('safe_success', 'violation', 'collision', 'arm_collision', 'payload_collision',
                    'episode_return', 'minimum_distance_m', 'time_to_goal_capped_seconds')}, records=records)
            if summary['safe_success_count'] + summary['violation_count'] + summary['safe_timeout_count'] != episodes:
                raise ValueError('Every Payload episode needs exactly one terminal class')
            results.append(summary)
    return dict(task='SafePayloadTransport-v1', conditions=results,
        training_transitions=0, adaptation_updates=0, training_rng_preserved=True,
        physical_evaluation_transitions=sum(r['physical_evaluation_transitions'] for r in results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--method', choices=['sac_lag', 's_iso', 's_ea'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--phase', choices=['engineering', 'development', 'confirmation'], default='development')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.output.exists():
        parser.error('Preserve every original run; use a new output directory')
    if args.phase == 'development' and args.seed not in config['development_seeds']:
        parser.error('Use predeclared development seeds')
    if args.phase == 'confirmation' and args.seed not in config['confirmation_seeds']:
        parser.error('Use predeclared confirmation seeds')
    if args.phase == 'engineering' and args.steps > 10000:
        parser.error('Engineering smoke cannot exceed10000 actual transitions')
    if args.steps > 1000000:
        parser.error('Payload per-trial target cannot exceed1M')
    num_envs = config['num_envs']
    if args.steps % num_envs:
        parser.error('Exact transition budget must be divisible by the environment batch')
    torch.set_num_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    learner = build_payload_learner(config, args.method, args.device)
    shared = config['shared_learner']
    envs = [SafePayloadTransport(PayloadConfig(**config['task'])) for _ in range(num_envs)]
    replay = Replay(shared['replay_capacity'], 45)
    steps = violations = episodes = successes = timeouts = evaluation_transitions = 0
    episode_returns = np.zeros(num_envs)
    episode_costs = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs, dtype=np.int64)
    recent = deque(maxlen=100)
    previous_elapsed = 0.
    if args.resume:
        saved = torch.load(args.resume, map_location=args.device, weights_only=False)
        if (saved.get('format') != FORMAT or saved['configuration'] != config or
                saved['method'] != args.method or saved['training_seed'] != args.seed):
            raise ValueError('Only an identical native Payload full checkpoint can continue')
        prior = json.loads((args.resume.parent/'run_config.json').read_text())
        proc = Path('/proc', str(prior['native_pid']), 'cmdline')
        if prior['host'] == platform.node() and proc.exists():
            command = proc.read_bytes()
            if b'payload_train' in command and str(args.resume.parent).encode() in command:
                raise ValueError('Do not duplicate a live native Payload run')
        learner.load_state_dict(saved['learner']); replay.load_state_dict(saved['replay'])
        steps, violations, episodes = saved['steps'], saved['violations'], saved['episodes']
        successes, timeouts = saved['safe_successes'], saved['safe_timeouts']
        evaluation_transitions = saved['evaluation_transitions']
        episode_returns, episode_costs, episode_lengths = [saved[k].copy() for k in
            ('episode_returns', 'episode_costs', 'episode_lengths')]
        recent = deque(saved['recent'], maxlen=100)
        previous_elapsed = saved['cumulative_elapsed_seconds']
        observations = np.stack([restore_payload_environment(e, s) for e, s in zip(envs, saved['environments'])])
        restore_rng(saved['rng'])
    else:
        observations = np.stack([env.reset(seed=args.seed + 997*i)[0] for i, env in enumerate(envs)])
    if steps >= args.steps:
        raise ValueError('Continuation must add actual transitions')
    learner.set_transitions(steps)
    initial_steps, initial_evaluation = steps, evaluation_transitions
    args.output.mkdir(parents=True)
    manifest = dict(protocol=config['protocol'], task='SafePayloadTransport-v1', phase=args.phase,
        method=args.method, training_seed=args.seed, source_commit=args.source_commit,
        configuration=config, task_definition=envs[0].definition(),
        native_pid=os.getpid(), host=platform.node(), device=args.device,
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        initialization='native_full_continuation' if args.resume else 'fresh',
        parent_checkpoint=str(args.resume) if args.resume else None,
        initial_steps=steps, total_transition_budget=args.steps,
        cost_mode='episodic_binary', episode_budget=learner.cost_limit,
        full_resume='learner_optimizers_replay_RNG_partial_episodes_robot_and_payload_observed_state;not_bitwise_solver_cache',
        training_transitions_definition='sum of actual env.step calls including random/preheat',
        scientific_early_stopping=False, automatic_restart=False,
        physical_hardware_used=False, software=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
            numpy=np.__version__, **{name: version(name) for name in ('panda-gym', 'pybullet', 'gymnasium')}))
    manifest['source_files_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('payload_environment.py', 'payload_models.py', 'payload_train.py')}
    atomic_json(args.output/'run_config.json', manifest)
    started = time.monotonic()
    last_log = last_checkpoint = steps
    milestones = [] if args.phase == 'engineering' else sorted(set(
        int(args.steps * fraction) for fraction in config['model_selection']['checkpoint_fractions']
        if steps < int(args.steps * fraction) <= args.steps))
    if any(s % num_envs for s in milestones):
        raise ValueError('Candidate checkpoint fractions must land on exact vector boundaries')
    stream = (args.output/'progress.jsonl').open('w')

    def checkpoint(filename='latest_full.pt'):
        state = dict(format=FORMAT, configuration=config, method=args.method, training_seed=args.seed,
            source_commit=args.source_commit, learner=learner.state_dict(), replay=replay.state_dict(),
            steps=steps, violations=violations, episodes=episodes, safe_successes=successes,
            safe_timeouts=timeouts, evaluation_transitions=evaluation_transitions,
            episode_returns=episode_returns.copy(), episode_costs=episode_costs.copy(),
            episode_lengths=episode_lengths.copy(), recent=list(recent),
            environments=[capture_payload_environment(env) for env in envs], rng=capture_rng(),
            cumulative_elapsed_seconds=previous_elapsed + time.monotonic() - started)
        temporary = args.output/(filename + '.tmp')
        torch.save(state, temporary); temporary.replace(args.output/filename)

    try:
        while steps < args.steps:
            actions = (np.random.uniform(-1., 1., (num_envs, 3)).astype(np.float32)
                if steps < shared['random_transitions'] else learner.action(observations))
            for index, env in enumerate(envs):
                observation, reward, done, truncated, info = env.step(actions[index])
                replay.add(obs=observations[index], action=actions[index], reward=reward,
                    next_obs=observation, done=float(done or truncated), cost=info['cost'], margin=info['local_margin'])
                violations += int(info['violation'])
                episode_returns[index] += reward; episode_costs[index] += info['cost']; episode_lengths[index] += 1
                if done or truncated:
                    successes += int(info['safe_success'])
                    timeouts += int(not info['safe_success'] and not info['violation'])
                    recent.append(dict(episode_return=float(episode_returns[index]),
                        cost=float(episode_costs[index]), length=int(episode_lengths[index]),
                        safe_success=bool(info['safe_success'])))
                    learner.record_episode(episode_costs[index])
                    episodes += 1
                    episode_returns[index] = episode_costs[index] = episode_lengths[index] = 0
                    observation, _ = env.reset()
                observations[index] = observation
            steps += num_envs
            learner.set_transitions(steps)
            if steps > shared['random_transitions'] and replay.count >= shared['batch_size']:
                for _ in range(num_envs * shared['learner_updates_per_transition']):
                    learner.update(replay.sample(shared['batch_size'], learner.device, recent=True))
            if steps - last_log >= config['log_every'] or steps == args.steps:
                elapsed = time.monotonic() - started
                row = dict(steps=steps, episodes=episodes, training_violation_count=violations,
                    training_safe_successes=successes, training_safe_timeouts=timeouts,
                    recent_success=float(np.mean([r['safe_success'] for r in recent])) if recent else 0.,
                    recent_episode_length=float(np.mean([r['length'] for r in recent])) if recent else 0.,
                    recent_return=float(np.mean([r['episode_return'] for r in recent])) if recent else 0.,
                    cumulative_elapsed_seconds=previous_elapsed+elapsed, segment_elapsed_seconds=elapsed,
                    complete_train_sps=(steps-initial_steps)/max(elapsed, 1e-9), **learner.metrics())
                if not all(np.isfinite(value) for value in row.values()):
                    raise FloatingPointError('Nonfinite Payload learning diagnostic')
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                atomic_json(args.output/'status.json', dict(state='running', **row))
                print(json.dumps(row), flush=True); last_log = steps
            if milestones and steps == milestones[0]:
                milestone = milestones.pop(0)
                count = config['evaluation']['confirmation_episodes_per_condition'] if args.phase == 'confirmation' else config['evaluation']['development_episodes_per_condition']
                result = evaluate_payload(learner, config, count)
                result.update(method=args.method, training_seed=args.seed, checkpoint_steps=steps,
                    fixed_endpoint=steps==args.steps, training_violation_count=violations, training_episodes=episodes)
                evaluation_transitions += result['physical_evaluation_transitions']
                atomic_json(args.output/f'evaluation_{milestone}.json', result)
                torch.save(learner.state_dict(), args.output/f'policy_{milestone}.pt')
                checkpoint(f'full_{milestone}.pt')
            if steps - last_checkpoint >= config['checkpoint_every'] or steps == args.steps:
                checkpoint(); last_checkpoint = steps
        atomic_json(args.output/'receipt.json', dict(status='completed', method=args.method,
            training_seed=args.seed, phase=args.phase, steps=steps, initial_steps=initial_steps,
            newly_collected_transitions=steps-initial_steps, training_violation_count=violations,
            training_episodes=episodes, training_safe_successes=successes, training_safe_timeouts=timeouts,
            new_physical_evaluation_transitions=evaluation_transitions-initial_evaluation,
            physical_evaluation_transitions=evaluation_transitions,
            segment_elapsed_seconds=time.monotonic()-started,
            cumulative_elapsed_seconds=previous_elapsed+time.monotonic()-started,
            learner_updates=learner.updates, reward_updates=learner.reward_updates, actor_updates=learner.actor_updates,
            source_commit=args.source_commit))
        atomic_json(args.output/'status.json', dict(state='completed', steps=steps))
    except BaseException as error:
        checkpoint('interrupted_full.pt')
        atomic_json(args.output/'status.json', dict(state='engineering_interruption', steps=steps,
            error_type=type(error).__name__, error=str(error), automatic_restart=False))
        raise
    finally:
        stream.close()
        for env in envs:
            env.close()


if __name__ == '__main__':
    main()

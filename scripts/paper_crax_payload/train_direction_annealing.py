"""Explicit full-state v5->v6 fork, NOT a bypass of original resume checks.

Copies no old trace or replay to local/GitHub. Each branch restores identical
parent models/targets, optimizers, replay/cursor, RNG, environments and partial
episodes. New interactions are exactly absolute steps minus100K.
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

from scripts.paper_crax_payload.direction_annealing import (
    FORMAT, build, fixed_probe, gradient_diagnosis, sha256, validate_config, validate_parent)
from scripts.paper_crax_payload.evaluate_direction_annealing import evaluate
from scripts.star.arm_safe_entropy.stability_runtime import capture_rng, restore_rng, capture_environment, restore_environment


def main():
    from scripts.star.arm_safe_entropy.environment import TaskConfig, make_environment
    from scripts.star.arm_safe_entropy.train import Replay, atomic_json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--arm', required=True)
    origin = p.add_mutually_exclusive_group(required=True)
    origin.add_argument('--fork-parent', type=Path); origin.add_argument('--resume', type=Path)
    p.add_argument('--parent-manifest', type=Path); p.add_argument('--parent-sha256')
    p.add_argument('--output', type=Path, required=True); p.add_argument('--source-commit', required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--cuda-memory-fraction', type=float, default=.025)
    p.add_argument('--engineering-steps', type=int, default=0)
    args = p.parse_args(); config = json.loads(args.config.read_text()); validate_config(config)
    arm = next(a for a in config['arms'] if a['name'] == args.arm)
    if (not 0 < args.cuda_memory_fraction <= .15
            or args.engineering_steps not in (0, 8) or args.resume and args.engineering_steps):
        p.error('Only original150K suffix or separately accounted8-step engineering probe')
    if args.fork_parent and (not args.parent_manifest or not args.parent_sha256 or args.output.exists()):
        p.error('Explicit fork requires original parent+manifest+SHA and previously absent output')
    if args.resume and (not args.output.exists() or args.parent_manifest or args.parent_sha256):
        p.error('Resume only own v6 safe-boundary directory; do not combine with parent import')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < 40*1024**3: p.error('Require40GiB data-disk reserve')
    torch.set_num_threads(1)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, torch.device(args.device))
    learner = build(config, arm, args.device)
    target = 100000+(args.engineering_steps or 150000)
    loaded = torch.load(args.fork_parent or args.resume, map_location=args.device, weights_only=False)
    if args.fork_parent:
        manifest = json.loads(args.parent_manifest.read_text())
        if (args.parent_sha256 != config['parent']['checkpoint_sha256']
                or sha256(args.parent_manifest) != config['parent']['manifest_sha256']):
            raise ValueError('Parent SHA must equal the registered common full100K identity')
        prefix = validate_parent(loaded, manifest, args.parent_sha256, args.fork_parent)
        # Same learner constants, not merely same tensor shapes.
        if (loaded['configuration']['shared_learner'] != config['shared_learner']
                or loaded['configuration']['geometry'] != config['geometry']
                or loaded['configuration']['task_version'] != config['task_version']):
            raise ValueError('No undeclared task/learner/geometry changes in the fork')
        learner.load_parent_state(loaded['learner'])
        probe = fixed_probe(loaded['replay'], config)
        parent = dict(checkpoint_sha256=args.parent_sha256, manifest_sha256=sha256(args.parent_manifest),
            checkpoint=str(args.fork_parent.resolve()), manifest=str(args.parent_manifest.resolve()),
            source_commit=loaded['source_commit'], initial_network_sha256=manifest['initial_network_sha256'],
            fork_kind='explicit_new_development_branch_not_same_configuration_resume',
            observed_parent_selection=True, prefix=prefix,
            inherited_environment_lengths=loaded['lengths'], inherited_episode_ids=loaded['episode_ids'])
        evaluations = []; diagnosis_steps = []; evaluation_controls = 0; prior_elapsed = 0.
        args.output.mkdir()
        atomic_json(args.output/'run_config.json', dict(format=FORMAT, configuration=config, arm=arm,
            parent=parent, native_pid=os.getpid(), host=platform.node(), source_commit=args.source_commit,
            training_seed=2028, target_transitions=target, new_target_transitions=target-100000,
            parent_checkpoint_sha256=parent['checkpoint_sha256'], probe_sha256=probe['sha256'],
            config_sha256=sha256(args.config), engineering_only=bool(args.engineering_steps),
            device=args.device, cuda_allocator_fraction_cap=args.cuda_memory_fraction,
            automatic_retry=False, physical_robot_interactions=0,
            state_portability='observed_markov_state_not_solver_cache_bitwise_equivalence'))
    else:
        status = json.loads((args.output/'status.json').read_text())
        manifest = json.loads((args.output/'run_config.json').read_text())
        if (args.resume.resolve() != (args.output/'interrupted_full.pt').resolve()
                or status['state'] != 'stopped_at_safe_boundary' or status['steps'] != loaded['steps']
                or loaded['format'] != FORMAT or loaded['configuration'] != config or loaded['arm'] != arm
                or loaded['source_commit'] != args.source_commit or loaded['steps'] >= target
                or manifest['engineering_only'] or manifest['configuration'] != config
                or manifest['parent'] != loaded['parent']):
            raise ValueError('Same v6 schedule/config/source and explicit signal-safe state required for resume')
        learner.load_state_dict(loaded['learner']); parent = loaded['parent']; prefix = parent['prefix']
        probe = loaded['probe']; evaluations = loaded['evaluations']; diagnosis_steps = loaded['diagnosis_steps']
        evaluation_controls = loaded['evaluation_controls']; prior_elapsed = loaded['elapsed_seconds']
    steps = loaded['steps']; episodes = loaded['episodes']; violations = loaded['violations']
    successes = loaded['successes']; timeouts = loaded['timeouts']
    episode_ids = list(loaded['episode_ids']); lengths = list(loaded['lengths']); returns = list(loaded['returns'])
    started = time.monotonic(); stop_requested = False; envs = []
    trajectory = gzip.open(args.output/'trajectories.jsonl.gz', 'at' if args.resume else 'xt', compresslevel=3)
    progress = (args.output/'progress.jsonl').open('a' if args.resume else 'x')
    gradient_log = (args.output/'gradient_diagnostics.jsonl').open('a' if args.resume else 'x')
    def elapsed(): return prior_elapsed+time.monotonic()-started
    def record(row): trajectory.write(json.dumps(row, separators=(',', ':'))+'\n')
    def reset(index):
        obs, initial = envs[index].reset()  # Original per-environment RNG restored, no reseeding.
        record(dict(record_type='initial', env=index, episode_id=episode_ids[index], transitions=steps,
            observation=obs.tolist(), tcp_xyz=envs[index].ee_position().tolist(), start=envs[index].start.tolist(),
            goal=envs[index].goal.tolist(), obstacle=envs[index].obstacle_position.tolist(),
            minimum_distance=initial['minimum_distance']))
        return obs
    def checkpoint(name):
        trajectory.flush(); progress.flush(); gradient_log.flush()
        state = dict(format=FORMAT, configuration=config, arm=arm, source_commit=args.source_commit,
            seed=2028, parent=parent, probe=probe, steps=steps, new_training_transitions=steps-100000,
            episodes=episodes, violations=violations, successes=successes, timeouts=timeouts,
            episode_ids=episode_ids.copy(), lengths=lengths.copy(), returns=returns.copy(),
            learner=learner.state_dict(), replay=replay.state_dict(), rng=capture_rng(),
            environments=[capture_environment(e) for e in envs], observations=observations.copy(),
            evaluations=evaluations.copy(), diagnosis_steps=diagnosis_steps.copy(),
            evaluation_controls=evaluation_controls, elapsed_seconds=elapsed())
        tmp = args.output/(name+'.tmp'); torch.save(state, tmp); tmp.replace(args.output/name)
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, request_stop)
    try:
        envs = [make_environment(TaskConfig(version=config['task_version'])) for _ in range(2)]
        replay = Replay(config['shared_learner']['replay_capacity'], 25); replay.load_state_dict(loaded['replay'])
        observations = np.stack([restore_environment(e, s) for e,s in zip(envs, loaded['environments'])])
        if not np.allclose(observations, loaded['observations'], rtol=0, atol=2e-6):
            raise ValueError('Parent observed/environment state mismatch; do not reset partial episodes')
        restore_rng(loaded['rng']); del loaded
        atomic_json(args.output/'task_definition.json', envs[0].definition())
        record(dict(record_type='resume' if args.resume else 'fork', transitions=steps,
            native_pid=os.getpid(), host=platform.node(), parent_sha256=parent['checkpoint_sha256']))
        if args.fork_parent:
            for index, env in enumerate(envs):
                record(dict(record_type='initial', env=index, episode_id=episode_ids[index], transitions=steps,
                    observation=observations[index].tolist(), tcp_xyz=env.ee_position().tolist(),
                    start=env.start.tolist(), goal=env.goal.tolist(), obstacle=env.obstacle_position.tolist(),
                    inherited_partial_episode_length=lengths[index], no_reset_at_fork=True))
            atomic_json(args.output/'restoration_receipt.json', dict(parent=parent,
                observed_state_restored=True, replay_count=replay.count, replay_cursor=replay.cursor,
                rng_restored=True, all_learner_state_restored=True, no_environment_controls_for_restore=True,
                solver_warm_start_cache_bitwise_equivalence_claimed=False,
                learner_updates=learner.updates, actor_updates=learner.actor_updates,
                reward_updates=learner.reward_updates, absolute_steps=steps))
        if not args.engineering_steps and steps == 100000:
            gradient_log.write(json.dumps(gradient_diagnosis(learner, probe), allow_nan=False)+'\n')
            gradient_log.flush(); diagnosis_steps.append(steps)
        while steps < target and not stop_requested:
            learner.set_transitions(steps); actions = learner.action(observations)
            for index, env in enumerate(envs):
                action = np.clip(actions[index], -1., 1.)
                obs, reward, done, truncated, info = env.step(action)
                replay.add(obs=observations[index], action=action, reward=reward, next_obs=obs,
                    done=float(done or truncated), cost=info['cost'], margin=info['local_margin'])
                steps += 1; lengths[index] += 1; returns[index] += reward; violations += int(info['violation'])
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
            for _ in range(2): learner.update(replay.sample(config['shared_learner']['batch_size'], learner.device, recent=True))
            if steps % 1000 == 0 or steps == target:
                status = dict(state='running', steps=steps, new_training_transitions=steps-100000,
                    episodes=episodes, training_violations=violations, safe_successes=successes, safe_timeouts=timeouts,
                    new_safe_successes=successes-prefix['safe_successes'],
                    new_training_violations=violations-prefix['training_violations'],
                    new_safe_timeouts=timeouts-prefix['safe_timeouts'], elapsed_seconds=elapsed(),
                    suffix_train_sps=(steps-100000)/max(elapsed(), 1e-9), **learner.metrics())
                if not all(np.isfinite(v) for k,v in status.items() if k != 'state'):
                    raise FloatingPointError('Nonfinite v6 training metrics')
                progress.write(json.dumps(status)+'\n'); progress.flush(); trajectory.flush()
                atomic_json(args.output/'status.json', status); print(json.dumps(status), flush=True)
            if not args.engineering_steps and steps % 10000 == 0 and steps not in diagnosis_steps:
                gradient_log.write(json.dumps(gradient_diagnosis(learner, probe), allow_nan=False)+'\n')
                gradient_log.flush(); diagnosis_steps.append(steps)
            if not args.engineering_steps and steps in config['evaluation']['milestones'] and steps not in evaluations:
                checkpoint(f'full_{steps}.pt')
                value = evaluate(learner, config, steps); evaluation_controls += value['native_controls']
                atomic_json(args.output/f'evaluation_{steps}.json', value); evaluations.append(steps)
            if steps % 25000 == 0 or steps == target: checkpoint('latest_full.pt')
        if stop_requested and steps < target:
            checkpoint('interrupted_full.pt')
            atomic_json(args.output/'status.json', dict(state='stopped_at_safe_boundary', steps=steps, automatic_retry=False))
            return
        record(dict(record_type='end', transitions=steps, new_transitions=steps-100000,
            unfinished_episode_lengths=lengths, unfinished_episodes_not_counted_as_terminal=True))
        atomic_json(args.output/'receipt.json', dict(status='completed', format=FORMAT, arm=arm, parent=parent,
            configuration_sha256=sha256(args.config), source_commit=args.source_commit, training_seed=2028,
            steps=steps, new_training_transitions=steps-100000, parent_prefix_steps=100000,
            safe_successes=successes, training_violations=violations, safe_timeouts=timeouts, episodes=episodes,
            new_safe_successes=successes-prefix['safe_successes'],
            new_training_violations=violations-prefix['training_violations'], new_safe_timeouts=timeouts-prefix['safe_timeouts'],
            learner_updates=learner.updates, new_learner_updates=learner.updates-prefix['learner_updates'],
            actor_updates=learner.actor_updates, reward_updates=learner.reward_updates,
            elapsed_seconds=elapsed(), evaluations=evaluations, evaluation_controls=evaluation_controls,
            diagnosis_steps=diagnosis_steps, probe_sha256=probe['sha256'],
            engineering_only=bool(args.engineering_steps), independent_endpoint_test_executed=False,
            all_suffix_step_trajectories_retained=True, physical_robot_interactions=0))
        atomic_json(args.output/'status.json', dict(state='completed', steps=steps, new_training_transitions=steps-100000))
    except BaseException as error:
        atomic_json(args.output/'status.json', dict(state='engineering_interruption', steps=steps,
            error_type=type(error).__name__, error=str(error), automatic_retry=False))
        raise
    finally:
        trajectory.close(); progress.close(); gradient_log.close()
        for env in envs: env.close()


if __name__ == '__main__': main()

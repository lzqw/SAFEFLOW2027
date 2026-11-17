"""Finite v11 native-task paired adaptation; data-disk outputs, no hardware."""
import argparse
from collections import deque
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

from scripts.paper_crax_payload.native_p5_adaptation import (
    FORMAT, attach_adaptation, build_native, evaluation_config, validate_parent, target_deployment, actor_digest)
from scripts.paper_crax_payload.piper_train import evaluate_piper
from scripts.paper_crax_payload.piper_parallel import PiperProcessEnvironment
from scripts.paper_crax_payload.piper_helpers import capture_piper_environment, restore_piper_environment
from scripts.paper_crax_payload.payload_environment import (
    PayloadConfig, PayloadFiniteRadiusEnvironment, capture_payload_environment, restore_payload_environment)
from scripts.paper_crax_payload.payload_train import evaluate_payload
from scripts.star.arm_safe_entropy.stability_runtime import capture_rng, restore_rng
from scripts.star.arm_safe_entropy.train import Replay, atomic_json


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def storage_check(path, growth_gib=10):
    if shutil.disk_usage(path).free < (40+growth_gib)*1024**3:
        raise OSError('Preserve40GiB data-disk headroom plus projected10GiB growth')


def restore_fork_rng(state, line):
    # Original Piper ran on cuda:1 with TWO visible GPUs. The finite shared
    # slot exposes ONE GPU: retain the actually used parent's generator, not
    # index0 by accident, and do not restore nonexistent generators.
    mapped = dict(state)
    if torch.cuda.is_available():
        if torch.cuda.device_count() != 1:
            raise ValueError('Fork requires one explicitly bound shared GPU')
        mapped['cuda'] = [state['cuda'][line['parent_rng_cuda_index']]]
    else:
        mapped['cuda'] = []
    restore_rng(mapped)


def train(args):
    plan = json.loads(args.plan.read_text())
    line = plan['lines'][args.line]
    if (plan['protocol'] != 'native_task_p5_paired_adaptation_v11'
            or plan['first_stage_training_steps'] != 600000
            or plan['development_training_ceiling'] != 900000
            or plan['additional_steps_per_run'] != 100000):
        raise ValueError('Use the registered bounded six-run first stage')
    # Exact root admission prevents accidentally writing large archives to home.
    root = args.data_root.resolve(strict=True)
    output = args.output.resolve()
    if not output.is_relative_to(root) or root == Path('/home/lzqw'):
        raise ValueError('Output must resolve beneath the explicitly admitted data disk')
    storage_check(root)
    if args.resume:
        if args.resume.parent.resolve() != output or not output.is_dir():
            raise ValueError('Resume in the original output directory only')
    else:
        output.mkdir(parents=True, exist_ok=False)
    lock = (output/'run.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(1)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, torch.device(args.device))
    checkpoint = args.parent or Path(line['parent_path'])
    parent = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_parent(parent, plan, args.line, sha256(checkpoint))
    native = deepcopy(parent['configuration'])
    learner = build_native(native, line, args.device)
    learner.load_state_dict(parent['learner'])  # Original strict native guard.
    initial_identity = dict(online_actor=actor_digest(learner.actor), target_actor=actor_digest(learner.target_actor))
    learner = attach_adaptation(learner, plan, args.line, args.arm)
    dim = 34 if line['robot']=='Piper' else 45
    shared = native['shared_learner']
    replay = Replay(shared['replay_capacity'], dim)
    replay.load_state_dict(parent['replay'])
    assets = json.loads(args.assets_receipt.read_text()) if line['robot']=='Piper' else None
    if assets and (assets['model_commit'] != parent['model_commit']
                   or assets['source_sha256'] != parent['model_source_urdf_sha256']
                   or sha256(assets['source_urdf']) != assets['source_sha256']
                   or sha256(assets['resolved_urdf']) != assets['resolved_sha256']):
        raise ValueError('Pinned native Piper official assets differ')
    if line['robot']=='Piper':
        if native.get('training_radius_values') != line.get('training_radius_values'):
            raise ValueError('The first Piper pair must not change training geometry')
        settings = dict(native['simulation'], task=line['task'])
        if native.get('training_radius_values'):
            settings['training_radius_values'] = native['training_radius_values']
        envs = [PiperProcessEnvironment(assets, settings) for _ in range(native['num_envs'])]
        capture, restore = capture_piper_environment, restore_piper_environment
    else:
        if tuple(line['training_radius_family']) != PayloadFiniteRadiusEnvironment.radius_values:
            raise ValueError('Both Panda arms retain the registered repair family')
        envs = [PayloadFiniteRadiusEnvironment(PayloadConfig(**native['task'])) for _ in range(native['num_envs'])]
        capture, restore = capture_payload_environment, restore_payload_environment
    steps = parent['steps']
    names = ('violations', 'episodes', 'safe_successes', 'safe_timeouts')
    prefix = {k: parent[k] for k in names}
    counts = deepcopy(prefix)
    partial_returns = np.zeros(len(envs)); partial_costs = np.zeros(len(envs))
    partial_lengths = np.zeros(len(envs), dtype=np.int64)
    recent = deque(maxlen=100)
    evaluation_controls = 0
    elapsed_before = 0.
    pkeys = ('partial_returns', 'partial_costs', 'partial_lengths') if assets else (
        'episode_returns', 'episode_costs', 'episode_lengths')
    censored = [dict(index=i, episode_return=float(parent[pkeys[0]][i]),
        cost=float(parent[pkeys[1]][i]), length=int(parent[pkeys[2]][i])) for i in range(len(envs))]
    resumed = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if resumed:
        if (resumed['format'] != FORMAT or resumed['plan'] != plan or resumed['line'] != args.line
                or resumed['arm'] != args.arm or resumed['source_commit'] != args.source_commit):
            raise ValueError('Strict identical v11 resume only')
        learner.load_state_dict(resumed['learner']); replay.load_state_dict(resumed['replay'])
        steps, counts = resumed['steps'], resumed['counts']
        partial_returns, partial_costs, partial_lengths = [resumed[k].copy() for k in
            ('partial_returns', 'partial_costs', 'partial_lengths')]
        recent = deque(resumed['recent'], maxlen=100)
        evaluation_controls = resumed['evaluation_controls']
        elapsed_before = resumed['elapsed_seconds']
        observations = np.stack([restore(e, s) for e, s in zip(envs, resumed['environments'])])
        restore_rng(resumed['rng'])
    else:
        # Restore native physical/RNG state, then explicitly censor partials
        # through a fresh reset. Both arms have identical reset treatment.
        for env, state in zip(envs, parent['environments']):
            restore(env, state)
        observations = np.stack([env.reset()[0] for env in envs])
        restore_fork_rng(parent['rng'], line)
    del parent
    learner.set_transitions(steps)
    evaluation = evaluation_config(native, plan, line)
    started = time.monotonic()
    initial_segment_steps = steps
    target = line['parent_steps']+plan['additional_steps_per_run']
    if steps >= target:
        raise ValueError('Do not duplicate a completed endpoint')
    milestones = [line['parent_steps']+s for s in plan['milestones'] if line['parent_steps']+s > steps]
    update_rounds = shared.get('updates_per_vector_step', len(envs)*shared['learner_updates_per_transition'])
    manifest = dict(protocol=plan['protocol'], plan=plan, configuration=native, line=args.line, arm=args.arm,
        source_commit=args.source_commit, parent_checkpoint=str(checkpoint), parent_sha256=line['parent_sha256'],
        initial_networks=initial_identity, initial_counts=prefix, censored_parent_partial_episodes=censored,
        censored_partials_counted_as_completed=False, native_pid=os.getpid(), host=platform.node(), device=args.device,
        output=str(output), allocator_fraction=args.cuda_memory_fraction, training_updates_per_vector_step=update_rounds,
        scientific_early_stopping=False, automatic_restart=False, physical_robot_interactions=0,
        common_deployment=plan['deployment'], task_definition=envs[0].definition(),
        source_files_sha256={name:sha256(Path(__file__).with_name(name)) for name in
            ('native_p5_adaptation.py', 'train_native_p5_adaptation.py')})
    atomic_json(output/('resume_manifest.json' if resumed else 'run_config.json'), manifest)
    stream = (output/'progress.jsonl').open('a' if resumed else 'x')
    last_log = last_checkpoint = steps

    def elapsed():
        return elapsed_before+time.monotonic()-started

    def checkpoint_out(name):
        storage_check(root)
        state = dict(format=FORMAT, plan=plan, configuration=native, line=args.line, arm=args.arm,
            source_commit=args.source_commit, steps=steps, counts=counts, prefix=prefix,
            learner=learner.state_dict(), replay=replay.state_dict(), rng=capture_rng(),
            environments=[capture(e) for e in envs], partial_returns=partial_returns.copy(),
            partial_costs=partial_costs.copy(), partial_lengths=partial_lengths.copy(), recent=list(recent),
            evaluation_controls=evaluation_controls, elapsed_seconds=elapsed())
        temporary = output/(name+'.tmp')
        torch.save(state, temporary); temporary.replace(output/name)

    try:
        while steps < target:
            actions = learner.action(observations)
            if assets:
                for env, action in zip(envs, actions):
                    env.begin_step(action)
            for i, env in enumerate(envs):
                observation, reward, done, truncated, info = env.step(actions[i])
                replay.add(obs=observations[i], action=actions[i], reward=reward, next_obs=observation,
                    done=float(done or truncated), cost=info['cost'], margin=info['local_margin'])
                steps += 1; counts['violations'] += int(info['violation'])
                partial_returns[i] += reward; partial_costs[i] += info['cost']; partial_lengths[i] += 1
                if done or truncated:
                    counts['episodes'] += 1; counts['safe_successes'] += int(info['safe_success'])
                    counts['safe_timeouts'] += int(not info['safe_success'] and not info['violation'])
                    learner.record_episode(partial_costs[i])
                    recent.append(dict(episode_return=float(partial_returns[i]),
                        length=int(partial_lengths[i]), safe_success=bool(info['safe_success'])))
                    partial_returns[i] = partial_costs[i] = partial_lengths[i] = 0
                    observation, _ = env.reset()
                observations[i] = observation
            learner.set_transitions(steps)
            for _ in range(update_rounds):
                learner.update(replay.sample(shared['batch_size'], learner.device, recent=True))
            new_steps = steps-line['parent_steps']
            if steps-last_log >= 1000 or steps==target:
                row = dict(steps=steps, new_steps=new_steps, **{k:counts[k]-prefix[k] for k in names},
                    elapsed_seconds=elapsed(), segment_sps=(steps-initial_segment_steps)/max(time.monotonic()-started,1e-9),
                    recent_success=float(np.mean([r['safe_success'] for r in recent])) if recent else 0.,
                    **learner.metrics())
                if not all(np.isfinite(v) for v in row.values()):
                    raise FloatingPointError('Nonfinite training diagnostic; no automatic retry')
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                atomic_json(output/'status.json', dict(state='running', native_pid=os.getpid(), **row))
                print(json.dumps(row), flush=True); last_log = steps
            if milestones and steps==milestones[0]:
                with target_deployment(learner):
                    result = (evaluate_piper(learner, evaluation, assets, line['task'],
                        plan['development_episodes_per_condition'], line['development_layout_seed_start']) if assets else
                        evaluate_payload(learner, evaluation, plan['development_episodes_per_condition']))
                controls = result.get('physical_evaluation_transitions', result.get('new_simulation_evaluation_control_intervals', 0))
                if assets:
                    controls = sum(r['length'] for c in result['conditions'] for r in c['records'])
                evaluation_controls += controls
                result.update(line=args.line, arm=args.arm, new_steps=new_steps, deployment=plan['deployment'])
                atomic_json(output/f'development_{new_steps}.json', result)
                checkpoint_out(f'full_new_{new_steps}.pt'); milestones.pop(0)
            if steps-last_checkpoint >= 10000 or steps==target:
                checkpoint_out('latest_full.pt'); last_checkpoint = steps
        atomic_json(output/'receipt.json', dict(status='completed', steps=steps,
            new_training_steps=steps-line['parent_steps'], segment_counts={k:counts[k]-prefix[k] for k in names},
            evaluation_controls=evaluation_controls, elapsed_seconds=elapsed(), source_commit=args.source_commit,
            line=args.line, arm=args.arm, physical_robot_interactions=0))
        atomic_json(output/'status.json', dict(state='completed', steps=steps, new_steps=steps-line['parent_steps']))
    except BaseException as error:
        atomic_json(output/'status.json', dict(state='engineering_interruption', steps=steps,
            error_type=type(error).__name__, error=str(error), automatic_restart=False))
        # A failed environment step can leave a physical worker unavailable.
        # Preserve the last valid checkpoint; never mask the original failure.
        try:
            checkpoint_out('interrupted_full.pt')
        except BaseException as checkpoint_error:
            atomic_json(output/'checkpoint_failure.json', dict(error=str(checkpoint_error), last_valid_checkpoint_preserved=True))
        raise
    finally:
        stream.close()
        for env in envs:
            env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--line', choices=['piper_reach','piper_payload','panda_payload'], required=True)
    p.add_argument('--arm', choices=['control','p5'], required=True)
    p.add_argument('--parent', type=Path)
    p.add_argument('--assets-receipt', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cuda-memory-fraction', type=float, default=.04)
    train(p.parse_args())


if __name__=='__main__':
    main()

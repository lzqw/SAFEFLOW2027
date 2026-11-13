"""Fresh official-Piper simulation training, frozen evaluation and safe export.

No Panda policy is loaded. No CAN/SDK/socket transport exists in this runner.
Physical deployment remains disabled even after a useful simulated candidate.
"""
import argparse
from collections import deque
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from deployment.piper.simulation import PiperSimConfig
from deployment.piper.bundle import export_bundle
from scripts.star.arm_safe_entropy.stability_models import StableArmLearner
from scripts.star.arm_safe_entropy.stability_runtime import capture_rng, restore_rng, isolated_rng
from scripts.star.arm_safe_entropy.train import Replay, atomic_json
from scripts.paper_crax_payload.piper_helpers import (
    PiperTrainingEnvironment, capture_piper_environment, restore_piper_environment)


FORMAT = 'paper_native_piper_sim_v1_full'


class PiperFlowLearner(StableArmLearner):
    def __init__(self, config, method, device):
        shared = config['shared_learner']
        if method not in config['methods']:
            raise ValueError('Use the declared native Piper Iso/Iso-Low/EA controls')
        variant = 's_ea' if method == 'EA' else 's_iso'
        super().__init__(obs_dim=34, variant=variant, device=device, width=shared['hidden_width'],
            actor_lr=shared['actor_lr'], critic_lr=shared['critic_lr'],
            critic_only_updates=shared['critic_only_updates'], margin_ema_tau=shared['margin_ema_tau'],
            actor_interval=shared['actor_update_interval'], rho=shared['rho'])
        variance = shared['iso_low_reference_variance'] if method == 'Iso-Low' else shared['reference_variance']
        if not 0 < variance <= shared['reference_variance']:
            raise ValueError('Invalid native reference variance')
        # Existing stable constructor is .04-only. Explicitly change BOTH
        # native samplers, the inverse-reference energy target AND alpha units.
        # Preserve the scalar parameter/optimizer identity by copy_, not assign.
        self.actor.variance = self.target_actor.variance = variance
        self.actor.num_steps = self.target_actor.num_steps = shared['flow_steps']
        self.energy_target = 3. * shared['energy_target_per_action'] / variance
        with torch.no_grad():
            self.log_alpha.copy_(torch.tensor(math.log(shared['kinetic_alpha_per_variance']*variance), device=self.device))
        self._last_metrics['alpha'] = self.log_alpha.detach().exp()
        self.cost_limit = shared['episode_budget']
        self.settings.update(piper_method=method, piper_reference_variance=variance,
            piper_energy_target=self.energy_target,
            piper_initial_alpha=shared['kinetic_alpha_per_variance']*variance,
            piper_flow_steps=shared['flow_steps'], piper_robot_family='Piper')

    def update(self, batch):
        super().update(batch)
        # The parent diagnostic indexes Panda25. Piper34 uses TCP12:15 and
        # goal15:18. This diagnostic never clips Q or affects an update.
        with torch.no_grad():
            upper = 5.*(batch['obs'][:, 12:15]-batch['obs'][:, 15:18]).norm(dim=-1) + 10.
            values = self.q(batch['obs'], batch['action']).min(0).values
            self._last_metrics['q_replay_above_upper_fraction'] = (values > upper+.01).float().mean()


def make_piper_env(config, assets, task, overrides=None):
    if task not in config['tasks'] or assets['model_commit'] != config['model_commit']:
        raise ValueError('Native task and pinned official model must match')
    settings = dict(config['simulation']) | (overrides or {})
    return PiperTrainingEnvironment(assets, PiperSimConfig(task=task, **settings))


def evaluate_piper(learner, config, assets, task, episodes, seed_start, conditions=None):
    results = []
    before = (learner.transitions, learner.updates, learner.actor_updates, learner.safety_multiplier)
    with isolated_rng():
        for ci, condition in enumerate(conditions or config['evaluation']['conditions']):
            torch.manual_seed(config['evaluation']['action_sampling_seed'] + ci)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config['evaluation']['action_sampling_seed'] + ci)
            env = make_piper_env(config, assets, task, condition.get('simulation_overrides'))
            rows = []
            try:
                for episode in range(episodes):
                    observation, initial = env.reset(seed=seed_start+episode)
                    minimum = initial['minimum_distance']
                    total = 0.
                    trajectory = []
                    before_controls = env.total_control_intervals
                    for step in range(env.config.horizon):
                        raw = learner.action(observation[None])[0]
                        executed = np.clip(raw*condition.get('action_scale', 1.), -1., 1.)
                        observation, reward, done, truncated, info = env.step(executed)
                        trajectory.append(env.task_position().tolist())
                        minimum = min(minimum, info['minimum_distance'])
                        total += reward
                        if done or truncated:
                            break
                    rows.append(dict(seed=seed_start+episode, episode_return=total,
                        safe_success=bool(info['safe_success']), violation=bool(info['violation']),
                        collision=bool(info['collision']), arm_collision=bool(info['arm_collision']),
                        payload_collision=bool(info['payload_collision']),
                        true_horizon_terminal=bool(info['true_horizon_terminal']),
                        length=env.total_control_intervals-before_controls,
                        minimum_distance_m=minimum, trajectory=trajectory,
                        time_to_goal_seconds=(step+1)*env.control_dt if info['safe_success'] else None,
                        time_to_goal_capped_seconds=((step+1) if info['safe_success'] else env.config.horizon)*env.control_dt))
                successful = [r['time_to_goal_seconds'] for r in rows if r['safe_success']]
                results.append(dict(condition=condition['name'], episodes=episodes,
                    safe_success_count=sum(r['safe_success'] for r in rows),
                    violation_count=sum(r['violation'] for r in rows),
                    safe_timeout_count=sum(not r['safe_success'] and not r['violation'] for r in rows),
                    time_to_goal_seconds=float(np.mean(successful)) if successful else None,
                    **{key: float(np.mean([r[key] for r in rows])) for key in
                       ('safe_success', 'violation', 'collision', 'arm_collision', 'payload_collision',
                        'episode_return', 'minimum_distance_m', 'time_to_goal_capped_seconds')},
                    new_simulation_control_intervals=env.total_control_intervals,
                    new_physics_substeps=env.total_physics_substeps,
                    distance_queries=env.total_distance_queries, records=rows))
            finally:
                env.close()
    if before != (learner.transitions, learner.updates, learner.actor_updates, learner.safety_multiplier):
        raise ValueError('Frozen evaluation altered training/dual state')
    return dict(robot_family='Piper', task=task, conditions=results,
        new_training_transitions=0, adaptation_updates=0, training_rng_preserved=True,
        new_simulation_control_intervals=sum(r['new_simulation_control_intervals'] for r in results),
        new_physics_substeps=sum(r['new_physics_substeps'] for r in results),
        distance_queries=sum(r['distance_queries'] for r in results), physical_robot_interactions=0)


def train(args):
    config = json.loads(args.config.read_text())
    assets = json.loads(args.assets_receipt.read_text())
    if args.output.exists():
        raise FileExistsError('Preserve every native Piper trial')
    if args.engineering and args.steps > 10000:
        raise ValueError('Engineering smoke is bounded10000 total target steps')
    if not args.engineering and args.seed != config['development_seed']:
        raise ValueError('Use the registered common native Piper development seed')
    if not 0 < args.steps <= 1000000 or args.steps % config['num_envs']:
        raise ValueError('Use a finite exact vector-aligned Piper target <=1M')
    torch.set_num_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    learner = PiperFlowLearner(config, args.method, args.device)
    shared = config['shared_learner']
    envs = [make_piper_env(config, assets, args.task) for _ in range(config['num_envs'])]
    replay = Replay(shared['replay_capacity'], 34)
    steps = violations = episodes = successes = timeouts = evaluation_controls = evaluation_substeps = 0
    partial_returns = np.zeros(len(envs)); partial_costs = np.zeros(len(envs))
    partial_lengths = np.zeros(len(envs), dtype=np.int64)
    recent = deque(maxlen=100)
    previous_elapsed = 0.
    if args.resume:
        saved = torch.load(args.resume, map_location=args.device, weights_only=False)
        if (saved.get('format') != FORMAT or saved['configuration'] != config or saved['task'] != args.task or
                saved['method'] != args.method or saved['training_seed'] != args.seed):
            raise ValueError('Only an identical native Piper full checkpoint may continue')
        prior = json.loads((args.resume.parent/'run_config.json').read_text())
        proc = Path('/proc', str(prior['native_pid']), 'cmdline')
        if prior['host'] == platform.node() and proc.exists():
            command = proc.read_bytes()
            if b'piper_train' in command and str(args.resume.parent).encode() in command:
                raise ValueError('Do not duplicate live native Piper training')
        learner.load_state_dict(saved['learner']); replay.load_state_dict(saved['replay'])
        steps, violations, episodes = saved['steps'], saved['violations'], saved['episodes']
        successes, timeouts = saved['safe_successes'], saved['safe_timeouts']
        evaluation_controls, evaluation_substeps = saved['evaluation_controls'], saved['evaluation_substeps']
        partial_returns, partial_costs, partial_lengths = [saved[key].copy() for key in
            ('partial_returns', 'partial_costs', 'partial_lengths')]
        previous_elapsed = saved['cumulative_elapsed_seconds']; recent = deque(saved['recent'], maxlen=100)
        observations = np.stack([restore_piper_environment(env, state) for env, state in zip(envs, saved['environments'])])
        restore_rng(saved['rng'])
    else:
        observations = np.stack([env.reset(seed=args.seed+997*i)[0] for i, env in enumerate(envs)])
    if steps >= args.steps:
        raise ValueError('Native continuation must collect additional actual transitions')
    initial_steps, initial_evaluation_controls, initial_evaluation_substeps = steps, evaluation_controls, evaluation_substeps
    initial_training_physics_substeps = sum(env.total_physics_substeps for env in envs) if args.resume else 0
    initial_training_distance_queries = sum(env.total_distance_queries for env in envs) if args.resume else 0
    learner.set_transitions(steps)
    args.output.mkdir(parents=True)
    source_files = dict(piper_train=Path(__file__), piper_helpers=Path(__file__).with_name('piper_helpers.py'),
        simulator=Path('deployment/piper/simulation.py'))
    atomic_json(args.output/'run_config.json', dict(robot_family='Piper', task=args.task, method=args.method,
        training_seed=args.seed, source_commit=args.source_commit, model_commit=config['model_commit'],
        configuration=config, asset_receipt=assets, task_definition=envs[0].definition(),
        native_pid=os.getpid(), host=platform.node(), device=args.device,
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), initial_steps=steps, total_transition_budget=args.steps,
        initialization='native_Piper_full_continuation' if args.resume else 'fresh_Piper_no_Panda_transfer',
        parent_checkpoint=str(args.resume) if args.resume else None, engineering=args.engineering,
        reference_variance=learner.actor.variance, energy_target=learner.energy_target,
        initial_alpha=shared['kinetic_alpha_per_variance']*learner.actor.variance,
        sampling_and_energy_same_reference=True, normalizer='identity_after_explicit_simulator_units',
        scientific_early_stopping=False, automatic_restart=False, physical_motion_enabled=False,
        source_files_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_files.items()},
        software=dict(torch=str(torch.__version__), cuda=torch.version.cuda, numpy=np.__version__,
            **{name: version(name) for name in ('gymnasium', 'pybullet')})))
    started = time.monotonic()
    last_log = last_checkpoint = steps
    milestones = [] if args.engineering else sorted(set(int(args.steps*f) for f in config['checkpoint_fractions'] if steps < int(args.steps*f) <= args.steps))
    if any(s % len(envs) for s in milestones):
        raise ValueError('Checkpoint fractions must align with vector boundaries')
    stream = (args.output/'progress.jsonl').open('w')

    def checkpoint(filename='latest_full.pt'):
        state = dict(format=FORMAT, robot_family='Piper', configuration=config, model_commit=config['model_commit'],
            model_source_urdf_sha256=assets['source_sha256'],
            source_commit=args.source_commit, task=args.task, method=args.method, training_seed=args.seed,
            engineering=args.engineering, learner=learner.state_dict(), replay=replay.state_dict(),
            steps=steps, violations=violations, episodes=episodes, safe_successes=successes, safe_timeouts=timeouts,
            evaluation_controls=evaluation_controls, evaluation_substeps=evaluation_substeps,
            partial_returns=partial_returns.copy(), partial_costs=partial_costs.copy(), partial_lengths=partial_lengths.copy(),
            recent=list(recent), environments=[capture_piper_environment(env) for env in envs], rng=capture_rng(),
            cumulative_elapsed_seconds=previous_elapsed+time.monotonic()-started)
        temporary = args.output/(filename+'.tmp')
        torch.save(state, temporary); temporary.replace(args.output/filename)

    def receipt(status):
        return dict(status=status, robot_family='Piper', training_robot_family='Piper', task=args.task,
            method=args.method, training_seed=args.seed, training_steps=steps, initial_steps=initial_steps,
            newly_collected_transitions=steps-initial_steps, model_commit=config['model_commit'],
            training_source_commit=args.source_commit, training_violation_count=violations,
            training_episodes=episodes, training_safe_successes=successes, training_safe_timeouts=timeouts,
            reference_variance=learner.actor.variance, energy_target=learner.energy_target,
            new_simulation_evaluation_control_intervals=evaluation_controls-initial_evaluation_controls,
            new_simulation_evaluation_physics_substeps=evaluation_substeps-initial_evaluation_substeps,
            new_training_physics_substeps=sum(env.total_physics_substeps for env in envs)-initial_training_physics_substeps,
            new_training_distance_queries=sum(env.total_distance_queries for env in envs)-initial_training_distance_queries,
            learner_updates=learner.updates, reward_updates=learner.reward_updates, actor_updates=learner.actor_updates,
            segment_elapsed_seconds=time.monotonic()-started,
            cumulative_elapsed_seconds=previous_elapsed+time.monotonic()-started,
            engineering=args.engineering, physical_robot_interactions=0, physical_motion_enabled=False)

    try:
        while steps < args.steps:
            actions = (np.random.uniform(-1., 1., (len(envs), 3)).astype(np.float32)
                if steps < shared['random_transitions'] else learner.action(observations))
            for index, env in enumerate(envs):
                observation, reward, done, truncated, info = env.step(actions[index])
                replay.add(obs=observations[index], action=actions[index], reward=reward,
                    next_obs=observation, done=float(done or truncated), cost=info['cost'], margin=info['local_margin'])
                violations += int(info['violation'])
                partial_returns[index] += reward; partial_costs[index] += info['cost']; partial_lengths[index] += 1
                if done or truncated:
                    successes += int(info['safe_success'])
                    timeouts += int(not info['safe_success'] and not info['violation'])
                    recent.append(dict(episode_return=float(partial_returns[index]), cost=float(partial_costs[index]),
                        length=int(partial_lengths[index]), safe_success=bool(info['safe_success'])))
                    learner.record_episode(partial_costs[index]); episodes += 1
                    partial_returns[index] = partial_costs[index] = partial_lengths[index] = 0
                    observation, _ = env.reset()
                observations[index] = observation
            steps += len(envs); learner.set_transitions(steps)
            if steps > shared['random_transitions'] and replay.count >= shared['batch_size']:
                for _ in range(len(envs)*shared['learner_updates_per_transition']):
                    learner.update(replay.sample(shared['batch_size'], learner.device, recent=True))
            if steps-last_log >= config['log_every'] or steps == args.steps:
                elapsed = time.monotonic()-started
                row = dict(steps=steps, episodes=episodes, training_violation_count=violations,
                    training_safe_successes=successes, training_safe_timeouts=timeouts,
                    recent_safe_success=float(np.mean([r['safe_success'] for r in recent])) if recent else 0.,
                    recent_episode_length=float(np.mean([r['length'] for r in recent])) if recent else 0.,
                    recent_return=float(np.mean([r['episode_return'] for r in recent])) if recent else 0.,
                    segment_elapsed_seconds=elapsed, cumulative_elapsed_seconds=previous_elapsed+elapsed,
                    complete_training_sps=(steps-initial_steps)/max(elapsed, 1e-9), **learner.metrics())
                if not all(np.isfinite(value) for value in row.values()):
                    raise FloatingPointError('Nonfinite native Piper learning diagnostic')
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                atomic_json(args.output/'status.json', dict(state='running', **row))
                print(json.dumps(row), flush=True); last_log = steps
            if milestones and steps == milestones[0]:
                milestone = milestones.pop(0)
                result = evaluate_piper(learner, config, assets, args.task,
                    config['evaluation']['development_episodes_per_condition'], config['evaluation']['development_layout_seed_start'])
                result.update(method=args.method, training_seed=args.seed, checkpoint_steps=steps, fixed_endpoint=steps==args.steps)
                evaluation_controls += result['new_simulation_control_intervals']; evaluation_substeps += result['new_physics_substeps']
                atomic_json(args.output/f'evaluation_{milestone}.json', result)
                checkpoint(f'full_{milestone}.pt')
                atomic_json(args.output/f'training_receipt_{milestone}.json', receipt('completed_native_checkpoint'))
            if steps-last_checkpoint >= config['checkpoint_every'] or steps == args.steps:
                checkpoint(); last_checkpoint = steps
        atomic_json(args.output/'receipt.json', receipt('completed'))
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


def load_frozen(args):
    config = json.loads(args.config.read_text()); assets = json.loads(args.assets_receipt.read_text())
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if (saved.get('format') != FORMAT or saved.get('robot_family') != 'Piper' or saved['configuration'] != config or
            saved['model_commit'] != config['model_commit'] or assets['model_commit'] != config['model_commit'] or
            saved['model_source_urdf_sha256'] != assets['source_sha256']):
        raise ValueError('Only a native matching Piper checkpoint, never renamed Panda weights')
    learner = PiperFlowLearner(config, saved['method'], args.device)
    learner.load_state_dict(saved['learner'])
    return config, assets, saved, learner


def evaluate_command(args):
    if args.output.exists():
        raise FileExistsError('Preserve frozen evaluation evidence')
    config, assets, saved, learner = load_frozen(args)
    plan = config['evaluation']
    default_episodes = plan.get(args.split+'_episodes_per_condition', 2)
    count = args.episodes if args.episodes is not None else default_episodes
    if args.split == 'engineering':
        if count > 2:
            raise ValueError('Bound engineering frozen rollouts to two episodes/condition')
        start = 699000
        conditions = [plan['conditions'][0]]
    else:
        start = plan[args.split+'_layout_seed_start']; conditions = None
    result = evaluate_piper(learner, config, assets, saved['task'], count, start, conditions)
    result.update(method=saved['method'], training_seed=saved['training_seed'], checkpoint_steps=saved['steps'],
        split=args.split, source_checkpoint=str(args.checkpoint), policy_adaptation=False)
    args.output.parent.mkdir(parents=True, exist_ok=True); atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'conditions'}))


def export_command(args):
    config, assets, saved, learner = load_frozen(args)
    if saved.get('engineering'):
        raise ValueError('An engineering smoke is not an eligible deployment candidate')
    selected = json.loads(args.selection_record.read_text())
    training_receipt = json.loads(args.training_receipt.read_text())
    if (training_receipt.get('training_source_commit') != saved['source_commit'] or
            training_receipt.get('method') != saved['method']):
        raise ValueError('Training receipt method/source do not match this actual native checkpoint')
    required = dict(robot_family='Piper', task=saved['task'], method=saved['method'],
        selected_checkpoint_steps=saved['steps'], selection_split='validation', independent_test_used_for_selection=False)
    if any(selected.get(k) != v for k, v in required.items()):
        raise ValueError('Require a matching common-rule validation selection record, not test-based selection')
    if selected.get('risk_qualified') is not True and selected.get('status') != 'pareto_tradeoff_not_risk_qualified':
        raise ValueError('Do not hide risk failure in a deployment export')
    metadata = dict(robot_family='Piper', training_robot_family='Piper', task=saved['task'],
        method=saved['method'], observation_dim=34, action_dim=3, training_steps=saved['steps'],
        model_commit=assets['model_commit'], training_source_commit=saved['source_commit'],
        source_urdf_sha256=saved['model_source_urdf_sha256'],
        normalizer_frozen=True, action_frame='robot_base', action_unit='normalized_delta_xyz',
        action_scale_m=config['simulation'].get('displacement_scale_m', .01),
        hidden_width=config['shared_learner']['hidden_width'], flow_steps=config['shared_learner']['flow_steps'],
        reference_variance=learner.actor.variance, iso_reference_variance=config['shared_learner']['reference_variance'],
        physical_motion_enabled=False, hardware_matched=False, selection=selected,
        task_semantics='native Piper simulation; IK1200/shared endpoint admissibility; Payload centroid success')
    normalizer = dict(kind='identity', dimension=34, frozen=True,
        units='explicit q/pi,dq/4; world SI payload pose/dimensions/mass, as native simulator observation')
    manifest = export_bundle(args.output, learner.state_dict(), metadata, normalizer, args.training_receipt)
    print(json.dumps(manifest))


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--config', type=Path, required=True)
    common.add_argument('--assets-receipt', type=Path, required=True)
    common.add_argument('--output', type=Path, required=True)
    common.add_argument('--device', default='cuda:0')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train', parents=[common])
    p.add_argument('--task', choices=['Reach', 'Payload'], required=True)
    p.add_argument('--method', choices=['Iso', 'Iso-Low', 'EA'], required=True)
    p.add_argument('--seed', type=int, default=41027)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--engineering', action='store_true')
    p.set_defaults(run=train)
    p = sub.add_parser('evaluate', parents=[common])
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--split', choices=['engineering', 'development', 'validation', 'independent_test'], required=True)
    p.add_argument('--episodes', type=int)
    p.set_defaults(run=evaluate_command)
    p = sub.add_parser('export', parents=[common])
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--training-receipt', type=Path, required=True)
    p.add_argument('--selection-record', type=Path, required=True)
    p.set_defaults(run=export_command)
    args = parser.parse_args(); args.run(args)


if __name__ == '__main__':
    main()

"""Finite-transition arm development runner. Large replay stays on the server."""
import argparse
from collections import deque
from dataclasses import asdict
import csv
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
import random
import time

import numpy as np
import torch

from scripts.star.arm_safe_entropy.environment import TaskConfig, make_environment
from scripts.star.arm_safe_entropy.models import ArmLearner
from scripts.star.arm_safe_entropy.run_protocol import (
    evaluation_schedule, resolve_phase, validate_main_freeze, validate_resume_origin)


class Replay:
    def __init__(self, capacity, obs_dim):
        self.capacity, self.count, self.cursor = capacity, 0, 0
        self.data = {key: np.zeros((capacity, width), np.float32) for key, width in
                     [('obs', obs_dim), ('next_obs', obs_dim), ('action', 3)]}
        self.data.update({key: np.zeros(capacity, np.float32) for key in ['reward', 'done', 'cost', 'margin']})

    def add(self, **values):
        for key, value in values.items():
            self.data[key][self.cursor] = value
        self.cursor = (self.cursor + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, batch_size, device, recent=False):
        indices = np.random.randint(self.count, size=batch_size)
        if recent:
            n = batch_size // 2
            indices[:n] = (self.cursor - 1 - np.random.randint(min(self.count, 10000), size=n)) % self.capacity
        return {key: torch.as_tensor(value[indices], device=device) for key, value in self.data.items()}

    def state_dict(self):
        return dict(capacity=self.capacity, count=self.count, cursor=self.cursor,
                    data={k: v[:self.count].copy() for k, v in self.data.items()})

    def load_state_dict(self, state):
        assert self.capacity == state['capacity']
        self.count, self.cursor = state['count'], state['cursor']
        for key, value in state['data'].items():
            self.data[key][:self.count] = value


def evaluate(learner, config, episodes=30, seed_start=52027):
    env = make_environment(config)
    records = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            obs, initial = env.reset(seed=seed)
            total = 0.
            minimum = initial['minimum_distance']
            for step in range(config.horizon):
                action = learner.action(obs[None])[0]
                obs, reward, done, truncated, info = env.step(action)
                total += reward
                minimum = min(minimum, info['minimum_distance'])
                if done or truncated:
                    break
            records.append(dict(seed=seed, episode_return=total, length=step + 1,
                minimum_distance_m=minimum,
                time_to_goal_seconds=(step+1)*env.control_dt if info['safe_success'] else None,
                time_to_goal_capped_seconds=((step+1) if info['safe_success'] else config.horizon)*env.control_dt,
                **{k: info[k] for k in ['safe_success', 'success', 'violation', 'collision',
                                      'clearance_violation', 'true_horizon_terminal']}))
    finally:
        env.close()
    successful_times = [r['time_to_goal_seconds'] for r in records if r['safe_success']]
    return dict(episodes=episodes, fixed_layout_seeds=list(range(seed_start, seed_start + episodes)),
                policy_sampling='stochastic_same_behavior_policy',
                time_to_goal_seconds=float(np.mean(successful_times)) if successful_times else None,
                time_to_goal_successes=len(successful_times),
                time_to_goal_capped_seconds=float(np.mean([r['time_to_goal_capped_seconds'] for r in records])),
                minimum_distance_m=float(np.mean([r['minimum_distance_m'] for r in records])),
                **{key: float(np.mean([r[key] for r in records])) for key in
                   ['safe_success', 'success', 'violation', 'collision', 'clearance_violation', 'episode_return']},
                records=records)


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['sac', 'flow', 'sac_lag', 'flow_iso', 'flow_oracle',
                                           'flow_learned', 'flow_iso_low', 'flow_random_aniso'], required=True)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--num-envs', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--resume', type=Path)
    evaluation_options = parser.add_mutually_exclusive_group()
    evaluation_options.add_argument('--eval-every', type=int)
    evaluation_options.add_argument('--eval-steps', type=int, nargs='+',
                                    help='Explicit fixed milestones; formal runs use100k/300k/600k/1M')
    parser.add_argument('--eval-episodes', type=int, default=30)
    parser.add_argument('--eval-seed-start', type=int)
    parser.add_argument('--checkpoint-every', type=int, default=10000)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--oracle-workers', type=int, default=8)
    parser.add_argument('--oracle-backend', choices=['exact', 'controller_jacobian'], default='exact')
    parser.add_argument('--oracle-cache-entries', type=int, default=40000000)
    parser.add_argument('--reference-variance', type=float)
    parser.add_argument('--normal-variance-ratio', type=float, default=.25)
    parser.add_argument('--revision-id')
    parser.add_argument('--revision-reason')
    parser.add_argument('--reference-match-run', help='LearnedDir run used to measure Iso-Low normal reference noise')
    parser.add_argument('--phase', choices=['ability', 'obstacle_development', 'mechanism_control', 'relative_layout_screening', 'main'])
    parser.add_argument('--task-version', choices=['SafeReach-Obstacle-v2', 'SafeReach-RelativeLayout-v3'],
                        default='SafeReach-Obstacle-v2')
    parser.add_argument('--frozen-config', type=Path, help='Required for main; must declare a genuinely frozen development configuration')
    args = parser.parse_args()
    try:
        phase = resolve_phase(args.method, args.seed, args.steps, args.eval_episodes, args.phase, args.smoke)
    except ValueError as error:
        parser.error(str(error))
    if phase == 'relative_layout_screening' and args.task_version != 'SafeReach-RelativeLayout-v3':
        parser.error('Relative-layout screening must retain its explicitly named task')
    if args.task_version == 'SafeReach-RelativeLayout-v3' and phase not in {'relative_layout_screening', 'main'}:
        parser.error('The relative-layout task is only for its screening stage or a genuinely frozen main stage')
    args.eval_seed_start = args.eval_seed_start if args.eval_seed_start is not None else (62027 if phase == 'main' else 52027)
    args.eval_every = args.eval_every if args.eval_every is not None else (25000 if phase == 'ability' else 100000)
    try:
        declared_evaluation_steps = evaluation_schedule(args.steps, args.eval_every, args.eval_steps, args.num_envs)
    except ValueError as error:
        parser.error(str(error))
    if args.method == 'flow_iso_low' and (args.reference_variance is None or not args.reference_match_run):
        parser.error('Iso-Low requires explicitly measured --reference-variance and --reference-match-run')
    variance = .04 if args.reference_variance is None else args.reference_variance
    if not 0 < variance <= .04:
        parser.error('Use a positive reference variance no larger than the fixed .04 development reference')
    if not 0 < args.normal_variance_ratio <= 1:
        parser.error('Normal variance ratio must be in (0,1]')
    if args.normal_variance_ratio != .25 and phase != 'main' and not (args.revision_id and args.revision_reason):
        parser.error('A targeted revision must record its id and reason')
    frozen_record = None
    if phase == 'main':
        if args.frozen_config is None:
            parser.error('Main requires --frozen-config after useful directional development evidence')
        frozen_record = json.loads(args.frozen_config.read_text())
        try:
            validate_main_freeze(frozen_record, args.method, args.seed, variance,
                                 args.normal_variance_ratio, args.task_version)
        except ValueError as error:
            parser.error(str(error))
        if args.eval_seed_start != frozen_record['evaluation_layout_seed_start']:
            parser.error('Use the frozen common main evaluation layout seeds')
        if declared_evaluation_steps != frozen_record['evaluation_steps']:
            parser.error('Use the frozen main evaluation schedule')
    resume_manifest = None
    if args.resume:
        origin = args.resume.parent / 'run_config.json'
        if origin.exists():
            resume_manifest = json.loads(origin.read_text())
            try:
                validate_resume_origin(args.seed, phase, resume_manifest)
            except ValueError as error:
                parser.error(str(error))
            if phase == 'main' and resume_manifest.get('frozen_config') != frozen_record:
                parser.error('Do not change the frozen main configuration on resume')
            if resume_manifest.get('revision_id') != args.revision_id:
                parser.error('Resume the same revision identity')
    assert args.steps % args.num_envs == 0
    assert not args.output.exists(), 'Do not overwrite another run'
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = TaskConfig(version=args.task_version, obstacle=args.method not in ['sac', 'flow'])
    envs = [make_environment(config) for _ in range(args.num_envs)]
    oracle = None
    if args.method == 'flow_oracle':
        from scripts.star.arm_safe_entropy.oracle import SimulatorOracle, ControllerJacobianOracle
        if args.oracle_backend == 'exact':
            oracle = SimulatorOracle(workers=args.oracle_workers)
        else:
            oracle = ControllerJacobianOracle(workers=args.oracle_workers,
                                               cache_size=args.oracle_cache_entries)
    learner = ArmLearner(25, args.method, args.device, oracle_geometry=oracle,
                         variance=variance, rho=args.normal_variance_ratio)
    replay = Replay(300000, 25)
    steps, violations, episodes = 0, 0, 0
    if args.resume:
        saved = torch.load(args.resume, map_location=args.device, weights_only=False)
        assert saved['method'] == args.method and saved['task_config'] == asdict(config)
        if 'training_seed' in saved:
            assert saved['training_seed'] == args.seed, 'Checkpoint seed cannot be relabelled'
        else:
            assert resume_manifest is not None, 'Legacy resume requires the accompanying native seed manifest'
        if phase == 'main':
            assert saved.get('stage', (resume_manifest or {}).get('stage')) == 'main'
            assert saved.get('frozen_config', (resume_manifest or {}).get('frozen_config')) == frozen_record
        assert saved.get('reference_variance', .04) == variance, 'Do not change reference noise on resume'
        if learner.flow:
            assert saved.get('normal_variance_ratio', .25) == args.normal_variance_ratio, 'Do not change normal ratio on resume'
        if oracle:
            assert saved['oracle_backend'] == args.oracle_backend, 'Do not silently change oracle model on resume'
        learner.load_state_dict(saved['learner'])
        replay.load_state_dict(saved['replay'])
        steps, violations, episodes = saved['steps'], saved['violations'], saved['episodes']
        random.setstate(saved['rng']['python'])
        np.random.set_state(saved['rng']['numpy'])
        torch.set_rng_state(saved['rng']['torch'].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in saved['rng']['cuda']])
    initial_steps = steps
    obs = np.stack([env.reset(seed=args.seed + 997 * index)[0] for index, env in enumerate(envs)])
    episode_returns = np.zeros(args.num_envs)
    recent = deque(maxlen=100)
    elapsed_start = time.monotonic()
    last_log = steps
    last_checkpoint = steps
    # Optimization/physics are unchanged; phases make the evaluation protocol explicit.
    manifest = dict(stage='engineering_smoke' if args.smoke else 'unobstructed_ability',
        method=args.method, training_seed=args.seed, source_commit=args.source_commit,
        task_config=asdict(config), task_definition=envs[0].definition(),
        device=args.device, num_envs=args.num_envs, total_transition_budget=args.steps,
        initial_steps=initial_steps, transition_count='sum of actual train env.step calls',
        evaluation_layout_seed_start=args.eval_seed_start, evaluation_episodes=args.eval_episodes,
        evaluation_every=args.eval_every if args.eval_steps is None else None,
        evaluation_steps=declared_evaluation_steps,
        resume_path=str(args.resume) if args.resume else None,
        resume_episode_handling='new episode on learner/replay resume; partial prior episodes not claimed exact',
        batch_size=256, replay_capacity=300000, random_transitions=2000, utd=1.0,
        reward_critic_updates_per_update=2 if learner.flow else 1,
        actor_update_interval=2 if learner.flow else 1, reward_target_samples=4 if learner.flow else 1,
        actor_lr=.0003, critic_lr=.0003, reward_target_polyak=.001 if learner.flow else .005,
        actor_target_polyak=.001 if learner.flow else None,
        reward_actor_sampling='half uniform + half recent10000' if learner.flow else 'uniform',
        geometry_source='none', directional_loss=False, inward_drift=False, action_projection=False,
        reference_variance=variance if learner.flow else None,
        normal_variance_ratio=args.normal_variance_ratio if learner.flow else None,
        revision_id=args.revision_id, revision_reason=args.revision_reason,
        kinetic_alpha_initial=.02 if learner.flow else None,
        scaled_alpha_initial=variance * .02 if learner.flow else .1,
        kinetic_energy_target_per_action=.25 if learner.flow else None,
        initial_base_variance=0. if learner.flow else None,
        global_cost_critic='two sigmoid episode-risk Q heads' if learner.safety else None,
        safety_budget=.1 if learner.safety else None,
        safety_dual_update='completed-episode EMA .05, projected lr .05 after random warmup, initial0, max100' if learner.safety else None,
        actor_objective_normalized_by_dual=False,
        safety_replay_sampling='uniform+recent same batch; no safety priority controls actor')
    manifest['stage'] = ('engineering_smoke' if args.smoke else
                         'unobstructed_ability' if phase == 'ability' else phase)
    manifest['frozen_config'] = frozen_record
    manifest['geometry_source'] = {'flow_learned': 'learned_margin', 'flow_random_aniso': 'learned_margin',
                                   'flow_oracle': 'oracle'}.get(args.method, 'none')
    manifest['reference_control'] = ('global_low_variance' if args.method == 'flow_iso_low' else
                                     'random_direction_fixed_per_action' if args.method == 'flow_random_aniso' else None)
    manifest['reference_match_run'] = args.reference_match_run
    manifest['reference_match_definition'] = ('mean normal reference variance, NOT total trace' if args.method == 'flow_iso_low' else
        'same eigenvalue/trace formula and learned-q rule, independent direction; q need not numerically match a separately trained policy'
        if args.method == 'flow_random_aniso' else None)
    manifest['run_host'] = platform.node()
    manifest['gpu_name'] = torch.cuda.get_device_name(learner.device) if learner.device.type == 'cuda' else None
    manifest['visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    manifest['software'] = dict(python=platform.python_version(), torch=str(torch.__version__),
        cuda=torch.version.cuda, numpy=np.__version__,
        **{name: version(name) for name in ['panda-gym', 'pybullet', 'gymnasium']})
    manifest['oracle_information'] = ('known robot/controller/obstacle model; exact simulator finite differences'
        if args.oracle_backend == 'exact' else
        'privileged first-order actual-controller interval-distance approximation; not an error-free upper bound') if oracle else None
    manifest['oracle_workers'] = args.oracle_workers if oracle else None
    manifest['oracle_backend'] = args.oracle_backend if oracle else None
    manifest['oracle_cache_entries'] = args.oracle_cache_entries if oracle else None
    atomic_json(args.output / 'run_config.json', manifest)
    fields = ['steps', 'elapsed_seconds', 'train_fps', 'episodes', 'training_violation_count',
              'recent_success', 'recent_return', 'reward_td', 'reward_q_abs_max', 'actor_loss', 'energy', 'clipping_rate', 'alpha',
              'cost_loss', 'margin_loss', 'safety_multiplier']
    stream = (args.output / 'progress.csv').open('w', newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    evaluation_steps = [s for s in declared_evaluation_steps if s > steps]
    latest = {k: 0. for k in fields[7:]}
    latest['alpha'] = float(learner.log_alpha.detach().exp())

    def checkpoint(destination='latest_full.pt'):
        payload = dict(method=args.method, task_config=asdict(config), learner=learner.state_dict(),
            replay=replay.state_dict(), steps=steps, violations=violations, episodes=episodes,
            source_commit=args.source_commit, rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                                      torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
        payload['oracle_backend'] = args.oracle_backend if oracle else None
        payload['reference_variance'] = variance
        payload['normal_variance_ratio'] = args.normal_variance_ratio if learner.flow else None
        payload['training_seed'] = args.seed
        payload['stage'] = manifest['stage']
        payload['frozen_config'] = frozen_record
        temporary = args.output / 'latest_full.tmp'
        torch.save(payload, temporary)
        temporary.replace(args.output / destination)

    try:
        while steps < args.steps:
            actions = np.random.uniform(-1, 1, size=(args.num_envs, 3)).astype(np.float32) if steps < 2000 else learner.action(obs)
            for index, env in enumerate(envs):
                next_obs, reward, done, truncated, info = env.step(actions[index])
                replay.add(obs=obs[index], action=actions[index], reward=reward, next_obs=next_obs,
                           done=float(done or truncated), cost=info['cost'], margin=info['local_margin'])
                violations += int(info['violation'])
                episode_returns[index] += reward
                if done or truncated:
                    recent.append((float(episode_returns[index]), float(info['success'])))
                    learner.record_episode(info['cost'])
                    episodes += 1
                    episode_returns[index] = 0.
                    next_obs, _ = env.reset()
                obs[index] = next_obs
            steps += args.num_envs
            if steps > 2000 and replay.count >= 256:
                for _ in range(args.num_envs):
                    latest = learner.update(replay.sample(256, learner.device, recent=learner.flow))
            if steps - last_log >= 1000 or steps == args.steps:
                elapsed = time.monotonic() - elapsed_start
                row = dict(steps=steps, elapsed_seconds=elapsed, train_fps=(steps - initial_steps) / elapsed,
                           episodes=episodes, training_violation_count=violations,
                           recent_success=float(np.mean([r[1] for r in recent])) if recent else 0.,
                           recent_return=float(np.mean([r[0] for r in recent])) if recent else 0., **latest)
                if not all(np.isfinite(v) for v in row.values()):
                    raise FloatingPointError('Nonfinite training diagnostic')
                writer.writerow(row)
                stream.flush()
                atomic_json(args.output / 'status.json', dict(state='running', **row))
                print(json.dumps(row), flush=True)
                last_log = steps
            if evaluation_steps and steps >= evaluation_steps[0]:
                milestone = evaluation_steps.pop(0)
                result = dict(milestone_steps=steps, training_violation_count=violations,
                              **evaluate(learner, config, args.eval_episodes, args.eval_seed_start))
                result['evaluation_transitions'] = sum(row['length'] for row in result['records'])
                atomic_json(args.output / f'evaluation_{milestone}.json', result)
                torch.save(learner.state_dict(), args.output / f'policy_{milestone}.pt')
            if steps - last_checkpoint >= args.checkpoint_every or steps == args.steps:
                checkpoint()
                last_checkpoint = steps
        atomic_json(args.output / 'receipt.json', dict(status='completed', steps=steps,
            newly_collected_transitions=steps - initial_steps, training_violation_count=violations,
            source_commit=args.source_commit, elapsed_seconds=time.monotonic() - elapsed_start,
            learner_updates=learner.updates, reward_updates=learner.reward_updates, actor_updates=learner.actor_updates,
            final_evaluation=f'evaluation_{steps}.json',
            oracle_statistics=oracle.statistics() if oracle else None))
        atomic_json(args.output / 'status.json', dict(state='completed', steps=steps))
    except BaseException as error:
        checkpoint(destination='interrupted_full.pt')
        atomic_json(args.output / 'status.json', dict(state='engineering_interruption', steps=steps,
                                                      error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        stream.close()
        for env in envs:
            env.close()
        if oracle is not None:
            oracle.close()


if __name__ == '__main__':
    main()

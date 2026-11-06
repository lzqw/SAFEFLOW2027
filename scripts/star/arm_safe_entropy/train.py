"""Finite-transition arm development runner. Large replay stays on the server."""
import argparse
from collections import deque
from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from scripts.star.arm_safe_entropy.environment import SafeReachEnv, TaskConfig
from scripts.star.arm_safe_entropy.models import ArmLearner


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


def evaluate(learner, config, episodes=30):
    env = SafeReachEnv(config)
    records = []
    try:
        for seed in range(52027, 52027 + episodes):
            obs, _ = env.reset(seed=seed)
            total = 0.
            for step in range(config.horizon):
                action = learner.action(obs[None])[0]
                obs, reward, done, truncated, info = env.step(action)
                total += reward
                if done or truncated:
                    break
            records.append(dict(seed=seed, episode_return=total, length=step + 1,
                                **{k: info[k] for k in ['safe_success', 'success', 'violation', 'collision', 'clearance_violation']}))
    finally:
        env.close()
    return dict(episodes=episodes, fixed_layout_seeds=list(range(52027, 52027 + episodes)),
                policy_sampling='stochastic_same_behavior_policy',
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
    parser.add_argument('--eval-every', type=int, default=25000)
    parser.add_argument('--eval-episodes', type=int, default=30)
    parser.add_argument('--checkpoint-every', type=int, default=10000)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--oracle-workers', type=int, default=8)
    parser.add_argument('--oracle-backend', choices=['exact', 'controller_jacobian'], default='exact')
    parser.add_argument('--oracle-cache-entries', type=int, default=40000000)
    parser.add_argument('--reference-variance', type=float)
    parser.add_argument('--reference-match-run', help='LearnedDir run used to measure Iso-Low normal reference noise')
    args = parser.parse_args()
    if args.method == 'flow_iso_low' and (args.reference_variance is None or not args.reference_match_run):
        parser.error('Iso-Low requires explicitly measured --reference-variance and --reference-match-run')
    variance = .04 if args.reference_variance is None else args.reference_variance
    if not 0 < variance <= .04:
        parser.error('Use a positive reference variance no larger than the fixed .04 development reference')
    assert args.steps % args.num_envs == 0
    assert not args.output.exists(), 'Do not overwrite another run'
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = TaskConfig(obstacle=args.method not in ['sac', 'flow'])
    envs = [SafeReachEnv(config) for _ in range(args.num_envs)]
    oracle = None
    if args.method == 'flow_oracle':
        from scripts.star.arm_safe_entropy.oracle import SimulatorOracle, ControllerJacobianOracle
        if args.oracle_backend == 'exact':
            oracle = SimulatorOracle(workers=args.oracle_workers)
        else:
            oracle = ControllerJacobianOracle(workers=args.oracle_workers,
                                               cache_size=args.oracle_cache_entries)
    learner = ArmLearner(25, args.method, args.device, oracle_geometry=oracle, variance=variance)
    replay = Replay(300000, 25)
    steps, violations, episodes = 0, 0, 0
    if args.resume:
        saved = torch.load(args.resume, map_location=args.device, weights_only=False)
        assert saved['method'] == args.method and saved['task_config'] == asdict(config)
        assert saved.get('reference_variance', .04) == variance, 'Do not change reference noise on resume'
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
    # Explicit common initial configuration. Safety variants are not yet enabled.
    manifest = dict(stage='engineering_smoke' if args.smoke else 'unobstructed_ability',
        method=args.method, training_seed=args.seed, source_commit=args.source_commit,
        task_config=asdict(config), task_definition=envs[0].definition(),
        device=args.device, num_envs=args.num_envs, total_transition_budget=args.steps,
        initial_steps=initial_steps, transition_count='sum of actual train env.step calls',
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
        kinetic_alpha_initial=.02 if learner.flow else None,
        scaled_alpha_initial=variance * .02 if learner.flow else .1,
        kinetic_energy_target_per_action=.25 if learner.flow else None,
        initial_base_variance=0. if learner.flow else None,
        global_cost_critic='two sigmoid episode-risk Q heads' if learner.safety else None,
        safety_budget=.1 if learner.safety else None,
        safety_dual_update='completed-episode EMA .05, projected lr .05 after random warmup, initial0, max100' if learner.safety else None,
        actor_objective_normalized_by_dual=False,
        safety_replay_sampling='uniform+recent same batch; no safety priority controls actor')
    manifest['stage'] = 'engineering_smoke' if args.smoke else 'obstacle_development' if learner.safety else 'unobstructed_ability'
    manifest['geometry_source'] = {'flow_learned': 'learned_margin', 'flow_oracle': 'oracle'}.get(args.method, 'none')
    if args.method == 'flow_random_aniso':
        manifest['geometry_source'] = 'learned_margin_weight_random_direction'
    manifest['reference_control'] = ('global_low_variance' if args.method == 'flow_iso_low' else
                                     'random_direction_fixed_per_action' if args.method == 'flow_random_aniso' else None)
    manifest['reference_match_run'] = args.reference_match_run
    manifest['reference_match_definition'] = ('mean normal reference variance, NOT total trace' if args.method == 'flow_iso_low' else
        'same eigenvalue/trace formula and learned-q rule, independent direction; q need not numerically match a separately trained policy'
        if args.method == 'flow_random_aniso' else None)
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
    evaluation_steps = sorted(set(range(args.eval_every, args.steps + 1, args.eval_every)) | {args.steps})
    evaluation_steps = [s for s in evaluation_steps if s > steps]
    latest = {k: 0. for k in fields[7:]}
    latest['alpha'] = float(learner.log_alpha.detach().exp())

    def checkpoint(destination='latest_full.pt'):
        payload = dict(method=args.method, task_config=asdict(config), learner=learner.state_dict(),
            replay=replay.state_dict(), steps=steps, violations=violations, episodes=episodes,
            source_commit=args.source_commit, rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                                      torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
        payload['oracle_backend'] = args.oracle_backend if oracle else None
        payload['reference_variance'] = variance
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
                              **evaluate(learner, config, args.eval_episodes))
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

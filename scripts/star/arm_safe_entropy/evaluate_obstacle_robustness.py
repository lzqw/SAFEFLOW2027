"""One predeclared full-episode zero-shot comparison, with no learner updates.

Physics remains the original SafeReach-v2 controller, reward and first-violation
termination. Only sphere radius and its lateral world position change.
"""
import argparse
from dataclasses import replace
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from scripts.star.arm_safe_entropy.environment import SafeReachEnv, TaskConfig


def atomic_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class PerturbedReach(SafeReachEnv):
    def reset_condition(self, seed, offset):
        _, info = super().reset(seed=seed)
        self.obstacle_position = self.obstacle_position + np.array([offset, 0., 0.])
        self.sim.set_base_pose('obstacle', self.obstacle_position, np.array([0., 0., 0., 1.]))
        minimum, collision = self._distance()
        if collision or minimum <= self.config.safety_distance + .005:
            raise RuntimeError(f'Unsafe perturbed initial layout seed={seed}, offset={offset}: {minimum}')
        layout = dict(start=self.start.tolist(), goal=self.goal.tolist(),
                      obstacle=self.obstacle_position.tolist(), radius_m=self.config.obstacle_radius,
                      initial_minimum_distance_m=minimum)
        return self._obs(), layout


def physics_worker(connection):
    env = None
    try:
        while True:
            command, payload = connection.recv()
            if command == 'close':
                break
            if command == 'reset':
                if env is None or env.config.obstacle_radius != payload['radius_m']:
                    if env is not None:
                        env.close()
                    env = PerturbedReach(replace(TaskConfig(), obstacle_radius=payload['radius_m']))
                result = env.reset_condition(payload['seed'], payload['offset_m'])
            elif command == 'step':
                result = env.step(payload)
            elif command == 'detour_action':
                position = env.ee_position()
                # One fixed constructive path, not a teacher for any learned policy.
                waypoints = [env.start + np.array([0., -.07, 0.]),
                             env.start + np.array([.03, -.07, .20]),
                             env.goal + np.array([.03, 0., .20]), env.goal]
                target = payload
                if np.linalg.norm(position - waypoints[target]) < .025 and target < 3:
                    target += 1
                action = np.clip((waypoints[target] - position) / env.config.displacement_scale, -1., 1.)
                result = (action, target)
            else:
                raise ValueError(command)
            connection.send(result)
    except BaseException as error:
        connection.send({'worker_error': f'{type(error).__name__}: {error}'})
    finally:
        if env is not None:
            env.close()
        connection.close()


def receive(connection):
    if not connection.poll(60):
        raise TimeoutError('A specific live physics worker did not answer; no automatic job restart')
    result = connection.recv()
    if isinstance(result, dict) and 'worker_error' in result:
        raise RuntimeError(result['worker_error'])
    return result


def load_policy(run, device):
    import torch
    from scripts.star.arm_safe_entropy.models import ArmFlowActor, GaussianActor
    config = json.loads((run / 'run_config.json').read_text())
    receipt = json.loads((run / 'receipt.json').read_text())
    if receipt['status'] != 'completed' or receipt['steps'] != 300000 or config['total_transition_budget'] != 300000:
        raise ValueError('Use the fixed final300k native run, never selected milestones')
    if config['task_config'] != TaskConfig().__dict__:
        raise ValueError('Do not pool other task versions or changed nominal physics')
    state = torch.load(run / 'policy_300000.pt', map_location=device, weights_only=False)
    if state['updates'] != 298000:
        raise ValueError('Unexpected final policy update count')
    flow = config['method'].startswith('flow')
    actor = (ArmFlowActor(25, variance=config['reference_variance'],
                          rho=config.get('normal_variance_ratio') or .25) if flow
             else GaussianActor(25)).to(device)
    actor.load_state_dict(state['actor'])
    actor.eval().requires_grad_(False)
    margin = None
    if config['method'] == 'flow_learned':
        from scripts.star.arm_safe_entropy.geometry import LocalMarginEnsemble
        margin = LocalMarginEnsemble().to(device)
        margin.load_state_dict(state['margin'])
        margin.eval().requires_grad_(False)
    lineage_path = run / 'resume_lineage.json'
    return actor, margin, dict(method=config['method'], training_seed=config['training_seed'],
        policy_source_commit=config['source_commit'], checkpoint_steps=300000,
        run=str(run), checkpoint=str(run / 'policy_300000.pt'),
        reference_variance=config.get('reference_variance'),
        normal_variance_ratio=config.get('normal_variance_ratio'),
        resume_lineage=json.loads(lineage_path.read_text()) if lineage_path.exists() else None)


def rollout_wave(pipes, condition, indices, plan, actor=None, margin=None, training_seed=None):
    import torch
    active = np.ones(len(indices), dtype=bool)
    connections = pipes[:len(indices)]
    for pipe, index in zip(connections, indices):
        sign = -1 if index % 2 == 0 else 1
        pipe.send(('reset', dict(seed=plan['layout_seed_start'] + index,
                                radius_m=condition['radius_m'], offset_m=sign*condition['lateral_offset_m'])))
    initial = [receive(pipe) for pipe in connections]
    observations = np.stack([value[0] for value in initial])
    records = [dict(layout_seed=plan['layout_seed_start'] + index,
                    lateral_sign=-1 if index % 2 == 0 else 1, layout=value[1],
                    minimum_distance_m=value[1]['initial_minimum_distance_m'],
                    episode_return=0., length=0) for index, value in zip(indices, initial)]
    targets = np.zeros(len(indices), dtype=int)
    sampling_seed = plan['action_sampling_seed'] + (training_seed or 0)*1000 + indices[0]
    if actor is not None:
        torch.manual_seed(sampling_seed)
    for step in range(TaskConfig().horizon):
        if actor is None:
            for index in np.flatnonzero(active):
                connections[index].send(('detour_action', int(targets[index])))
            actions = np.zeros((len(indices), 3))
            for index in np.flatnonzero(active):
                actions[index], targets[index] = receive(connections[index])
        else:
            with torch.no_grad():
                batch = torch.as_tensor(observations, device=next(actor.parameters()).device)
                result = actor(batch, geometry=margin.geometry if margin else None) if hasattr(actor, 'variance') else actor(batch)
                actions = result[0].cpu().numpy()
        # Inactive rows still consume native sampling noise, but never execute a
        # second action after success/violation/horizon. Batch RNG is stable.
        for index in np.flatnonzero(active):
            connections[index].send(('step', actions[index]))
        for index in np.flatnonzero(active):
            observation, reward, done, truncated, info = receive(connections[index])
            observations[index] = observation
            record = records[index]
            record['minimum_distance_m'] = min(record['minimum_distance_m'], info['minimum_distance'])
            record['episode_return'] += reward
            record['length'] = step + 1
            if done or truncated:
                active[index] = False
                record.update({key: info[key] for key in
                    ['safe_success', 'success', 'violation', 'collision', 'clearance_violation', 'true_horizon_terminal']})
                record['time_to_goal_seconds'] = (record['length']*info['control_dt'] if info['safe_success'] else None)
                record['time_to_goal_capped_seconds'] = (record['time_to_goal_seconds'] if info['safe_success']
                                                       else TaskConfig().horizon*info['control_dt'])
                record['sampling_seed_for_fixed_wave'] = sampling_seed if actor is not None else None
        if not active.any():
            break
    if active.any():
        raise RuntimeError('The true finite-horizon episode did not terminate')
    return records


def summarize(records):
    successes = [r['time_to_goal_seconds'] for r in records if r['safe_success']]
    return dict(episodes=len(records), **{key: float(np.mean([r[key] for r in records])) for key in
        ['safe_success', 'success', 'violation', 'collision', 'clearance_violation', 'episode_return',
         'minimum_distance_m', 'time_to_goal_capped_seconds']},
        time_to_goal_seconds=float(np.mean(successes)) if successes else None,
        time_to_goal_successes=len(successes), evaluation_transitions=sum(r['length'] for r in records))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--previous-layout-validity', type=Path,
                        help='Reuse already successful physical detours; retry only layouts lacking a constructive proof')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    plan = json.loads(args.plan.read_text())
    configs = [json.loads((run / 'run_config.json').read_text()) for run in args.runs]
    expected = {(method, seed) for method in plan['training_methods'] for seed in plan['training_seeds']}
    if len(configs) != 8 or {(c['method'], c['training_seed']) for c in configs} != expected:
        parser.error('Keep all four methods and both training seeds, without duplicates')
    if args.output.exists():
        parser.error('Never overwrite an earlier evaluation')
    args.output.mkdir(parents=True)
    context = mp.get_context('spawn')
    pipes, workers = [], []
    started = time.monotonic()
    try:
        for _ in range(plan['action_batch_size']):
            parent, child = context.Pipe()
            process = context.Process(target=physics_worker, args=(child,))
            process.start()
            child.close()
            pipes.append(parent)
            workers.append(process)
        batches = [list(range(start, min(start+plan['action_batch_size'], 50)))
                   for start in range(0, 50, plan['action_batch_size'])]
        validity = []
        if args.previous_layout_validity:
            previous = json.loads(args.previous_layout_validity.read_text())
            validity = [r for r in previous['records'] if r['safe_success']]
            expected_layouts = {(c['name'], seed) for c in plan['conditions']
                                for seed in range(plan['layout_seed_start'], plan['layout_seed_start']+50)}
            if {(r['condition'], r['layout_seed']) for r in previous['records']} != expected_layouts:
                raise ValueError('Previous validity evidence is not the identical condition/layout set')
        reused = {(r['condition'], r['layout_seed']) for r in validity}
        validation_transitions = 0
        for condition in plan['conditions']:
            for indices in batches:
                indices = [index for index in indices if (condition['name'], plan['layout_seed_start']+index) not in reused]
                if not indices:
                    continue
                rows = rollout_wave(pipes, condition, indices, plan)
                validation_transitions += sum(r['length'] for r in rows)
                validity.extend([dict(condition=condition['name'], **r) for r in rows])
        atomic_json(args.output / 'layout_validity.json', dict(records=validity,
            safe_detours=sum(r['safe_success'] for r in validity), expected_layouts=250,
            validation_transitions_this_invocation=validation_transitions, training_transitions=0,
            reused_safe_detours=len(reused), previous_artifact=str(args.previous_layout_validity),
            source_commit=args.source_commit))
        if not all(r['safe_success'] for r in validity):
            raise RuntimeError('The fixed detour did not establish every layout; do not run or select policy comparisons yet')
        fixed_layouts = {(r['condition'], r['layout_seed']): r['layout'] for r in validity}
        results = []
        for run in args.runs:
            actor, margin, provenance = load_policy(run, args.device)
            for condition in plan['conditions']:
                records = []
                for indices in batches:
                    records.extend(rollout_wave(pipes, condition, indices, plan, actor, margin, provenance['training_seed']))
                for record in records:
                    reference = fixed_layouts[(condition['name'], record['layout_seed'])]
                    for field in ['start', 'goal', 'obstacle', 'radius_m']:
                        if not np.allclose(record['layout'][field], reference[field], rtol=0., atol=1e-9):
                            raise RuntimeError('Actual layout differs from the shared predeclared condition')
                result = dict(**provenance, condition=condition, **summarize(records), records=records,
                              evaluation_source_commit=args.source_commit, training_transitions=0,
                              adaptation_updates=0, sampling='frozen_native_stochastic_policy')
                results.append(result)
                filename = f"{provenance['method']}-seed{provenance['training_seed']}-{condition['name']}.json"
                atomic_json(args.output / filename, result)
                atomic_json(args.output / 'status.json', dict(state='running', completed_conditions=len(results),
                            expected_conditions=40, elapsed_seconds=time.monotonic()-started))
                print(json.dumps({key: result[key] for key in
                                  ['method', 'training_seed', 'condition', 'episodes', 'safe_success', 'violation']}), flush=True)
            del actor, margin
        atomic_json(args.output / 'receipt.json', dict(status='completed', policy_episodes=2000,
                    completed_conditions=40, training_transitions=0, adaptation_updates=0,
                    evaluation_transitions=sum(r['evaluation_transitions'] for r in results),
                    validation_transitions_this_invocation=validation_transitions,
                    reused_validity_artifact=str(args.previous_layout_validity),
                    plan=plan, source_commit=args.source_commit, device=args.device,
                    elapsed_seconds=time.monotonic()-started))
        atomic_json(args.output / 'status.json', dict(state='completed', completed_conditions=40))
    except BaseException as error:
        atomic_json(args.output / 'status.json', dict(state='engineering_interruption',
                    error=f'{type(error).__name__}: {error}'))
        raise
    finally:
        for pipe in pipes:
            try:
                pipe.send(('close', None))
            except (OSError, EOFError):
                pass
        for process in workers:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()  # Only this invocation's own simulator child.
                process.join()
        for pipe in pipes:
            pipe.close()


if __name__ == '__main__':
    main()

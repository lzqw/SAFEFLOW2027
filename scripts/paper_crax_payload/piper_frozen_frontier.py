"""Registered E02 risk/task frontier and E03 unseen conditions; CPU, no RL/hardware."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch

from scripts.paper_crax_payload.piper_train import load_frozen, make_piper_env
from scripts.star.arm_safe_entropy.stability_runtime import isolated_rng
from scripts.star.arm_safe_entropy.train import atomic_json


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def wilson(events, count):
    if not count:
        return [None, None]
    z = 1.959963984540054
    p, a = events/count, z*z/count
    center = (p+a/2)/(1+a)
    radius = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))/(1+a)
    return [max(0., center-radius), min(1., center+radius)]


def episode_metrics(*, success, length, horizon, dt, positions):
    points = np.asarray(positions, dtype=float)
    return dict(time_to_goal_capped_seconds=(length if success else horizon)*dt,
        time_to_goal_seconds=length*dt if success else None,
        path_length_m=float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()))


def select_scale(candidates, cap):
    eligible = [c for c in candidates if c['static_invalid_layouts'] == 0 and
        c['worst_condition_violation'] <= cap]
    if eligible:
        best = min(eligible, key=lambda c: (-c['macro_safe_success'], c['macro_capped_seconds'], c['action_scale']))
        status = 'risk_qualified' if best['macro_safe_success'] > 0 else 'risk_qualified_but_stagnant'
    else:
        best = min(candidates, key=lambda c: (c['worst_condition_violation'],
            -c['macro_safe_success'], c['macro_capped_seconds'], c['action_scale']))
        status = 'nonqualified_tradeoff'
    return dict(action_scale=best['action_scale'], risk_cap=cap, status=status,
        validation_score=best, independent_test_used=False)


def frontier_score(result, scale):
    values = result['conditions']
    return dict(action_scale=scale,
        static_invalid_layouts=sum(c['static_invalid_layouts'] for c in values),
        worst_condition_violation=max(c['violation'] if c['violation'] is not None else 1. for c in values),
        macro_safe_success=float(np.mean([c['safe_success'] if c['safe_success'] is not None else 0. for c in values])),
        macro_capped_seconds=float(np.mean([c['time_to_goal_capped_seconds'] for c in values])))


def evaluate(learner, config, assets, task, conditions, scale, split, action_seed, heartbeat):
    before = (learner.transitions, learner.updates, learner.actor_updates, learner.safety_multiplier)
    results = []
    with isolated_rng():
        for ci, condition in enumerate(conditions):
            env = make_piper_env(config, assets, task, condition.get('simulation_overrides'))
            rows = []
            try:
                for index in range(split['episodes']):
                    seed = split['layout_seed_start'] + index
                    # Episode-local common random numbers, not shifted by the
                    # previous policy's episode length or another scale choice.
                    torch.manual_seed(action_seed + 10000*ci + index)
                    try:
                        observation, initial = env.reset(seed=seed)
                    except RuntimeError as error:
                        if not str(error).startswith('No safe reachable Piper start/goal'):
                            raise
                        rows.append(dict(layout_seed=seed, static_admissible=False,
                            error=str(error), physical_control_intervals=0))
                        heartbeat(condition['name'], index+1)
                        continue
                    minimum = initial['minimum_distance']
                    if minimum <= env.config.safety_distance_m:
                        raise ValueError('Never evaluate a policy from an unsafe initial state')
                    initial_position = env.task_position().tolist()
                    positions = [initial_position]
                    reward_sum, clipped = 0., 0
                    for step in range(env.config.horizon):
                        raw = learner.action(observation[None])[0]
                        proposed = raw*scale*condition.get('action_scale', 1.)
                        executed = np.clip(proposed, -1., 1.)
                        clipped += int(np.any(executed != proposed))
                        observation, reward, done, truncated, info = env.step(executed)
                        positions.append(env.task_position().tolist())
                        reward_sum += reward
                        minimum = min(minimum, info['minimum_distance'])
                        if done or truncated:
                            break
                    length = step+1
                    rows.append(dict(layout_seed=seed, static_admissible=True,
                        safe_success=bool(info['safe_success']), violation=bool(info['violation']),
                        collision=bool(info['collision']), episode_return=reward_sum,
                        physical_control_intervals=length, clipped_control_intervals=clipped,
                        minimum_distance_m=minimum, initial_minimum_distance_m=initial['minimum_distance'],
                        trajectory=positions, **episode_metrics(success=info['safe_success'], length=length,
                            horizon=env.config.horizon, dt=env.control_dt, positions=positions)))
                    if (index+1) % 10 == 0:
                        heartbeat(condition['name'], index+1)
                valid = [r for r in rows if r['static_admissible']]
                successes = sum(r['safe_success'] for r in valid)
                violations = sum(r['violation'] for r in valid)
                count = len(valid)
                cap = env.config.horizon*env.control_dt
                results.append(dict(condition=condition['name'], episodes_attempted=len(rows),
                    episodes_admitted=count, static_invalid_layouts=len(rows)-count,
                    safe_success_count=successes, violation_count=violations,
                    safe_success=successes/count if count else None,
                    violation=violations/count if count else None,
                    violation_wilson95=wilson(violations, count), success_wilson95=wilson(successes, count),
                    time_to_goal_capped_seconds=float(np.mean([r['time_to_goal_capped_seconds'] for r in valid])) if valid else cap,
                    mean_path_length_m=float(np.mean([r['path_length_m'] for r in valid])) if valid else None,
                    mean_success_path_length_m=float(np.mean([r['path_length_m'] for r in valid if r['safe_success']])) if successes else None,
                    mean_minimum_distance_m=float(np.mean([r['minimum_distance_m'] for r in valid])) if valid else None,
                    physical_control_intervals=env.total_control_intervals,
                    physics_substeps=env.total_physics_substeps, distance_queries=env.total_distance_queries,
                    failure_cap_seconds=cap, records=rows))
            finally:
                env.close()
    if before != (learner.transitions, learner.updates, learner.actor_updates, learner.safety_multiplier):
        raise ValueError('Frozen evaluation changed training or dual counters')
    return dict(task=task, conditions=results, training_transitions=0,
        physical_robot_interactions=0, action_scale=scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan['data_root']).resolve()
    if root not in args.output.resolve().parents or args.output.exists():
        raise ValueError('Use a new output on the registered data disk; preserve previous evidence')
    if plan['device'] != 'cpu' or plan['training_transitions'] != 0 or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This is the registered CPU-only frozen experiment, not another training job')
    if shutil.disk_usage(root).free < plan['storage_headroom_bytes']:
        raise RuntimeError('Less than40GiB data-disk headroom')
    torch.set_num_threads(plan['torch_threads'])
    args.output.mkdir(parents=True)
    started = time.monotonic()
    state = dict(state='loading_frozen_identities', native_pid=os.getpid(),
        source_commit=args.source_commit, plan_sha256=digest(args.plan), training_transitions=0,
        physical_robot_interactions=0, automatic_restart=False, results=[])
    atomic_json(args.output/'plan.json', plan | dict(source_commit=args.source_commit, plan_sha256=state['plan_sha256']))

    def heartbeat(condition=None, episodes=None):
        state.update(condition=condition, active_condition_completed_episodes=episodes,
            elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/'status.json', state)

    identities = {}
    summary_rows, episode_rows = [], []

    def load(stage, method):
        spec = plan[stage]
        identity = identities[stage][method]
        config, assets, saved, learner = load_frozen(SimpleNamespace(config=Path(identity['config']),
            assets_receipt=Path(plan['assets_receipt']), checkpoint=Path(identity['checkpoint']), device='cpu'))
        if (saved['task'] != spec['task'] or saved['method'] != method or
                saved['steps'] != spec['checkpoint_steps'][method] or saved['training_seed'] != spec['training_seed']):
            raise ValueError('The predeclared native candidate identity changed')
        return config, assets, learner

    def run(stage, method, scale, split_name, config, assets, learner, freeze_sha=None):
        spec = plan[stage]
        state.update(state='evaluating', stage=stage, method=method, action_scale=scale, split=split_name)
        heartbeat()
        result = evaluate(learner, config, assets, spec['task'], spec['conditions'], scale,
            spec[split_name], spec['action_sampling_seed_start'], heartbeat)
        result.update(stage=stage, method=method, split=split_name, selection_sha256=freeze_sha,
            frozen_identity=identities[stage][method], source_commit=args.source_commit)
        name = f'{stage}-{split_name}-{method.lower()}-scale{scale:g}.json'
        atomic_json(args.output/name, result)
        state['results'].append(name)
        for condition in result['conditions']:
            fields = {k: v for k, v in condition.items() if k not in ('records','violation_wilson95','success_wilson95')}
            summary_rows.append(dict(stage=stage, split=split_name, method=method, action_scale=scale,
                **fields, violation_ci_low=condition['violation_wilson95'][0],
                violation_ci_high=condition['violation_wilson95'][1]))
            for row in condition['records']:
                episode_rows.append(dict(stage=stage, split=split_name, method=method, action_scale=scale,
                    condition=condition['condition'], **{k:v for k,v in row.items() if k != 'trajectory'}))
        heartbeat()
        return result

    try:
        # Resolve and hash ALL identities before acquiring any new labels.
        for stage in ('E02', 'E03'):
            prior = json.loads(Path(plan[stage]['prior_selection']).read_text())
            identities[stage] = {}
            for method in plan[stage]['methods']:
                candidate = prior['methods'][method]['candidate']
                value = digest(candidate['checkpoint'])
                if value != plan[stage]['full_checkpoint_sha256'][method] or value != candidate['full_checkpoint_sha256']:
                    raise ValueError('Previously frozen checkpoint SHA256 mismatch')
                identities[stage][method] = {k:candidate[k] for k in ('checkpoint','config','full_checkpoint_sha256','steps')}
        atomic_json(args.output/'identities.json', identities)
        scores = {}
        for method in plan['E02']['methods']:
            config, assets, learner = load('E02', method)
            scores[method] = [frontier_score(run('E02', method, scale, 'validation', config, assets, learner), scale)
                for scale in plan['E02']['action_scales']]
            del learner
        selection = dict(identities=identities['E02'], all_scale_candidates_frozen=plan['E02']['action_scales'],
            independent_test_used=False, no_final_test_reselection=True,
            candidate_count_each=len(plan['E02']['action_scales']), validation_scores=scores,
            selected={method:{str(cap):select_scale(scores[method], cap) for cap in plan['E02']['risk_caps_each_condition']}
                for method in plan['E02']['methods']})
        atomic_json(args.output/'selection.json', selection)
        freeze_sha = digest(args.output/'selection.json')
        for method in plan['E02']['methods']:
            config, assets, learner = load('E02', method)
            for scale in plan['E02']['action_scales']:
                run('E02', method, scale, 'independent_test', config, assets, learner, freeze_sha)
            del learner
        if digest(args.output/'selection.json') != freeze_sha:
            raise ValueError('Selection changed after test labels')
        for method in plan['E03']['methods']:
            config, assets, learner = load('E03', method)
            run('E03', method, 1., 'independent_test', config, assets, learner)
            del learner
        # Static admissibility must be method-independent on each shared layout.
        admitted = {}
        for row in episode_rows:
            key = (row['stage'],row['split'],row['condition'],row['layout_seed'])
            if key in admitted and admitted[key] != row['static_admissible']:
                raise ValueError('Static admissibility differs across methods/scales; do not pool')
            admitted[key] = row['static_admissible']
        for filename, rows in (('summary.csv', summary_rows), ('episodes.csv', episode_rows)):
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with (args.output/filename).open('x',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        atomic_json(args.output/'receipt.json', dict(status='completed', source_commit=args.source_commit,
            plan_sha256=state['plan_sha256'], selection_sha256=freeze_sha, training_transitions=0,
            attempted_episodes=len(episode_rows), static_admissible_episodes=sum(r['static_admissible'] for r in episode_rows),
            physical_control_intervals=sum(r['physical_control_intervals'] for r in summary_rows),
            physics_substeps=sum(r['physics_substeps'] for r in summary_rows),
            distance_queries=sum(r['distance_queries'] for r in summary_rows),
            static_admissibility_shared=True, elapsed_seconds=time.monotonic()-started,
            physical_robot_interactions=0, actual_deployment_pc_timing=False, automatic_restart=False))
        state['state'] = 'completed'; heartbeat()
    except BaseException as error:
        state.update(state='engineering_failure_no_automatic_retry', error_type=type(error).__name__, error=str(error))
        heartbeat()
        raise


if __name__ == '__main__':
    main()

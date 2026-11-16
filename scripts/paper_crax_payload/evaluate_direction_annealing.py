"""Development only; same dynamics with interval link-distance observation.

All three conditions and every outcome retained. No oracle action selection or
known-scale compensation. execution085 scales the generated action AFTER policy.
"""
import numpy as np
import torch

from scripts.star.arm_safe_entropy.stability_runtime import isolated_rng


def attach_interval_observer(env):
    original = env._advance
    def observed(action, distance_trace=False):
        result = original(action, distance_trace=True)
        env._v6_link_interval_minima = result[-1].min(0)
        return result if distance_trace else result[:-1]
    env._advance = observed


def evaluate(learner, config, absolute_steps):
    from scripts.star.arm_safe_entropy.environment import TaskConfig, make_environment
    plan = config['evaluation']; count = plan['development_episodes_each']; results = []
    with isolated_rng():
        for ci, condition in enumerate(plan['conditions']):
            env = make_environment(TaskConfig(version=config['task_version'], obstacle_radius=condition['radius']))
            attach_interval_observer(env); records = []
            try:
                for index in range(count):
                    seed = plan['development_layout_seed_start']+index
                    torch.manual_seed(plan['action_seed']+ci*10000+index)
                    if torch.cuda.is_available(): torch.cuda.manual_seed_all(plan['action_seed']+ci*10000+index)
                    obs, initial = env.reset(seed=seed); total = 0.; actions = []
                    minimum = initial['minimum_distance']; positions = [env.ee_position().tolist()]
                    for step in range(env.config.horizon):
                        raw = learner.action(obs[None])[0]
                        executed = np.clip(raw*condition['action_scale'], -1., 1.)
                        before = env.ee_position().copy(); distance_before = float(np.linalg.norm(before-env.goal))
                        obs, reward, done, truncated, info = env.step(executed)
                        after = env.ee_position(); minima = env._v6_link_interval_minima
                        minimum = min(minimum, info['minimum_distance']); total += reward
                        actions.append(dict(step=step+1, raw_action=raw.tolist(), executed_action=executed.tolist(),
                            raw_norm=float(np.linalg.norm(raw)), executed_norm=float(np.linalg.norm(executed)),
                            actual_tcp_displacement_m=(after-before).tolist(),
                            actual_tcp_displacement_norm_m=float(np.linalg.norm(after-before)),
                            minimum_distance_m=info['minimum_distance'],
                            protected_link_interval_minima_m=minima.tolist(),
                            closest_protected_link=int(env.protected_links[int(np.argmin(minima))]),
                            distance_to_goal_before_m=distance_before, distance_to_goal_after_m=info['distance_to_goal'],
                            task_progress_m=distance_before-info['distance_to_goal'],
                            safety_clearance_deficit_m=max(0., env.config.safety_distance-info['minimum_distance']),
                            violation=info['violation'], collision=info['collision']))
                        positions.append(after.tolist())
                        if done or truncated: break
                    records.append(dict(layout_seed=seed, safe_success=info['safe_success'],
                        violation=info['violation'], safe_timeout=not info['safe_success'] and not info['violation'],
                        length=step+1, episode_return=total, minimum_distance_m=minimum,
                        capped_seconds=((step+1) if info['safe_success'] else env.config.horizon)*env.control_dt,
                        layout=dict(start=env.start.tolist(), goal=env.goal.tolist(), obstacle=env.obstacle_position.tolist(),
                            radius_m=condition['radius'], initial_minimum_distance_m=initial['minimum_distance']),
                        trajectory_tcp_xyz=positions, control_details=actions,
                        closest_link_scope='actual_control_interval_not_posthoc_counterfactual'))
            finally: env.close()
            results.append(dict(condition=condition['name'], episodes=count,
                safe_success=sum(r['safe_success'] for r in records)/count,
                violation=sum(r['violation'] for r in records)/count,
                safe_timeout=sum(r['safe_timeout'] for r in records)/count,
                capped_seconds=float(np.mean([r['capped_seconds'] for r in records])), records=records))
    return dict(milestone=absolute_steps, split='paired_fork_development_only',
        independent_confirmation=False, training_seed_count=1, parent_seen_for_selection=True,
        layout_seed_start=plan['development_layout_seed_start'], paired_unique_layout_seeds=count,
        episodes=sum(c['episodes'] for c in results), conditions=results,
        native_controls=sum(r['length'] for c in results for r in c['records']),
        direction_weight=learner.current_direction_weight, rho=learner.actor.rho,
        new_training_transitions=0, action_compensation=False)

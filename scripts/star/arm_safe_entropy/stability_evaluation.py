"""Reuse native stochastic full-episode tests with isolated training randomness."""
from types import SimpleNamespace

from scripts.star.arm_safe_entropy.evaluate_main_robustness import physics_pool
from scripts.star.arm_safe_entropy.evaluate_obstacle_robustness import rollout_wave, summarize
from scripts.star.arm_safe_entropy.stability_runtime import isolated_rng


def evaluate_stability(learner,config,training_seed,steps,workers=10):
    evaluation=config['evaluation']
    final=steps==evaluation['final_steps']
    seed_start=(evaluation['final_development_layout_seed_start'] if final
                else evaluation['intermediate_layout_seed_start'])
    plan=dict(task_version=config['task_version'],layout_seed_start=seed_start,
              action_sampling_seed=evaluation['action_sampling_seed'])
    geometry=SimpleNamespace(geometry=learner.geometry) if learner.geometry is not None else None
    results=[]
    # Includes Python/NumPy/Torch CPU/CUDA state. Separate physics clients and
    # environment RNGs also keep evaluation resets out of training streams.
    with isolated_rng(),physics_pool(workers) as pipes:
        for condition in evaluation['conditions']:
            count=(evaluation['final_episodes_per_condition'] if final else
                   evaluation['intermediate_nominal_episodes'] if condition['name']=='nominal'
                   else evaluation['intermediate_stress_episodes_per_condition'])
            records=[]
            for start in range(0,count,workers):
                records.extend(rollout_wave(pipes,condition,list(range(start,min(start+workers,count))),
                                           plan,learner.actor,geometry,training_seed))
            success=sum(bool(r['safe_success']) for r in records)
            violation=sum(bool(r['violation']) for r in records)
            safe_timeout=sum(not r['safe_success'] and not r['violation'] and r['true_horizon_terminal'] for r in records)
            if success+violation+safe_timeout!=count:
                raise ValueError('Every episode must retain success,safe timeout or violating failure')
            results.append(dict(condition=condition,**summarize(records),
                safe_success_count=success,violation_failure_count=violation,safe_timeout_count=safe_timeout,
                records=records))
    return dict(stage='ea_stability_v2',variant=learner.variant,training_seed=training_seed,
        milestone_steps=steps,geometry_beta=min(1.,max(0.,(steps-10000)/20000)),
        fixed_final_1M=final,layout_seed_start=seed_start,training_rng_preserved=True,
        training_transitions=0,adaptation_updates=0,
        physical_evaluation_transitions=sum(r['evaluation_transitions'] for r in results),
        conditions=results)

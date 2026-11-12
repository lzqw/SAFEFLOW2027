"""Bounded official JAX SAC-Lag reference, with explicit API/reporting fixes.

Losses/optimizers/dual use pinned official implementation. Only environment reset
and transition plumbing are repaired: fresh native full-state resets, preserved
pre-reset next observations, and actual collected-cost accounting. The reference
saves final policy/normalizer only, NOT a resumable full learner/replay checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
from jax import numpy as jp
import numpy as np
from crax import envs
from crax.envs.base import Wrapper
from crax.envs.wrappers.training import EpisodeWrapper,VmapWrapper
from crax.training import acting,logger
from crax.training.agents.sac_lag import train as native_train


def merge_reset(done, new, old):
    return jax.tree.map(lambda n,o:jp.where(done.reshape((done.shape[0],)+(1,)*(o.ndim-1)),n,o)
        if hasattr(o,'ndim') and o.ndim else o,new,old)


class CompleteReset(Wrapper):
    """Retain native physics/task, fix autoreset state/observation bookkeeping."""
    def __init__(self, env, episode_length, action_repeat=1,randomization_fn=None):
        if randomization_fn is not None or action_repeat!=1:
            raise ValueError('Registered native state/F1 reference only')
        self.base=env.unwrapped
        super().__init__(EpisodeWrapper(VmapWrapper(env),episode_length,1))

    def reset(self, keys):
        state=self.env.reset(keys)
        state.info.update(paper_reset_keys=keys,terminal_observation=state.obs,
            paper_total_cost=jp.zeros_like(state.done),paper_total_goals=jp.zeros_like(state.done))
        state.info['episode_metrics'].update(paper_total_cost=jp.zeros_like(state.done),
            paper_total_goals=jp.zeros_like(state.done),paper_episode_goals=jp.zeros_like(state.done))
        return state

    def step(self, state, action):
        # Preserve completed metrics for the callback, clear before the NEXT
        # transition so a reset neither leaks old sums nor drops its first step.
        metrics={k:jp.where(state.done,0.,v) for k,v in state.info['episode_metrics'].items()}
        episode_goals=metrics['paper_episode_goals']
        clean=state.replace(done=jp.zeros_like(state.done),info=dict(state.info,
            episode_metrics=metrics,episode_done=jp.zeros_like(state.done)))
        terminal=self.env.step(clean,action)
        goals=jp.rint((terminal.reward-(state.info['last_dist_goal']-
            terminal.metrics['distance_to_goal'])*self.base._reward_distance)/self.base._reward_goal)
        total_cost=state.info['paper_total_cost']+terminal.info['cost']
        total_goals=state.info['paper_total_goals']+goals
        terminal.info['episode_metrics'].update(paper_total_cost=total_cost,
            paper_total_goals=total_goals,paper_episode_goals=episode_goals+goals)
        keys=jax.vmap(lambda k:jax.random.split(k)[0])(state.info['paper_reset_keys'])
        done=terminal.done.astype(bool)
        policy=jax.lax.cond(jp.any(done),lambda _:merge_reset(done,self.reset(keys),terminal),lambda _:terminal,None)
        info=dict(policy.info,terminal_observation=terminal.obs,cost=terminal.info['cost'],
            truncation=terminal.info['truncation'],episode_metrics=terminal.info['episode_metrics'],
            paper_total_cost=total_cost,paper_total_goals=total_goals,paper_reset_keys=keys)
        return policy.replace(done=terminal.done,reward=terminal.reward,metrics=terminal.metrics,info=info)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--steps',type=int,default=1000000);p.add_argument('--seed',type=int,default=41027)
    p.add_argument('--num-envs',type=int,default=256);p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--updates',type=int,default=4);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Do not overwrite a reference')
    if args.steps>3000000 or args.steps<25000:raise ValueError('Bounded1M reference,at most3M extension')
    args.output.mkdir(parents=True);started=time.monotonic();totals={};last_step=0
    metrics_path=args.output/'progress.jsonl'
    def progress(step, values):
        nonlocal last_step
        last_step=max(last_step,int(step))
        row=dict(steps=int(step),elapsed_seconds=time.monotonic()-started,
            metrics={k:float(np.asarray(v).mean()) for k,v in values.items()},**totals)
        with metrics_path.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        (args.output/'status.json').write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    original_logger=logger.MetricsLogger
    class CostLogger(original_logger):
        def update_step_metrics(self,episode_metrics,dones,train_metrics,env_steps):
            totals.update(training_cumulative_cost=float(np.asarray(episode_metrics['paper_total_cost']).sum()),
                training_cumulative_goals=float(np.asarray(episode_metrics['paper_total_goals']).sum()))
            super().update_step_metrics(episode_metrics,dones,train_metrics,env_steps)
    logger.MetricsLogger=CostLogger
    original_actor_step=acting.actor_step
    def actor_step(*a,**kw):
        state,transition=original_actor_step(*a,**kw)
        return state,transition._replace(next_observation=state.info['terminal_observation'])
    acting.actor_step=actor_step
    env=envs.get_environment('safe_goal_point',level=1)
    settings=dict(crax_commit='63504b540705be98024abd675e646ff5e046c6c2',source_commit=args.source_commit,
        env_id='safe_goal_point',difficulty=1,requested_steps=args.steps,seed=args.seed,
        episode_length=env.unwrapped.episode_length,num_envs=args.num_envs,batch_size=args.batch_size,
        grad_updates_per_step=args.updates,replay_exposure=args.updates*args.batch_size/args.num_envs,
        safety_bound=25.,cost_q='official_unbounded_discounted_cost',discount=.99,
        reward_scaling=1.,cost_scaling=1.,normalize_observations=False,
        dual='official_perstep_bound25/horizon;effective_lr0.01/horizon;lambda_max2',
        api_fixes=['fresh_full_state_autoreset','pre_reset_terminal_observation','raw_collected_cost'],
        checkpoint_kind='policy_and_normalizer_only_not_full_resume',training_early_stop=False)
    (args.output/'configuration.json').write_text(json.dumps(settings,indent=2)+'\n')
    try:
        make_policy,params,metrics,eval_env=native_train(environment=env,num_timesteps=args.steps,
            episode_length=env.unwrapped.episode_length,num_envs=args.num_envs,num_eval_envs=50,
            learning_rate=3e-4,discounting=.99,seed=args.seed,batch_size=args.batch_size,
            num_evals=3,normalize_observations=False,max_devices_per_host=1,reward_scaling=1.,
            min_replay_size=20000,max_replay_size=1000000,grad_updates_per_step=args.updates,
            deterministic_eval=False,progress_fn=progress,training_metrics_steps=50000,
            safety_bound=25.,lagrangian_lr=.01,initial_lambda=0.,lambda_max=2.,wrap_env_fn=CompleteReset)
        jax.effects_barrier()
        with (args.output/'policy_normalizer.pkl').open('wb') as stream:pickle.dump(jax.device_get(params),stream)
        result=dict(status='completed',steps=last_step,new_training_transitions=last_step,
            requested_steps=args.steps,source_commit=args.source_commit,
            elapsed_seconds=time.monotonic()-started,**totals,
            final_metrics={k:float(np.asarray(v).mean()) for k,v in metrics.items()},
            evaluation_query_accounting='official3passes_50envs_nativehorizon;separate_from_training',
            evaluation_controls=3*50*env.unwrapped.episode_length,
            full_learner_resume_saved=False,new_real_robot_interactions=0)
        (args.output/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    finally:
        logger.MetricsLogger=original_logger;acting.actor_step=original_actor_step


if __name__=='__main__':main()

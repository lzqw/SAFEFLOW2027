"""Frozen native SAC checkpoints on COMMON DEVELOPMENT layouts only.

One JAX-compiled native environment unroll is reused across every saved Actor/
normalizer checkpoint. No optimizer/training state is loaded or changed. The
710xx independent test family is intentionally rejected by this entry point.
"""
import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np


VALIDATION_SEEDS=(61027,61028,61029,61030,61031)
CRAX_COMMIT='63504b540705be98024abd675e646ff5e046c6c2'


def require_validation_seeds(seeds):
    if not seeds or len(set(seeds))!=len(seeds) or any(seed not in VALIDATION_SEEDS for seed in seeds):
        raise ValueError('Use only distinct predeclared development validation seeds61027..61031;not independent test')


def read_plan(run):
    run=Path(run)
    config=json.loads((run/'configuration.json').read_text())
    status=json.loads((run/'checkpoint_status.json').read_text())
    receipt=json.loads((run/'receipt.json').read_text())
    settings=config['settings']
    if receipt['status']!='completed' or settings.get('engineering_check'):
        raise ValueError('A completed scientific native-SAC lineage is required')
    if settings['crax_commit']!=CRAX_COMMIT or settings['env_id']!='safe_goal_point' or settings['difficulty']!=1:
        raise ValueError('Wrong public native task/version')
    candidates=status['candidate_checkpoints']
    if len(candidates)!=5 or len({c['steps'] for c in candidates})!=5 or len({c['completed_epoch'] for c in candidates})!=5:
        raise ValueError('Retain exactly five distinct prospective candidate checkpoints')
    if sorted(c['completed_epoch'] for c in candidates)!=status['epoch_plan']['candidate_epochs']:
        raise ValueError('Candidates do not follow the recorded prospective selection schedule')
    for candidate in candidates:
        path=run/f"policy_normalizer_{candidate['steps']}.pkl"
        if not path.is_file():raise FileNotFoundError(path)
        candidate['local_path']=str(path)
    return dict(run=str(run),name=run.name,settings=settings,source_commit=config['source_commit'],
                final_steps=receipt['steps'],candidates=sorted(candidates,key=lambda c:c['steps']))


def summarize(raw,cost_budget,control_period):
    raw=np.asarray(raw)
    if raw.ndim!=2 or raw.shape[1]!=5 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Finite[episodes,return/cost/goals/length/first_goal] records required')
    qualified=raw[:,1]<=cost_budget;reached=raw[:,2]>=1
    return dict(episodes=len(raw),native_return=float(raw[:,0].mean()),episode_cost=float(raw[:,1].mean()),
        goals_per_episode=float(raw[:,2].mean()),at_least_one_goal=float(reached.mean()),
        budget_compliant_fraction=float(qualified.mean()),budget_compliant_episodes=int(qualified.sum()),
        budget_compliant_return=float(raw[qualified,0].mean()) if qualified.any() else None,
        capped_first_goal_seconds=float(raw[:,4].mean())*control_period,
        conditional_first_goal_seconds=float(raw[reached,4].mean())*control_period if reached.any() else None,
        records=[dict(native_return=float(r[0]),episode_cost=float(r[1]),goals=float(r[2]),length=int(r[3]),
            capped_first_goal_seconds=float(r[4])*control_period,
            first_goal_seconds=float(r[4])*control_period if r[2]>=1 else None) for r in raw])


def atomic_json(path,payload):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unreplicate_saved_policy(params,jax):
    """Convert the explicitly replicated1-device TrainingState export once.

    Old FullCheckpointManager policy exports retained their pmap leading axis;
    official final `train` params are already unpmapped. Detect by Dense kernel
    rank, require ALL leaves to have the same singleton device axis, and never
    silently discard a real multi-device shard.
    """
    policy_leaves=jax.tree_util.tree_leaves(params[1])
    if not policy_leaves:raise ValueError('Empty native policy parameters')
    maximum_rank=max(np.ndim(value) for value in policy_leaves)
    if maximum_rank==3:
        leaves=jax.tree_util.tree_leaves(params)
        if any(np.ndim(value)<1 or np.shape(value)[0]!=1 for value in leaves):
            raise ValueError('Expected one consistent singleton native pmap device axis')
        return jax.tree_util.tree_map(lambda value:value[0],params),0
    if maximum_rank==2:return params,None
    raise ValueError('Unknown saved native MLP parameter layout')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[61027])
    parser.add_argument('--episodes-per-seed',type=int,default=50)
    parser.add_argument('--analysis-sha256',required=True)
    args=parser.parse_args()
    require_validation_seeds(args.seeds)
    if args.episodes_per_seed not in (30,50):parser.error('Registered development screen uses30or50episodes,not independent final test')
    if sha256(__file__)!=args.analysis_sha256:raise ValueError('Analysis source differs from explicit pre-execution hash')
    if args.output.exists():raise FileExistsError('Refuse to overwrite prior evaluations or restart automatically')
    plans=[read_plan(run) for run in args.runs]
    if len({p['name'] for p in plans})!=len(plans):raise ValueError('Duplicate run input')
    task_keys=('env_id','difficulty','episode_length','physics_timestep','frame_skip','action_dim','obs_dim',
               'action_repeat','normalize_observations','safety_bound')
    if any({k:p['settings'][k] for k in task_keys}!={k:plans[0]['settings'][k] for k in task_keys} for p in plans):
        raise ValueError('Shared compiled evaluator requires matching native task and observation units')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    os.environ.setdefault('JAX_COMPILATION_CACHE_DIR',str(args.output.parent/'jax-compilation-cache'))
    import jax
    from jax import numpy as jp
    from crax import envs
    from crax.training.acme import running_statistics
    from crax.training.agents.sac_lag import networks
    from scripts.paper_crax_payload.native_sac_reference import CompleteReset
    if len(jax.local_devices())!=1 or jax.local_devices()[0].platform!='gpu':
        raise ValueError('Use one explicitly visible shared GPU for this bounded native validation')
    # Pin actual wrapper and official module bytes against EACH training run.
    for plan in plans:
        settings=plan['settings']
        wrapper=importlib.import_module('scripts.paper_crax_payload.native_sac_reference')
        if sha256(wrapper.__file__)!=settings['paper_reset_wrapper_file_sha256']:
            raise ValueError('Reset/terminal semantics changed from the frozen candidate training source')
        for name,expected in settings['native_module_sha256'].items():
            if sha256(importlib.import_module(name).__file__)!=expected:raise ValueError(f'Native dependency source changed:{name}')
    started=time.monotonic();args.output.mkdir(parents=True)
    settings=plans[0]['settings']
    base=envs.get_environment('safe_goal_point',level=1);native=base.unwrapped
    if (native.episode_length!=settings['episode_length'] or native._n_frames!=settings['frame_skip']
            or native.action_size!=settings['action_dim'] or native.observation_size!=settings['obs_dim']):
        raise ValueError('Native effective task dimensions changed')
    environment=CompleteReset(base,native.episode_length)
    normalize=running_statistics.normalize if settings['normalize_observations'] else lambda obs,params:obs
    network=networks.make_sac_lag_networks(observation_size=base.observation_size,
        action_size=base.action_size,preprocess_observations_fn=normalize)
    make_policy=networks.make_inference_fn(network)
    count=args.episodes_per_seed;horizon=native.episode_length
    @jax.jit
    def evaluate(params,reset_keys,noise_key):
        policy=make_policy(params,deterministic=False)
        state=environment.reset(reset_keys)
        totals=jp.zeros((count,4));active=jp.ones((count,),dtype=bool)
        first_goal=jp.full((count,),horizon)
        def step(carry,index):
            state,totals,active,key,first_goal=carry
            key,noise=jax.random.split(key)
            action,_=policy(state.obs,noise)
            following=environment.step(state,action)
            goals=jp.rint((following.reward-(state.info['last_dist_goal']-
                following.metrics['distance_to_goal'])*native._reward_distance)/native._reward_goal)
            increment=jp.stack((following.reward,following.info['cost'],goals,jp.ones_like(following.reward)),-1)
            first_goal=jp.where(active&(totals[:,2]==0)&(goals>0),index+1,first_goal)
            totals=totals+increment*active[:,None]
            return (following,totals,active&~following.done.astype(bool),key,first_goal),None
        final=jax.lax.scan(step,(state,totals,active,noise_key,first_goal),jp.arange(horizon))[0]
        return jp.concatenate((final[1],final[4][:,None]),-1)
    def keys(seed):
        # Exact PointGoalTorch.reset split, not merely the same seed label.
        _,reset=jax.random.split(jax.random.PRNGKey(seed))
        return jax.random.split(reset,count),jax.random.PRNGKey(seed+1)
    policy_data=[]
    for plan in plans:
        for candidate in plan['candidates']:
            path=Path(candidate['local_path'])
            with path.open('rb') as stream:saved=pickle.load(stream)
            if (saved['steps']!=candidate['steps'] or saved['completed_epoch']!=candidate['completed_epoch']
                    or saved.get('normalizer_saved') is not True or len(saved['params'])!=2):
                raise ValueError('Wrong frozen policy/normalizer artifact schema')
            params,replica_axis=unreplicate_saved_policy(saved['params'],jax)
            policy_data.append((plan,candidate,params,sha256(path),replica_axis))
    reset_keys,noise_key=keys(args.seeds[0])
    first_params=jax.device_put(policy_data[0][2])
    compile_started=time.monotonic()
    compiled=evaluate.lower(first_params,reset_keys,noise_key).compile()
    compile_seconds=time.monotonic()-compile_started
    total_planned_controls=len(policy_data)*len(args.seeds)*count*horizon
    atomic_json(args.output/'plan.json',dict(role='development_common_validation_not_independent_test',
        analysis_source_sha256=args.analysis_sha256,seeds=args.seeds,episodes_per_seed=count,
        candidates_per_run=5,runs=[dict(name=p['name'],source_commit=p['source_commit'],settings=p['settings']) for p in plans],
        control_period_seconds=float(native.dt),native_horizon=horizon,frame_skip=native._n_frames,
        environment_instances_created=1,compilation_calls=1,compile_seconds=compile_seconds,
        compilation_generated_physical_controls=0,total_planned_physical_controls=total_planned_controls,
        initial_layouts='identicalPointGoalTorch.resetPRNGsplit;policy_noise_separate;nativegoalrefreshandcostunchanged'))
    rows=[];controls=0
    for plan,candidate,params,policy_sha,replica_axis in policy_data:
        for seed in args.seeds:
            reset_keys,noise_key=keys(seed)
            atomic_json(args.output/'status.json',dict(status='evaluating_specific_candidate',run=plan['name'],
                checkpoint_steps=candidate['steps'],validation_seed=seed,completed_physical_controls=controls,
                in_flight_control_upper_bound=count*horizon,automatic_restart=False))
            evaluation_started=time.monotonic()
            raw=np.asarray(compiled(jax.device_put(params),reset_keys,noise_key))
            elapsed=time.monotonic()-evaluation_started;controls+=count*horizon
            result=summarize(raw,settings['safety_bound'],float(native.dt))
            result.update(run=plan['name'],training_source_commit=plan['source_commit'],checkpoint_steps=candidate['steps'],
                completed_epoch=candidate['completed_epoch'],validation_seed=seed,policy_sha256=policy_sha,
                stored_policy_replica_axis_removed=replica_axis,
                frozen_actor=True,frozen_normalizer=True,optimizer_updates=0,dual_updates=0,
                training_transitions=0,physical_controls=count*horizon,evaluation_seconds=elapsed,
                role='development_common_validation_not_independent_test')
            atomic_json(args.output/f"{plan['name']}__{candidate['steps']}__{seed}.json",result)
            row={key:value for key,value in result.items() if key!='records'};rows.append(row)
            atomic_json(args.output/'status.json',dict(status='candidate_completed',completed_rows=len(rows),
                completed_physical_controls=controls,in_flight_control_upper_bound=0))
            print(json.dumps(row,allow_nan=False),flush=True)
    with (args.output/'COMMON_VALIDATION_RESULTS.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(args.output/'receipt.json',dict(status='completed',analysis_source_sha256=args.analysis_sha256,
        training_transitions=0,optimizer_updates=0,dual_updates=0,real_interactions=0,
        physical_controls=controls,expected_physical_controls=total_planned_controls,
        result_rows=len(rows),candidates_per_run=5,runs=len(plans),seeds=args.seeds,
        episodes_per_seed=count,environment_instances_created=1,compilation_calls=1,
        compile_seconds=compile_seconds,elapsed_seconds=time.monotonic()-started,
        original_checkpoints_modified=False,independent_test_executed=False,
        primary_endpoint_rows_not_replaced=True,automatic_restart=False))


if __name__=='__main__':main()

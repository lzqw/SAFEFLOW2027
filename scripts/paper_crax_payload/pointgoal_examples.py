"""Two prospective native trajectories of the frozen selected Flow policies.

One predeclared development layout61027 per selected model, horizon1000.
Never resample for a prettier outcome, train, select a policy or connect robots.
CPU-only so Piper retains both shared GPU slots.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Retain the first admitted examples; no retry/reselection')
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
    import torch
    import jax
    from jax import numpy as jp
    from crax import envs
    from crax.envs.wrappers.training import EpisodeWrapper,VmapWrapper
    from scripts.paper_crax_payload.crax_models import CumulativeFlowLearner
    from scripts.paper_crax_payload.observation_normalizer import ObservationNormalizer
    torch.set_num_threads(1)
    if any(d.platform!='cpu' for d in jax.local_devices()): raise ValueError('CPU-only illustrative rollout')
    selection=json.loads(args.selection.read_text())
    if selection.get('independent_test_used') is not False: raise ValueError('Require original selection freeze')
    base=envs.get_environment('safe_goal_point',level=1);native=base.unwrapped
    if native.episode_length!=1000: raise ValueError('Preserve native1000-step horizon')
    environment=EpisodeWrapper(VmapWrapper(base),native.episode_length,1)
    reset=jax.jit(environment.reset);step=jax.jit(environment.step)
    _,key=jax.random.split(jax.random.PRNGKey(61027))
    keys=jax.random.split(key,1)
    args.output.mkdir(parents=True)
    plan=dict(layout_seed=61027,episodes_per_model=1,prospective_first_layout=True,
        methods=['stable_iso','ea'],native_horizon=1000,planned_controls=2000,
        role='illustrative_development_trajectories_not_independent_test_or_mechanism',
        training_transitions=0,optimizer_updates=0,physical_robot_interactions=0,
        selection_freeze_digest=selection['freeze_digest'],CPU_only=True,
        layout_resampling_allowed=False,native_goal_refresh_unchanged=True)
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    controls=0;summaries=[];started=time.monotonic();initial_signature=None
    for method in plan['methods']:
        candidate=selection['methods'][method]['selected']
        path=args.run_root/candidate['run']/f"policy_{candidate['checkpoint_steps']}.pt"
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=candidate['policy_sha256']: raise ValueError('Exact selected model required')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        settings=saved['learner']['settings']
        allowed=set(inspect.signature(CumulativeFlowLearner.__init__).parameters)-{'self','device'}
        constructor={k:v for k,v in settings.items() if k in allowed}
        if 'lagrangian_base_lr' in settings:constructor['lagrangian_lr']=settings['lagrangian_base_lr']
        learner=CumulativeFlowLearner(**constructor,device='cpu');learner.load_state_dict(saved['learner'])
        learner.actor.eval().requires_grad_(False);learner.margin_geometry.eval().requires_grad_(False)
        normalizer=ObservationNormalizer(settings['obs_dim'],learner.device,saved['normalizer']['mode'])
        normalizer.load_state_dict(saved['normalizer'])
        state=reset(keys)
        physical=[np.asarray(state.pipeline_state.qpos),np.asarray(state.pipeline_state.qvel),
            np.asarray(state.info['goal_positions']),np.asarray(state.info['hazard_positions'])]
        signature=hashlib.sha256(b''.join(a.tobytes() for a in physical)).hexdigest()
        if initial_signature is not None and signature!=initial_signature: raise ValueError('Initial layouts must match')
        initial_signature=signature
        hazards=dict(positions=np.asarray(state.info['hazard_positions'])[0].tolist(),
            types=[h.hazard_type for h in native._hazard_manager.hazards],
            sizes=[float(h.size) if np.isscalar(h.size) else np.asarray(h.size).tolist() for h in native._hazard_manager.hazards],
            half_extents=np.asarray(native._hazard_half_extents).tolist())
        initial=dict(agent_xy=np.asarray(state.pipeline_state.xpos)[0,native._agent_body,:2].tolist(),
            goals=np.asarray(state.info['goal_positions'])[0].tolist(),hazards=hazards)
        rng=torch.Generator(device='cpu').manual_seed(61028)
        totals=dict(native_return=0.,cost=0.,goals=0.)
        records=[]
        for index in range(1000):
            with torch.no_grad():
                observation=torch.from_numpy(np.array(state.obs,copy=True))
                action,_,_=learner.actor(normalizer.transform(observation),geometry=learner.geometry,generator=rng)
            following=step(state,jp.asarray(action.numpy()))
            reward=float(np.asarray(following.reward)[0]);cost=float(np.asarray(following.info['cost'])[0])
            distance_reward=(float(np.asarray(state.info['last_dist_goal'])[0])-
                float(np.asarray(following.metrics['distance_to_goal'])[0]))*native._reward_distance
            goals=float(np.rint((reward-distance_reward)/native._reward_goal))
            position=np.asarray(following.pipeline_state.xpos)[0,native._agent_body,:2]
            if not np.isfinite(position).all() or not np.isfinite([reward,cost,goals]).all():
                raise ValueError('Nonfinite native illustrative rollout; retain prefix, do not replace')
            records.append(dict(step=index+1,position_xy=position.tolist(),action=action[0].tolist(),
                native_reward=reward,native_cost=cost,native_goals=goals,
                goals_after=np.asarray(following.info['goal_positions'])[0].tolist()))
            totals['native_return']+=reward;totals['cost']+=cost;totals['goals']+=goals
            controls+=1;state=following
            if (index+1)%100==0:
                (args.output/'status.json').write_text(json.dumps(dict(state='running',method=method,
                    completed_controls=controls,pid=os.getpid()))+'\n')
            if bool(np.asarray(following.done)[0]): break
        summary=dict(method=method,selected_run=candidate['run'],selected_steps=candidate['checkpoint_steps'],
            policy_sha256=digest,initial_physical_state_sha256=signature,controls=len(records),**totals)
        (args.output/f'{method}.json').write_text(json.dumps(dict(summary=summary,initial=initial,records=records),indent=2)+'\n')
        summaries.append(summary)
        del learner,saved
    receipt=dict(status='completed',summaries=summaries,actual_native_controls=controls,
        planned_native_control_upper_bound=2000,training_transitions=0,optimizer_updates=0,
        physical_robot_interactions=0,no_layout_or_policy_reselection=True,
        not_independent_test=True,elapsed_seconds=time.monotonic()-started)
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (args.output/'status.json').write_text(json.dumps(dict(state='completed',completed_controls=controls,pid=os.getpid()))+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()

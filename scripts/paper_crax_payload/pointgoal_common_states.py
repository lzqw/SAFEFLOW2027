"""Prepared SMALL PointGoal common-state mechanism, NOT run without admission.

32 genuine frozen MJX states x16 draws x2 Actors x2 samplers, plus160
finite-difference labels =2208 native one-control-step queries, zero SGD.
Oracle labels are computed only AFTER all learned actions are fixed; they
never choose/replace an action. The actual native horizon remains1000.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch


def select_strata(margins,fingerprints,seen,seed,count=16,band=.15):
    """Prospective observable-state stratification, not oracle-action filtering."""
    margins=np.asarray(margins,dtype=float)
    if len(margins)!=len(fingerprints) or not np.isfinite(margins).all():raise ValueError('Invalid state candidates')
    available=[];local_seen=set(seen)
    for i,key in enumerate(fingerprints):
        if key not in local_seen:available.append(i);local_seen.add(key)
    if len(available)<count:raise ValueError('Not enough distinct physical snapshot states;never duplicate/pad')
    rng=np.random.default_rng(seed);selected=[]
    def take(group,n):
        candidates=[i for i in group if i in available and i not in selected]
        if candidates:selected.extend(rng.choice(candidates,size=min(n,len(candidates)),replace=False).tolist())
    unsafe=[i for i in available if margins[i]>0]
    near=[i for i in available if -band<=margins[i]<=0]
    interior=[i for i in available if margins[i]<-band]
    take(unsafe,min(2,count//2));take(near,count//2-len(selected));take(interior,count-len(selected))
    take(available,count-len(selected))
    return sorted(selected)


def endpoint_projection(actions,normal):
    actions=np.asarray(actions,dtype=float);normal=np.asarray(normal,dtype=float)
    if actions.ndim!=2 or len(actions)<2 or actions.shape[1]<2:raise ValueError('Within-state multivariate draws required')
    matrix=np.cov(actions,rowvar=False,ddof=1)
    result=dict(covariance=matrix.tolist(),trace=float(np.trace(matrix)),vn=None,vt=None,
                common_axis_valid=bool(np.linalg.norm(normal)>1e-7))
    if result['common_axis_valid']:
        normal=normal/np.linalg.norm(normal)
        vn=float(normal@matrix@normal)
        result.update(vn=vn,vt=(float(np.trace(matrix))-vn)/(actions.shape[1]-1))
    return result


def static_proxy(native,state):
    """Exact same endpoint-clearance PROXY used by PointGoalTorch margin labels."""
    position=np.asarray(state.pipeline_state.xpos)[:,native._agent_body,:2]
    delta=position[:,None,:]-np.asarray(state.info['hazard_positions'])[...,:2]
    half=np.asarray(native._hazard_half_extents)
    rect=np.array([h.hazard_type=='rect' for h in native._hazard_manager.hazards])
    sizes=np.array([float(h.size) if np.isscalar(h.size) else 0. for h in native._hazard_manager.hazards])
    circle=np.linalg.norm(delta,axis=-1)-sizes
    q=np.abs(delta)-half
    box=np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(np.max(q,axis=-1),0)
    return -np.min(np.where(rect,box,circle),axis=-1)


def physical_fingerprints(state):
    arrays=[np.asarray(state.pipeline_state.qpos),np.asarray(state.pipeline_state.qvel),
            np.asarray(state.info['goal_positions']),np.asarray(state.info['hazard_positions']),
            np.asarray(state.info['respawn_rng'])]
    return [hashlib.sha256(b''.join(value[i].tobytes() for value in arrays)).hexdigest() for i in range(len(state.obs))]


def atomic_json(path,payload):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def file_sha(path):
    return subprocess.run(['sha256sum','--',str(path)],check=True,capture_output=True,text=True).stdout.split()[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints',nargs=2,type=Path,required=True,metavar=('ISO_FULL','EA_FULL'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--analysis-sha256',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--minimum-training-steps',type=int,default=2000000)
    parser.add_argument('--seed',type=int,default=8630027)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('No overwrite or automatic restart of an admitted analysis')
    if file_sha(__file__)!=args.analysis_sha256:raise ValueError('Analysis source not pinned to admitted bytes')
    import jax
    from jax import numpy as jp
    from scripts.paper_crax_payload.crax_env import PointGoalTorch
    from scripts.paper_crax_payload.crax_models import CumulativeFlowLearner,UpdateGeometry
    from scripts.paper_crax_payload.observation_normalizer import ObservationNormalizer
    if torch.device(args.device).type!='cuda' or len(jax.local_devices())!=1:raise ValueError('One shared admitted GPU required')
    torch.cuda.set_device(args.device);torch.set_num_threads(1);started=time.monotonic()
    loaded=[]
    for path in args.checkpoints:
        if path.name=='full_latest.pt':
            receipt_path=path.parent/'receipt.json'
            if not receipt_path.is_file() or json.loads(receipt_path.read_text())['status']!='completed':
                raise ValueError('Active parent requires an immutable numbered full snapshot,not changing full_latest')
        before=path.stat();digest=file_sha(path)
        state=torch.load(path,map_location='cpu',weights_only=False)
        after=path.stat()
        if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError('Snapshot changed while being read')
        required={'learner','environment','normalizer','replay','partial','rng','rollout_rng','steps','source_commit'}
        if not required<=set(state) or state['steps']<args.minimum_training_steps:raise ValueError('Genuine sufficiently trained fullMJX checkpoint required')
        settings=state['learner']['settings']
        permitted=set(inspect.signature(CumulativeFlowLearner.__init__).parameters)-{'self','device'}
        constructor={k:v for k,v in settings.items() if k in permitted}
        if 'lagrangian_base_lr' in settings:constructor['lagrangian_lr']=settings['lagrangian_base_lr']
        learner=CumulativeFlowLearner(**constructor,device=args.device)
        learner.load_state_dict(state['learner']);learner.actor.requires_grad_(False)
        learner.margin_geometry.requires_grad_(False)
        normalizer=ObservationNormalizer(settings['obs_dim'],learner.device,state['normalizer']['mode'])
        normalizer.load_state_dict(state['normalizer'])
        loaded.append(dict(path=path,state=state,learner=learner,normalizer=normalizer,sha256=digest,
            stat=(before.st_ino,before.st_size,before.st_mtime_ns)))
    if {x['learner'].method for x in loaded}!={'stable_iso','ea'}:raise ValueError('Retain both Iso and EA,not two good Actors')
    shared=('obs_dim','action_dim','variance','rho','flow_steps')
    if any(tuple(x['learner'].settings[k] for k in shared)!=tuple(loaded[0]['learner'].settings[k] for k in shared) for x in loaded):
        raise ValueError('Shared common-noise comparison requires the same action/reference/discretization dimensions')
    environment=PointGoalTorch(32,args.seed,args.device);native=environment.native
    effective=environment.effective_config()
    physical_keys=('env_id','difficulty','episode_length','physics_timestep','frame_skip','control_period',
        'action_dim','obs_dim','goals','hazards','hazard_types','hazard_collidable','reward_goal',
        'reward_distance_scale','proximity_cost_scale','collision_cost','native_goal_refresh','cost_termination_added')
    if effective['episode_length']!=1000:raise ValueError('No shortened native episodes')
    selected_parts=[];pool=[];seen=set()
    for source_index,item in enumerate(loaded):
        saved_env=item['state']['environment']
        if any(saved_env['config'][k]!=effective[k] for k in physical_keys):raise ValueError('Native physical/task profile changed')
        source=saved_env['state'];n=saved_env['num_envs']
        if source.obs.shape[0]!=n:raise ValueError('Unexpected MJX batch/shard shape')
        margins=static_proxy(native,source);fingerprints=physical_fingerprints(source)
        indices=select_strata(margins,fingerprints,seen,args.seed+source_index)
        def select(value):
            if not hasattr(value,'ndim') or value.ndim==0:return value
            if value.shape[0]!=n:raise ValueError('Unexpected nonbatched snapshot array leaf')
            return jp.asarray(value)[jp.asarray(indices)]
        selected_parts.append(jax.tree_util.tree_map(select,source))
        for index in indices:
            seen.add(fingerprints[index])
            pool.append(dict(source_index=source_index,source_method=item['learner'].method,
                source_checkpoint_steps=item['state']['steps'],source_environment_index=index,
                physical_state_sha256=fingerprints[index],state_margin_proxy_m=float(margins[index]),
                state_stratum='unsafe_proxy' if margins[index]>0 else 'near_boundary' if margins[index]>=-.15 else 'interior',
                observation=np.asarray(source.obs[index]).tolist()))
    def join(a,b):
        if hasattr(a,'ndim') and a.ndim>0:return jp.concatenate((a,b),axis=0)
        if not np.array_equal(a,b):raise ValueError('Shared static state leaf differs')
        return a
    common=jax.tree_util.tree_map(join,*selected_parts)
    strata={name:sum(r['state_stratum']==name for r in pool) for name in ('unsafe_proxy','near_boundary','interior')}
    if len(pool)!=32 or len(seen)!=32 or strata['near_boundary']+strata['unsafe_proxy']<8 or strata['interior']<8:
        raise ValueError('Require32distinct mixed-source states with both near-boundary and interior strata')
    args.output.mkdir(parents=True)
    atomic_json(args.output/'pool.json',dict(records=pool,strata=strata,old_pool_reused=False,
        selection='16distinct states/source;prioritizeup-to2unsafe,8near-or-unsafe,remainderinterior;nooracle-actionfilter',
        selection_margin_scope='observable static endpoint-clearance proxy,not native cost or interval minimum',
        native_resets_executed=0))
    draws=16;dimension=environment.action_dim
    torch_obs=torch.utils.dlpack.from_dlpack(common.obs)
    batch=torch_obs.repeat_interleave(draws,0)
    common_device=loaded[0]['learner'].device
    common_noise=torch.randn((loaded[0]['learner'].actor.num_steps,len(batch),dimension),device=common_device,
        generator=torch.Generator(device=common_device).manual_seed(args.seed+20))
    interventions=[];owners=[]
    for item in loaded:
        learner=item['learner'];obs=item['normalizer'].transform(batch)
        for sampler in ('iso_reference','endpoint_pullback'):
            geometry=None if sampler=='iso_reference' else UpdateGeometry(learner.margin_geometry,'endpoint_pullback')
            if geometry is not None:geometry.set_margin_updates(learner.margin_updates)
            with torch.no_grad():actions,energy,diag=learner.actor(obs,geometry=geometry,noise_increments=common_noise)
            owners.append(actions)
            interventions.append(dict(method=learner.method,checkpoint_steps=item['state']['steps'],sampler=sampler,
                actions=actions,energy=energy,clipping=diag['clipping'],activation=diag['geometry_weight']))
    if any(not torch.isfinite(x['actions']).all() for x in interventions):
        raise FloatingPointError('Nonfinite frozen actions;no native queries have executed')
    # All learned actions are now fixed. Oracle finite differences are LABELS
    # only and cannot affect any sampler or action-selection branch above.
    torch.cuda.synchronize()
    positions=lambda state:state.pipeline_state.xpos[:,native._agent_body,:2]
    rect=jp.array([h.hazard_type=='rect' for h in native._hazard_manager.hazards])
    sizes=jp.array([float(h.size) if np.isscalar(h.size) else 0. for h in native._hazard_manager.hazards])
    @jax.jit
    def native_query(state,actions):
        terminal=environment.env.step(state,actions)
        delta=positions(terminal)[:,None,:]-terminal.info['hazard_positions'][...,:2]
        q=jp.abs(delta)-native._hazard_half_extents
        box=jp.linalg.norm(jp.maximum(q,0),axis=-1)+jp.minimum(jp.max(q,axis=-1),0)
        circle=jp.linalg.norm(delta,axis=-1)-sizes
        margin=-jp.min(jp.where(rect,box,circle),axis=-1)
        progress=state.info['last_dist_goal']-terminal.metrics['distance_to_goal']
        goals=jp.rint((terminal.reward-progress*native._reward_distance)/native._reward_goal)
        source_goals=state.info['goal_positions'][...,:2]
        center_before=jp.linalg.norm(positions(state)[:,None,:]-source_goals,axis=-1).min(-1)
        center_after=jp.linalg.norm(positions(terminal)[:,None,:]-source_goals,axis=-1).min(-1)
        return jp.concatenate((terminal.info['cost'][:,None],goals[:,None],progress[:,None],
            (positions(terminal)-positions(state)),margin[:,None],terminal.done[:,None],
            (center_before-center_after)[:,None]),axis=-1)
    def repeat_state(state,count):
        return jax.tree_util.tree_map(lambda v:jp.repeat(v,count,axis=0) if hasattr(v,'ndim') and v.ndim else v,state)
    epsilon=.05
    offsets=np.concatenate((np.zeros((1,dimension)),epsilon*np.eye(dimension),-epsilon*np.eye(dimension)),axis=0).astype(np.float32)
    fd_actions=jp.tile(jp.asarray(offsets),(32,1));fd_state=repeat_state(common,len(offsets))
    expected_fd=32*(1+2*dimension);expected_samples=len(interventions)*32*draws
    atomic_json(args.output/'plan.json',dict(analysis_sha256=args.analysis_sha256,states=32,draws_per_state=16,
        checkpoints=[dict(path=str(i['path']),sha256=i['sha256'],steps=i['state']['steps'],source_commit=i['state']['source_commit'],
            method=i['learner'].method,normalizer_mode=i['normalizer'].mode,
            frozen_normalizer_count=float(i['normalizer'].count.cpu())) for i in loaded],
        runtime_source_sha256=dict(learner=file_sha(inspect.getfile(CumulativeFlowLearner)),
            environment=file_sha(inspect.getfile(PointGoalTorch)),normalizer=file_sha(inspect.getfile(ObservationNormalizer))),
        physical_profile=effective,fd_epsilon=epsilon,planned_fd_controls=expected_fd,
        planned_sampling_controls=expected_samples,planned_total_controls=expected_fd+expected_samples,
        training_transitions=0,optimizer_steps=0,independent_test=False,
        matching='sameActor/commonGaussianincrements;Iso-vs-PBsameinjectiontrace,notmatchedspectrumorendpointtrace'))
    atomic_json(args.output/'status.json',dict(status='fd_label_query',completed_controls=0,in_flight_upper_bound=expected_fd))
    fd=np.asarray(native_query(fd_state,fd_actions)).reshape(32,len(offsets),-1)
    if not np.isfinite(fd).all():
        atomic_json(args.output/'receipt.json',dict(status='failed_nonfinite_native_FD_labels',physical_controls=expected_fd,
            training_transitions=0,optimizer_steps=0,real_interactions=0,automatic_restart=False))
        raise FloatingPointError('Native FD labels nonfinite;retain actual charged prefix')
    gradient=(fd[:,1:1+dimension,5]-fd[:,1+dimension:,5])/(2*epsilon)
    magnitude=np.linalg.norm(gradient,axis=-1)
    normals=np.divide(gradient,magnitude[:,None],out=np.zeros_like(gradient),where=magnitude[:,None]>1e-7)
    sample_state=repeat_state(common,draws);rows=[];per_state=[];controls=expected_fd
    for value in interventions:
        atomic_json(args.output/'status.json',dict(status='sample_label_query',method=value['method'],sampler=value['sampler'],
            completed_controls=controls,in_flight_upper_bound=32*draws))
        # Modern object protocol plus retained Torch owners avoids temporary
        # external-buffer lifetime pitfalls; no per-control host action copies.
        action_jax=jax.dlpack.from_dlpack(value['actions'].detach().contiguous())
        labels=np.asarray(native_query(sample_state,action_jax)).reshape(32,draws,-1);controls+=32*draws
        if not np.isfinite(labels).all():
            atomic_json(args.output/'receipt.json',dict(status='failed_nonfinite_native_sample_labels',physical_controls=controls,
                training_transitions=0,optimizer_steps=0,real_interactions=0,automatic_restart=False))
            raise FloatingPointError('Native sample labels nonfinite;retain actual charged prefix')
        actions=value['actions'].detach().cpu().numpy().reshape(32,draws,dimension)
        state_rows=[]
        for index in range(32):
            state_rows.append(dict(state_index=index,source_stratum=pool[index]['state_stratum'],
                native_mean_cost=float(labels[index,:,0].mean()),native_cost_positive_fraction=float((labels[index,:,0]>0).mean()),
                native_mean_goals=float(labels[index,:,1].mean()),native_distance_reward_delta_m=float(labels[index,:,2].mean()),
                mean_source_goal_center_progress_m=float(labels[index,:,7].mean()),
                mean_displacement_m=float(np.linalg.norm(labels[index,:,3:5],axis=-1).mean()),
                endpoint_proxy_unsafe_fraction=float((labels[index,:,5]>0).mean()),
                native_done_fraction=float(labels[index,:,6].mean()),
                action_endpoint=endpoint_projection(actions[index],normals[index])))
        valid=[r['action_endpoint'] for r in state_rows if r['action_endpoint']['common_axis_valid']]
        rows.append(dict(method=value['method'],checkpoint_steps=value['checkpoint_steps'],sampler=value['sampler'],states=32,draws=16,
            native_mean_cost=float(labels[:,:,0].mean()),native_cost_positive_fraction=float((labels[:,:,0]>0).mean()),
            native_mean_goals=float(labels[:,:,1].mean()),native_distance_reward_delta_m=float(labels[:,:,2].mean()),
            mean_source_goal_center_progress_m=float(labels[:,:,7].mean()),
            endpoint_proxy_unsafe_fraction=float((labels[:,:,5]>0).mean()),valid_common_action_axes=len(valid),
            endpoint_vn=float(np.mean([x['vn'] for x in valid])) if valid else None,
            endpoint_vt=float(np.mean([x['vt'] for x in valid])) if valid else None,
            clipping_fraction=float(value['clipping'].mean().cpu()),geometry_activation=float(value['activation'].mean().cpu())))
        per_state.append(dict(method=value['method'],sampler=value['sampler'],states=state_rows))
    for item in loaded:
        stat=item['path'].stat()
        if (stat.st_ino,stat.st_size,stat.st_mtime_ns)!=item['stat']:raise ValueError('Input checkpoint changed during analysis')
    atomic_json(args.output/'results.json',dict(summaries=rows,per_state=per_state,
        source_zero_action_oracle=fd[:,0].tolist(),common_action_normals=normals.tolist(),common_gradient_magnitudes=magnitude.tolist(),
        normal_scope='native one-step endpoint-clearance PROXY finite differences at commonzeroaction;not cost-Q or interval-minimum certificate'))
    atomic_json(args.output/'receipt.json',dict(status='completed',analysis_sha256=args.analysis_sha256,
        states=32,draws_per_state=16,models=2,samplers_per_actor=2,physical_controls=controls,
        finite_difference_controls=expected_fd,sampled_action_controls=expected_samples,
        native_horizon=native.episode_length,
        training_transitions=0,optimizer_steps=0,real_interactions=0,independent_test=False,
        oracle_actions_selected=0,native_resets_executed=0,elapsed_seconds=time.monotonic()-started,
        scope='small training-distribution common-state mechanism;not main episode evaluation'))


if __name__=='__main__':main()

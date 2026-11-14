"""CPU-only E04: method-independent constructive whole-arm route qualification."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from deployment.piper.bundle import FrozenPiperPolicy
from scripts.paper_crax_payload.piper_train import make_piper_env
from scripts.star.arm_safe_entropy.train import atomic_json


def shifted_reset(env, seed, shift):
    obs,info=env.reset(seed=seed)
    env.obstacle_position=env.obstacle_position+np.asarray([shift,0.,0.])
    env.b('resetBasePositionAndOrientation',env.obstacle,env.obstacle_position,[0.,0.,0.,1.])
    arm,payload=env.distances()
    admissible=min(arm,payload)>env.config.safety_distance_m and env._goal_is_admissible()
    return env._obs(),dict(initial_safe=bool(min(arm,payload)>env.config.safety_distance_m),
        goal_safe_reachable=bool(admissible),minimum_distance=min(arm,payload),
        start=env.task_position().tolist(),goal=env.goal.tolist(),obstacle=env.obstacle_position.tolist())


def rollout(env, seed, shift, *, plan, side=None, policy=None, replay_actions=None):
    obs,initial=shifted_reset(env,seed,shift)
    if not initial['initial_safe'] or not initial['goal_safe_reachable']:
        return dict(admissible=False,initial=initial,controls=0,positions=[],actions=[])
    positions=[env.task_position().tolist()];actions=[];minimum=initial['minimum_distance']
    if side is not None:
        offset=np.array([side*plan['constructive_detour_x_m'],0.,plan['constructive_lift_z_m']])
        waypoints=[np.asarray(initial['start'])+offset,env.goal+offset,env.goal.copy()]
    waypoint=0;info=None
    limit=env.config.horizon if replay_actions is None else len(replay_actions)
    for step in range(limit):
        if side is not None:
            position=env.task_position()
            if waypoint<2 and np.linalg.norm(position-waypoints[waypoint])<plan['waypoint_tolerance_m']:
                waypoint+=1
            raw=(waypoints[waypoint]-position)/env.config.displacement_scale_m
        elif replay_actions is not None:raw=replay_actions[step]
        else:raw=policy.action(obs[None])[0]
        action=np.clip(raw,-1,1)
        obs,_,done,truncated,info=env.step(action)
        minimum=min(minimum,info['minimum_distance']);positions.append(env.task_position().tolist())
        actions.append(action.tolist())
        if done or truncated:break
    success=bool(info and info['safe_success']);violation=bool(info and info['violation'])
    nearest=min(positions,key=lambda p:abs(p[1]-initial['obstacle'][1]))
    return dict(admissible=True,initial=initial,safe_success=success,violation=violation,
        safe_timeout=not success and not violation,controls=len(actions),minimum_distance_m=minimum,
        capped_seconds=(len(actions) if success else env.config.horizon)*env.control_dt,
        route_side='negative_x' if nearest[0]<initial['obstacle'][0] else 'nonnegative_x',
        actual_y_plane_crossed=any((a[1]-initial['obstacle'][1])*(b[1]-initial['obstacle'][1])<=0
            for a,b in zip(positions[:-1],positions[1:])),positions=positions,actions=actions)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    if args.output.exists() or Path(plan['data_root']).resolve() not in args.output.resolve().parents:
        raise ValueError('Preserve evidence; use a fresh registered data-disk output')
    if plan['device']!='cpu' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Bounded CPU simulation only, no GPU or physical transport')
    torch.set_num_threads(1);args.output.mkdir();started=time.monotonic()
    config=json.loads(Path(plan['frozen_config']).read_text())
    assets=json.loads(Path(plan['assets_receipt']).read_text())
    env=make_piper_env(config,assets,'Reach')
    witnesses=[];admitted=[];records=[]
    def heartbeat(stage):
        atomic_json(args.output/'status.json',dict(stage=stage,native_pid=os.getpid(),
            attempted_layout_pairs=len(witnesses),admitted_pairs=len(admitted),policy_episodes=len(records),
            controls=env.total_control_intervals,elapsed_seconds=time.monotonic()-started))
    atomic_json(args.output/'plan.json',dict(plan=plan,source_commit=args.source_commit))
    try:
        for seed in range(plan['layout_seed_start'],plan['layout_seed_start']+plan['maximum_layout_attempts']):
            trials=[]
            try:
                for shift,side in [(0.,-1),(0.,1),(-plan['obstacle_shift_x_m'],-1),
                    (-plan['obstacle_shift_x_m'],1),(plan['obstacle_shift_x_m'],-1),
                    (plan['obstacle_shift_x_m'],1)]:
                    trials.append(rollout(env,seed,shift,plan=plan,side=side))
                ok=(all(t['admissible'] for t in trials) and
                    all(trials[i]['safe_success'] for i in (0,1,3,4)) and
                    all(trials[i]['violation'] for i in (2,5)))
                row=dict(layout_seed=seed,admitted=bool(ok),trials=trials)
            except RuntimeError as error:
                row=dict(layout_seed=seed,admitted=False,reset_error=str(error),trials=trials);ok=False
            witnesses.append(row)
            if ok:admitted.append(seed)
            heartbeat('qualifying_fixed_routes_no_policy_loaded')
            if len(admitted)==plan['target_admitted_pairs']:break
        freeze=dict(admitted_seeds=admitted,qualification_policy_independent=True,
            full_arm_and_gripper_checked=True,payload_present=env.payload is not None,
            all_physics_substeps_checked=True,source_commit=args.source_commit,witnesses=witnesses)
        atomic_json(args.output/'layout_freeze.json',freeze)
        freeze_hash=hashlib.sha256((args.output/'layout_freeze.json').read_bytes()).hexdigest()
        # Do not load a learned policy, let alone revise geometry based on it,
        # until the complete symmetric constructive qualification is frozen.
        if admitted:
            for method,bundle in plan['bundles'].items():
                policy=FrozenPiperPolicy(Path(bundle),'cpu')
                if policy.manifest['policy_sha256']!=plan['policy_sha256'][method]:
                    raise ValueError('Previously frozen policy identity changed')
                for index,seed in enumerate(admitted):
                    torch.manual_seed(plan['action_seed_start']+index)
                    nominal=rollout(env,seed,0.,plan=plan,policy=policy)
                    records.append(dict(method=method,layout_seed=seed,condition='nominal',**nominal))
                    for direction in (-1,1):
                        shift=direction*plan['obstacle_shift_x_m']
                        torch.manual_seed(plan['action_seed_start']+index)
                        result=rollout(env,seed,shift,plan=plan,policy=policy)
                        # Diagnostic only: replay exactly the learned nominal
                        # action sequence. No proposal selection/reranking.
                        replay=rollout(env,seed,shift,plan=plan,replay_actions=nominal['actions'])
                        records.append(dict(method=method,layout_seed=seed,condition=f'shift_{direction}',
                            nominal_action_replay=replay,**result))
                        heartbeat('frozen_policies_on_prequalified_pairs')
        if hashlib.sha256((args.output/'layout_freeze.json').read_bytes()).hexdigest()!=freeze_hash:
            raise ValueError('Layout freeze changed after policy labels')
        summary=[]
        for method in plan['bundles']:
            for condition in ('nominal','shift_-1','shift_1'):
                rows=[r for r in records if r['method']==method and r['condition']==condition]
                if not rows:continue
                summary.append(dict(method=method,condition=condition,episodes=len(rows),
                    safe_success=float(np.mean([r['safe_success'] for r in rows])),
                    violation=float(np.mean([r['violation'] for r in rows])),
                    safe_timeout=float(np.mean([r['safe_timeout'] for r in rows])),
                    capped_seconds=float(np.mean([r['capped_seconds'] for r in rows])),
                    actual_nominal_route_blocked_fraction=float(np.mean([r['nominal_action_replay']['violation'] for r in rows]))
                        if condition!='nominal' else None))
        atomic_json(args.output/'result.json',dict(summary=summary,records=records))
        atomic_json(args.output/'receipt.json',dict(status='completed',source_commit=args.source_commit,
            qualified_changed_layouts=2*len(admitted),target_changed_layouts=2*plan['target_admitted_pairs'],
            full_target_met=len(admitted)==plan['target_admitted_pairs'],policy_episodes=len(records),
            constructive_attempts=len(witnesses),controls=env.total_control_intervals,
            physics_substeps=env.total_physics_substeps,distance_queries=env.total_distance_queries,
            layout_freeze_sha256=freeze_hash,elapsed_seconds=time.monotonic()-started,
            new_training_transitions=0,physical_robot_interactions=0,
            more_training_routes_discovered_claim=False,
            actual_usual_route_restriction_requires_nominal_replay_analysis=True))
        heartbeat('completed')
    finally:env.close()


if __name__=='__main__':main()

"""Small prospective terminal-action DEV probe; labels never select an action."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from deployment.piper.simulation import PiperSimConfig
from scripts.paper_crax_payload.diagnose_native_piper_v11 import DiagnosticPiper
from scripts.paper_crax_payload.native_p5_adaptation import build_native, target_deployment
from scripts.paper_crax_payload.train_native_p5_warmup_v12 import attach, FORMAT
from scripts.paper_crax_payload.train_native_p5_adaptation import sha256, storage_check
from scripts.star.arm_safe_entropy.train import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('plan','protocol','runs','assets-receipt','data-root','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    args=p.parse_args();torch.set_num_threads(1)
    plan=json.loads(args.plan.read_text());spec=json.loads(args.protocol.read_text())
    expected=Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_reach_terminal_probe.json'
    if spec!=json.loads(expected.read_text()):raise ValueError('Use exact prospective protocol')
    assets=json.loads(args.assets_receipt.read_text())
    if sha256(assets['source_urdf'])!=assets['source_sha256'] or sha256(assets['resolved_urdf'])!=assets['resolved_sha256']:
        raise ValueError('Pinned assets changed')
    if not args.output.resolve().is_relative_to(args.data_root.resolve(strict=True)):
        raise ValueError('Output must remain on data disk')
    storage_check(args.data_root);args.output.mkdir(exist_ok=False,parents=True)
    rows=[];visits=[];work=[];cases=[]
    for arm in spec['arms']:
        d=args.runs/('piper_reach-'+arm);receipt=json.loads((d/'receipt.json').read_text())
        if receipt['status']!='completed' or receipt['new_training_steps']!=100000:raise ValueError('Require completed100K')
        c=d/'full_new_100000.pt';cases.append(dict(arm=arm,checkpoint=str(c),sha256=sha256(c)))
    atomic_json(args.output/'registration.json',dict(protocol=spec,cases=cases,source_commit=args.source_commit))
    for case in cases:
        arm=case['arm'];saved=torch.load(case['checkpoint'],map_location='cpu',weights_only=False)
        line=plan['lines']['piper_reach']
        if saved['format']!=FORMAT or saved['plan']!=plan or saved['arm']!=arm or saved['steps']!=500000:
            raise ValueError('Exact registered fixed endpoint required')
        native=saved['configuration'];learner=build_native(native,line,'cpu');learner.set_transitions(400000)
        attach(learner,plan,'piper_reach',arm);learner.load_state_dict(saved['learner']);del saved
        learner.margin_geometry.requires_grad_(False)
        env=DiagnosticPiper(assets,PiperSimConfig(task='Reach',**(dict(native['simulation'])|dict(obstacle_radius_m=spec['radius_m']))))
        try:
            with target_deployment(learner):
                for i in range(spec['episodes_per_arm']):
                    seed=spec['layout_seed_start']+i;torch.manual_seed(spec['action_seed_start']+i)
                    obs,_=env.reset(seed=seed)
                    for t in range(env.config.horizon):
                        action=learner.action(obs[None])[0]
                        variable=torch.as_tensor(action[None]).requires_grad_(True)
                        heads=learner.margin_geometry(torch.as_tensor(obs[None]),variable)
                        grad=torch.autograd.grad(heads.mean(),variable)[0][0].numpy();pred=float(heads.detach().mean())
                        snapshot=env.b('saveState');elapsed,done,previous=env.elapsed,env._done,env.previous_goal_distance
                        env.closest_interval=None;position=env.task_position().copy()
                        try:
                            obs,_,terminal,truncated,info=env.step(action)
                            row=dict(arm=arm,seed=seed,t=t,predicted_margin_m=pred,true_margin_m=info['local_margin'],
                                error_m=pred-info['local_margin'],false_safe=bool(pred<=0<info['local_margin']),
                                violation=bool(info['violation']),safe_success=bool(info['safe_success']))
                            visits.append(row)
                            if terminal or truncated:
                                nearest=env.closest_interval;probes=[action]
                                for axis in np.eye(3):
                                    probes.extend([np.clip(action+spec['action_difference_epsilon']*axis,-1,1),np.clip(action-spec['action_difference_epsilon']*axis,-1,1)])
                                labels=[]
                                for probe in probes:
                                    env.b('restoreState',snapshot);env.elapsed,env._done,env.previous_goal_distance=elapsed,done,previous
                                    _,_,_,_,label=env.step(probe);labels.append(label['local_margin'])
                                if not np.isclose(labels[0],info['local_margin'],rtol=0,atol=2e-7):raise ValueError('Behavior replay changed')
                                truth=np.array([(labels[1+2*j]-labels[2+2*j])/(probes[1+2*j][j]-probes[2+2*j][j]) for j in range(3)])
                                denom=np.linalg.norm(grad)*np.linalg.norm(truth)
                                rows.append(row|dict(gradient_cosine=float(grad@truth/denom) if denom>1e-12 else None,
                                    closest_part=nearest[1],closest_link=nearest[2],action=action.tolist(),
                                    predicted_gradient=grad.tolist(),true_finite_difference=truth.tolist(),
                                    native_temperature_m=learner.margin_geometry.temperature))
                                break
                        finally:env.b('removeState',snapshot)
        finally:
            work.append(dict(arm=arm,controls=env.total_control_intervals,substeps=env.total_physics_substeps,distance_calls=env.total_distance_queries))
            env.close()
    for name,values in [('terminals.csv',rows),('visited.csv',visits)]:
        with (args.output/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    summary=[]
    for arm in spec['arms']:
        group=[r for r in rows if r['arm']==arm];bad=[r for r in group if r['violation']]
        summary.append(dict(arm=arm,episodes=len(group),violations=len(bad),false_safe=sum(r['false_safe'] for r in bad),
            unsafe_errors_mm=[1000*r['error_m'] for r in bad],unsafe_gradient_cosines=[r['gradient_cosine'] for r in bad]))
    atomic_json(args.output/'receipt.json',dict(status='completed',summary=summary,work=work,new_training_steps=0,
        independent_test_used=False,physical_robot_interactions=0,counterfactual_action_selection=False))


if __name__=='__main__':main()

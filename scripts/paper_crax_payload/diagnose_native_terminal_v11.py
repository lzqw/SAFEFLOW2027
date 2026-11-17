"""Prospective fixed-new-layout terminal diagnosis; no training or action selection."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from deployment.piper.simulation import PiperSimConfig
from scripts.paper_crax_payload.diagnose_native_piper_v11 import DiagnosticPiper
from scripts.paper_crax_payload.native_p5_adaptation import (
    FORMAT, attach_adaptation, build_native, target_deployment)
from scripts.paper_crax_payload.train_native_p5_adaptation import sha256, storage_check
from scripts.star.arm_safe_entropy.train import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'protocol', 'runs', 'assets-receipt', 'data-root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    args = p.parse_args(); torch.set_num_threads(1)
    plan = json.loads(args.plan.read_text()); protocol = json.loads(args.protocol.read_text())
    assets = json.loads(args.assets_receipt.read_text())
    if (sha256(assets['source_urdf']) != assets['source_sha256']
            or sha256(assets['resolved_urdf']) != assets['resolved_sha256']):
        raise ValueError('Pinned official simulation assets changed')
    if protocol['protocol'] != 'v11_final100k_prospective_terminal_diagnostic':
        raise ValueError('Use the registered terminal diagnostic')
    if not args.output.resolve().is_relative_to(args.data_root.resolve(strict=True)):
        raise ValueError('Diagnostic output must remain on the data disk')
    storage_check(args.data_root)
    cases = []
    # Require all four completed endpoints before allocating a diagnostic run.
    for line_name in protocol['lines']:
        for arm in protocol['arms']:
            directory = args.runs/(line_name+'-'+arm)
            receipt = json.loads((directory/'receipt.json').read_text())
            if receipt['status'] != 'completed' or receipt['new_training_steps'] != 100000:
                raise ValueError('Wait for full new100K endpoints')
            cases.append(dict(line=line_name, arm=arm, checkpoint=str(directory/'full_new_100000.pt'),
                checkpoint_sha256=sha256(directory/'full_new_100000.pt'),
                receipt_sha256=sha256(directory/'receipt.json')))
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output/'registration.json', dict(protocol=protocol, cases=cases,
        source_commit=args.source_commit, automatic_retry=False, independent_test_used=False))
    visits, terminals, work = [], [], []
    for case in cases:
        line_name, arm = case['line'], case['arm']; line = plan['lines'][line_name]
        spec = protocol['lines'][line_name]
        saved = torch.load(case['checkpoint'], map_location='cpu', weights_only=False)
        if (saved['format'] != FORMAT or saved['line'] != line_name or saved['arm'] != arm
                or saved['steps'] != line['parent_steps']+100000 or saved['plan'] != plan):
            raise ValueError('Only the identical registered fixed100K native checkpoint is admitted')
        native = saved['configuration']
        if native['model_commit'] != assets['model_commit']:
            raise ValueError('Checkpoint and official Piper asset commit differ')
        learner = build_native(native, line, 'cpu'); learner.set_transitions(line['parent_steps'])
        attach_adaptation(learner, plan, line_name, arm); learner.load_state_dict(saved['learner'])
        del saved
        learner.actor.requires_grad_(False); learner.target_actor.requires_grad_(False)
        learner.margin_geometry.requires_grad_(False)
        env = DiagnosticPiper(assets, PiperSimConfig(task=line['task'],
            **(dict(native['simulation']) | dict(obstacle_radius_m=spec['radius_m']))))
        try:
            with target_deployment(learner):
                for episode in range(protocol['episodes_per_case']):
                    seed = spec['layout_seed_start']+episode
                    torch.manual_seed(spec['action_seed_start']+episode)
                    obs, _ = env.reset(seed=seed)
                    for t in range(env.config.horizon):
                        action = learner.action(obs[None])[0]
                        tensor = torch.as_tensor(obs[None]); variable = torch.as_tensor(action[None]).requires_grad_(True)
                        heads = learner.margin_geometry(tensor, variable)
                        gradient = torch.autograd.grad(heads.mean(), variable)[0][0].numpy()
                        predicted = float(heads.detach().mean())
                        snapshot = env.b('saveState')
                        elapsed, done, previous = env.elapsed, env._done, env.previous_goal_distance
                        position = env.task_position().copy(); env.closest_interval = None
                        try:
                            next_obs, _, terminal, truncated, info = env.step(action)
                            nearest = env.closest_interval
                            common = dict(line=line_name, arm=arm, layout_seed=seed, state_time=t,
                                predicted_margin_m=predicted, true_margin_m=info['local_margin'],
                                margin_error_m=predicted-info['local_margin'],
                                false_safe=bool(predicted<=0. and info['local_margin']>0.),
                                violation=bool(info['violation']), safe_success=bool(info['safe_success']))
                            visits.append(common)
                            if terminal or truncated:
                                probes = [action]
                                epsilon = protocol['action_difference_epsilon']
                                for axis in np.eye(3):
                                    probes.extend([np.clip(action+epsilon*axis,-1.,1.),np.clip(action-epsilon*axis,-1.,1.)])
                                labels = []; responses = []
                                for probe in probes:
                                    env.b('restoreState', snapshot)
                                    env.elapsed, env._done, env.previous_goal_distance = elapsed, done, previous
                                    env.closest_interval = None
                                    _, _, _, _, label = env.step(probe)
                                    labels.append(label['local_margin'])
                                    responses.append((env.task_position()-position).tolist())
                                if not np.isclose(labels[0],info['local_margin'],rtol=0,atol=2e-7):
                                    raise ValueError('Exact-state behavior replay changed its interval label')
                                actual = np.array([(labels[1+2*j]-labels[2+2*j]) /
                                    (probes[1+2*j][j]-probes[2+2*j][j]) for j in range(3)])
                                norm = np.linalg.norm(gradient)*np.linalg.norm(actual)
                                terminals.append(common | dict(
                                    interval_minimum_distance_m=info['minimum_distance'],
                                    closest_part=nearest[1], closest_link=nearest[2],
                                    predicted_action_gradient=gradient.tolist(), true_action_finite_difference=actual.tolist(),
                                    gradient_cosine=float(np.dot(gradient,actual)/norm) if norm>1e-12 else None,
                                    action=action.tolist(), actual_task_displacement_m=responses[0],
                                    commanded_tcp_displacement_m=(action*env.config.displacement_scale_m).tolist(),
                                    native_margin_temperature=learner.margin_geometry.temperature,
                                    finite_difference_actions=[np.asarray(a).tolist() for a in probes[1:]]))
                                break
                            obs = next_obs
                        finally:
                            env.b('removeState', snapshot)
        finally:
            work.append(dict(line=line_name, arm=arm, simulation_controls=env.total_control_intervals,
                physics_substeps=env.total_physics_substeps, distance_calls=env.total_distance_queries,
                extra_closest_queries=env.closest_queries))
            env.close()
    for name, rows in [('visited_actions',visits),('terminal_actions',terminals)]:
        with (args.output/(name+'.csv')).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary=[]
    for case in cases:
        rows=[r for r in terminals if r['line']==case['line'] and r['arm']==case['arm']]
        unsafe=[r for r in rows if r['violation']]
        summary.append(dict(line=case['line'],arm=case['arm'],episodes=len(rows),
            successes=sum(r['safe_success'] for r in rows),violations=len(unsafe),
            false_safe_terminal_count=sum(r['false_safe'] for r in unsafe),
            unsafe_mean_margin_error_m=float(np.mean([r['margin_error_m'] for r in unsafe])) if unsafe else None,
            unsafe_gradient_cosines=[r['gradient_cosine'] for r in unsafe]))
    atomic_json(args.output/'summary.json',dict(status='completed',summary=summary,work=work,
        new_training_steps=0,physical_robot_interactions=0,independent_test_used=False,
        all_simulation_controls=sum(w['simulation_controls'] for w in work),
        behavior_actions_selected_by_counterfactuals=False))


if __name__ == '__main__':
    main()

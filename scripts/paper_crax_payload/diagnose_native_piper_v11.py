"""Small prospective DEV-only native Piper margin/dynamics diagnosis on CPU.

Counterfactual physics labels never select a behavior action. Exact Bullet
save/restore keeps the physical state common to all small-action differences.
All extra simulated controls, substeps and closest-point queries are counted.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from deployment.piper.simulation import PiperSimConfig
from scripts.paper_crax_payload.piper_helpers import PiperTrainingEnvironment
from scripts.paper_crax_payload.native_p5_adaptation import build_native, validate_parent, target_deployment
from scripts.paper_crax_payload.train_native_p5_adaptation import sha256, storage_check
from scripts.star.arm_safe_entropy.train import atomic_json


class DiagnosticPiper(PiperTrainingEnvironment):
    def __init__(self, *args, **kwargs):
        self.closest_queries = 0
        self.closest_interval = None
        super().__init__(*args, **kwargs)

    def distances(self):
        values = super().distances()
        points = self.b('getClosestPoints', self.robot, self.obstacle, distance=2.)
        self.closest_queries += 1
        candidates = [(float(p[8]), 'arm', int(p[3])) for p in points if p[3] in self.protected_links]
        if self.payload is not None:
            points = self.b('getClosestPoints', self.payload, self.obstacle, distance=2.)
            self.closest_queries += 1
            candidates.extend((float(p[8]), 'payload', -1) for p in points)
        nearest = min(candidates, default=(2., 'none', -1))
        if self.closest_interval is None or nearest[0] < self.closest_interval[0]:
            self.closest_interval = nearest
        return values


def run(args):
    torch.set_num_threads(1)
    plan = json.loads(args.plan.read_text())
    assets = json.loads(args.assets_receipt.read_text())
    root = args.data_root.resolve(strict=True)
    if not args.output.resolve().is_relative_to(root):
        raise ValueError('Diagnostic output must remain on the admitted data disk')
    storage_check(root)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, missing, provenance = [], [], []
    for line_name in ('piper_reach', 'piper_payload'):
        line = plan['lines'][line_name]
        parent = torch.load(line['parent_path'], map_location='cpu', weights_only=False)
        validate_parent(parent, plan, line_name, sha256(line['parent_path']))
        learner = build_native(parent['configuration'], line, 'cpu')
        learner.load_state_dict(parent['learner'])
        learner.actor.requires_grad_(False); learner.target_actor.requires_grad_(False)
        learner.margin_geometry.requires_grad_(False)
        native = parent['configuration']
        del parent
        radii = (.045, .0575, .06) if line['task']=='Reach' else (.045, .055)
        for radius in radii:
            env = DiagnosticPiper(assets, PiperSimConfig(task=line['task'],
                **(dict(native['simulation']) | dict(obstacle_radius_m=radius))))
            try:
                with target_deployment(learner):
                    for index in range(plan['diagnostic']['layouts']):
                        layout = line['diagnostic_layout_seed_start']+index
                        torch.manual_seed(line['action_sampling_seed']+2000+index)
                        obs, _ = env.reset(seed=layout)
                        visited = []
                        for t in range(max(plan['diagnostic']['state_times'])+1):
                            action = learner.action(obs[None])[0]
                            if t in plan['diagnostic']['state_times']:
                                visited.append(t)
                                snapshot = env.b('saveState')
                                elapsed, done, previous = env.elapsed, env._done, env.previous_goal_distance
                                position = env.task_position().copy()
                                tensor = torch.as_tensor(obs[None])
                                variable = torch.as_tensor(action[None]).requires_grad_(True)
                                heads = learner.margin_geometry(tensor, variable)
                                gradient = torch.autograd.grad(heads.mean(), variable)[0][0].numpy()
                                epsilon = plan['diagnostic']['action_difference_epsilon']
                                actions = [action]
                                for axis in np.eye(3):
                                    actions.extend([np.clip(action+epsilon*axis,-1.,1.), np.clip(action-epsilon*axis,-1.,1.)])
                                labels, responses, nearest = [], [], []
                                def restore():
                                    env.b('restoreState', snapshot)
                                    env.elapsed, env._done, env.previous_goal_distance = elapsed, done, previous
                                    env.closest_interval = None
                                try:
                                    for probe in actions:
                                        restore()
                                        _, _, _, _, info = env.step(probe)
                                        labels.append(info)
                                        responses.append((env.task_position()-position).tolist())
                                        nearest.append(env.closest_interval)
                                    actual = np.array([(labels[1+2*j]['local_margin']-labels[2+2*j]['local_margin']) /
                                        (actions[1+2*j][j]-actions[2+2*j][j]) for j in range(3)])
                                    predicted = float(heads.detach().mean())
                                    truth = labels[0]['local_margin']
                                    norm = float(np.linalg.norm(gradient)*np.linalg.norm(actual))
                                    rows.append(dict(line=line_name, layout_seed=layout, state_time=t, radius_m=radius,
                                        predicted_margin_m=predicted, true_margin_m=truth, margin_error_m=predicted-truth,
                                        false_safe=bool(predicted<=0. and truth>0.),
                                        interval_minimum_distance_m=labels[0]['minimum_distance'],
                                        closest_part=nearest[0][1], closest_link=nearest[0][2],
                                        predicted_action_gradient=gradient.tolist(), true_action_finite_difference=actual.tolist(),
                                        gradient_cosine=float(np.dot(gradient,actual)/norm) if norm>1e-12 else None,
                                        action=action.tolist(), actual_task_displacement_m=responses[0],
                                        commanded_tcp_displacement_m=(action*env.config.displacement_scale_m).tolist(),
                                        finite_difference_actions=[np.asarray(a).tolist() for a in actions[1:]]))
                                finally:
                                    restore()
                                    env.b('removeState', snapshot)
                            if t==max(plan['diagnostic']['state_times']):
                                break
                            obs, _, done, truncated, _ = env.step(action)
                            if done or truncated:
                                break
                        missing.extend(dict(line=line_name, layout_seed=layout, radius_m=radius, state_time=t)
                            for t in plan['diagnostic']['state_times'] if t not in visited)
            finally:
                provenance.append(dict(line=line_name, radius_m=radius,
                    parent_sha256=line['parent_sha256'], controls=env.total_control_intervals,
                    physics_substeps=env.total_physics_substeps, native_distance_calls=env.total_distance_queries,
                    diagnostic_extra_closest_point_queries=env.closest_queries))
                env.close()
        del learner
    with (args.output/'samples.csv').open('x', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for line_name in ('piper_reach','piper_payload'):
        for radius in sorted({r['radius_m'] for r in rows if r['line']==line_name}):
            group=[r for r in rows if r['line']==line_name and r['radius_m']==radius]
            cosine=[r['gradient_cosine'] for r in group if r['gradient_cosine'] is not None]
            summary.append(dict(line=line_name,radius_m=radius,samples=len(group),
                false_safe_count=sum(r['false_safe'] for r in group),
                true_unsafe_count=sum(r['true_margin_m']>0 for r in group),
                mean_margin_error_m=float(np.mean([r['margin_error_m'] for r in group])),
                mean_absolute_margin_error_m=float(np.mean([abs(r['margin_error_m']) for r in group])),
                mean_gradient_cosine=float(np.mean(cosine)) if cosine else None))
    atomic_json(args.output/'summary.json',dict(status='completed',protocol='v11_prospective_native_margin_diagnosis',
        new_training_steps=0,physical_robot_interactions=0,source_commit=args.source_commit,
        samples=len(rows),missing_prospective_states=missing,summary=summary,work=provenance,
        total_simulated_controls=sum(p['controls'] for p in provenance),
        actor='original_native_parent_existing_target',finite_difference_state='exact Bullet save/restore',
        actions_not_selected_from_counterfactual_labels=True,independent_test_used=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--assets-receipt',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    run(p.parse_args())


if __name__=='__main__':
    main()

"""Frozen terminal noise attribution; counterfactual labels never choose actions."""
import argparse
import csv
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from deployment.piper.simulation import PiperSimConfig
from scripts.paper_crax_payload.diagnose_native_piper_v11 import DiagnosticPiper
from scripts.paper_crax_payload.native_p5_adaptation import build_native, target_deployment
from scripts.paper_crax_payload.train_native_p5_adaptation import sha256, storage_check
from scripts.star.arm_safe_entropy.stability_runtime import isolated_rng
from scripts.star.arm_safe_entropy.train import atomic_json


PLAN = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_terminal_noise_probe.json'


def load(case, device='cpu'):
    module = importlib.import_module(case['module'])
    checkpoint = Path(case['checkpoint'])
    if sha256(checkpoint) != case['checkpoint_sha256']:
        raise ValueError('Frozen completed checkpoint changed')
    receipt = json.loads(Path(case['receipt']).read_text())
    if receipt['status'] != 'completed' or receipt['new_training_steps'] != 100000:
        raise ValueError('Require the original complete100K native endpoint')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    training = json.loads(Path(case['plan']).read_text())
    line = training['lines'][case['line']]
    if (saved['format'] != module.FORMAT or saved['plan'] != training or
            saved['line'] != case['line'] or saved['arm'] != case['arm'] or
            saved['steps'] != line['parent_steps']+100000):
        raise ValueError('Native format, training plan, arm and budget must match')
    learner = build_native(saved['configuration'], line, device)
    learner.set_transitions(line['parent_steps'])
    module.attach(learner, training, case['line'], case['arm'])
    learner.load_state_dict(saved['learner'])
    return learner, saved['configuration'], line


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'assets-receipt', 'data-root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    args = p.parse_args(); torch.set_num_threads(1)
    spec = json.loads(args.plan.read_text()); assets = json.loads(args.assets_receipt.read_text())
    if spec != json.loads(PLAN.read_text()):
        raise ValueError('Use exact prospective diagnostic specification')
    if not args.output.resolve().is_relative_to(args.data_root.resolve(strict=True)):
        raise ValueError('Keep all new data on the admitted data disk')
    storage_check(args.data_root)
    if sha256(assets['source_urdf']) != assets['source_sha256'] or sha256(assets['resolved_urdf']) != assets['resolved_sha256']:
        raise ValueError('Pinned official assets changed')
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output/'registration.json', dict(protocol=spec, source_commit=args.source_commit,
        plan_sha256=sha256(args.plan), automatic_retry=False, independent_test_used=False))
    rows = []; work = []
    for case in spec['cases']:
        learner, native, line = load(case)
        env = DiagnosticPiper(assets, PiperSimConfig(task=line['task'],
            **(dict(native['simulation']) | dict(obstacle_radius_m=case['radius_m']))))
        try:
            with target_deployment(learner):
                for episode in range(spec['episodes_per_case']):
                    layout = case['layout_seed_start']+episode
                    torch.manual_seed(case['action_seed_start']+episode)
                    obs, _ = env.reset(seed=layout)
                    for t in range(env.config.horizon):
                        tensor = torch.as_tensor(obs[None])
                        with torch.no_grad():
                            action, _, _, trace = learner.actor(tensor, geometry=learner.geometry, return_reference=True)
                        action = action[0].numpy()
                        # No random draw and no diagnostic actor trajectory affects behavior RNG.
                        with isolated_rng(), torch.no_grad():
                            deterministic, _, _, clean_trace = learner.actor(tensor, geometry=learner.geometry,
                                deterministic=True, return_reference=True)
                        clean = deterministic[0].numpy()
                        with torch.no_grad():
                            predicted = float(learner.margin_online(tensor, torch.as_tensor(action[None])).mean())
                            predicted_clean = float(learner.margin_online(tensor, torch.as_tensor(clean[None])).mean())
                        before = env.b('saveState')
                        elapsed, done, previous = env.elapsed, env._done, env.previous_goal_distance
                        env.closest_interval = None
                        try:
                            next_obs, _, terminal, truncated, info = env.step(action)
                            if terminal or truncated:
                                after = env.b('saveState')
                                actual_elapsed, actual_done, actual_previous = env.elapsed, env._done, env.previous_goal_distance
                                nearest = env.closest_interval
                                try:
                                    env.b('restoreState', before)
                                    env.elapsed, env._done, env.previous_goal_distance = elapsed, done, previous
                                    _, _, _, _, clean_info = env.step(clean)
                                    summary = dict(case=case['name'], line=case['line'], layout_seed=layout,
                                        terminal_step=t+1, safe_success=bool(info['safe_success']), violation=bool(info['violation']),
                                        safe_timeout=bool(not info['safe_success'] and not info['violation']),
                                        behavior_margin_m=info['local_margin'], deterministic_margin_m=clean_info['local_margin'],
                                        deterministic_violation=bool(clean_info['violation']),
                                        sampled_unsafe_clean_safe=bool(info['violation'] and not clean_info['violation']),
                                        predicted_online_margin_m=predicted, predicted_online_clean_margin_m=predicted_clean,
                                        action=action.tolist(), deterministic_action=clean.tolist(),
                                        action_noise_difference_norm=float(np.linalg.norm(action-clean)),
                                        behavior_mean_weight=float(trace['weights'].mean()),
                                        behavior_mean_vn_over_vt=float((trace['vn']/trace['vt']).mean()),
                                        clean_mean_weight=float(clean_trace['weights'].mean()),
                                        clean_mean_vn_over_vt=float((clean_trace['vn']/clean_trace['vt']).mean()),
                                        geometry_valid_fraction=float((~trace['geometry_zero_gradient']).float().mean()),
                                        closest_part=nearest[1], closest_link=nearest[2])
                                    if 'velocities' in trace:
                                        summary['mean_outward_speed'] = float((trace['velocities']*trace['normals']).sum(-1).clamp_min(0).mean())
                                    rows.append(summary)
                                finally:
                                    env.b('restoreState', after); env.b('removeState', after)
                                    env.elapsed, env._done, env.previous_goal_distance = actual_elapsed, actual_done, actual_previous
                                break
                            obs = next_obs
                        finally:
                            env.b('removeState', before)
        finally:
            work.append(dict(case=case['name'], controls=env.total_control_intervals,
                physics_substeps=env.total_physics_substeps, distance_calls=env.total_distance_queries))
            env.close()
    with (args.output/'terminals.csv').open('x', newline='') as stream:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    summary = []
    for case in spec['cases']:
        group = [r for r in rows if r['case'] == case['name']]; unsafe = [r for r in group if r['violation']]
        summary.append(dict(case=case['name'], episodes=len(group), violations=len(unsafe),
            sampled_unsafe_clean_safe=sum(r['sampled_unsafe_clean_safe'] for r in group),
            unsafe_mean_vn_over_vt=float(np.mean([r['behavior_mean_vn_over_vt'] for r in unsafe])) if unsafe else None,
            unsafe_mean_weight=float(np.mean([r['behavior_mean_weight'] for r in unsafe])) if unsafe else None))
    atomic_json(args.output/'receipt.json', dict(status='completed', summary=summary, work=work,
        new_training_steps=0, independent_test_used=False, physical_robot_interactions=0,
        counterfactual_action_selection=False,
        interpretation='same terminal state action comparison only;not whole-policy deterministic evaluation or unique causal proof'))


if __name__ == '__main__': main()

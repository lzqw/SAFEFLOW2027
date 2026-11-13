"""One fixed, frozen Piper margin/gradient diagnostic; no training or hardware.

The development pool is prospective: ten existing development layouts, and
the same four time indices on each original EA trajectory. Early terminations
are reported, never replaced with more discriminating states. Physical
counterfactuals label actions; they never choose a behavior action.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from scripts.paper_crax_payload.piper_helpers import capture_piper_environment, restore_piper_environment
from scripts.paper_crax_payload.piper_train import load_frozen, make_piper_env
from scripts.star.arm_safe_entropy.endpoint_geometry import EndpointGeometry
from scripts.star.arm_safe_entropy.train import atomic_json


LAYOUTS = tuple(range(650000, 650010))
STATE_TIMES = (0, 10, 20, 30)
RADII = (.045, .055)


class RadiusResponse(nn.Module):
    """Diagnostic spherical-radius prior, not a dynamics/safety guarantee."""
    def __init__(self, margin, reference_radius=.045):
        super().__init__()
        self.margin = margin
        self.temperature = margin.temperature
        self.reference_radius = reference_radius

    def forward(self, observation, action):
        canonical = observation.clone()
        canonical[..., 21] = self.reference_radius
        return self.margin(canonical, action) + observation[..., 21] - self.reference_radius


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def counterfactual(env, state, action):
    """Restore actual physical state; count only work done after restoration."""
    adjusted = deepcopy(state)
    adjusted['observation'][21] = env.config.obstacle_radius_m
    restore_piper_environment(env, adjusted)
    before = (env.total_control_intervals, env.total_physics_substeps, env.total_distance_queries)
    _, _, _, _, info = env.step(np.asarray(action))
    after = (env.total_control_intervals, env.total_physics_substeps, env.total_distance_queries)
    return info, tuple(a-b for a, b in zip(after, before))


def summarize(rows):
    result = []
    for task in ('Reach', 'Payload'):
        for radius in RADII:
            group = [r for r in rows if r['task'] == task and r['radius_m'] == radius]
            for model in ('original', 'radius_response'):
                predicted = np.array([r[model+'_margin_m'] for r in group])
                truth = np.array([r['true_margin_m'] for r in group])
                safe = predicted <= 0
                unsafe = truth > 0
                valid = np.array([r['gradient_valid'] for r in group], dtype=bool)
                response = np.array([r['true_plus_minus_margin_m'] for r in group])
                result.append(dict(task=task, radius_m=radius, model=model, samples=len(group),
                    mean_error_m=float(np.mean(predicted-truth)),
                    mean_absolute_error_m=float(np.mean(abs(predicted-truth))),
                    true_unsafe_count=int(unsafe.sum()), predicted_safe_count=int(safe.sum()),
                    false_safe_count=int((safe & unsafe).sum()),
                    false_safe_among_predicted_safe=float((safe & unsafe).sum()/safe.sum()) if safe.any() else None,
                    missed_unsafe_fraction=float((safe & unsafe).sum()/unsafe.sum()) if unsafe.any() else None,
                    valid_gradient_count=int(valid.sum()),
                    mean_true_outward_minus_inward_margin_m=float(response[valid].mean()) if valid.any() else None,
                    outward_increases_true_risk_fraction=float((response[valid] > 1e-6).mean()) if valid.any() else None,
                    near_zero_physical_direction_response_fraction=float((abs(response[valid]) <= 1e-6).mean()) if valid.any() else None,
                    direction_is_original_model_finite_difference=True))
    return result


def run(args):
    if args.output.exists():
        raise FileExistsError('Preserve the fixed development experiment; no pool reselection')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    rows, episodes, pools, provenance = [], [], {}, []
    work = np.zeros(3, dtype=np.int64)
    for task in ('Reach', 'Payload'):
        checkpoint = args.original_root/'runs'/f'piper-{task.lower()}-ea-41027-v1/full_300000.pt'
        config, assets, saved, learner = load_frozen(SimpleNamespace(config=args.config,
            assets_receipt=args.assets_receipt, checkpoint=checkpoint, device='cpu'))
        if saved['steps'] != 300000 or saved['task'] != task or saved.get('engineering'):
            raise ValueError('Expected the frozen original native EA300K candidate')
        learner.actor.eval().requires_grad_(False)
        learner.margin_geometry.eval().requires_grad_(False)
        corrected = RadiusResponse(learner.margin_geometry)
        provenance.append(dict(task=task, checkpoint=str(checkpoint), sha256=sha256(checkpoint),
            training_seed=saved['training_seed'], source_commit=saved['source_commit']))
        env = make_piper_env(config, assets, task)
        pool, missing = [], []
        try:
            for layout in LAYOUTS:
                torch.manual_seed(691000+layout-LAYOUTS[0])
                obs, _ = env.reset(seed=layout)
                visited = []
                for time in range(STATE_TIMES[-1]+1):
                    if time in STATE_TIMES:
                        state = capture_piper_environment(env)
                        pool.append(dict(layout=layout, time=time, state=state))
                        visited.append(time)
                    if time == STATE_TIMES[-1]:
                        break
                    action = learner.action(obs[None])[0]
                    obs, _, done, truncated, _ = env.step(action)
                    if done or truncated:
                        break
                missing.extend(dict(layout=layout, time=t) for t in STATE_TIMES if t not in visited)
            pools[task] = dict(states=pool, missing_prospective_states=missing)
        finally:
            pool_work = dict(controls=env.total_control_intervals, substeps=env.total_physics_substeps,
                distance_queries=env.total_distance_queries)
            env.close()
        provenance[-1].update(pool_states=len(pool), missing_prospective_states=missing,
                              pool_generation_work=pool_work)
        for radius in RADII:
            probe = make_piper_env(config, assets, task, dict(obstacle_radius_m=radius))
            try:
                for index, item in enumerate(pool):
                    obs = np.asarray(item['state']['observation']).copy(); obs[21] = radius
                    tensor = torch.as_tensor(obs[None])
                    # One deterministic action and three common-seed samples,
                    # fixed before reading any simulator counterfactual label.
                    torch.manual_seed(692000+index)
                    with torch.no_grad():
                        actions = [learner.actor(tensor, deterministic=True, geometry=learner.geometry)[0][0].numpy()]
                        actions.extend(learner.actor(tensor.expand(3, -1), geometry=learner.geometry)[0].numpy())
                    for draw, action in enumerate(actions):
                        variable = torch.as_tensor(action[None]).requires_grad_(True)
                        heads = learner.margin_geometry(tensor, variable)
                        gradient = torch.autograd.grad(heads.mean(), variable)[0][0].numpy()
                        norm = float(np.linalg.norm(gradient))
                        normal = gradient/norm if norm > 1e-7 else np.zeros(3)
                        plus, minus = np.clip(action+.2*normal, -1, 1), np.clip(action-.2*normal, -1, 1)
                        base, count = counterfactual(probe, item['state'], action); work += count
                        outward, count = counterfactual(probe, item['state'], plus); work += count
                        inward, count = counterfactual(probe, item['state'], minus); work += count
                        with torch.no_grad():
                            corr = float(corrected(tensor, variable.detach()).mean())
                        rows.append(dict(task=task, layout_seed=item['layout'], state_time=item['time'],
                            radius_m=radius, draw=draw, original_margin_m=float(heads.detach().mean()),
                            radius_response_margin_m=corr, true_margin_m=base['local_margin'],
                            original_gradient_norm=norm, gradient_valid=norm > 1e-7,
                            true_plus_minus_margin_m=outward['local_margin']-inward['local_margin'],
                            clipped_direction_delta_norm=float(np.linalg.norm(plus-minus)),
                            action_x=float(action[0]), action_y=float(action[1]), action_z=float(action[2])))
            finally:
                probe.close()
        # Freeze Actor and compare only this proposed margin prior under the
        # held-fixed new-radius development layouts and common episode RNG.
        for model in ('original', 'radius_response'):
            geometry = learner.geometry if model == 'original' else EndpointGeometry(
                corrected, 'endpoint_pullback', learner.transitions)
            rollout = make_piper_env(config, assets, task, dict(obstacle_radius_m=.055))
            try:
                for layout in LAYOUTS:
                    torch.manual_seed(693000+layout-LAYOUTS[0])
                    obs, initial = rollout.reset(seed=layout)
                    minimum, total = initial['minimum_distance'], 0.
                    for time in range(rollout.config.horizon):
                        with torch.no_grad():
                            action = learner.actor(torch.as_tensor(obs[None]), geometry=geometry)[0][0].numpy()
                        obs, reward, done, truncated, info = rollout.step(action)
                        minimum = min(minimum, info['minimum_distance']); total += reward
                        if done or truncated:
                            break
                    episodes.append(dict(task=task, model=model, layout_seed=layout, radius_m=.055,
                        safe_success=info['safe_success'], violation=info['violation'], steps=time+1,
                        episode_return=total, minimum_distance_m=minimum))
            finally:
                provenance[-1][model+'_rollout_work'] = dict(controls=rollout.total_control_intervals,
                    substeps=rollout.total_physics_substeps, distance_queries=rollout.total_distance_queries)
                rollout.close()
        del learner, saved
    torch.save(pools, args.output/'physical_pool.pt')
    write_csv(args.output/'margin_samples.csv', rows)
    write_csv(args.output/'radius_response_episodes.csv', episodes)
    atomic_json(args.output/'summary.json', dict(protocol='fixed_piper_margin_diagnosis_v1',
        split='development', training_transitions=0, physical_robot_interactions=0,
        layout_seeds=LAYOUTS, prospective_state_times=STATE_TIMES, no_state_pool_reselection=True,
        labels='true control-interval minimum distance; positive margin is unsafe',
        finite_difference_action_epsilon=.2, provenance=provenance, margin_summary=summarize(rows),
        rollout_summary=[dict(task=task, model=model, episodes=10,
            safe_success_count=sum(r['safe_success'] for r in episodes if r['task']==task and r['model']==model),
            violation_count=sum(r['violation'] for r in episodes if r['task']==task and r['model']==model))
            for task in ('Reach', 'Payload') for model in ('original', 'radius_response')],
        counterfactual_work=dict(zip(('controls', 'substeps', 'distance_queries'), map(int, work))),
        radius_response_not_a_dynamics_or_safety_guarantee=True,
        no_adaptation_or_deployment_decision_automatically_made=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

"""Bounded exact-teacher probe of frozen Learned flow midpoints; no training."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from scripts.star.arm_safe_entropy.environment import TaskConfig
from scripts.star.arm_safe_entropy.geometry import LocalMarginEnsemble
from scripts.star.arm_safe_entropy.models import ArmFlowActor
from scripts.star.arm_safe_entropy.oracle import SimulatorOracle


def mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs=2, type=Path, required=True)
    parser.add_argument('--state-pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    configs = [json.loads((run / 'run_config.json').read_text()) for run in args.runs]
    if {c['training_seed'] for c in configs} != {2027, 2028} or any(
            c['method'] != 'flow_learned' or c['task_config'] != asdict(TaskConfig()) for c in configs):
        parser.error('Use both matched Learned development seeds, never a selected or formal seed')
    if args.output.exists():
        parser.error('Never overwrite another diagnostic')
    pool = json.loads(args.state_pool.read_text())
    eligible = [i for i, row in enumerate(pool)
                if row['region'] == 'near_boundary' and row['common_gradient_norm'] > 1e-7]
    chosen = np.sort(np.random.default_rng(1337).choice(eligible, 16, replace=False))
    observations = np.repeat(np.asarray([pool[i]['observation'] for i in chosen], np.float32), 4, axis=0)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    teacher = SimulatorOracle(workers=args.workers)
    records = []
    try:
        for run, config in zip(args.runs, configs):
            receipt = json.loads((run / 'receipt.json').read_text())
            state = torch.load(run / 'policy_300000.pt', map_location=args.device, weights_only=False)
            if receipt['status'] != 'completed' or receipt['steps'] != 300000 or state['updates'] != 298000:
                raise ValueError('Use the fixed completed300k policy, not an earlier selected milestone')
            actor = ArmFlowActor(25, variance=config['reference_variance']).to(args.device)
            margin = LocalMarginEnsemble().to(args.device)
            actor.load_state_dict(state['actor'])
            margin.load_state_dict(state['margin'])
            obs = torch.as_tensor(observations, device=args.device)
            # Same Gaussian sampling seed; teacher is called only AFTER the
            # unchanged Learned policy generated all diagnostic paths.
            torch.manual_seed(1337)
            with torch.no_grad():
                _, _, _, trace = actor(obs, geometry=margin.geometry, return_reference=True)
                mids = trace['midpoints'].reshape(-1, 3)
                repeated_obs = obs.repeat_interleave(actor.num_steps, dim=0)
                predicted_h = margin(repeated_obs, mids).mean(0).cpu().numpy()
            normal, weight, factors = margin.geometry(repeated_obs, mids, return_diagnostics=True)
            predicted_n = normal.cpu().numpy()
            q = weight.cpu().numpy()
            gradient_norm = factors['gradient_norm'].cpu().numpy()
            actions = mids.cpu().numpy()
            truth = teacher.evaluate_numpy(repeated_obs.cpu().numpy(), actions)
            true_valid = truth[:, 4] > 1e-7
            pred_valid = np.linalg.norm(predicted_n, axis=1) > 1e-7
            both = true_valid & pred_valid
            cosine = np.sum(predicted_n * truth[:, 1:4], axis=1)
            vn = trace['vn'].cpu().numpy().reshape(-1)
            vt = trace['vt'].cpu().numpy().reshape(-1)
            true_axis_vn = vt + (vn - vt) * cosine**2
            summary = dict(training_seed=config['training_seed'], source_commit=config['source_commit'],
                points=len(actions), true_gradient_valid=int(true_valid.sum()),
                true_gradient_flat=int((~true_valid).sum()), valid_comparisons=int(both.sum()),
                signed_cosine=mean_or_none(cosine[both]), axis_cosine_squared=mean_or_none(cosine[both]**2),
                margin_rmse_m=float(np.sqrt(np.mean((predicted_h - truth[:, 0])**2))),
                mean_reference_normal_variance_on_true_valid_axis=mean_or_none(true_axis_vn[true_valid]))
            for name, mask in [('true_valid', true_valid), ('true_flat', ~true_valid)]:
                summary[name] = dict(points=int(mask.sum()), mean_q=mean_or_none(q[mask]),
                    predicted_gradient_norm_quantiles=np.quantile(gradient_norm[mask], [0, .25, .5, .75, 1]).tolist()
                        if mask.any() else None)
            points = [dict(source_pool_index=int(chosen[i // (4 * actor.num_steps)]),
                draw=(i // actor.num_steps) % 4, quadrature_point=i % actor.num_steps,
                midpoint_action=actions[i].tolist(), predicted_margin_m=float(predicted_h[i]),
                exact_margin_m=float(truth[i, 0]), exact_gradient_norm=float(truth[i, 4]),
                predicted_gradient_norm=float(gradient_norm[i]), q=float(q[i]),
                predicted_normal=predicted_n[i].tolist(), exact_normal=truth[i, 1:4].tolist(),
                normal_cosine=float(cosine[i]) if both[i] else None,
                true_axis_reference_normal_variance=float(true_axis_vn[i]) if true_valid[i] else None)
                for i in range(len(actions))]
            records.append(dict(run=str(run), summary=summary, points=points))
            print(json.dumps(summary), flush=True)
        payload = dict(kind='bounded_internal_flow_geometry_diagnosis_not_policy_evaluation',
            analysis_source_commit=args.source_commit, policy_steps=300000, state_pool=str(args.state_pool),
            selected_pool_indices=chosen.tolist(), states=16, paths_per_state=4, quadrature_points=4,
            training_transitions=0, teacher_use='independent exact model queries after frozen Learned sampling; never training labels',
            limitations='small targeted near-boundary probe; flat true gradients are not assigned a normal; not an episode-risk or final covariance estimate',
            records=records, oracle_statistics=teacher.statistics())
        (args.output / 'diagnostic.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    finally:
        teacher.close()


if __name__ == '__main__':
    main()

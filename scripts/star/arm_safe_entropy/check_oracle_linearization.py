"""Bounded comparison of cached controller Jacobians against exact simulator FD."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.star.arm_safe_entropy.environment import SafeReachEnv, TaskConfig
from scripts.star.arm_safe_entropy.oracle import ControllerJacobianOracle, SimulatorOracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--state-pool', type=Path,
                        help='Use recorded common visited states/actions rather than the initial small probe')
    args = parser.parse_args()
    assert not args.output.exists()
    env = SafeReachEnv(TaskConfig(randomize_layout=False))
    model = ControllerJacobianOracle(workers=args.workers)
    exact = SimulatorOracle(workers=args.workers)
    rng = np.random.default_rng(2028)
    states = []
    try:
        if args.state_pool:
            pool = json.loads(args.state_pool.read_text())
            states = [np.asarray(row['observation'], dtype=np.float32) for row in pool]
            observations = np.stack(states)
            actions = np.asarray([row['reference_action'] for row in pool], dtype=np.float32)
            regions = np.asarray([row['region'] for row in pool])
        else:
            for index in range(12):
                env.reset(seed=2027 + index)
                # Real visited states along safe and near-obstacle controls.
                action = [np.array([0., .2, 0.]), np.array([.2, -.2, .8]),
                          np.array([-.3, 0., .5])][index % 3]
                for step in range(index % 4):
                    _, _, done, _, _ = env.step(action)
                    if done:
                        env.reset(seed=2027 + index)
                        break
                states.append(env._obs())
            observations = np.repeat(np.stack(states), 4, axis=0)
            actions = rng.uniform(-.8, .8, (len(observations), 3)).astype(np.float32)
            regions = np.repeat('initial_scripted_states', len(actions))
        approx = model.evaluate_numpy(observations, actions)
        truth = exact.evaluate_numpy(observations, actions)
        error = np.abs(approx[:, 0] - truth[:, 0])
        actual_valid, predicted_valid = truth[:, 4] > 1e-7, approx[:, 4] > 1e-7
        both = actual_valid & predicted_valid
        cosine = (approx[both, 1:4] * truth[both, 1:4]).sum(-1)
        start = time.monotonic()
        model.evaluate_numpy(observations, actions)
        cached_seconds = time.monotonic() - start
        start = time.monotonic()
        model.evaluate_numpy(observations, rng.uniform(-.8, .8, actions.shape).astype(np.float32))
        new_action_seconds = time.monotonic() - start
        payload = dict(kind='controller_jacobian_approximation_check_not_task_efficacy',
            task_version=TaskConfig().version, states=len(states), state_actions=len(actions),
            source_state_pool=str(args.state_pool) if args.state_pool else None,
            approximate_model_explicit=True, exact_backend='7 actual simulator control intervals per FD query',
            margin_absolute_error_m=dict(mean=float(error.mean()), maximum=float(error.max()),
                                         percentile95=float(np.quantile(error, .95))),
            normals=dict(exact_valid=int(actual_valid.sum()), model_valid=int(predicted_valid.sum()),
                         both_valid=int(both.sum()), missed_exact_valid=int((actual_valid & ~predicted_valid).sum()),
                         extra_model_valid=int((~actual_valid & predicted_valid).sum()),
                         mean_cosine=float(cosine.mean()) if len(cosine) else None,
                         minimum_cosine=float(cosine.min()) if len(cosine) else None,
                         percentile10_cosine=float(np.quantile(cosine,.1)) if len(cosine) else None),
            violation_sign_disagreements=int(((approx[:,0]>0)!=(truth[:,0]>0)).sum()),
            cached_field_queries_per_second=len(actions) / cached_seconds,
            reused_states_new_action_queries_per_second=len(actions) / new_action_seconds,
            approximation_statistics=model.statistics(), exact_statistics=exact.statistics(),
            rows=[dict(region=str(region), observation=s.tolist(), action=a.tolist(), approximate=p.tolist(), exact=t.tolist())
                  for region,s,a,p,t in zip(regions,observations,actions,approx,truth)],
            new_training_transitions=0, trained_policy_claim=False,
            limitations='Local numerical controller model, not an exact guarantee or trained-method advantage; invalid gradients and all mismatch rows retained.')
        payload['by_region'] = {}
        for region in np.unique(regions):
            mask = regions == region
            valid = mask & both
            region_cosine = (approx[valid, 1:4] * truth[valid, 1:4]).sum(-1)
            payload['by_region'][str(region)] = dict(states=int(mask.sum()),
                exact_valid=int((mask & actual_valid).sum()), both_valid=int(valid.sum()),
                missed_exact_valid=int((mask & actual_valid & ~predicted_valid).sum()),
                extra_model_valid=int((mask & ~actual_valid & predicted_valid).sum()),
                mean_cosine=float(region_cosine.mean()) if valid.any() else None,
                minimum_cosine=float(region_cosine.min()) if valid.any() else None,
                percentile10_cosine=float(np.quantile(region_cosine,.1)) if valid.any() else None,
                mean_margin_error_m=float(error[mask].mean()),
                maximum_margin_error_m=float(error[mask].max()),
                violation_sign_disagreements=int((mask & ((approx[:,0]>0)!=(truth[:,0]>0))).sum()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + '\n')
        print(json.dumps({k:v for k,v in payload.items() if k!='rows'}),flush=True)
    finally:
        env.close()
        model.close()
        exact.close()


if __name__ == '__main__':
    main()

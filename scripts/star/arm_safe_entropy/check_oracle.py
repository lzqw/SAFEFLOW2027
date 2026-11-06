"""One bounded controller-coordinate/replica check and an oracle throughput sample."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.star.arm_safe_entropy.environment import SafeReachEnv, TaskConfig
from scripts.star.arm_safe_entropy.oracle import SimulatorOracle, finite_difference_geometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    assert not args.output.exists()
    env = SafeReachEnv(TaskConfig(randomize_layout=False))
    oracle = SimulatorOracle(workers=args.workers)
    rng = np.random.default_rng(2027)
    observations, actions, live = [], [], []
    try:
        for index in range(12):
            obs, _ = env.reset(seed=2027 + index)
            for step in range(index % 4):
                _, _, done, _, _ = env.step(np.array([.2, -.2, .8]))
                if done:
                    break
            obs = env._obs()
            action = rng.uniform(-.8, .8, 3).astype(np.float32)
            before = env._obs().copy()
            exact = finite_difference_geometry(env, action)
            assert np.allclose(before, env._obs(), atol=1e-7), 'Oracle changed live simulator state'
            observations.append(obs); actions.append(action); live.append(exact)
        observations, actions, live = map(np.stack, [observations, actions, live])
        replica = oracle.evaluate_numpy(observations, actions)
        error = np.abs(replica[:, 0] - live[:, 0])
        valid = (replica[:, 4] > 1e-7) & (live[:, 4] > 1e-7)
        cosine = (replica[valid, 1:4] * live[valid, 1:4]).sum(-1)
        start = time.monotonic()
        # Different counterfactual actions avoid an artificially perfect cache benchmark.
        benchmark = oracle.evaluate_numpy(np.repeat(observations, 4, 0),
                                          rng.uniform(-.8, .8, (48, 3)).astype(np.float32))
        elapsed = time.monotonic() - start
        payload = dict(kind='bounded_oracle_controller_check_not_method_result',
            states=len(observations), maximum_margin_replica_error_m=float(error.max()),
            valid_normals_compared=int(valid.sum()), minimum_normal_cosine=float(cosine.min()) if len(cosine) else None,
            live_state_unchanged=True, benchmark_unique_state_actions=48,
            benchmark_seconds=elapsed, benchmark_state_actions_per_second=48 / elapsed,
            oracle_statistics=oracle.statistics(), rows=[dict(live=x.tolist(), replica=y.tolist()) for x, y in zip(live, replica)],
            new_training_transitions=0, new_behavior_policy_runs=0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + '\n')
        print(json.dumps({k: v for k, v in payload.items() if k != 'rows'}), flush=True)
        assert error.max() < .0002, 'Simulator reconstruction is not an adequate oracle'
        assert len(cosine) and cosine.min() > .98, 'Oracle normal mismatch; do not launch direction training'
    finally:
        env.close()
        oracle.close()


if __name__ == '__main__':
    main()

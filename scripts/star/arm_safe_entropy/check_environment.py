"""Only required task/controller checks and three explanatory simulation clips."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from scripts.star.arm_safe_entropy.environment import SafeReachEnv, TaskConfig


def demonstration(path, obstacle, detour):
    env = SafeReachEnv(TaskConfig(obstacle=obstacle, randomize_layout=False))
    frames, costs, records = [], [], []
    try:
        env.reset(seed=2027)
        waypoints = ([np.array([-.22, -.18, .45]), np.array([-.22, .17, .45]), env.goal.copy()]
                     if detour else [env.goal.copy()])
        target = 0
        frames.append(env.render())
        for step in range(env.config.horizon):
            position = env.robot.get_ee_position()
            if np.linalg.norm(position - waypoints[target]) < .04 and target < len(waypoints) - 1:
                target += 1
            action = np.clip((waypoints[target] - position) / env.config.displacement_scale, -1., 1.)
            _, reward, done, truncated, info = env.step(action)
            costs.append(info['cost'])
            records.append(dict(step=step + 1, action=action.tolist(), reward=reward, **info))
            frames.append(env.render())
            if done or truncated:
                break
        imageio.mimsave(path, frames, fps=25, macro_block_size=1)
        return dict(obstacle=obstacle, detour=detour, terminal=records[-1],
                    episode_cost=sum(costs), records=records)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Do not overwrite earlier environment evidence'
    args.output.mkdir(parents=True)
    cases = {name: demonstration(args.output / (name + '.mp4'), obstacle, detour)
             for name, obstacle, detour in [('unobstructed_reach', False, False),
                                          ('straight_path_violation', True, False),
                                          ('collision_free_detour', True, True)]}
    checks = dict(unobstructed_reachable=cases['unobstructed_reach']['terminal']['safe_success'],
                  straight_path_unsafe=cases['straight_path_violation']['episode_cost'] == 1.,
                  collision_free_detour=cases['collision_free_detour']['terminal']['safe_success'])
    env = SafeReachEnv(TaskConfig(obstacle=False, randomize_layout=False))
    try:
        obs, _ = env.reset(seed=2027)
        for _ in range(100):
            obs, _, done, truncated, info = env.step(np.zeros(3))
            if done:
                break
        checks['finite_horizon_terminal_no_bootstrap'] = done and not truncated and info['true_horizon_terminal'] and obs[-1] == 0.
        checks['binary_cost'] = all(c['episode_cost'] in [0., 1.] for c in cases.values())
    finally:
        env.close()
    checks = {key: bool(value) for key, value in checks.items()}
    (args.output / 'environment_checks.json').write_text(json.dumps(dict(checks=checks, cases=cases), indent=2) + '\n')
    print(json.dumps(checks), flush=True)
    if not all(checks.values()):
        raise SystemExit('Task demonstration incomplete; do not launch training until repaired')


if __name__ == '__main__':
    main()

"""One task-only extension: independently sampled relative endpoint/obstacle positions."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.star.arm_safe_entropy.environment import SafeReachEnv, TaskConfig


LAYOUT = dict(
    start_center=[-.25, -.17, .25], start_jitter_halfwidth=[.04, .02, .03],
    goal_center=[-.25, .17, .25], goal_jitter_halfwidth=[.05, .02, .03],
    obstacle_center=[-.25, 0., .25], obstacle_jitter_halfwidth=[.045, .015, .025],
    sampling='independent uniform coordinates and independent start/goal/obstacle draws; no shared translation',
    rejection='same initial protected-link clearance plus a kinematic safe/reachable goal; no policy-return selection')


class RelativeLayoutReach(SafeReachEnv):
    def _sample_layout(self, fixed):
        positions = []
        for name in ['start', 'goal', 'obstacle']:
            halfwidth = np.asarray(LAYOUT[name+'_jitter_halfwidth'])
            positions.append(np.asarray(LAYOUT[name+'_center']) +
                             (np.zeros(3) if fixed else self.np_random.uniform(-halfwidth, halfwidth)))
        return positions

    def _additional_layout_validity(self):
        # Reset-layout admissibility only, applied identically to all methods.
        # Never queried by the Learned actor or used to choose behavior actions.
        snapshot = self.client.saveState()
        try:
            joints = self.robot.inverse_kinematics(link=self.robot.ee_link, position=self.goal,
                                                   orientation=np.array([1., 0., 0., 0.]))[:7]
            self.robot.set_joint_angles(np.concatenate([joints, [0., 0.]]))
            minimum, collision = self._distance()
            return (not collision and minimum > self.config.safety_distance + .005 and
                    np.linalg.norm(self.ee_position()-self.goal) < .02)
        finally:
            self.client.restoreState(snapshot)
            self.client.removeState(snapshot)

    def definition(self):
        return super().definition() | dict(relative_layout_distribution=LAYOUT,
                                          algorithm_change=False, reward_or_cost_change=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed-start', type=int, default=82027)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve earlier layout evidence')
    env = RelativeLayoutReach(TaskConfig(version='SafeReach-RelativeLayout-v3'))
    records = []
    try:
        for seed in range(args.seed_start, args.seed_start+30):
            _, info = env.reset(seed=seed)
            minimum = info['minimum_distance']
            waypoints = [env.start+np.array([0., -.07, 0.]), env.start+np.array([.03, -.07, .20]),
                         env.goal+np.array([.03, 0., .20]), env.goal]
            target = 0
            for step in range(env.config.horizon):
                position = env.ee_position()
                if np.linalg.norm(position-waypoints[target]) < .025 and target < 3:
                    target += 1
                action = np.clip((waypoints[target]-position)/env.config.displacement_scale, -1., 1.)
                _, _, done, truncated, result = env.step(action)
                minimum = min(minimum, result['minimum_distance'])
                if done or truncated:
                    break
            records.append(dict(layout_seed=seed, start=env.start.tolist(), goal=env.goal.tolist(),
                obstacle=env.obstacle_position.tolist(), safe_success=result['safe_success'],
                violation=result['violation'], length=step+1, minimum_distance_m=minimum))
            segment = env.goal-env.start
            fraction = np.clip(np.dot(env.obstacle_position-env.start, segment)/np.dot(segment, segment), 0., 1.)
            line_distance = float(np.linalg.norm(env.start+fraction*segment-env.obstacle_position))
            records[-1]['straight_segment_intersects_inflated_sphere'] = bool(
                line_distance < env.config.obstacle_radius+env.config.safety_distance)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(task_definition=env.definition(),
            safe_detours=sum(r['safe_success'] for r in records), layouts=30,
            validation_transitions=sum(r['length'] for r in records), training_transitions=0,
            policy_evaluations=0, records=records), indent=2)+'\n')
        if not all(r['safe_success'] for r in records):
            raise RuntimeError('The fixed constructive path did not establish every screening layout')
        print(json.dumps(dict(safe_detours=30, layouts=30, training_transitions=0)))
    finally:
        env.close()


if __name__ == '__main__':
    main()

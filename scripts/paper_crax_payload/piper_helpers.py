"""Training-only Piper adapters; official model/simulator and hardware code unchanged."""
from copy import deepcopy

import numpy as np
import pybullet as bullet

from deployment.piper.simulation import PiperSimulation


class PiperTrainingEnvironment(PiperSimulation):
    """Shared admissible resets and actual payload-centroid task completion.

    Reset rejection uses only static endpoint reachability/clearance, not policy
    performance. No proposal/action is selected using the diagnostic oracle.
    The native34D interface and all official model assets remain unchanged.
    """

    def __init__(self, *args, **kwargs):
        self.total_distance_queries = self.goal_admissibility_queries = self.rejected_resets = 0
        super().__init__(*args, **kwargs)

    def distances(self):
        self.total_distance_queries += 1
        return super().distances()

    def _joint_target(self, tcp, orientation):
        # The provisional200-iteration solver stopped124.6mm short on the
        # cross-workspace goal. Same actual kinematics, more solver iterations;
        # no action reranking, oracle safety projection or robot teleport here.
        rotation = np.asarray(bullet.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        flange = np.asarray(tcp)-rotation@np.asarray(self.config.tcp_from_flange_m)
        solution = self.b('calculateInverseKinematics', self.robot, self.flange,
            flange.tolist(), list(orientation), maxNumIterations=1200, residualThreshold=1e-6)
        return np.asarray(solution[:6])

    def task_position(self):
        if self.payload is not None:
            return np.asarray(self.b('getBasePositionAndOrientation', self.payload)[0])
        return self.tcp_pose()[0]

    def _goal_is_admissible(self):
        snapshot = self.b('saveState')
        self.goal_admissibility_queries += 1
        try:
            target = self._joint_target(self.goal, self.orientation)
            for joint, angle in zip(self.arm_joints, target):
                self.b('resetJointState', self.robot, joint, float(angle))
            if self.payload is not None:
                position, orientation = self.tcp_pose()
                self.b('resetBasePositionAndOrientation', self.payload, position, orientation)
            arm, payload = self.distances()
            return bool(min(arm, payload) > self.config.safety_distance_m + .002 and
                        np.linalg.norm(self.task_position()-self.goal) < .025)
        finally:
            self.b('restoreState', snapshot)
            self.b('removeState', snapshot)

    def reset(self, *, seed=None, options=None):
        fixed = bool((options or {}).get('fixed_layout', False))
        for attempt in range(30):
            try:
                observation, info = super().reset(seed=seed if attempt == 0 else None, options=options)
                admissible = self._goal_is_admissible()
            except RuntimeError:
                admissible = False
            if admissible:
                self.previous_goal_distance = float(np.linalg.norm(self.task_position()-self.goal))
                return observation, info | dict(reset_rejections=attempt,
                    goal_admissibility='static arm AND payload endpoint clearance and IK only')
            self.rejected_resets += 1
            if fixed:
                break
        raise RuntimeError('No safe reachable Piper start/goal after bounded static rejection')

    def step(self, action):
        previous = self.previous_goal_distance
        observation, _, _, truncated, info = super().step(action)
        distance = float(np.linalg.norm(self.task_position()-self.goal))
        success = bool(distance < self.config.success_distance_m and not info['violation'])
        reward = 5.*(previous-distance)-.01+10.*success-10.*info['violation']
        self.previous_goal_distance = distance
        self._done = bool(success or info['violation'] or self.elapsed >= self.config.horizon)
        return observation, float(reward), self._done, truncated, info | dict(
            safe_success=success, distance_to_goal=distance,
            arm_collision=info['arm_minimum_distance'] <= 0.,
            payload_collision=info['payload_minimum_distance'] <= 0.)

    def definition(self):
        return super().definition() | dict(
            task_completion='actual payload centroid for Payload; TCP for Reach',
            reset_admissibility='bounded static safe/reachable start AND goal; common to all policies',
            inverse_kinematics='same native solver1200maxiterations vs provisional200; common to all policies',
            learned_action_oracle_selection=False)


def capture_piper_environment(env):
    joints = env.arm_joints + env.finger_joints
    states = [env.b('getJointState', env.robot, joint) for joint in joints]
    state = dict(joints=list(joints), positions=[s[0] for s in states], velocities=[s[1] for s in states],
        goal=env.goal.copy(), obstacle_position=env.obstacle_position.copy(),
        orientation=env.orientation.copy(), elapsed=env.elapsed, done=env._done,
        previous_goal_distance=env.previous_goal_distance,
        rng=deepcopy(env.np_random.bit_generator.state), observation=env._obs().copy(),
        total_control_intervals=env.total_control_intervals,
        total_physics_substeps=env.total_physics_substeps,
        total_distance_queries=env.total_distance_queries,
        goal_admissibility_queries=env.goal_admissibility_queries,
        rejected_resets=env.rejected_resets)
    if env.payload is not None:
        position, orientation = env.b('getBasePositionAndOrientation', env.payload)
        linear, angular = env.b('getBaseVelocity', env.payload)
        state['payload'] = dict(position=position, orientation=orientation, linear=linear, angular=angular)
    else:
        state['payload'] = None
    return state


def restore_piper_environment(env, state):
    if (state['payload'] is None) != (env.payload is None):
        raise ValueError('Cannot restore a Reach state into Payload or vice versa')
    if state['joints'] != env.arm_joints + env.finger_joints:
        raise ValueError('Native official joint ordering differs')
    env.goal = np.asarray(state['goal']).copy()
    env.obstacle_position = np.asarray(state['obstacle_position']).copy()
    env.orientation = np.asarray(state['orientation']).copy()
    env.b('resetBasePositionAndOrientation', env.obstacle, env.obstacle_position, [0., 0., 0., 1.])
    for joint, position, velocity in zip(state['joints'], state['positions'], state['velocities']):
        env.b('resetJointState', env.robot, joint, float(position), targetVelocity=float(velocity))
    env._motor(state['positions'][:6])
    env.b('setJointMotorControlArray', env.robot, env.finger_joints, 2,
          targetPositions=state['positions'][6:], forces=[10.]*2)
    if env.payload is not None:
        payload = state['payload']
        env.b('resetBasePositionAndOrientation', env.payload, payload['position'], payload['orientation'])
        env.b('resetBaseVelocity', env.payload, payload['linear'], payload['angular'])
    env.b('performCollisionDetection')
    env.elapsed, env._done = state['elapsed'], state['done']
    env.previous_goal_distance = state['previous_goal_distance']
    env.np_random.bit_generator.state = deepcopy(state['rng'])
    for key in ('total_control_intervals', 'total_physics_substeps', 'total_distance_queries',
                'goal_admissibility_queries', 'rejected_resets'):
        setattr(env, key, state[key])
    observation = env._obs()
    if not np.allclose(observation, state['observation'], rtol=0., atol=2e-6):
        raise ValueError('Piper observed reconstruction mismatch; do not reset its partial episode')
    return observation

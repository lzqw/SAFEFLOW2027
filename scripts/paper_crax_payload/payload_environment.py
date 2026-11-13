"""SafePayloadTransport-v1: an actually held body, not Reach relabelled.

The gripper is not learned. A fixed Bullet grasp constraint holds a 50g box
with fixed commanded orientation. The interval oracle includes the protected
arm and the carried box at every physical substep. No physical hardware API.
"""
from dataclasses import asdict, dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from scripts.star.arm_safe_entropy.environment import SafeReachEnv, TaskConfig
from scripts.star.arm_safe_entropy.relative_layout import LAYOUT
from scripts.star.arm_safe_entropy.stability_runtime import capture_environment, restore_environment


@dataclass
class PayloadConfig(TaskConfig):
    version: str = 'SafePayloadTransport-v1'
    payload_half_extents: tuple = (.025, .018, .020)
    payload_mass_kg: float = .050
    payload_grasp_offset: tuple = (0., 0., .035)
    grasp_max_force_n: float = 80.
    obstacle_offset: tuple = (0., 0., 0.)


class SafePayloadTransport(SafeReachEnv):
    """Three-dimensional EE deltas, fixed attitude, already grasped payload."""

    allowed_grasp_links = (8, 9, 10, 11)

    def __init__(self, config=None, render_mode=None):
        config = config if isinstance(config, PayloadConfig) else PayloadConfig(**(config or {}))
        if config.version != 'SafePayloadTransport-v1':
            raise ValueError('Payload is an independent named task')
        self._collect_interval = False
        self._interval_component_min = np.full(2, np.inf)
        self.physical_control_intervals = 0
        SafeReachEnv.__init__(self, config, render_mode)
        self.sim.create_box('payload', half_extents=np.asarray(config.payload_half_extents),
            mass=config.payload_mass_kg, position=np.array([2., 2., 2.]),
            rgba_color=np.array([.1, .35, .9, 1.]), lateral_friction=.8)
        self.payload_body = self.sim._bodies_idx['payload']
        for link in self.allowed_grasp_links:
            self.client.setCollisionFilterPair(self.body, self.payload_body, link, -1, 0)
        self.grasp_constraint = self.client.createConstraint(
            self.body, self.robot.ee_link, self.payload_body, -1,
            self.client.JOINT_FIXED, [0., 0., 0.], list(config.payload_grasp_offset), [0., 0., 0.])
        self.client.changeConstraint(self.grasp_constraint, maxForce=config.grasp_max_force_n)
        # Native25 + payload position3/quaternion4/dimensions3/mass1/velocity6/goal3.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(45,), dtype=np.float32)

    def _sample_layout(self, fixed):
        positions = []
        for name in ['start', 'goal', 'obstacle']:
            half = np.asarray(LAYOUT[name + '_jitter_halfwidth'])
            positions.append(np.asarray(LAYOUT[name + '_center']) +
                             (np.zeros(3) if fixed else self.np_random.uniform(-half, half)))
        positions[2] += np.asarray(self.config.obstacle_offset)
        return positions

    def _grasp_pose(self):
        link = self.client.getLinkState(self.body, self.robot.ee_link, computeForwardKinematics=1)
        return self.client.multiplyTransforms(link[4], link[5], self.config.payload_grasp_offset,
                                             [0., 0., 0., 1.])

    def _place_payload(self):
        position, orientation = self._grasp_pose()
        self.client.resetBasePositionAndOrientation(self.payload_body, position, orientation)
        self.client.resetBaseVelocity(self.payload_body, [0., 0., 0.], [0., 0., 0.])
        self.client.performCollisionDetection()

    def payload_position(self):
        return np.asarray(self.client.getBasePositionAndOrientation(self.payload_body)[0])

    @property
    def payload_goal(self):
        # The fixed commanded attitude is Rx(pi); local +z points down in world.
        return self.goal + np.asarray(self.config.payload_grasp_offset) * [1., -1., -1.]

    def _obs(self):
        native = SafeReachEnv._obs(self)
        position, quaternion = self.client.getBasePositionAndOrientation(self.payload_body)
        linear, angular = self.client.getBaseVelocity(self.payload_body)
        return np.concatenate([native, position, quaternion, self.config.payload_half_extents,
                               [self.config.payload_mass_kg], linear, angular,
                               self.payload_goal]).astype(np.float32)

    def _component_distance(self):
        if not self.config.obstacle:
            return 2., 2., np.full(len(self.protected_links), 2.)
        arm, _, links = SafeReachEnv._distance(self, components=True)
        points = self.client.getClosestPoints(self.payload_body, self.obstacle_body, distance=2.)
        payload = min((float(p[8]) for p in points), default=2.)
        return arm, payload, links

    def _distance(self, components=False):
        arm, payload, links = self._component_distance()
        if self._collect_interval:
            self._interval_component_min = np.minimum(self._interval_component_min, [arm, payload])
        minimum = min(arm, payload)
        if components:
            return minimum, minimum <= 0., np.concatenate([links, [payload]])
        return minimum, minimum <= 0.

    def _advance(self, action, distance_trace=False):
        self.physical_control_intervals += 1
        self._interval_component_min = np.full(2, np.inf)
        self._collect_interval = True
        try:
            return SafeReachEnv._advance(self, action, distance_trace)
        finally:
            self._collect_interval = False

    def _additional_layout_validity(self):
        snapshot = self.client.saveState()
        try:
            joints = self.robot.inverse_kinematics(link=self.robot.ee_link, position=self.goal,
                orientation=np.array([1., 0., 0., 0.]))[:7]
            self.robot.set_joint_angles(np.concatenate([joints, [0., 0.]]))
            self._place_payload()
            distance, collision = self._distance()
            return bool(not collision and distance > self.config.safety_distance + .005 and
                        np.linalg.norm(self.payload_position() - self.payload_goal) < .02)
        finally:
            self.client.restoreState(snapshot)
            self.client.removeState(snapshot)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self._elapsed, self._done = 0, False
        fixed = bool((options or {}).get('fixed_layout', False)) or not self.config.randomize_layout
        for _ in range(30):
            self.start, self.goal, self.obstacle_position = self._sample_layout(fixed)
            if not self.config.obstacle:
                self.obstacle_position = np.array([2., 2., 2.])
            self.sim.set_base_pose('target', self.payload_goal, np.array([0., 0., 0., 1.]))
            self.sim.set_base_pose('obstacle', self.obstacle_position, np.array([0., 0., 0., 1.]))
            self.robot.reset()
            q = self.robot.inverse_kinematics(link=self.robot.ee_link, position=self.start,
                orientation=np.array([1., 0., 0., 0.]))[:7]
            target = np.concatenate([q, [0., 0.]])
            self.robot.set_joint_angles(target)
            self.robot.control_joints(target)
            # Move the already held object with the reset robot BEFORE settling.
            self._place_payload()
            for _ in range(20):
                self.client.stepSimulation()
            distance, collision = self._distance()
            if not collision and distance > self.config.safety_distance + .005 and self._additional_layout_validity():
                break
        else:
            raise RuntimeError('Cannot obtain safe reachable arm AND payload endpoints')
        self._previous_distance = float(np.linalg.norm(self.payload_position() - self.payload_goal))
        return self._obs(), dict(task_version=self.config.version, minimum_distance=distance,
            payload_grasped=True, payload_mass_kg=self.config.payload_mass_kg)

    def step(self, action):
        if self._done:
            raise RuntimeError('Call reset after episode termination')
        minimum, collision, violation, workspace_clipped = self._advance(action)
        self._elapsed += 1
        distance = float(np.linalg.norm(self.payload_position() - self.payload_goal))
        success = distance < self.config.success_distance and not violation
        horizon = self._elapsed >= self.config.horizon
        reward = self.config.progress_scale * (min(self._previous_distance, 1.) - min(distance, 1.)) - self.config.step_penalty
        reward += self.config.success_bonus * success - self.config.failure_penalty * violation
        if violation:
            reward -= self.config.step_penalty * (self.config.horizon - self._elapsed)
        if horizon and not success and not violation:
            reward -= self.config.timeout_penalty
        self._previous_distance = distance
        self._done = bool(success or violation or horizon)
        arm, payload = self._interval_component_min
        info = dict(cost=float(violation), violation=bool(violation), collision=bool(collision),
            clearance_violation=bool(minimum < self.config.safety_distance),
            arm_collision=bool(arm <= 0.), payload_collision=bool(payload <= 0.),
            arm_minimum_distance=float(arm), payload_minimum_distance=float(payload),
            minimum_distance=float(minimum), local_margin=float(self.config.safety_distance - minimum),
            success=bool(success), safe_success=bool(success and not violation),
            distance_to_goal=distance, true_horizon_terminal=bool(horizon),
            workspace_clipped=workspace_clipped, control_dt=self.control_dt)
        return self._obs(), float(reward), self._done, False, info

    def definition(self):
        return dict(**asdict(self.config), control_dt=self.control_dt,
            observation_dimension=45, action_dimension=3, relative_layout_distribution=LAYOUT,
            protected_robot_links=self.protected_links, allowed_grasp_contact_links=self.allowed_grasp_links,
            held_payload='dynamic box (configured mass) with fixed Bullet grasp constraint; no learned grasp',
            safety_label='minimum signed arm-obstacle OR payload-obstacle distance over start and all physical substeps',
            payload_goal='EE goal plus fixed Rx(pi) grasp offset; reward uses actual payload centroid',
            observations=['native25', 'payload_world_xyz', 'payload_world_quaternion_xyzw',
                          'payload_half_extents_m', 'payload_mass_kg', 'payload_world_linear_velocity',
                          'payload_world_angular_velocity', 'payload_goal_world_xyz'])


def capture_payload_environment(env):
    state = capture_environment(env)
    position, orientation = env.client.getBasePositionAndOrientation(env.payload_body)
    linear, angular = env.client.getBaseVelocity(env.payload_body)
    return state | dict(payload_position=position, payload_orientation=orientation,
                        payload_linear_velocity=linear, payload_angular_velocity=angular)


def restore_payload_environment(env, state):
    # Restore the carried body's pose/velocity BEFORE base reconstruction checks
    # the full45D observation; restore does not step or reset a partial episode.
    env.client.resetBasePositionAndOrientation(env.payload_body, state['payload_position'], state['payload_orientation'])
    env.client.resetBaseVelocity(env.payload_body, state['payload_linear_velocity'], state['payload_angular_velocity'])
    observation = restore_environment(env, state)
    env.sim.set_base_pose('target', env.payload_goal, np.array([0., 0., 0., 1.]))
    return observation


if 'SafePayloadTransport-v1' not in gym.envs.registry:
    gym.register(id='SafePayloadTransport-v1',
                 entry_point='scripts.paper_crax_payload.payload_environment:SafePayloadTransport')

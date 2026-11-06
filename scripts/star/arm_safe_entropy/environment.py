"""SAFEFLOW Panda SafeReach-Obstacle-v1, a project panda-gym variant.

Real PyBullet joint motors, fixed-gripper Cartesian displacement commands,
signed closest distances on protected robot links, and interval-min labels.
This module never commands physical hardware.
"""
from dataclasses import asdict, dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet


@dataclass
class TaskConfig:
    version: str = 'SafeReach-Obstacle-v1'
    obstacle: bool = True
    horizon: int = 100
    substeps: int = 20
    displacement_scale: float = 0.025
    safety_distance: float = 0.015
    success_distance: float = 0.035
    obstacle_radius: float = 0.060
    progress_scale: float = 5.0
    success_bonus: float = 10.0
    failure_penalty: float = 10.0
    timeout_penalty: float = 2.0
    step_penalty: float = 0.01
    randomize_layout: bool = True


class SafeReachEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 25}
    protected_links = tuple(range(2, 12))

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        self.config = config if isinstance(config, TaskConfig) else TaskConfig(**(config or {}))
        self.render_mode = render_mode
        self.sim = PyBullet(render_mode='rgb_array', renderer='Tiny', n_substeps=self.config.substeps)
        self.client = self.sim.physics_client
        self.robot = Panda(self.sim, block_gripper=True, base_position=np.array([-0.6, 0., 0.]), control_type='ee')
        self.body = self.sim._bodies_idx['panda']
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.9, height=0.4, x_offset=-0.3)
        self.sim.create_sphere('target', radius=self.config.success_distance, mass=0., position=np.zeros(3),
                               ghost=True, rgba_color=np.array([0.1, 0.85, 0.2, 0.7]))
        self.sim.create_sphere('obstacle', radius=self.config.obstacle_radius, mass=0., position=np.array([2., 2., 2.]),
                               ghost=False, rgba_color=np.array([0.85, 0.12, 0.12, 1.]))
        self.obstacle_body = self.sim._bodies_idx['obstacle']
        self.action_space = spaces.Box(-1., 1., shape=(3,), dtype=np.float32)
        # q/pi, qdot/4, EE, goal, obstacle, radius, remaining time.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float32)
        self._elapsed = 0
        self._done = False
        self.goal = np.zeros(3)
        self.obstacle_position = np.zeros(3)

    @property
    def control_dt(self):
        return float(self.sim.dt)

    def _distance(self):
        if not self.config.obstacle:
            return 2.0, False
        points = self.client.getClosestPoints(self.body, self.obstacle_body, distance=2.)
        distances = [float(p[8]) for p in points if p[3] in self.protected_links]
        distance = min(distances, default=2.)
        return distance, distance <= 0.

    def _obs(self):
        q = np.array([self.client.getJointState(self.body, j)[0] for j in range(7)])
        dq = np.array([self.client.getJointState(self.body, j)[1] for j in range(7)])
        return np.concatenate([q / np.pi, dq / 4., self.robot.get_ee_position(), self.goal,
                               self.obstacle_position, [self.config.obstacle_radius if self.config.obstacle else 0.],
                               [max(0., 1. - self._elapsed / self.config.horizon)]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._elapsed, self._done = 0, False
        options = options or {}
        fixed = bool(options.get('fixed_layout', False)) or not self.config.randomize_layout
        for _ in range(30):
            jitter = np.zeros(3) if fixed else self.np_random.uniform([-.025, -.015, -.025], [.025, .015, .025])
            self.start = np.array([-.25, -.17, .25]) + jitter
            self.goal = np.array([-.25, .17, .25]) + jitter
            self.obstacle_position = (self.start + self.goal) / 2
            if not self.config.obstacle:
                self.obstacle_position = np.array([2., 2., 2.])
            self.sim.set_base_pose('target', self.goal, np.array([0., 0., 0., 1.]))
            self.sim.set_base_pose('obstacle', self.obstacle_position, np.array([0., 0., 0., 1.]))
            self.robot.reset()
            q = self.robot.inverse_kinematics(link=self.robot.ee_link, position=self.start,
                                              orientation=np.array([1., 0., 0., 0.]))[:7]
            target = np.concatenate([q, [0., 0.]])
            self.robot.set_joint_angles(target)
            self.robot.control_joints(target)
            for _ in range(20):
                self.client.stepSimulation()
            distance, collision = self._distance()
            if not collision and distance > self.config.safety_distance + .005:
                break
        else:
            raise RuntimeError('Could not sample a safe reachable start; do not train on invalid resets')
        self._previous_distance = float(np.linalg.norm(self.robot.get_ee_position() - self.goal))
        return self._obs(), {'task_version': self.config.version, 'minimum_distance': distance}

    def _advance(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1., 1.)
        proposed = self.robot.get_ee_position() + action * self.config.displacement_scale
        target_position = np.clip(proposed, [-.45, -.35, .08], [.10, .35, .55])
        joints = self.robot.inverse_kinematics(link=self.robot.ee_link, position=target_position,
                                               orientation=np.array([1., 0., 0., 0.]))[:7]
        self.robot.control_joints(np.concatenate([joints, [0., 0.]]))
        minimum, collision = self._distance()
        violation = minimum < self.config.safety_distance or collision
        # Every actual 2ms physics frame is checked. Stop at the first violation.
        for _ in range(self.config.substeps):
            if violation:
                break
            self.client.stepSimulation()
            distance, contact = self._distance()
            minimum, collision = min(minimum, distance), collision or contact
            violation = minimum < self.config.safety_distance or collision
        return minimum, collision, violation, bool(np.any(np.abs(proposed - target_position) > 1e-8))

    def step(self, action):
        if self._done:
            raise RuntimeError('Call reset after episode termination')
        minimum, collision, violation, workspace_clipped = self._advance(action)
        self._elapsed += 1
        distance = float(np.linalg.norm(self.robot.get_ee_position() - self.goal))
        success = distance < self.config.success_distance and not violation
        horizon = self._elapsed >= self.config.horizon
        reward = self.config.progress_scale * (min(self._previous_distance, 1.) - min(distance, 1.)) - self.config.step_penalty
        reward += self.config.success_bonus * success - self.config.failure_penalty * violation
        if violation:
            # No early-termination discount of the remaining step charge.
            reward -= self.config.step_penalty * (self.config.horizon - self._elapsed)
        if horizon and not success and not violation:
            reward -= self.config.timeout_penalty
        self._previous_distance = distance
        self._done = bool(success or violation or horizon)
        info = dict(cost=float(violation), violation=bool(violation), collision=bool(collision),
                    clearance_violation=bool(minimum < self.config.safety_distance),
                    minimum_distance=float(minimum), local_margin=float(self.config.safety_distance - minimum),
                    success=bool(success), safe_success=bool(success and not violation),
                    distance_to_goal=distance, true_horizon_terminal=bool(horizon),
                    workspace_clipped=workspace_clipped, control_dt=self.control_dt)
        # Horizon is part of the observed finite-horizon MDP, NOT a bootstrap timeout.
        return self._obs(), float(reward), self._done, False, info

    def oracle_margin(self, action):
        """Exact same-control-interval simulation risk, not a task-space normal.

        No reward, counter, RNG or live physical state changes. Diagnostic only
        unless geometry_source=oracle is explicitly selected by the learner.
        """
        snapshot = self.client.saveState()
        try:
            minimum, _, _, _ = self._advance(action)
            return self.config.safety_distance - minimum
        finally:
            self.client.restoreState(snapshot)
            self.client.removeState(snapshot)

    def oracle_normal(self, action, epsilon=0.002):
        action = np.asarray(action, dtype=float)
        gradient = np.zeros(3)
        for axis in range(3):
            offset = np.eye(3)[axis] * epsilon
            plus, minus = np.clip(action + offset, -1., 1.), np.clip(action - offset, -1., 1.)
            width = plus[axis] - minus[axis]
            gradient[axis] = (self.oracle_margin(plus) - self.oracle_margin(minus)) / width if width > 0 else 0.
        norm = np.linalg.norm(gradient)
        return (gradient / norm if norm > 1e-7 else np.zeros(3)), bool(norm > 1e-7)

    def render(self):
        return self.sim.render(width=640, height=480, target_position=np.array([-.25, 0., .2]),
                               distance=1.0, yaw=50, pitch=-30)

    def close(self):
        self.sim.close()

    def definition(self):
        return dict(**asdict(self.config), control_dt=self.control_dt, protected_links=self.protected_links,
                    observations=['q/pi', 'qdot/4', 'ee_world_xyz', 'goal_world_xyz',
                                  'obstacle_world_xyz', 'obstacle_radius', 'remaining_time'])

"""Simulator-derived action normals for the exact configured arm controller.

No candidate action is selected or replaced. Counterfactual control intervals
only supply detached local reference geometry. The learned-margin policy does
not instantiate this module's oracle pool.
"""
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import time

import numpy as np

_ENV = None


def _initialize_worker():
    global _ENV
    # Bullet prints renderer argv on destruction; keep worker noise out of IPC logs.
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.close(null)
    from scripts.star.arm_safe_entropy.environment import SafeReachEnv, TaskConfig
    _ENV = SafeReachEnv(TaskConfig())


def restore_observation(env, observation):
    """Reconstruct the observed fixed-base, closed-gripper simulation state.

    All arm q/dq, static obstacle/goal pose and remaining time are observed.
    Motor targets are overwritten by _advance, exactly as in the real step.
    The narrow replica check reports any residual simulator-state discrepancy.
    """
    observation = np.asarray(observation, dtype=float)
    if observation.shape != (25,):
        raise ValueError('Oracle requires the shared 25-dimensional arm state')
    radius = float(observation[23])
    env.config.obstacle = radius > 0
    if radius > 0 and abs(radius - env.config.obstacle_radius) > 1e-6:
        env.client.removeBody(env.obstacle_body)
        del env.sim._bodies_idx['obstacle']
        env.sim.create_sphere('obstacle', radius=radius, mass=0., position=observation[20:23],
                              rgba_color=np.array([.85, .12, .12, 1.]))
        env.obstacle_body = env.sim._bodies_idx['obstacle']
        env.config.obstacle_radius = radius
    env.goal = observation[17:20].copy()
    env.obstacle_position = observation[20:23].copy()
    env.sim.set_base_pose('target', env.goal, np.array([0., 0., 0., 1.]))
    env.sim.set_base_pose('obstacle', env.obstacle_position, np.array([0., 0., 0., 1.]))
    q, dq = observation[:7] * np.pi, observation[7:14] * 4.
    for joint, value, velocity in zip(range(7), q, dq):
        env.client.resetJointState(env.body, joint, float(value), targetVelocity=float(velocity))
    for joint in [9, 10]:
        env.client.resetJointState(env.body, joint, 0., targetVelocity=0.)
    env.robot.control_joints(np.concatenate([q, [0., 0.]]))
    env.client.performCollisionDetection()
    env._elapsed = int(round((1. - observation[24]) * env.config.horizon))
    env._done = False
    env._previous_distance = float(np.linalg.norm(env.ee_position() - env.goal))


def finite_difference_geometry(env, action, epsilon=.002):
    action = np.asarray(action, dtype=float)
    margin = env.oracle_margin(action)
    gradient = np.zeros(3)
    for axis in range(3):
        offset = np.eye(3)[axis] * epsilon
        plus, minus = np.clip(action + offset, -1, 1), np.clip(action - offset, -1, 1)
        width = plus[axis] - minus[axis]
        if width > 0:
            gradient[axis] = (env.oracle_margin(plus) - env.oracle_margin(minus)) / width
    norm = float(np.linalg.norm(gradient))
    normal = gradient / norm if norm > 1e-7 else np.zeros(3)
    return np.asarray([margin, *normal, norm], dtype=np.float32)


def _query(item):
    observation, action, epsilon = item
    restore_observation(_ENV, observation)
    return finite_difference_geometry(_ENV, action, epsilon)


def _margin_query(item):
    observation, action = item
    restore_observation(_ENV, observation)
    return _ENV.oracle_margin(action)


def _state_distance_query(observation):
    restore_observation(_ENV, observation)
    return _ENV._distance()[0]


def _interval_jacobian_query(item):
    """Differentiate every link/frame distance at a nearby numerical anchor.

    Retaining individual frame/link planes avoids differentiating only the
    already-minimized zero-action distance, which can have a flat subgradient.
    This is a local first-order controller model, not exact risk at all actions.
    """
    observation, epsilon, anchor = item
    restore_observation(_ENV, observation)
    snapshot = _ENV.client.saveState()
    def trace(action):
        _ENV.client.restoreState(snapshot)
        return _ENV._advance(action, distance_trace=True)[4]
    try:
        base = trace(anchor)
        derivative = np.zeros((*base.shape, 3))
        for axis in range(3):
            offset = np.eye(3)[axis] * epsilon
            plus, minus = np.clip(anchor + offset, -1, 1), np.clip(anchor - offset, -1, 1)
            derivative[..., axis] = (trace(plus) - trace(minus)) / (plus[axis] - minus[axis])
        intercept = base - (derivative * anchor).sum(-1)
        return np.concatenate([intercept[..., None], derivative], -1).reshape(-1, 4).astype(np.float32)
    finally:
        _ENV.client.restoreState(snapshot)
        _ENV.client.removeState(snapshot)


class SimulatorOracle:
    def __init__(self, workers=8, epsilon=.002, temperature=.02, cache_size=60000):
        if workers < 1 or epsilon <= 0 or temperature <= 0:
            raise ValueError('Invalid exact-oracle settings')
        self.workers, self.epsilon, self.temperature = workers, epsilon, temperature
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                            initializer=_initialize_worker)
        self.control_interval_queries = self.cache_hits = self.unique_state_actions = 0
        self.elapsed_query_seconds = 0.
        self.margin_only_queries = self.static_distance_queries = 0

    def margins_numpy(self, observations, actions):
        """Independent true interval labels, without unnecessary gradient queries."""
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if observations.shape != (len(actions), 25) or actions.shape != (len(actions), 3):
            raise ValueError('Margin batch state/action shapes differ')
        values = list(self.executor.map(_margin_query, zip(observations, actions),
            chunksize=max(1, len(actions) // (self.workers * 4))))
        self.margin_only_queries += len(actions)
        return np.asarray(values, dtype=np.float32)

    def state_distances_numpy(self, observations):
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != 25:
            raise ValueError('Distance batch requires shared 25-dimensional states')
        values = list(self.executor.map(_state_distance_query, observations,
            chunksize=max(1, len(observations) // (self.workers * 4))))
        self.static_distance_queries += len(observations)
        return np.asarray(values, dtype=np.float32)

    def evaluate_numpy(self, observations, actions):
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != 25 or actions.shape != (len(observations), 3):
            raise ValueError('Oracle batch state/action shapes differ')
        start = time.monotonic()
        missing = OrderedDict()
        keys = []
        for obs, action in zip(observations, actions):
            key = obs.tobytes() + action.tobytes()
            keys.append(key)
            if key in self.cache:
                self.cache_hits += 1
            elif key not in missing:
                missing[key] = (obs, action, self.epsilon)
        if missing:
            values = list(self.executor.map(_query, missing.values(),
                chunksize=max(1, len(missing) // (self.workers * 4))))
            self.cache.update(zip(missing, values))
            self.unique_state_actions += len(missing)
            self.control_interval_queries += 7 * len(missing)
        result = np.stack([self.cache[key] for key in keys])
        for key in keys:
            self.cache.move_to_end(key)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        self.elapsed_query_seconds += time.monotonic() - start
        return result

    def __call__(self, obs, action):
        import torch
        values = self.evaluate_numpy(obs.detach().cpu().numpy(), action.detach().cpu().numpy())
        margin = torch.as_tensor(values[:, 0], device=action.device, dtype=action.dtype)
        normal = torch.as_tensor(values[:, 1:4], device=action.device, dtype=action.dtype)
        valid = torch.as_tensor(values[:, 4] > 1e-7, device=action.device)
        weight = torch.sigmoid(margin / self.temperature) * valid
        return normal.detach(), weight.detach()

    def statistics(self):
        return dict(oracle_kind='actual_controller_simulator_finite_difference', workers=self.workers,
                    epsilon=self.epsilon, temperature=self.temperature,
                    unique_state_actions=self.unique_state_actions,
                    counterfactual_control_intervals=self.control_interval_queries,
                    cached_rows=self.cache_hits, query_seconds=self.elapsed_query_seconds,
                    margin_only_queries=self.margin_only_queries,
                    static_distance_queries=self.static_distance_queries,
                    learned_model_queries=0, candidate_actions_selected=0)

    def close(self):
        self.executor.shutdown(wait=True)


class ControllerJacobianOracle(SimulatorOracle):
    """State-cached first-order approximation through the real IK/motor mapping.

    Per-state/anchor 7 simulator intervals supply all 21-frame x10-link distance
    intercepts/Jacobians. Every flow midpoint uses their lower distance envelope.
    Exact finite-difference SimulatorOracle remains the independent diagnostic;
    this approximation must never be presented as an error-free oracle bound.
    """
    def __init__(self, workers=8, epsilon=.05, temperature=.02, cache_size=4000000,
                 action_grid_step=.5):
        super().__init__(workers, epsilon, temperature, cache_size)
        if action_grid_step <= 0:
            raise ValueError('The local action anchor spacing must be positive')
        self.action_grid_step = action_grid_step
        self.device_batches = OrderedDict()
        self.reference_rows = 0
        self.states_seen = set()

    def anchors_numpy(self, actions):
        anchors = np.clip(np.round(actions / self.action_grid_step) * self.action_grid_step, -1, 1).astype(np.float32)
        anchors[anchors == 0] = 0.  # canonical positive zero in numerical cache keys
        return anchors

    def evaluate_fields_numpy(self, observations, actions):
        observations = np.asarray(observations, dtype=np.float32)
        anchors = self.anchors_numpy(np.asarray(actions, dtype=np.float32))
        missing = OrderedDict()
        keys = []
        start = time.monotonic()
        for obs, anchor in zip(observations, anchors):
            state_key = obs.tobytes()
            self.states_seen.add(state_key)
            key = state_key + anchor.tobytes()
            keys.append(key)
            if key in self.cache:
                self.cache_hits += 1
            elif key not in missing:
                missing[key] = (obs, self.epsilon, anchor)
        if missing:
            values = list(self.executor.map(_interval_jacobian_query, missing.values(),
                chunksize=max(1, len(missing) // (self.workers * 4))))
            self.cache.update(zip(missing, values))
            self.unique_state_actions += len(missing)
            self.control_interval_queries += 7 * len(missing)
        fields = np.stack([self.cache[key] for key in keys])
        for key in keys:
            self.cache.move_to_end(key)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        self.elapsed_query_seconds += time.monotonic() - start
        return fields

    def evaluate_numpy(self, observations, actions):
        fields = self.evaluate_fields_numpy(observations, actions)
        actions = np.asarray(actions, dtype=np.float32)
        distances = fields[..., 0] + np.einsum('bpc,bc->bp', fields[..., 1:], actions)
        selected = distances.argmin(-1)
        gradient = -fields[np.arange(len(actions)), selected, 1:]
        norm = np.linalg.norm(gradient, axis=-1)
        normal = gradient / np.where(norm > 1e-7, norm, 1.)[:, None]
        normal[norm <= 1e-7] = 0.
        return np.column_stack([.015 - distances.min(-1), normal, norm]).astype(np.float32)

    def __call__(self, obs, action):
        import torch
        observations = obs.detach().cpu().numpy()
        actions = action.detach().cpu().numpy()
        anchors = self.anchors_numpy(actions)
        key = (observations.tobytes(), anchors.tobytes(), str(action.device), str(action.dtype))
        if key not in self.device_batches:
            fields = self.evaluate_fields_numpy(observations, actions)
            self.device_batches[key] = torch.as_tensor(fields, dtype=action.dtype, device=action.device)
            while len(self.device_batches) > 4:
                self.device_batches.popitem(last=False)
        else:
            self.device_batches.move_to_end(key)
        fields = self.device_batches[key]
        distances = fields[..., 0] + (fields[..., 1:] * action.detach()[:, None]).sum(-1)
        minimum, selected = distances.min(-1)
        gradient = -fields[torch.arange(len(action), device=action.device), selected, 1:]
        norm = gradient.norm(dim=-1)
        valid = norm > 1e-7
        denominator = torch.where(valid, norm, torch.ones_like(norm))
        normal = torch.where(valid[:, None], gradient / denominator[:, None], torch.zeros_like(gradient))
        weight = torch.sigmoid((.015 - minimum) / self.temperature) * valid
        self.reference_rows += len(action)
        return normal.detach(), weight.detach()

    def statistics(self):
        result = super().statistics()
        result.pop('unique_state_actions')
        result.update(oracle_kind='first_order_controller_distance_oracle',
                      backend='interval_distance_controller_jacobian',
                      unique_states=len(self.states_seen), unique_state_anchors=self.unique_state_actions,
                      reference_rows=self.reference_rows,
                      reference_action='nearest numerical cache anchor, never an executed candidate',
                      action_anchor_spacing=self.action_grid_step, linearization_epsilon=self.epsilon,
                      planes_per_state=210, exact_for_arbitrary_actions=False)
        return result

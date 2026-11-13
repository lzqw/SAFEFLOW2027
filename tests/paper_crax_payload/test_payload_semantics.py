import json
from pathlib import Path

import numpy as np
import torch

from scripts.paper_crax_payload.payload_environment import (
    SafePayloadTransport, capture_payload_environment, restore_payload_environment)
from scripts.paper_crax_payload.payload_models import build_payload_learner


def test_payload_is_observed_held_body_and_own_collision_is_counted():
    env = SafePayloadTransport()
    try:
        obs, _ = env.reset(seed=42027)
        assert obs.shape == (45,)
        np.testing.assert_allclose(obs[25:28], env.payload_position(), atol=1e-7)
        assert env.client.getDynamicsInfo(env.payload_body, -1)[0] == .05
        # A collision confined to the lower carried body must not disappear
        # merely because arm-obstacle distance is positive.
        env.obstacle_position = env.payload_position() + [0., 0., -.07]
        env.sim.set_base_pose('obstacle', env.obstacle_position, [0., 0., 0., 1.])
        env.client.performCollisionDetection()
        _, _, done, truncated, info = env.step(np.zeros(3))
        assert done and not truncated and info['payload_collision'] and info['violation']
        assert info['arm_minimum_distance'] > 0.
        assert info['cost'] == 1.
        assert info['minimum_distance'] == min(info['arm_minimum_distance'], info['payload_minimum_distance'])
        assert env.physical_control_intervals == 1
    finally:
        env.close()


def test_interval_min_includes_arm_and_payload_at_every_substep():
    env = SafePayloadTransport()
    try:
        env.reset(seed=42028)
        result = env._advance(np.array([.1, .1, -.1]), distance_trace=True)
        minimum, collision, violation, _, trace = result
        assert trace.shape == (env.config.substeps + 1, len(env.protected_links) + 1)
        assert abs(minimum - trace.min()) < 1e-12
        assert violation == (minimum < env.config.safety_distance or collision)
        assert np.allclose(env._interval_component_min, [trace[:, :-1].min(), trace[:, -1].min()])
        assert env.physical_control_intervals == 1
    finally:
        env.close()


def test_payload_resume_preserves_partial_episode_and_observed_body_state():
    a, b = SafePayloadTransport(), SafePayloadTransport()
    try:
        a.reset(seed=42027)
        a.step(np.array([.1, .1, 0.]))
        state = capture_payload_environment(a)
        observation = restore_payload_environment(b, state)
        np.testing.assert_allclose(observation, state['observation'], rtol=0., atol=2e-6)
        assert a._elapsed == b._elapsed == 1
        action = np.array([.1, -.1, .05])
        left, right = a.step(action), b.step(action)
        # Solver warm-start caches are not claimed bitwise portable. Preserve
        # Markov observation/partials, then check actual next physics interval.
        np.testing.assert_allclose(left[0], right[0], rtol=0., atol=5e-4)
        assert abs(left[1] - right[1]) < 5e-4
        assert left[2:4] == right[2:4]
        assert a.physical_control_intervals + b.physical_control_intervals == 3
    finally:
        a.close(); b.close()


def test_all_payload_methods_have_shared45d_observation_and_binary_cost():
    config = json.loads(Path('configs/paper_crax_payload/payload_v1.json').read_text())
    torch.set_num_threads(1)
    for method in config['methods']:
        learner = build_payload_learner(config, method, 'cpu')
        assert learner.cost_limit == .1
        assert learner.cost_q.probability
        action = learner.action(np.zeros((2, 45), np.float32))
        assert action.shape == (2, 3) and np.isfinite(action).all()
        batch = dict(obs=torch.zeros(4, 45), next_obs=torch.zeros(4, 45), action=torch.zeros(4, 3),
                     reward=torch.zeros(4), done=torch.zeros(4), cost=torch.zeros(4), margin=torch.zeros(4))
        learner.update(batch)
        state = learner.state_dict()
        restored = build_payload_learner(config, method, 'cpu')
        restored.load_state_dict(state)
        assert restored.updates == 1 and restored.actor_updates == 0

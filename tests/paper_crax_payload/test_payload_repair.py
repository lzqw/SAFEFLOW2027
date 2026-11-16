from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.paper_crax_payload.payload_environment import (
    PayloadFiniteRadiusEnvironment, SafePayloadTransport,
    capture_payload_environment, restore_payload_environment)
from scripts.paper_crax_payload.payload_train import FORMAT, validate_parent


def test_only_registered_distribution_can_change_at_explicit_fork():
    config = json.loads(Path('configs/paper_crax_payload/payload_v1.json').read_text())
    saved = dict(format=FORMAT, configuration=config, method='s_ea', training_seed=41028)
    child = deepcopy(config)
    child['training_radius_family'] = [.060, .0675, .075]
    validate_parent(saved, child, 's_ea', 41028, adaptation=True)
    with pytest.raises(ValueError):
        validate_parent(saved, child, 's_ea', 41028)
    child['shared_learner']['actor_lr'] *= 2
    with pytest.raises(ValueError):
        validate_parent(saved, child, 's_ea', 41028, adaptation=True)


def test_radius_fork_uses_actual_collision_body_and_preserves_schedule_on_resume():
    a, b, nominal = PayloadFiniteRadiusEnvironment(), PayloadFiniteRadiusEnvironment(), SafePayloadTransport()
    try:
        seen = []
        for index in range(3):
            observation, _ = a.reset(seed=42027 if index == 0 else None)
            seen.append(a.config.obstacle_radius)
            assert observation[23] == np.float32(a.config.obstacle_radius)
            assert a.obstacle_body == a.sim._bodies_idx['obstacle']
            shape = a.client.getCollisionShapeData(a.obstacle_body, -1)[0]
            assert abs(shape[3][0] - a.config.obstacle_radius) < 1e-12
        assert set(seen) == set(a.radius_values)
        a.step(np.array([.1, .1, 0.]))
        state = capture_payload_environment(a)
        observation = restore_payload_environment(b, state)
        np.testing.assert_allclose(observation, state['observation'], rtol=0., atol=2e-6)
        assert a.radius_next_index == b.radius_next_index and b._elapsed == 1
        with pytest.raises(ValueError):
            restore_payload_environment(nominal, state)
        a.reset(); b.reset()
        assert a.config.obstacle_radius == b.config.obstacle_radius
        np.testing.assert_allclose(a._obs(), b._obs(), rtol=0., atol=2e-6)
    finally:
        a.close(); b.close(); nominal.close()


def test_static_retry_does_not_advance_radius_or_relax_safety(monkeypatch):
    env = PayloadFiniteRadiusEnvironment()
    native = SafePayloadTransport.reset
    calls = []
    def transient_failure(self, **kwargs):
        calls.append((self.config.obstacle_radius, kwargs.get('seed')))
        if len(calls) == 1:
            raise RuntimeError('Cannot obtain safe reachable arm AND payload endpoints')
        return native(self, **kwargs)
    monkeypatch.setattr(SafePayloadTransport, 'reset', transient_failure)
    try:
        _, info = env.reset(seed=42027)
        assert len(calls) == 2 and calls[0][0] == calls[1][0]
        assert calls[1][1] is None and env.radius_next_index == 1
        assert env.radius_reset_retry_batches == 1
        assert info['minimum_distance'] > env.config.safety_distance + .005
    finally:
        env.close()

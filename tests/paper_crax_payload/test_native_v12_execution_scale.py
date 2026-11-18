from types import SimpleNamespace

import numpy as np
import torch

from scripts.paper_crax_payload import train_native_p5_warmup_v12 as native


def test_shared_evaluator_executes_declared_scale_and_retains_control_horizon(monkeypatch):
    actions = []

    class Environment:
        config = SimpleNamespace(horizon=1, success_distance_m=.03)
        control_dt = .04
        goal = np.zeros(3)

        def reset(self, seed):
            return np.zeros(34, np.float32), dict(minimum_distance=.2)

        def task_position(self):
            return np.zeros(3)

        def step(self, action):
            actions.append(action.copy())
            return np.zeros(34, np.float32), 10., True, False, dict(
                minimum_distance=.2, distance_to_goal=0., safe_success=True,
                violation=False, collision=False)

        def close(self):
            pass

    monkeypatch.setattr(native, 'make_piper_env', lambda *a, **k: Environment())
    raw = np.array([[.4, -.2, .8]], np.float32)
    learner = SimpleNamespace(transitions=120000, updates=100, actor_updates=50,
                              device=torch.device('cpu'), action=lambda obs: raw.copy())
    cfg = dict(evaluation=dict(action_sampling_seed=11871000, conditions=[
        dict(name='nominal'), dict(name='execution_scale_085', action_scale=.85)]))
    result = native.evaluate(learner, cfg, {}, 'Payload', 1, 11870000)
    assert np.array_equal(actions[0], raw[0])
    assert np.allclose(actions[1], .85*raw[0], rtol=0, atol=1e-7)
    assert all(c['records'][0]['length'] == 1 for c in result['conditions'])

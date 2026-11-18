from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.paper_crax_payload.native_p5_adaptation import build_native, PandaAdaptationLearner
from scripts.paper_crax_payload import train_panda_reference_v12 as reference


def test_native45d_weights_optimizers_and_strict_arm_restore():
    torch.set_num_threads(1)
    # Constructor-only fixture; no dependency on locally archived raw reports.
    native = dict(shared_learner=dict(hidden_width=128, actor_lr=.0001, critic_lr=.0003,
        critic_only_updates=1000, actor_update_interval=2, margin_ema_tau=.005, rho=.25),
        cost=dict(episode_budget=.1))
    plan = json.loads(reference.PLAN.read_text())
    for arm in plan['arms']:
        learner = build_native(native, plan['lines']['panda_payload'], 'cpu')
        learner.set_transitions(240000)
        actor = deepcopy(learner.actor.state_dict()); optimizer = learner.actor_opt
        reference.attach(learner, plan, 'panda_payload', arm)
        assert isinstance(learner, PandaAdaptationLearner)
        assert learner.actor_opt is optimizer
        assert all(torch.equal(v, learner.actor.state_dict()[k]) for k, v in actor.items())
        assert learner.reference_identity['obs_dim'] == 45
        if arm == 'Panda-Metric-only':
            assert learner.actor.sampling_directional is False
            assert learner.actor.energy_directional is True
            assert learner.actor_regularizer({}) == 0.
        saved = deepcopy(learner.state_dict())
        other = build_native(native, plan['lines']['panda_payload'], 'cpu'); other.set_transitions(240000)
        reference.attach(other, plan, 'panda_payload', arm); other.load_state_dict(saved)
        wrong = build_native(native, plan['lines']['panda_payload'], 'cpu'); wrong.set_transitions(240000)
        other_arm = next(a for a in plan['arms'] if a != arm)
        reference.attach(wrong, plan, 'panda_payload', other_arm)
        with pytest.raises(ValueError, match='identical native Panda'):
            wrong.load_state_dict(saved)


def test_payload_centroid_threshold_scale_and_failures_retained(monkeypatch):
    actions = []

    class Environment:
        config = SimpleNamespace(horizon=1, success_distance=.035)
        payload_goal = np.zeros(3)
        control_dt = .04

        def __init__(self, config): pass
        def payload_position(self): return np.zeros(3)
        def reset(self, seed): return np.zeros(45, np.float32), dict(minimum_distance=.02)
        def step(self, action):
            actions.append(action.copy())
            return np.zeros(45, np.float32), -10., True, False, dict(
                distance_to_goal=.033, safe_success=False, violation=True,
                collision=False, minimum_distance=.014)
        def close(self): pass

    monkeypatch.setattr(reference, 'SafePayloadTransport', Environment)
    raw = np.array([[.4, -.2, .8]], np.float32)
    learner = SimpleNamespace(transitions=240000, updates=10, actor_updates=5,
        safety_multiplier=.5, device=torch.device('cpu'), action=lambda obs: raw.copy())
    config = dict(task={}, evaluation=dict(development_layout_seed_start=11880000,
        action_sampling_seed=11881000, conditions=[dict(name='nominal'),
        dict(name='execution_scale_085', action_scale=.85)]))
    result = reference.evaluate(learner, config, 1)
    assert np.array_equal(actions[0], raw[0]) and np.allclose(actions[1], .85*raw[0])
    assert result['physical_evaluation_transitions'] == 2
    for group in result['conditions']:
        assert group['violation_count'] == 1 and group['safe_success_count'] == 0
        assert group['records'][0]['ordinary_terminal_task_success'] is True

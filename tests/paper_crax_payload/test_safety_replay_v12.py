from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.paper_crax_payload.boundary_replay_v12 import SafetyOnlyBoundaryReplay
from scripts.paper_crax_payload.direction_development import DirectionDevelopmentLearner
from scripts.paper_crax_payload.train_safety_replay_v12 import build, validate
from scripts.star.arm_safe_entropy.train import Replay


def batch():
    return dict(obs=torch.randn(8, 25), next_obs=torch.randn(8, 25), action=torch.rand(8, 3),
                reward=torch.randn(8), done=torch.ones(8), cost=torch.ones(8), margin=torch.ones(8)*.001)


def test_explicit_same_safety_batch_matches_default_native_update():
    torch.set_num_threads(1); torch.manual_seed(923)
    a = DirectionDevelopmentLearner(device='cpu', critic_only_updates=0,
                                    component='full', normal_shrink=True, direction_weight=1.)
    torch.manual_seed(923)
    b = DirectionDevelopmentLearner(device='cpu', critic_only_updates=0,
                                    component='full', normal_shrink=True, direction_weight=1.)
    value = batch()
    for _ in range(2):
        rng = torch.get_rng_state(); a.update(value)
        torch.set_rng_state(rng); b.update(value, safety_batch=value)
    for name in ('actor', 'target_actor', 'q', 'qt', 'cost_q', 'cost_qt', 'margin_online', 'margin_geometry'):
        assert all(torch.equal(v, getattr(b, name).state_dict()[k])
                   for k, v in getattr(a, name).state_dict().items())
    assert a.metrics() == b.metrics()


def test_safety_batch_changes_only_safety_updates_before_geometry_and_actor():
    torch.manual_seed(923)
    a = DirectionDevelopmentLearner(device='cpu', critic_only_updates=1000)
    torch.manual_seed(923)
    b = DirectionDevelopmentLearner(device='cpu', critic_only_updates=1000)
    ordinary = batch(); safety = deepcopy(ordinary)
    safety['cost'].zero_(); safety['margin'].fill_(-.03)
    rng = torch.get_rng_state(); a.update(ordinary)
    torch.set_rng_state(rng); b.update(ordinary, safety_batch=safety)
    assert all(torch.equal(v, b.q.state_dict()[k]) for k, v in a.q.state_dict().items())
    assert all(torch.equal(v, b.actor.state_dict()[k]) for k, v in a.actor.state_dict().items())
    assert any(not torch.equal(v, b.margin_online.state_dict()[k]) for k, v in a.margin_online.state_dict().items())
    assert any(not torch.equal(v, b.cost_q.state_dict()[k]) for k, v in a.cost_q.state_dict().items())


def test_native_actor_pool_fourfold_safety_odds_and_full_resume():
    np.random.seed(92); replay = SafetyOnlyBoundaryReplay(1000, 1)
    for i in range(1000):
        margin = -.001 if i % 5 == 0 else -.02
        replay.add(obs=[i], next_obs=[i], action=[0, 0, 0], reward=0, done=1, cost=0, margin=margin)
        replay.observe_transition(0, True, dict(local_margin=margin, violation=False, collision=False))
    state = deepcopy(replay.state_dict()); rng = np.random.get_state()
    actual = replay.sample(30000, 'cpu', recent=True)
    np.random.set_state(rng); reference = Replay(1000, 1); reference.load_state_dict(state)
    expected = reference.sample(30000, 'cpu', recent=True)
    assert torch.equal(actual['obs'], expected['obs'])
    assert abs(float((actual['obs'].flatten() % 5 == 0).float().mean())-.2) < .015
    assert abs(float((actual['safety_batch']['obs'].flatten() % 5 == 0).float().mean())-.5) < .015
    restored = SafetyOnlyBoundaryReplay(1000, 1); restored.load_state_dict(state)
    cfg = json.loads(Path('configs/paper_crax_payload/simulation_v12_safereach_safety_replay.json').read_text()); validate(cfg)
    for arm in cfg['arms']:
        learner = build(cfg, arm, 2033, 'cpu'); saved = deepcopy(learner.state_dict())
        learner.update(batch() | dict(safety_batch=batch()))
        assert learner.updates == 1 and learner.reward_updates == 2 and learner.actor_updates == 0
        other = build(cfg, arm, 2033, 'cpu'); other.load_state_dict(saved)
        bad = deepcopy(saved); bad.pop('safety_only_replay_v12')
        with pytest.raises(ValueError, match='identical'):
            other.load_state_dict(bad)

"""Targeted numerical tests; no training campaign or simulator required."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.paper_crax_payload.direction_annealing import (
    AnnealingLearner, fixed_probe, gradient_diagnosis, schedule_values, validate_config)
from scripts.paper_crax_payload.direction_development import DirectionDevelopmentLearner
from scripts.paper_crax_payload.evaluate_direction_annealing import attach_interval_observer


def test_schedule_absolute_clock_and_same_covariance_for_target():
    assert schedule_values('fixed', 250000) == (1., .25)
    assert schedule_values('loss_only', 100000) == (1., .25)
    assert schedule_values('loss_only', 150000) == (.1, .25)
    assert schedule_values('loss_and_rho', 125000) == pytest.approx((.55, .375))
    assert schedule_values('loss_and_rho', 250000) == (.1, .5)
    model = AnnealingLearner('loss_and_rho', device='cpu')
    model.set_transitions(150000)
    assert model.actor.rho == model.target_actor.rho == .5
    assert model.actor.variance == model.target_actor.variance == .04
    config = json.loads(Path('configs/paper_crax_payload/direction_annealing_v6.json').read_text())
    validate_config(config)
    config['schedule']['lambda_final'] = 0.
    with pytest.raises(ValueError, match='entire registered'): validate_config(config)


def test_fixed_branch_import_matches_original_updates_and_schedule_resume():
    torch.set_num_threads(1)
    torch.manual_seed(80)
    parent = DirectionDevelopmentLearner(component='full', rho=.25, normal_shrink=True,
        direction_weight=1., device='cpu', critic_only_updates=0, actor_interval=1)
    parent.set_transitions(100000)
    state = deepcopy(parent.state_dict())
    fork = AnnealingLearner('fixed', device='cpu', critic_only_updates=0, actor_interval=1)
    fork.load_parent_state(state)
    batch = dict(obs=torch.zeros(8,25), next_obs=torch.zeros(8,25), action=torch.zeros(8,3),
        reward=torch.ones(8), done=torch.zeros(8), cost=torch.zeros(8), margin=torch.zeros(8))
    for model in (parent, fork):
        torch.manual_seed(99); model.update(batch)
    for name in ('actor', 'target_actor', 'q', 'qt', 'cost_q', 'margin_geometry'):
        assert all(torch.equal(v, getattr(fork,name).state_dict()[k])
                   for k,v in getattr(parent,name).state_dict().items())
    assert fork.updates == parent.updates and fork.metrics()['alpha'] == parent.metrics()['alpha']
    revised = AnnealingLearner('loss_and_rho', device='cpu', critic_only_updates=0, actor_interval=1)
    revised.load_parent_state(state); revised.set_transitions(175000)
    saved = deepcopy(revised.state_dict())
    restored = AnnealingLearner('loss_and_rho', device='cpu', critic_only_updates=0, actor_interval=1)
    restored.load_state_dict(saved)
    assert restored.actor.rho == restored.target_actor.rho == .5
    assert restored.current_direction_weight == .1 and restored.transitions == 175000
    with pytest.raises(ValueError, match='same v6 schedule'):
        AnnealingLearner('loss_only', device='cpu').load_state_dict(saved)
    with pytest.raises(ValueError, match='reinterpret'):
        parent.load_state_dict(saved)


def test_gradient_probe_leaves_rng_parameters_grads_and_optimizer_untouched():
    torch.set_num_threads(1)
    model = AnnealingLearner('loss_only', device='cpu'); model.set_transitions(150000)
    model.safety_multiplier = 0.
    config = json.loads(Path('configs/paper_crax_payload/direction_annealing_v6.json').read_text())
    replay = dict(count=70, data=dict(obs=np.zeros((70,25), np.float32)))
    probe = fixed_probe(replay, config); assert probe['sha256'] == fixed_probe(replay, config)['sha256']
    parameters_before = [p.detach().clone() for p in model.actor.parameters()]
    rng_before = torch.get_rng_state().clone(); metrics = deepcopy(model.metrics())
    count = model.updates
    result = gradient_diagnosis(model, probe)
    assert set(result['norms']) == {'reward','cost','energy','direction'}
    assert result['norms']['cost'] == 0 and result['cosines']['cost__energy'] is None
    assert result['direction_weight'] == .1 and result['direction_denominator'] > 0
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert model.metrics() == metrics and model.updates == count
    assert all(torch.equal(p,q) and p.grad is None for p,q in zip(model.actor.parameters(),parameters_before))
    assert model.actor_opt.state == {}


def test_interval_observer_uses_one_actual_advance_and_keeps_original_results():
    class Dummy:
        def __init__(self): self.calls = 0
        def _advance(self, action, distance_trace=False):
            self.calls += 1
            assert distance_trace
            return (.01, False, True, False, np.array([[.1,.02],[.03,.01]]))
    env = Dummy(); attach_interval_observer(env)
    assert env._advance(np.zeros(3)) == (.01, False, True, False)
    assert env.calls == 1
    np.testing.assert_array_equal(env._v6_link_interval_minima, [.03,.01])

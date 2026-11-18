from copy import deepcopy
import json

import pytest
import torch

from scripts.paper_crax_payload.native_p5_adaptation import build_native
from scripts.paper_crax_payload.train_localized_geometry_v12 import attach, PLANS


def native_config():
    return dict(methods=['EA'], shared_learner=dict(hidden_width=128, actor_lr=.0001, critic_lr=.0003,
        critic_only_updates=1000, margin_ema_tau=.005, actor_update_interval=2, rho=.25,
        reference_variance=.04, flow_steps=4, energy_target_per_action=.25,
        kinetic_alpha_per_variance=.02, episode_budget=.1))


def test_temperature_changes_weights_only_and_resume_cannot_reinterpret_it():
    torch.set_num_threads(1)
    plan = json.loads(PLANS['piper_reach'].read_text()); arm = plan['arms'][0]
    native = native_config(); torch.manual_seed(92)
    learner = build_native(native, plan['lines']['piper_reach'], 'cpu'); learner.set_transitions(400000)
    obs = torch.zeros(3, 34); action = torch.randn(3, 3, requires_grad=True)
    online = learner.margin_online(obs, action).detach().clone()
    geometry_weights = deepcopy(learner.margin_geometry.state_dict()); optimizer = learner.margin_opt
    attach(learner, plan, 'piper_reach', arm)
    shifted = learner.margin_geometry(obs, action)
    gradient = torch.autograd.grad(shifted.sum(), action)[0]
    learner.margin_geometry.temperature = .02
    original_gradient = torch.autograd.grad(learner.margin_geometry(obs, action).sum(), action)[0]
    assert torch.equal(gradient, original_gradient)
    assert torch.equal(learner.margin_online(obs, action), online)
    assert learner.margin_opt is optimizer
    assert all(torch.equal(v, learner.margin_geometry.state_dict()[k]) for k, v in geometry_weights.items())
    learner.margin_geometry.temperature = .005
    # Same signed margin/activation: stronger near danger, weaker safely inside.
    raw = torch.tensor([-.02, -.01, 0., .002, .01])
    short, wide = torch.sigmoid(raw/.005), torch.sigmoid(raw/.02)
    assert torch.all(short[:2] < wide[:2]) and short[2] == wide[2]
    assert torch.all(short[3:] > wide[3:])
    saved = deepcopy(learner.state_dict())
    other = build_native(native, plan['lines']['piper_reach'], 'cpu'); other.set_transitions(400000)
    attach(other, plan, 'piper_reach', arm); other.load_state_dict(saved)
    assert other.margin_geometry.temperature == .005
    wrong = deepcopy(saved); wrong['localized_geometry_v12']['geometry_temperature_m'] = .02
    with pytest.raises(ValueError, match='identical localized'):
        other.load_state_dict(wrong)

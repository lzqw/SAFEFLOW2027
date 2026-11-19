from copy import deepcopy
import json

import pytest
import torch

from scripts.paper_crax_payload.direction_development import NormalShrinkActor
from scripts.paper_crax_payload.endpoint_residual_v12 import EndpointRecordedActor,endpoint_residual
from scripts.paper_crax_payload.train_endpoint_residual_v12 import PLAN,build
from scripts.paper_crax_payload.train_matched_ablation_v12 import AblationLearner
from scripts.star.arm_safe_entropy.endpoint_geometry import EndpointGeometry
from scripts.star.arm_safe_entropy.geometry import LocalMarginEnsemble


def test_recording_preserves_exact_sampler_energy_geometry_trace_and_rng():
    torch.set_num_threads(1);torch.manual_seed(2034)
    base = NormalShrinkActor(25)
    candidate = EndpointRecordedActor(25);candidate.load_state_dict(base.state_dict())
    geometry = EndpointGeometry(LocalMarginEnsemble(),transitions=100000)
    obs = torch.randn(12,25); noise = torch.randn(4,12,3)
    before = torch.random.get_rng_state()
    old = base(obs,geometry=geometry,return_reference=True,noise_increments=noise)
    new = candidate(obs,geometry=geometry,return_reference=True,noise_increments=noise)
    assert torch.equal(before,torch.random.get_rng_state())
    assert all(torch.equal(l,r) for l,r in zip(old[:3],new[:3]))
    assert all(torch.equal(old[3][key],new[3][key]) for key in old[3])
    assert new[3]['endpoint_action'] is new[0]
    assert new[3]['endpoint_action'].requires_grad


def test_loss_only_penalizes_positive_margin_and_backpropagates_toward_safety():
    margin = torch.tensor([-.015,0.,.015],requires_grad=True)
    loss = endpoint_residual(margin)
    torch.testing.assert_close(loss,torch.tensor(1./3))
    loss.backward()
    assert margin.grad[0]==0 and margin.grad[-1]>0


def test_native_initialization_and_control_update_are_exact_without_new_parameters():
    torch.set_num_threads(1);config = json.loads(PLAN.read_text())
    spec = config['arms'][0]
    torch.manual_seed(2034);original = AblationLearner(deepcopy(spec),device='cpu')
    state = torch.random.get_rng_state()
    torch.manual_seed(2034);control = build(config,spec,2034,'cpu')
    assert torch.equal(state,torch.random.get_rng_state())
    assert set(dict(original.actor.named_parameters()))==set(dict(control.actor.named_parameters()))
    original.set_transitions(50000);control.set_transitions(50000)
    obs = torch.randn(12,25);noise = torch.randn(4,12,3)
    old = original.actor(obs,geometry=original.geometry,return_reference=True,noise_increments=noise)
    new = control.actor(obs,geometry=control.geometry,return_reference=True,noise_increments=noise)
    assert all(torch.equal(l,r) for l,r in zip(old[:3],new[:3]))
    torch.testing.assert_close(original.actor_regularizer(old[3]),control.actor_regularizer(new[3]),rtol=0,atol=0)
    original.updates = control.updates = 1001
    batch = dict(obs=obs,action=torch.randn(12,3).clamp(-1,1),next_obs=torch.randn(12,25),
        reward=torch.randn(12),cost=torch.zeros(12),done=torch.zeros(12),margin=-.02*torch.ones(12))
    state = torch.random.get_rng_state()
    original.update(batch)
    torch.random.set_rng_state(state)
    control.update(batch)
    assert original.actor_updates==control.actor_updates==1
    for attr in ('actor','target_actor','q','cost_q','margin_geometry'):
        left,right = getattr(original,attr).state_dict(),getattr(control,attr).state_dict()
        assert all(torch.equal(left[k],right[k]) for k in left)
    saved = control.state_dict();saved.pop('endpoint_residual_v12')
    with pytest.raises(ValueError,match='reinterpret'):
        control.load_state_dict(saved)


def test_candidate_uses_frozen_existing_margin_and_finite_actor_gradient():
    torch.set_num_threads(1);torch.manual_seed(2034)
    config = json.loads(PLAN.read_text());learner = build(config,config['arms'][1],2034,'cpu')
    learner.set_transitions(50000)
    # Force positive predictions while retaining native differentiable action inputs.
    with torch.no_grad():
        for head in learner.margin_geometry.heads:
            head[-1].bias.add_(1.)
    obs = torch.randn(12,25)
    result = learner.actor(obs,geometry=learner.geometry,return_reference=True)
    loss = learner.actor_regularizer(result[3]);loss.backward()
    assert learner._last_metrics['endpoint_residual_loss']>0
    assert all(p.grad is None for p in learner.margin_geometry.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in learner.actor.parameters())

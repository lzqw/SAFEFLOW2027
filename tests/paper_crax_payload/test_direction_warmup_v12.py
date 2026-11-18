from copy import deepcopy

import pytest
import torch

from scripts.paper_crax_payload.direction_development import DirectionDevelopmentLearner
from scripts.paper_crax_payload.direction_optimization import OptimizationLearner
from scripts.paper_crax_payload.direction_warmup_v12 import DirectionWarmup, SafeReachWarmupLearner


def trace():
    # Positive outward velocity makes the real outward loss differentiable.
    velocity = torch.ones((2, 4, 3), requires_grad=True)
    normal = torch.zeros_like(velocity); normal[..., 0] = 1.
    weight = torch.full((2, 4), .5)
    return dict(velocities=velocity, normals=normal, weights=weight,
        vn=torch.full_like(weight, .025), vt=torch.full_like(weight, .04))


def test_effective_direction_gradient_and_covariance_at_mid_ramp():
    torch.set_num_threads(1)
    fixed = SafeReachWarmupLearner(DirectionWarmup(0), device='cpu')
    ramp = SafeReachWarmupLearner(DirectionWarmup(50000), device='cpu')
    ramp.set_transitions(25000)
    a, b = trace(), trace()
    fixed_loss = fixed.actor_regularizer(a); ramp_loss = ramp.actor_regularizer(b)
    fixed_loss.backward(); ramp_loss.backward()
    assert torch.allclose(b['velocities'].grad, .5*a['velocities'].grad)
    assert torch.allclose(ramp_loss.detach(), .5*fixed_loss.detach())
    assert ramp.actor.rho == ramp.target_actor.rho == .25
    assert ramp.actor.variance == ramp.target_actor.variance == .04
    ramp.set_transitions(50000)
    assert ramp.metrics()['direction_weight_effective'] == 1.


def test_fixed_baseline_matches_registered_v10_schedule_without_physics():
    torch.set_num_threads(1)
    candidate = dict(name='P5-A-lr', direction=dict(start=100000,end=150000,floor=1),
        actor_lr=dict(start=150000,end=200000,final=.00003))
    torch.manual_seed(73001)
    old = OptimizationLearner(candidate, device='cpu')
    torch.manual_seed(73001)
    new = SafeReachWarmupLearner(DirectionWarmup(0), device='cpu')
    for name in ('actor','target_actor','q','qt','cost_q','cost_qt','margin_geometry'):
        assert all(torch.equal(t,getattr(new,name).state_dict()[k])
                   for k,t in getattr(old,name).state_dict().items())
    for step in (0,50000,100000,150000,175000,200000,250000):
        old.set_transitions(step); new.set_transitions(step)
        assert old.current_direction_weight == new.metrics()['direction_weight_effective']
        assert old.actor_opt.param_groups[0]['lr'] == new.actor_opt.param_groups[0]['lr']


def test_strict_resume_retains_optimizer_state_and_rejects_other_schedules():
    torch.set_num_threads(1)
    original = SafeReachWarmupLearner(DirectionWarmup(75000,.25), device='cpu')
    # Nonempty Adam state verifies full native optimizer resume, no environment steps.
    for p in original.actor.parameters():
        p.grad = torch.ones_like(p)*.001
    original.actor_opt.step(); original.actor_opt.zero_grad(set_to_none=True)
    original.set_transitions(175000)
    saved = deepcopy(original.state_dict())
    restored = SafeReachWarmupLearner(DirectionWarmup(75000,.25), device='cpu')
    restored.load_state_dict(saved)
    assert restored.transitions == 175000
    assert restored.actor_opt.state
    assert restored.metrics()['direction_weight_effective'] == 1.
    assert restored.actor_opt.param_groups[0]['lr'] == pytest.approx(.000065)
    assert restored.q_opt.param_groups[0]['lr'] == .0003
    for setting in (DirectionWarmup(0),DirectionWarmup(50000),DirectionWarmup(75000)):
        with pytest.raises(ValueError,match='identical v12'):
            SafeReachWarmupLearner(setting,device='cpu').load_state_dict(saved)
    bad = deepcopy(saved); bad['actor_opt']['param_groups'][0]['lr'] = .1
    with pytest.raises(ValueError,match='optimizer LR'):
        SafeReachWarmupLearner(DirectionWarmup(75000,.25),device='cpu').load_state_dict(bad)
    missing = DirectionDevelopmentLearner(component='full',rho=.25,normal_shrink=True,
        direction_weight=1.,device='cpu').state_dict()
    with pytest.raises(ValueError,match='identical v12'):
        SafeReachWarmupLearner(DirectionWarmup(75000,.25),device='cpu').load_state_dict(missing)

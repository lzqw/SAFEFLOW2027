"""Only the numerical contracts required for EA-STAB; no simulation suite."""
import torch
from torch import nn

from scripts.star.arm_safe_entropy.endpoint_geometry import (
    EndpointGeometry, StableFlowActor, deterministic_tail, geometry_ramp)
from scripts.star.arm_safe_entropy.geometry import reference_variances
from scripts.star.arm_safe_entropy.models import ArmFlowActor


class LinearMargin(nn.Module):
    temperature = .02

    def __init__(self):
        super().__init__()
        self.axis = nn.Parameter(torch.tensor([1., 0., 0.]))

    def forward(self, obs, action):
        value = (action * self.axis).sum(-1)
        return torch.stack([value, value])


class LinearGenerator(nn.Module):
    num_steps = 2

    def __init__(self, rotate=True):
        super().__init__()
        self.matrix = nn.Parameter(torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]])
                                   if rotate else torch.zeros(3, 3))

    def velocity(self, obs, action, time):
        return action @ self.matrix.T


def test_zero_ramp_is_exact_legacy_iso_with_paired_noise_and_energy():
    torch.manual_seed(91)
    old, new = ArmFlowActor(25), StableFlowActor(25)
    new.load_state_dict(old.state_dict())
    obs = torch.randn(17, 25)
    geometry = EndpointGeometry(LinearMargin(), transitions=10000)
    torch.manual_seed(43)
    expected = old(obs)
    torch.manual_seed(43)
    actual = new(obs, geometry=geometry)
    for x, y in zip(expected, actual):
        torch.testing.assert_close(x, y, rtol=0., atol=0.)
    assert [geometry_ramp(n) for n in (0, 10000, 20000, 30000, 50000)] == [0., 0., .5, 1., 1.]


def test_linear_tail_pullback_uses_transpose_jacobian_and_current_generator():
    actor, margin = LinearGenerator(), LinearMargin()
    obs, u = torch.zeros(2, 25), torch.tensor([[.1, .2, .1], [.2, -.1, .0]])
    geometry = EndpointGeometry(margin, transitions=30000)
    n, q = geometry.query(actor, obs, u, u, 0)
    matrix = torch.eye(3) + .5*actor.matrix.detach() + .5*.5**2*(actor.matrix.detach()@actor.matrix.detach())
    expected = matrix.T @ torch.tensor([1., 0., 0.])
    expected /= expected.norm()
    torch.testing.assert_close(n, expected.expand_as(n))
    direct = EndpointGeometry(margin, 'endpoint_direct', 30000).query(actor, obs, u, u, 0)[0]
    torch.testing.assert_close(direct, torch.tensor([1., 0., 0.]).expand_as(n))
    other = geometry.query(LinearGenerator(False), obs, u, u, 0)[0]
    torch.testing.assert_close(other, direct)
    assert not torch.allclose(n, direct)
    vn, vt = reference_variances(q)
    assert (vn > 0).all() and (vt > 0).all()
    torch.testing.assert_close(vn + 2*vt, torch.full_like(vn, .12))


def test_last_step_matches_direct_gradient_and_real_clamp_has_no_straight_through():
    actor, margin = LinearGenerator(), LinearMargin()
    obs = torch.zeros(2, 25)
    u = torch.tensor([[.2, .1, -.1], [2., 3., 4.]])
    g = EndpointGeometry(margin, transitions=30000)
    normal, weight, diagnostic = g.query(actor, obs, u, u, 1, True)
    torch.testing.assert_close(normal[0], torch.tensor([1., 0., 0.]))
    torch.testing.assert_close(normal[1], torch.zeros(3))
    assert weight[1] == 0 and diagnostic['zero_gradient'][1]
    assert diagnostic['proposal_clipped'][1]
    vn, vt = reference_variances(weight)
    assert vn[1] == vt[1] == torch.tensor(.04)
    torch.testing.assert_close(deterministic_tail(actor, obs, u, 1), u.clamp(-1, 1))


def test_no_grad_calls_enable_input_gradients_but_never_parameter_gradients():
    actor, margin = LinearGenerator(), LinearMargin()
    u, obs = torch.ones(2, 3)*.1, torch.zeros(2, 25)
    with torch.no_grad():
        n, q = EndpointGeometry(margin, transitions=30000).query(actor, obs, u, u, 0)
    assert not n.requires_grad and not q.requires_grad
    assert actor.matrix.requires_grad and margin.axis.requires_grad
    assert actor.matrix.grad is None and margin.axis.grad is None
    assert (n.norm(dim=-1) - 1).abs().max() < 1e-6


def test_actual_proposal_site_and_actor_backpropagation_are_preserved():
    torch.manual_seed(1)
    actor, margin = StableFlowActor(25), LinearMargin()
    obs = torch.randn(5, 25)
    action, energy, _, trace = actor(obs, geometry=EndpointGeometry(margin, transitions=30000),
                                    return_reference=True, deterministic=True)
    current = torch.zeros(5, 3)
    for k in range(actor.num_steps):
        t = torch.full((5, 1), k / actor.num_steps)
        mid = (current + .5/actor.num_steps*actor.velocity(obs, current, t)).clamp(-1, 1)
        proposed = current + actor.velocity(obs, mid, t+.5/actor.num_steps)/actor.num_steps
        torch.testing.assert_close(trace['injection_proposals'][:, k], proposed)
        current = proposed.clamp(-1, 1)
    (action.square().mean() + .001*energy.mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())
    assert all(p.grad is None for p in margin.parameters())


def test_ramp_is_applied_once_to_both_sampling_and_energy_covariance():
    actor, margin = StableFlowActor(25), LinearMargin()
    obs, u = torch.zeros(3, 25), torch.ones(3, 3)*.1
    half = EndpointGeometry(margin, 'local', 20000)
    full = EndpointGeometry(margin, 'local', 30000)
    _, q1 = half.query(actor, obs, u, u, 0)
    _, q2 = full.query(actor, obs, u, u, 0)
    torch.testing.assert_close(q1, .5*q2)
    _, _, _, trace = actor(obs, geometry=half, return_reference=True)
    expected_vn, expected_vt = reference_variances(trace['weights'])
    torch.testing.assert_close(trace['vn'], expected_vn)
    torch.testing.assert_close(trace['vt'], expected_vt)

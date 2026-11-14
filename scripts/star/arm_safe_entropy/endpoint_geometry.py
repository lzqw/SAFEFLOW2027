"""EA-STAB geometry at the actual noise injection point; legacy sampler unchanged.

The tail is a frozen copy *of the currently queried generator's parameters*,
not a second behavior policy. It predicts no future random noise or physics.
All geometry is detached before constructing the shared sampling/energy D.
"""
from contextlib import contextmanager
import math

import torch

from scripts.star.arm_safe_entropy.geometry import strict_normals, reference_variances
from scripts.star.arm_safe_entropy.models import ArmFlowActor


@contextmanager
def frozen_parameters(*modules):
    parameters = [p for module in modules for p in module.parameters()]
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def geometry_ramp(transitions, start=10000, end=30000):
    if end <= start:
        raise ValueError('Geometry ramp must have positive duration')
    return min(1., max(0., (transitions - start) / (end - start)))


def deterministic_tail(actor, obs, proposal, step):
    """First apply the injection-position clamp, then remaining midpoint steps.

    This is differentiable with respect to proposal, including the true clamp
    Jacobian; callers freeze parameters but MUST NOT disable input autograd.
    """
    action = proposal.clamp(-1., 1.)
    dt = 1. / actor.num_steps
    for index in range(step + 1, actor.num_steps):
        t = torch.full_like(action[..., :1], index * dt)
        midpoint = (action + .5 * dt * actor.velocity(obs, action, t)).clamp(-1., 1.)
        velocity = actor.velocity(obs, midpoint, t + .5 * dt)
        action = (action + dt * velocity).clamp(-1., 1.)
    return action


class EndpointGeometry:
    MODES = ('none', 'local', 'endpoint_direct', 'endpoint_pullback')

    def __init__(self, margin, mode='endpoint_pullback', transitions=0):
        if mode not in self.MODES or (mode != 'none' and margin is None):
            raise ValueError('Geometry needs a known mode and its own margin model')
        self.margin, self.mode, self.transitions = margin, mode, transitions

    def query(self, actor, obs, midpoint, proposal, step, return_diagnostics=False):
        beta = geometry_ramp(self.transitions)
        if self.mode == 'none' or beta == 0.:
            zero = torch.zeros_like(proposal[..., 0])
            normal, weight = torch.zeros_like(proposal), zero
            diagnostics = dict(valid=zero.bool(), beta=torch.full_like(zero, beta),
                raw_weight=zero, zero_gradient=zero.bool(),
                proposal_clipped=(proposal.detach().abs() > 1.).any(-1))
        else:
            with frozen_parameters(actor, self.margin), torch.enable_grad():
                if self.mode == 'local':
                    variable = midpoint.detach().requires_grad_(True)
                    endpoint = variable
                elif self.mode == 'endpoint_direct':
                    # Diagnostic: terminal normal without the tail Jacobian.
                    with torch.no_grad():
                        predicted = deterministic_tail(actor, obs.detach(), proposal.detach(), step)
                    variable = predicted.detach().requires_grad_(True)
                    endpoint = variable
                else:
                    variable = proposal.detach().requires_grad_(True)
                    endpoint = deterministic_tail(actor, obs.detach(), variable, step)
                margins = self.margin(obs.detach(), endpoint)
                gradients = torch.stack([torch.autograd.grad(margins[j].sum(), variable,
                    retain_graph=j < len(margins)-1, create_graph=False)[0]
                    for j in range(len(margins))])
            units, head_valid = strict_normals(gradients)
            normal, mean_valid = strict_normals(gradients.mean(0))
            valid = head_valid.all(0) & mean_valid
            agreement = (units[0] * units[1]).sum(-1).clamp(0., 1.)
            raw_weight = (margins.detach().mean(0) / self.margin.temperature).sigmoid() * agreement * valid
            weight = beta * raw_weight  # Exactly one mixture/ramp factor.
            diagnostics = dict(valid=valid, zero_gradient=~valid,
                beta=torch.full_like(weight, beta), raw_weight=raw_weight,
                agreement=agreement, endpoint=endpoint.detach(),
                proposal_clipped=(proposal.detach().abs() > 1.).any(-1))
        normal, weight = normal.detach(), weight.detach()
        if return_diagnostics:
            return normal, weight, {k:v.detach() for k,v in diagnostics.items()}
        return normal, weight


class StableFlowActor(ArmFlowActor):
    """Same velocity network, K/noise/energy, with an explicit geometry site."""

    def forward(self, obs, deterministic=False, geometry=None, return_reference=False,
                noise_increments=None, sampling_directional=True, energy_directional=True):
        if self.reference_control != 'aligned':
            raise ValueError('EA-STAB does not add a random-direction behavior variant')
        action = torch.zeros((*obs.shape[:-1], 3), device=obs.device, dtype=obs.dtype)
        energy = torch.zeros_like(action[..., 0])
        clips = torch.zeros_like(energy)
        dt = 1. / self.num_steps
        normals, weights, vns, vts, midpoints, proposals, invalid, proposal_clips = ([] for _ in range(8))
        for index in range(self.num_steps):
            t = torch.full_like(action[..., :1], index * dt)
            midpoint = (action + .5 * dt * self.velocity(obs, action, t)).clamp(-1., 1.)
            velocity = self.velocity(obs, midpoint, t + .5 * dt)
            injection = action + dt * velocity
            if geometry is None:
                normal, weight = torch.zeros_like(action), torch.zeros_like(energy)
                bad = torch.zeros_like(energy, dtype=torch.bool)
            else:
                normal, weight, diagnostic = geometry.query(self, obs, midpoint, injection, index, True)
                bad = diagnostic['zero_gradient']
            vn, vt = reference_variances(weight, self.variance, self.rho)
            # Opt-in component study: legacy callers retain the identical shared-D path.
            energy_vn, energy_vt = (vn, vt) if energy_directional else (
                torch.full_like(vn, self.variance), torch.full_like(vt, self.variance))
            vnormal = (velocity * normal).sum(-1)
            vtangent2 = (velocity.square().sum(-1) - vnormal.square()).clamp_min(0.)
            energy = energy + .5 * dt * (vnormal.square() / energy_vn + vtangent2 / energy_vt)
            if not sampling_directional:
                vn, vt = torch.full_like(vn, self.variance), torch.full_like(vt, self.variance)
            if deterministic:
                noise = torch.zeros_like(action)
            elif noise_increments is not None:
                noise = noise_increments[index]
            else:
                noise = torch.randn_like(action)
            increment = vt.sqrt()[..., None] * noise + ((vn.sqrt()-vt.sqrt()) *
                (noise*normal).sum(-1))[..., None] * normal
            proposal = injection + math.sqrt(dt) * increment
            clips = clips + (proposal.abs() > 1.).to(obs.dtype).mean(-1) / self.num_steps
            action = proposal.clamp(-1., 1.)
            if return_reference:
                normals.append(normal); weights.append(weight); vns.append(vn); vts.append(vt)
                midpoints.append(midpoint.detach()); proposals.append(injection.detach())
                invalid.append(bad); proposal_clips.append((injection.detach().abs() > 1.).any(-1))
        if return_reference:
            return action, energy, clips, dict(normals=torch.stack(normals, -2),
                weights=torch.stack(weights, -1), vn=torch.stack(vns, -1), vt=torch.stack(vts, -1),
                midpoints=torch.stack(midpoints, -2), injection_proposals=torch.stack(proposals, -2),
                geometry_zero_gradient=torch.stack(invalid, -1),
                geometry_proposal_clipped=torch.stack(proposal_clips, -1),
                terminal_clipping_any=(proposal.abs() > 1.).any(-1).to(obs.dtype))
        return action, energy, clips

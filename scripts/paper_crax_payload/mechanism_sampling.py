"""Frozen-Actor interventions for M1/M2, not differently trained algorithms.

The normal oracle is deliberately absent from the sampling API.  Learned
Local/Direct/Pullback geometry uses the original differentiable clamped tail.
Random uses an independent axis held throughout a sampled action, with the
same learned-Pullback validity gate, never a true-oracle action selection.
"""
import math

import torch

from scripts.star.arm_safe_entropy.endpoint_geometry import EndpointGeometry
from scripts.star.arm_safe_entropy.geometry import strict_normals


M1_MODES = ('iso_high', 'iso_low', 'normal_shrink', 'directional_full')
M2_MODES = ('local', 'endpoint_direct', 'endpoint_pullback', 'random')


def reference_spectrum(mode, weight, variance, rho, action_dim):
    if mode not in M1_MODES + M2_MODES or action_dim < 2 or variance <= 0 or not 0 < rho <= 1:
        raise ValueError('Known mechanism mode, d>=2 and positive covariance required')
    if mode == 'iso_high':
        return torch.full_like(weight, variance), torch.full_like(weight, variance)
    if mode == 'iso_low':
        return torch.full_like(weight, rho*variance), torch.full_like(weight, rho*variance)
    vn = variance*(1-weight.clamp(0,1)*(1-rho))
    vt = (torch.full_like(vn, variance) if mode == 'normal_shrink'
          else (action_dim*variance-vn)/(action_dim-1))
    return vn, vt


def covariance_matching_description(variance=.04, rho=.25, action_dim=3):
    """Analytical reference spectra at valid fully enabled geometry sites.

    Invalid learned-gradient sites fall back to isotropy, except Iso-Low's
    globally reduced variance. Empirical joint-valid spectra are separately
    counted; this declaration is not an assertion of matched terminal trace.
    """
    result = {}
    for mode in M1_MODES:
        vn, vt = reference_spectrum(mode, torch.tensor(1., dtype=torch.float64), variance, rho, action_dim)
        result[mode] = dict(normal_variance=float(vn), tangent_variance=float(vt),
            trace=float(vn+(action_dim-1)*vt), energy_target=action_dim*.25/variance,
            interpretation='frozen_sampler_intervention_not_a_retrained_temperature_or_policy')
    result['matching'] = dict(iso_high_vs_directional_full='same total injection covariance trace',
        iso_low_vs_normal_shrink_vs_directional_full='same aligned injection normal variance at valid q=1 sites',
        normal_shrink_vs_directional_full='same aligned normal variance;different tangent variance and total trace',
        no_joint_claim='No claim that total trace, normal noise, tangent noise, endpoint covariance and energy are all simultaneously matched')
    return result


class FixedGeometry(EndpointGeometry):
    def __init__(self, margin, mode, random_axis=None, fixed_q=1.):
        if not 0 < fixed_q <= 1:
            raise ValueError('Fixed q must be in (0,1]')
        self.output_mode = mode
        self.random_axis, self.fixed_q = random_axis, fixed_q
        super().__init__(margin, 'endpoint_pullback' if mode == 'random' else mode, transitions=1000000)

    def query(self, actor, obs, midpoint, proposal, step, return_diagnostics=False):
        normal, _, diagnostic = super().query(actor, obs, midpoint, proposal, step, True)
        if self.output_mode == 'random':
            if self.random_axis is None or self.random_axis.shape != proposal.shape:
                raise ValueError('Independent per-action random axes must be supplied before sampling')
            normal = torch.where(diagnostic['valid'][...,None], self.random_axis, torch.zeros_like(normal))
        weight = self.fixed_q*diagnostic['valid'].to(proposal.dtype)
        return (normal, weight, diagnostic) if return_diagnostics else (normal, weight)


def common_noise(actor, obs, draws=64, seed=8112026):
    if draws < 2:
        raise ValueError('At least two within-state draws are required')
    device = obs.device
    generator = torch.Generator(device=device).manual_seed(seed)
    # Existing arm actor has an explicit three-dimensional head.  New
    # arbitrary-action actors should expose action_dim rather than this legacy
    # layout being inferred from an observation's width.
    action_dim = getattr(actor, 'action_dim', actor.net[-1].out_features)
    batch = len(obs)*draws
    noise = torch.randn((actor.num_steps, batch, action_dim), device=device, dtype=obs.dtype, generator=generator)
    axes = torch.randn((batch, action_dim), device=device, dtype=obs.dtype,
                       generator=torch.Generator(device=device).manual_seed(seed+1))
    axes, valid = strict_normals(axes)
    if not valid.all():
        raise RuntimeError('Degenerate independently generated random axis')
    return noise, axes


def frozen_sample(actor, observations, margin, mode, noise, random_axes, *, variance=None, rho=None, fixed_q=1.):
    """One batched Actor call with externally fixed noise and no parameter edits.

    `observations` is already repeated state-major.  K is actor.num_steps;
    callers testing K sensitivity must pass a frozen copied actor and record
    that this changes the discretization, not pretend it is a retrained model.
    """
    if mode not in M1_MODES+M2_MODES:
        raise ValueError('Unknown frozen sampler intervention')
    variance = actor.variance if variance is None else variance
    rho = actor.rho if rho is None else rho
    if len(noise) != actor.num_steps or len(observations) != noise.shape[1]:
        raise ValueError('Precomputed common noise must match K and the repeated state batch')
    action_dim = noise.shape[-1]
    action = torch.zeros_like(noise[0]); energy = torch.zeros_like(action[...,0])
    clipping = torch.zeros_like(energy)
    geometry = FixedGeometry(margin, mode if mode in M2_MODES else 'endpoint_pullback', random_axes, fixed_q)
    references = {key:[] for key in ('vn','vt','weights','normals','valid','proposal_clipped')}
    dt = 1/actor.num_steps
    # Keep geometry input gradients even under a no-grad sampling caller.
    # EndpointGeometry locally re-enables autograd while freezing parameters.
    with torch.no_grad():
        for index in range(actor.num_steps):
            t = torch.full_like(action[...,:1], index*dt)
            midpoint = (action+.5*dt*actor.velocity(observations,action,t)).clamp(-1,1)
            velocity = actor.velocity(observations,midpoint,t+.5*dt)
            injection = action+dt*velocity
            normal, weight, diagnostic = geometry.query(actor,observations,midpoint,injection,index,True)
            vn,vt = reference_spectrum(mode,weight,variance,rho,action_dim)
            vnormal = (velocity*normal).sum(-1)
            tangent_square = (velocity.square().sum(-1)-vnormal.square()).clamp_min(0)
            energy += .5*dt*(vnormal.square()/vn+tangent_square/vt)
            increment = vt.sqrt()[...,None]*noise[index]+(vn.sqrt()-vt.sqrt())[...,None]*(noise[index]*normal).sum(-1)[...,None]*normal
            proposal = injection+math.sqrt(dt)*increment
            clipping += (proposal.abs()>1).to(action.dtype).mean(-1)/actor.num_steps
            action = proposal.clamp(-1,1)
            for key,value in dict(vn=vn,vt=vt,weights=weight,normals=normal,
                valid=diagnostic['valid'],proposal_clipped=diagnostic['proposal_clipped']).items():
                references[key].append(value)
    return action, energy, clipping, {key:torch.stack(value,dim=1) for key,value in references.items()}

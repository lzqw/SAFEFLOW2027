"""Opt-in v5: shared normal-shrink covariance and detached-scale outward loss.

Historical actors/checkpoints retain their exact reference. New actors keep
tangential variance .04, rather than redistributing removed normal variance.
No environment/oracle calls, new critics, or action-candidate selection.
"""
from copy import deepcopy
import math

import torch

from scripts.paper_crax_payload.component_models import ComponentLearner
from scripts.star.arm_safe_entropy.endpoint_geometry import StableFlowActor
from scripts.star.arm_safe_entropy.geometry import reference_variances


def outward_loss(velocities, normals, weights, dt):
    outward = (velocities * normals).sum(-1).clamp_min(0.)
    numerator = (weights * outward.square()).sum(-1).mean() * dt
    scale = (velocities.square().sum(-1).sum(-1).mean() * dt).detach()
    return numerator / (scale + 1e-6)


class NormalShrinkActor(StableFlowActor):
    def forward(self, obs, deterministic=False, geometry=None, return_reference=False,
                noise_increments=None):
        action = torch.zeros((*obs.shape[:-1], 3), device=obs.device, dtype=obs.dtype)
        energy = torch.zeros_like(action[..., 0]); clips = torch.zeros_like(energy)
        dt = 1. / self.num_steps
        trace = {k: [] for k in ('normals', 'weights', 'vn', 'vt', 'velocities',
            'geometry_zero_gradient', 'geometry_proposal_clipped')}
        for index in range(self.num_steps):
            t = torch.full_like(action[..., :1], index * dt)
            midpoint = (action + .5 * dt * self.velocity(obs, action, t)).clamp(-1., 1.)
            velocity = self.velocity(obs, midpoint, t + .5 * dt)
            injection = action + dt * velocity
            if geometry is None:
                normal, weight = torch.zeros_like(action), torch.zeros_like(energy)
                invalid = torch.zeros_like(weight, dtype=torch.bool)
            else:
                normal, weight, diagnostic = geometry.query(self, obs, midpoint, injection, index, True)
                invalid = diagnostic['zero_gradient']
            vn, _ = reference_variances(weight, self.variance, self.rho)
            vt = torch.full_like(vn, self.variance)
            vnormal = (velocity * normal).sum(-1)
            tangent2 = (velocity.square().sum(-1) - vnormal.square()).clamp_min(0.)
            energy = energy + .5 * dt * (vnormal.square() / vn + tangent2 / vt)
            noise = (torch.zeros_like(action) if deterministic else
                     noise_increments[index] if noise_increments is not None else torch.randn_like(action))
            increment = vt.sqrt()[..., None] * noise + ((vn.sqrt() - vt.sqrt()) *
                (noise * normal).sum(-1))[..., None] * normal
            proposal = injection + math.sqrt(dt) * increment
            clips = clips + (proposal.abs() > 1.).to(obs.dtype).mean(-1) / self.num_steps
            action = proposal.clamp(-1., 1.)
            if return_reference:
                values = dict(normals=normal, weights=weight, vn=vn, vt=vt, velocities=velocity,
                    geometry_zero_gradient=invalid,
                    geometry_proposal_clipped=(injection.detach().abs() > 1.).any(-1))
                for key, value in values.items(): trace[key].append(value)
        if return_reference:
            return action, energy, clips, {key: torch.stack(value, -2 if key in
                ('normals', 'velocities') else -1) for key, value in trace.items()}
        return action, energy, clips


class DirectionDevelopmentLearner(ComponentLearner):
    def __init__(self, *, normal_shrink=False, direction_weight=0., **kwargs):
        if direction_weight < 0 or (direction_weight and not normal_shrink):
            raise ValueError('Outward development is registered only with normal shrink')
        super().__init__(**kwargs)
        self.development_settings = dict(normal_shrink=normal_shrink, direction_weight=direction_weight,
            loss='weighted_outward_squared_over_detached_unweighted_speed_squared_eps1e-6')
        if normal_shrink:
            if self.component != 'full': raise ValueError('Normal shrink needs shared sampling and energy')
            for actor in (self.actor, self.target_actor): actor.__class__ = NormalShrinkActor
        if normal_shrink:
            for key in ('direction_loss', 'high_weight_reference_vn_over_vt',
                        'high_weight_outward_speed', 'high_weight_geometry_weight', 'reference_trace'):
                self._last_metrics[key] = torch.zeros((), device=self.device)

    def actor_regularizer(self, trace):
        if not self.development_settings['normal_shrink']: return 0.
        loss = outward_loss(trace['velocities'], trace['normals'], trace['weights'], 1./self.actor.num_steps)
        with torch.no_grad():
            # Exactly top20% by the preexisting detached reliability/risk weight;
            # never filter by eventual outcome. This is model-risk, not true risk.
            indices = trace['weights'].mean(-1).topk(max(1, math.ceil(len(trace['weights'])*.2))).indices
            outward = (trace['velocities']*trace['normals']).sum(-1).clamp_min(0.)
            self._last_metrics.update(direction_loss=loss.detach(),
                high_weight_reference_vn_over_vt=(trace['vn'][indices]/trace['vt'][indices]).mean(),
                high_weight_outward_speed=outward[indices].mean(),
                high_weight_geometry_weight=trace['weights'][indices].mean(),
                reference_trace=(trace['vn']+2*trace['vt']).mean())
        return self.development_settings['direction_weight'] * loss

    def state_dict(self):
        return super().state_dict() | {'direction_development_v5': deepcopy(self.development_settings)}

    def load_state_dict(self, state):
        if state.get('direction_development_v5') != self.development_settings:
            raise ValueError('Never reinterpret a covariance or direction-loss checkpoint')
        super().load_state_dict(state)


def build_development(config, arm, seed, device):
    if seed not in config['development_training_seeds']:
        raise ValueError('Use a predeclared development training seed')
    shared = config['shared_learner']
    return DirectionDevelopmentLearner(component=arm['component'], rho=arm['rho'], device=device,
        normal_shrink=arm['normal_shrink'], direction_weight=arm['direction_weight'],
        actor_lr=shared['actor_lr'], critic_lr=shared['critic_lr'],
        critic_only_updates=shared['critic_only_updates'],
        actor_interval=shared['actor_update_interval'], margin_ema_tau=config['geometry']['ema_tau'])

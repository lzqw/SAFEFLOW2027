"""Opt-in direction-loss-only taper. Sampling/energy/critics remain original P5."""
from copy import deepcopy

import torch

from scripts.paper_crax_payload.direction_development import NormalShrinkActor, outward_loss
from scripts.paper_crax_payload.train_matched_ablation_v12 import AblationLearner


SPEC = dict(support_start_m=-.01, full_strength_m=0.,
            shape='cubic_smoothstep', scope='direction_loss_only', detached=True)


def direction_taper(margin):
    x = ((margin.detach()-SPEC['support_start_m']) /
         (SPEC['full_strength_m']-SPEC['support_start_m'])).clamp(0., 1.)
    return x.square()*(3.-2.*x)


class RecordedGeometry:
    def __init__(self, geometry):
        self.geometry = geometry
        self.margins = []

    def query(self, actor, obs, midpoint, proposal, step, return_diagnostics=False):
        normal, weight, diagnostics = self.geometry.query(actor, obs, midpoint, proposal, step, True)
        self.margins.append(diagnostics['predicted_margin_m'])
        return (normal, weight, diagnostics) if return_diagnostics else (normal, weight)


class DirectionTaperActor(NormalShrinkActor):
    def forward(self, obs, deterministic=False, geometry=None, return_reference=False, noise_increments=None):
        if not return_reference or geometry is None:
            return super().forward(obs, deterministic, geometry, return_reference, noise_increments)
        recorded = RecordedGeometry(geometry)
        action, energy, clips, trace = super().forward(obs, deterministic, recorded, True, noise_increments)
        margins = torch.stack(recorded.margins, -1)
        trace.update(direction_margin_m=margins,
                     direction_weights=trace['weights']*direction_taper(margins))
        return action, energy, clips, trace


class DirectionTaperLearner(AblationLearner):
    def __init__(self, specification, **kwargs):
        super().__init__(specification, **kwargs)
        if not specification['normal_shrink'] or specification['direction_weight'] != 1:
            raise ValueError('Taper is a single factor on fixed full P5 only')
        for actor in (self.actor, self.target_actor):
            actor.__class__ = DirectionTaperActor

    def actor_regularizer(self, trace):
        original = super().actor_regularizer(trace)
        loss = outward_loss(trace['velocities'], trace['normals'], trace['direction_weights'], 1./self.actor.num_steps)
        self._last_metrics.update(direction_loss=loss.detach(),
            untapered_direction_loss=original.detach(),
            direction_taper_supported_fraction=(trace['direction_margin_m']>-.01).float().mean(),
            mean_direction_loss_weight=trace['direction_weights'].mean())
        return loss

    def state_dict(self):
        return super().state_dict() | dict(direction_taper_v12=deepcopy(SPEC))

    def load_state_dict(self, state):
        if state.get('direction_taper_v12') != SPEC:
            raise ValueError('Never reinterpret an untapered or differently localized checkpoint')
        super().load_state_dict(state)

"""Opt-in existing-EMA final-action margin residual; full original P5 retained."""
from copy import deepcopy

from scripts.paper_crax_payload.direction_development import NormalShrinkActor
from scripts.paper_crax_payload.train_matched_ablation_v12 import AblationLearner
from scripts.star.arm_safe_entropy.endpoint_geometry import geometry_ramp


SPEC = dict(coefficient=1., scale_m=.015, shape='positive_part_linear',
    model='existing_two_head_geometry_EMA_mean', site='final_generated_action',
    multiplier='original_geometry_ramp_10K_30K', original_direction_loss='unchanged')


def endpoint_residual(margin):
    return margin.clamp_min(0.).mean()/SPEC['scale_m']


class EndpointRecordedActor(NormalShrinkActor):
    def forward(self, obs, deterministic=False, geometry=None, return_reference=False,
                noise_increments=None):
        result = super().forward(obs,deterministic,geometry,return_reference,noise_increments)
        if return_reference:
            result[3].update(endpoint_observation=obs.detach(), endpoint_action=result[0])
        return result


class EndpointResidualLearner(AblationLearner):
    def __init__(self,specification,**kwargs):
        if (specification['component']!='full' or not specification['normal_shrink']
                or specification['direction_weight']!=1
                or not isinstance(specification['endpoint_residual'],bool)):
            raise ValueError('Only original full P5 with the registered optional residual')
        super().__init__(specification,**kwargs)
        for actor in (self.actor,self.target_actor):
            actor.__class__ = EndpointRecordedActor

    def actor_regularizer(self,trace):
        original = super().actor_regularizer(trace)
        if not self.ablation['endpoint_residual']:
            return original
        # Geometry parameters are natively frozen; gradients flow through action only.
        # This terminal query does not change sampling, the reference D or its energy.
        margin = self.margin_geometry(trace['endpoint_observation'],trace['endpoint_action']).mean(0)
        residual = endpoint_residual(margin)
        beta = geometry_ramp(self.transitions)
        self._last_metrics.update(endpoint_residual_loss=residual.detach(),
            endpoint_residual_effective=(beta*residual).detach(),
            endpoint_predicted_unsafe_fraction=(margin>0.).float().mean().detach(),
            endpoint_positive_margin_m=margin.clamp_min(0.).mean().detach())
        return original+SPEC['coefficient']*beta*residual

    def state_dict(self):
        return super().state_dict() | dict(endpoint_residual_v12=deepcopy(SPEC))

    def load_state_dict(self,state):
        if state.get('endpoint_residual_v12')!=SPEC:
            raise ValueError('Do not reinterpret original, tapered or other residual checkpoints')
        super().load_state_dict(state)

"""Independent sampling/energy switches; shared native stable learner updates."""
from copy import deepcopy

from scripts.star.arm_safe_entropy.endpoint_geometry import StableFlowActor
from scripts.star.arm_safe_entropy.stability_models import StableArmLearner


SWITCHES = {'iso': (False, False), 'noise_only': (True, False),
            'metric_only': (False, True), 'full': (True, True)}


class ComponentActor(StableFlowActor):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs, sampling_directional=self.sampling_directional,
                               energy_directional=self.energy_directional)


class ComponentLearner(StableArmLearner):
    def __init__(self, component='full', rho=.25, device='cpu', **kwargs):
        if component not in SWITCHES:
            raise ValueError('Use a registered component arm')
        super().__init__(variant='s_ea', rho=rho, device=device, **kwargs)
        self.component = component
        # Reuse exact initial weights, generator and optimizer references, no RNG draws.
        for actor in (self.actor, self.target_actor):
            actor.__class__ = ComponentActor
            actor.sampling_directional, actor.energy_directional = SWITCHES[component]
        self.component_settings = dict(component=component, rho=rho,
            sampling_directional=SWITCHES[component][0], energy_directional=SWITCHES[component][1])

    def state_dict(self):
        return super().state_dict() | {'component_study': deepcopy(self.component_settings)}

    def load_state_dict(self, state):
        if state.get('component_study') != self.component_settings:
            raise ValueError('Do not reinterpret the component or rho of a native checkpoint')
        super().load_state_dict(state)


def build_component(config, component, rho, device):
    shared = config['shared_learner']
    return ComponentLearner(component, rho, device,
        actor_lr=shared['actor_lr'], critic_lr=shared['critic_lr'],
        critic_only_updates=shared['critic_only_updates'],
        actor_interval=shared['actor_update_interval'], margin_ema_tau=config['geometry']['ema_tau'])

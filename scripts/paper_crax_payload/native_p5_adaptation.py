"""Explicit native34D/45D full-state forks; historical loaders stay strict.

P5 changes the native EA sampler and adds the frozen outward loss. Both arms
share a learning-rate schedule in NEW adaptation steps and target deployment.
No cross-robot weight import, new critic, test-set selection or hardware API.
"""
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import math

import torch

from scripts.paper_crax_payload.direction_development import NormalShrinkActor, outward_loss
from scripts.paper_crax_payload.piper_train import PiperFlowLearner
from scripts.paper_crax_payload.payload_models import PayloadFlowLearner, build_payload_learner


FORMAT = 'native_task_p5_paired_adaptation_v11_full'


def actor_digest(actor):
    digest = hashlib.sha256()
    for name, tensor in sorted(actor.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def target_deployment(learner):
    """Same existing-target deployment rule as v10, without importing its CLI."""
    original = learner.actor
    if (learner.target_actor.num_steps != original.num_steps
            or learner.target_actor.rho != original.rho
            or learner.target_actor.variance != original.variance):
        raise ValueError('Deployment must retain the native sampler settings')
    learner.actor = learner.target_actor
    try:
        yield
    finally:
        learner.actor = original


def scheduled_lr(step, schedule):
    fraction = min(1., max(0., (step-schedule['start'])/(schedule['end']-schedule['start'])))
    return schedule['floor'] + (schedule['initial']-schedule['floor'])*.5*(1.+math.cos(math.pi*fraction))


class AdaptationMixin:
    def set_transitions(self, transitions):
        super().set_transitions(transitions)
        if hasattr(self, 'adaptation'):
            step = self.transitions-self.adaptation['parent_steps']
            if not 0 <= step <= self.adaptation['additional_steps']:
                raise ValueError('Adaptation counter must remain within the registered new budget')
            for group in self.actor_opt.param_groups:
                group['lr'] = scheduled_lr(step, self.adaptation['actor_lr'])

    def actor_regularizer(self, trace):
        if self.adaptation['arm'] == 'control':
            return 0.
        loss = outward_loss(trace['velocities'], trace['normals'], trace['weights'], 1./self.actor.num_steps)
        with torch.no_grad():
            indices = trace['weights'].mean(-1).topk(max(1, math.ceil(len(trace['weights'])*.2))).indices
            outward = (trace['velocities']*trace['normals']).sum(-1).clamp_min(0.)
            self._last_metrics.update(direction_loss=loss.detach(),
                high_weight_reference_vn_over_vt=(trace['vn'][indices]/trace['vt'][indices]).mean(),
                high_weight_outward_speed=outward[indices].mean(),
                high_weight_geometry_weight=trace['weights'][indices].mean())
        return self.adaptation['p5']['direction_weight']*loss

    def metrics(self):
        return super().metrics() | dict(adaptation_steps=self.transitions-self.adaptation['parent_steps'],
            actor_lr_effective=self.actor_opt.param_groups[0]['lr'])

    def state_dict(self):
        return super().state_dict() | {'native_p5_adaptation_v11': deepcopy(self.adaptation)}

    def load_state_dict(self, state):
        if state.get('native_p5_adaptation_v11') != self.adaptation:
            raise ValueError('Resume only the identical task, parent, arm and new-step schedule')
        super().load_state_dict(state)


class PiperAdaptationLearner(AdaptationMixin, PiperFlowLearner):
    pass


class PandaAdaptationLearner(AdaptationMixin, PayloadFlowLearner):
    pass


def build_native(config, line, device):
    if line['robot'] == 'Piper':
        return PiperFlowLearner(config, 'EA', device)
    return build_payload_learner(config, 's_ea', device)


def attach_adaptation(learner, plan, line_name, arm):
    line = plan['lines'][line_name]
    if arm not in plan['arms'] or learner.transitions != line['parent_steps']:
        raise ValueError('Attach only after strict native full-parent import')
    learner.__class__ = PiperAdaptationLearner if line['robot']=='Piper' else PandaAdaptationLearner
    learner.adaptation = dict(line=line_name, arm=arm, parent_steps=line['parent_steps'],
        parent_sha256=line['parent_sha256'], additional_steps=plan['additional_steps_per_run'],
        actor_lr=deepcopy(plan['actor_lr']), p5=deepcopy(plan['p5']))
    if arm == 'p5':
        if (learner.actor.rho != .25 or learner.actor.variance != .04
                or plan['p5'] != dict(direction_weight=1., rho=.25, tangential_variance=.04)):
            raise ValueError('Retain the frozen P5 covariance and direction strength')
        for actor in (learner.actor, learner.target_actor):
            actor.__class__ = NormalShrinkActor
        for name in ('direction_loss', 'high_weight_reference_vn_over_vt',
                     'high_weight_outward_speed', 'high_weight_geometry_weight'):
            learner._last_metrics[name] = torch.zeros((), device=learner.device)
    learner.set_transitions(learner.transitions)
    return learner


def validate_parent(saved, plan, line_name, digest):
    line = plan['lines'][line_name]
    if (digest != line['parent_sha256'] or saved['format'] != line['parent_format']
            or saved['steps'] != line['parent_steps'] or saved['training_seed'] != line['seed']
            or saved['method'] != ('EA' if line['robot']=='Piper' else 's_ea')
            or saved.get('engineering') or saved['learner']['ea_stability_v2']['transitions'] != saved['steps']):
        raise ValueError('Require the exact registered native full checkpoint')
    dim = 34 if line['robot']=='Piper' else 45
    if (saved['replay']['data']['obs'].shape[-1] != dim
            or len(saved['environments']) != saved['configuration']['num_envs']):
        raise ValueError('Native task dimensions or complete environment count differ')
    if line['robot']=='Piper' and (saved.get('robot_family') != 'Piper' or saved['task'] != line['task']):
        raise ValueError('Never rename weights across robots or native Piper tasks')


def evaluation_config(native, plan, line):
    config = deepcopy(native)
    config['evaluation'].update(conditions=deepcopy(line['conditions']),
        development_layout_seed_start=line['development_layout_seed_start'],
        action_sampling_seed=line['action_sampling_seed'])
    return config

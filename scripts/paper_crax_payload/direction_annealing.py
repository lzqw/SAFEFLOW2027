"""v6 explicit fork method; no weakening of historical native resume guards."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.paper_crax_payload.direction_development import DirectionDevelopmentLearner, outward_loss
from scripts.star.arm_safe_entropy.endpoint_geometry import frozen_parameters
from scripts.star.arm_safe_entropy.stability_runtime import isolated_rng


FORMAT = 'direction_annealing_v6_full'
PARENT_SOURCE = '721bf6bbdba5943c800498557e9b0203f73cc3bf'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def validate_config(config):
    # This finite campaign is not a generic arbitrary tuning launcher.
    registered = json.loads(Path(__file__).resolve().parents[2].joinpath(
        'configs/paper_crax_payload/direction_annealing_v6.json').read_text())
    if config != registered:
        raise ValueError('Use the entire registered v6 finite fork protocol without implicit retuning')
    if (config['new_training_budget'] != 450000 or config['absolute_target_steps'] != 250000
            or config['budget_reallocation']['remaining_confirmation'] != 2250000):
        raise ValueError('Preserve the reallocated3.45M total ceiling')


def schedule_values(name, absolute_steps):
    if name not in ('fixed', 'loss_only', 'loss_and_rho'):
        raise ValueError('Unknown frozen annealing branch')
    x = min(1., max(0., (absolute_steps-100000)/50000))
    f = .5*(1.+math.cos(math.pi*x))
    return (1. if name == 'fixed' else .1+.9*f,
            .5-.25*f if name == 'loss_and_rho' else .25)


class AnnealingLearner(DirectionDevelopmentLearner):
    def __init__(self, schedule, **kwargs):
        super().__init__(component='full', rho=.25, normal_shrink=True, direction_weight=1., **kwargs)
        schedule_values(schedule, 100000)
        self.annealing_schedule = schedule
        self.current_direction_weight = 1.

    def set_transitions(self, transitions):
        super().set_transitions(transitions)
        self.current_direction_weight, rho = schedule_values(self.annealing_schedule, transitions)
        self.actor.rho = self.target_actor.rho = rho

    def actor_regularizer(self, trace):
        return self.current_direction_weight * super().actor_regularizer(trace)

    def metrics(self):
        return super().metrics() | dict(direction_weight_effective=self.current_direction_weight,
            rho_effective=self.actor.rho, target_rho_effective=self.target_actor.rho)

    def state_dict(self):
        return super().state_dict() | dict(direction_annealing_v6=dict(schedule=self.annealing_schedule,
            absolute_steps=self.transitions, direction_weight=self.current_direction_weight,
            rho=self.actor.rho))

    def load_parent_state(self, state):
        if 'direction_annealing_v6' in state or self.transitions != 0:
            raise ValueError('Explicit parent import only once from the original v5 state')
        if state['ea_stability_v2']['transitions'] != 100000:
            raise ValueError('Import only the observed original100K full parent')
        # All original component, learner, covariance, optimizer guards apply.
        super().load_state_dict(state)
        self.set_transitions(100000)

    def load_state_dict(self, state):
        meta = state.get('direction_annealing_v6')
        if not meta or meta['schedule'] != self.annealing_schedule or self.transitions != 0:
            raise ValueError('Resume only the same v6 schedule into a fresh learner')
        step = meta['absolute_steps']; weight, rho = schedule_values(self.annealing_schedule, step)
        if (step != state['ea_stability_v2']['transitions']
                or (weight, rho) != (meta['direction_weight'], meta['rho'])):
            raise ValueError('Scheduled covariance/weight identity mismatch')
        self.actor.rho = self.target_actor.rho = rho
        super().load_state_dict(state)


def build(config, arm, device):
    s = config['shared_learner']
    return AnnealingLearner(arm['schedule'], device=device, actor_lr=s['actor_lr'], critic_lr=s['critic_lr'],
        critic_only_updates=s['critic_only_updates'], actor_interval=s['actor_update_interval'],
        margin_ema_tau=config['geometry']['ema_tau'])


def validate_parent(checkpoint, manifest, expected_sha, checkpoint_path):
    from scripts.paper_crax_payload.train_direction_development import validate_config as validate_v5
    validate_v5(manifest['configuration'])
    if (Path(checkpoint_path).name != 'full_100000.pt' or sha256(checkpoint_path) != expected_sha
            or len(expected_sha) != 64 or manifest['engineering_only']
            or manifest['source_commit'] != PARENT_SOURCE
            or manifest['arm'] != dict(name='normal-shrink-dir', component='full', rho=.25,
                                      normal_shrink=True, direction_weight=1.)
            or manifest['training_seed'] != 2028
            or checkpoint['format'] != 'direction_development_v5_full'
            or checkpoint['source_commit'] != PARENT_SOURCE or checkpoint['steps'] != 100000
            or checkpoint['seed'] != 2028 or checkpoint['configuration'] != manifest['configuration']
            or checkpoint['arm'] != manifest['arm']
            or checkpoint['learner']['ea_stability_v2']['transitions'] != 100000
            or checkpoint['replay']['count'] != 100000 or checkpoint['replay']['capacity'] != 150000
            or len(checkpoint['environments']) != 2 or len(checkpoint['observations']) != 2):
        raise ValueError('Require the exact original Normal-shrink-dir2028 full100K parent and SHA')
    return dict(steps=checkpoint['steps'], episodes=checkpoint['episodes'],
        training_violations=checkpoint['violations'], safe_successes=checkpoint['successes'],
        safe_timeouts=checkpoint['timeouts'], learner_updates=checkpoint['learner']['updates'],
        actor_updates=checkpoint['learner']['actor_updates'], reward_updates=checkpoint['learner']['reward_updates'])


def fixed_probe(replay_state, config):
    d = config['gradient_diagnosis']
    indices = np.random.default_rng(d['parent_replay_index_seed']).integers(
        replay_state['count'], size=d['batch_size'])
    observations = torch.tensor(replay_state['data']['obs'][indices].copy())
    generator = torch.Generator().manual_seed(d['noise_seed'])
    noise = torch.randn(4, len(indices), 3, generator=generator)
    digest = hashlib.sha256(observations.numpy().tobytes()+noise.numpy().tobytes()).hexdigest()
    return dict(observations=observations, noise=noise, indices=indices.tolist(), sha256=digest)


def gradient_diagnosis(learner, probe):
    # Autograd only: no parameter/optimizer/EMA updates, no global RNG drift.
    obs = probe['observations'].to(learner.device); noise = probe['noise'].to(learner.device)
    parameters = list(learner.actor.parameters())
    with isolated_rng(), frozen_parameters(learner.q, learner.cost_q):
        action, energy, clips, trace = learner.actor(obs, geometry=learner.geometry,
            noise_increments=noise, return_reference=True)
        outward = (trace['velocities']*trace['normals']).sum(-1).clamp_min(0.)
        numerator = (trace['weights']*outward.square()).sum(-1).mean()/learner.actor.num_steps
        denominator = (trace['velocities'].square().sum(-1).sum(-1).mean()/learner.actor.num_steps).detach()
        terms = dict(reward=-learner.q(obs, action).min(0).values.mean(),
            cost=learner.safety_multiplier*learner.cost_q(obs, action).mean(),
            energy=learner.log_alpha.detach().exp()*energy.mean(),
            direction=learner.current_direction_weight*outward_loss(trace['velocities'], trace['normals'],
                trace['weights'], 1./learner.actor.num_steps))
        vectors = {}
        for name, value in terms.items():
            gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
            vectors[name] = torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten()
                                       for p, g in zip(parameters, gradients)]).detach()
        norms = {k: float(v.norm().cpu()) for k, v in vectors.items()}
        cosines = {a+'__'+b: float(torch.dot(vectors[a], vectors[b]).cpu()/(norms[a]*norms[b]))
                   if norms[a]*norms[b] > 0 else None
                   for i, a in enumerate(vectors) for b in list(vectors)[i+1:]}
        result = dict(absolute_steps=learner.transitions, probe_sha256=probe['sha256'],
            norms=norms, cosines=cosines, loss_values={k: float(v.detach().cpu()) for k,v in terms.items()},
            direction_numerator=float(numerator.detach().cpu()), direction_denominator=float(denominator.cpu()),
            alpha=float(learner.log_alpha.detach().exp().cpu()), energy=float(energy.detach().mean().cpu()),
            clipping_rate=float(clips.detach().mean().cpu()), safety_multiplier=learner.safety_multiplier,
            direction_weight=learner.current_direction_weight, rho=learner.actor.rho,
            new_training_transitions=0, new_simulation_controls=0, optimizer_updates=0,
            scope='fixed_parent_training_replay_not_independent_or_causal')
    json.dumps(result, allow_nan=False)
    return result

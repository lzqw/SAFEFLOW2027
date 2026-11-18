"""v7 one-factor schedule candidates with explicit full-prefix import.

Historical v5/v6 loaders are unchanged. Import is a separately validated fork,
never a relaxation of a same-configuration resume guard.
"""
from copy import deepcopy
import math

from scripts.paper_crax_payload.direction_development import DirectionDevelopmentLearner
from scripts.paper_crax_payload.direction_annealing import sha256

FORMAT = 'direction_optimization_v7_full'


def cosine_value(step, schedule, initial):
    x = min(1., max(0., (step-schedule['start'])/(schedule['end']-schedule['start'])))
    return schedule.get('floor', schedule.get('final')) + (
        initial-schedule.get('floor', schedule.get('final'))) * .5*(1.+math.cos(math.pi*x))


def registered_development(config):
    from pathlib import Path
    import json
    root = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload'
    for name in ('direction_optimization_v7.json', 'direction_optimization_v7_round2.json',
                 'direction_optimization_v7_round3.json'):
        path = root/name
        if path.exists() and json.loads(path.read_text()) == config:
            return path
    raise ValueError('Use the complete registered round; later candidates require a new preregistration')


def validate_config(config):
    registered_development(config)
    budget = config['budget']
    if (budget['development_ceiling'] != 750000 or budget['confirmation_reserved'] != 1500000
            or sum(budget[k] for k in ('historical_v5','historical_v6','development_ceiling','confirmation_reserved')) != 3450000
            or budget['confirmation_seeds'] != [2031,2032]):
        raise ValueError('Preserve the new750K/1500K allocation and exactly two fresh seeds')


class OptimizationLearner(DirectionDevelopmentLearner):
    def __init__(self, candidate, **kwargs):
        super().__init__(component='full', rho=.25, normal_shrink=True, direction_weight=1., **kwargs)
        self.candidate = deepcopy(candidate)
        self.current_direction_weight = 1.

    def set_transitions(self, transitions):
        super().set_transitions(transitions)
        self.current_direction_weight = cosine_value(transitions, self.candidate['direction'], 1.)
        self.current_actor_lr = cosine_value(transitions, self.candidate['actor_lr'], .0001)
        for group in self.actor_opt.param_groups:
            group['lr'] = self.current_actor_lr

    def actor_regularizer(self, trace):
        return self.current_direction_weight * super().actor_regularizer(trace)

    def metrics(self):
        return super().metrics() | dict(direction_weight_effective=self.current_direction_weight,
            actor_lr_effective=self.actor_opt.param_groups[0]['lr'], rho_effective=self.actor.rho,
            target_rho_effective=self.target_actor.rho)

    def state_dict(self):
        return super().state_dict() | dict(direction_optimization_v7=dict(
            candidate=deepcopy(self.candidate), absolute_steps=self.transitions,
            direction_weight=self.current_direction_weight, actor_lr=self.actor_opt.param_groups[0]['lr']))

    def import_parent(self, state):
        if self.transitions or 'direction_optimization_v7' in state:
            raise ValueError('Explicit historical prefix import only into a fresh learner')
        # v6 metadata is verified by validate_parent, not reinterpreted as own resume.
        super().load_state_dict(state)
        self.set_transitions(self.transitions)

    def load_state_dict(self, state):
        meta = state.get('direction_optimization_v7')
        if (not meta or self.transitions or meta['candidate'] != self.candidate
                or meta['absolute_steps'] != state['ea_stability_v2']['transitions']
                or meta['direction_weight'] != cosine_value(meta['absolute_steps'],self.candidate['direction'],1.)
                or meta['actor_lr'] != cosine_value(meta['absolute_steps'],self.candidate['actor_lr'],.0001)):
            raise ValueError('Resume only identical v7 candidate and absolute schedule state')
        super().load_state_dict(state)
        self.set_transitions(self.transitions)


def build(config, candidate, device):
    s = config['shared_learner']
    return OptimizationLearner(candidate, device=device, actor_lr=s['actor_lr'],critic_lr=s['critic_lr'],
        critic_only_updates=s['critic_only_updates'],actor_interval=s['actor_update_interval'],
        margin_ema_tau=config['geometry']['ema_tau'])


def counts(state):
    return dict(steps=state['steps'], episodes=state['episodes'], safe_successes=state['successes'],
        training_violations=state['violations'], safe_timeouts=state['timeouts'])


def validate_parent(state, manifest, checkpoint, manifest_path, config, candidate):
    spec = config['parents'][candidate['parent']]
    if (sha256(checkpoint) != spec['checkpoint_sha256'] or sha256(manifest_path) != spec['manifest_sha256']
            or state['format'] != spec['format'] or state['source_commit'] != spec['source_commit']
            or manifest['source_commit'] != spec['source_commit'] or state['steps'] != spec['steps']
            or state['seed'] != 2028 or state['arm']['name'] != spec['arm']
            or manifest['arm'] != state['arm'] or manifest['configuration'] != state['configuration']
            or manifest['engineering_only']):
        raise ValueError('Exact registered complete prefix and original manifest required')
    for key in ('task_version','shared_learner','geometry'):
        if state['configuration'][key] != config[key]:
            raise ValueError('Do not change common task/learner/geometry in prefix reuse')
    if spec['steps'] == 150000:
        schedule = state['learner']['direction_annealing_v6']['schedule']
        expected = 'fixed' if candidate['parent']=='v6-A150' else 'loss_only'
        if schedule != expected or state['parent']['checkpoint_sha256'] != config['common_parent100K_sha256']:
            raise ValueError('A/B150K must descend from the common original100K')
        # Equality for the ENTIRE reused100K..150K, not only endpoint lambda.
        from scripts.paper_crax_payload.direction_annealing import schedule_values
        for step in range(100000,150001):
            if (cosine_value(step,candidate['direction'],1.) != schedule_values(schedule,step)[0]
                    or cosine_value(step,candidate['actor_lr'],.0001) != .0001):
                raise ValueError('Candidate algorithm differs before imported prefix end')
        common = state['parent']['prefix']
    else:
        common = counts(state)
    if (state['replay']['capacity'] != 150000 or state['learner']['ea_stability_v2']['transitions'] != state['steps']
            or state['learner']['component_study']['rho'] != .25
            or len(state['environments']) != 2 or len(state['observations']) != 2):
        raise ValueError('Require complete native optimizer/replay/RNG/environment prefix')
    return common

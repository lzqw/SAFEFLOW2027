"""Fresh250K mechanism attribution, not promotion of an unqualified candidate."""
from copy import deepcopy
import json
from pathlib import Path

from scripts.paper_crax_payload.direction_development import DirectionDevelopmentLearner
from scripts.paper_crax_payload.direction_warmup_v12 import p5_actor_lr
from scripts.paper_crax_payload.train_safereach_early_v12 import deployed as evaluate_target, SafeReachTracker
from scripts.paper_crax_payload.train_direction_development import main as run

PLAN = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_matched_ablation.json'
FORMAT = 'safereach_fresh250k_matched_ablation_v12_full'


class AblationLearner(DirectionDevelopmentLearner):
    def __init__(self, specification, **kwargs):
        self.ablation = deepcopy(specification)
        super().__init__(component=specification['component'], rho=.25,
            normal_shrink=specification['normal_shrink'], direction_weight=specification['direction_weight'], **kwargs)
        self.current_direction_weight = specification['direction_weight']
        self.set_transitions(0)

    def set_transitions(self, steps):
        super().set_transitions(steps)
        for group in self.actor_opt.param_groups:
            group['lr'] = p5_actor_lr(steps, self.ablation['late_actor_lr_decay'])

    def metrics(self):
        return super().metrics() | dict(direction_weight_effective=self.current_direction_weight,
            actor_lr_effective=self.actor_opt.param_groups[0]['lr'])

    def state_dict(self):
        return super().state_dict() | dict(matched_ablation_v12=deepcopy(self.ablation))

    def load_state_dict(self, state):
        if state.get('matched_ablation_v12') != self.ablation:
            raise ValueError('Only identical matched-ablation factor settings may resume')
        step = state['ea_stability_v2']['transitions']
        if any(g['lr'] != p5_actor_lr(step, self.ablation['late_actor_lr_decay'])
               for g in state['actor_opt']['param_groups']):
            raise ValueError('Saved actor optimizer LR differs from the registered ablation schedule')
        super().load_state_dict(state)


def validate(config):
    if config != json.loads(PLAN.read_text()):
        raise ValueError('Use exact fresh250K matched-ablation protocol')


def build(config, arm, seed, device):
    if seed not in config['development_training_seeds'] or arm not in config['arms']:
        raise ValueError('Use the registered fresh seed and arm')
    shared = config['shared_learner']
    return AblationLearner(arm, device=device, actor_lr=shared['actor_lr'], critic_lr=shared['critic_lr'],
        critic_only_updates=shared['critic_only_updates'], actor_interval=shared['actor_update_interval'],
        margin_ema_tau=config['geometry']['ema_tau'])


def deployed(learner, config, steps):
    result = evaluate_target(learner, config, steps)
    result.update(split='v12_fresh250k_matched_ablation_development_only',
        parent_seen_for_selection=False, algorithm_factors=deepcopy(learner.ablation),
        scientific_purpose='mechanism_attribution_not_winner_confirmation')
    return result


if __name__ == '__main__':
    run(validator=validate, builder=build, format_name=FORMAT, evaluator=deployed,
        tracker_factory=SafeReachTracker)

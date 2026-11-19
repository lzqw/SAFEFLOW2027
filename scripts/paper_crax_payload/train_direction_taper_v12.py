"""Single fixed250K direction-loss-support candidate; original sampling retained."""
import json
from pathlib import Path

from scripts.paper_crax_payload.direction_taper_v12 import DirectionTaperLearner, SPEC
from scripts.paper_crax_payload.train_direction_development import main as run
from scripts.paper_crax_payload.train_matched_ablation_v12 import deployed as evaluate_target
from scripts.paper_crax_payload.train_safereach_early_v12 import SafeReachTracker

PLAN = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_direction_taper.json'
FORMAT = 'safereach_fresh250k_direction_loss_taper_v12_full'


def validate(config):
    if config != json.loads(PLAN.read_text()) or config['direction_taper'] != SPEC:
        raise ValueError('Use only the exact single-factor registered direction-only taper')


def build(config, arm, seed, device):
    if seed not in config['development_training_seeds'] or arm not in config['arms']:
        raise ValueError('Use the registered single candidate/seed')
    shared = config['shared_learner']
    return DirectionTaperLearner(arm, device=device, actor_lr=shared['actor_lr'],
        critic_lr=shared['critic_lr'], critic_only_updates=shared['critic_only_updates'],
        actor_interval=shared['actor_update_interval'], margin_ema_tau=config['geometry']['ema_tau'])


def deployed(learner, config, steps):
    result = evaluate_target(learner, config, steps)
    result.update(split='v12_direction_loss_taper_development_only',
        scientific_purpose=config['scientific_purpose'], direction_taper=config['direction_taper'])
    return result


if __name__ == '__main__':
    run(validator=validate, builder=build, format_name=FORMAT, evaluator=deployed,
        tracker_factory=SafeReachTracker)

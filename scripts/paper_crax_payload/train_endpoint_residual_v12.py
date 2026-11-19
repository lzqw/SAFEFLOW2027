"""Fixed fresh250K matched control/candidate; not a winner or independent test."""
import json
from pathlib import Path

from scripts.paper_crax_payload.endpoint_residual_v12 import EndpointResidualLearner,SPEC
from scripts.paper_crax_payload.train_direction_development import main as run
from scripts.paper_crax_payload.train_matched_ablation_v12 import deployed as evaluate_target
from scripts.paper_crax_payload.train_safereach_early_v12 import SafeReachTracker


PLAN = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_endpoint_residual.json'
FORMAT = 'safereach_fresh250k_endpoint_margin_residual_v12_full'


def validate(config):
    if config!=json.loads(PLAN.read_text()) or config['endpoint_residual']!=SPEC:
        raise ValueError('Use the one registered factor, both arms and the complete fixed budget')


def build(config,arm,seed,device):
    if seed not in config['development_training_seeds'] or arm not in config['arms']:
        raise ValueError('Use registered native seed/control/candidate')
    shared = config['shared_learner']
    return EndpointResidualLearner(arm,device=device,actor_lr=shared['actor_lr'],
        critic_lr=shared['critic_lr'],critic_only_updates=shared['critic_only_updates'],
        actor_interval=shared['actor_update_interval'],margin_ema_tau=config['geometry']['ema_tau'])


def deployed(learner,config,steps):
    result = evaluate_target(learner,config,steps)
    result.update(split='v12_endpoint_margin_residual_development_only',
        scientific_purpose=config['scientific_purpose'],endpoint_residual=config['endpoint_residual'])
    return result


if __name__=='__main__':
    run(validator=validate,builder=build,format_name=FORMAT,evaluator=deployed,
        tracker_factory=SafeReachTracker)

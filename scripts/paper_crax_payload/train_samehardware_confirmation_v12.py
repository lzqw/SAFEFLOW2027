"""Two fixed250K controls; unchanged native learner and existing DEV protocol."""
import json
from pathlib import Path

from scripts.paper_crax_payload.train_direction_development import main as run
from scripts.paper_crax_payload.train_matched_ablation_v12 import build, deployed as ablation_deployed
from scripts.paper_crax_payload.train_safereach_early_v12 import SafeReachTracker

PLAN = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_samehardware_confirmation.json'
FORMAT = 'safereach_fresh250k_samehardware_confirmation_v12_full'


def validate(config):
    if config != json.loads(PLAN.read_text()):
        raise ValueError('Use the exact two-arm samehardware confirmation registration')


def deployed(learner, config, steps):
    result = ablation_deployed(learner, config, steps)
    result.update(split='v12_samehardware_confirmation_development_only',
        scientific_purpose=config['scientific_purpose'])
    return result


if __name__ == '__main__':
    run(validator=validate, builder=build, format_name=FORMAT, evaluator=deployed,
        tracker_factory=SafeReachTracker)

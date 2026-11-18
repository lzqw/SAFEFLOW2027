"""v8 saved-label boundary emphasis; old registrations and checkpoints unchanged."""
from pathlib import Path
import json

import torch

from scripts.paper_crax_payload.direction_optimization import OptimizationLearner, validate_parent as original_parent

FORMAT = 'direction_refinement_v8_full'


def boundary_weights(labels, step, settings):
    if step <= settings['start']:
        return None
    return (1.+settings['gain']*torch.sigmoid(labels.detach()/settings['temperature_m'])).detach()


def validate_config(config):
    path = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/direction_refinement_v8_round1.json'
    if config != json.loads(path.read_text()):
        raise ValueError('Complete preregistration required for each new refinement round')
    if (config['protocol'] != 'direction_refinement_v8_continuous_uncapped'
            or config['budget']['development_ceiling'] is not None
            or config['budget']['user_authorized_no_total_training_cap'] is not True
            or config['absolute_target_steps'] != 250000 or config['training_seed'] != 2028
            or config['scientific_early_stopping'] or config['automatic_failed_run_retry']
            or config['evaluation']['split'] != 'development_only_even_at_final'):
        raise ValueError('Retain the current uncapped authorization and fixed comparable endpoints')
    for candidate in config['candidates']:
        if candidate['margin_emphasis'] != dict(start=150000, gain=9., temperature_m=.005):
            raise ValueError('No undisclosed label, margin threshold, capacity, or actor modification')


class RefinementLearner(OptimizationLearner):
    def margin_supervision_weights(self, batch):
        weights = boundary_weights(batch['margin'], self.transitions, self.candidate['margin_emphasis'])
        if weights is not None:
            self._last_metrics['margin_supervision_mean_weight'] = weights.mean()
        return weights

    def load_state_dict(self, state):
        if state.get('direction_refinement_v8') != self.candidate['margin_emphasis']:
            raise ValueError('Same margin-emphasis revision required for native v8 resume')
        super().load_state_dict(state)

    def state_dict(self):
        return super().state_dict() | dict(direction_refinement_v8=dict(self.candidate['margin_emphasis']))


def build(config, candidate, device):
    s = config['shared_learner']
    return RefinementLearner(candidate, device=device, actor_lr=s['actor_lr'], critic_lr=s['critic_lr'],
        critic_only_updates=s['critic_only_updates'], actor_interval=s['actor_update_interval'],
        margin_ema_tau=config['geometry']['ema_tau'])


def validate_parent(state, manifest, checkpoint, manifest_path, config, candidate):
    if (candidate['margin_emphasis']['start'] < config['parents'][candidate['parent']]['steps']
            or state['steps'] != 150000 or 'direction_refinement_v8' in state['learner']):
        raise ValueError('Historical complete prefix must precede the new margin-emphasis rule')
    return original_parent(state, manifest, checkpoint, manifest_path, config, candidate)

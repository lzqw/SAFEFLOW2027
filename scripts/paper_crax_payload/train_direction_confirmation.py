"""Frozen v5 three-seed confirmation training, NOT independent evaluation.

This entry point cannot run from a development config or an unfrozen selection.
All intermediate/final training evaluations remain on development layouts. A
separate closure must freeze all nine final policy identities before any fresh
independent layout is instantiated. No confirmation config is provided yet.
"""
import hashlib
import json

from scripts.paper_crax_payload.direction_development import build_registered_model
from scripts.paper_crax_payload.train_direction_development import main as train_native


PROTOCOL = 'direction_confirmation_v5_frozen'
FORMAT = 'direction_confirmation_v5_full'


def configuration_digest(config):
    value = {key: item for key, item in config.items() if key != 'freeze_sha256'}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_confirmation(config):
    if (config.get('protocol') != PROTOCOL
            or config.get('status') != 'frozen_before_confirmation_training_and_independent_test'
            or config.get('freeze_sha256') != configuration_digest(config)):
        raise ValueError('Require an explicit immutable confirmation freeze, not a development config')
    if (config['task_version'] != 'SafeReach-RelativeLayout-v3' or config['num_envs'] != 2
            or config['training_seeds'] != [2031, 2032, 2033] or config['steps_per_run'] != 300000
            or config['new_training_budget'] != 2700000 or len(config['arms']) != 3
            or config['scientific_early_stopping'] or config['automatic_failed_run_retry']
            or config['physical_robot_interactions'] != 0):
        raise ValueError('Retain the finite three-method, three-seed fresh300K confirmation budget')
    review = config['development_selection']
    if (review['completed_runs'] != 5 or review['actual_training_steps'] != 750000
            or review['selected_arm'] not in ('normal-shrink', 'normal-shrink-dir')
            or not review['review_rationale'] or not review['full_learning_curve_review']
            or review['independent_test_used_for_selection']
            or len(review['run_receipt_sha256']) != 5 or len(review['trajectory_summary_sha256']) != 5
            or any(len(v) != 64 for v in [*review['run_receipt_sha256'].values(),
                                         *review['trajectory_summary_sha256'].values()])):
        raise ValueError('Finish and review all five complete development arms before freezing selection')
    expected_names = {'iso', 'metric-only', review['selected_arm']}
    if {arm['name'] for arm in config['arms']} != expected_names:
        raise ValueError('Keep both declared controls and exactly one selected new method')
    for arm in config['arms']:
        expected = {
            'iso': ('iso', False, 0.), 'metric-only': ('metric_only', False, 0.),
            'normal-shrink': ('full', True, 0.), 'normal-shrink-dir': ('full', True, 1.)}[arm['name']]
        if ((arm['component'], arm['normal_shrink'], arm['direction_weight']) != expected
                or arm['rho'] != .25):
            raise ValueError('Confirmation must retain the selected development method, not retune it')
    shared = config['shared_learner']
    expected_shared = dict(actor_lr=.0001, critic_lr=.0003, batch_size=256, replay_capacity=300000,
        random_transitions=2000, critic_only_updates=1000, actor_update_interval=2, utd=1,
        reward_updates_per_round=2, target_samples=4, reference_variance=.04, energy_target=18.75,
        flow_steps=4, width=128)
    if shared != expected_shared or config['geometry'] != dict(ema_tau=.005, ramp_start=10000, ramp_end=30000):
        raise ValueError('No unregistered learner or geometry changes after selection')
    plan = config['evaluation']
    if (plan['milestones'] != [50000, 100000, 150000, 200000, 250000, 300000]
            or plan['development_layout_seed_start'] != 7100027 or plan['action_seed'] != 7200027
            or plan['development_episodes_each'] != 30
            or plan['split'] != 'development_only_even_at_final'
            or any('independent' in key for key in plan)
            or plan['conditions'] != [dict(name='nominal', radius=.06, action_scale=1.),
                                     dict(name='radius065', radius=.065, action_scale=1.),
                                     dict(name='execution085', radius=.06, action_scale=.85)]):
        raise ValueError('Training runner can evaluate development layouts only, even at300K')
    closure = config['independent_closure']
    if (closure['status'] != 'not_opened_requires_all_nine_final_policy_hashes'
            or closure['checkpoint_rule'] != 'fixed300K_all_three_seeds_no_reselection'
            or closure['episodes_per_condition_per_policy'] != 100
            or not closure['fresh_layout_seed_commitment_sha256']
            or len(closure['fresh_layout_seed_commitment_sha256']) != 64):
        raise ValueError('Independent testing is a separate all-policy-frozen closure, never an online tuning signal')


def build_confirmation(config, arm, seed, device):
    if seed not in config['training_seeds']:
        raise ValueError('Use a preregistered confirmation seed')
    return build_registered_model(config, arm, device)


if __name__ == '__main__':
    train_native(validator=validate_confirmation, builder=build_confirmation, format_name=FORMAT)

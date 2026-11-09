"""Small argument/config contracts for the requested development/main split."""
MAIN_SEEDS = {2030, 2031, 2032}
MAIN_METHODS = {'sac_lag', 'flow_iso', 'flow_learned'}
CONTROL_METHODS = {'flow_iso_low', 'flow_random_aniso'}


def resolve_phase(method, seed, steps, evaluation_episodes, phase=None, smoke=False):
    phase = phase or ('ability' if method in {'sac', 'flow'} else
                      'mechanism_control' if method in CONTROL_METHODS else 'obstacle_development')
    if phase == 'main':
        if smoke or method not in MAIN_METHODS or seed not in MAIN_SEEDS or steps != 1000000 or evaluation_episodes != 100:
            raise ValueError('Main uses SAC-Lag/Iso/Learned, held-out seeds2030/2031/2032,1M transitions and100 evaluation episodes')
    elif seed in MAIN_SEEDS:
        raise ValueError('Seeds2030/2031/2032 are reserved for frozen main runs, not development')
    if phase == 'relative_layout_screening':
        if method not in MAIN_METHODS or seed != 2027 or steps != 300000 or evaluation_episodes != 30 or smoke:
            raise ValueError('Relative-layout screening is only SAC-Lag/Iso/Learned,seed2027,300k each,30 evaluation episodes')
    if phase == 'ability' and method not in {'sac', 'flow'}:
        raise ValueError('Ability phase is the unobstructed SAC/reward-only flow check')
    if phase == 'mechanism_control' and method not in CONTROL_METHODS:
        raise ValueError('Mechanism-control phase is only for the two requested controls')
    return phase


def validate_main_freeze(record, method, seed, variance, rho=.25):
    if not record.get('frozen'):
        raise ValueError('Development configuration is not frozen; do not launch main training')
    if method not in record['methods'] or seed not in record['training_seeds']:
        raise ValueError('Method/seed is absent from the frozen configuration')
    required = dict(steps_per_seed=1000000, evaluation_episodes=100,
                    reference_variance=variance, normal_variance_ratio=rho, cost_limit=.1,
                    directional_loss=False, inward_drift=False, action_projection=False)
    for key, expected in required.items():
        if record.get(key) != expected:
            raise ValueError(f'Frozen {key} differs from the implemented main setting')
    if not record.get('frozen_method_code_commit'):
        raise ValueError('Record the frozen method code commit before main training')


def validate_resume_origin(seed, phase, manifest):
    if manifest.get('training_seed') != seed:
        raise ValueError('Checkpoint training seed differs; do not relabel development or another seed')
    if phase == 'main' and manifest.get('stage') != 'main':
        raise ValueError('A main run may resume only its own formal run, never a development checkpoint')

"""Small CPU-only admission guard for a prospectively frozen PointGoal plan.

This module does not freeze a plan, start jobs, import GPU libraries, or infer
scientific readiness from scores. Phase/provenance metadata stays OUTSIDE the
learner/native settings used by existing exact development continuations.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
METHODS = ('sac_lag', 'stable_iso', 'ea')
SEEDS = (51027, 51028, 51029)
BUDGETS = (5000000, 10000000)
FRACTIONS = (.2, .4, .6, .8, 1.)
TASK = dict(crax_commit='63504b540705be98024abd675e646ff5e046c6c2',
            env_id='safe_goal_point', difficulty=1, task_mode='cumulative',
            episodic_cost_budget=25.)
ALGORITHM_FILES = (
    'scripts/paper_crax_payload/train_crax.py',
    'scripts/paper_crax_payload/crax_models.py',
    'scripts/paper_crax_payload/crax_env.py',
    'scripts/paper_crax_payload/observation_normalizer.py',
    'scripts/paper_crax_payload/native_sac_checkpoint.py',
    'scripts/paper_crax_payload/native_sac_reference.py',
    'scripts/paper_crax_payload/formal_admission.py',
    'scripts/paper_crax_payload/configuration.py',
    'scripts/star/arm_safe_entropy/models.py',
    'scripts/star/arm_safe_entropy/endpoint_geometry.py',
    'scripts/star/arm_safe_entropy/geometry.py',
    'scripts/star/arm_safe_entropy/stability_runtime.py')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def algorithm_file_hashes():
    """Read-only helper for a later explicit freeze; never writes the plan."""
    return {name:_sha((ROOT/name).read_bytes()) for name in ALGORITHM_FILES}


def flow_runtime_parameters(master, profile_name, eval_envs):
    return dict(pointgoal=deepcopy(master['pointgoal']), profile_name=profile_name,
        profile=deepcopy(master['pointgoal']['profiles'][profile_name]), eval_envs=eval_envs,
        checkpoint_fractions=master['selection']['candidate_checkpoint_fractions'],
        maximum_checkpoints_per_seed=master['selection']['maximum_checkpoints_per_seed'])


def native_runtime_parameters(args):
    return dict(num_envs=args.num_envs, batch_size=args.batch_size, updates=args.updates,
        lagrangian_lr=args.lagrangian_lr, normalize_observations=args.normalize_observations,
        min_replay_size=args.min_replay_size, max_replay_size=args.max_replay_size,
        learning_rate=3e-4, discount=.99, reward_scaling=1., cost_scaling=1.,
        safety_bound=25., env_id='safe_goal_point', difficulty=1,
        checkpoint_fractions=list(FRACTIONS), maximum_checkpoints_per_seed=5)


def admit_run(*, phase, plan_path, method, seed, steps, runtime_parameters,
              source_commit, forbidden_flags=None):
    if phase == 'development':
        if plan_path is not None:
            raise ValueError('--formal-plan requires --phase formal')
        return None
    if phase != 'formal' or plan_path is None:
        raise ValueError('Formal runs require --phase formal and --formal-plan')
    data = Path(plan_path).read_bytes()
    plan = json.loads(data)
    if plan.get('status') != 'frozen':
        raise ValueError('Formal plan is not frozen; prepared_not_frozen does not admit training')
    if plan.get('schema_version') != 1 or plan.get('phase') != 'formal' or not plan.get('plan_id'):
        raise ValueError('Unsupported formal plan schema/phase')
    if (tuple(plan.get('methods', [])) != METHODS or tuple(plan.get('training_seeds', [])) != SEEDS
            or tuple(plan.get('allowed_common_budgets', [])) != BUDGETS):
        raise ValueError('Formal plan must retain all three registered methods/seeds and budgets')
    if (tuple(plan.get('checkpoint_fractions', [])) != FRACTIONS
            or plan.get('maximum_checkpoints_per_seed') != 5 or plan.get('task') != TASK):
        raise ValueError('Exact five fractions and native shared task/budget must be retained')
    common = plan.get('common_steps_per_seed')
    if type(common) is not int or common not in BUDGETS or steps != common:
        raise ValueError('Every method/seed must use the frozen common5M or10M budget')
    canonical = 'sac_lag' if method == 'sac_lag_native' else method
    if canonical not in METHODS or seed not in SEEDS:
        raise ValueError('Unregistered formal method or training seed')
    if any((forbidden_flags or {}).values()):
        raise ValueError('Engineering/profile/individual-target-extension flags are forbidden in formal runs')
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    runs = plan.get('runs', [])
    if (len(runs) != 9 or {(r.get('method'), r.get('training_seed')) for r in runs} != expected
            or any(r.get('steps', common) != common for r in runs)):
        raise ValueError('Formal plan must register the full nine-run common-budget matrix')
    freeze = plan.get('freeze_record')
    if (not isinstance(freeze, dict) or not freeze.get('frozen_at_utc')
            or not freeze.get('decision_evidence') or freeze.get('all_methods_common_budget_confirmed') is not True):
        raise ValueError('An explicit prospective freeze record is required')
    specs = plan.get('method_configurations', {})
    if set(specs) != set(METHODS) or any(not isinstance(specs[m].get('runtime_parameters'), dict)
                                          or not specs[m]['runtime_parameters'] for m in METHODS):
        raise ValueError('All three runtime configurations, including native SAC LR/normalization, must be frozen')
    if specs['stable_iso']['runtime_parameters'] != specs['ea']['runtime_parameters']:
        raise ValueError('Stable-Iso and EA must retain the same frozen Flow shared configuration')
    flow_task = specs['stable_iso']['runtime_parameters'].get('pointgoal', {})
    if (not isinstance(flow_task, dict)
            or any(flow_task.get(key) != TASK[key] for key in
                   ('env_id', 'difficulty', 'task_mode', 'episodic_cost_budget'))
            or flow_task.get('environment_overrides') != {}):
        raise ValueError('Frozen Flow runtime disagrees with the native task/budget or adds environment overrides')
    if specs[canonical]['runtime_parameters'] != runtime_parameters:
        raise ValueError('Runtime CLI/config differs from the frozen method configuration')
    if (tuple(runtime_parameters.get('checkpoint_fractions', [])) != FRACTIONS
            or runtime_parameters.get('maximum_checkpoints_per_seed') != 5):
        raise ValueError('Runtime must preserve exactly five declared checkpoint fractions')
    if not re.fullmatch('[0-9a-fA-F]{40}', source_commit):
        raise ValueError('Record the explicit runtime source commit')
    hashes = plan.get('algorithm_file_sha256')
    if not isinstance(hashes, dict) or set(hashes) != set(ALGORITHM_FILES):
        raise ValueError('Frozen algorithm file hashes are required; plan commit is not its own source hash')
    if any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)
           for value in hashes.values()) or hashes != algorithm_file_hashes():
        raise ValueError('Frozen scientific/entrypoint files changed')
    return dict(plan_id=plan['plan_id'], plan_sha256=_sha(data), plan_path=str(plan_path),
        method=canonical, training_seed=seed, common_steps_per_seed=common,
        checkpoint_fractions=list(FRACTIONS), nominal_checkpoint_steps=[int(common*f) for f in FRACTIONS],
        registered_training_seeds=list(SEEDS), registered_methods=list(METHODS),
        runtime_configuration_sha256=_sha(json.dumps(runtime_parameters,sort_keys=True,separators=(',',':')).encode()),
        algorithm_file_sha256=hashes, source_binding='exact_algorithm_file_hashes_not_self_referential_plan_commit',
        runtime_source_commit=source_commit, phase='formal')


def phase_fields(admission):
    return {} if admission is None else dict(phase='formal', formal_admission=admission)


def require_resume_phase(admission, payload):
    if admission is None:
        if payload.get('phase') == 'formal':
            raise ValueError('Do not relabel a formal checkpoint as development')
        return
    stored = payload.get('formal_admission', {})
    keys = ('plan_sha256', 'method', 'training_seed', 'common_steps_per_seed',
            'runtime_configuration_sha256', 'algorithm_file_sha256')
    if payload.get('phase') != 'formal' or any(stored.get(key) != admission[key] for key in keys):
        raise ValueError('Formal continuation requires the same frozen plan; development cannot be relabelled')

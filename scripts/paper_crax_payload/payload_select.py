"""Prospective equal-opportunity Payload selection; no physics or training here."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def candidates(plan, cohort):
    specification = plan['cohorts'][cohort]
    rows = []
    for method, short in plan['methods'].items():
        for seed in specification['training_seeds']:
            run = specification['run_template'].format(method=method, short=short, seed=seed)
            for fraction in plan['checkpoint_fractions']:
                step = round(specification['target_steps']*fraction)
                rows.append(dict(candidate_id=f'{cohort}:{method}:{seed}:{step:09d}', method=method,
                    training_seed=seed, checkpoint_steps=step, run_id=run,
                    policy_filename=f'policy_{step}.pt', full_filename=f'full_{step}.pt',
                    fixed_endpoint=step==specification['target_steps']))
    return rows


def summarize_validation(document, plan):
    conditions = document['result']['conditions']
    expected = [c['name'] for c in plan['validation']['conditions']]
    if [c['condition'] for c in conditions] != expected:
        raise ValueError('Every registered condition, including offset, is mandatory and ordered')
    risks, successes, capped, returns = [], [], [], []
    count = plan['validation']['episodes_per_condition']
    for condition in conditions:
        records = condition['records']
        if condition['episodes'] != count or len(records) != count:
            raise ValueError('Validation requires the same complete episode count for every candidate')
        wanted = list(range(plan['validation']['layout_seed_start'], plan['validation']['layout_seed_start']+count))
        if [r['layout_seed'] for r in records] != wanted:
            raise ValueError('Validation layout seeds differ from the registered common stream')
        for record in records:
            if record['safe_success'] and record['violation']:
                raise ValueError('Success and violation terminal classes overlap')
            if not record['safe_success'] and not record['violation'] and not record['true_horizon_terminal']:
                raise ValueError('Unclassified failure cannot disappear')
            if not record['safe_success'] and abs(record['time_to_goal_capped_seconds']-plan['horizon_seconds']) > 1e-8:
                raise ValueError('Failed/timeout episodes must retain the full capped horizon')
            if not all(math.isfinite(record[k]) for k in ('episode_return', 'time_to_goal_capped_seconds')):
                raise ValueError('Nonfinite result')
        risks.append(mean(float(r['violation']) for r in records))
        successes.append(mean(float(r['safe_success']) for r in records))
        capped.append(mean(r['time_to_goal_capped_seconds'] for r in records))
        returns.append(mean(r['episode_return'] for r in records))
    threshold = plan['risk_qualification']
    qualified = risks[0] <= threshold['nominal_max_violation'] and all(
        risk <= threshold['each_stress_max_violation'] for risk in risks[1:])
    return dict(risk_qualified=qualified, task_progress_observed=any(value>0. for value in successes),
        condition_names=expected, per_condition_violation=risks,
        per_condition_safe_success=successes, macro_safe_success=mean(successes),
        macro_capped_seconds=mean(capped), macro_native_return=mean(returns),
        stress_mean_violation_diagnostic_only=mean(risks[1:]))


def dominates(left, right):
    a = left['score']['per_condition_violation'] + [-left['score']['macro_safe_success'], left['score']['macro_capped_seconds']]
    b = right['score']['per_condition_violation'] + [-right['score']['macro_safe_success'], right['score']['macro_capped_seconds']]
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def select(plan, cohort, documents):
    registered = {c['candidate_id']: c for c in candidates(plan, cohort)}
    supplied = {}
    for document in documents:
        identifier = document['candidate_id']
        if identifier not in registered or identifier in supplied:
            raise ValueError('Unknown or duplicate candidate; do not add extra checkpoints to one method')
        if document.get('plan_digest') != digest(plan) or document.get('cohort') != cohort:
            raise ValueError('Plan/cohort mismatch')
        if document.get('candidate') != registered[identifier]:
            raise ValueError('Candidate method/seed/checkpoint identity mismatch')
        if document.get('status') == 'unavailable_verified_terminal':
            proof = document.get('evidence', {})
            if proof.get('native_process_confirmed_terminal') is not True or not proof.get('retained_failure_receipt'):
                raise ValueError('Missing candidate needs explicit terminal failure evidence, not a stale status')
            supplied[identifier] = dict(candidate=registered[identifier], unavailable=True, evidence=proof)
            continue
        if document.get('status') != 'completed' or document.get('split') != 'validation':
            raise ValueError('Only complete common validation, never development/test scores, can select')
        if not document.get('policy_sha256') or not document.get('training_config_digest'):
            raise ValueError('Missing native policy/config provenance')
        if document.get('frozen_unchanged') is not True:
            raise ValueError('Actor/margins/normalizer/dual were not proven frozen')
        supplied[identifier] = dict(candidate=registered[identifier], score=summarize_validation(document, plan),
            policy_sha256=document['policy_sha256'], training_config_digest=document['training_config_digest'],
            validation_digest=digest(document), unavailable=False)
    if set(supplied) != set(registered):
        raise ValueError('Incomplete fair opportunity set: finish validation or preserve verified terminal dispositions')
    configs = {r['training_config_digest'] for r in supplied.values() if not r['unavailable']}
    if len(configs) > 1:
        raise ValueError('Do not compare unequal shared training configuration under one selection freeze')
    chosen = {}
    for method in plan['methods']:
        observed = [row for row in supplied.values() if row['candidate']['method']==method and not row['unavailable']]
        if not observed:
            chosen[method] = dict(status='all_registered_candidates_failed', selected=None,
                risk_qualified=False, task_progress_observed=False, eligible_for_deployment=False)
            continue
        eligible = [row for row in observed if row['score']['risk_qualified']]
        front = [row for row in observed if not any(dominates(other, row) for other in observed)]
        pool = eligible or front
        winner = min(pool, key=lambda row: (-row['score']['macro_safe_success'],
            row['score']['macro_capped_seconds'], row['candidate']['candidate_id']))
        progress=winner['score']['task_progress_observed']
        status=('risk_qualified' if progress else 'risk_qualified_but_stagnant') if eligible else 'pareto_tradeoff_not_risk_qualified'
        chosen[method] = dict(status=status,
            selected=winner, risk_qualified=bool(eligible), task_progress_observed=progress,
            eligible_for_deployment=bool(eligible) and progress,
            allowed_candidate_count=sum(c['method']==method for c in registered.values()),
            evaluated_candidate_count=len(observed), pareto_candidate_ids=[r['candidate']['candidate_id'] for r in front])
    result = dict(schema='payload_frozen_selection_v1', status='complete_frozen_selection',
        cohort=cohort, plan_digest=digest(plan), registered_candidates=list(registered.values()),
        candidate_evidence=supplied, methods=chosen, independent_test_used_for_selection=False,
        fixed_endpoints_replaced=False, training_seeds_are_replication_unit=True,
        new_training_transitions=0, new_simulation_control_intervals=0)
    result['freeze_digest'] = digest(result)
    return result


def verify_test_authorization(plan, cohort, candidate, policy_sha256, selection):
    expected = selection.get('freeze_digest')
    payload = {k: v for k, v in selection.items() if k != 'freeze_digest'}
    if expected != digest(payload) or selection.get('status') != 'complete_frozen_selection':
        raise ValueError('Incomplete or altered model-selection freeze')
    if (selection.get('plan_digest') != digest(plan) or selection.get('cohort') != cohort or
            selection.get('independent_test_used_for_selection') is not False):
        raise ValueError('Independent test cannot choose or modify the selection')
    record = selection['methods'][candidate['method']]['selected']
    if record is None or record['candidate'] != candidate or record['policy_sha256'] != policy_sha256:
        raise ValueError('Only the exact frozen selected native policy may enter independent test')
    return selection['methods'][candidate['method']]['status']


def validation_manifest(plan, cohort, plan_path, training_config, run_root, evaluation_root, python):
    registered = candidates(plan, cohort)
    commands = []
    for candidate in registered:
        target = Path(evaluation_root)/(candidate['candidate_id'].replace(':', '_')+'.json')
        commands.append(dict(candidate_id=candidate['candidate_id'], output=str(target), command=[python,
            '-m', 'scripts.paper_crax_payload.payload_frozen', '--plan', str(plan_path),
            '--training-config', str(training_config), '--cohort', cohort,
            '--candidate-id', candidate['candidate_id'], '--run-root', str(run_root),
            '--split', 'validation', '--device', 'cpu', '--output', str(target)]))
    return dict(schema='payload_prospective_candidate_manifest_v1', cohort=cohort,
        plan_digest=digest(plan), candidates=registered, validation_commands=commands,
        independent_test_commands_not_authorized=True,
        independent_test_executed=False, new_training_transitions=0, new_simulation_control_intervals=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--validation-directory', type=Path)
    parser.add_argument('--training-config', type=Path, default=Path('configs/paper_crax_payload/payload_v1.json'))
    parser.add_argument('--run-root', type=Path, default=Path('/home/carla/4T/SAFEFLOW2027_PAPER/runs'))
    parser.add_argument('--evaluation-root', type=Path, default=Path('/home/carla/4T/SAFEFLOW2027_PAPER/analysis/payload-common-validation-v1'))
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve previous manifests/selection freezes')
    plan = json.loads(args.plan.read_text())
    if args.validation_directory is None:
        result = validation_manifest(plan, args.cohort, args.plan, args.training_config,
            args.run_root, args.evaluation_root, args.python)
    else:
        documents = [json.loads(p.read_text()) for p in sorted(args.validation_directory.glob('*.json'))]
        result = select(plan, args.cohort, documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('candidates', 'validation_commands', 'registered_candidates', 'candidate_evidence', 'methods')}))


if __name__ == '__main__':
    main()

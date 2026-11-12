"""Freeze fair PointGoal model selection from common validation only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def num(row, key):
    value = row.get(key)
    return None if value in (None, '', 'None') else float(value)


def method_from_run(run):
    for method in ('sac_lag', 'stable_iso', 'ea'):
        if f'formal-{method}-' in run:
            return method
    raise ValueError(f'Unknown formal method in {run}')


def seed_from_run(run):
    return int(run.split('-')[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sac-csv', type=Path, required=True)
    parser.add_argument('--flow-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    rows = read_rows(args.sac_csv) + read_rows(args.flow_csv)
    if len(rows) != 225:
        raise ValueError(f'Expected 45 candidates × 5 validation seeds = 225 rows, got {len(rows)}')
    grouped = {}
    for row in rows:
        run = row['run']; checkpoint = int(row['checkpoint_steps'])
        method = row.get('method') or method_from_run(run)
        training_seed = int(row.get('training_seed') or seed_from_run(run))
        key = (method, run, checkpoint)
        grouped.setdefault(key, []).append(row)
    if len(grouped) != 45 or any(len(v) != 5 for v in grouped.values()):
        raise ValueError('Common validation must contain exactly five seeds for every 45 candidate')
    candidates = []
    for (method, run, checkpoint), entries in sorted(grouped.items()):
        seeds = sorted(int(r['validation_seed']) for r in entries)
        if seeds != [61027, 61028, 61029, 61030, 61031]:
            raise ValueError(f'Unexpected validation seeds for {run}/{checkpoint}: {seeds}')
        policy_hashes = sorted({r['policy_sha256'] for r in entries})
        if len(policy_hashes) != 1:
            raise ValueError('Candidate policy hash changed across validation seeds')
        summary = {
            'method': method, 'run': run, 'training_seed': int(entries[0].get('training_seed') or seed_from_run(run)),
            'checkpoint_steps': checkpoint, 'policy_sha256': policy_hashes[0],
            'validation_seeds': seeds,
            'mean_native_return': mean(num(r, 'native_return') for r in entries),
            'mean_episode_cost': mean(num(r, 'episode_cost') for r in entries),
            'mean_goals_per_episode': mean(num(r, 'goals_per_episode') for r in entries),
            'mean_at_least_one_goal': mean(num(r, 'at_least_one_goal') for r in entries),
            'mean_budget_compliant_fraction': mean(num(r, 'budget_compliant_fraction') for r in entries),
            'mean_budget_compliant_return': mean(num(r, 'budget_compliant_return') for r in entries if num(r, 'budget_compliant_return') is not None),
            'mean_capped_first_goal_seconds': mean(num(r, 'capped_first_goal_seconds') for r in entries),
            'per_seed_episode_cost': [num(r, 'episode_cost') for r in entries],
            'per_seed_goals_per_episode': [num(r, 'goals_per_episode') for r in entries],
            'risk_qualified': mean(num(r, 'episode_cost') for r in entries) <= 25.0,
            'independent_test_used': False,
        }
        candidates.append(summary)
    methods = {}
    for method in ('sac_lag', 'stable_iso', 'ea'):
        pool = [c for c in candidates if c['method'] == method]
        eligible = [c for c in pool if c['risk_qualified']]
        front = [c for c in pool if not any(
            other['mean_episode_cost'] <= c['mean_episode_cost'] and
            other['mean_goals_per_episode'] >= c['mean_goals_per_episode'] and
            (other['mean_episode_cost'] < c['mean_episode_cost'] or other['mean_goals_per_episode'] > c['mean_goals_per_episode'])
            for other in pool)]
        ranking_pool = eligible or front
        winner = min(ranking_pool, key=lambda c: (-c['mean_goals_per_episode'],
            c['mean_episode_cost'], c['mean_capped_first_goal_seconds'],
            c['run'], c['checkpoint_steps']))
        methods[method] = dict(
            status='risk_qualified' if eligible else 'pareto_tradeoff_not_risk_qualified',
            selected=winner, eligible_for_deployment=bool(eligible),
            candidate_count=len(pool), risk_qualified_count=len(eligible),
            pareto_candidates=front)
    deployment_pool = [v['selected'] for v in methods.values() if v['eligible_for_deployment']]
    overall = max(deployment_pool, key=lambda c: (c['mean_goals_per_episode'],
        -c['mean_episode_cost'], -c['mean_capped_first_goal_seconds'])) if deployment_pool else None
    result = dict(schema='pointgoal_common_validation_selection_v1', status='complete_frozen_selection',
        source_commit=args.source_commit, common_validation_seeds=[61027,61028,61029,61030,61031],
        episodes_per_seed=50, risk_rule={'mean_episode_cost_max':25.0},
        independent_test_used=False, candidates=candidates, methods=methods,
        overall_deployment_candidate=overall)
    result['freeze_digest'] = digest(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'freeze_digest': result['freeze_digest'],
                      'overall': overall, 'methods': methods}, sort_keys=True))


if __name__ == '__main__':
    main()

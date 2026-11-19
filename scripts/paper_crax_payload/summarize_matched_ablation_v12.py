"""Review all four complete250K ablations; never substitute early/partial nodes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from scripts.paper_crax_payload.summarize_simulation_v12 import output


COUNTS = ('training_violations', 'safe_successes', 'safe_timeouts', 'training_stall_episodes',
          'training_stall_windows', 'training_cumulative_return', 'ordinary_terminal_task_successes')
ARMS = ('A-full-P5', 'A-no-direction', 'A-no-lateLR', 'A-Metric-matchedLR')
PLAN = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_matched_ablation.json'
CONFIRMATION_PLAN = PLAN.with_name('simulation_v12_samehardware_confirmation.json')


def registered_protocols(same_hardware=False, direction_taper=False, endpoint_residual=False):
    original = json.loads(PLAN.read_text())
    if not same_hardware and not direction_taper and not endpoint_residual:
        return {arm: original for arm in ARMS}
    confirmation = json.loads(CONFIRMATION_PLAN.read_text())
    for key in ('task_version', 'num_envs', 'development_training_seeds', 'steps_per_run',
                'shared_learner', 'geometry', 'deployment', 'evaluation'):
        if confirmation[key] != original[key]:
            raise ValueError('Samehardware confirmation must retain the original scientific protocol')
    result = {'A-full-P5': confirmation, 'A-Iso-matchedLR': confirmation,
              'A-no-lateLR': original, 'A-Metric-matchedLR': original}
    if direction_taper:
        candidate = json.loads(PLAN.with_name('simulation_v12_direction_taper.json').read_text())
        for key in ('task_version', 'num_envs', 'development_training_seeds', 'steps_per_run',
                    'shared_learner', 'geometry', 'deployment', 'evaluation'):
            if candidate[key] != original[key]:
                raise ValueError('Direction-taper candidate must retain the common scientific protocol')
        result['D-direction-only-taper'] = candidate
    if endpoint_residual:
        candidate = json.loads(PLAN.with_name('simulation_v12_endpoint_residual.json').read_text())
        for key in ('task_version', 'num_envs', 'development_training_seeds', 'steps_per_run',
                    'shared_learner', 'geometry', 'deployment', 'evaluation'):
            if candidate[key] != original[key]:
                raise ValueError('Endpoint residual pair must retain the common scientific protocol')
        result.update({arm['name']:candidate for arm in candidate['arms']})
    return result


def candidate_checks(windows, nodes, candidate='D-direction-only-taper',
                     baselines=('A-full-P5', 'A-Iso-matchedLR', 'A-Metric-matchedLR')):
    """Separate every explicit DEV criterion; no independent confirmation claim."""
    checks = []
    def window(arm, start, end):
        return next(r for r in windows if r['arm'] == arm and r['start_transitions'] == start and r['end_transitions'] == end)
    def add(name, baseline, scope, observed, limit, relation):
        checks.append(dict(criterion=name, baseline=baseline, scope=scope, candidate_value=observed,
            comparison_limit=limit, relation=relation,
            satisfied=observed<=limit if relation=='<=' else observed>=limit,
            independent_confirmation=False))
    early = window(candidate, 0, 100000); total = window(candidate, 0, 250000)
    add('historical_first100K_violations', 'historical_Iso_331.5', '0-100K', early['training_violations'], 331.5, '<=')
    for baseline in baselines:
        reference = window(baseline, 0, 250000)
        add('first100K_violations_no_higher', baseline, '0-100K', early['training_violations'],
            window(baseline,0,100000)['training_violations'], '<=')
        add('late_violations_no_higher', baseline, '100-250K', window(candidate,100000,250000)['training_violations'],
            window(baseline,100000,250000)['training_violations'], '<=')
        for key in ('safe_timeouts','training_stall_episodes'):
            for start, end in ((0,100000),(100000,250000),(0,250000)):
                add(key+'_no_higher', baseline, f'{start}-{end}', window(candidate,start,end)[key],
                    window(baseline,start,end)[key], '<=')
        for start, end in ((0,100000),(100000,250000),(0,250000)):
            add('training_safe_success_count_no_lower', baseline, f'{start}-{end}',
                window(candidate,start,end)['safe_successes'], window(baseline,start,end)['safe_successes'], '>=')
        # For negative reference returns, permit a drop of1% of its magnitude.
        add('cumulative_return_drop_at_most1percent', baseline, '0-250K', total['training_cumulative_return'],
            reference['training_cumulative_return']-.01*abs(reference['training_cumulative_return']), '>=')
        if baseline in ('A-Iso-matchedLR', 'A-Metric-matchedLR'):
            add('training_violations_at_least20percent_lower', baseline, '0-250K', total['training_violations'],
                .8*reference['training_violations'], '<=')
        for row in (r for r in nodes if r['arm'] == candidate):
            ref = next(r for r in nodes if r['arm'] == baseline and r['transitions'] == row['transitions'] and r['condition'] == row['condition'])
            for key, relation in (('safe_successes','>='),('violations','<='),('timeouts','<='),('stall_count','<=')):
                add('DEV_'+key+'_no_worse', baseline, f"{row['transitions']}:{row['condition']}", row[key], ref[key], relation)
    return checks


def training_curve_comparisons(progress, candidate, baselines):
    indexed = {(r['arm'],r['steps']):r for r in progress}
    comparisons = []
    for row in (r for r in progress if r['arm'] == candidate):
        for baseline in baselines:
            ref = indexed[baseline,row['steps']]
            comparisons.append(dict(candidate=candidate, baseline=baseline, transitions=row['steps'],
                **{'delta_'+key:row[key]-ref[key] for key in COUNTS}))
    return comparisons


def pre_decay_prefix(progress):
    """Disclose numeric divergence before LR differs; never interpolate it away."""
    indexed = {arm: {r['steps']: r for r in progress if r['arm'] == arm}
               for arm in ('A-full-P5', 'A-no-lateLR')}
    keys = COUNTS + ('actor_loss', 'reward_loss', 'margin_loss', 'cost_loss')
    differences = []
    for step in range(1000, 150001, 1000):
        left, right = (indexed[arm][step] for arm in indexed)
        differing = [key for key in keys if left[key] != right[key]]
        if differing:
            differences.append(dict(transitions=step, differing_metrics=differing))
    return dict(exact_recorded_prefix_match=not differences,
        first_observed_difference=differences[0] if differences else None,
        different_nodes=len(differences), compared_nodes=150,
        strict_lr_causal_claim_permitted=False,
        interpretation='Same host removes hardware split; one seed and any early divergence still limit LR attribution')


def training_window(rows, start, end):
    indexed = {int(r['steps']): r for r in rows}
    if len(indexed) != len(rows) or end not in indexed or (start and start not in indexed):
        raise ValueError('Exact saved boundaries required; do not interpolate cumulative counts')
    result = {key: indexed[end][key]-(indexed[start][key] if start else 0) for key in COUNTS}
    if any(result[key] < 0 for key in COUNTS if key != 'training_cumulative_return'):
        raise ValueError('Cumulative terminal counts cannot decrease')
    if result['training_violations']+result['safe_successes']+result['safe_timeouts'] != indexed[end]['episodes']-(indexed[start]['episodes'] if start else 0):
        raise ValueError('Every completed training episode must have exactly one terminal class')
    return dict(start_transitions=start, end_transitions=end, actual_transitions=end-start, **result,
        violations_per100k_steps=100000*result['training_violations']/(end-start))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--same-hardware', action='store_true', help='Use registered4080 P5/Iso plus original4080 noLR/Metric')
    group.add_argument('--direction-taper', action='store_true', help='Compare one complete candidate with all four4080 controls')
    group.add_argument('--endpoint-residual', action='store_true', help='Compare complete new pair with all four historical4080 controls')
    args = p.parse_args(); protocols = registered_protocols(args.same_hardware,args.direction_taper,args.endpoint_residual)
    windows = []; nodes = []; all_episodes = []; progress = []; provenance = []; identities = set()
    for arm, config in protocols.items():
        directory = args.runs/(arm+'-2034')
        manifest = json.loads((directory/'run_config.json').read_text())
        receipt = json.loads((directory/'receipt.json').read_text())
        expected = next(a for a in config['arms'] if a['name'] == arm)
        if args.same_hardware or args.direction_taper or args.endpoint_residual:
            source = ('499a4b6d' if arm.startswith('E-') else
                      '61bd30f0' if arm == 'D-direction-only-taper' else
                      '2c06ca2a' if arm in ('A-full-P5', 'A-Iso-matchedLR') else '2f09231b')
            if manifest['host'] != '447' or manifest['source_commit'] != source:
                raise ValueError('Require the exact registered4080 native controls, not the A100 P5 substitute')
        if (manifest['configuration'] != config or manifest['arm'] != expected or manifest['training_seed'] != 2034
                or receipt['arm'] != expected or receipt['training_seed'] != 2034
                or receipt['status'] != 'completed' or receipt['steps'] != 250000
                or receipt['new_training_transitions'] != 250000 or receipt['engineering_only']
                or receipt['source_commit'] != manifest['source_commit']):
            raise ValueError('Need every exact registered scientific250K receipt, never a partial/best endpoint')
        identities.add(receipt['initial_network_sha256'])
        if args.endpoint_residual and arm.startswith('E-'):
            if (manifest['format'] != 'safereach_fresh250k_endpoint_margin_residual_v12_full'
                    or receipt['evaluations'] != config['evaluation']['milestones']
                    or manifest['initial_network_sha256'] != receipt['initial_network_sha256']
                    or receipt['initial_network_sha256'] != '6eed16a4650593305c7ecb157ffe6d61c990b7c2fa1dad307a1a8025b6f31e6a'
                    or receipt['independent_endpoint_test_executed'] is not False):
                raise ValueError('Require all native residual-pair DEV receipts; no independent test')
        rows = [json.loads(s) for s in (directory/'progress.jsonl').read_text().splitlines()]
        if len(rows) != 250 or [r['steps'] for r in rows] != list(range(1000,250001,1000)):
            raise ValueError('Retain the full1K..250K training curve')
        for key in ('training_violations', 'safe_successes', 'safe_timeouts'):
            if rows[-1][key] != receipt[key]: raise ValueError('Receipt differs from full endpoint counts')
        for key in ('training_stall_episodes', 'training_stall_windows', 'training_cumulative_return'):
            if rows[-1][key] != receipt['training_metrics'][key]: raise ValueError('Tracker receipt changed')
        for start, end in ((0,100000),(100000,250000),(0,250000)):
            windows.append(dict(arm=arm, training_seed=2034, **training_window(rows,start,end)))
        progress.extend(dict(arm=arm, **r) for r in rows)
        for step in config['evaluation']['milestones']:
            value = json.loads((directory/f'evaluation_{step}.json').read_text())
            if value['milestone'] != step or value.get('independent_test_used') is not False:
                raise ValueError('Use the registered complete DEV nodes')
            if [c['condition'] for c in value['conditions']] != [c['name'] for c in config['evaluation']['conditions']]:
                raise ValueError('All registered conditions required')
            for c in value['conditions']:
                rs = c['records']; n = len(rs)
                if n != 30 or [r['layout_seed'] for r in rs] != list(range(11910000,11910030)):
                    raise ValueError('Do not drop/replace failed DEV layouts')
                success = sum(r['safe_success'] for r in rs); bad = sum(r['violation'] for r in rs)
                if any(r['safe_success'] and r['violation'] for r in rs): raise ValueError('Invalid terminal classes')
                nodes.append(dict(arm=arm, transitions=step, condition=c['condition'], episodes=n,
                    safe_successes=success, violations=bad, timeouts=n-success-bad,
                    stall_count=sum(r['stall'] for r in rs), ordinary_terminal_task_successes=sum(r['ordinary_terminal_task_success'] for r in rs),
                    mean_return=sum(r['episode_return'] for r in rs)/n,
                    mean_capped_seconds=sum(r['capped_seconds'] for r in rs)/n))
                all_episodes.extend(dict(arm=arm, transitions=step, condition=c['condition'],
                    **{k:v for k,v in r.items() if k not in ('control_details','trajectory_tcp_xyz','layout')}) for r in rs)
        provenance.append(dict(arm=arm, native_host=manifest['host'], source_commit=manifest['source_commit'],
            receipt_sha256=hashlib.sha256((directory/'receipt.json').read_bytes()).hexdigest(),
            initial_network_sha256=receipt['initial_network_sha256']))
    if len(identities) != 1: raise ValueError('All registered arms must share identical initialization')
    paired = []
    candidate_arm = ('E-endpoint-residual' if args.endpoint_residual else
                     'D-direction-only-taper' if args.direction_taper else 'A-full-P5')
    for row in [r for r in nodes if r['arm'] == candidate_arm]:
        for baseline in (arm for arm in protocols if arm != candidate_arm):
            ref = next(r for r in nodes if r['arm'] == baseline and r['transitions'] == row['transitions'] and r['condition'] == row['condition'])
            paired.append(dict(candidate=candidate_arm, baseline=baseline, transitions=row['transitions'],condition=row['condition'],
                **{'delta_'+k:row[k]-ref[k] for k in ('safe_successes','violations','timeouts','stall_count','mean_return','mean_capped_seconds')}))
    args.output.mkdir(parents=True,exist_ok=False)
    for name, values in [('training_windows.csv',windows),('development_nodes.csv',nodes),
                         ('development_episodes.csv',all_episodes),('training.csv',progress),('paired_nodes.csv',paired)]:
        output(args.output/name,values)
    if args.direction_taper or args.endpoint_residual:
        baselines = ('E-control-P5','A-full-P5','A-Iso-matchedLR','A-Metric-matchedLR') if args.endpoint_residual else (
            'A-full-P5','A-Iso-matchedLR','A-Metric-matchedLR')
        output(args.output/'development_criteria.csv',candidate_checks(windows,nodes,candidate_arm,baselines))
        output(args.output/'training_curve_comparisons.csv',training_curve_comparisons(progress,candidate_arm,
            [arm for arm in protocols if arm != candidate_arm]))
    if args.endpoint_residual:
        output(args.output/'control_against_historical_P5.csv',training_curve_comparisons(
            progress,'E-control-P5',['A-full-P5']))
    (args.output/'receipt.json').write_text(json.dumps(dict(status='complete_offline_review',
        runs=len(protocols), actual_completed_training_transitions=len(protocols)*250000, development_nodes=len(nodes),
        development_episodes=len(all_episodes), training_seed_count=1, provenance=provenance,
        all_registered_nodes_retained=True, new_training_steps=0, new_simulation_controls=0,
        hardware_split_disclosed=not (args.same_hardware or args.direction_taper or args.endpoint_residual), strict_causal_crosshardware_claim=False,
        same_hardware_confirmation=args.same_hardware,
        direction_taper_screen=args.direction_taper,
        endpoint_residual_screen=args.endpoint_residual,
        new_campaign_completed_training_transitions=500000 if args.endpoint_residual else None,
        historical_comparison_training_transitions=1000000 if args.endpoint_residual else None,
        contemporary_baselines=['E-control-P5'] if args.endpoint_residual else None,
        historical_samehardware_baselines=['A-full-P5','A-Iso-matchedLR','A-no-lateLR','A-Metric-matchedLR'] if args.endpoint_residual else None,
        pre_decay_prefix_comparison=pre_decay_prefix(progress) if args.same_hardware or args.direction_taper or args.endpoint_residual else None,
        independent_test_used=False, winner_confirmation=False),indent=2)+'\n')


if __name__ == '__main__': main()

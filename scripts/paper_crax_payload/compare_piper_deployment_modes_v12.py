"""Strict complete4200-episode DEV comparison; equally treat Metric deployment."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.paper_crax_payload.analyze_piper_failures_v12 import paired
from scripts.paper_crax_payload.summarize_simulation_v12 import output


PLAN = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_piper_deployment_modes.json'


def metrics(records):
    n = len(records)
    return dict(episodes=n,safe_successes=sum(r['safe_success'] for r in records),
        ordinary_terminal_task_successes=sum(r['ordinary_terminal_task_success'] for r in records),
        violations=sum(r['violation'] for r in records),timeouts=sum(r['safe_timeout'] for r in records),
        stall_count=sum(r['stall'] for r in records),
        mean_return=sum(r['episode_return'] for r in records)/n,
        mean_capped_seconds=sum(r['time_to_goal_capped_seconds'] for r in records)/n,
        mean_episode_inference_p50_ms=sum(r['inference_p50_ms'] for r in records)/n,
        mean_episode_inference_p95_ms=sum(r['inference_p95_ms'] for r in records)/n)


def comparison(candidate, baseline):
    left,right = metrics(candidate),metrics(baseline)
    return dict(**paired(candidate,baseline),
        **{'delta_'+key:left[key]-right[key] for key in left if key!='episodes'})


def validate_records(records, first, count):
    if [r['seed'] for r in records] != list(range(first,first+count)):
        raise ValueError('Every registered layout in original order is required')
    for row in records:
        if int(row['safe_success'])+int(row['violation'])+int(row['safe_timeout']) != 1:
            raise ValueError('Every episode must retain exactly one native terminal class')
        if row['length']<1 or len(row['trajectory'])!=row['length']+1:
            raise ValueError('Keep the complete native episode, including reset position')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(); spec = json.loads(PLAN.read_text())
    receipt = json.loads((args.runs/'receipt.json').read_text())
    manifest = json.loads((args.runs/'run_config.json').read_text())
    if (receipt['status']!='completed' or receipt['conditions']!=42 or receipt['evaluated_episodes']!=4200
            or receipt['source_commit']!='c7cb61a9' or manifest['source_commit']!='c7cb61a9'
            or manifest['protocol']!=spec or manifest['host']!='CTAlab'
            or receipt['new_training_steps']!=0 or receipt['optimizer_updates']!=0
            or receipt['independent_test_used'] is not False or not receipt['whole_policy_trajectories']):
        raise ValueError('Require the original complete whole-policy4200-episode DEV experiment')
    records = {}; summaries = []; provenance = []; all_episodes = []
    for case in spec['cases']:
        training = json.loads(Path(case['plan']).read_text())
        expected = [c['name'] for c in training['lines'][case['line']]['conditions']]
        for mode in spec['deployment_modes']:
            path = args.runs/(case['name']+'__'+mode+'.json'); value = json.loads(path.read_text())
            if (value['case']!=case['name'] or value['line']!=case['line'] or value['deployment_mode']!=mode
                    or value['independent_test_used'] is not False or value['new_training_transitions']!=0
                    or [g['condition'] for g in value['conditions']]!=expected):
                raise ValueError('Retain all predeclared models, modes and conditions')
            provenance.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            for group in value['conditions']:
                rs = group['records']; validate_records(rs,spec['layout_seed_start'][case['line']],100)
                records[case['line'],case['label'],mode,group['condition']] = rs
                summaries.append(dict(case=case['name'],line=case['line'],label=case['label'],mode=mode,
                    condition=group['condition'],**metrics(rs)))
                all_episodes.extend(dict(case=case['name'],line=case['line'],mode=mode,condition=group['condition'],
                    **{k:v for k,v in row.items() if k!='trajectory'}) for row in rs)
    if len(summaries)!=42 or len(all_episodes)!=4200:
        raise ValueError('Do not substitute or omit any policy/geometry episode')
    mode_changes = []; against_metric = []; checks = []
    def check(name,case,mode,condition,observed,limit,relation,baseline):
        checks.append(dict(criterion=name,case=case,mode=mode,condition=condition,baseline=baseline,
            value=observed,limit=limit,relation=relation,satisfied=observed<=limit if relation=='<=' else observed>=limit,
            independent_confirmation=False))
    for row in summaries:
        key = (row['line'],row['label'],row['mode'],row['condition'])
        if row['mode']=='zero_noise':
            baseline = records[row['line'],row['label'],'sampled',row['condition']]
            mode_changes.append(dict(case=row['case'],line=row['line'],condition=row['condition'],
                candidate_mode='zero_noise',baseline_mode='sampled',**comparison(records[key],baseline)))
        if row['label']!='Metric':
            baseline = records[row['line'],'Metric',row['mode'],row['condition']]
            against_metric.append(dict(candidate=row['case'],mode=row['mode'],condition=row['condition'],
                baseline='Metric_same_mode',**comparison(records[key],baseline)))
            ref = metrics(baseline)
            for field,relation in (('safe_successes','>='),('violations','<='),('timeouts','<='),('stall_count','<=')):
                check(field+'_no_worse',row['case'],row['mode'],row['condition'],row[field],ref[field],relation,'Metric_same_mode')
            check('return_drop_at_most1percent',row['case'],row['mode'],row['condition'],row['mean_return'],
                  ref['mean_return']-.01*abs(ref['mean_return']),'>=','Metric_same_mode')
        if row['line']=='piper_reach' and row['label']!='Metric' and row['condition'] in ('radius_0575','radius_060'):
            threshold = 93 if row['condition']=='radius_0575' else 90
            check('absolute_reach_success',row['case'],row['mode'],row['condition'],row['safe_successes'],threshold,'>=','user_target')
    args.output.mkdir(parents=True,exist_ok=False)
    for name,rows in (('summary.csv',summaries),('episodes.csv',all_episodes),('paired_mode_changes.csv',mode_changes),
                      ('paired_against_metric_same_mode.csv',against_metric),('development_criteria.csv',checks)):
        output(args.output/name,rows)
    (args.output/'receipt.json').write_text(json.dumps(dict(status='complete_deployment_mode_DEV_review',
        cases=6,modes=2,conditions=42,episodes=4200,provenance=provenance,
        native_receipt_sha256=hashlib.sha256((args.runs/'receipt.json').read_bytes()).hexdigest(),
        training_seed_count=1,new_training_steps=0,new_simulation_controls=0,independent_test_used=False,
        winner_confirmation=False,training_risk_reduction_demonstrated=False,
        caution='Zero noise is a separate deployment policy,not evidence of stochastic-policy or training-risk improvement'),indent=2)+'\n')


if __name__=='__main__': main()

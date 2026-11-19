"""Compact full3400-episode source records; no new physics, selection or training."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.paper_crax_payload.compare_piper_deployment_modes_v12 import comparison,metrics,validate_records
from scripts.paper_crax_payload.summarize_simulation_v12 import output

PLAN = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_panda_safereach_deployment_modes.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args = p.parse_args();spec = json.loads(PLAN.read_text())
    receipt = json.loads((args.runs/'receipt.json').read_text())
    manifest = json.loads((args.runs/'run_config.json').read_text())
    if (receipt['status']!='completed' or receipt['source_commit']!='f54300e1'
            or receipt['evaluated_episodes']!=3400 or receipt['conditions']!=34
            or receipt['new_training_steps']!=0 or receipt['optimizer_updates']!=0
            or receipt['independent_test_used'] is not False or manifest['protocol']!=spec
            or manifest['source_commit']!='f54300e1' or manifest['host']!='CTAlab'):
        raise ValueError('Use the entire registered native3400-episode DEV campaign')
    summaries=[];episodes=[];records={};provenance=[]
    for case in spec['cases']:
        plan=json.loads(Path(case['plan']).read_text())
        conditions=(plan['lines'][case['line']]['conditions'] if case['line']=='panda_payload'
                    else plan['evaluation']['conditions'])
        for mode in spec['deployment_modes']:
            path=args.runs/(case['name']+'__'+mode+'.json');raw=json.loads(path.read_text())
            if (raw['case']!=case['name'] or raw['line']!=case['line'] or raw['deployment_mode']!=mode
                    or raw['new_training_steps']!=0 or raw['independent_test_used'] is not False
                    or [g['condition'] for g in raw['conditions']]!=[c['name'] for c in conditions]):
                raise ValueError('Preserve exact native model/mode/geometry identity')
            provenance.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            for group in raw['conditions']:
                rs=group['records'];validate_records(rs,spec['layout_seed_start'][case['line']],100)
                records[case['line'],case['label'],mode,group['condition']]=rs
                summaries.append(dict(case=case['name'],line=case['line'],label=case['label'],mode=mode,
                    condition=group['condition'],**metrics(rs)))
                episodes.extend(dict(case=case['name'],line=case['line'],mode=mode,condition=group['condition'],
                    **{k:v for k,v in row.items() if k not in ('trajectory','trajectory_tcp_xyz','control_details','layout')}) for row in rs)
    if len(summaries)!=34 or len(episodes)!=3400:raise ValueError('Every registered episode required')
    mode_changes=[];comparisons=[];guard=[]
    for row in summaries:
        key=(row['line'],row['label'],row['mode'],row['condition'])
        if row['mode']=='zero_noise':
            sampled=records[row['line'],row['label'],'sampled',row['condition']]
            mode_changes.append(dict(case=row['case'],condition=row['condition'],**comparison(records[key],sampled)))
            if row['line']=='panda_payload' and row['label']=='P5':
                ref=metrics(sampled)
                guard.append(dict(condition=row['condition'],zero_noise_successes=row['safe_successes'],
                    sampled_successes=ref['safe_successes'],success_drop_no_more2pp=row['safe_successes']>=ref['safe_successes']-2,
                    zero_noise_timeouts=row['timeouts'],sampled_timeouts=ref['timeouts'],
                    zero_noise_stalls=row['stall_count'],sampled_stalls=ref['stall_count'],
                    success_at_least95percent=row['safe_successes']>=95,independent_confirmation=False))
        if row['label']=='P5':
            for label in ('Metric','Iso') if row['line']=='safereach' else ('Metric',):
                other=records[row['line'],label,row['mode'],row['condition']]
                comparisons.append(dict(case=row['case'],mode=row['mode'],condition=row['condition'],
                    baseline=label+'_same_mode',**comparison(records[key],other)))
    args.output.mkdir(parents=True,exist_ok=False)
    for name,values in (('summary.csv',summaries),('episodes.csv',episodes),('paired_mode_changes.csv',mode_changes),
                        ('paired_against_baselines_same_mode.csv',comparisons),('panda_development_regression.csv',guard)):
        output(args.output/name,values)
    (args.output/'receipt.json').write_text(json.dumps(dict(status='complete_cross_task_deployment_DEV_review',
        cases=5,modes=2,conditions=34,episodes=3400,provenance=provenance,
        native_receipt_sha256=hashlib.sha256((args.runs/'receipt.json').read_bytes()).hexdigest(),
        new_training_steps=0,new_simulation_controls=0,independent_test_used=False,
        training_seed_count_per_task=1,winner_confirmation=False,
        panda_guard_scope='fixed policy deployment only;not final algorithm multi seed regression',
        training_risk_reduction_demonstrated=False),indent=2)+'\n')


if __name__=='__main__':main()

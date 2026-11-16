"""Reuse actual parent/development measurements; never evaluate or select."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from scripts.paper_crax_payload.plot_learning_curves import snapshot, payload_figure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    if args.output.exists():
        raise FileExistsError('Preserve existing prefixes; use a new snapshot name')
    rows=[];sources={}
    for job in plan['jobs']:
        run=args.sources/job['run']
        parent=run/'parent_evaluation.json'
        if not parent.exists():
            continue
        paths=[parent]+sorted(run.glob('evaluation_*.json'))
        parent_value=json.loads(parent.read_text())
        for path in paths:
            data=path.read_bytes();value=json.loads(data)
            sources[str(path)]=hashlib.sha256(data).hexdigest()
            actual=value['checkpoint_steps'];new_steps=actual-job['initial_steps']
            if (value['method']!=job['method'] or value['training_seed']!=plan['training_seed'] or
                    new_steps not in [0,*plan['new_candidate_steps']]):
                raise ValueError('Do not mix native identities/ages or invent a baseline')
            for c in value['conditions']:
                seeds=[r['layout_seed'] for r in c['records']]
                if seeds!=list(range(450000,450000+c['episodes'])):
                    raise ValueError('Require the original fixed development layout stream')
                rows.append(dict(trial_id=job['label'],method=job['method'],training_seed=plan['training_seed'],
                    steps=new_steps,full_checkpoint_steps=actual,phase='reference' if new_steps==0 else 'development',
                    condition=c['condition'],evaluation_seed=450000,episodes=c['episodes'],
                    safe_success=c['safe_success'],violation=c['violation'],
                    time_to_goal_capped_seconds=c['time_to_goal_capped_seconds'],
                    new_training_violations=value['training_violation_count']-parent_value['training_violation_count'],
                    historical_parent_reference=new_steps==0,final_independent_test=False))
    if not rows:
        raise ValueError('No actual recorded parent/development data')
    args.output.mkdir(parents=True)
    csv_path=args.output/'development_curve.csv'
    with csv_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    parsed,receipt=snapshot(csv_path,['safe_success','violation'])
    payload_figure(parsed,receipt,args.output/'development_curve.png')
    (args.output/'receipt.json').write_text(json.dumps(dict(status='development_prefix_only',
        source_sha256=sources,csv_snapshot=receipt,all_recorded_points_retained=True,
        zero_points_are_real_historical_parent_evaluations=True,
        x_axis_is_new_transitions_not_unequal_parent_age=True,
        missing_checkpoints_filled=False,new_training_transitions=0,new_simulation_controls=0,
        new_evaluation_episodes=0,independent_test_claim=False,alternative_route_legality_claim=False),indent=2)+'\n')


if __name__=='__main__':
    main()

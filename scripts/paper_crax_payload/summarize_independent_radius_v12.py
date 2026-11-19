"""Both complete native100K radius forks;no best-seed/node selection or new physics."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

from scripts.paper_crax_payload.summarize_simulation_v12 import output


PROJECT=Path(__file__).resolve().parents[2]
PLAN=PROJECT/'configs/paper_crax_payload/simulation_v12_reach_independent_radius.json'
FIELDS=('training_violation_count','training_safe_successes','training_safe_timeouts','training_episodes')
ROW_FIELDS=('training_violation_count','training_safe_successes','training_safe_timeouts','episodes')


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def stage_counts(parent,receipt,rows,seed):
    if (receipt['status']!='completed' or receipt['training_steps']!=400000
            or receipt['initial_steps']!=300000 or receipt['newly_collected_transitions']!=100000
            or receipt['training_seed']!=seed or receipt['engineering']
            or receipt['training_source_commit']!='10ba914c'
            or parent['status']!='completed' or parent['training_steps']!=300000
            or parent['training_seed']!=seed or parent['engineering']
            or [r['steps'] for r in rows]!=list(range(301000,400001,1000))):
        raise ValueError('Both complete scientific100K segments,not relabeled cumulative400K or partial nodes')
    previous={k:parent[k] for k in FIELDS}
    for row in rows:
        current={k:row[rk] for k,rk in zip(FIELDS,ROW_FIELDS)}
        if (any(current[k]<previous[k] for k in FIELDS)
                or current['training_episodes']!=sum(current[k] for k in FIELDS[:3])
                or any(isinstance(v,float) and not math.isfinite(v) for v in row.values())):
            raise ValueError('Preserve every episode class and monotone finite native counters')
        previous=current
    if any(receipt[k]!=previous[k] for k in FIELDS):
        raise ValueError('Native endpoint receipt and last progress counters disagree')
    differences={k:receipt[k]-parent[k] for k in FIELDS}
    if (receipt['new_training_violations']!=differences['training_violation_count']
            or receipt['new_training_safe_successes']!=differences['training_safe_successes']):
        raise ValueError('Never report inherited parent violations/successes as newly collected')
    return dict(seed=seed,parent_steps=300000,total_steps=400000,new_training_steps=100000,
        **{'new_'+k:v for k,v in differences.items()},
        **{'cumulative_'+k:receipt[k] for k in FIELDS})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();plan=json.loads(PLAN.read_text())
    parent_plan=json.loads((PROJECT/plan['parent_plan']).read_text())
    base_path=PROJECT/parent_plan['base_config']
    if sha256(base_path)!=parent_plan['base_config_sha256']:
        raise ValueError('The registered original scientific recipe changed')
    training,endpoints,nodes,episodes,provenance=[],[],[],[],[]
    for entry in plan['parents']:
        seed=entry['seed'];directory=args.runs/f'Reach-EA-{seed}';parent_dir=args.parents/f'Reach-EA-{seed}'
        parent=json.loads((parent_dir/'receipt.json').read_text())
        parent_manifest=json.loads((parent_dir/'run_config.json').read_text())
        manifest=json.loads((directory/'run_config.json').read_text())
        receipt=json.loads((directory/'receipt.json').read_text())
        expected=json.loads(base_path.read_text());expected['development_seed']=seed
        expected['evaluation']['development_layout_seed_start']=parent_plan['development_layout_seed_start']
        expected['evaluation']['action_sampling_seed']=parent_plan['action_sampling_seed']
        parent_config=deepcopy(expected);expected['training_radius_values']=plan['training_radius_values']
        adaptation=manifest['geometry_adaptation']
        if (parent_manifest['configuration']!=parent_config or parent_manifest['source_commit']!='22d4966f'
                or manifest['configuration']!=expected or manifest['source_commit']!='10ba914c'
                or manifest['training_seed']!=seed or manifest['initial_steps']!=300000
                or manifest['total_transition_budget']!=400000 or manifest['engineering']
                or manifest['initialization']!='Piper_Reach_finite_radius_adaptation'
                or manifest['host']!='CTAlab' or manifest['task']!='Reach' or manifest['method']!='EA'
                or adaptation['parent_sha256']!=entry['checkpoint_sha256']
                or adaptation['parent_training_steps']!=300000
                or adaptation['radius_values']!=plan['training_radius_values']
                or receipt['geometry_adaptation']!=adaptation):
            raise ValueError('Require actual matching independent parent and original single-radius-field change')
        rows=[json.loads(s) for s in (directory/'progress.jsonl').read_text().splitlines()]
        result=stage_counts(parent,receipt,rows,seed)
        checkpoint=directory/'full_400000.pt'
        checkpoint_receipt=json.loads((directory/'training_receipt_400000.json').read_text())
        if (checkpoint_receipt['status']!='completed_native_checkpoint'
                or checkpoint_receipt['training_seed']!=seed or checkpoint_receipt['training_steps']!=400000
                or any(checkpoint_receipt[k]!=receipt[k] for k in FIELDS)
                or [p.name for p in directory.glob('evaluation_*.json')]!=['evaluation_400000.json']):
            raise ValueError('Fixed native400K checkpoint and sole registered DEV endpoint required')
        digest=sha256(checkpoint)
        result.update(parent_checkpoint_sha256=entry['checkpoint_sha256'],checkpoint_sha256=digest,
            checkpoint_bytes=checkpoint.stat().st_size,saved_cuda_rng_index=0,
            output_path=str(directory),downstream_training_started=False,
            scientific_scope='EA radius coverage;not P5 vs Metric confirmation')
        endpoints.append(result)
        training.extend(dict(seed=seed,new_steps=r['steps']-300000,**r) for r in rows)
        raw=json.loads((directory/'evaluation_400000.json').read_text())
        conditions=expected['evaluation']['conditions'];start=expected['evaluation']['development_layout_seed_start']
        if (raw['checkpoint_steps']!=400000 or raw['training_seed']!=seed or not raw['fixed_endpoint']
                or raw['new_training_transitions']!=0 or raw['adaptation_updates']!=0
                or not raw['training_rng_preserved']
                or [c['condition'] for c in raw['conditions']]!=[c['name'] for c in conditions]):
            raise ValueError('Every registered frozen400K DEV condition required')
        for group in raw['conditions']:
            rs=group['records']
            if (len(rs)!=30 or [r['seed'] for r in rs]!=list(range(start,start+30))
                    or any(r['safe_success'] and r['violation'] for r in rs)):
                raise ValueError('Never omit or replace failed DEV episodes')
            counts=dict(safe_success_count=sum(r['safe_success'] for r in rs),
                violation_count=sum(r['violation'] for r in rs),
                safe_timeout_count=sum(not r['safe_success'] and not r['violation'] for r in rs))
            if any(group[k]!=v for k,v in counts.items()):
                raise ValueError('Actual DEV episodes and native totals disagree')
            nodes.append(dict(seed=seed,condition=group['condition'],episodes=30,**counts,
                mean_return=statistics.mean(r['episode_return'] for r in rs),
                mean_capped_seconds=statistics.mean(r['time_to_goal_capped_seconds'] for r in rs),
                geometry_scope='trained_radius' if group['condition']=='new_geometry' else 'nominal_or_execution',
                independent_test_used=False))
            episodes.extend(dict(training_seed=seed,condition=group['condition'],
                **{k:v for k,v in r.items() if k!='trajectory'}) for r in rs)
        provenance.append(dict(seed=seed,checkpoint_sha256=digest,
            parent_checkpoint_sha256=entry['checkpoint_sha256'],source_commit=manifest['source_commit'],
            receipt_sha256=sha256(directory/'receipt.json'),parent_receipt_sha256=sha256(parent_dir/'receipt.json'),
            native_dev_sha256=sha256(directory/'evaluation_400000.json'),progress_sha256=sha256(directory/'progress.jsonl')))
    aggregates=[]
    for condition in conditions:
        subset=[r for r in nodes if r['condition']==condition['name']]
        rates=[r['safe_success_count']/30 for r in subset]
        if len(rates)!=2:
            raise ValueError('Both seeds required equally')
        aggregates.append(dict(condition=condition['name'],training_seeds=2,episodes_per_seed=30,
            success_mean=statistics.mean(rates),success_best=max(rates),success_worst=min(rates),
            success_sd_sample=statistics.stdev(rates),
            violation_mean=statistics.mean(r['violation_count']/30 for r in subset),
            timeout_mean=statistics.mean(r['safe_timeout_count']/30 for r in subset),independent_test_used=False))
    args.output.mkdir(parents=True,exist_ok=False)
    for filename,values in [('training.csv',training),('endpoints.csv',endpoints),('development_nodes.csv',nodes),
                            ('development_episodes.csv',episodes),('equal_seed_development.csv',aggregates)]:
        output(args.output/filename,values)
    (args.output/'receipt.json').write_text(json.dumps(dict(status='complete_independent_radius200K_summary',
        training_seeds=plan['training_seeds'],new_completed_training_transitions=200000,
        cumulative_lineage_transitions=800000,completed_runs=2,development_conditions=6,development_episodes=180,
        provenance=provenance,new_training_steps=0,new_simulation_controls=0,optimizer_steps=0,
        independent_test_used=False,winner_confirmation=False,downstream_training_started=False,
        training_radius_055_is_not_unseen_geometry=True,automatic_retry=False),indent=2)+'\n')


if __name__=='__main__':
    main()

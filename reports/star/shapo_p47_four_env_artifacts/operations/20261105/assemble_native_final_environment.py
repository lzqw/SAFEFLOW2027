#!/usr/bin/env python3
"""Summarize all five fixed-seed native final evaluations, locally only.

No policy rollout, tensor load, checkpoint/seed selection or final audit.
Refuse missing or conflicting inputs before creating any output directory.
"""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

BASE=Path(__file__).resolve().parent
PARENT=Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/partial_milestones')
OLD=PARENT/'terminal_native_receipts_available8_20261024'
NATIVE=PARENT/'native_terminal_receipts'
COMMIT='62ca7399357c10645a955dcbead7ed2153a82c69'
ENVS={'point-goal':('SafetyPointGoal1-v0',10000000),'point-button':('SafetyPointButton1-v0',10000000),
      'ant':('Ant-v4',5000000),'walker2d':('Walker2d-v4',5000000)}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(job,steps):
    found=[]
    for manifest in [OLD/'manifest.json',NATIVE/job/'manifest.json']:
        if not manifest.is_file():continue
        m=json.loads(manifest.read_text());assert m['scientific_source_commit']==COMMIT
        records=m.get('records',[m.get('source',{})])
        for record in records:
            if record.get('job')!=job:continue
            selected={};proof=[]
            for entry in record['files']:
                rel=Path(entry.get('local_relative_path',entry['path']))
                assert not rel.is_absolute() and '..' not in rel.parts
                p=manifest.parent/rel
                p.resolve().relative_to(PARENT.resolve())
                assert p.is_file() and not p.is_symlink() and p.stat().st_size==entry['bytes']
                assert digest(p)==entry['sha256']
                selected[p.name]=p
                proof.append(dict(relative_path=str(p.relative_to(PARENT)),sha256=entry['sha256']))
            required={'receipt.json','training_summary.json',f'evaluation_step_{steps}.json'}
            assert set(selected)==required
            found.append((selected,dict(copy_manifest=str(manifest.relative_to(PARENT)),
                copy_manifest_sha256=digest(manifest),native_input_files=proof)))
    if not found:raise FileNotFoundError('Await the native terminal copy for '+job)
    for selected,_ in found[1:]:
        assert all(digest(selected[k])==digest(found[0][0][k]) for k in selected), 'Conflicting copies; never select by metric'
    return found[0]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--environment',choices=ENVS,required=True);args=parser.parse_args()
    slug=args.environment;environment,steps=ENVS[slug];out=PARENT/(slug+'_native_final_all5')
    assert not out.exists(),'Preserve the existing immutable five-seed assembly'
    rows=[];evaluations=[]
    required=['mean_return','mean_cost','cost_cvar_20','training_cumulative_cost',
              'success_rate' if slug.startswith('point-') else 'failure_rate']
    geometry={'high_risk_normal_tangent_variance_ratio':'normal_to_tangent_variance_ratio',
              'high_risk_dangerous_outward_velocity':'dangerous_outward_velocity',
              'high_risk_reliability':'reliability','high_risk_safety_weight':'safety_weight'}
    fields=required+list(geometry)
    for seed in range(2027,2032):
        job=f'{slug}-{seed}';selected,proof=inputs(job,steps)
        r=json.loads(selected['receipt.json'].read_text());s=json.loads(selected['training_summary.json'].read_text())
        epath=selected[f'evaluation_step_{steps}.json'];e=json.loads(epath.read_text())
        assert r['status']=='passed' and r['environment']==environment and r['seed']==seed and r['steps']==steps
        assert r['source_commit']==COMMIT and r['source_dirty'] is False
        assert s['status']=='completed' and s['env_steps']==steps and s['engineering_hard_stop_reason'] is None
        assert e['milestone']==steps and e['episodes']==100 and len(e['returns'])==len(e['costs'])==100
        assert e['base_seeds']==list(range(32027,32032)) and e['episodes_per_base_seed']==20
        row=dict(job=job,environment=environment,training_seed=seed,milestone_steps=steps,input_provenance=proof)
        for field in required:
            value=e[field];assert isinstance(value,(int,float)) and math.isfinite(value), (job,field)
            row[field]=value
        missing=[];g=e.get('direction_geometry_high_risk_top_20_percent',{})
        for field,key in geometry.items():
            value=g.get(key)
            if not isinstance(value,(int,float)) or not math.isfinite(value):value=None;missing.append(field)
            row[field]=value
        row['missing_or_nonfinite_optional_diagnostics']=missing
        row['native_evaluation']=dict(relative_path=f'native_evaluations/{job}/evaluation_step_{steps}.json',sha256=digest(epath))
        rows.append(row);evaluations.append(epath)
    assert len(rows)==5 and [r['training_seed'] for r in rows]==list(range(2027,2032))
    groups={}
    for field in fields:
        values=[r[field] for r in rows if r[field] is not None]
        groups[field]=dict(n_valid_training_seeds=len(values),mean=statistics.mean(values) if values else None,
            sample_sd=statistics.stdev(values) if len(values)>1 else None,
            minimum=min(values) if values else None,maximum=max(values) if values else None)
    summary=dict(kind='descriptive_native_final_budget_one_environment_not_final_matrix_audit',
        assembled_at_utc=datetime.now(timezone.utc).isoformat(),scientific_source_commit=COMMIT,
        environment=environment,n_training_seeds=5,milestone_steps=steps,evaluation_episodes_per_training_seed=100,
        statistical_unit='training_seed_not_individual_episode',descriptive_metrics=groups,records=rows,
        helper_sha256=digest(Path(__file__)),additional_evaluation_rollouts=0,model_or_tensor_loads=0,
        remote_operations=0,new_scientific_cells=0,scientific_parameters_changed=False,
        matched_control_comparison_performed=False,inferential_test_performed=False,
        full_checkpoint_provenance_audit_executed=False,strict_final_matrix_audit_executed=False,
        full_scientific_objective_complete=False,
        caveats=['All five prespecified seeds at the final budget are required; no best-checkpoint or metric selection.',
                 'Means and sample SDs summarize five training seeds, not500 independent training replicates.',
                 'CVaR summaries average per-training-seed CVaR-20 values; episode pools are not merged.',
                 'Training cumulative costs are accepted-lineage counters, not total physical recovery work.',
                 'Optional diagnostic omissions are reported; no training seed is dropped.',
                 'Locomotion failure rate is not Point-task success rate.',
                 'Native receipts and JSON hashes do not establish the unchanged final scientific gates.',
                 'Known original-parent/path gaps remain unresolved; no matched-control or causal superiority claim is made.'])
    # Only now, after all five native inputs are present and coherent, write output.
    out.mkdir(parents=True)
    for row,source in zip(rows,evaluations):
        target=out/row['native_evaluation']['relative_path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);assert digest(target)==row['native_evaluation']['sha256']
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    columns=['job','environment','training_seed','milestone_steps']+fields
    with (out/'per_training_seed_metrics.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in rows:writer.writerow({k:row.get(k) for k in columns})
    print(json.dumps(dict(output=str(out),environment=environment,training_seeds=5,descriptive_metrics=groups),indent=2))


if __name__=='__main__':main()

"""Prepare exact per-seed150K plans only after both real400K parents complete.

No torch, model loading, policy action, GPU allocation, or training launch.
This compiler never invents a missing parent SHA or substitutes the old41027.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


PROJECT=Path(__file__).resolve().parents[2]
TEMPLATE=PROJECT/'configs/paper_crax_payload/simulation_v12_reach_crossseed_template.json'
RADIUS_PLAN=PROJECT/'configs/paper_crax_payload/simulation_v12_reach_independent_radius.json'


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def bind(template,summary,endpoints,seed,destination_parent):
    if template!=json.loads(TEMPLATE.read_text()) or seed not in template['training_seeds']:
        raise ValueError('Only the registered two-seed fixed three-arm150K recipe')
    if (summary.get('status')!='complete_independent_radius200K_summary'
            or summary.get('completed_runs')!=2 or summary.get('new_completed_training_transitions')!=200000
            or summary.get('training_seeds')!=[41028,41029] or summary.get('development_episodes')!=180
            or summary.get('independent_test_used') is not False
            or summary.get('downstream_training_started') is not False
            or sorted(r['seed'] for r in endpoints)!=[41028,41029]):
        raise ValueError('Both actual independent400K native endpoints must have completed;never best-only')
    radius=json.loads(RADIUS_PLAN.read_text())
    for row in endpoints:
        original=next(r for r in radius['parents'] if r['seed']==row['seed'])
        digest=row['checkpoint_sha256']
        if (row['parent_steps']!=300000 or row['total_steps']!=400000 or row['new_training_steps']!=100000
                or row['saved_cuda_rng_index']!=0 or row['downstream_training_started'] is not False
                or row['parent_checkpoint_sha256']!=original['checkpoint_sha256']
                or not isinstance(digest,str) or len(digest)!=64
                or any(c not in '0123456789abcdef' for c in digest)
                or digest==original['checkpoint_sha256']):
            raise ValueError('Need this seed actual400K digest and original lineage,RNG0,budget')
        evidence=next((r for r in summary['provenance'] if r['seed']==row['seed']),{})
        if evidence.get('checkpoint_sha256')!=digest or evidence.get('source_commit')!='10ba914c':
            raise ValueError('Completed summary provenance and endpoint identity disagree')
    row=next(r for r in endpoints if r['seed']==seed)
    plan=deepcopy(template)
    for key in ('training_seeds','parent_steps','total_new_training_steps','development_layout_seed_start',
                'action_sampling_seed','conditions','parent_rng_cuda_index','training_radius_values'):
        plan.pop(key)
    plan['compiled_training_seed']=seed
    plan['new_training_steps_this_seed']=450000
    plan['template_sha256']=sha256(TEMPLATE)
    plan['lines']={'piper_reach':dict(robot='Piper',task='Reach',seed=seed,parent_steps=400000,
        parent_format='paper_native_piper_sim_v1_full',parent_path=str(destination_parent),
        parent_sha256=row['checkpoint_sha256'],parent_source_commit='10ba914c',parent_rng_cuda_index=0,
        training_radius_values=[.045,.05,.055],
        development_layout_seed_start=template['development_layout_seed_start'],
        action_sampling_seed=template['action_sampling_seed'],conditions=deepcopy(template['conditions']))}
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--completed-analysis',type=Path,required=True)
    p.add_argument('--parents',type=Path,required=True,help='Physical source directories containing both actual400K checkpoints')
    p.add_argument('--destination-parent-root',type=Path,required=True,help='Declared target data-disk input directory;no copy or launch')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();template=json.loads(TEMPLATE.read_text())
    summary=json.loads((args.completed_analysis/'receipt.json').read_text())
    import csv
    rows=list(csv.DictReader((args.completed_analysis/'endpoints.csv').read_text().splitlines()))
    for row in rows:
        for key in ('seed','parent_steps','total_steps','new_training_steps','saved_cuda_rng_index'):
            row[key]=int(row[key])
        if row['downstream_training_started']!='False':
            raise ValueError('Compiler cannot reset an already launched downstream lineage')
        row['downstream_training_started']=False
    plans,provenance={},[]
    # Bind and verify every parent BEFORE creating any resolved plan directory.
    for seed in template['training_seeds']:
        directory=args.parents/f'Reach-EA-{seed}'
        checkpoint=directory/'full_400000.pt'
        destination=args.destination_parent_root/f'Reach-EA-{seed}'/'full_400000.pt'
        plan=bind(template,summary,rows,seed,destination)
        receipt=json.loads((directory/'receipt.json').read_text())
        manifest=json.loads((directory/'run_config.json').read_text())
        if (sha256(checkpoint)!=plan['lines']['piper_reach']['parent_sha256']
                or receipt['status']!='completed' or receipt['training_steps']!=400000
                or receipt['newly_collected_transitions']!=100000 or receipt['training_seed']!=seed
                or receipt['engineering'] or manifest['source_commit']!='10ba914c'
                or manifest['training_seed']!=seed or manifest['configuration']['development_seed']!=seed
                or manifest['configuration']['training_radius_values']!=[.045,.05,.055]):
            raise ValueError('Physical fixed400K parents and native metadata must match completed summary')
        plans[seed]=plan
        provenance.append(dict(seed=seed,checkpoint_sha256=sha256(checkpoint),
            receipt_sha256=sha256(directory/'receipt.json'),manifest_sha256=sha256(directory/'run_config.json')))
    args.output.mkdir(parents=True,exist_ok=False)
    for seed,plan in plans.items():
        (args.output/f'reach-three-arm150K-{seed}.json').write_text(json.dumps(plan,indent=2)+'\n')
    (args.output/'preparation_receipt.json').write_text(json.dumps(dict(
        status='prepared_only_not_training_authorization_or_completed_confirmation',
        template_sha256=sha256(TEMPLATE),parent_analysis_sha256=sha256(args.completed_analysis/'receipt.json'),
        provenance=provenance,training_seeds=[41028,41029],proposed_training_steps=900000,
        new_training_steps=0,training_launches=0,new_simulation_controls=0,independent_test_used=False,
        same_mode_baselines_required=True,requires_separate_native_runner_and_resource_admission=True),indent=2)+'\n')


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Copy the eight already cached terminal JSON receipts to the local host.

No evaluation, model load, checkpoint selection, provenance audit or mutation
on a training host. These are native training receipts, not final gate passes.
"""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

BASE=Path(__file__).resolve().parent
OUT=Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/partial_milestones/terminal_native_receipts_available8_20261024')
ROOT='/storage_0/SAFEFLOW2027_lzqw_uid1007/terminal_matrix_20261024/runs/shapo_p47_four_env'
COMMIT='62ca7399357c10645a955dcbead7ed2153a82c69'


def main():
    assert not OUT.exists(),'Do not overwrite an existing immutable receipt snapshot'
    migration=json.loads((BASE/'terminal_result_cache_migration_20261024_state.json').read_text())
    assert migration['phase']=='completed_new_collector_started'
    jobs=sorted(migration['terminal_jobs']);assert len(jobs)==8
    remote=f'''
import base64,hashlib,json
from pathlib import Path
root=Path({ROOT!r});out=[]
for job in {jobs!r}:
 d=root/'runs'/job
 assert d.is_dir() and not d.is_symlink()
 r=json.loads((d/'receipt.json').read_text())
 assert r['status']=='passed' and r['steps']==5000000 and r['source_commit']=={COMMIT!r} and r['source_dirty'] is False
 assert r['seed']==int(job.rsplit('-',1)[1])
 edge=json.loads((d/'artifact_relocation.json').read_text())
 rel=Path(r['log_dir']).relative_to(Path(edge['original_container']))
 assert rel.parts[0]=='training' and '..' not in rel.parts
 log=d/rel
 summary=json.loads((log/'artifacts/training_summary.json').read_text())
 assert summary['status']=='completed' and summary['env_steps']==5000000 and summary['engineering_hard_stop_reason'] is None
 evaluation=log/'artifacts/evaluation_step_5000000.json'
 e=json.loads(evaluation.read_text())
 assert e['milestone']==5000000 and e['episodes']==100
 assert e['base_seeds']==list(range(32027,32032)) and e['episodes_per_base_seed']==20
 selected=[d/'receipt.json',log/'artifacts/training_summary.json',evaluation]
 files=[]
 for p in selected:
  assert p.is_file() and not p.is_symlink()
  data=p.read_bytes();files.append(dict(path=str(p.relative_to(d)),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),base64=base64.b64encode(data).decode('ascii')))
 out.append(dict(job=job,source_container=str(d),files=files))
print(json.dumps(out))
'''
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','tjucta-a100',
                           'nice -n 19 python3 -c '+shlex.quote(remote)],check=True,capture_output=True,text=True,timeout=60)
    rows=json.loads(result.stdout)
    assert sorted(r['job'] for r in rows)==jobs
    OUT.mkdir(parents=True)
    for row in rows:
        for item in row['files']:
            rel=Path(item['path']);assert not rel.is_absolute() and '..' not in rel.parts
            data=base64.b64decode(item.pop('base64'),validate=True)
            assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
            path=OUT/row['job']/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            item['local_relative_path']=str(path.relative_to(OUT))
    manifest=dict(kind='existing_native_terminal_receipt_copy_not_final_scientific_audit',
        created_at_utc=datetime.now(timezone.utc).isoformat(),scientific_source_commit=COMMIT,
        completed_training_cells=8,full_matrix_cells=20,files_copied=sum(len(r['files']) for r in rows),
        selection_rule='All eight terminal cells already present in the verified cache migration; no metric selection',
        additional_evaluation_rollouts=0,model_or_tensor_loads=0,training_processes_signalled=0,
        remote_files_modified=0,scientific_parameters_changed=False,matched_control_comparison_performed=False,
        strict_final_matrix_audit_executed=False,full_scientific_objective_complete=False,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=rows,
        caveats=['Only the eight cells already terminal at cache migration are included; not all five seeds of either environment.',
                 'Native receipt status is not the final60/80/60/120 scientific gate result.',
                 'Known original-parent and multi-hop path limitations remain unresolved.',
                 'No performance superiority, significance or causal claim is made from this incomplete matrix.'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(OUT),training_cells=len(rows),native_json_files=manifest['files_copied'],additional_rollouts=0)))


if __name__=='__main__':main()

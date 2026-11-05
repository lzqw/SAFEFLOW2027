#!/usr/bin/env python3
"""Copy one existing fixed-budget native terminal result; never evaluate it.

The source must have a passed native receipt, completed training summary and
the prespecified final evaluation. This is not the final matrix scientific audit.
"""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

BASE=Path(__file__).resolve().parent
OUT=Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/partial_milestones/native_terminal_receipts')
ROOTS={'4080':'/home/carla/SAFEFLOW2027_4080/runs/shapo_p47_four_env',
       'tjucta-a100':'/home/lzqw/SAFEFLOW2027_A100/runs/shapo_p47_four_env',
       '5080':'/home/autodrive/LZQW/SAFEFLOW2027/runs/shapo_p47_four_env'}
CACHE='/storage_0/SAFEFLOW2027_lzqw_uid1007/terminal_matrix_20261024/runs/shapo_p47_four_env'
ENVS={'point-goal':('SafetyPointGoal1-v0',10000000),'point-button':('SafetyPointButton1-v0',10000000),
      'ant':('Ant-v4',5000000),'walker2d':('Walker2d-v4',5000000)}
COMMIT='62ca7399357c10645a955dcbead7ed2153a82c69'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',choices=ROOTS,required=True)
    parser.add_argument('--job',required=True);parser.add_argument('--from-cache',action='store_true');args=parser.parse_args()
    slug,seed=args.job.rsplit('-',1);assert slug in ENVS and int(seed) in range(2027,2032)
    env,steps=ENVS[slug]
    assert not args.from_cache or args.host=='tjucta-a100'
    root=CACHE if args.from_cache else ROOTS[args.host]
    target=OUT/args.job
    assert not target.exists(),'Preserve the existing immutable native receipt copy'
    remote=f'''
import base64,hashlib,json
from pathlib import Path
d=Path({root!r})/'runs'/{args.job!r};assert d.is_dir() and not d.is_symlink()
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  a=[v.decode(errors='replace') for v in p.read_bytes().split(b'\\0') if v]
  if any(v.endswith('scripts/star/train_star.py') for v in a) and '--config' in a:
   assert a[a.index('--config')+1]!=str(d/'resolved_formal_config.yaml'),'Training worker is still active'
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
r=json.loads((d/'receipt.json').read_text())
assert r['status']=='passed', 'Native receipt is not passed: '+str(r.get('status'))
assert r['steps']=={steps} and r['seed']=={int(seed)} and r['environment']=={env!r}
assert r['source_commit']=={COMMIT!r} and r['source_dirty'] is False
try:rel=Path(r['log_dir']).relative_to(d)
except ValueError:
 edge=json.loads((d/'artifact_relocation.json').read_text())
 rel=Path(r['log_dir']).relative_to(Path(edge['original_container']))
assert rel.parts[0]=='training' and '..' not in rel.parts
log=d/rel;summary_path=log/'artifacts/training_summary.json'
summary=json.loads(summary_path.read_text())
assert summary['status']=='completed' and summary['env_steps']=={steps} and summary['engineering_hard_stop_reason'] is None
evaluation=log/'artifacts/evaluation_step_{steps}.json';e=json.loads(evaluation.read_text())
assert e['milestone']=={steps} and e['episodes']==100 and e['base_seeds']==list(range(32027,32032)) and e['episodes_per_base_seed']==20
files=[]
for p in [d/'receipt.json',summary_path,evaluation]:
 assert p.is_file() and not p.is_symlink()
 data=p.read_bytes();files.append(dict(path=str(p.relative_to(d)),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),
  mtime_ns=p.stat().st_mtime_ns,base64=base64.b64encode(data).decode('ascii')))
print(json.dumps(dict(job={args.job!r},source_container=str(d),files=files)))
'''
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',args.host,
                           'nice -n 19 python3 -c '+shlex.quote(remote)],check=True,capture_output=True,text=True,timeout=60)
    row=json.loads(result.stdout);assert row['job']==args.job and len(row['files'])==3
    target.mkdir(parents=True)
    for item in row['files']:
        rel=Path(item['path']);assert not rel.is_absolute() and '..' not in rel.parts
        data=base64.b64decode(item.pop('base64'),validate=True)
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
        path=target/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    manifest=dict(kind='native_terminal_receipt_copy_not_final_scientific_audit',
        copied_at_utc=datetime.now(timezone.utc).isoformat(),source_host=args.host,scientific_source_commit=COMMIT,
        full_matrix_cells=20,terminal_training_budget_steps=steps,source=row,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection_rule='Prespecified final-budget evaluation of the named cell, never best-checkpoint selection',
        additional_evaluation_rollouts=0,model_or_tensor_loads=0,training_processes_signalled=0,
        remote_files_modified=0,scientific_parameters_changed=False,matched_control_comparison_performed=False,
        strict_final_matrix_audit_executed=False,full_scientific_objective_complete=False,
        caveat='A passed native receipt is not a final60/80/60/120 scientific-gate pass; original-parent and path limitations remain unresolved.')
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(target),job=args.job,full_budget_steps=steps,native_json_files=3,additional_rollouts=0)))


if __name__=='__main__':main()

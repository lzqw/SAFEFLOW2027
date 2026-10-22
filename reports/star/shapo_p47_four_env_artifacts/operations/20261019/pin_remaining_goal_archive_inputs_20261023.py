#!/usr/bin/env python3
"""Pin existing 4T raw manifests for the three stopped cells; no tensor loads."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

BASE=Path(__file__).resolve().parent
OUTPUT=BASE/'remaining_goal_archive_inputs_20261023.json'
ROOT='/home/carla/4T/SAFEFLOW2027_4080/scientific_evidence_archive'
COMMIT='62ca7399357c10645a955dcbead7ed2153a82c69'
JOBS=('point-goal-2029','point-goal-2028','point-goal-2030')


def main():
    assert not OUTPUT.exists(), 'Inspect pinned evidence instead of overwriting it'
    archived=json.loads((BASE/'scientific_evidence_archive_state.json').read_text())
    inventory=json.loads((BASE/'incident_5080_2334_20261022_checkpoint_inventory.json').read_text())
    rows=[]
    for job in JOBS:
        source=next(r for r in inventory['rows'] if r['job']==job)
        assert source['full_replay_rng_optimizers_present'] is True
        container='/home/autodrive/LZQW/SAFEFLOW2027/runs/shapo_p47_four_env/runs/'+job
        relative=str(Path(source['checkpoint']).relative_to(container))
        manifest=archived['archived']['5080/'+job]['manifest']
        assert Path(manifest).parent==Path(ROOT)/'manifests/5080'/job
        rows.append(dict(job=job,manifest=manifest,source_container=container,
            checkpoint_relative_path=relative,checkpoint_sha256=source['checkpoint_sha256'],
            counters_from_existing_incident_inventory=source['counters'],
            resume_steps=source['counters']['env_steps'],total_budget_steps=10000000,
            remaining_steps=10000000-source['counters']['env_steps'],
            conditional_repeated_steps_not_executed=source['known_repeated_steps_if_resumed'],
            conditional_extra_cost_not_executed=source['known_additional_training_cost_if_resumed']))
    remote='''
import hashlib,json,stat
from pathlib import Path
rows=ROWS
root=Path(ROOT)
for row in rows:
 p=Path(row['manifest']);assert p.is_file() and not p.is_symlink()
 raw=p.read_bytes();m=json.loads(raw)
 sha=hashlib.sha256(raw).hexdigest();assert p.name.split('-',1)[1]==sha+'.json'
 assert m['format']=='safeflow-raw-evidence-snapshot-v1'
 assert m['source_commit']==COMMIT and m['origin']=='5080'
 assert m['job']==row['job'] and m['source_container']==row['source_container']
 entries=[r for r in m['files'] if r['path']==row['checkpoint_relative_path']]
 assert len(entries)==1 and entries[0]['sha256']==row['checkpoint_sha256']
 for entry in m['files']:
  rel=Path(entry['path']);assert not rel.is_absolute() and '..' not in rel.parts
  blob=root/'blobs'/entry['sha256'];st=blob.lstat()
  assert stat.S_ISREG(st.st_mode) and st.st_size==entry['bytes']
 row.update(manifest_sha256=sha,manifest_created_at_ns=m['created_at_ns'],
  archived_raw_files=len(m['files']),archived_logical_bytes=sum(r['bytes'] for r in m['files']),
  checkpoint_blob=str(root/'blobs'/row['checkpoint_sha256']),checkpoint_bytes=entries[0]['bytes'],
  blob_presence_and_sizes_checked=True,fresh_blob_content_hash_performed=False,
  raw_scope='Recorded archive suffixes; not a newly inventoried complete filesystem capsule')
print(json.dumps(rows))
'''.replace('rows=ROWS','rows='+repr(rows)).replace('Path(ROOT)','Path('+repr(ROOT)+')').replace("==COMMIT","=="+repr(COMMIT))
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','4080',
        'taskset -c 16-27 nice -n 10 python3 -c '+shlex.quote(remote)],
        text=True,capture_output=True,check=True,timeout=60)
    rows=json.loads(result.stdout)
    assert tuple(r['job'] for r in rows)==JOBS
    assert [r['remaining_steps'] for r in rows]==sorted((r['remaining_steps'] for r in rows),reverse=True)
    record=dict(phase='existing_raw_archives_pinned_for_future_resource_admission',
        source_commit=COMMIT,rows=rows,priority_rule='Largest remaining fixed budget first; not reward/cost selection',
        observed_at_unix=time.time(),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_data_transfers=0,model_or_tensor_loads=0,additional_evaluation_rollouts=0,
        trainers_signalled=0,training_launched=0,training_authorized_by_this_receipt=False,
        new_scientific_cells=0,scientific_parameters_changed=False,
        historical_2027_parent_gaps_repaired=False,
        liveness_caveat='These cells were stopped in the07:23 snapshot. Recheck current hosts, placement, resources and exact input bytes before any launch.',
        archive_caveat='Existing immutable raw manifests with prior SHA-verified ingestion; this check reads manifest hashes and blob sizes, not fresh blob content hashes or a final scientific audit.')
    OUTPUT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(receipt=OUTPUT.name,phase=record['phase'],
        jobs=[(r['job'],r['resume_steps'],r['archived_raw_files']) for r in rows])),flush=True)


if __name__=='__main__':
    main()

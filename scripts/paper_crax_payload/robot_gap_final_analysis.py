"""Run the single final PointGoal mechanism when Piper work/memory allow it.

Standard-library scheduler:30minute checks, no job preemption or training.
The native query job is not retried if it fails or has partial output.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def save(path, data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n'); temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--native-python',type=Path,required=True)
    parser.add_argument('--native-source-root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    native_source=args.native_source_root.resolve(strict=True)
    if not (native_source/'crax/__init__.py').is_file():
        raise ValueError('Require the installed pinned native CRAX checkout')
    status_path=root/'final_analysis_status.json'
    output=root/'pointgoal-final64x32'
    if status_path.exists() or output.exists():
        raise FileExistsError('Preserve existing scheduler/partial analysis; no automatic rerun')
    script=Path('scripts/paper_crax_payload/pointgoal_common_states.py').resolve()
    digest=hashlib.sha256(script.read_bytes()).hexdigest()
    status=dict(state='waiting_for_Piper_slots_and_shared_GPU_memory',controller_pid=os.getpid(),
        interval_seconds=1800,minimum_free_GPU_MiB=8192,training_transitions=0,
        planned_native_controls=16704,automatic_failed_job_restart=False,physical_robot_interactions=0)
    status['native_source_root']=str(native_source)
    while True:
        followup_path=root/'followup_status.json'
        followup=json.loads(followup_path.read_text()) if followup_path.exists() else {}
        jobs=followup.get('jobs',[])
        idle=len(jobs)==2 and all(j['status'] in ('completed','failed_no_automatic_restart') for j in jobs)
        continuation_path=root/'continuation_status.json'
        continuation=json.loads(continuation_path.read_text()) if continuation_path.exists() else {}
        if continuation and continuation.get('state') not in ('completed_pending_scientific_decision','needs_inspection_no_restart'):
            idle=False
        status.update(updated_unix=time.time(),Piper_followup_state=followup.get('state'))
        free={}
        if idle:
            try:
                rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                    '--format=csv,noheader,nounits'],text=True,timeout=10)
                free={int(r.split(',')[0]):int(r.split(',')[1]) for r in rows.splitlines()}
            except (OSError,subprocess.SubprocessError) as error:
                status['deferred_reason']=str(error)
        eligible=[gpu for gpu,memory in free.items() if memory>=8192]
        if idle and eligible:
            gpu=max(eligible,key=free.get)
            original=root.parent/'home_offload_20261113/SAFEFLOW2027_PAPER/runs'
            command=[str(args.native_python),'-u','-m','scripts.paper_crax_payload.pointgoal_common_states',
                '--checkpoints',str(original/'crax-formal-stable_iso-51028-10m-v1/full_10000128.pt'),
                str(original/'crax-formal-ea-51029-10m-v1/full_8000000.pt'),
                '--output',str(output),'--analysis-sha256',digest,'--device','cuda:0',
                '--states','64','--draws','32','--samplers','iso_reference','local','endpoint_direct','endpoint_pullback',
                '--selection-record','reports/paper_crax_payload/pointgoal-formal-selection-20261113/POINTGOAL_MODEL_SELECTION.json',
                '--minimum-training-steps','8000000','--query-batch-size','64']
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
                JAX_COMPILATION_CACHE_DIR=str(root/'cache/jax-final-mechanism'))
            environment['PYTHONPATH']=str(native_source)+os.pathsep+environment.get('PYTHONPATH','')
            with (root/'logs/pointgoal-final64x32.log').open('x') as stream:
                process=subprocess.Popen(command,env=environment,stdout=stream,stderr=subprocess.STDOUT)
                status.update(state='running',GPU=gpu,free_GPU_MiB_at_launch=free[gpu],pid=process.pid,
                    command=command,analysis_sha256=digest)
                save(status_path,status)
                code=process.wait()
            receipt_path=output/'receipt.json'
            receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
            ok=code==0 and receipt.get('status')=='completed' and receipt.get('physical_controls')==16704
            status.update(state='completed' if ok else 'failed_no_automatic_restart',exit_code=code,
                actual_native_controls=receipt.get('physical_controls'),updated_unix=time.time())
            save(status_path,status)
            break
        save(status_path,status)
        time.sleep(1800)


if __name__=='__main__':
    main()

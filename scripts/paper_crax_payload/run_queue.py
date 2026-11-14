"""Simple finite shared-GPU job slots. No restart daemon or foreign-process signals."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def next_slot(records, slots):
    used={record['slot_id'] for record in records}
    free=set(range(slots))-used
    if not free:raise ValueError('No released finite job slot')
    return min(free)


def validate_slot_environments(plan):
    environments=plan.get('slot_environments')
    if environments is None:return
    if len(environments)!=plan['slots'] or any(not isinstance(env,dict) for env in environments):
        raise ValueError('Exactly one environment mapping per finite slot required')
    if any('CUDA_VISIBLE_DEVICES' in job.get('environment',{}) for job in plan['jobs']):
        raise ValueError('GPU binding belongs to the released slot, not the next pending job')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    if plan.get('superseded_by') or plan.get('admission',{}).get('superseded_by') or plan.get('admission',{}).get('status','').startswith('superseded'):
        raise ValueError('Superseded queue plan: preserve historical registration, never duplicate its jobs')
    project=Path(plan['project']);state_path=Path(plan['state']);state_path.parent.mkdir(parents=True,exist_ok=True)
    if state_path.exists():raise FileExistsError('Inspect previous native handles; do not restart a queue blindly')
    if not 1<=plan['slots']<=2:raise ValueError('Only bounded shared slots allowed')
    validate_slot_environments(plan)
    jobs=plan['jobs'];live={};records=[];admission_wait_reason=None
    def save():
        state=dict(pid=os.getpid(),source_commit=args.source_commit,status='running' if jobs or live else 'completed',
            jobs=records,pending=len(jobs),live_native_pids=list(live),automatic_restart=False,foreign_processes_changed=False,
            admission_wait_reason=admission_wait_reason)
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
    while jobs or live:
        for pid,(proc,stream,record) in list(live.items()):
            code=proc.poll()
            if code is not None:
                stream.close();record.update(exit_code=code,status='completed' if code==0 else 'failed_not_restarted',finished_unix=time.time());del live[pid]
        while jobs and len(live)<plan['slots']:
            minimum_free=plan.get('minimum_free_disk_bytes',0)
            if minimum_free and shutil.disk_usage(project).free<minimum_free:
                admission_wait_reason='waiting_for_disk_space;active_jobs_not_stopped'
                break
            admission_wait_reason=None
            if plan.get('gpu_admission'):
                admission = plan['gpu_admission']
                free = int(subprocess.check_output(['nvidia-smi', '-i', str(admission['index']),
                    '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=15).strip())
                if free < admission['minimum_free_mib']:
                    admission_wait_reason='waiting_for_shared_GPU_memory;foreign_jobs_not_stopped'
                    break
            slot=next_slot([item[2] for item in live.values()],plan['slots'])
            job=jobs.pop(0);log=Path(job['log']);log.parent.mkdir(parents=True,exist_ok=True)
            if Path(job['output']).exists():raise FileExistsError('Preserve original run output')
            command=[plan['python'],'-u','-m',job['module'],*job['arguments'],'--output',job['output'],'--source-commit',args.source_commit]
            env=os.environ|{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','XLA_PYTHON_CLIENT_PREALLOCATE':'false'}
            env.update(job.get('environment',{}))
            if 'slot_environments' in plan:env.update(plan['slot_environments'][slot])
            stream=log.open('x');proc=subprocess.Popen(command,cwd=project,env=env,stdout=stream,stderr=subprocess.STDOUT)
            record=dict(name=job['name'],pid=proc.pid,command=command,output=job['output'],log=str(log),status='running',started_unix=time.time(),slot_id=slot,visible_devices=env.get('CUDA_VISIBLE_DEVICES'))
            records.append(record);live[proc.pid]=(proc,stream,record)
        save()
        if jobs or live:time.sleep(plan.get('poll_interval_seconds',5))


if __name__=='__main__':main()

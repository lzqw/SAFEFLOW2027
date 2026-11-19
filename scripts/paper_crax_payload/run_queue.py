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


def validate_shared_slots(plan):
    if 1<=plan['slots']<=2:return
    # Only the fixed three-method/one-seed campaign may opt into a third
    # shared slot. Other historic queues retain their original two-slot cap.
    seed=plan.get('crossseed_three_arm_single_seed')
    jobs=plan['jobs']
    if (plan['slots']!=3 or seed not in (41028,41029) or len(jobs)!=3
            or any(j.get('module')!='scripts.paper_crax_payload.train_reach_crossseed_v12' for j in jobs)):
        raise ValueError('Only the explicit fixed crossseed three-arm campaign may use three shared slots')
    arms=set();registrations=set();plans=set()
    for job in jobs:
        args=job['arguments']
        def value(flag):
            if args.count(flag)!=1:raise ValueError('Exact registered crossseed arguments required')
            return args[args.index(flag)+1]
        arms.add(value('--arm'));registrations.add(value('--execution-registration'));plans.add(value('--plan'))
        if (not job.get('require_completed_receipt') or job.get('required_steps')!=550000
                or value('--line')!='piper_reach'):
            raise ValueError('Three complete registered150K suffixes from fixed400K parents required')
    if arms!={'P5-fixed','P5-activation2mm','Metric-only'} or len(registrations)!=1 or len(plans)!=1:
        raise ValueError('Every method once,one seed plan and one exact execution registration')
    environments=plan.get('slot_environments')
    if (not isinstance(environments,list) or len(environments)!=3
            or any(not isinstance(env,dict) or not str(env.get('CUDA_VISIBLE_DEVICES','')).isdigit()
                   for env in environments)
            or len({env['CUDA_VISIBLE_DEVICES'] for env in environments})!=3):
        raise ValueError('The explicit third-slot campaign requires three distinct single-GPU bindings')


def prerequisites_ready(plan):
    """Optional finite campaign dependency, not permission to restart failed work."""
    for spec in plan.get('prerequisite_receipts',[]):
        path=Path(spec['path'])
        if not path.exists():return False
        record=json.loads(path.read_text())
        if (record.get('status') != spec.get('expected_status', 'completed')
                or ('steps' in spec and record.get('steps') != spec['steps'])):
            raise ValueError('Prerequisite exists but did not complete the registered budget')
        if record.get('engineering_only'):raise ValueError('Engineering smoke is not a scientific prerequisite')
    return True


def receipt_completed(receipt, job):
    if receipt.get('status') != 'completed':
        return False
    if job.get('required_steps') is not None:
        return receipt.get(job.get('required_steps_field','steps')) == job['required_steps']
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    if plan.get('superseded_by') or plan.get('admission',{}).get('superseded_by') or plan.get('admission',{}).get('status','').startswith('superseded'):
        raise ValueError('Superseded queue plan: preserve historical registration, never duplicate its jobs')
    project=Path(plan['project']);state_path=Path(plan['state']);state_path.parent.mkdir(parents=True,exist_ok=True)
    if state_path.exists():raise FileExistsError('Inspect previous native handles; do not restart a queue blindly')
    validate_shared_slots(plan)
    validate_slot_environments(plan)
    jobs=plan['jobs'];live={};records=[];admission_wait_reason=None
    def save():
        terminal='completed_with_failures' if any(r['status']=='failed_not_restarted' for r in records) else 'completed'
        state=dict(pid=os.getpid(),source_commit=args.source_commit,status='running' if jobs or live else terminal,
            jobs=records,pending=len(jobs),live_native_pids=list(live),automatic_restart=False,foreign_processes_changed=False,
            admission_wait_reason=admission_wait_reason)
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
    while jobs or live:
        for pid,(proc,stream,record) in list(live.items()):
            code=proc.poll()
            if code is not None:
                stream.close()
                success=code==0
                if success and record.get('require_completed_receipt'):
                    path=Path(record['output'])/'receipt.json'
                    try:
                        receipt=json.loads(path.read_text())
                        success=receipt_completed(receipt,record)
                    except (OSError,ValueError):success=False
                record.update(exit_code=code,status='completed' if success else 'failed_not_restarted',finished_unix=time.time());del live[pid]
        while jobs and len(live)<plan['slots']:
            if not prerequisites_ready(plan):
                admission_wait_reason='waiting_for_registered_predecessor_receipts;no_extra_GPU_slot'
                break
            occupied = 0
            for external in plan.get('existing_healthy_jobs', []):
                try:
                    cmd = Path('/proc',str(external['pid']),'cmdline').read_bytes()
                    occupied += int(external['expected_output'].encode() in cmd and b'payload_train' in cmd)
                except OSError:
                    pass
            if len(live) + occupied >= plan['slots']:
                admission_wait_reason='existing_healthy_project_job_retains_shared_slot'
                break
            minimum_free=plan.get('minimum_free_disk_bytes',0)
            if minimum_free and shutil.disk_usage(project).free<minimum_free:
                admission_wait_reason='waiting_for_disk_space;active_jobs_not_stopped'
                break
            admission_wait_reason=None
            slot=next_slot([item[2] for item in live.values()],plan['slots'])
            if plan.get('gpu_admission'):
                admission = plan['gpu_admission']
                index=admission['index']
                if index=='slot':
                    index=plan['slot_environments'][slot].get('CUDA_VISIBLE_DEVICES','')
                    if not index.isdigit():raise ValueError('One physical GPU per admitted slot required')
                free = int(subprocess.check_output(['nvidia-smi', '-i', str(index),
                    '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=15).strip())
                if free < admission['minimum_free_mib']:
                    admission_wait_reason='waiting_for_shared_GPU_memory;foreign_jobs_not_stopped'
                    break
            job=jobs.pop(0);log=Path(job['log']);log.parent.mkdir(parents=True,exist_ok=True)
            if Path(job['output']).exists():raise FileExistsError('Preserve original run output')
            command=[plan['python'],'-u','-m',job['module'],*job['arguments'],'--output',job['output'],'--source-commit',args.source_commit]
            env=os.environ|{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','XLA_PYTHON_CLIENT_PREALLOCATE':'false'}
            env.update(job.get('environment',{}))
            if 'slot_environments' in plan:env.update(plan['slot_environments'][slot])
            stream=log.open('x');proc=subprocess.Popen(command,cwd=project,env=env,stdout=stream,stderr=subprocess.STDOUT)
            record=dict(name=job['name'],pid=proc.pid,command=command,output=job['output'],log=str(log),status='running',started_unix=time.time(),slot_id=slot,visible_devices=env.get('CUDA_VISIBLE_DEVICES'))
            record.update({k:job[k] for k in ('require_completed_receipt','required_steps','required_steps_field') if k in job})
            records.append(record);live[proc.pid]=(proc,stream,record)
        save()
        if jobs or live:time.sleep(plan.get('poll_interval_seconds',5))


if __name__=='__main__':main()

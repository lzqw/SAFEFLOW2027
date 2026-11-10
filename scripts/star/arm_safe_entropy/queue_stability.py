"""Finite,resource-only queue for the six predeclared development trials.

Standard-library parent,one learner per A100 initially. The4080 may admit its
second slot after observed productive progress; no training mutation/restart.
GPU checks happen at resource admission,not on every owned-handle poll.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def resources(project):
    query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=15)
    devices={}
    for line in query.stdout.splitlines():
        index,free,utilization=[int(v.strip()) for v in line.split(',')]
        devices[index]=dict(free_mib=free,utilization=utilization)
    return dict(gpus=devices,load=os.getloadavg()[0],cpus=len(os.sched_getaffinity(0)),
                disk_free_gib=shutil.disk_usage(project).free/2**30)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--host-role',choices=['a100','4080'],required=True)
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args()
    source=Path(__file__).resolve().parents[3]
    plan=read_json(args.manifest)
    role=plan['host_roles'][args.host_role]
    all_jobs=[job for host in plan['host_roles'].values() for job in host['jobs']]
    if (len(all_jobs)!=6 or {(j['variant'],j['seed']) for j in all_jobs}!=
            {(v,s) for v in ('s_iso','s_local','s_ea') for s in (2027,2028)} or
            any(j['steps']!=300000 for j in all_jobs) or
            sum(j['steps'] for j in all_jobs)!=plan['reserved_new_training_transitions']):
        raise ValueError('This queue can launch only the bounded initial1.8M matrix')
    output_root=args.project/'runs/ea_stability_v2'
    output_root.mkdir(parents=True,exist_ok=True)
    state_path=args.project/'queues'/f'ea-stability-v2-{args.host_role}-{args.source_commit[:7]}.json'
    state_path.parent.mkdir(exist_ok=True)
    if state_path.exists(): raise ValueError('Existing queue evidence is not a restart authorization')
    pending=[dict(job) for job in role['jobs']]
    running=[];finished=[];next_resource_attempt={gpu:0. for gpu in role['gpus']}

    def emit(event,**values):
        print(json.dumps(dict(event=event,utc=datetime.now(timezone.utc).isoformat(),**values)),flush=True)

    def snapshot(status):
        atomic_json(state_path,dict(status=status,queue_pid=os.getpid(),source_commit=args.source_commit,
            host_role=args.host_role,manifest=str(args.manifest),pending=pending,
            running=[{k:v for k,v in r.items() if k!='process'} for r in running],finished=finished,
            automatic_restart=False,foreign_workloads_changed=False))

    snapshot('running')
    while pending or running:
        for job in list(running):
            returncode=job['process'].poll()
            if returncode is None: continue
            receipt=read_json(Path(job['output'])/'receipt.json')
            complete=(returncode==0 and receipt.get('status')=='completed' and receipt.get('steps')==300000
                      and receipt.get('variant')==job['variant'] and receipt.get('training_seed')==job['seed'])
            finished.append({k:v for k,v in job.items() if k!='process'} |
                            dict(returncode=returncode,status='completed' if complete else 'failed_no_restart'))
            running.remove(job)
            next_resource_attempt[job['gpu']]=0.
            emit('native_completed' if complete else 'native_failed_no_restart',variant=job['variant'],
                 seed=job['seed'],pid=job['pid'],output=job['output'],returncode=returncode)
        for gpu in role['gpus']:
            active=[r for r in running if r['gpu']==gpu]
            maximum=role['initial_slots_per_gpu']
            if (role['maximum_slots_per_gpu']>maximum and
                    any(read_json(Path(r['output'])/'status.json').get('steps',0)>=role.get('expand_after_observed_steps',5000)
                        for r in active)):
                maximum=role['maximum_slots_per_gpu']
            if not pending or len(active)>=maximum or time.monotonic()<next_resource_attempt[gpu]: continue
            try:
                observed=resources(args.project)
                admission=plan['admission'];card=observed['gpus'][gpu]
                admitted=(card['free_mib']>=admission['minimum_free_gpu_mib'] and
                    card['utilization']<=admission['maximum_gpu_utilization_percent'] and
                    observed['load']<=admission['maximum_host_load_fraction']*observed['cpus'] and
                    observed['disk_free_gib']>=admission['minimum_disk_free_gib'])
            except (OSError,subprocess.SubprocessError,ValueError,KeyError) as error:
                emit('resource_observation_failed_no_training_restart',gpu=gpu,error=str(error))
                next_resource_attempt[gpu]=time.monotonic()+plan['resource_retry_seconds'];continue
            if not admitted:
                emit('waiting_resource_admission',gpu=gpu,observed=observed)
                next_resource_attempt[gpu]=time.monotonic()+plan['resource_retry_seconds'];continue
            job=pending[0]
            output=output_root/job['name']
            log=args.project/'logs'/f"ea-stability-v2-{job['name']}.log"
            if output.exists() or log.exists():
                snapshot('existing_output_requires_inspection_no_restart')
                raise ValueError('Never overwrite/restart an existing native trial')
            command=[str(args.project/'env/bin/python'),'-u','-m','scripts.star.arm_safe_entropy.train_stability',
                '--config',str(source/plan['config']),'--variant',job['variant'],'--seed',str(job['seed']),
                '--steps','300000','--source-commit',args.source_commit,'--output',str(output)]
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',
                             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            with log.open('xb') as stream:
                process=subprocess.Popen(command,cwd=source,env=environment,stdout=stream,
                                         stderr=subprocess.STDOUT,start_new_session=True)
            pending.pop(0)
            running.append(dict(**job,gpu=gpu,pid=process.pid,output=str(output),log=str(log),process=process))
            emit('native_launched',variant=job['variant'],seed=job['seed'],gpu=gpu,pid=process.pid,
                 output=str(output),log=str(log),resource_admission=observed)
        snapshot('running')
        if pending or running: time.sleep(plan['owned_handle_poll_seconds'])
    success=all(r['status']=='completed' for r in finished)
    snapshot('completed_all_predeclared_jobs' if success else 'finished_with_failures_no_restarts')
    emit('queue_finished',completed=sum(r['status']=='completed' for r in finished),failed=sum(r['status']!='completed' for r in finished))


if __name__=='__main__':
    main()

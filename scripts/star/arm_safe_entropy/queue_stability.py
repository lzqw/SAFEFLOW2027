"""Finite resource-only queue for initial trials or explicit native followups.

Standard-library parent,one learner per A100 initially. The4080 may admit its
second slot after observed productive progress; no training mutation/restart.
GPU checks happen at resource admission,not on every owned-handle poll.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
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


class ExistingNativeHandle:
    """Read-only adoption of a detached native child; never signal/restart it."""
    def __init__(self,job,commit):
        self.job,self.commit=job,commit

    def poll(self):
        path=Path('/proc')/str(self.job['pid'])/'cmdline'
        try: command=path.read_bytes().replace(bytes([0]),b' ').decode()
        except FileNotFoundError: command=''
        if command:
            if ('train_stability' not in command or self.job['output'] not in command or self.commit not in command):
                raise ValueError('An adopted native PID no longer has its exact training identity')
            return None
        receipt=read_json(Path(self.job['output'])/'receipt.json')
        # Logical terminal result only,NOT an observed child OS wait status.
        return 0 if receipt.get('status')=='completed' else 255


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--host-role',choices=['a100','4080'],required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--learner-source',type=Path,help='Immutable native learner source;independent of resource-controller code')
    parser.add_argument('--adopt-state',type=Path,help='Explicit controller-only handoff;previous controller must already be stopped')
    parser.add_argument('--maximum-gpu-utilization',type=int,choices=range(1,101),metavar='1..100')
    parser.add_argument('--queue-id',help='Distinct named finite followup,not a restart of the initial queue')
    args=parser.parse_args()
    controller_source=Path(__file__).resolve().parents[3]
    source=args.learner_source.resolve() if args.learner_source else controller_source
    plan=read_json(args.manifest)
    if args.maximum_gpu_utilization is not None:
        plan['admission']['maximum_gpu_utilization_percent']=args.maximum_gpu_utilization
    role=plan['host_roles'][args.host_role]
    all_jobs=[job for host in plan['host_roles'].values() for job in host['jobs']]
    followup=plan.get('queue_kind')=='explicit_native_development_followup'
    if args.queue_id and not re.fullmatch('[a-z0-9-]{1,40}',args.queue_id):
        parser.error('Queue id must be a short safe filename component')
    if followup:
        if not args.queue_id or not 1<=len(all_jobs)<=4:
            raise ValueError('A finite followup requires a distinct id and at most four declared native segments')
        for job in all_jobs:
            initial=job.get('initial_steps',-1)
            if job['variant'] not in ('s_iso','s_local','s_ea') or job['seed'] not in (2027,2028):
                raise ValueError('Followups require the declared development methods and seeds')
            if job.get('engineering_recovery'):
                if not 0<initial<job['steps']==300000:
                    raise ValueError('Engineering followup only completes the initial300k endpoint')
            elif not initial==300000<job['steps']==1000000:
                raise ValueError('Scientific followup only declares a native300k-to1M continuation')
        reservation=sum(j['steps']-j['initial_steps'] for j in all_jobs)
        if reservation!=plan['reserved_new_training_transitions'] or not 0<reservation<=2800000:
            raise ValueError('Followup reservation must exactly match its bounded incremental transitions')
        if any(not Path(job.get('resume','')).is_file() for job in role['jobs']):
            raise ValueError('Each local followup requires its existing native full checkpoint')
    elif (len(all_jobs)!=6 or {(j['variant'],j['seed']) for j in all_jobs}!=
              {(v,s) for v in ('s_iso','s_local','s_ea') for s in (2027,2028)} or
              any(j['steps']!=300000 for j in all_jobs) or
              sum(j['steps'] for j in all_jobs)!=plan['reserved_new_training_transitions']):
        raise ValueError('The initial queue can launch only the bounded1.8M matrix')
    output_root=args.project/'runs/ea_stability_v2'
    output_root.mkdir(parents=True,exist_ok=True)
    state_path=args.project/'queues'/f'ea-stability-v2-{args.host_role}-{args.queue_id or args.source_commit[:7]}.json'
    state_path.parent.mkdir(exist_ok=True)
    if state_path.exists() and args.adopt_state!=state_path:
        raise ValueError('Existing queue evidence is not a restart authorization')
    pending=[dict(job) for job in role['jobs']]
    running=[];finished=[];next_resource_attempt={gpu:0. for gpu in role['gpus']}
    handoff=None
    if args.adopt_state:
        prior=read_json(args.adopt_state)
        if prior.get('source_commit')!=args.source_commit or prior.get('host_role')!=args.host_role:
            raise ValueError('Adopt only the same finite native cohort and host role')
        old_command=Path('/proc')/str(prior['queue_pid'])/'cmdline'
        if old_command.exists() and old_command.read_bytes().strip(bytes([0])):
            raise ValueError('Previous controller is still live;do not create two launch authorities')
        expected={(j['variant'],j['seed']) for j in role['jobs']}
        preserved=prior['pending']+prior['running']+prior['finished']
        if len(preserved)!=len(expected) or {(j['variant'],j['seed']) for j in preserved}!=expected:
            raise ValueError('Controller handoff must retain every original trial exactly once')
        backup=args.adopt_state.with_name(args.adopt_state.stem+f'.before-controller-{os.getpid()}.json')
        if backup.exists(): raise ValueError('Never replace previous handoff evidence')
        shutil.copy2(args.adopt_state,backup)
        pending=prior['pending'];finished=prior['finished']
        for job in prior['running']:
            handle=ExistingNativeHandle(job,args.source_commit)
            handle.poll()  # Validate the actual native identity before adoption.
            running.append(dict(**job,process=handle,adopted_native_handle=True))
        handoff=dict(previous_controller_pid=prior['queue_pid'],preserved_state=str(backup),
                     native_training_processes_changed=False)

    def emit(event,**values):
        print(json.dumps(dict(event=event,utc=datetime.now(timezone.utc).isoformat(),**values)),flush=True)

    def snapshot(status):
        atomic_json(state_path,dict(status=status,queue_pid=os.getpid(),source_commit=args.source_commit,
            controller_source=str(controller_source),learner_source=str(source),
            effective_resource_admission=plan['admission'],controller_handoff=handoff,
            host_role=args.host_role,manifest=str(args.manifest),pending=pending,
            running=[{k:v for k,v in r.items() if k!='process'} for r in running],finished=finished,
            automatic_restart=False,foreign_workloads_changed=False))

    snapshot('running')
    while pending or running:
        for job in list(running):
            returncode=job['process'].poll()
            if returncode is None: continue
            receipt=read_json(Path(job['output'])/'receipt.json')
            complete=(returncode==0 and receipt.get('status')=='completed' and receipt.get('steps')==job['steps']
                      and receipt.get('variant')==job['variant'] and receipt.get('training_seed')==job['seed'])
            adopted=job.get('adopted_native_handle',False)
            finished.append({k:v for k,v in job.items() if k!='process'} |
                            dict(returncode=None if adopted else returncode,
                                 terminal_evidence='completed_receipt_and_owned_PID_disappearance' if adopted else 'native_child_exit_and_receipt',
                                 status='completed' if complete else 'failed_no_restart'))
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
            prerequisite=pending[0].get('wait_for_queue_completion')
            if prerequisite:
                predecessor=read_json(Path(prerequisite))
                predecessor_pid=predecessor.get('queue_pid')
                # Resource sequencing only:finish the already running own
                # recovery slot,not a success-rate or scientific early-stop gate.
                if (predecessor.get('status')!='completed_all_predeclared_jobs' or
                        predecessor_pid is None or (Path('/proc')/str(predecessor_pid)/'cmdline').exists()):
                    continue
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
                '--steps',str(job['steps']),'--source-commit',args.source_commit,'--output',str(output)]
            if followup:
                command.extend(['--resume',job['resume']])
                if job.get('engineering_recovery'): command.append('--engineering-recovery')
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

"""Read-only half-hour native/resource snapshots;never restart a learner.

Run detached with persistent stdout/latest JSON so observation does not depend
on an interactive tool session. A completed monitored phase is not the whole
EA-STAB objective;transient SSH errors are not evidence of stopped training.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


REMOTE = r'''
import json,os,subprocess
from pathlib import Path
def read(path):
 p=Path(path)
 return json.loads(p.read_text()) if p.exists() else {}
def command(pid):
 try:return (Path('/proc')/str(pid)/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
 except FileNotFoundError:return ''
def native(job,commit):
 cmd=command(job['pid']);folder=Path(job['output'])
 receipt=read(folder/'receipt.json');status=read(folder/'status.json')
 alive='train_stability' in cmd and job['output'] in cmd and commit in cmd
 complete=receipt.get('status')=='completed' and receipt.get('steps')==job['steps']
 return dict(pid=job['pid'],variant=job['variant'],seed=job['seed'],gpu=job['gpu'],identity_alive=alive,
  steps=receipt.get('steps',status.get('steps')),target_steps=job['steps'],completed=complete,
  unexpected_terminal=not alive and not complete,
  receipt_is_archive_of_completed_A100_run=bool(job.get('receipt_is_archive_of_completed_A100_run')))
queues=[];direct=[]
for definition in cfg['queues']:
 state=read(definition['state']);cmd=command(state['queue_pid'])
 alive='queue_stability' in cmd and definition['identity_token'] in cmd
 rows=[native(job,state['source_commit']) for job in state['running']+state['finished']]
 queues.append(dict(state=definition['state'],queue_pid=state['queue_pid'],identity_alive=alive,
  status=state['status'],pending=state['pending'],rows=rows,
  completed=state['status']=='completed_all_predeclared_jobs',
  unexpected_terminal=(not alive and state['status']!='completed_all_predeclared_jobs') or any(row['unexpected_terminal'] for row in rows)))
for job in cfg['direct']:direct.append(native(job,job['source_commit']))
result=dict(host=cfg['host'],queues=queues,direct=direct)
if include_resources:
 try:
  result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15).strip()
 except (OSError,subprocess.SubprocessError) as error:result['gpu_observation_error']=str(error)
 result.update(load=os.getloadavg()[0],cpus=len(os.sched_getaffinity(0)))
print(json.dumps(result))
'''


def inspect_host(config,include_resources=True):
    try:
        command=([sys.executable,'-'] if config.get('local_execution') else
                 ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',config['host'],'python3','-'])
        result=subprocess.run(command,
            input='cfg='+repr(config)+'\ninclude_resources='+repr(include_resources)+'\n'+REMOTE,
            text=True,capture_output=True,timeout=35,check=True)
        return json.loads(result.stdout)
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        return dict(host=config['host'],observation_error=str(error))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--latest',type=Path,required=True)
    parser.add_argument('--first-check-at',help='ISO timestamp;align replacement to the existing30-minute schedule')
    args=parser.parse_args()
    if args.first_check_at:
        remaining=(datetime.fromisoformat(args.first_check_at)-datetime.now().astimezone()).total_seconds()
        if remaining>0:
            print(json.dumps(dict(event='waiting_scheduled_observation',at=args.first_check_at)),flush=True)
            time.sleep(remaining)
    while True:
        plan=json.loads(args.plan.read_text())
        if plan['interval_seconds']!=1800: raise ValueError('Retain the requested half-hour inspection cadence')
        observations=[inspect_host(config) for config in plan['hosts']]
        snapshot=dict(local_time=datetime.now().astimezone().isoformat(),observations=observations,
                      automatic_restart=False,whole_goal_complete=False)
        temporary=args.latest.with_suffix('.tmp')
        temporary.write_text(json.dumps(snapshot,indent=2)+'\n');temporary.replace(args.latest)
        print(json.dumps(snapshot),flush=True)
        if all('observation_error' not in value for value in observations):
            jobs=[job for value in observations for job in value['queues']+value['direct']]
            if any(job['unexpected_terminal'] for job in jobs):
                print('EA_STABILITY_NATIVE_TERMINAL_INSPECT_NO_AUTORESTART',flush=True);break
            if jobs and all(job['completed'] for job in jobs):
                print('EA_STABILITY_MONITORED_PHASE_COMPLETE_NOT_WHOLE_GOAL_COMPLETE',flush=True);break
        time.sleep(plan['interval_seconds'])


if __name__=='__main__':
    main()

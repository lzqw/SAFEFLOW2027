"""One-shot native completion -> direct server archive -> declared long queue.

No model selection,training restart,foreign-process signals or local replay
copy. A missing native handle without its completed receipt stops this stage;
a transient observation error does not declare training terminal.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time


def ssh(host,*command):
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,*command]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--monitor-plan',type=Path)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Do not repeat a completed staging operation')
    plan=json.loads(args.plan.read_text());parent=plan['parent'];destination=plan['destination']
    if (parent['variant']!='s_iso' or parent['seed']!=2028 or parent['target_steps']!=300000
            or destination['target_steps']!=1000000 or plan['reserved_new_training_transitions']!=700000):
        raise ValueError('This one-shot stage only completes the declared last Iso2028 continuation')
    print(json.dumps(dict(event='waiting_owned_native_parent',host=parent['host'],pid=parent['pid'])),flush=True)
    probe="""
import json
from pathlib import Path
folder=Path(parent['output']);receipt_path=folder/'receipt.json';status_path=folder/'status.json'
receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
status=json.loads(status_path.read_text()) if status_path.exists() else {}
try: command=(Path('/proc')/str(parent['pid'])/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
except FileNotFoundError: command=''
alive='train_stability' in command and parent['output'] in command and parent['source_commit'] in command
completed=(receipt.get('status')=='completed' and receipt.get('steps')==parent['target_steps']
 and receipt.get('variant')==parent['variant'] and receipt.get('training_seed')==parent['seed'])
print(json.dumps(dict(alive=alive,completed=completed,steps=receipt.get('steps',status.get('steps')),
 protected_full_exists=(folder/'full_300000.pt').is_file(),receipt=receipt)))
"""
    while True:
        try:
            result=subprocess.run(ssh(parent['host'],'python3','-'),input='parent='+repr(parent)+'\n'+probe,
                text=True,capture_output=True,timeout=35,check=True)
            observed=json.loads(result.stdout)
        except (OSError,ValueError,subprocess.SubprocessError) as error:
            print(json.dumps(dict(event='observation_error_not_training_restart',error=str(error))),flush=True)
            time.sleep(60);continue
        if observed['completed'] and not observed['alive']:
            if not observed['protected_full_exists']: raise ValueError('Completed parent is missing its protected native full snapshot')
            break
        if not observed['alive']: raise RuntimeError('Native parent terminal without the matching completed300k receipt;inspect,do not restart')
        time.sleep(60)
    name=Path(parent['output']).name
    source_root=str(Path(parent['output']).parent)
    target_root=destination['project']+'/runs/ea_stability_v2'
    # Names/paths come from this explicit owned plan,not shell interpolation
    # of user input. Refuse an existing destination rather than overwriting it.
    producer=subprocess.Popen(ssh(parent['host'],f'tar --exclude=latest_full.pt -cf - -C {source_root} {name}'),
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    consumer=subprocess.Popen(ssh(destination['host'],f'test ! -e {target_root}/{name} && tar -xf - -C {target_root}'),
                              stdin=producer.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    producer.stdout.close()
    _,consumer_error=consumer.communicate(timeout=300)
    _,producer_error=producer.communicate(timeout=30)
    if producer.returncode or consumer.returncode:
        raise RuntimeError('Archive failed;preserve the partial destination for inspection: '+
                           (producer_error+consumer_error).decode()[-1000:])
    launcher="""
from pathlib import Path
import json,subprocess
base=Path(destination['project']);state=base/'queues'/('ea-stability-v2-4080-'+destination['queue_id']+'.json')
log=base/'logs'/('ea-stability-v2-queue-4080-'+destination['queue_id']+'.log')
assert not state.exists() and not log.exists()
assert (base/'runs/ea_stability_v2'/destination['parent_name']/'full_300000.pt').is_file()
controller=base/'code'/destination['controller_commit'];learner=base/'code'/destination['learner_commit']
command=[str(base/'env/bin/python'),'-u','-m','scripts.star.arm_safe_entropy.queue_stability',
 '--manifest',destination['queue_manifest'],'--project',str(base),'--host-role','4080',
 '--source-commit',destination['learner_commit'],'--learner-source',str(learner),'--queue-id',destination['queue_id']]
with log.open('x') as stream:
 process=subprocess.Popen(command,cwd=controller,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(queue_pid=process.pid,state=str(state),log=str(log))))
"""
    launch=subprocess.run(ssh(destination['host'],'python3','-'),input='destination='+repr(destination)+'\n'+launcher,
        text=True,capture_output=True,timeout=35,check=True)
    receipt=dict(status='archived_and_declared_queue_launched_not_training_complete',
        utc=datetime.now(timezone.utc).isoformat(),source_commit=args.source_commit,plan=plan,parent_receipt=observed['receipt'],
        destination_controller=json.loads(launch.stdout),local_large_files_created=0,
        new_training_transitions_launched_by_this_stager=0,reserved_continuation_transitions=700000,
        native_training_processes_stopped=0,foreign_processes_changed=0)
    temporary=args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt,indent=2)+'\n');temporary.replace(args.output)
    if args.monitor_plan:
        monitoring=json.loads(args.monitor_plan.read_text())
        host=next(value for value in monitoring['hosts'] if value['host']==destination['host'])
        definition=dict(state=receipt['destination_controller']['state'],identity_token=destination['queue_id'])
        if not any(value['state']==definition['state'] for value in host['queues']): host['queues'].append(definition)
        temporary=args.monitor_plan.with_suffix('.staging.tmp')
        temporary.write_text(json.dumps(monitoring,indent=2)+'\n');temporary.replace(args.monitor_plan)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()

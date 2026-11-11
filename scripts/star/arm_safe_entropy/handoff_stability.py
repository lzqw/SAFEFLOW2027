"""One explicit EA2027 resource handoff,not a restart daemon or new trial.

The existing scientific profile is immutable. On any failure before the
source is terminal,resume that exact owned native process;never stop others.
"""
import argparse
import json
from pathlib import Path
import subprocess


def remote(host,script,python='python3',timeout=40):
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,python,'-'],
                          input=script,text=True,capture_output=True,timeout=timeout,check=True)
    return json.loads(result.stdout)


def transfer(plan,extra_only=False):
    source=Path(plan['source_output']);destination=Path(plan['destination_parent_archive'])
    if extra_only:
        source_command=f'tar -C {source} -cf - resource_handoff.json engineering_interruption.json status.json'
        destination_command=f'tar -C {destination} -xf -'
    else:
        source_command=f'tar -C {source.parent} --exclude=latest_full.pt --exclude=*.tmp -cf - {source.name}'
        destination_command=f'mkdir {destination} && tar -C {destination.parent} -xf -'
    with subprocess.Popen(['ssh',plan['source_host'],source_command],stdout=subprocess.PIPE,stderr=subprocess.PIPE) as first:
        with subprocess.Popen(['ssh',plan['destination_host'],destination_command],stdin=first.stdout,
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE) as second:
            first.stdout.close()
            _,error=second.communicate(timeout=100)
            source_error=first.communicate(timeout=10)[1]
            if second.returncode or first.returncode:
                raise RuntimeError(f'Archive transfer failed:{source_error!r}:{error!r}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    if (plan['source_pid']!=1997214 or plan['variant']!='s_ea' or plan['training_seed']!=2027 or
            plan['pin_checkpoint_steps']!=450000 or plan['target_steps']!=1000000):
        raise ValueError('This one-shot handoff is not a generic migration/restart service')
    prefix='p='+repr(plan)+'\n'
    source_python='/home/lzqw/SAFEFLOW2027_ARM/env/bin/python'
    destination_python='/home/carla/4T/SAFEFLOW2027_ARM/env/bin/python'
    destination_guard=r'''
import os,shutil,subprocess
from pathlib import Path
out=Path(p['destination_output']);archive=Path(p['destination_parent_archive'])
assert not out.exists(),'Never overwrite a destination trial'
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[p['destination_gpu']])
mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
assert free>=p['minimum_free_vram_mib']
assert mem['MemAvailable']>=p['minimum_available_ram_gib']*1024**2
assert os.getloadavg()[0]<=p['maximum_cpu_load_per_cpu']*len(os.sched_getaffinity(0))
assert shutil.disk_usage(archive.parent).free>=p['minimum_free_disk_gib']*1024**3
'''
    remote(plan['destination_host'],prefix+destination_guard+"assert not archive.exists()\nimport json;print(json.dumps(dict(admitted=True)))\n")
    terminal=False;frozen=False
    try:
        # Import/load only from CPU. Wait for the already scheduled atomic full
        # save,then freeze its exact native owner;do not capture mid-update state.
        frozen=True  # On any uncertain SSH outcome,continue only the exact source identity.
        frozen_state=remote(plan['source_host'],prefix+r'''
import json,os,signal,time,platform
from datetime import datetime,timezone
from pathlib import Path
import torch
folder=Path(p['source_output']);proc=Path('/proc')/str(p['source_pid'])
def verify():
 command=(proc/'cmdline').read_bytes()
 assert b'train_stability' in command and p['source_output'].encode() in command and p['source_driver'].encode() in command
 assert proc.stat().st_uid==os.getuid(),'Only the owned exact source native may be signalled'
verify();started=time.monotonic();old=(folder/'latest_full.pt').stat().st_mtime_ns
while (folder/'latest_full.pt').stat().st_mtime_ns==old:
 verify()
 assert time.monotonic()-started<900,'No automatic extension or fallback checkpoint'
 time.sleep(.25)
verify();os.kill(p['source_pid'],signal.SIGSTOP)
try:
 state=torch.load(folder/'latest_full.pt',map_location='cpu',weights_only=False)
 assert state['format']=='ea_stability_v2_full' and state['steps']==p['pin_checkpoint_steps']
 assert state['variant']==p['variant'] and state['training_seed']==p['training_seed']
 assert state['source_commit']==p['source_driver']
 assert state['learner']['ea_stability_v2']['transitions']==state['steps']
 assert all(k in state for k in ('rng','replay','environments','episode_returns','episode_lengths'))
 assert len(state['environments'])==2 and len(state['rng']['cuda'])==1
 pin=folder/p['checkpoint_filename'];os.link(folder/'latest_full.pt',pin)
 log=json.loads((folder/'status.json').read_text())
 ticks=float((proc/'stat').read_text().rsplit(')',1)[1].split()[19])/os.sysconf('SC_CLK_TCK')
 boot=int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
 print(json.dumps(dict(steps=state['steps'],violations=state['violations'],episodes=state['episodes'],log=log,
  source_node=platform.node(),native_started_utc=datetime.fromtimestamp(boot+ticks,timezone.utc).isoformat())),flush=True)
except BaseException:
 os.kill(p['source_pid'],signal.SIGCONT);raise
''',source_python,timeout=920)
        frozen=True
        transfer(plan)
        remote(plan['destination_host'],prefix+destination_guard+r'''
import json,torch
state=torch.load(archive/p['checkpoint_filename'],map_location='cpu',weights_only=False)
parent=json.loads((archive/'run_config.json').read_text())
assert state['steps']==p['pin_checkpoint_steps'] and state['variant']==p['variant'] and state['training_seed']==p['training_seed']
assert state['configuration']==parent['configuration'] and parent['total_transition_budget']==p['target_steps']
assert state['learner']['ea_stability_v2']['transitions']==state['steps']
print(json.dumps(dict(verified_full_checkpoint=True)))
''',destination_python)
        incident=dict(plan,native_pid=plan['source_pid'],source_node=frozen_state['source_node'],
            native_started_utc=frozen_state['native_started_utc'],reason='resource_handoff',
            recovery_checkpoint_steps=frozen_state['steps'],
            last_logged_steps=frozen_state['log']['steps'],
            logged_lost_tail_transitions=frozen_state['log']['steps']-frozen_state['steps'],
            logged_lost_tail_violations=frozen_state['log']['training_violation_count']-frozen_state['violations'],
            logged_lost_tail_episodes=frozen_state['log']['episodes']-frozen_state['episodes'])
        stopped=remote(plan['source_host'],prefix+'incident='+repr(incident)+'\n'+r'''
import json,os,signal,time
from datetime import datetime,timezone
from pathlib import Path
proc=Path('/proc')/str(p['source_pid']);folder=Path(p['source_output'])
command=(proc/'cmdline').read_bytes()
assert b'train_stability' in command and p['source_output'].encode() in command and p['source_driver'].encode() in command
assert proc.stat().st_uid==os.getuid()
os.kill(p['source_pid'],signal.SIGTERM);os.kill(p['source_pid'],signal.SIGCONT)
for _ in range(100):
 if not (proc/'cmdline').exists() or b'train_stability' not in (proc/'cmdline').read_bytes():break
 time.sleep(.1)
else:raise RuntimeError('Source termination not confirmed;do not start a duplicate')
incident.update(native_process_confirmed_terminal=True,native_terminal_observed_utc=datetime.now(timezone.utc).isoformat())
for name in ('resource_handoff.json','engineering_interruption.json'):
 (folder/name).write_text(json.dumps(incident,indent=2)+'\n')
status=json.loads((folder/'status.json').read_text());status['state']='intentional_resource_handoff'
(folder/'status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(incident))
''')
        terminal=True;transfer(plan,extra_only=True)
        launched=remote(plan['destination_host'],prefix+destination_guard+r'''
import json,subprocess
base=Path('/home/carla/4T/SAFEFLOW2027_ARM');code=base/'code'/p['handoff_driver']
log=base/'logs/ea-stability-v2-s-ea-seed2027-v2-long-handoff1.log'
assert not log.exists()
command=[str(base/'env/bin/python'),'-u','-m','scripts.star.arm_safe_entropy.train_stability',
 '--config',str(code/'configs/star/arm_ea_stability/development_v2.json'),'--variant',p['variant'],
 '--seed',str(p['training_seed']),'--steps',str(p['target_steps']),'--source-commit',p['handoff_driver'],
 '--output',p['destination_output'],'--resume',str(archive/p['checkpoint_filename']),'--resource-handoff']
with log.open('x') as stream:
 child=subprocess.Popen(command,cwd=code,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(p['destination_gpu']),
  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,log=str(log),command=command)))
''')
        receipt=dict(status='source_terminal_full_state_archived_destination_launched_not_training_complete',
                     incident=stopped,destination=launched,whole_goal_complete=False)
        args.receipt.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    finally:
        if frozen and not terminal:
            remote(plan['source_host'],prefix+r'''
import json,os,signal
from pathlib import Path
proc=Path('/proc')/str(p['source_pid'])
if (proc/'cmdline').exists():
 command=(proc/'cmdline').read_bytes()
 if b'train_stability' in command and p['source_output'].encode() in command and proc.stat().st_uid==os.getuid():
  os.kill(p['source_pid'],signal.SIGCONT)
print(json.dumps(dict(source_continue_requested_if_exact_identity_alive=True)))
''')


if __name__=='__main__':main()

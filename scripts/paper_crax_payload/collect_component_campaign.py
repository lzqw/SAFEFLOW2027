"""Finite half-hour collection of six existing jobs; never launch/restart RL.

Only completed compact artifacts and one server-side CPU summary are copied.
The original large checkpoints, trajectories and training processes stay put.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from scripts.paper_crax_payload.analyze_component_trajectories import compare_summaries
from scripts.paper_crax_payload.train_crax import atomic_json


ARMS=('iso','noise-only','metric-only','full','full-rho010','full-rho050')
FILES=('run_config.json','receipt.json','progress.jsonl',
       'evaluation_100000.json','evaluation_200000.json','evaluation_300000.json')
A100='/storage_0/SAFEFLOW2027_lzqw_uid1007/nonhardware_v4'
RTX='/home/carla/4T/SAFEFLOW2027_NONHARDWARE_V4'


def placements(pending_parameters_on4080=False):
    on_rtx=('iso','metric-only','full-rho010','full-rho050') if pending_parameters_on4080 else ('iso','metric-only')
    return [dict(arm=arm,host='4080' if arm in on_rtx else 'tjucta-a100',
        root=RTX if arm in on_rtx else A100,
        python='/home/carla/4T/SAFEFLOW2027_ARM/env/bin/python' if arm in on_rtx
               else '/home/lzqw/SAFEFLOW2027_ARM/env/bin/python') for arm in ARMS]


def ssh(host,command,timeout=30):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',host,command],
        text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,check=True).stdout


def remote_inventory(host,items):
    # Every observed job is bound to its own native manifest/output, not just
    # a stale status file or a coincidentally reused process number.
    code='''import json,pathlib
items=json.loads(PAYLOAD);out=[]
for item in items:
    directory=pathlib.Path(item['root'])/'runs'/item['arm']
    row=dict(arm=item['arm'],host=HOST)
    for name in ('receipt','status','run_config'):
        path=directory/(name+'.json')
        row[name]=json.loads(path.read_text()) if path.exists() else None
    manifest=row.pop('run_config')
    pid=manifest.get('native_pid') if manifest else None
    row['native_pid']=pid;row['matching_native_process']=False
    if pid:
        try:
            command=(pathlib.Path('/proc')/str(pid)/'cmdline').read_bytes()
            row['matching_native_process']=str(directory).encode() in command and b'train_component_study' in command
        except OSError:pass
    out.append(row)
print(json.dumps(out))
'''.replace('PAYLOAD',repr(json.dumps(items))).replace('HOST',repr(host))
    return json.loads(ssh(host,shlex.join([items[0]['python'],'-c',code])))


def require_completion(receipt,arm):
    if (not receipt or receipt.get('status')!='completed' or receipt.get('engineering_only')
            or receipt.get('steps')!=300000 or receipt.get('training_seed')!=2027
            or receipt.get('arm',{}).get('name')!=arm
            or receipt.get('source_commit')!='b31878c0'):
        raise ValueError('Require the original registered full300K scientific run')


def fetch(item,remote,local):
    temporary=local.with_suffix(local.suffix+'.transfer')
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=12',
        f'{item["host"]}:{remote}',str(temporary)],timeout=90,check=True)
    temporary.replace(local)


def collect_one(item,output):
    directory=output/'runs'/item['arm'];directory.mkdir(parents=True,exist_ok=True)
    if (directory/'collected.json').exists():return
    if shutil.disk_usage(output).free<1024**3:raise RuntimeError('Preserve1GiB local disk headroom')
    remote=f'{item["root"]}/runs/{item["arm"]}'
    for name in FILES:
        if not (directory/name).exists():fetch(item,f'{remote}/{name}',directory/name)
    require_completion(json.loads((directory/'receipt.json').read_text()),item['arm'])
    summary=f'{item["root"]}/analysis-tools-0425d452/{item["arm"]}-summary-v1.json'
    marker=directory/'remote_summary_started.json'
    exists=ssh(item['host'],f'test -f {shlex.quote(summary)} && echo yes || echo no').strip()=='yes'
    if not exists:
        if marker.exists():raise RuntimeError('A prior remote summary started without a result; inspect, do not launch twice')
        atomic_json(marker,dict(started_utc=datetime.now(timezone.utc).isoformat(),remote_output=summary,
            automatic_retry=False,new_training_transitions=0))
        command=[item['python'],f'{item["root"]}/analysis-tools-0425d452/analyze_component_trajectories.py',
            '--summarize-run',remote,'--output',summary]
        # Deliberately no GPU, no environment import and one CPU thread. A
        # timeout is not permission to relaunch: retain the started marker.
        ssh(item['host'],'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '+shlex.join(command),timeout=900)
    fetch(item,summary,directory/'trajectory_summary.json')
    data=json.loads((directory/'trajectory_summary.json').read_text())
    if data['arm']!=item['arm'] or data['steps']!=300000 or not data['explicit_trace_end']:
        raise ValueError('Incomplete or wrong remote trajectory summary')
    atomic_json(directory/'collected.json',dict(status='collected_complete_native_result',
        arm=item['arm'],host=item['host'],raw_trace_copied=False,full_checkpoint_copied=False,
        new_training_transitions=0,new_simulation_controls=0))


def finalize(output):
    from scripts.paper_crax_payload.report_component_study import report as report_study
    from scripts.paper_crax_payload.report_component_exploration import report as report_exploration
    runs=[output/'runs'/arm for arm in ARMS]
    summaries=[json.loads((p/'trajectory_summary.json').read_text()) for p in runs]
    if (len({r['configuration_digest'] for r in summaries})!=1
            or len({r['initial_network_sha256'] for r in summaries})!=1
            or summaries[0]['initial_network_sha256'] is None):
        raise ValueError('Different cohort configurations or initial networks')
    comparison=compare_summaries(summaries)
    atomic_json(output/'exploration_comparison.json',comparison)
    # If any report is interrupted, preserve its partial output for inspection.
    # Do not silently overwrite it or rerun the evaluation/train protocol.
    report_study(runs,output/'component_results')
    report_exploration(comparison,output/'exploration_results')
    atomic_json(output/'receipt.json',dict(status='component_collection_and_reports_completed',
        all_six_arms_retained=True,training_transitions_represented=1800000,
        new_training_transitions=0,new_simulation_controls=0,physical_robot_interactions=0,
        whole_goal_complete=False,CVPO_results_still_separate=True,
        github_publication='pending_next_agent_review'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--interval',type=int,default=1800)
    p.add_argument('--initial-delay-seconds',type=int,default=0)
    p.add_argument('--samples',type=int,default=336)
    p.add_argument('--pending-parameters-on4080',action='store_true',
        help='Read only the two never-started relocated parameter arms from4080; no training action')
    args=p.parse_args()
    if args.interval<1800 or not 1<=args.samples<=336 or not 0<=args.initial_delay_seconds<=1800:
        p.error('Finite half-hour observation only')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'collector.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'receipt.json').exists():raise FileExistsError('Already collected; no new report run')
    items=placements(args.pending_parameters_on4080);errors={}
    if args.initial_delay_seconds:
        atomic_json(output/'status.json',dict(state='initial_wait_for_next_half_hour',collector_pid=os.getpid(),
            next_check_unix=time.time()+args.initial_delay_seconds,interval_seconds=args.interval,
            new_training_transitions=0,new_simulation_controls=0,automatic_training_restart=False))
        time.sleep(args.initial_delay_seconds)
    for sample in range(args.samples):
        inventory=[]
        for host in ('tjucta-a100','4080'):
            try:inventory.extend(remote_inventory(host,[i for i in items if i['host']==host]))
            except (OSError,ValueError,subprocess.SubprocessError) as error:
                errors[host]=str(error);continue
            errors.pop(host,None)
        for row in inventory:
            if not row['receipt']:continue
            try:
                require_completion(row['receipt'],row['arm'])
                collect_one(next(i for i in items if i['arm']==row['arm']),output)
                errors.pop(row['arm'],None)
            except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
                errors[row['arm']]=str(error)
        collected=[arm for arm in ARMS if (output/'runs'/arm/'collected.json').exists()]
        status=dict(state='waiting_for_existing_native_runs',collector_pid=os.getpid(),sample=sample+1,
            time_utc=datetime.now(timezone.utc).isoformat(),interval_seconds=args.interval,
            collected=collected,errors=errors,native_inventory=inventory,
            new_training_transitions=0,new_simulation_controls=0,automatic_training_restart=False)
        atomic_json(output/'status.json',status)
        print(json.dumps({k:v for k,v in status.items() if k!='native_inventory'}),flush=True)
        if len(collected)==6:
            try:finalize(output)
            except BaseException as error:
                atomic_json(output/'report_interruption.json',dict(error=str(error),automatic_retry=False))
                raise
            return
        if sample+1<args.samples:time.sleep(args.interval)


if __name__=='__main__':main()

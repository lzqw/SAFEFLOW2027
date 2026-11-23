"""Half-hour read-only archive of the frozen selected-model experiment; no repeat evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


ROOT='/home/carla/4T/SAFEFLOW2027_SIMULATION_V12'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-source',required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    for check in range(48):
        state='observation_unavailable_not_job_failure'
        try:
            result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','4080',
                'cat '+ROOT+'/queue_selected_holdout_20261123_v2.json'],capture_output=True,text=True,timeout=45)
            if result.returncode==0:
                queue=json.loads(result.stdout)
                if queue['source_commit']!=args.expected_source:
                    raise ValueError('Observe only the exact frozen experiment source')
                state=queue['status']
                if state in ('completed','completed_with_failures'):
                    if queue['pending'] or queue['live_native_pids'] or len(queue['jobs'])!=2:
                        raise ValueError('All registered selected-model tasks must be terminal')
                    if shutil.disk_usage(args.output).free<1024**3:
                        raise OSError('Preserve1GiB local free space')
                    for task in ('Reach','Payload'):
                        job=next(j for j in queue['jobs'] if j['name']==task+'-selected-seed-holdout')
                        remote=ROOT+'/selected-seed-frozen-holdout-20261123/'+task
                        if job['output']!=remote:
                            raise ValueError('Archive only exact registered outputs')
                        target=args.output/task;target.mkdir(exist_ok=True)
                        subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes -o ConnectTimeout=12',
                            '--include=*.json','--include=*.csv','--exclude=*',
                            '4080:'+remote+'/',str(target)+'/'],check=True,timeout=120)
                        if job['status']=='completed':
                            receipt=json.loads((target/'receipt.json').read_text())
                            if (receipt['status']!='completed' or receipt['task']!=task
                                    or receipt['evaluated_episodes']!=1600 or not receipt['independent_test_used']
                                    or receipt['holdout_used_for_selection'] or receipt['source_commit']!=args.expected_source):
                                raise ValueError('Never turn failed or partial holdout into confirmation')
                    files=[p for p in args.output.rglob('*') if p.is_file() and p.name!='latest_observation.json']
                    (args.output/'archive_receipt.json').write_text(json.dumps(dict(queue=queue,
                        complete_selected_model_experiment=state=='completed',
                        hashes={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                        evaluations_launched=0,training_launches=0,github_writes=0,messages_sent=0),indent=2)+'\n')
                    state='terminal_archived'
        except (OSError,subprocess.SubprocessError,json.JSONDecodeError) as error:
            state='observation_or_archive_error:'+type(error).__name__
        record=dict(check=check+1,unix_time=time.time(),state=state,poll_interval_seconds=1800,
                    evaluations_launched=0,training_launches=0,github_writes=0,messages_sent=0)
        (args.output/'latest_observation.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        if state=='terminal_archived':return
        if check<47:time.sleep(1800)
    raise TimeoutError('Finite archive observation ended; no remote job or policy changed')


if __name__=='__main__':main()

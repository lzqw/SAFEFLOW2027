"""Finite30min read-only archive of the exact two4080 queues; no GPU or GitHub writes."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time


ROOT = '/home/carla/4T/SAFEFLOW2027_SIMULATION_V12'
QUEUES = {
    'payload42028_recovery': {'Payload42028-recovery150Kto300K'},
    'third_confirmation': {'Reach-41030-'+arm for arm in ('P5-fixed','P5-activation2mm','Metric-only')},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    archived = set()
    for check in range(48):
        states = {}
        for kind, names in QUEUES.items():
            if kind in archived:
                states[kind] = 'archived_terminal'
                continue
            path = ROOT+'/queue_'+kind+'_20261121.json'
            try:
                result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','4080',
                    'cat '+shlex.quote(path)], capture_output=True, text=True, timeout=45)
                if result.returncode:
                    states[kind] = 'observation_failed_not_training_failure'
                    continue
                queue = json.loads(result.stdout)
                if queue['source_commit'] != '075f8f5ec117e2b331b3909ba1521c0a02981b7c':
                    raise ValueError('Exact queue source required')
                states[kind] = queue['status']
                if queue['status'] not in ('completed','completed_with_failures'):
                    continue
                if (queue['pending'] or queue['live_native_pids']
                        or {j['name'] for j in queue['jobs']} != names):
                    raise ValueError('Retain all registered jobs, not a favorable subset')
                if shutil.disk_usage(args.output).free < 1024**3:
                    raise OSError('Leave1GiB local space; never fetch model files')
                destination = args.output/kind
                destination.mkdir(exist_ok=True)
                for job in queue['jobs']:
                    expected = (ROOT+'/runs-payload-recovery-20261121/Payload-EA-42028'
                        if kind == 'payload42028_recovery' else
                        ROOT+'/runs-third-confirmation-20261121/'+job['name'])
                    if job['output'] != expected:
                        raise ValueError('Only the exact registered data-disk outputs')
                    target = destination/job['name']
                    target.mkdir(exist_ok=True)
                    subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes -o ConnectTimeout=12',
                        '--include=*.json','--include=progress.jsonl','--exclude=*',
                        '4080:'+expected+'/',str(target)+'/'],check=True,timeout=120)
                    if job['status'] == 'completed':
                        receipt = json.loads((target/'receipt.json').read_text())
                        steps = receipt.get('training_steps',receipt.get('steps'))
                        if receipt['status'] != 'completed' or steps != job['required_steps']:
                            raise ValueError('Never mark a partial/failed run complete')
                files = sorted(p for p in destination.rglob('*') if p.is_file())
                (destination/'archive_receipt.json').write_text(json.dumps(dict(
                    queue=queue,sha256={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in files},complete_scientific_campaign=queue['status']=='completed',
                    independent_test_used=False,training_launches=0,github_writes=0),indent=2)+'\n')
                archived.add(kind)
                states[kind] = 'archived_terminal'
            except (subprocess.SubprocessError,OSError,json.JSONDecodeError) as error:
                states[kind] = 'archive_observation_error:'+type(error).__name__
        record = dict(check=check+1,unix_time=time.time(),states=states,poll_interval_seconds=1800,
                      training_launches=0,github_writes=0,messages_sent=0)
        (args.output/'latest_observation.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        if len(archived)==len(QUEUES):
            return
        if check<47:
            time.sleep(1800)
    raise TimeoutError('Finite archive wait ended; remote training untouched')


if __name__ == '__main__':
    main()

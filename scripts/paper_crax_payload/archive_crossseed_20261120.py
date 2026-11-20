"""Fetch compact completed host summaries every30min, then merge exactly once.

Read-only on servers. Never launches/restarts training, deletes files, publishes
to GitHub, or sends messages. A failed SSH observation is not a training failure.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


HOSTS = {
    'tjucta-a100': '/storage_0/SAFEFLOW2027_lzqw_uid1007/simulation_v12',
    '4080': '/home/carla/4T/SAFEFLOW2027_SIMULATION_V12',
}
FILES = ('receipt.json', 'training.csv', 'development_nodes.csv', 'development_episodes.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--maximum-checks', type=int, default=48)
    args = parser.parse_args()
    if not 1 <= args.maximum_checks <= 48:
        raise ValueError('Finite 24h observation only')
    args.output.mkdir(parents=True, exist_ok=False)
    for check in range(args.maximum_checks):
        states = {}
        for host, root in HOSTS.items():
            destination = args.output/host
            if (destination/'receipt.json').is_file():
                states[host] = 'archived'
                continue
            remote = root+'/crossseed900K-complete-analysis-20261120'
            try:
                observed = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                    host, 'test -f '+shlex.quote(remote+'/receipt.json')+
                    ' && cat '+shlex.quote(remote+'/receipt.json')],
                    capture_output=True, text=True, timeout=45)
            except subprocess.TimeoutExpired:
                states[host] = 'SSH_observation_timed_out;training_state_not_inferred'
                continue
            if observed.returncode:
                states[host] = 'summary_not_observed;training_state_not_inferred'
                continue
            receipt = json.loads(observed.stdout)
            if receipt.get('status') != 'complete_crossseed_one_host_development_summary':
                raise ValueError('Unexpected native completion summary')
            if shutil.disk_usage(args.output).free < 1024**3:
                raise OSError('Preserve at least1GiB local free space; no model downloads')
            destination.mkdir(exist_ok=True)
            # Archive receipt LAST so an interrupted transfer is never trusted
            # as complete. Each retry reads only already finalized compact files.
            try:
                for name in (*FILES[1:], FILES[0]):
                    subprocess.run(['rsync', '-a', '-e',
                        'ssh -o BatchMode=yes -o ConnectTimeout=12',
                        host+':'+remote+'/'+name, str(destination)+'/'], check=True, timeout=120)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                states[host] = 'compact_archive_incomplete;retry_readonly_transfer_next_poll'
                continue
            states[host] = 'archived'
        snapshot = dict(check=check+1, unix_time=time.time(), states=states,
            poll_interval_seconds=1800, training_launches=0, github_writes=0, messages_sent=0)
        (args.output/'latest_observation.json').write_text(json.dumps(snapshot, indent=2)+'\n')
        print(json.dumps(snapshot), flush=True)
        if all(state == 'archived' for state in states.values()):
            subprocess.run([sys.executable, '-m', 'scripts.paper_crax_payload.merge_reach_crossseed_20261120',
                '--a100', str(args.output/'tjucta-a100'), '--rtx4080', str(args.output/'4080'),
                '--registration', str(args.registration), '--output', str(args.output/'combined')], check=True)
            print('Both seeds archived and merged; no candidate frozen and no GitHub write.', flush=True)
            return
        if check+1 < args.maximum_checks:
            time.sleep(1800)
    raise TimeoutError('Finite archive wait ended; native training remains untouched')


if __name__ == '__main__':
    main()

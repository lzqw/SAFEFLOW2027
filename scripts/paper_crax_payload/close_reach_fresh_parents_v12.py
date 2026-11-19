"""Finite30min native completion wait; never launch downstream training."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.paper_crax_payload.close_samehardware_v12 import terminal_state


ROOT = Path('/storage_0/SAFEFLOW2027_lzqw_uid1007/simulation_v12')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maximum-checks',type=int,default=32)
    p.add_argument('--independent-radius',action='store_true',help='Observe the fixed300K to400K two-seed radius stage only')
    args = p.parse_args()
    if not 1<=args.maximum_checks<=48:
        raise ValueError('Finite observation only; at most24h')
    if (not ROOT.is_dir() or shutil.disk_usage(ROOT).free<41*1024**3):
        raise OSError('Preserve data-disk40GiB plus1GiB summary allowance')
    label='independent-radius' if args.independent_radius else 'reach-fresh-parents'
    operation = ROOT/(label+'-closure')
    operation.mkdir(exist_ok=False)
    for check in range(args.maximum_checks):
        queue = json.loads((ROOT/('queue_independent_radius.json' if args.independent_radius else 'queue_reach_fresh_parents.json')).read_text())
        source_commit='10ba914c' if args.independent_radius else '22d4966f'
        prefix='Reach-EA-independent-radius100K-' if args.independent_radius else 'Reach-EA-fresh300K-'
        if (queue['source_commit']!=source_commit
                or [j['name'] for j in queue['jobs']]!=[prefix+'41028',prefix+'41029']):
            raise ValueError('Observe only the registered two independent native parents')
        complete = terminal_state(queue)
        snapshot = dict(check=check+1,unix_time=time.time(),queue_status=queue['status'],
            ready_for_complete_summary=complete,interval_seconds=1800,training_launches=0,
            downstream_training_started=False,automatic_retry=False)
        (operation/f'check_{check+1:02d}.json').write_text(json.dumps(snapshot,indent=2)+'\n')
        print(json.dumps(snapshot),flush=True)
        if complete:
            command=([sys.executable,'-m','scripts.paper_crax_payload.summarize_independent_radius_v12',
                '--runs',str(ROOT/'runs-independent-radius'),'--parents',str(ROOT/'runs-reach-fresh-parents'),
                '--output',str(ROOT/'independent-radius-complete-analysis')] if args.independent_radius else
                [sys.executable,'-m','scripts.paper_crax_payload.summarize_reach_fresh_parents_v12',
                '--runs',str(ROOT/'runs-reach-fresh-parents'),
                '--output',str(ROOT/'reach-fresh-parents-complete-analysis'),'--require-complete'])
            subprocess.run(command,
                cwd=Path(__file__).resolve().parents[2],check=True)
            print('Both registered native endpoints and all DEV nodes summarized;no downstream launch.',flush=True)
            return
        if check+1<args.maximum_checks:
            time.sleep(1800)
    raise TimeoutError('Observation ended;native training remains untouched and is not marked failed')


if __name__=='__main__':
    main()

"""Node-local half-hour READ-ONLY resource/process snapshots, never restart jobs.

One recorder per participating host. Finite queues remain the only launchers.
No remote credential changes, foreign-process commands or scientific decisions.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


MODULES={'scripts.paper_crax_payload.train_crax','scripts.paper_crax_payload.native_sac_reference',
    'scripts.paper_crax_payload.native_sac_checkpoint','scripts.paper_crax_payload.payload_train',
    'scripts.paper_crax_payload.piper_train','scripts.paper_crax_payload.run_queue',
    'scripts.paper_crax_payload.train_component_study','scripts.paper_crax_payload.train_cvpo',
    'scripts.paper_crax_payload.cvpo_campaign','scripts.paper_crax_payload.train_direction_development',
    'scripts.paper_crax_payload.train_direction_confirmation',
    'scripts.paper_crax_payload.train_direction_annealing',
    'scripts.paper_crax_payload.train_direction_optimization',
    'scripts.paper_crax_payload.train_direction_confirmation_v7'}


def own_processes(project):
    rows=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            if entry.stat().st_uid!=os.getuid() or Path(os.readlink(entry/'cwd')).resolve()!=project:continue
            args=[p.decode() for p in (entry/'cmdline').read_bytes().split(b'\0') if p]
            if '-m' not in args:continue
            module=args[args.index('-m')+1]
            if module not in MODULES:continue
            rows.append(dict(pid=int(entry.name),module=module,command=args))
        except (OSError,IndexError,UnicodeError):continue
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--interval',type=int,default=1800);p.add_argument('--samples',type=int,default=336)
    args=p.parse_args();project=args.project.resolve()
    if args.interval<1800 or args.samples<1:raise ValueError('Half-hour snapshots,not busy polling')
    args.snapshot.parent.mkdir(parents=True,exist_ok=True)
    for index in range(args.samples):
        if args.snapshot.with_suffix('.stop').exists():break
        processes=own_processes(project)
        try:
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader'],text=True,timeout=15).strip()
        except (OSError,subprocess.SubprocessError):gpu=None
        completed=[]
        for receipt in sorted((project.parent/'runs').glob('*/receipt.json')):
            try:
                raw=json.loads(receipt.read_text())
                completed.append(dict(run=receipt.parent.name,status=raw.get('status'),steps=raw.get('steps')))
            except (OSError,ValueError):continue
        snapshot=dict(time_utc=datetime.now(timezone.utc).isoformat(),observer_pid=os.getpid(),
            interval_seconds=args.interval,sample=index+1,project=str(project),actual_native_processes=processes,
            nvidia_smi=gpu,load_average=list(os.getloadavg()),free_disk_bytes=shutil.disk_usage(project).free,
            native_receipts=completed,whole_goal_complete=False,automatic_restart=False,
            training_launches=0,foreign_processes_changed=False,
            idle_observation_only_not_goal_completion=not bool(processes))
        temp=args.snapshot.with_suffix('.tmp');temp.write_text(json.dumps(snapshot,indent=2)+'\n');temp.replace(args.snapshot)
        print(json.dumps(dict(time_utc=snapshot['time_utc'],sample=index+1,
            actual_native_pids=[r['pid'] for r in processes],observer_only=True)),flush=True)
        if index+1<args.samples:time.sleep(args.interval)


if __name__=='__main__':main()

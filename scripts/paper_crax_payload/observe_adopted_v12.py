"""Observe two explicitly adopted live training PIDs; never launch/restart/signal."""
import argparse
import json
from pathlib import Path
import time

from scripts.star.arm_safe_entropy.train import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True)
    args=p.parse_args();value=json.loads(args.registration.read_text())
    records=value['adopted_jobs']
    if args.state.exists():raise FileExistsError('Do not restart adopted observers blindly')
    for r in records:
        cmd=Path('/proc',str(r['pid']),'cmdline').read_bytes()
        if str(r['output']).encode() not in cmd or b'train_safereach_early_v12' not in cmd:
            raise ValueError('Adopt only exact currently live native handles')
        r['proc_start_ticks']=Path('/proc',str(r['pid']),'stat').read_text().split(') ',1)[1].split()[19]
    while True:
        live=[]
        for r in records:
            if r['status'] in ('completed','failed_not_restarted'):continue
            d=Path('/proc',str(r['pid']))
            try:
                cmd=(d/'cmdline').read_bytes();ticks=(d/'stat').read_text().split(') ',1)[1].split()[19]
            except FileNotFoundError:cmd=b'';ticks=None
            if str(r['output']).encode() in cmd and ticks==r['proc_start_ticks']:
                live.append(r['pid']);continue
            receipt=Path(r['output'])/'receipt.json'
            result=json.loads(receipt.read_text()) if receipt.exists() else {}
            ok=result.get('status')=='completed' and result.get('steps')==100000
            r.update(status='completed' if ok else 'failed_not_restarted',finished_unix=time.time(),
                exit_code=None,completion_evidence='native100Kreceipt_and_handle_exit' if ok else 'handle_exit_without_complete_receipt')
        status='running' if live else ('completed_with_failures' if any(r['status']=='failed_not_restarted' for r in records) else 'completed')
        atomic_json(args.state,dict(pid=__import__('os').getpid(),status=status,jobs=records,pending=0,
            live_native_pids=live,automatic_restart=False,foreign_processes_changed=False,
            source_commit=value['training_source_commit'],offloaded_pending_jobs=value['offloaded_pending_jobs'],
            handoff_registration=str(args.registration)))
        if not live:break
        time.sleep(30)


if __name__=='__main__':main()

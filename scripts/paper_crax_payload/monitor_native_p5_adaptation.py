"""Finite read-only half-hour resource/progress monitor; never signals jobs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue-state',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(exist_ok=False)
    while True:
        q=json.loads(args.queue_state.read_text())
        rows=[]
        for job in q['jobs']:
            native=Path('/proc',str(job['pid']),'cmdline')
            command=native.read_bytes() if native.exists() else b''
            expected_module = 'scripts.paper_crax_payload.train_native_p5_adaptation'
            registered = job.get('command', [])
            if '-m' in registered:
                expected_module = registered[registered.index('-m')+1]
            live=str(job['output']).encode() in command and expected_module.encode() in command
            path=Path(job['output'])/'status.json'
            rows.append(dict(name=job['name'],pid=job['pid'],native_handle_live=live,
                queue_status=job['status'],progress=json.loads(path.read_text()) if path.exists() else None))
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
            '--format=csv,noheader'],text=True,timeout=20).strip()
        now=time.time()
        result=dict(unix_time=now,queue_status=q['status'],pending=q['pending'],jobs=rows,
            gpu_free_memory_and_utilization=gpu,data_disk_free_gib=shutil.disk_usage(args.queue_state.parent).free/1024**3,
            poll_interval_seconds=1800,foreign_processes_changed=False)
        (args.output/f'{int(now)}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        if q['status'] in ('completed','completed_with_failures') and not any(r['native_handle_live'] for r in rows):
            break
        time.sleep(1800)


if __name__=='__main__':
    main()

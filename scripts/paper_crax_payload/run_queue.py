"""Simple finite shared-GPU job slots. No restart daemon or foreign-process signals."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    project=Path(plan['project']);state_path=Path(plan['state']);state_path.parent.mkdir(parents=True,exist_ok=True)
    if state_path.exists():raise FileExistsError('Inspect previous native handles; do not restart a queue blindly')
    if not 1<=plan['slots']<=2:raise ValueError('Only bounded shared slots allowed')
    jobs=plan['jobs'];live={};records=[]
    def save():
        state=dict(pid=os.getpid(),source_commit=args.source_commit,status='running' if jobs or live else 'completed',
            jobs=records,pending=len(jobs),live_native_pids=list(live),automatic_restart=False,foreign_processes_changed=False)
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
    while jobs or live:
        for pid,(proc,stream,record) in list(live.items()):
            code=proc.poll()
            if code is not None:
                stream.close();record.update(exit_code=code,status='completed' if code==0 else 'failed_not_restarted',finished_unix=time.time());del live[pid]
        while jobs and len(live)<plan['slots']:
            job=jobs.pop(0);log=Path(job['log']);log.parent.mkdir(parents=True,exist_ok=True)
            if Path(job['output']).exists():raise FileExistsError('Preserve original run output')
            command=[plan['python'],'-u','-m',job['module'],*job['arguments'],'--output',job['output'],'--source-commit',args.source_commit]
            env=os.environ|{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','XLA_PYTHON_CLIENT_PREALLOCATE':'false'}
            env.update(job.get('environment',{}))
            stream=log.open('x');proc=subprocess.Popen(command,cwd=project,env=env,stdout=stream,stderr=subprocess.STDOUT)
            record=dict(name=job['name'],pid=proc.pid,command=command,output=job['output'],log=str(log),status='running',started_unix=time.time())
            records.append(record);live[proc.pid]=(proc,stream,record)
        save()
        if jobs or live:time.sleep(5)


if __name__=='__main__':main()

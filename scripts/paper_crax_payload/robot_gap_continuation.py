"""One evidence-triggered Payload continuation pair; no curriculum or retries.

Continue each completed100K transfer endpoint to300K, preserving native full
state and identical settings. The plan authorizes200K/method after learning.
All resource checks are30minutes; foreign jobs are never modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def evidence(root):
    parents = []
    for method in ('Iso', 'EA'):
        run = root/'runs'/f'transfer-{method.lower()}-100k'
        receipt = json.loads((run/'receipt.json').read_text())
        evaluation = json.loads((run/'evaluation_100000.json').read_text())
        nominal = next(c for c in evaluation['conditions'] if c['condition']=='nominal')
        if (receipt['status']!='completed' or receipt['newly_collected_transitions']!=100000
                or nominal['safe_success'] < .5 or receipt['task']!='Payload'
                or receipt['method']!=method or not (run/'full_100000.pt').is_file()):
            raise ValueError('Require the completed paired100K endpoints and≥50% nominal learning')
        parents.append(receipt['task_transfer']['parent_sha256'])
    if len(set(parents))!=1:
        raise ValueError('Both transfers must share their original Reach Actor parent')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--assets-receipt', type=Path, required=True)
    parser.add_argument('--training-source-commit', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    path = root/'continuation_status.json'
    if path.exists():
        raise FileExistsError('Inspect existing continuation; never duplicate/restart')
    if os.stat(root).st_dev==os.stat('/home/lzqw').st_dev:
        raise ValueError('Use the separate A100 data filesystem')
    evidence(root)
    if args.validate_only:
        print('Completed paired100K learning evidence accepted; no training launched')
        return
    status = dict(state='waiting_for_shared_resources', controller_pid=os.getpid(),
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training_source_commit=args.training_source_commit, interval_seconds=1800,
        decision='Both100K endpoints show≥50% nominal safe success; continue the matched endpoint pair, no curriculum',
        reserved_new_training=400000, total_round_reservation=830000, round_cap=2200000,
        piper_payload_reserved_total=600000, piper_payload_cap=800000,
        endpoint='300K cumulative Payload,200K additional per method',
        no_intermediate_checkpoint_reselection=True, automatic_failure_restart=False,
        other_projects_modified=False, physical_robot_interactions=0, jobs=[])
    processes = {}
    while True:
        for method, gpu in (('Iso',0),('EA',1)):
            job = next((j for j in status['jobs'] if j['method']==method),None)
            if job is not None:
                if job['status']=='running':
                    code = processes[method].poll()
                    if code is not None:
                        receipt_path = Path(job['output'])/'receipt.json'
                        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
                        ok = (code==0 and receipt.get('status')=='completed'
                            and receipt.get('newly_collected_transitions')==200000)
                        job.update(status='completed' if ok else 'failed_no_automatic_restart',
                            exit_code=code,actual_steps=receipt.get('newly_collected_transitions'),
                            finished_unix=time.time())
                continue
            if shutil.disk_usage(root).free < 40*2**30:
                status['deferred_reason']='Data disk below40GiB headroom'; continue
            # The older mechanism controller owns its own analysis slot. Do
            # not overlap a live native query process on that same GPU.
            analysis_path=root/'final_analysis_status.json'
            analysis=json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
            if analysis.get('state')=='running' and analysis.get('GPU')==gpu:
                status['deferred_reason']='Own final mechanism occupies this slot'; continue
            try:
                rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                    '--format=csv,noheader,nounits'],text=True,timeout=10)
                free={int(r.split(',')[0]):int(r.split(',')[1]) for r in rows.splitlines()}
            except (OSError,subprocess.SubprocessError) as error:
                status['deferred_reason']=str(error); continue
            if free.get(gpu,0)<2048:
                status['deferred_reason']=f'GPU{gpu} below2GiB free; sharing preserved'; continue
            name=f'transfer-{method.lower()}-300k'
            output=root/'runs'/name
            if output.exists():
                raise FileExistsError(f'Preserve existing run: {output}')
            parent=root/'runs'/f'transfer-{method.lower()}-100k/full_100000.pt'
            command=[sys.executable,'-u','-m','scripts.paper_crax_payload.piper_train','train',
                '--config',str(root/'configs'/f'transfer-{method.lower()}-100k.json'),
                '--assets-receipt',str(args.assets_receipt),'--output',str(output),
                '--device',f'cuda:{gpu}','--task','Payload','--method',method,
                '--steps','300000','--seed','41027','--source-commit',args.training_source_commit,
                '--resume',str(parent)]
            with (root/'logs'/f'{name}.log').open('x') as stream:
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            processes[method]=process
            status['jobs'].append(dict(name=name,method=method,pid=process.pid,status='running',
                target_steps=200000,cumulative_target_steps=300000,initial_steps=100000,
                engineering=False,output=str(output),command=command,started_unix=time.time()))
        status['updated_unix']=time.time()
        if len(status['jobs'])==2 and all(j['status']!='running' for j in status['jobs']):
            status['state']='completed_pending_scientific_decision' if all(j['status']=='completed' for j in status['jobs']) else 'needs_inspection_no_restart'
            save(path,status); break
        status['state']='running_or_waiting_for_shared_resources'
        save(path,status)
        time.sleep(1800)


if __name__=='__main__':
    main()

"""Queue one paired finite-Reach adaptation after each own transfer slot ends.

Checks every30minutes. Never pauses foreign jobs, launches a5080 job, restarts
a failed run, continues PointGoal RL, or spends optional continuation budgets.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.star.arm_safe_entropy.train import atomic_json


def native_process_alive(pid, output):
    path = Path('/proc', str(pid), 'cmdline')
    try:
        command = path.read_bytes()
        return b'piper_train' in command and str(output).encode() in command
    except FileNotFoundError:
        return False


def free_gpu_mib():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    return {int(row.split(',')[0]): int(row.split(',')[1]) for row in rows.splitlines()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--assets-receipt', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    status_path = root/'followup_status.json'
    if status_path.exists():
        raise FileExistsError('Inspect existing scheduler/jobs instead of duplicating or restarting them')
    diagnosis = json.loads((root/'risk-diagnostic/summary.json').read_text())
    reach = [r for r in diagnosis['rollout_summary'] if r['task']=='Reach']
    if len(reach) != 2 or len({(r['safe_success_count'],r['violation_count']) for r in reach}) != 1:
        raise ValueError('This frozen decision is the no-observed-benefit radius-prior case only')
    if diagnosis['training_transitions'] != 0 or not diagnosis['no_state_pool_reselection']:
        raise ValueError('Use the actual predeclared frozen development diagnostic')
    status = dict(protocol='SAFEFlow_Robot_Gap_v2_Reach_followup', controller_pid=__import__('os').getpid(),
        state='waiting_for_transfer_slots', source_commit=args.source_commit,
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reserved_new_training=200000, total_round_reservation_including_initial_phase=430000,
        new_round_cap=2200000, category='piper_reach_geometry_adaptation', category_cap=200000,
        decision='radius value prior improved calibration but not paired frozen behavior; use shared finite training radii',
        geometry_family=[.045,.05,.055], parent_budget_not_charged_again=True,
        interval_seconds=1800, physical_robot_interactions=0, pointgoal_new_training=0,
        healthy_other_projects_untouched=True, automatic_failure_restart=False, jobs=[])
    processes = {}
    while True:
        campaign = json.loads((root/'campaign_status.json').read_text())
        for method, gpu in (('Iso',0), ('EA',1)):
            job = next((j for j in status['jobs'] if j['method']==method), None)
            if job is not None:
                if job['status']=='running':
                    process = processes[method]
                    code = process.poll()
                    if code is not None:
                        receipt_path = Path(job['output'])/'receipt.json'
                        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
                        ok = code==0 and receipt.get('status')=='completed' and receipt.get('newly_collected_transitions')==100000
                        job.update(status='completed' if ok else 'failed_no_automatic_restart',
                            exit_code=code, finished_unix=time.time(), actual_steps=receipt.get('newly_collected_transitions'))
                continue
            prior = next((j for j in campaign['jobs'] if j['name']==f'transfer-{method.lower()}-100k'), None)
            if prior is None:
                continue
            receipt_path = Path(prior['output'])/'receipt.json'
            if native_process_alive(prior['pid'], prior['output']) or not receipt_path.exists():
                continue
            receipt = json.loads(receipt_path.read_text())
            if receipt.get('status')!='completed' or receipt.get('newly_collected_transitions')!=100000:
                continue
            if shutil.disk_usage(root).free < 40*2**30:
                status['deferred_reason']='data-disk headroom below40GiB'; continue
            try:
                available = free_gpu_mib().get(gpu,0)
            except (OSError,subprocess.SubprocessError) as error:
                status['deferred_reason']=str(error); continue
            if available < 2048:
                status['deferred_reason']=f'GPU{gpu} free memory below2GiB; other projects retained'; continue
            config = json.loads((root/'configs'/f'transfer-{method.lower()}-100k.json').read_text())
            config = deepcopy(config)
            config['training_radius_values']=[.045,.05,.055]
            name=f'reach-radius-{method.lower()}-100k'
            cfg, out = root/'configs'/f'{name}.json', root/'runs'/name
            if out.exists():
                raise FileExistsError(f'Existing branch requires inspection: {out}')
            atomic_json(cfg,config)
            parent=root.parent/'home_offload_20261113'/'SAFEFLOW2027_PAPER/runs'/f'piper-reach-{method.lower()}-41027-v1/full_300000.pt'
            command=[sys.executable,'-u','-m','scripts.paper_crax_payload.piper_train','train',
                '--config',str(cfg),'--assets-receipt',str(args.assets_receipt),'--output',str(out),
                '--device',f'cuda:{gpu}','--task','Reach','--method',method,'--steps','400000',
                '--seed','41027','--source-commit',args.source_commit,'--adapt-from',str(parent)]
            with (root/'logs'/f'{name}.log').open('x') as stream:
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            processes[method]=process
            status['jobs'].append(dict(name=name,method=method,pid=process.pid,device=f'cuda:{gpu}',
                status='running',target_steps=100000,cumulative_target_steps=400000,
                initial_steps=300000,engineering=False,output=str(out),command=command,started_unix=time.time()))
        status['updated_unix']=time.time()
        if len(status['jobs'])==2 and all(j['status']!='running' for j in status['jobs']):
            status['state']='completed_pending_scientific_decision' if all(j['status']=='completed' for j in status['jobs']) else 'needs_inspection_no_restart'
            atomic_json(status_path,status)
            break
        status['state']='adaptation_running_or_waiting_for_own_transfer_slots'
        atomic_json(status_path,status)
        time.sleep(1800)


if __name__=='__main__':
    main()

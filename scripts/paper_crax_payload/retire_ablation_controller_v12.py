"""One-time pending-only handoff; signal exact queue controller, never training."""
import json
import os
from pathlib import Path
import signal
import time


ROOT = Path('/storage_0/SAFEFLOW2027_lzqw_uid1007/simulation_v12')


def main():
    state = ROOT/'queue_matched_ablation.json'
    evidence = ROOT/'matched_ablation_handoff_registration.json'
    if evidence.exists():
        raise FileExistsError('Inspect the completed/partial handoff before doing anything again')
    value = json.loads(state.read_text()); pid = value['pid']
    if pid != 786815 or value['pending'] != 2 or len(value['jobs']) != 2:
        raise ValueError('Require exact controller and two unstarted jobs')
    cmd = Path('/proc', str(pid), 'cmdline').read_bytes()
    if b'scripts.paper_crax_payload.run_queue' not in cmd or b'simulation_v12_queue_a100_matched_ablation.json' not in cmd:
        raise ValueError('Controller command identity differs')
    expected = {'A-full-P5-2034': 786854, 'A-no-direction-2034': 786855}
    for record in value['jobs']:
        if record['name'] not in expected or record['pid'] != expected[record['name']] or record['status'] != 'running':
            raise ValueError('Native children changed; do not hand off automatically')
        child = Path('/proc', str(record['pid']), 'cmdline').read_bytes()
        if record['output'].encode() not in child or b'train_matched_ablation_v12' not in child:
            raise ValueError('Exact original training handle must still be live')
    # Stop only the queue controller to close the pending-launch race. Native
    # training children are not in a signalled process group and keep running.
    os.kill(pid, signal.SIGSTOP)
    retired = False
    try:
        frozen = json.loads(state.read_text())
        if frozen['pending'] != 2 or frozen['jobs'] != value['jobs']:
            raise ValueError('Queue changed before freezing; keep all original work')
        record = dict(protocol='v12_matched_ablation_pending_only_handoff',
            original_queue_pid=pid, original_queue_path=str(state),
            training_source_commit=value['source_commit'], training_processes_signalled=False,
            training_steps_replayed=0, unix_time=time.time(), adopted_jobs=value['jobs'],
            offloaded_pending_jobs=[dict(name=name, host='4080',
                output='/home/carla/4T/SAFEFLOW2027_SIMULATION_V12/runs-matched-offload/'+name)
                for name in ('A-no-lateLR-2034', 'A-Metric-matchedLR-2034')])
        for job in record['adopted_jobs']:
            job['expected_module'] = 'scripts.paper_crax_payload.train_matched_ablation_v12'
        with (ROOT/'queue_matched_ablation_before_handoff.json').open('x') as stream:
            json.dump(frozen, stream, indent=2)
        with evidence.open('x') as stream:
            json.dump(record, stream, indent=2)
        os.kill(pid, signal.SIGTERM); retired = True
    finally:
        os.kill(pid, signal.SIGCONT)
    print(json.dumps(dict(controller_retired=retired, registration=str(evidence),
        training_processes_signalled=False, automatic_restart=False)), flush=True)


if __name__ == '__main__': main()

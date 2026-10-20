#!/usr/bin/env python3
"""Restart only an idle archive coordinator; never signal a remote learner."""
import concurrent.futures
import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

import archive_scientific_evidence as archive

BASE = Path(__file__).resolve().parent
UNIT = 'safeflow-scientific-evidence-archive.service'
RECEIPT = BASE / 'archive_background_priority_deployment_20261021_receipt.json'
BOOTS = {'tjucta-a100': '6928f333-1036-473c-92d5-4c482d5fef59',
         '4080': '3b362c03-4ed3-4e03-aea4-cf7339356026',
         '5080': '5f46bb0c-b4ec-4802-8f6e-01ac611c0f2f'}


def show():
    r = subprocess.run(['systemctl', '--user', 'show', UNIT, '-p', 'MainPID', '-p', 'ActiveState'],
                       capture_output=True, text=True, check=True, timeout=15)
    return dict(line.split('=', 1) for line in r.stdout.splitlines() if '=' in line)


def probe(host):
    code = "import json,os;from pathlib import Path;print(json.dumps({'nice':os.getpriority(os.PRIO_PROCESS,0),'affinity':sorted(os.sched_getaffinity(0)),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))"
    command = archive.background_prefix(host) + 'python3 -c ' + shlex.quote(code)
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', host, command],
                       capture_output=True, text=True, check=True, timeout=30)
    value = json.loads(r.stdout)
    assert value['nice'] >= 10 and value['boot_id'] == BOOTS[host]
    if host == '4080':
        assert value['affinity'] == list(range(16, 28))
    return host, value


def main():
    assert not RECEIPT.exists(), 'Inspect the completed deployment; never replay'
    assert archive.SOURCE == '62ca7399357c10645a955dcbead7ed2153a82c69'
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        probes = dict(pool.map(probe, archive.HOSTS))
    print(json.dumps({'helper_priority_probes_passed': probes}), flush=True)
    old = 993
    process = Path('/proc', str(old))
    ticks = '736'
    assert show() == {'MainPID': str(old), 'ActiveState': 'active'}
    assert process.stat().st_uid == os.getuid()
    assert 'archive_scientific_evidence.py --run' in (process / 'cmdline').read_bytes().decode().replace(chr(0), ' ')
    frozen = False
    deadline = time.monotonic() + 900
    try:
        while time.monotonic() < deadline:
            assert (process / 'stat').read_text().split()[21] == ticks
            state = json.loads(archive.STATE.read_text())
            idle = (not state['pending'] and not (process / 'task/993/children').read_text().strip()
                    and (process / 'wchan').read_text().strip() in ('do_select', 'hrtimer_nanosleep')
                    and state.get('last_success_at_unix', 0) + 1800 - time.time() >= 60)
            if idle:
                os.kill(old, signal.SIGSTOP)
                frozen = True
                stable = json.loads(archive.STATE.read_text())
                assert (process / 'stat').read_text().split()[21] == ticks
                assert not stable['pending'] and not (process / 'task/993/children').read_text().strip()
                assert stable['last_success_at_unix'] == state['last_success_at_unix']
                break
            time.sleep(5)
        else:
            raise RuntimeError('No idle archive window; do not stop active snapshot/transfer')
        started = time.time()
        subprocess.run(['systemctl', '--user', 'restart', UNIT], check=True, capture_output=True,
                       text=True, timeout=45)
        frozen = False
        new = show()
        assert new['ActiveState'] == 'active' and int(new['MainPID']) not in (0, old)
        new_pid = int(new['MainPID'])
        assert 'archive_scientific_evidence.py --run' in Path('/proc', str(new_pid), 'cmdline').read_bytes().decode().replace(chr(0), ' ')
        after = json.loads(archive.STATE.read_text())
        assert not after['pending'] and after['last_success_at_unix'] == stable['last_success_at_unix']
        receipt = {
            'kind': 'archive_helper_background_priority_not_learner_or_scientific_change',
            'completed_at_unix': time.time(), 'local_restart_seconds': time.time() - started,
            'old_idle_coordinator_PID': old, 'old_start_ticks': int(ticks), 'new_local_coordinator_PID': new_pid,
            'idle_pending_transfers': 0, 'idle_children': 0, 'archive_unit': UNIT,
            'verified_new_remote_helper_process_properties': probes,
            'command_prefixes': {h: archive.background_prefix(h) for h in archive.HOSTS},
            'before_on_disk_helper_sha256': hashlib.sha256((BASE / 'archive_scientific_evidence_before_background_priority_20261021.py').read_bytes()).hexdigest(),
            'new_helper_sha256': hashlib.sha256((BASE / 'archive_scientific_evidence.py').read_bytes()).hexdigest(),
            'existing_last_success_unix': stable['last_success_at_unix'],
            'next_archive_not_before_unix': stable['last_success_at_unix'] + 1800,
            'no_extra_archive_cycle_on_restart': True,
            'archive_bytes_hash_mode_mtime_and_reserve_rules_unchanged': True,
            'learner_signals_restarts_or_affinity_changes': 0,
            'foreign_projects_modified': False, 'scientific_parameters_or_budgets_changed': False,
            'extra_evaluation_rollouts_model_loads_or_final_audits': 0,
            'GPU_resource_monitor986_not_restarted': True}
        with RECEIPT.open('x') as stream:
            stream.write(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)
    finally:
        if frozen and process.exists() and (process / 'stat').read_text().split()[21] == ticks:
            os.kill(old, signal.SIGCONT)


if __name__ == '__main__':
    main()

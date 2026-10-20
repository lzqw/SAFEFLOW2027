#!/usr/bin/env python3
"""Finish evidence transfer for the already-live GPU recovery; never relaunch it."""
import hashlib
import importlib.util
import json
import re
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
LAUNCHER = BASE / 'recover_goal2028_cpu_failure_to_5080_20261020_0830.py'
LAUNCHER_SHA = 'dee09ea813b961829887ab740c6cce80cf327927d0f36192d6af2f4c0469a5fb'
PREFIX = 'point-goal-2028_cpu_failure_gpu_recovery_20261020_0830'
STATE = BASE / (PREFIX + '_completion_state.json')
OBSERVATION = BASE / 'capacity_progress_observation_20261020_goal2028_cpu_failure_gpu_target_launch.json'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    assert not STATE.exists(), 'Inspect prior completion, never duplicate'
    assert digest(LAUNCHER) == LAUNCHER_SHA
    spec = importlib.util.spec_from_file_location('original_gpu_recovery_launcher', LAUNCHER)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    h = original.load_helper()
    raw = json.loads(original.STATE.read_text())
    receipt = json.loads(original.RECEIPT.read_text())
    assert raw['phase'] == 'needs_inspection_after_cpu_failure_gpu_recovery_error'
    assert raw['target_launch_requested'] is True and 'scp' in raw['error']
    assert raw['controller_sha256'] == receipt['controller_sha256'] == LAUNCHER_SHA
    observed = json.loads(OBSERVATION.read_text())
    expected = next(w for w in observed['hosts']['5080']['training'] if w['job'] == original.JOB)
    assert expected['pid'] == 186026 and expected['start_ticks'] == 8241742
    assert expected['device'] == 'cuda:0' and expected['progress_csv']
    assert expected['boot_id'] == raw['plan']['expected_target_boot_id']
    identity_keys = ('pid', 'start_ticks', 'device')
    live = h.status(h.DST)
    assert len(live['workers']) == 1 and live['boot_id'] == expected['boot_id']
    assert all(live['workers'][0][key] == expected[key] for key in identity_keys)
    assert not h.status(h.SRC)['workers']
    state = {**raw, 'phase': 'finishing_host_copy_without_restarting_live_gpu_target',
             'completion_controller_sha256': digest(Path(__file__)),
             'completion_controller_filename': Path(__file__).name,
             'raw_launch_controller_error_record': original.STATE.name,
             'raw_launch_controller_error_record_sha256': digest(original.STATE),
             'prior_launch_controller_error': raw['error'],
             'target_process_not_restarted': True,
             'independent_target_launch_observation_file': OBSERVATION.name,
             'expected_live_target_identity': expected}
    state.pop('error', None)
    original.save(STATE, state)
    try:
        original.HOST.mkdir(parents=True, exist_ok=True)
        files = []
        for artifact in receipt['source_artifacts']:
            # Each path is a single argv item. Extra literal shell quotes made
            # legacy SCP's filename check reject the previous transfer.
            source = artifact['source_path']
            assert re.fullmatch(r'[A-Za-z0-9_./{}-]+', source)
            destination = original.HOST / artifact['relative_path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            valid = (destination.is_file() and destination.stat().st_size == artifact['bytes']
                     and digest(destination) == artifact['sha256'])
            if not valid:
                if destination.exists():
                    destination.rename(destination.with_name(destination.name + '.unverified_copy_' + str(time.time_ns())))
                subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                                h.SRC + ':' + source, str(destination)], check=True, timeout=1800)
            assert destination.stat().st_size == artifact['bytes'] and digest(destination) == artifact['sha256']
            files.append({**artifact, 'host_path': str(destination)})
        manifest = {'job': original.JOB, 'source_input_sha256': receipt['checkpoint_sha256'],
                    'files': files, 'raw_native_bytes_preserved': True,
                    'launch_controller_sha256': LAUNCHER_SHA,
                    'completion_controller_sha256': state['completion_controller_sha256']}
        original.save(original.HOST / 'host_artifact_manifest.json', manifest)
        original.save(BASE / (PREFIX + '_host_manifest.json'), manifest)
        state.update(phase='host_evidence_preserved_waiting_existing_gpu_first_epoch',
                     host_native_evidence_and_full_input_preserved=True,
                     host_copy_error_resolved=True)
        original.save(STATE, state)
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            live = h.status(h.DST)
            workers = live['workers']
            assert live['boot_id'] == expected['boot_id']
            if workers:
                assert len(workers) == 1
                assert all(workers[0][key] == expected[key] for key in identity_keys)
            new = [r for r in live['progress'] if r['csv'] == expected['progress_csv']
                   and r['steps'] >= receipt['resume_steps'] + 5000]
            if workers and new: break
            if not workers and h.ssh(h.DST, 'test ! -e ' + original.shlex.quote(h.container(h.DST) + '/exit_code.txt')
                    + ' || cat ' + original.shlex.quote(h.container(h.DST) + '/exit_code.txt')).strip():
                raise RuntimeError('Existing GPU recovery became terminal before first epoch; inspect without duplicating')
            time.sleep(30)
        else: raise RuntimeError('First epoch not observed; inspect live target, do not restart')
        target = workers[0]
        assert target['cuda_visible_devices'] == '0'
        assert not h.status(h.SRC)['workers']
        gpu_uuid = receipt['plan']['rebalance_qualification']['uuid']
        h.py(h.DST, f'''
import csv,subprocess
rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines())
assert any(uuid.strip()=={gpu_uuid!r} and int(pid.strip())=={target['pid']} for uuid,pid in rows)
''')
        src, dst = h.container(h.SRC), h.container(h.DST)
        archive = h.ROOTS[h.SRC] + '/recovery_backups/retired_cpu_workers/' + PREFIX + '_' + str(time.time_ns())
        h.py(h.SRC, f'''
from pathlib import Path
d=Path({src!r});assert (d/'exit_code.txt').read_text().strip()=='1'
archive=Path({archive!r});archive.parent.mkdir(parents=True,exist_ok=True)
assert not archive.exists() and d.stat().st_dev==archive.parent.stat().st_dev
d.rename(archive)
''')
        state.update(phase='recovery_completed_first_gpu_epoch_verified',
                     target_identity=target, first_verified_target_steps=new[-1]['steps'],
                     first_verified_target_fps=new[-1]['fps'], first_verified_target_cost=new[-1]['cost'],
                     first_verified_target_csv=new[-1]['csv'], target_gpu_uuid_verified=gpu_uuid,
                     failed_source_remains_terminal=True, failed_cpu_not_restarted=True,
                     failed_source_archive=archive, source_canonical_path_vacated=True)
        original.save(STATE, state)
        h.py(h.DST, f'from pathlib import Path; Path({dst+"/handoff_"+original.SIDE+"_completion_state.json"!r}).write_text({json.dumps(state,indent=2)+chr(10)!r})')
    except Exception as error:
        state.update(phase='needs_inspection_during_existing_gpu_recovery_completion',
                     error=str(error), automatic_retry=False, target_process_not_restarted=True)
        original.save(STATE, state)
        raise


if __name__ == '__main__':
    main()

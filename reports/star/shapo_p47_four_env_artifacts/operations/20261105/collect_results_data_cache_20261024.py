#!/usr/bin/env python3
"""Gather terminal training receipts, then strictly audit the twenty-run matrix once."""
import datetime
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from host_full_input_cache_20261020 import ensure_cache

BASE = Path(__file__).resolve().parent
STATE = BASE / 'result_collection_state.json'
COMMIT = '62ca7399357c10645a955dcbead7ed2153a82c69'
A100_ROOT = '/home/lzqw/SAFEFLOW2027_A100'
A100_PRIMARY_RUNS = A100_ROOT + '/runs/shapo_p47_four_env'
A100_RUNS = '/storage_0/SAFEFLOW2027_lzqw_uid1007/terminal_matrix_20261024/runs/shapo_p47_four_env'
TARGET_ROOT = '/home/carla/SAFEFLOW2027_4080/runs/shapo_p47_four_env'
REMOTE_RUNS = {'4080': TARGET_ROOT,
               '5080': '/home/autodrive/LZQW/SAFEFLOW2027/runs/shapo_p47_four_env'}
SOURCE_RUNS = {'tjucta-a100': A100_PRIMARY_RUNS, **REMOTE_RUNS}

HOST_FINAL = Path('/home/lzqw/SAFEFLOW2027_artifacts/recovered_a100_4080/final')
ENVIRONMENTS = {'point-goal': ('SafetyPointGoal1-v0', 10000000), 'point-button': ('SafetyPointButton1-v0', 10000000), 'ant': ('Ant-v4', 5000000), 'walker2d': ('Walker2d-v4', 5000000)}


def log(message):
    print(datetime.datetime.now().astimezone().isoformat(), message, flush=True)


def save(state):
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    os.replace(temporary, STATE)


def ssh(host, command, timeout=60):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', host, command], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=timeout)


def completed(host, root, seeds):
    code = f'''
import json
from pathlib import Path
active=set()
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=[x.decode(errors='replace') for x in p.read_bytes().split(b'\\0') if x]
  if any(x.endswith('scripts/star/train_star.py') for x in args):
   active.add(str(Path(args[args.index('--config')+1]).parent))
  if any(x.endswith('scripts/star/validate_shapo_p47_run.py') for x in args):
   active.add(args[args.index('--container')+1])
 except (OSError,ValueError,IndexError): continue
root=Path({root!r})
result=[]
for seed in {seeds!r}:
 for slug,(env,steps) in {ENVIRONMENTS!r}.items():
  d=root/'runs'/(slug+'-'+str(seed))
  if str(d) in active: continue
  try:
   r=json.loads((d/'receipt.json').read_text())
   log_dir=Path(r['log_dir'])
   if not log_dir.exists():
    relocation=json.loads((d/'artifact_relocation.json').read_text())
    relative=log_dir.relative_to(Path(relocation['original_container']))
    if relative.parts[0]!='training' or '..' in relative.parts:continue
    log_dir=d/relative
   summary=json.loads((log_dir/'artifacts/training_summary.json').read_text())
   ok=r.get('status')=='passed' and int(r.get('steps',-1))==steps and r.get('environment')==env and int(r.get('seed',-1))==seed and r.get('source_commit')=={COMMIT!r} and r.get('source_dirty') is False and summary.get('status')=='completed' and int(summary.get('env_steps',-1))==steps and summary.get('engineering_hard_stop_reason') is None
   if ok: result.append(d.name)
  except (OSError,ValueError,TypeError,KeyError,IndexError): continue
print(json.dumps(result))
'''
    return json.loads(ssh(host, 'python3 -c ' + shlex.quote(code)).stdout)


def relocation_validation_code(name, host):
    roots = SOURCE_RUNS
    if host not in SOURCE_RUNS or name not in {
        f'{slug}-{seed}' for slug in ENVIRONMENTS for seed in range(2027, 2032)
    }:
        raise ValueError('unapproved completed-container import identity')
    return f'''
import json
from pathlib import Path
def validate_previous_relocation(d):
 p=d/'artifact_relocation.json'
 if not p.exists():return
 old=json.loads(p.read_text())
 roots={roots!r}
 origin=old.get('original_host')
 if (old.get('format')!='shapo-artifact-relocation-v1'
     or old.get('job')!={name!r}
     or origin not in roots
     or old.get('target_host')!={host!r}
     or old.get('original_container')!=roots[origin]+'/runs/'+{name!r}
     or old.get('scientific_protocol_changed') is not False
     or old.get('original_evidence_modified') is not False):
  raise ValueError('unapproved relocation edge; preserve source and strict scientific gates')
'''


def import_job(name, host='4080', terminal_reserve_gib=100):
    # Transfer only a terminal container; promotion is an atomic directory rename.
    validation = relocation_validation_code(name, host)
    source_container = SOURCE_RUNS[host] + '/runs/' + name
    preflight = validation + f'''
import subprocess
d=Path({source_container!r})
if d.resolve()!=d or not d.is_dir():raise ValueError('source container is missing or symlinked')
validate_previous_relocation(d)
print(json.dumps({{'incoming_bytes':int(subprocess.check_output(['du','-sb','--',str(d)],text=True).split()[0])}}))
'''
    incoming_bytes = json.loads(ssh(host, 'python3 -c ' + shlex.quote(preflight)).stdout)['incoming_bytes']
    # Engineering-only exception for the last small, completed primary run.
    # Ordinary transfers retain the original 100 GiB reserve. Never delete
    # originals or relax any scientific verification to complete collection.
    if terminal_reserve_gib != 100 and not (
        terminal_reserve_gib == 32 and host == 'tjucta-a100'
        and name == 'point-goal-2030' and incoming_bytes <= 3 * 1024**3
    ):
        raise ValueError('terminal reserve exception is restricted to the last small A100 run')
    headroom = f'''
import shutil
free=shutil.disk_usage({A100_RUNS!r}).free
if free < {incoming_bytes}+{terminal_reserve_gib}*1024**3:
 raise RuntimeError('completed-container import deferred to preserve data-volume reserve')
'''
    relocation = {
        'format': 'shapo-artifact-relocation-v1', 'job': name,
        'original_container': SOURCE_RUNS[host] + '/runs/' + name,
        'original_host': host, 'target_host': 'tjucta-a100',
        'scientific_protocol_changed': False, 'original_evidence_modified': False,
    }
    annotate = validation + f'''
import json,sys,time
from pathlib import Path
d=Path(sys.argv[1])
p=d/'artifact_relocation.json'
if p.exists():
 validate_previous_relocation(d)
 p.rename(d/('artifact_relocation.before_collection_'+str(time.time_ns())+'.json'))
p.write_text({json.dumps(relocation, indent=2) + chr(10)!r})
'''
    command = ('set -e; python3 -c ' + shlex.quote(headroom)
               + '; import_stage=$(mktemp -d ' + shlex.quote(A100_RUNS + '/runs/.completed.XXXXXX')
               + '); tar -xzf - --no-same-owner -C "$import_stage"; test ! -e ' + shlex.quote(A100_RUNS + '/runs/' + name)
               + '; python3 -c ' + shlex.quote(annotate) + ' "$import_stage"/' + shlex.quote(name)
               + '; mv "$import_stage"/' + shlex.quote(name) + ' ' + shlex.quote(A100_RUNS + '/runs/' + name)
               + '; rmdir "$import_stage"')
    sender = 'tar -czf - -C ' + shlex.quote(SOURCE_RUNS[host] + '/runs') + ' -- ' + shlex.quote(name)
    if host == 'tjucta-a100':
        # Same-host completed sources cross filesystems by an ordinary copy;
        # never round-trip their weights through the local workstation.
        command = ('set -e; python3 -c ' + shlex.quote(headroom)
                   + '; import_stage=$(mktemp -d ' + shlex.quote(A100_RUNS + '/runs/.completed.XXXXXX')
                   + '); cp -a --reflink=auto -- ' + shlex.quote(source_container)
                   + ' "$import_stage"/' + shlex.quote(name)
                   + '; test ! -e ' + shlex.quote(A100_RUNS + '/runs/' + name)
                   + '; python3 -c ' + shlex.quote(annotate) + ' "$import_stage"/' + shlex.quote(name)
                   + '; mv "$import_stage"/' + shlex.quote(name) + ' ' + shlex.quote(A100_RUNS + '/runs/' + name)
                   + '; rmdir "$import_stage"')
        ssh('tjucta-a100', command, timeout=14400)
        log('Cached terminal primary-source run on separate data disk: ' + name)
        return
    with subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', host, sender], stdout=subprocess.PIPE) as source:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'tjucta-a100', command], stdin=source.stdout, timeout=14400)
        source.stdout.close()
        if source.wait() or result.returncode:
            raise RuntimeError('completed run transfer failed for ' + name)
    log('Imported terminal run to the A100 matrix: ' + name)


def finalize():
    prepare = f'''
from pathlib import Path
root=Path({A100_RUNS!r})
p=root/'launch_manifest.txt'
if not p.exists():
 p.write_text('source_commit={COMMIT}\\ntraining_seeds=2027,2028,2029,2030,2031\\nschedule=recovery_on_a100_and_4080_with_spare_cpu_workers\\nscientific_early_stop=false\\n')
(root/'matrix_failure_count.txt').write_text('0\\n')
(root/'formal_matrix_complete').touch()
'''
    ssh('tjucta-a100', 'python3 -c ' + shlex.quote(prepare))
    public = A100_RUNS + '/public_artifacts'
    command = ('test -f ' + shlex.quote(public + '/finalization_complete') + ' || bash '
               + shlex.quote(A100_ROOT + '/code/controller-portable-20261013/scripts/star/finalize_shapo_p47_formal.sh')
               + ' ' + shlex.quote(A100_ROOT + '/code/controller-portable-20261013') + ' ' + shlex.quote(A100_RUNS)
               + ' ' + shlex.quote(A100_ROOT + '/env/bin/python') + ' ' + shlex.quote(public)
               + ' >' + shlex.quote(A100_RUNS + '/finalization_recovered.log') + ' 2>&1')
    ssh('tjucta-a100', command, timeout=14400)
    total = int(ssh('tjucta-a100', 'du -sb ' + shlex.quote(public)).stdout.split()[0])
    if shutil.disk_usage(BASE).free < 2 * total + 1024**3:
        raise RuntimeError('not enough host disk space for the final report and Git publication')
    HOST_FINAL.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', '--partial', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=12', 'tjucta-a100:' + public + '/', str(HOST_FINAL) + '/'], check=True, timeout=14400)
    subprocess.run(['sha256sum', '-c', 'checksums.sha256'], cwd=HOST_FINAL, check=True, stdout=subprocess.DEVNULL, timeout=1800)
    log('Final twenty-run evidence is on the host: ' + str(HOST_FINAL))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finish-last-cache-with-reserve-gib', type=int,
                        choices=(32, 100), default=100)
    args = parser.parse_args()
    state = json.loads(STATE.read_text()) if STATE.exists() else {'imported': []}
    while True:
        try:
            a100_done = completed('tjucta-a100', A100_RUNS, [2027,2028,2029,2030,2031])
            training_done = set(a100_done)
            sources = {}
            for host, root in SOURCE_RUNS.items():
                target_done = completed(host, root, [2027,2028,2029,2030,2031])
                training_done.update(target_done)
                sources[host] = target_done
            # Persist the true training count before an archive failure can
            # leave the dashboard claiming an already finished run is active.
            state['training_completed_runs'] = len(training_done)
            state['completed_runs'] = len(a100_done)
            state['pending_cache_runs'] = sorted(training_done - set(a100_done))
            state['observed_at'] = datetime.datetime.now().astimezone().isoformat()
            state['completion_stage'] = ('training_complete_archive_pending'
                                         if len(training_done) == 20 and len(a100_done) < 20
                                         else 'training_receipts_collected_pending_final_strict_audit')
            save(state)
            for host, target_done in sources.items():
                for name in target_done:
                    if name not in a100_done:
                        reserve = 100
                        if (len(training_done) == 20 and len(a100_done) == 19
                                and name == 'point-goal-2030' and host == 'tjucta-a100'):
                            reserve = args.finish_last_cache_with_reserve_gib
                        import_job(name, host, terminal_reserve_gib=reserve)
                        a100_done.append(name)
                    if host != 'tjucta-a100' and name not in state['imported']:
                        state['imported'].append(name)
            state['completed_runs'] = len(a100_done)
            state['training_completed_runs'] = len(training_done)
            state['pending_cache_runs'] = sorted(training_done - set(a100_done))
            state['terminal_cache_reserve_gib'] = args.finish_last_cache_with_reserve_gib
            state['terminal_cache_run_root'] = A100_RUNS
            state['primary_source_run_root'] = A100_PRIMARY_RUNS
            state['completion_stage'] = 'training_receipts_collected_pending_final_strict_audit'
            state['observed_at'] = datetime.datetime.now().astimezone().isoformat()
            save(state)
            if len(a100_done) == 20:
                ensure_cache(BASE)
                if not state.get('final_delivered'):
                    finalize()
                    state['final_delivered'] = str(HOST_FINAL)
                    save(state)
                publication = subprocess.run(
                    [sys.executable, str(BASE / 'publish_final.py')],
                    check=True, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, timeout=7200,
                )
                state['github_publication'] = json.loads(publication.stdout)
                state['completion_stage'] = 'final_strict_audit_passed_and_published'
                save(state)
                log('Published complete report to GitHub: ' + state['github_publication']['report_url'])
                return
            log(f'Terminal training receipts: {len(a100_done)}/20; final strict audit pending')
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            log('Result collection needs another observation or repair: ' + str(error))
            state['last_error'] = str(error)
            state['error_observed_at'] = datetime.datetime.now().astimezone().isoformat()
            if isinstance(error, subprocess.CalledProcessError) and error.stderr:
                state['last_error_stderr'] = error.stderr[-8000:]
            save(state)
            # Finalization is one final verification, not a repeatedly scheduled test.
            if state.get('completed_runs') == 20:
                raise
        time.sleep(1800)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""One bounded OS/log observation; no GPU query, model load, or signals."""
import json
import shlex
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'CPU_4080_Point_slowdown_20261021_0922.json'


def main():
    assert not OUT.exists(), 'Use the existing one-shot evidence, not a repeat probe'
    frame = json.loads((BASE / 'capacity_progress_observation_20261021_0922.json').read_text())
    assert frame['observed_at_unix'] == 1792545723.653781
    assert 0 <= time.time() - frame['observed_at_unix'] < 1800
    expected = frame['hosts']['4080']['training']
    code = r'''
import csv,io,json,os,stat,time
from pathlib import Path
expected=json.loads(EXPECTED)
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='3b362c03-4ed3-4e03-aea4-cf7339356026'
def sample(pid):
 p=Path('/proc')/str(pid);v=(p/'stat').read_text().rsplit(')',1)[1].split()
 return {'ticks':int(v[11])+int(v[12]),'start_ticks':int(v[19]),'processor':int(v[36]),'state':v[0]}
first={w['pid']:sample(w['pid']) for w in expected}
for w in expected:assert first[w['pid']]['start_ticks']==w['start_ticks']
started=time.monotonic();time.sleep(1);seconds=time.monotonic()-started
workers=[]
for w in expected:
 p=Path('/proc')/str(w['pid']);s=sample(w['pid']);assert s['start_ticks']==w['start_ticks']
 args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
 assert args[args.index('--config')+1]=='/home/carla/SAFEFLOW2027_4080/runs/shapo_p47_four_env/runs/'+w['job']+'/resolved_formal_config.yaml'
 row={'job':w['job'],'pid':w['pid'],'start_ticks':s['start_ticks'],'state':s['state'],'processor_before':first[w['pid']]['processor'],'processor_after':s['processor'],'CPU_percent_single_core_basis':100*(s['ticks']-first[w['pid']]['ticks'])/os.sysconf('SC_CLK_TCK')/seconds,'affinity':sorted(os.sched_getaffinity(w['pid']))}
 if w['device']!='cpu':
  f=Path(w['progress_csv']);assert f.is_file()
  with f.open('rb') as stream:
   header=stream.readline().decode();size=f.stat().st_size;pos=max(len(header.encode()),size-16384);stream.seek(pos)
   if pos>len(header.encode()):stream.readline()
   tail=stream.read(16384).decode(errors='replace')
  if tail and not tail.endswith('\n'):tail=tail.rsplit('\n',1)[0]+'\n' if '\n' in tail else ''
  rows=list(csv.DictReader(io.StringIO(header+tail)))
  row['recent_progress_rows']=rows[-8:]
  row['progress_mtime_unix']=f.stat().st_mtime
  row['native_4M_file_exists']=(f.parent/'artifacts/evaluation_step_4000000.json').is_file()
  row['stdout_target']=os.readlink(p/'fd/1');row['stderr_target']=os.readlink(p/'fd/2')
 workers.append(row)
cores=[]
for i in range(os.cpu_count()):
 p=Path('/sys/devices/system/cpu')/('cpu'+str(i));freq=p/'cpufreq/scaling_cur_freq'
 cores.append({'logical_cpu':i,'physical_core':int((p/'topology/core_id').read_text()),'siblings':(p/'topology/thread_siblings_list').read_text().strip(),'current_khz':int(freq.read_text()) if freq.exists() else None})
print(json.dumps({'observed_at_unix':time.time(),'sample_seconds':seconds,'load_1m_5m_15m':os.getloadavg(),'workers':workers,'cores':cores,'GPU_queries':0,'signals':0,'extra_evaluation_rollouts':0}))
'''.replace('EXPECTED', repr(json.dumps(expected)))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                             '4080', 'python3 -c ' + shlex.quote(code)],
                            text=True, capture_output=True, check=True, timeout=60)
    evidence = {'kind': 'one_shot_OS_and_existing_progress_observation_after_new_Point_slowdown',
                'native_frame_unix': frame['observed_at_unix'],
                'not_a_scientific_test_or_configuration_change': True,
                'remote_observation': json.loads(result.stdout),
                'training_restarts': 0, 'foreign_projects_modified': False}
    with OUT.open('x') as stream:
        stream.write(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'evidence': str(OUT), 'workers': [
        {k: w[k] for k in ('job', 'CPU_percent_single_core_basis', 'processor_after', 'state')}
        for w in evidence['remote_observation']['workers']]}))


if __name__ == '__main__':
    main()

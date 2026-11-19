"""Finite30min observer then complete four-run Piper summary; no launch/retry/git."""
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.paper_crax_payload.close_samehardware_v12 import terminal_state


ROOT = Path('/storage_0/SAFEFLOW2027_lzqw_uid1007/simulation_v12')


def main():
    operation = ROOT/'piper-direction-taper-closure'; operation.mkdir(exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    queue_plan = json.loads((source/'configs/paper_crax_payload/simulation_v12_queue_a100_piper_direction_taper.json').read_text())
    for index in range(24):
        queue = json.loads((ROOT/'queue_piper_direction_taper.json').read_text())
        complete = terminal_state(queue)
        row = dict(check=index+1,unix_time=time.time(),queue_state=queue['status'],pending=queue['pending'],
            interval_seconds=1800,ready_for_summary=complete,training_launches=0,automatic_retry=False)
        (operation/f'check_{index+1:02d}.json').write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if complete:
            if len(queue['jobs']) != 4:
                raise ValueError('Need all four registered native controls/candidates')
            for job in queue_plan['jobs']:
                directory = Path(job['output'])
                receipt = json.loads((directory/'receipt.json').read_text())
                manifest = json.loads((directory/'run_config.json').read_text())
                arguments = job['arguments']; relative_plan = arguments[arguments.index('--plan')+1]
                plan = json.loads((source/relative_plan).read_text())
                arm = arguments[arguments.index('--arm')+1]; line = arguments[arguments.index('--line')+1]
                if (receipt['status'] != 'completed' or receipt['new_training_steps'] != 100000
                        or receipt['steps'] != job['required_steps'] or receipt['source_commit'] != 'd904b9c8'
                        or receipt['arm'] != arm or receipt['line'] != line
                        or manifest['plan'] != plan or manifest['parent_sha256'] != plan['lines'][line]['parent_sha256']
                        or manifest['host'] != 'CTAlab' or manifest['source_commit'] != 'd904b9c8'):
                    raise ValueError('Need the complete exact A100 native taper pair endpoints')
                progress = [json.loads(s) for s in (directory/'progress.jsonl').read_text().splitlines()]
                if [r['new_steps'] for r in progress] != list(range(1000,100001,1000)):
                    raise ValueError('All100 saved training nodes required')
                for step in plan['milestones']:
                    value = json.loads((directory/f'development_{step}.json').read_text())
                    if value['new_steps'] != step or value['arm'] != arm or value['line'] != line:
                        raise ValueError('Keep the registered DEV nodes')
                    if [g['condition'] for g in value['conditions']] != [c['name'] for c in plan['lines'][line]['conditions']]:
                        raise ValueError('Keep every registered geometry')
                    for group in value['conditions']:
                        seeds = [r['seed'] for r in group['records']]
                        start = plan['lines'][line]['development_layout_seed_start']
                        if seeds != list(range(start,start+100)):
                            raise ValueError('Keep all100 DEV layouts without replacement')
            subprocess.run([sys.executable,'-m','scripts.paper_crax_payload.summarize_simulation_v12',
                '--runs',str(ROOT/'runs-piper-direction-taper'),
                '--output',str(ROOT/'piper-direction-taper-analysis')],cwd=source,check=True)
            print('Complete four-run DEV summary saved;no independent test or winner promotion.',flush=True)
            return
        if index<23:
            time.sleep(1800)
    raise TimeoutError('Finite observation ended;native training untouched')


if __name__ == '__main__': main()

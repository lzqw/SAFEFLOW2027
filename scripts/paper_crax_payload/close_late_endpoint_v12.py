"""Finite30min receipt observer and exact-prefix offline review; no training launch."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts.paper_crax_payload.close_samehardware_v12 import ROOT, terminal_state
from scripts.paper_crax_payload.summarize_matched_ablation_v12 import (
    candidate_checks, registered_protocols, training_curve_comparisons, training_window)
from scripts.paper_crax_payload.summarize_simulation_v12 import output
from scripts.paper_crax_payload.train_late_endpoint_v12 import FORMAT, PLAN


def read(path):
    return json.loads(path.read_text())


def check_suffix(receipt, manifest, config, arm, parent):
    provenance = receipt['fork_provenance']
    if (receipt['status'] != 'completed' or receipt['engineering_only']
            or receipt['steps'] != 250000 or receipt['initial_steps'] != 100000
            or receipt['new_training_transitions'] != 150000
            or receipt['training_seed'] != 2034 or receipt['arm'] != arm
            or manifest['configuration'] != config or manifest['arm'] != arm
            or manifest['host'] != '447' or manifest['format'] != FORMAT
            or manifest['source_commit'] != '7eda9c8f'
            or receipt['source_commit'] != manifest['source_commit']
            or receipt['independent_endpoint_test_executed'] is not False
            or receipt['evaluations'] != config['evaluation']['milestones']
            or provenance != manifest['fork_provenance']
            or provenance['source_checkpoint_sha256'] != config['source_checkpoint_sha256']
            or provenance['prefix_steps'] != 100000
            or provenance['prefix_not_new_training'] is not True):
        raise ValueError('Require exact real100K-prefix plus150K suffix; no fresh/best/partial substitute')
    expected = {k: parent[k] for k in ('training_violations', 'safe_successes', 'safe_timeouts')}
    if provenance['prefix_counts'] != expected:
        raise ValueError('Actual shared prefix counts changed')
    for key in expected:
        if receipt['new_'+key] != receipt[key]-expected[key]:
            raise ValueError('Do not double-count the historical prefix as new training')
    if receipt['new_evaluation_controls'] != receipt['evaluation_controls']-provenance['prefix_evaluation_controls']:
        raise ValueError('Inherited DEV controls are not newly executed controls')


def summarize(destination):
    config = read(PLAN)
    protocols = registered_protocols(endpoint_residual=True)
    protocols.update({arm['name']: config for arm in config['arms']})
    original = ROOT/'runs-samehardware/A-full-P5-2034'
    prefix = [json.loads(s) for s in (original/'progress.jsonl').read_text().splitlines()][:100]
    windows, progress, nodes, episodes, provenance = [], [], [], [], []
    identities = set()
    for name, recipe in protocols.items():
        group = ('runs-late-endpoint' if name.startswith('L-') else
                 'runs-endpoint-residual' if name.startswith('E-') else
                 'runs-samehardware' if name in ('A-full-P5', 'A-Iso-matchedLR') else 'runs-matched-offload')
        directory = ROOT/group/(name+'-2034')
        receipt = read(directory/'receipt.json'); manifest = read(directory/'run_config.json')
        arm = next(a for a in recipe['arms'] if a['name'] == name)
        if name.startswith('L-'):
            check_suffix(receipt, manifest, recipe, arm, prefix[-1])
        elif (receipt['status'] != 'completed' or receipt['steps'] != 250000
                or receipt['new_training_transitions'] != 250000 or receipt['engineering_only']
                or manifest['configuration'] != recipe or manifest['host'] != '447'
                or receipt['arm'] != arm or receipt['training_seed'] != 2034):
            raise ValueError('Historical comparisons must also retain complete original scientific runs')
        expected_source = ('7eda9c8f' if name.startswith('L-') else '499a4b6d' if name.startswith('E-')
            else '2c06ca2a' if name in ('A-full-P5','A-Iso-matchedLR') else '2f09231b')
        if receipt['source_commit'] != manifest['source_commit'] or manifest['source_commit'] != expected_source:
            raise ValueError('Do not replace the registered native lineage with another run')
        identities.add(receipt['initial_network_sha256'])
        rows = [json.loads(s) for s in (directory/'progress.jsonl').read_text().splitlines()]
        if [r['steps'] for r in rows] != list(range(1000,250001,1000)):
            raise ValueError('Require all250 saved1K training nodes')
        if name.startswith('L-') and rows[:100] != prefix:
            raise ValueError('Inherited curves must be the actual identical observed prefix')
        for key in ('training_violations','safe_successes','safe_timeouts'):
            if receipt[key] != rows[-1][key]: raise ValueError('Final receipt count differs')
        for key in ('training_stall_episodes','training_stall_windows','training_cumulative_return'):
            if receipt['training_metrics'][key] != rows[-1][key]: raise ValueError('Tracker endpoint differs')
        progress.extend(dict(arm=name, **row) for row in rows)
        windows.extend(dict(arm=name, **training_window(rows,a,b)) for a,b in
                       ((0,100000),(100000,250000),(0,250000)))
        for step in recipe['evaluation']['milestones']:
            path = directory/f'evaluation_{step}.json'; value = read(path)
            if (value['milestone'] != step or value['independent_test_used'] is not False
                    or [c['condition'] for c in value['conditions']] !=
                    [c['name'] for c in recipe['evaluation']['conditions']]):
                raise ValueError('All original DEV conditions required; independent test forbidden')
            if name.startswith('L-') and step <= 100000:
                if path.read_bytes() != (original/path.name).read_bytes():
                    raise ValueError('Original DEV prefix files must be byte-identical')
            for condition in value['conditions']:
                records = condition['records']; n = len(records)
                if n != 30 or [r['layout_seed'] for r in records] != list(range(11910000,11910030)):
                    raise ValueError('Retain every registered DEV episode including failures')
                success = sum(r['safe_success'] for r in records); bad = sum(r['violation'] for r in records)
                nodes.append(dict(arm=name,transitions=step,condition=condition['condition'],episodes=n,
                    safe_successes=success,violations=bad,timeouts=n-success-bad,
                    stall_count=sum(r['stall'] for r in records),
                    mean_return=sum(r['episode_return'] for r in records)/n,
                    mean_capped_seconds=sum(r['capped_seconds'] for r in records)/n))
                episodes.extend(dict(arm=name,transitions=step,condition=condition['condition'],
                    inherited_prefix=name.startswith('L-') and step<=100000,
                    **{k:v for k,v in r.items() if k not in ('control_details','trajectory_tcp_xyz','layout')}) for r in records)
        provenance.append(dict(arm=name,receipt_sha256=hashlib.sha256((directory/'receipt.json').read_bytes()).hexdigest(),
            source_commit=manifest['source_commit'],native_host=manifest['host']))
    if len(identities) != 1: raise ValueError('Same original network initialization required')
    destination.mkdir(exist_ok=False)
    candidate = 'L-endpoint-ramp100K150K'
    baselines = ('L-control-continuation','A-full-P5','A-Iso-matchedLR','A-Metric-matchedLR','E-endpoint-residual')
    for filename, values in (('training.csv',progress),('training_windows.csv',windows),
            ('development_nodes.csv',nodes),('development_episodes.csv',episodes),
            ('development_criteria.csv',candidate_checks(windows,nodes,candidate,baselines)),
            ('training_curve_comparisons.csv',training_curve_comparisons(progress,candidate,baselines))):
        output(destination/filename,values)
    (destination/'receipt.json').write_text(json.dumps(dict(status='complete_offline_review',runs=len(protocols),
        new_campaign_completed_training_transitions=300000,shared_original_prefix_steps_not_new=100000,
        historical_comparison_training_transitions=1500000,new_simulation_controls=0,
        new_training_steps=0,training_seed_count=1,provenance=provenance,
        independent_test_used=False,winner_confirmation=False,all_registered_nodes_retained=True,
        inherited_first100K_not_new_confirmation=True,physical_robot_interactions=0),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maximum-checks',type=int,default=32)
    args=parser.parse_args()
    if not 1<=args.maximum_checks<=48: raise ValueError('Finite observation only,at most24h')
    operation=ROOT/'late-endpoint-closure';operation.mkdir(exist_ok=False)
    expected=[arm['name']+'-2034' for arm in read(PLAN)['arms']]
    for check in range(args.maximum_checks):
        queue=read(ROOT/'queue_late_endpoint.json')
        if queue['source_commit']!='7eda9c8f' or [j['name'] for j in queue['jobs']]!=expected:
            raise ValueError('Observe only the exact registered native continuation pair')
        complete=terminal_state(queue)
        snapshot=dict(check=check+1,unix_time=time.time(),queue_status=queue['status'],ready_for_summary=complete,
            interval_seconds=1800,downstream_training_launches=0,automatic_retry=False)
        (operation/f'check_{check+1:02d}.json').write_text(json.dumps(snapshot,indent=2)+'\n')
        print(json.dumps(snapshot),flush=True)
        if complete:
            summarize(ROOT/'late-endpoint-complete-analysis');return
        if check+1<args.maximum_checks:time.sleep(1800)
    raise TimeoutError('Finite observer ended; training untouched and not marked failed')


if __name__=='__main__':main()

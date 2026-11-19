"""Complete native Piper pair against contemporary and historical fixed references."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.paper_crax_payload.analyze_piper_failures_v12 import paired
from scripts.paper_crax_payload.summarize_simulation_v12 import output


SPECS = {
    'piper_reach': dict(plan='reach', candidate='Reach-direction-taper2mm', control='Reach-fixed2mm-control',
        fixed='round1_native/piper_reach-R0-fixed', metric='round3_native/piper_reach-Metric-only'),
    'piper_payload': dict(plan='payload', candidate='Payload-direction-taper4mm', control='Payload-fixed4mm-control',
        fixed='payload_round1_native/piper_payload-Payload-P5-fixed', metric='payload_round1_native/piper_payload-Payload-Metric-only'),
}
CONFIG_ROOT = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload'


def read_run(directory, line, arm, plan=None):
    receipt_path = directory/'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads((directory/'run_config.json').read_text())
    config = manifest['plan']; task = config['lines'][line]
    if (receipt['status'] != 'completed' or receipt['new_training_steps'] != 100000
            or receipt['arm'] != arm or receipt['line'] != line
            or receipt['steps'] != task['parent_steps']+100000
            or receipt['source_commit'] != manifest['source_commit']):
        raise ValueError('Every reference must retain its complete native100K endpoint')
    if plan is not None and (config != plan or manifest['source_commit'] != 'd904b9c8'
                             or manifest['host'] != 'CTAlab'):
        raise ValueError('Use the registered contemporary A100 pair without hardware substitution')
    rows = [json.loads(s) for s in (directory/'progress.jsonl').read_text().splitlines()]
    if [r['new_steps'] for r in rows] != list(range(1000,100001,1000)):
        raise ValueError('Keep all100 training nodes, including early regressions')
    for key, value in receipt['segment_counts'].items():
        if rows[-1][key] != value:
            raise ValueError('Native receipt and training endpoint differ')
    nodes = {}
    for step in config['milestones']:
        raw = json.loads((directory/f'development_{step}.json').read_text())
        if raw['new_steps'] != step or raw['arm'] != arm or raw['line'] != line:
            raise ValueError('Keep the original DEV milestone identity')
        if [g['condition'] for g in raw['conditions']] != [c['name'] for c in task['conditions']]:
            raise ValueError('Keep every registered geometry')
        for group in raw['conditions']:
            records = group['records']; start = task['development_layout_seed_start']
            if [r['seed'] for r in records] != list(range(start,start+100)):
                raise ValueError('No missing, substituted or duplicated DEV episodes')
            if any(r['safe_success'] and r['violation'] for r in records):
                raise ValueError('Success and violation are mutually exclusive')
            nodes[step,group['condition']] = records
    return dict(rows=rows,nodes=nodes,receipt=receipt,manifest=manifest,
        receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest())


def metrics(records):
    return dict(safe_successes=sum(r['safe_success'] for r in records),
        violations=sum(r['violation'] for r in records),
        timeouts=sum(not r['safe_success'] and not r['violation'] for r in records),
        stall_count=sum(r['stall'] for r in records),
        mean_return=sum(r['episode_return'] for r in records)/len(records),
        mean_capped_seconds=sum(r['time_to_goal_capped_seconds'] for r in records)/len(records))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True); p.add_argument('--reports',type=Path,required=True)
    p.add_argument('--line',choices=SPECS,required=True); p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(); spec = SPECS[args.line]
    plan = json.loads((CONFIG_ROOT/('simulation_v12_piper_direction_taper_'+spec['plan']+'.json')).read_text())
    runs = {label: read_run(args.runs/(args.line+'-'+spec[label]),args.line,spec[label],plan)
            for label in ('candidate','control')}
    for label in ('fixed','metric'):
        directory = args.reports/spec[label]
        manifest = json.loads((directory/'run_config.json').read_text())
        runs[label] = read_run(directory,args.line,manifest['arm'])
    identities = {r['manifest']['parent_sha256'] for r in runs.values()}
    if len(identities)!=1:
        raise ValueError('Every comparator must share the same native full parent')
    training = []; nodes = []; comparisons = []; checks = []
    for label, run in runs.items():
        end = run['rows'][-1]
        training.append(dict(reference=label,source_commit=run['manifest']['source_commit'],
            host=run['manifest']['host'],**{k:end[k] for k in ('violations','safe_successes','safe_timeouts',
                'training_stall_episodes','training_cumulative_return')}))
        nodes.extend(dict(reference=label,new_steps=step,condition=condition,**metrics(records))
                     for (step,condition),records in run['nodes'].items())
    candidate = runs['candidate']
    for baseline in ('control','fixed','metric'):
        for (step,condition),records in candidate['nodes'].items():
            reference = runs[baseline]['nodes'][step,condition]
            left,right = metrics(records),metrics(reference)
            comparisons.append(dict(baseline=baseline,new_steps=step,condition=condition,
                scope='samehardware_sameparent' if baseline=='control' else 'historical_crosshardware_reference',
                **paired(records,reference),**{'delta_'+key:left[key]-right[key] for key in left}))
    def check(name,baseline,observed,limit,relation):
        checks.append(dict(criterion=name,baseline=baseline,candidate_value=observed,limit=limit,relation=relation,
            satisfied=observed<=limit if relation=='<=' else observed>=limit,independent_confirmation=False))
    end = candidate['rows'][-1]
    for baseline in ('control','fixed','metric'):
        ref = runs[baseline]['rows'][-1]
        for key in ('safe_timeouts','training_stall_episodes'):
            check('training_'+key+'_no_higher',baseline,end[key],ref[key],'<=')
        check('return_drop_at_most1percent',baseline,end['training_cumulative_return'],
              ref['training_cumulative_return']-.01*abs(ref['training_cumulative_return']),'>=')
        for (step,condition),records in candidate['nodes'].items():
            if step!=100000: continue
            left,right = metrics(records),metrics(runs[baseline]['nodes'][step,condition])
            for key,relation in (('safe_successes','>='),('violations','<='),('timeouts','<='),('stall_count','<=')):
                check('DEV_'+condition+'_'+key,baseline,left[key],right[key],relation)
    check('training_violations_within_original_fixed_P5_110percent','fixed',end['violations'],
          1.1*runs['fixed']['rows'][-1]['violations'],'<=')
    if args.line=='piper_reach':
        for condition,threshold in (('radius_0575',93),('radius_060',90)):
            check('absolute_'+condition+'_success','user_target',metrics(candidate['nodes'][100000,condition])['safe_successes'],threshold,'>=')
    args.output.mkdir(parents=True,exist_ok=False)
    for name,rows in (('training_endpoints.csv',training),('all_dev_nodes.csv',nodes),
                      ('paired_nodes.csv',comparisons),('development_criteria.csv',checks)):
        output(args.output/name,rows)
    (args.output/'receipt.json').write_text(json.dumps(dict(status='complete_pair_comparison',line=args.line,
        native_runs=len(runs),new_training_steps=0,new_simulation_controls=0,independent_test_used=False,
        winner_confirmation=False,training_seed_count=1,all_registered_nodes_retained=True,
        historical_crosshardware_limits_disclosed=True,
        sources={label:dict(receipt_sha256=run['receipt_sha256'],source=run['manifest']['source_commit'],
                            host=run['manifest']['host']) for label,run in runs.items()}),indent=2)+'\n')


if __name__=='__main__': main()

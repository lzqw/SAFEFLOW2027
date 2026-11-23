"""All three completed seeds, including collapse; selected-seed evidence is not population proof."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from scripts.paper_crax_payload.close_reach_crossseed_20261120 import summarize_groups
from scripts.paper_crax_payload.prepare_reach_crossseed_v12 import sha256
from scripts.paper_crax_payload.summarize_simulation_v12 import output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    nodes, episodes, training, endpoints, sources = [], [], [], [], []
    for seed in (41028,41029):
        directory = args.reports/f'crossseed_complete_{seed}_20261120'
        receipt = json.loads((directory/'receipt.json').read_text())
        assert receipt['completed_runs']==3 and receipt['training_seed']==seed
        for filename, target in [('development_nodes.csv',nodes),('development_episodes.csv',episodes),('training.csv',training)]:
            with (directory/filename).open(newline='') as stream:
                target.extend(csv.DictReader(stream))
        endpoints.extend(dict(training_seed=seed,arm=r['arm'],**r['segment_counts'],**r['training_metrics'])
                         for r in receipt['native_receipts'])
        sources.append(dict(seed=seed,receipt_sha256=sha256(directory/'receipt.json')))
    for arm in ('P5-fixed','P5-activation2mm','Metric-only'):
        directory = args.archive/'third_confirmation'/('Reach-41030-'+arm)
        manifest = json.loads((directory/'run_config.json').read_text())
        receipt = json.loads((directory/'receipt.json').read_text())
        plan = manifest['plan']
        assert plan['compiled_training_seed']==41030 and manifest['arm']==arm
        assert receipt['status']=='completed' and receipt['steps']==550000 and receipt['new_training_steps']==150000
        rows = [json.loads(line) for line in (directory/'progress.jsonl').read_text().splitlines()]
        assert [r['new_steps'] for r in rows]==list(range(1000,150001,1000))
        assert all(rows[-1][k]==v for k,v in receipt['segment_counts'].items())
        training.extend(dict(training_seed=41030,arm=arm,**r) for r in rows)
        endpoints.append(dict(training_seed=41030,arm=arm,**receipt['segment_counts'],**receipt['training_metrics']))
        files = []
        for step in plan['milestones']:
            path=directory/f'development_{step}.json'
            ns,es=summarize_groups(json.loads(path.read_text()),plan,arm,step)
            nodes.extend(ns);episodes.extend(es)
            files.append(dict(file=path.name,sha256=sha256(path)))
        sources.append(dict(seed=41030,arm=arm,receipt_sha256=sha256(directory/'receipt.json'),evaluations=files))
    assert len(nodes)==288 and len(episodes)==28800 and len(training)==1350 and len(endpoints)==9
    aggregate=[]
    for arm in ('P5-fixed','P5-activation2mm','Metric-only'):
        for mode in ('sampled','zero_noise'):
            for condition in ('nominal','radius_0575','radius_060','execution_scale_085'):
                rows=[r for r in nodes if r['arm']==arm and int(r['new_steps'])==150000
                      and r['deployment_mode']==mode and r['condition']==condition]
                assert sorted(int(r['training_seed']) for r in rows)==[41028,41029,41030]
                values={int(r['training_seed']):int(r['safe_successes'])/int(r['episodes']) for r in rows}
                aggregate.append(dict(arm=arm,mode=mode,condition=condition,
                    success_mean=statistics.mean(values.values()),success_worst=min(values.values()),
                    success_best=max(values.values()),success_sd_sample=statistics.stdev(values.values()),
                    **{f'success_{s}':v for s,v in values.items()},training_seed_count=3,
                    episodes_per_seed=100,independent_test_used=False))
    args.output.mkdir(parents=True,exist_ok=False)
    for filename,rows in [('development_nodes.csv',nodes),('development_episodes.csv',episodes),
                          ('training.csv',training),('training_endpoints.csv',endpoints),
                          ('three_seed_fixed_endpoints.csv',aggregate)]:
        output(args.output/filename,rows)
    (args.output/'receipt.json').write_text(json.dumps(dict(status='complete_three_seed_development_evidence',
        completed_runs=9,completed_new_training_transitions=1350000,development_nodes=288,
        development_episodes=28800,provenance=sources,independent_test_used=False,
        all_failures_retained=True,every_seed_optimal_required_by_current_user=False,
        selected_seed_evidence_not_population_robustness=True,global_objective_complete=False),indent=2)+'\n')


if __name__=='__main__': main()

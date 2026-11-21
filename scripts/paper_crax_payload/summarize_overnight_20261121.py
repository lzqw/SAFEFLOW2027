"""Archive exact overnight scientific endpoints and an engineering-interrupted prefix."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.paper_crax_payload.snapshot_independent_parents_20261120 import (
    complete_rows, summarize_node)
from scripts.paper_crax_payload.summarize_simulation_v12 import output


CASES = [
    ('third_radius',41030,'Reach',300000,400000,'255b737b69c61481e094043789f1ac6ae53a04e2',11990000,True),
    ('payload42029',42029,'Payload',0,300000,'870e7e81ce29ab0b06a24fe546544d1f848ad666',12010000,True),
    ('payload42028_failure',42028,'Payload',0,300000,'870e7e81ce29ab0b06a24fe546544d1f848ad666',12010000,False),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    endpoints, training, nodes, episodes, provenance = [], [], [], [], []
    for name, seed, task, initial, budget, source, start, completed in CASES:
        root = args.native/name
        def read(path):
            raw = path.read_bytes()
            provenance.append(dict(run=name, file=str(path.relative_to(args.native)),
                sha256=hashlib.sha256(raw).hexdigest()))
            return raw
        manifest = json.loads(read(root/'run_config.json'))
        if (manifest['training_seed'] != seed or manifest['task'] != task
                or manifest['source_commit'] != source or manifest['method'] != 'EA'
                or manifest['initial_steps'] != initial or manifest['total_transition_budget'] != budget
                or manifest['engineering'] is not False):
            raise ValueError('Exact native scientific identity and budget required')
        rows = complete_rows(read(root/'progress.jsonl'))
        last = rows[-1]
        if [r['steps'] for r in rows] != list(range(initial+1000,last['steps']+1,1000)):
            raise ValueError('Whole native observed prefix required')
        for row in rows:
            if row['episodes'] != row['training_safe_successes']+row['training_violation_count']+row['training_safe_timeouts']:
                raise ValueError('Every terminal class retained')
        if completed:
            receipt = json.loads(read(root/'receipt.json'))
            if (receipt['status'] != 'completed' or receipt['training_steps'] != budget
                    or receipt['newly_collected_transitions'] != budget-initial
                    or receipt['training_source_commit'] != source or receipt['training_seed'] != seed
                    or last['steps'] != budget
                    or any(last[k] != receipt[k] for k in ('training_safe_successes',
                        'training_violation_count','training_safe_timeouts'))):
                raise ValueError('Real endpoint and observed native curve must agree')
        elif (root/'receipt.json').exists() or last['steps'] != 175000:
            raise ValueError('Keep the exact interrupted175K prefix explicitly incomplete')
        identity = dict(run=name, training_seed=seed, task=task)
        endpoints.append(dict(**identity, status='completed' if completed else 'engineering_interrupted',
            initial_steps=initial, observed_steps=last['steps'], target_steps=budget,
            observed_segment_steps=last['steps']-initial,
            **{k:last[k] for k in ('training_safe_successes','training_violation_count','training_safe_timeouts')},
            inherited_counts_included=bool(initial), independent_test_used=False))
        training.extend(dict(**identity, **r) for r in rows)
        for path in sorted(root.glob('evaluation_*.json')):
            value = json.loads(read(path))
            step = int(path.stem.split('_')[-1])
            if step > last['steps']:
                raise ValueError('No evaluation ahead of the observed native prefix')
            ns, es = summarize_node(value, seed, step, start, task)
            nodes.extend(dict(run=name,**r) for r in ns)
            episodes.extend(dict(run=name,**r) for r in es)
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, records in [('endpoints.csv',endpoints),('training.csv',training),
                              ('development_nodes.csv',nodes),('development_episodes.csv',episodes)]:
        output(args.output/filename,records)
    (args.output/'receipt.json').write_text(json.dumps(dict(status='two_completed_one_interrupted_native_review',
        complete_runs=2, interrupted_runs=1, completed_new_transitions=400000,
        interrupted_observed_transitions=175000, no_interrupted_budget_reported_complete=True,
        development_nodes=len(nodes), development_episodes=len(episodes), provenance=provenance,
        new_training_steps=0,new_simulation_controls=0,independent_test_used=False,
        global_objective_complete=False),indent=2)+'\n')


if __name__ == '__main__':
    main()

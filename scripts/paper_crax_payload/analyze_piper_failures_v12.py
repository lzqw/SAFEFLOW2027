"""Paired final DEV failures from existing records; no physics, retraining or oracle policy."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from scripts.paper_crax_payload.summarize_simulation_v12 import output


CASES = {
    'Reach': {
        'fixed': 'round1_native/piper_reach-R0-fixed',
        'ramp75k': 'round1_native/piper_reach-R1-ramp75k',
        'fixed2mm': 'round2_native/piper_reach-R2-activation2mm',
        'half75k2mm': 'followup_native/piper_reach-Reach-half75k-activation2mm',
        'Metric': 'round3_native/piper_reach-Metric-only',
    },
    'Payload': {
        'fixed': 'payload_round1_native/piper_payload-Payload-P5-fixed',
        'fixed2mm': 'payload_round1_native/piper_payload-Payload-activation2mm',
        'fixed4mm': 'followup_native/piper_payload-Payload-activation4mm',
        'Metric': 'payload_round1_native/piper_payload-Payload-Metric-only',
    },
}


def paired(candidate, baseline):
    left = {r['seed']: r for r in candidate}; right = {r['seed']: r for r in baseline}
    if len(left) != len(candidate) or len(right) != len(baseline) or left.keys() != right.keys():
        raise ValueError('Require every same-layout DEV record, never replace a failure')
    return dict(both_safe=sum(left[s]['safe_success'] and right[s]['safe_success'] for s in left),
        candidate_only_safe=sum(left[s]['safe_success'] and not right[s]['safe_success'] for s in left),
        baseline_only_safe=sum(not left[s]['safe_success'] and right[s]['safe_success'] for s in left),
        neither_safe=sum(not left[s]['safe_success'] and not right[s]['safe_success'] for s in left))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); summaries = []; comparisons = []; provenance = []; joint = []
    for task, cases in CASES.items():
        records = {}
        for arm, relative in cases.items():
            directory = args.reports/relative
            receipt = json.loads((directory/'receipt.json').read_text())
            if receipt['status'] != 'completed' or receipt['new_training_steps'] != 100000:
                raise ValueError('Use complete native100K endpoints only')
            path = directory/'development_100000.json'
            raw = json.loads(path.read_text())
            provenance.append(dict(task=task, arm=arm, relative_file=str(path.relative_to(args.reports)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            records[arm] = {}
            for condition in raw['conditions']:
                rows = condition['records']
                if len(rows) != 100 or len({r['seed'] for r in rows}) != 100:
                    raise ValueError('Keep all100 registered DEV episodes per condition')
                records[arm][condition['condition']] = rows
                failures = [r for r in rows if not r['safe_success']]
                summaries.append(dict(task=task, arm=arm, condition=condition['condition'],
                    episodes=len(rows), successes=sum(r['safe_success'] for r in rows),
                    violations=sum(r['violation'] for r in rows), timeouts=sum(r['safe_timeout'] for r in rows),
                    stalls=sum(r['stall'] for r in rows),
                    failures_in_first10_controls=sum(r['length'] <= 10 for r in failures),
                    failure_median_control=statistics.median(r['length'] for r in failures) if failures else None,
                    failure_mean_true_margin_mm=statistics.mean((.015-r['minimum_distance_m'])*1000 for r in failures) if failures else None))
        for arm, conditions in records.items():
            for condition, rows in conditions.items():
                comparisons.append(dict(task=task, candidate=arm, baseline='Metric', condition=condition,
                    **paired(rows, records['Metric'][condition])))
        p5_arms = [a for a in cases if a != 'Metric']
        for condition in records['Metric']:
            indexed = {a: {r['seed']: r for r in records[a][condition]} for a in cases}
            for seed in indexed['Metric']:
                # Outcome-conditioned diagnostics only; never used to select an action or training layout.
                joint.append(dict(task=task, condition=condition, layout_seed=seed,
                    every_listed_P5_failed=all(not indexed[a][seed]['safe_success'] for a in p5_arms),
                    Metric_safe=indexed['Metric'][seed]['safe_success'],
                    P5_failure_count=sum(not indexed[a][seed]['safe_success'] for a in p5_arms),
                    listed_P5_arms=len(p5_arms)))
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (('failure_timing.csv', summaries), ('paired_layouts.csv', comparisons), ('joint_failures.csv', joint)):
        output(args.output/name, rows)
    (args.output/'receipt.json').write_text(json.dumps(dict(status='complete_existing_DEV_analysis',
        provenance=provenance, new_training_steps=0, new_simulation_controls=0, independent_test_used=False,
        training_seed_count=1, winner_confirmation=False,
        caution='Paired initial layouts are not identical visited states; selected historical arms are not independent replications'), indent=2)+'\n')


if __name__ == '__main__': main()

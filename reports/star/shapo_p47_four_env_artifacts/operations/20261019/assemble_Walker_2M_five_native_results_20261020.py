#!/usr/bin/env python3
"""Summarize five already-archived native Walker2M evaluations; no new rollouts."""
import csv
import hashlib
import json
import math
import shlex
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
CACHE = Path('/home/lzqw/SAFEFLOW2027_artifacts/recovered_a100_4080/partial_milestones/walker_2M_available5_20261020')
OUT = Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/partial_milestones/walker_2M_available5_20261020')
SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    state = read(BASE / 'scientific_evidence_archive_state.json')
    selected = []
    for seed in range(2027, 2032):
        origin = '5080' if seed in (2028, 2029) else 'tjucta-a100'
        job = 'walker2d-' + str(seed)
        receipt = state['archived'][origin + '/' + job]
        assert receipt['archived'] is True
        selected.append({'training_seed': seed, 'origin': origin, 'job': job, 'manifest': receipt['manifest']})
    # Only the five indexed manifests are read. No historical manifest sweep,
    # checkpoint deserialization or remote training/evaluation command occurs.
    code = f'''
import hashlib,json,re
from pathlib import Path
rows=json.loads({json.dumps(selected)!r})
root=Path('/home/carla/4T/SAFEFLOW2027_4080/scientific_evidence_archive')
assert root.stat().st_dev!=Path('/home/carla/SAFEFLOW2027_4080').stat().st_dev
result=[]
for r in rows:
 p=Path(r['manifest']);p.relative_to(root/'manifests')
 m=json.loads(p.read_text())
 assert m['format']=='safeflow-raw-evidence-snapshot-v1'
 assert m['origin']==r['origin'] and m['job']==r['job'] and m['source_commit']=={SOURCE!r}
 entries=[f for f in m['files'] if f['path'].endswith('/artifacts/evaluation_step_2000000.json')]
 assert entries,'Missing native Walker2M evaluation: '+r['job']
 hashes={{f['sha256'] for f in entries}}
 assert len(hashes)==1,'Ambiguous same-seed milestone: never select by metric value: '+r['job']
 sha=next(iter(hashes));assert re.fullmatch('[0-9a-f]{{64}}',sha)
 blob=root/'blobs'/sha
 data=blob.read_bytes();assert hashlib.sha256(data).hexdigest()==sha
 assert all(f['bytes']==len(data) for f in entries)
 result.append(dict(r,sha256=sha,bytes=len(data),blob=str(blob),source_relative_paths=sorted(f['path'] for f in entries)))
print(json.dumps(result))
'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '4080',
                             'python3 -c ' + shlex.quote(code)], capture_output=True, text=True,
                            check=True, timeout=90)
    metadata = json.loads(result.stdout)
    assert len(metadata) == 5
    records = []
    for m in metadata:
        relative = Path('native_evaluations') / m['job'] / 'evaluation_step_2000000.json'
        native = CACHE / relative
        native.parent.mkdir(parents=True, exist_ok=True)
        if not native.exists():
            subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                            '4080:' + m['blob'], str(native)], check=True, capture_output=True, timeout=90)
        assert digest(native) == m['sha256'] and native.stat().st_size == m['bytes']
        e = read(native)
        assert e['milestone'] == 2000000 and e['episodes'] == 100 and e['episodes_per_base_seed'] == 20
        assert e['base_seeds'] == list(range(32027, 32032))
        assert e['trajectory_seeds'] == [32027, 32029, 32031]
        assert len(e['returns']) == len(e['costs']) == 100
        geometry = e['direction_geometry_high_risk_top_20_percent']
        record = {'training_seed': m['training_seed'], 'environment': 'Walker2d-v4',
                  'milestone_steps': 2000000, 'total_training_budget_steps': 5000000,
                  'mean_return': e['mean_return'], 'mean_cost': e['mean_cost'],
                  'cost_cvar_20': e['cost_cvar_20'], 'failure_rate': e['failure_rate'],
                  'time_limit_rate': e['time_limit_rate'],
                  'training_cumulative_cost_at_2M': e['training_cumulative_cost'],
                  'high_risk_normal_tangent_variance_ratio': geometry['normal_to_tangent_variance_ratio'],
                  'high_risk_dangerous_outward_velocity': geometry['dangerous_outward_velocity'],
                  'native_evaluation': {'relative_path': relative.as_posix(), 'sha256': m['sha256'],
                      'bytes': m['bytes'], 'archive_manifest': m['manifest'], 'archive_blob': m['blob'],
                      'all_identical_SHA_source_relative_paths': m['source_relative_paths'],
                      'archive_manifest_declared_source_commit': SOURCE},
                  'protocol': {k: e[k] for k in ('episodes', 'episodes_per_base_seed', 'base_seeds', 'trajectory_seeds')},
                  'strict_full_checkpoint_provenance_audit_executed': False}
        assert all(math.isfinite(record[k]) for k in ('mean_return', 'mean_cost', 'cost_cvar_20', 'failure_rate'))
        records.append(record)
        destination = OUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert digest(destination) == m['sha256'], 'Never overwrite differing existing native evidence'
        else:
            shutil.copyfile(native, destination)
    metrics = {k: {'mean_across_five_training_seeds': statistics.mean(r[k] for r in records),
                   'sd_across_five_training_seeds': statistics.stdev(r[k] for r in records)}
               for k in ('mean_return', 'mean_cost', 'cost_cvar_20', 'failure_rate')}
    summary = {'schema_version': 1, 'kind': 'existing_native_intermediate_milestone_descriptive_summary',
               'assembled_at_utc': datetime.now(timezone.utc).isoformat(),
               'environment': 'Walker2d-v4', 'milestone_steps': 2000000,
               'known_original_full_parent_gap_seeds': [2027],
               'descriptive_summary_includes_provenance_unresolved_history': True,
               'total_training_budget_steps': 5000000, 'scientific_source_commit': SOURCE,
               'planned_training_seeds': list(range(2027, 2032)), 'available_training_seeds': list(range(2027, 2032)),
               'availability_basis': 'Five exact latest indexed4T archive manifests; all same-seed2M copies must have one SHA, never selected by return/cost.',
               'records': records, 'descriptive_aggregation': {'unit': 'training seed', 'n': 5,
                   'spread': 'Sample standard deviation across five training seeds, not episode-level SE',
                   'cost_cvar_20_aggregation': 'Mean of five native per-training-seed CVaR20 values, not pooled500-episode CVaR',
                   'metrics': metrics},
               'native_evaluation_protocol_checked': {'episodes_per_training_seed': 100,
                   'base_seed_count': 5, 'episodes_per_base_seed': 20,
                   'source_JSON_bytes_and_SHA_match_archive_rows': True,
                   'only_metrics_JSON_protocol_checks_not_full_checkpoint_or_final_matrix_audit': True},
               'additional_evaluation_rollouts': 0, 'new_training_cells': 0,
               'scientific_parameters_changed': False,
               'matched_control_comparison_performed': False,
               'inferential_or_significance_test_performed': False,
               'early_scientific_stop_or_configuration_selection_performed': False,
               'full_checkpoint_provenance_audit_executed': False,
               'strict_final_matrix_audit_executed': False,
               'final_scientific_cohort_report': False, 'full_scientific_objective_complete': False,
               'notes': ['This2M snapshot is not the5M final Walker cohort or evidence of full experiment completion.',
                   'No matchedC0 control is supplied; these data alone do not establish causal safety or task-performance improvement.',
                   'Native mean cost equals failure rate for these files: average0.376 means37.6% failures, not a standalone low-cost success claim.',
                   'At this fixed2M milestone, failure varies from1% for recovered seed2027 to75% for seed2028; cross-seed task/safety behavior is not uniform.',
                   'High-risk normal-to-tangent variance ratios span0.6690-0.8302. This shows normal diffusion suppression in these diagnostic samples, not a causal task/safety improvement overC0.',
                   'The statistical unit is5 training seeds, not500 independent training runs.',
                   'High-risk direction geometry is diagnostic, not a new training or configuration-selection gate.',
                   'Native training cumulative cost is accepted-lineage cost, not all physically executed recovery work.',
                   'Seed2027 recovered-history full-parent identity remains unresolved; matching metric JSON/archive SHA does not repair it.']}
    for root in (CACHE, OUT):
        root.mkdir(parents=True, exist_ok=True)
        (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        fields = [k for k in records[0] if k not in ('native_evaluation', 'protocol', 'strict_full_checkpoint_provenance_audit_executed')]
        with (root / 'per_training_seed_metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: r[k] for k in fields} for r in records)
    print(json.dumps({'environment': 'Walker2d-v4', 'milestone': 2000000, 'n_training_seeds': 5,
                      'descriptive_metrics': metrics,
                      'per_seed': [{k: r[k] for k in ('training_seed', 'mean_return', 'mean_cost', 'cost_cvar_20', 'failure_rate')} for r in records],
                      'additional_rollouts': 0}), flush=True)


if __name__ == '__main__':
    main()

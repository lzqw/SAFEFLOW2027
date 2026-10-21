#!/usr/bin/env python3
"""Append one newly archived Button4M result; reuse seven published raw files."""
import csv
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from assemble_Point_4M_available7_20261021 import copy_exact, digest, metrics, protocol, read

BASE = Path(__file__).resolve().parent
PARENT = Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/partial_milestones')
OLD = PARENT / 'point_4M_available7_20261021'
OUT = PARENT / 'point_4M_available8_20261021'
CACHE = Path('/home/lzqw/SAFEFLOW2027_artifacts/recovered_a100_4080/partial_milestones/point_4M_available8_20261021')
SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'
JOB = 'point-button-2028'
MANIFEST = '/home/carla/4T/SAFEFLOW2027_4080/scientific_evidence_archive/manifests/4080/point-button-2028/1792548167804176754-f4c9316c5622eb015057f6361630a8835890560ee61f47c1c224ea58a3871664.json'


def main():
    assert not (OUT / 'summary.json').exists() and not (CACHE / 'summary.json').exists(), 'Do not replay a completed assembly'
    previous = read(OLD / 'summary.json')
    assert previous['scientific_source_commit'] == SOURCE and len(previous['records']) == 7
    assert JOB not in {r['job'] for r in previous['records']}
    code = f'''
import hashlib,json,re
from pathlib import Path
root=Path('/home/carla/4T/SAFEFLOW2027_4080/scientific_evidence_archive')
assert root.stat().st_dev!=Path('/home/carla/SAFEFLOW2027_4080').stat().st_dev
p=Path({MANIFEST!r});raw=p.read_bytes();manifest_sha=hashlib.sha256(raw).hexdigest()
assert p.parent==root/'manifests/4080/point-button-2028' and manifest_sha==p.stem.rsplit('-',1)[1]
m=json.loads(raw)
assert m['format']=='safeflow-raw-evidence-snapshot-v1' and m['origin']=='4080'
assert m['job']=={JOB!r} and m['source_commit']=={SOURCE!r}
entries=[r for r in m['files'] if r['path'].endswith('/artifacts/evaluation_step_4000000.json')]
assert entries,'No native4M JSON: wait for a genuinely new archive, never rerun evaluation'
assert len({{r['sha256'] for r in entries}})==1,'Ambiguous native result: never select by performance'
sha=entries[0]['sha256'];assert re.fullmatch('[0-9a-f]{{64}}',sha)
blob=root/'blobs'/sha;data=blob.read_bytes()
assert hashlib.sha256(data).hexdigest()==sha and all(r['bytes']==len(data) for r in entries)
print(json.dumps({{'manifest':str(p),'manifest_sha256':manifest_sha,'manifest_created_at_ns':m['created_at_ns'],
 'source_commit':m['source_commit'],'blob':str(blob),'sha256':sha,'bytes':len(data),
 'all_identical_SHA_source_relative_paths':[r['path'] for r in entries]}}))
'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '4080',
                             'python3 -c ' + shlex.quote(code)], text=True, capture_output=True, check=True, timeout=60)
    meta = json.loads(result.stdout)
    relative = Path('native_evaluations') / JOB / 'evaluation_step_4000000.json'
    native = CACHE / relative
    native.parent.mkdir(parents=True, exist_ok=True)
    assert not native.exists(), 'Inspect any previous partial transfer before retrying'
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                    '4080:' + meta['blob'], str(native)], capture_output=True, check=True, timeout=60)
    assert digest(native) == meta['sha256'] and native.stat().st_size == meta['bytes']
    evaluation = read(native)
    protocol(evaluation, 4000000)
    row = {'job': JOB, 'environment': 'SafetyPointButton1-v0', 'training_seed': 2028,
           'episodes_each_milestone': 100, 'episodes_per_base_seed': 20,
           'base_seeds': evaluation['base_seeds'], 'trajectory_seeds': evaluation['trajectory_seeds'],
           'native_4M': {'relative_path': relative.as_posix(), **meta, 'metrics': metrics(evaluation)},
           'legacy_provenance_notice': 'Metrics-only assembly does not certify full checkpoint ancestry or final provenance.'}
    one_relative = Path('../first_milestone_20_cells_20261020/native_evaluations') / JOB / 'evaluation_step_1000000.json'
    one_path = (OUT / one_relative).resolve()
    if one_path.is_file():
        one = read(one_path)
        protocol(one, 1000000)
        row['same_seed_native_1M'] = {'previously_published_relative_path': one_relative.as_posix(),
                                     'sha256': digest(one_path), 'metrics': metrics(one)}
        row['differences_4M_minus_1M'] = {
            'mean_return': evaluation['mean_return'] - one['mean_return'],
            'success_percentage_points': round(100 * (evaluation['success_rate'] - one['success_rate']), 10),
            'mean_cost': evaluation['mean_cost'] - one['mean_cost'],
            'cost_cvar_20': evaluation['cost_cvar_20'] - one['cost_cvar_20']}
    records = previous['records'] + [row]
    records.sort(key=lambda r: (0 if r['job'].startswith('point-goal') else 1, r['training_seed']))
    assert len(records) == len({r['job'] for r in records}) == 8
    for record in records:
        rel = Path(record['native_4M']['relative_path'])
        source = native if record['job'] == JOB else OLD / rel
        for destination in (CACHE / rel, OUT / rel):
            copy_exact(source, destination, record['native_4M']['sha256'])
    groups = {env: sorted(r['training_seed'] for r in records if r['environment'] == env)
              for env in ('SafetyPointGoal1-v0', 'SafetyPointButton1-v0')}
    notes = []
    for env, seeds in groups.items():
        selected = [r for r in records if r['environment'] == env]
        successes = [r['native_4M']['metrics']['success_rate'] for r in selected]
        ratios = [r['native_4M']['metrics']['direction_geometry_high_risk_top_20_percent']['normal_to_tangent_variance_ratio'] for r in selected]
        notes.append(f'{env}: available training seeds{seeds}, native4M success rates{successes}, high-risk20% variance-ratio range{min(ratios):.4f}-{max(ratios):.4f}. Diagnostic partial learning-curve context, not the full five-seed final cohort.')
    notes += [
        'Availability is determined by native milestone arrival, never by return, cost, success or a configuration gate.',
        'All eight results use100 fixed-schedule evaluation episodes. Episodes are not additional independent training runs.',
        'No subset mean, significance test, matchedC0 comparison or causal superiority claim is made.',
        'Native training cumulative costs describe retained lineages, not all physical recovery/discarded work or unknown unlogged tails.',
        'Historical2027 original-full-parent gaps remain unresolved. JSON/hash matching does not repair ancestry; the separate final multi-hop locator decision is still pending.',
        'All10M Point budgets and the frozen method remain unchanged; no early stopping, extra experiment cell or extra rollout.',
        'Only one newly indexed manifest and small native metrics blob were retrieved. The previous seven local raw results were reused, not re-evaluated.'
    ]
    summary = {**{k: previous[k] for k in ('schema_version', 'kind', 'scientific_source_commit',
                'point_total_training_budget_steps', 'milestone_steps', 'planned_seeds_per_environment',
                'statistical_unit', 'episodes_are_not_additional_independent_training_runs')},
               'assembled_at_utc': datetime.now(timezone.utc).isoformat(),
               'previous_summary': '../point_4M_available7_20261021/summary.json',
               'available_seeds_per_environment': groups,
               'seeds_not_yet_at_4M_per_environment': {k: sorted(set(range(2027, 2032)) - set(v)) for k, v in groups.items()},
               'available_training_seed_counts': {k: len(v) for k, v in groups.items()},
               'availability_basis': 'Seven previously published native4M results plus newly archived Button2028 native4M JSON; not selected by metrics.',
               'aggregate_or_significance_test_performed': False, 'additional_evaluation_rollouts': 0,
               'scientific_parameters_changed': False, 'early_stop_or_configuration_selection_performed': False,
               'full_checkpoint_provenance_audit_executed': False, 'final_scientific_audit_executed': False,
               'final_scientific_cohort_report': False, 'full_scientific_objective_complete': False,
               'new_evaluation_archive_receipt': meta, 'records': records, 'interpretation_limits': notes}
    fields = ['job', 'environment', 'training_seed', 'mean_return', 'success_rate', 'mean_cost',
              'cost_cvar_20', 'training_cumulative_cost', 'high_risk_normal_tangent_variance_ratio',
              'high_risk_dangerous_outward_velocity']
    for root in (CACHE, OUT):
        with (root / 'summary.json').open('x') as stream:
            stream.write(json.dumps(summary, indent=2) + '\n')
        with (root / 'per_training_seed_metrics.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in records:
                m = record['native_4M']['metrics']; g = m['direction_geometry_high_risk_top_20_percent']
                writer.writerow({**{k: record[k] for k in fields[:3]}, **{k: m[k] for k in fields[3:8]},
                                 fields[8]: g['normal_to_tangent_variance_ratio'], fields[9]: g['dangerous_outward_velocity']})
    print(json.dumps({'new_native_4M_metrics': metrics(evaluation),
                      'same_seed_4M_minus_1M': row.get('differences_4M_minus_1M'),
                      'available_training_seeds': groups, 'new_native_sha256': meta['sha256'],
                      'additional_rollouts': 0}), flush=True)


if __name__ == '__main__':
    main()

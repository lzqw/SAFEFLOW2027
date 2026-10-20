#!/usr/bin/env python3
"""Append one archived native evaluation to six published results; never evaluate."""
import csv
import hashlib
import json
import math
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
PARENT = Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/partial_milestones')
OLD = PARENT / 'point_4M_available6_20261020'
OUT = PARENT / 'point_4M_available7_20261021'
CACHE = Path('/home/lzqw/SAFEFLOW2027_artifacts/recovered_a100_4080/partial_milestones/point_4M_available7_20261021')
SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'
JOB = 'point-goal-2028'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_exact(source, destination, sha):
    assert digest(source) == sha
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert digest(destination) == sha, 'Never overwrite differing native evidence'
    else:
        shutil.copyfile(source, destination)


def metrics(e):
    return {k: e[k] for k in ('mean_return', 'success_rate', 'mean_cost', 'cost_cvar_20',
                              'training_cumulative_cost', 'direction_geometry_high_risk_top_20_percent')}


def current_interpretation_notes(records, previous_notes):
    notes = list(previous_notes)
    goals = [r for r in records if r['job'].startswith('point-goal')]
    buttons = [r for r in records if r['job'].startswith('point-button')]
    seeds = '/'.join(str(r['training_seed']) for r in goals)
    successes = '/'.join(f"{100 * r['native_4M']['metrics']['success_rate']:g}%" for r in goals)
    notes[0] = ('PointGoal4M success for available training seeds' + seeds + ' is' + successes
                + '. This is not the complete five-seed final cohort. Seed2027 original-full-parent provenance remains unresolved.')
    def span(rows):
        values = [r['native_4M']['metrics']['direction_geometry_high_risk_top_20_percent']['normal_to_tangent_variance_ratio'] for r in rows]
        return f'{min(values):.4f}-{max(values):.4f}'
    notes[2] = ('High-risk20% normal-to-tangent variance ratios at4M span' + span(goals)
                + ' for the available Goal seeds and' + span(buttons)
                + ' for Button. These are diagnostic ranges, not a causal gain overC0 or a training/configuration gate.')
    return notes


def refresh_existing_notes():
    # Update only scope-dependent descriptive text after adding a seed; keep all
    # native bytes, recorded metrics, paired1M context and CSV unchanged.
    s = read(OUT / 'summary.json')
    assert s == read(CACHE / 'summary.json')
    s['interpretation_limits'] = current_interpretation_notes(s['records'], s['interpretation_limits'])
    for root in (CACHE, OUT):
        temporary = root / 'summary.current_notes.tmp'
        with temporary.open('x') as stream:
            stream.write(json.dumps(s, indent=2) + '\n')
        temporary.replace(root / 'summary.json')
    print(json.dumps({'local_descriptive_notes_updated': True, 'native_metrics_or_CSV_changed': False,
                      'network_calls_or_evaluation_rollouts': 0}))


def protocol(e, milestone):
    assert e['milestone'] == milestone and e['episodes'] == 100
    assert e['base_seeds'] == list(range(32027, 32032)) and e['episodes_per_base_seed'] == 20
    assert e['trajectory_seeds'] == [32027, 32029, 32031]
    assert len(e['returns']) == len(e['costs']) == 100
    assert all(math.isfinite(e[k]) for k in ('mean_return', 'success_rate', 'mean_cost', 'cost_cvar_20'))
    assert 0 <= e['success_rate'] <= 1


def main():
    assert not (OUT / 'summary.json').exists(), 'Inspect the published summary; do not replay'
    previous = read(OLD / 'summary.json')
    assert previous['scientific_source_commit'] == SOURCE and len(previous['records']) == 6
    archive = read(BASE / 'scientific_evidence_archive_state.json')['archived']['5080/' + JOB]
    assert archive['archived'] is True
    # One exact newly indexed manifest and one small metrics blob. No historical
    # manifest scan, checkpoint load, policy inference, or evaluation command.
    code = f'''
import hashlib,json,re
from pathlib import Path
root=Path('/home/carla/4T/SAFEFLOW2027_4080/scientific_evidence_archive')
assert root.stat().st_dev!=Path('/home/carla/SAFEFLOW2027_4080').stat().st_dev
p=Path({archive['manifest']!r})
assert p.parent==root/'manifests/5080/point-goal-2028'
raw=p.read_bytes();manifest_sha=hashlib.sha256(raw).hexdigest()
assert manifest_sha==p.stem.rsplit('-',1)[1]
m=json.loads(raw)
assert m['format']=='safeflow-raw-evidence-snapshot-v1'
assert m['origin']=='5080' and m['job']=='point-goal-2028' and m['source_commit']=={SOURCE!r}
entries=[r for r in m['files'] if r['path'].endswith('/artifacts/evaluation_step_4000000.json')]
assert entries,'Native4M JSON not in this snapshot; wait for a new archive, never rerun evaluation'
assert len({{r['sha256'] for r in entries}})==1,'Ambiguous milestone bytes: do not select by performance'
sha=entries[0]['sha256'];assert re.fullmatch('[0-9a-f]{{64}}',sha)
blob=root/'blobs'/sha;data=blob.read_bytes()
assert hashlib.sha256(data).hexdigest()==sha and all(r['bytes']==len(data) for r in entries)
print(json.dumps({{'manifest':str(p),'manifest_sha256':manifest_sha,'manifest_created_at_ns':m['created_at_ns'],
 'source_commit':m['source_commit'],'blob':str(blob),'sha256':sha,'bytes':len(data),
 'all_identical_SHA_source_relative_paths':[r['path'] for r in entries]}}))
'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '4080',
                             'python3 -c ' + shlex.quote(code)], capture_output=True, text=True,
                            check=True, timeout=90)
    meta = json.loads(result.stdout)
    relative = Path('native_evaluations') / JOB / 'evaluation_step_4000000.json'
    native = CACHE / relative
    native.parent.mkdir(parents=True, exist_ok=True)
    if not native.exists():
        subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                        '4080:' + meta['blob'], str(native)], capture_output=True, check=True, timeout=90)
    assert digest(native) == meta['sha256'] and native.stat().st_size == meta['bytes']
    e = read(native)
    protocol(e, 4000000)
    row = {'job': JOB, 'environment': 'SafetyPointGoal1-v0', 'training_seed': 2028,
           'episodes_each_milestone': 100, 'episodes_per_base_seed': 20,
           'base_seeds': e['base_seeds'], 'trajectory_seeds': e['trajectory_seeds'],
           'native_4M': {'relative_path': relative.as_posix(), **meta, 'metrics': metrics(e)},
           'legacy_provenance_notice': 'This metrics-only assembly does not audit full checkpoint ancestry or certify final provenance.'}
    one_relative = Path('../first_milestone_20_cells_20261020/native_evaluations') / JOB / 'evaluation_step_1000000.json'
    one_path = (OUT / one_relative).resolve()
    if one_path.is_file():
        one = read(one_path)
        protocol(one, 1000000)
        row['same_seed_native_1M'] = {'previously_published_relative_path': one_relative.as_posix(),
                                      'sha256': digest(one_path), 'metrics': metrics(one)}
        row['differences_4M_minus_1M'] = {
            'mean_return': e['mean_return'] - one['mean_return'],
            'success_percentage_points': round(100 * (e['success_rate'] - one['success_rate']), 10),
            'mean_cost': e['mean_cost'] - one['mean_cost'],
            'cost_cvar_20': e['cost_cvar_20'] - one['cost_cvar_20']}
    records = previous['records'] + [row]
    records.sort(key=lambda r: (0 if r['job'].startswith('point-goal') else 1, r['training_seed']))
    assert len({r['job'] for r in records}) == 7
    for r in records:
        rel = Path(r['native_4M']['relative_path'])
        source = native if r['job'] == JOB else OLD / rel
        for destination in (CACHE / rel, OUT / rel):
            copy_exact(source, destination, r['native_4M']['sha256'])
    groups = {env: sorted(r['training_seed'] for r in records if r['environment'] == env)
              for env in ('SafetyPointGoal1-v0', 'SafetyPointButton1-v0')}
    summary = {
        'schema_version': 2, 'kind': 'available_native_Point_4M_learning_curve_context_not_final_cohort_report',
        'assembled_at_utc': datetime.now(timezone.utc).isoformat(),
        'previous_summary': '../point_4M_available6_20261020/summary.json',
        'scientific_source_commit': SOURCE, 'point_total_training_budget_steps': 10000000,
        'milestone_steps': 4000000, 'planned_seeds_per_environment': list(range(2027, 2032)),
        'available_seeds_per_environment': groups,
        'seeds_not_yet_at_4M_per_environment': {k: sorted(set(range(2027, 2032)) - set(v)) for k, v in groups.items()},
        'availability_basis': 'Six already-published native4M results plus newly archived Goal2028 native4M JSON; availability only, never selected by return/cost.',
        'statistical_unit': 'training_seed_per_environment',
        'available_training_seed_counts': {k: len(v) for k, v in groups.items()},
        'episodes_are_not_additional_independent_training_runs': True,
        'aggregate_or_significance_test_performed': False, 'additional_evaluation_rollouts': 0,
        'scientific_parameters_changed': False, 'early_stop_or_configuration_selection_performed': False,
        'full_checkpoint_provenance_audit_executed': False, 'final_scientific_audit_executed': False,
        'final_scientific_cohort_report': False, 'full_scientific_objective_complete': False,
        'new_evaluation_archive_receipt': meta, 'records': records,
        'interpretation_limits': current_interpretation_notes(records, previous['interpretation_limits']) + [
            'Available Goal n=4 and Button n=3, not complete five-training-seed final cohorts.',
            'Every reported native milestone uses100 fixed-schedule evaluation episodes, not extra independent training runs.',
            'No matched C0 comparison or causal superiority claim is made; all10M Point budgets remain.',
            'Historical2027 original-full-parent gaps remain unresolved; JSON/hash matching does not repair checkpoint ancestry.',
            'Logged training cumulative cost is accepted-lineage cost, not all physically executed recovery work.',
            'Only one newly indexed archive/metrics blob was read; previous six local native JSONs were reused, not re-evaluated.']}
    fields = ['job', 'environment', 'training_seed', 'mean_return', 'success_rate', 'mean_cost', 'cost_cvar_20',
              'training_cumulative_cost', 'high_risk_normal_tangent_variance_ratio', 'high_risk_dangerous_outward_velocity']
    for root in (CACHE, OUT):
        root.mkdir(parents=True, exist_ok=True)
        with (root / 'summary.json').open('x') as stream:
            stream.write(json.dumps(summary, indent=2) + '\n')
        with (root / 'per_training_seed_metrics.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for r in records:
                m = r['native_4M']['metrics'];g = m['direction_geometry_high_risk_top_20_percent']
                writer.writerow({**{k: r[k] for k in fields[:3]}, **{k: m[k] for k in fields[3:8]},
                                 fields[8]: g['normal_to_tangent_variance_ratio'], fields[9]: g['dangerous_outward_velocity']})
    print(json.dumps({'new_job': JOB, 'new_native_4M_metrics': metrics(e),
                      'same_seed_4M_minus_1M': row.get('differences_4M_minus_1M'),
                      'available_training_seeds': groups, 'additional_rollouts': 0}), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Copy existing milestone evaluations/segment logs, not checkpoints or rollouts.

This descriptive bundle cannot repair missing original checkpoint provenance.
Raw segment CSVs stay separate: retries and rollback tails are not silently
stitched, deduplicated, counted as independent runs, or labelled accepted work.
"""
import base64
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess

BASE = Path(__file__).resolve().parent
PARENT = Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/partial_milestones')
OUT = PARENT / 'native_learning_curves_all20_20261105'
ROOT = '/storage_0/SAFEFLOW2027_lzqw_uid1007/terminal_matrix_20261024/runs/shapo_p47_four_env/runs'
COMMIT = '62ca7399357c10645a955dcbead7ed2153a82c69'
ENVS = {
    'point-goal': ('SafetyPointGoal1-v0', [1000000, 4000000, 10000000]),
    'point-button': ('SafetyPointButton1-v0', [1000000, 4000000, 10000000]),
    'ant': ('Ant-v4', [500000, 2000000, 5000000]),
    'walker2d': ('Walker2d-v4', [500000, 2000000, 5000000]),
}
PUBLISHED_BACKUPS = {
    ('ant-2031', 500000): (
        'first_milestone_20_cells_20261020/native_evaluations/ant-2031/evaluation_step_500000.json',
        '634c3d9169fa714225690915b8a27eefad8db0c076f67f65a3fd764ba90f0570'),
    ('ant-2031', 2000000): (
        'ant_2M_available5_20261020/native_evaluations/ant-2031/evaluation_step_2000000.json',
        '85841f2d6e672f0db4d79a894f8db12c7e1cc080275a742ba0f9716f72292b46'),
}


def main():
    assert not OUT.exists(), 'Preserve immutable snapshots; never overwrite'
    remote = f'''
import base64,hashlib,json
from pathlib import Path
root=Path({ROOT!r}); records=[]; total=0
for slug,(environment,milestones) in {ENVS!r}.items():
 for seed in range(2027,2032):
  d=root/(slug+'-'+str(seed)); assert d.is_dir() and not d.is_symlink()
  r=json.loads((d/'receipt.json').read_text())
  assert r['status']=='passed' and r['source_commit']=={COMMIT!r} and r['source_dirty'] is False
  assert r['environment']==environment and r['seed']==seed and r['steps']==milestones[-1]
  chosen={{}}; selected=[d/'receipt.json']
  for step in milestones:
   matches=list(d.glob('training/**/artifacts/evaluation_step_'+str(step)+'.json'))
   if not matches:continue
   latest=max(p.stat().st_mtime_ns for p in matches)
   newest=[p for p in matches if p.stat().st_mtime_ns==latest]
   assert len(newest)==1,'Ambiguous native newest-artifact identity'
   chosen[str(step)]=str(newest[0].relative_to(d))
   selected.extend(sorted(matches))
  selected.extend(sorted(d.glob('training/**/progress.csv')))
  files=[]
  for p in selected:
   assert p.is_file() and not p.is_symlink()
   data=p.read_bytes(); total+=len(data)
   assert total<=150*1024**2,'Refuse unexpectedly large non-checkpoint export'
   files.append(dict(path=str(p.relative_to(d)),bytes=len(data),mtime_ns=p.stat().st_mtime_ns,
    sha256=hashlib.sha256(data).hexdigest(),base64=base64.b64encode(data).decode('ascii')))
  records.append(dict(job=d.name,environment=environment,training_seed=seed,
   source_container=str(d),selected_evaluations=chosen,files=files))
print(json.dumps(records))
'''
    response = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'tjucta-a100',
         'nice -n 19 python3 -c ' + shlex.quote(remote)],
        capture_output=True, text=True, timeout=180,
    )
    if response.returncode:
        raise SystemExit('Native export failed before writing any local output:\n' + response.stderr[-6000:])
    records = json.loads(response.stdout)
    assert len(records) == 20
    for record in records:
        slug = record['job'].rsplit('-', 1)[0]
        for step in ENVS[slug][1]:
            if str(step) in record['selected_evaluations']:
                continue
            source_rel, expected_hash = PUBLISHED_BACKUPS[(record['job'], step)]
            source = PARENT / source_rel
            data = source.read_bytes()
            assert hashlib.sha256(data).hexdigest() == expected_hash
            path = 'local_published_backups/evaluation_step_' + str(step) + '.json'
            record['selected_evaluations'][str(step)] = path
            record['files'].append(dict(path=path, sha256=expected_hash, bytes=len(data),
                                        published_source=source_rel,
                                        origin='existing_published_native_backup_not_remote_repair',
                                        base64=base64.b64encode(data).decode('ascii')))
    decoded = []
    metrics = []
    fields = ['mean_return', 'mean_cost', 'cost_cvar_20', 'training_cumulative_cost']
    for record in records:
        slug = record['job'].rsplit('-', 1)[0]
        task_metric = 'success_rate' if slug.startswith('point-') else 'failure_rate'
        payloads = {}
        for item in record['files']:
            rel = Path(record['job']) / item['path']
            assert not rel.is_absolute() and '..' not in rel.parts
            data = base64.b64decode(item.pop('base64'), validate=True)
            assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
            decoded.append((rel, data))
            payloads[item['path']] = data
        for step_text, path in record['selected_evaluations'].items():
            step = int(step_text)
            e = json.loads(payloads[path])
            assert e['milestone'] == step and e['episodes'] == 100
            assert e['base_seeds'] == list(range(32027, 32032)) and e['episodes_per_base_seed'] == 20
            row = dict(job=record['job'], environment=record['environment'],
                       training_seed=record['training_seed'], milestone_steps=step,
                       task_metric_name=task_metric, task_metric=e[task_metric],
                       native_artifact=str(Path(record['job']) / path))
            for field in fields:
                assert math.isfinite(float(e[field])), (record['job'], step, field)
                row[field] = e[field]
            assert math.isfinite(float(row['task_metric']))
            g = e.get('direction_geometry_high_risk_top_20_percent', {})
            for name in ['normal_to_tangent_variance_ratio', 'dangerous_outward_velocity']:
                value = g.get(name)
                row['high_risk_' + name] = value if isinstance(value, (int, float)) and math.isfinite(value) else None
            if step == ENVS[slug][1][-1]:
                reference = PARENT / (slug + '_native_final_all5') / 'native_evaluations' / record['job'] / ('evaluation_step_' + str(step) + '.json')
                assert hashlib.sha256(reference.read_bytes()).digest() == hashlib.sha256(payloads[path]).digest(), 'Final evaluation differs from published native results'
            metrics.append(row)
    assert len(metrics) == 60
    grouped = []
    metric_fields = fields + ['task_metric', 'high_risk_normal_to_tangent_variance_ratio', 'high_risk_dangerous_outward_velocity']
    for slug, (environment, milestones) in ENVS.items():
        for step in milestones:
            rows = [r for r in metrics if r['environment'] == environment and r['milestone_steps'] == step]
            assert sorted(r['training_seed'] for r in rows) == list(range(2027, 2032))
            values = {}
            for field in metric_fields:
                numbers = [r[field] for r in rows if r[field] is not None]
                values[field] = dict(n=len(numbers), mean=statistics.mean(numbers) if numbers else None,
                                     sample_sd=statistics.stdev(numbers) if len(numbers) > 1 else None)
            grouped.append(dict(environment=environment, milestone_steps=step,
                                task_metric_name=rows[0]['task_metric_name'], metrics=values))
    OUT.mkdir(parents=True)
    for rel, data in decoded:
        path = OUT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    columns = ['job', 'environment', 'training_seed', 'milestone_steps', 'task_metric_name'] + fields + metric_fields[4:] + ['native_artifact']
    with (OUT / 'per_training_seed_milestones.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(metrics)
    summary = dict(kind='descriptive_native_three_milestone_curves_not_final_strict_audit',
                   generated_at_utc=datetime.now(timezone.utc).isoformat(),
                   scientific_source_commit=COMMIT, training_cells=20,
                   prespecified_evaluation_cells=60, statistical_unit='training_seed',
                   evaluation_episodes_per_training_seed_per_milestone=100,
                   groups=grouped, records=records,
                   helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   scientific_parameters_changed=False, additional_rollouts=0,
                   checkpoint_payloads_loaded=0, remote_files_modified=0,
                   final_strict_audit_passed=False,
                   caveats=[
                       'Three prespecified evaluations are milestone curves, not dense held-out evaluations.',
                       'All raw progress segments are retained separately; no unverifiable lineage stitch or recovery-tail deduplication.',
                       'Latest modification time selects native evaluations, matching frozen newest() semantics, never performance.',
                       'Sample SD is across five training seeds, not individual evaluation episodes.',
                       'Training cumulative costs are native accepted-lineage counters, not total physical recovery costs.',
                       'Historical missing-parent and multi-hop lookup gaps remain unresolved; hashes are not provenance proof.',
                   ])
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(OUT), evaluation_cells=len(metrics),
                          copied_files=len(decoded), copied_bytes=sum(len(data) for _, data in decoded),
                          groups=grouped), indent=2))


if __name__ == '__main__':
    main()

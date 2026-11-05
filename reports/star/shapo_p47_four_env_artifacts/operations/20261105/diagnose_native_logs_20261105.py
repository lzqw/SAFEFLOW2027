#!/usr/bin/env python3
"""Summarize existing evaluation and final-segment logs without model execution."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from collect_native_learning_curves_20261105 import OUT as INPUT, ENVS

OUT = INPUT.parent / 'native_effect_diagnosis_20261105'
WINDOW_STEPS = 500000
LOG_FIELDS = [
    'Dual/GlobalSafety', 'Dual/Energy', 'Loss/Actor', 'Loss/Direction',
    'Reference/Reliability', 'Reference/BoundaryReliability',
    'Reference/GradientReliability', 'Reference/EnsembleReliability',
    'Reference/VNVT', 'Reference/RawDangerousNormalVelocity',
    'Reference/ClippingRate', 'Reference/SafetyBandSampleRatio',
    'Critic/RewardQAbsMax', 'Critic/RewardTD', 'Critic/CostQMean',
    'Critic/CostTD', 'Numerics/CostActionGradientNorm',
    'Metrics/EpRet', 'Metrics/EpCost',
]
EVALUATION_FIELDS = [
    'mean_action_norm', 'mean_path_length', 'mean_start_end_displacement',
    'mean_step_displacement', 'stall_ratio', 'step_stall_ratio',
    'cost_to_go_mae', 'calibration_error', 'brier_score', 'brier_skill',
]


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def stats(values):
    values = [v for v in values if v is not None]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values) > 1 else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def main():
    assert not OUT.exists(), 'Preserve immutable diagnostic snapshots'
    manifest = json.loads((INPUT / 'summary.json').read_text())
    logs = []
    evaluation_rows = []
    for record in manifest['records']:
        root = INPUT / record['job']
        receipt = json.loads((root / 'receipt.json').read_text())
        # This is local descriptive log selection, not historical parent repair.
        relative = 'training/' + receipt['log_dir'].split('/training/', 1)[1] + '/progress.csv'
        identity = next(f for f in record['files'] if f['path'] == relative)
        path = root / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity['sha256']
        with path.open(newline='') as stream:
            all_rows = list(csv.DictReader(stream))
        total = receipt['steps']
        rows = [r for r in all_rows if finite(r.get('TotalEnvSteps')) is not None
                and total - WINDOW_STEPS < float(r['TotalEnvSteps']) <= total]
        assert rows, record['job']
        log = dict(job=record['job'], environment=record['environment'],
                   training_seed=record['training_seed'],
                   source_csv=str(path.relative_to(INPUT)), source_sha256=identity['sha256'],
                   requested_tail_window_steps=WINDOW_STEPS, observed_epochs=len(rows),
                   first_observed_step=min(float(r['TotalEnvSteps']) for r in rows),
                   last_observed_step=max(float(r['TotalEnvSteps']) for r in rows))
        log['metrics'] = {key: stats([finite(row.get(key)) for row in rows]) for key in LOG_FIELDS}
        duals = [finite(row.get('Dual/GlobalSafety')) for row in rows]
        log['global_dual_at_100_fraction'] = statistics.mean(float(v >= 100 - 1e-6) for v in duals if v is not None)
        logs.append(log)
        for step_text, rel in record['selected_evaluations'].items():
            e = json.loads((root / rel).read_text())
            row = dict(job=record['job'], environment=record['environment'],
                       training_seed=record['training_seed'], milestone_steps=int(step_text))
            row.update({key: finite(e.get(key)) for key in EVALUATION_FIELDS})
            evaluation_rows.append(row)
    groups = []
    movements = []
    for slug, (environment, milestones) in ENVS.items():
        cells = [r for r in logs if r['environment'] == environment]
        assert len(cells) == 5
        groups.append(dict(environment=environment,
            global_dual_at_100_fraction=stats([r['global_dual_at_100_fraction'] for r in cells]),
            observed_epochs_per_training_seed=[r['observed_epochs'] for r in cells],
            metrics={key: stats([r['metrics'][key]['mean'] for r in cells]) for key in LOG_FIELDS}))
        for step in milestones:
            cells = [r for r in evaluation_rows if r['environment'] == environment and r['milestone_steps'] == step]
            movements.append(dict(environment=environment, milestone_steps=step,
                metrics={key: stats([r[key] for r in cells]) for key in EVALUATION_FIELDS}))
    OUT.mkdir(parents=True)
    summary = dict(kind='existing_native_logs_diagnosis_not_causal_experiment',
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        scientific_source_commit=manifest['scientific_source_commit'],
        user_instruction='Preserve existing results, diagnose effects, no additional training',
        statistical_unit='training_seed', training_cells=20,
        final_segment_tail_window_steps=WINDOW_STEPS, final_segment_groups=groups,
        evaluation_diagnostic_groups=movements, per_training_seed_tail=logs,
        source_manifest_sha256=hashlib.sha256((INPUT / 'summary.json').read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_training_steps=0, new_evaluation_rollouts=0, models_loaded=0,
        scientific_configuration_changed=False, strict_audit_passed=False,
        limitations=[
            'Tail statistics use only each native receipt final segment, never mixed discarded recovery tails.',
            'If the final segment is shorter than 500k, its observed row count is shown; no synthetic rows or extrapolation.',
            'Group means are means of per-training-seed row means; optional missing values retain their n counts.',
            'GlobalSafety=100 is checked against the frozen ceiling; saturation alone is not causal proof.',
            'Training-batch reference factors and evaluation high-risk diagnostics use different sample populations.',
            'Scalar loss magnitudes do not establish gradient dominance.',
            'No matched control, ablation, new rollout, or formal checkpoint provenance acceptance is implied.',
        ])
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    with (OUT / 'per_training_seed_evaluation_diagnostics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['job', 'environment', 'training_seed', 'milestone_steps'] + EVALUATION_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(evaluation_rows, key=lambda r: (r['job'], r['milestone_steps'])))
    print(json.dumps(dict(output=str(OUT), final_segment_groups=groups,
                          evaluation_diagnostic_groups=movements), indent=2))


if __name__ == '__main__':
    main()

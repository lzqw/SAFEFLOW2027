"""Frozen margin diagnostics on a common pool of 30K saved actual actions.

All three registered final policies contribute their complete last10K replay
window. No environment, actor, optimizer, resampling, or independent test.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from scripts.star.arm_safe_entropy.geometry import LocalMarginEnsemble


PROJECT = Path(__file__).resolve().parents[2]
PLAN = PROJECT/'configs/paper_crax_payload/simulation_v12_panda_safereach_deployment_modes.json'
ROOT = Path('/storage_0/SAFEFLOW2027_lzqw_uid1007/simulation_v12')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def replay_window(replay, size=10000):
    if not 0 < size <= replay['count'] <= replay['capacity']:
        raise ValueError('Require every transition in the fixed replay window')
    indices = (replay['cursor']-size+np.arange(size)) % replay['capacity']
    values = {k:np.asarray(replay['data'][k])[indices].copy() for k in ('obs','action','margin','cost')}
    if (values['obs'].shape != (size,25) or values['action'].shape != (size,3)
            or not all(np.isfinite(v).all() for v in values.values())
            or not np.array_equal(values['margin']>0., values['cost']>0.)):
        raise ValueError('Preserve actual signed-margin labels and native25D inputs')
    return values


def statistics(heads, actual, mask):
    heads, actual, mask = np.asarray(heads), np.asarray(actual), np.asarray(mask)
    if heads.shape != (2,len(actual)) or mask.shape != actual.shape:
        raise ValueError('Two exact native heads and one label per actual action')
    count = int(mask.sum())
    if not count:
        return dict(count=0)
    mean = heads[:,mask].mean(0)
    error = mean-actual[mask]
    unsafe = actual[mask]>0.
    return dict(count=count, actual_unsafe=int(unsafe.sum()),
        actual_mean_mm=float(actual[mask].mean()*1000), predicted_mean_mm=float(mean.mean()*1000),
        bias_mm=float(error.mean()*1000), mae_mm=float(np.abs(error).mean()*1000),
        rmse_mm=float(np.sqrt(np.mean(error**2))*1000),
        error_p05_mm=float(np.quantile(error,.05)*1000), error_p95_mm=float(np.quantile(error,.95)*1000),
        false_safe_mean_count=int((unsafe & (mean<=0.)).sum()),
        unsafe_both_heads_safe_count=int((unsafe & (heads[:,mask]<=0.).all(0)).sum()),
        mean_head_std_mm=float(heads[:,mask].std(0).mean()*1000))


def write_csv(path, rows):
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args(); torch.set_num_threads(1)
    if (not args.output.resolve().is_relative_to(ROOT.resolve(strict=True))
            or shutil.disk_usage(ROOT).free < 41*1024**3):
        raise ValueError('Use admitted data disk and preserve40GiB plus diagnostic growth')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.025, device)
    plan = json.loads(PLAN.read_text())
    cases = [c for c in plan['cases'] if c['line']=='safereach']
    if [c['label'] for c in cases] != ['P5','Iso','Metric']:
        raise ValueError('Retain all three frozen final references')
    args.output.mkdir(exist_ok=False)
    registration = dict(protocol='v12_safereach_retrospective_common_replay_margin_calibration',
        source_commit=args.source_commit, native_reference_plan_sha256=sha256(PLAN), cases=cases,
        actual_training_window=[240001,250000], controls_each=10000,
        input_selection='all actual actions in the fixed final10K of each of three saved replay buffers',
        models=['geometry_EMA','online_margin'],
        strata='all;actual_unsafe>0;abs_actual_margin<=5mm;safe_interior<-5mm;first3episode_controls',
        common_pool='all30K actions scored by each margin model; own-pool rows are potentially in-sample',
        model_site='direct executed-action margin;not sampler tail pullback normal or causal safety proof',
        independent_test_used=False, new_training_steps=0, new_simulation_controls=0,
        optimizer_steps=0, generated_policy_actions=0, no_new_critics=True)
    (args.output/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    inputs, models, provenance = {}, {}, []
    for case in cases:
        directory = ROOT/'panda-safereach-mode-inputs'/case['relative_run']
        checkpoint = directory/case['checkpoint_file']
        if sha256(checkpoint) != case['checkpoint_sha256']:
            raise ValueError('Require the registered final native checkpoint SHA')
        receipt = json.loads((directory/'receipt.json').read_text())
        saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        config = json.loads((PROJECT/case['plan']).read_text())
        arm = next(a for a in config['arms'] if a['name']==case['arm'])
        if (saved['format'] != case['format'] or saved['configuration'] != config or saved['arm'] != arm
                or saved['seed'] != 2034 or saved['steps'] != 250000
                or not saved['source_commit'].startswith(case['training_source'])
                or receipt['status'] != 'completed' or receipt['steps'] != 250000
                or receipt['source_commit'] != saved['source_commit'] or receipt['engineering_only']):
            raise ValueError('Retain native identity, completed budget and scientific receipt')
        inputs[case['label']] = replay_window(saved['replay'])
        for name, state in (('geometry_EMA',saved['learner']['ea_stability_v2']['margin_geometry']),
                            ('online_margin',saved['learner']['margin'])):
            model = LocalMarginEnsemble().to(device).requires_grad_(False).eval()
            model.load_state_dict(state,strict=True)
            models[case['label'],name] = model
        provenance.append(dict(label=case['label'],checkpoint_sha256=case['checkpoint_sha256'],
            receipt_sha256=sha256(directory/'receipt.json'),training_source=saved['source_commit']))
        del saved
    summaries, unsafe_records = [], []
    with torch.no_grad():
        for source, values in inputs.items():
            obs = torch.as_tensor(values['obs'],device=device)
            action = torch.as_tensor(values['action'],device=device)
            actual = values['margin']
            # Remaining-time input is native pre-action time:1,.99,.98 are the first3 controls.
            early = values['obs'][:,-1]>=.98-1e-6
            for (label,name), model in models.items():
                heads = torch.cat([model(obs[i:i+1024],action[i:i+1024])
                    for i in range(0,len(obs),1024)],dim=1).cpu().numpy()
                if not np.isfinite(heads).all():
                    raise FloatingPointError('Nonfinite frozen margin predictions')
                groups = dict(all=np.ones(len(obs),dtype=bool), actual_unsafe=actual>0.,
                    near_actual_boundary=np.abs(actual)<=.005, safe_interior=actual<-.005,
                    first3_episode_controls=early, first3_unsafe=early & (actual>0.))
                for group, mask in groups.items():
                    summaries.append(dict(model_policy=label,model=name,behavior_pool=source,
                        group=group,**statistics(heads,actual,mask)))
                for i in np.flatnonzero(actual>0.):
                    unsafe_records.append(dict(model_policy=label,model=name,behavior_pool=source,
                        original_transition=240001+int(i),actual_margin_mm=float(actual[i]*1000),
                        predicted_head0_mm=float(heads[0,i]*1000),predicted_head1_mm=float(heads[1,i]*1000),
                        first3_episode_controls=bool(early[i])))
    write_csv(args.output/'summary.csv',summaries)
    write_csv(args.output/'all_actual_unsafe_predictions.csv',unsafe_records)
    (args.output/'receipt.json').write_text(json.dumps(dict(status='completed',source_commit=args.source_commit,
        provenance=provenance,saved_actual_controls=30000,models_scored=6,model_action_scores=180000,
        unsafe_behavior_controls=sum(int((v['margin']>0).sum()) for v in inputs.values()),
        summary_rows=len(summaries),new_training_steps=0,new_simulation_controls=0,
        optimizer_steps=0,generated_policy_actions=0,independent_test_used=False,
        all_saved_failures_retained=True,checkpoint_mutations=0,
        uncertainty_is_not_calibrated_probability=True,causal_proof=False),indent=2)+'\n')


if __name__=='__main__':
    main()

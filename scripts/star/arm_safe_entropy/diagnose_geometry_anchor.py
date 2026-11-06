"""Frozen-weight sampler intervention, not a trained method or primary eval.

Only the geometry query location changes for the counterfactual: the same
actor's deterministic endpoint supplies one detached reference held through
the stochastic path. There is no candidate scoring, selection or execution.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.star.arm_safe_entropy.conditional_metrics import endpoint_metrics, reference_metrics
from scripts.star.arm_safe_entropy.geometry import LocalMarginEnsemble
from scripts.star.arm_safe_entropy.models import ArmFlowActor


def average(rows, group, key):
    values = [r[group][key] for r in rows if r[group][key] is not None]
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs=4, type=Path, required=True)
    parser.add_argument('--state-pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    configs = [json.loads((r / 'run_config.json').read_text()) for r in args.runs]
    if {(c['method'], c['training_seed']) for c in configs} != {
            (method, seed) for method in ['flow_iso', 'flow_learned'] for seed in [2027, 2028]}:
        parser.error('Use both fixed-final Iso/Learned development seeds')
    if args.output.exists():
        parser.error('Never replace an earlier sampler diagnosis')
    pool = json.loads(args.state_pool.read_text())
    if len(pool) != 256 or len({tuple(r['observation']) for r in pool}) != 256:
        parser.error('Reuse the256-distinct-state pool, not a newly selected favorable subset')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    obs = torch.as_tensor(np.asarray([r['observation'] for r in pool], np.float32), device=args.device)
    batch = obs.repeat_interleave(128, dim=0)
    records, summaries = [], []
    for run, config in zip(args.runs, configs):
        state = torch.load(run / 'policy_300000.pt', map_location=args.device, weights_only=False)
        rho = config.get('normal_variance_ratio') or .25
        if state['updates'] != 298000 or state.get('normal_variance_ratio', rho) != rho:
            raise ValueError('Use the native fixed300k policy and its own covariance settings')
        actor = ArmFlowActor(25, variance=config['reference_variance'], rho=rho).to(args.device)
        actor.load_state_dict(state['actor'])
        margin = None
        if config['method'] == 'flow_learned':
            margin = LocalMarginEnsemble().to(args.device)
            margin.load_state_dict(state['margin'])
        variants = ['native_sampler', 'deterministic_endpoint_anchor_cf'] if margin is not None else ['native_sampler']
        for variant in variants:
            geometry = margin.geometry if margin is not None else None
            if variant == 'deterministic_endpoint_anchor_cf':
                with torch.no_grad():
                    anchor, _, _ = actor(obs, deterministic=True, geometry=None)
                n, q = margin.geometry(obs, anchor)
                n, q = n.repeat_interleave(128, 0), q.repeat_interleave(128, 0)
                geometry = lambda o, a, n=n, q=q: (n, q)
            # Same latent Gaussian increments for all variants and methods.
            # The deterministic auxiliary pass consumes no random draws.
            torch.manual_seed(1337)
            with torch.no_grad():
                actions, _, clips, trace = actor(batch, geometry=geometry, return_reference=True)
            actions = actions.cpu().numpy().reshape(256, 128, 3)
            normals = trace['normals'].cpu().numpy().reshape(256, 128, actor.num_steps, 3)
            vn = trace['vn'].cpu().numpy().reshape(256, 128, actor.num_steps)
            vt = trace['vt'].cpu().numpy().reshape(256, 128, actor.num_steps)
            clips = clips.cpu().numpy().reshape(256, 128)
            states = [dict(state_index=i, region=row['region'],
                common_normal_valid=bool(row['common_gradient_norm'] > 1e-7),
                endpoint=endpoint_metrics(actions[i], row['common_normal']),
                reference=reference_metrics(normals[i], vn[i], vt[i], row['common_normal']),
                clipping_rate=float(clips[i].mean()), geometry_normal_cosine_at_reference_action=None)
                for i, row in enumerate(pool)]
            name = config['method'] if variant == 'native_sampler' else 'flow_learned_anchor_cf'
            provenance = dict(method=name, sampling_variant=variant, training_seed=config['training_seed'],
                training_steps=300000, source_commit=config['source_commit'], policy_file=str(run / 'policy_300000.pt'),
                normal_variance_ratio=rho, parameter_group=f"{config['method']}-seed{config['training_seed']}")
            summary = dict(**provenance, states=256, actions_per_state=128,
                mean_clipping_rate=float(clips.mean()))
            for region in ['safe_interior', 'near_boundary']:
                selected = [r for r in states if r['region'] == region and r['common_normal_valid']]
                summary[region] = dict(valid_normal_states=len(selected),
                    reference_vn=average(selected, 'reference', 'vn'), reference_vt=average(selected, 'reference', 'vt'),
                    **{key:average(selected, 'endpoint', key) for key in ['vn', 'vt', 'gaussian_proxy_hn', 'gaussian_proxy_ht']})
            records.append(dict(**provenance, states=states))
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    payload = dict(kind='frozen_weight_geometry_location_intervention_not_trained_policy_evidence',
        analysis_source_commit=args.source_commit, requested_checkpoint_steps=300000,
        state_pool=str(args.state_pool), states=256, actions_per_state=128,
        sampling_intervention=True, training_transitions=0, actor_updates=0, new_teacher_queries=0,
        candidate_scoring_or_selection=False, auxiliary_anchor_actions_executed=0,
        paired_latent_sampling_seed=1337,
        geometry='same learned heads and q rule; only query location and holding that reference through the path change',
        energy='same inverse-covariance formula, paired to the reference actually used in each generated path',
        limitations='not retrained, no episode success/risk measured; the common true axis is at the pool source action, not newly queried at anchors; no formal method claim',
        records=records, summaries=summaries)
    (args.output / 'diagnostic.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()

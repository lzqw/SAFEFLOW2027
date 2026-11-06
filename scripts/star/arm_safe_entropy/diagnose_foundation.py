"""One interim common-state SAC/flow diagnostic; no new training or tuning loop.

Not the frozen final-method mechanism comparison. Native policy snapshots and
all undefined normals remain explicit. Run on the server: full replay is large.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from scripts.star.arm_safe_entropy.conditional_metrics import endpoint_metrics, reference_metrics
from scripts.star.arm_safe_entropy.environment import TaskConfig
from scripts.star.arm_safe_entropy.models import ArmLearner
from scripts.star.arm_safe_entropy.oracle import SimulatorOracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs=2, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite an earlier diagnostic')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    rng = np.random.default_rng(1337)
    torch.manual_seed(1337)
    oracle = SimulatorOracle(workers=args.workers)
    pool, policies = [], []
    expected_config = asdict(TaskConfig())
    try:
        for run in args.runs:
            config = json.loads((run / 'run_config.json').read_text())
            saved = torch.load(run / 'latest_full.pt', map_location='cpu', weights_only=False)
            if config['task_config'] != expected_config or saved['task_config'] != expected_config:
                raise ValueError('Use a matched current obstacle task, never pool v1 and v2')
            if saved['method'] not in ['sac_lag', 'flow_iso']:
                raise ValueError('This bounded diagnostic is only for the two foundation policies')
            replay = saved['replay']
            n = replay['count']
            data = replay['data']
            # Preferential candidate coverage near recorded risky transitions;
            # final stratification uses TRUE current-state distances, not labels.
            candidates = np.unique(np.concatenate([rng.choice(n, min(1024, n), replace=False),
                np.argsort(np.abs(data['margin'][:n]))[:256]]))
            states = data['obs'][candidates]
            clearance = oracle.state_distances_numpy(states) - expected_config['safety_distance']
            for region, mask in [('near_boundary', (clearance >= 0) & (clearance <= .02)),
                                 ('safe_interior', clearance > .02)]:
                available = np.flatnonzero(mask)
                if len(available) < 64:
                    raise RuntimeError(f'{run.name}: only {len(available)} genuine {region} states; do not fabricate balance')
                for index in rng.choice(available, 64, replace=False):
                    native_index = int(candidates[index])
                    pool.append(dict(observation=states[index].tolist(),
                        reference_action=data['action'][native_index].tolist(),
                        state_clearance_m=float(clearance[index]), region=region,
                        source_run=str(run), source_replay_index=native_index,
                        source_training_steps=saved['steps']))
            learner = ArmLearner(25, saved['method'], args.device)
            learner.load_state_dict(saved['learner'])
            policies.append((learner, dict(run=str(run), method=saved['method'],
                source_commit=saved['source_commit'], training_steps=saved['steps'],
                training_seed=config['training_seed'])))
            del saved, replay, data
        if {p[1]['method'] for p in policies} != {'sac_lag', 'flow_iso'}:
            raise ValueError('Compare one SAC-Lag and one Flow-Iso, not duplicated policies')
        observations = np.asarray([r['observation'] for r in pool], dtype=np.float32)
        reference_actions = np.asarray([r['reference_action'] for r in pool], dtype=np.float32)
        truth = oracle.evaluate_numpy(observations, reference_actions)
        for row, value in zip(pool, truth):
            row.update(common_normal=value[1:4].tolist(), common_gradient_norm=float(value[4]),
                       reference_interval_margin_m=float(value[0]))
        (args.output / 'state_pool.json').write_text(json.dumps(pool, indent=2) + '\n')
        all_records, summaries = [], []
        for learner, provenance in policies:
            per_state = []
            # Independent action samples for each fixed state; no pooled entropy.
            for index, (observation, shared) in enumerate(zip(observations, pool)):
                batch = torch.as_tensor(np.repeat(observation[None], 128, axis=0), device=learner.device)
                with torch.no_grad():
                    if learner.flow:
                        actions, energy, clip, trace = learner.actor(batch, geometry=learner.geometry,
                                                                  return_reference=True)
                        reference = reference_metrics(trace['normals'].cpu().numpy(),
                            trace['vn'].cpu().numpy(), trace['vt'].cpu().numpy(), shared['common_normal'])
                        clipping = float(clip.mean())
                    else:
                        actions, _ = learner.actor(batch)
                        reference, clipping = None, 0.  # smooth tanh, no hard clipping
                actions = actions.cpu().numpy()
                margins = oracle.margins_numpy(np.repeat(observation[None], 128, axis=0), actions)
                per_state.append(dict(state_index=index, region=shared['region'],
                    common_normal_valid=bool(shared['common_gradient_norm'] > 1e-7),
                    endpoint=endpoint_metrics(actions, shared['common_normal']), reference=reference,
                    clipping_rate=clipping, action_bound_saturation_rate=float((np.abs(actions) > .999).mean()),
                    mean_interval_margin_m=float(margins.mean()),
                    local_risk_increment_m=float(margins.mean() - shared['reference_interval_margin_m']),
                    one_interval_violation_probability=float((margins > 0).mean())))
            summary = dict(**provenance, states=len(pool), actions_per_state=128,
                common_normal_valid_states=sum(r['common_normal_valid'] for r in per_state),
                mean_conditional_action_trace=float(np.mean([r['endpoint']['trace'] for r in per_state])),
                mean_clipping_rate=float(np.mean([r['clipping_rate'] for r in per_state])),
                mean_action_bound_saturation_rate=float(np.mean([r['action_bound_saturation_rate'] for r in per_state])),
                one_interval_violation_probability=float(np.mean([r['one_interval_violation_probability'] for r in per_state])))
            for region in ['safe_interior', 'near_boundary']:
                valid = [r for r in per_state if r['region'] == region and r['common_normal_valid']]
                summary[region] = dict(states=sum(r['region'] == region for r in per_state),
                    valid_normal_states=len(valid),
                    **{key: float(np.mean([r['endpoint'][key] for r in valid])) if valid else None
                       for key in ['vn', 'vt', 'gaussian_proxy_hn', 'gaussian_proxy_ht']})
            all_records.append(dict(**provenance, states=per_state))
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
        payload = dict(kind='interim_foundation_conditional_sampling_not_final_mechanism_result',
            source_commit=args.source_commit, task_config=expected_config, training_transitions=0,
            state_pool='state_pool.json', states=256, actions_per_state=128,
            common_normal='exact interval-margin gradient at each source replay action, shared across methods',
            state_regions='128 states with current static clearance0..20mm above safety threshold;128 above20mm',
            entropy_definition='per-state Gaussian marginal covariance proxies; H_T is two-dimensional; H_N+H_T not claimed joint entropy',
            risk_definition='actual next-interval margin minus common replay-action margin, not episode risk',
            limitations='interim different training ages; not an efficacy comparison or a completed directional experiment',
            summaries=summaries, records=all_records, oracle_statistics=oracle.statistics())
        (args.output / 'diagnostic.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    finally:
        oracle.close()


if __name__ == '__main__':
    main()

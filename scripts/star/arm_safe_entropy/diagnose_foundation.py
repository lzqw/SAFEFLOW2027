"""Common-state conditional sampling; no new training or tuning loop.

Default: interim foundation/directional diagnosis. --final-mechanism requires
the completed two-seed directional/control cohort, not selected checkpoints.
Native snapshots and all undefined normals remain explicit. Run on the server.
"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from scripts.star.arm_safe_entropy.conditional_metrics import endpoint_metrics, reference_metrics
from scripts.star.arm_safe_entropy.environment import TaskConfig
from scripts.star.arm_safe_entropy.models import ArmLearner
from scripts.star.arm_safe_entropy.oracle import SimulatorOracle, ControllerJacobianOracle


def mean_defined(rows, group, key):
    values = [r[group][key] for r in rows if r.get(group) is not None and r[group][key] is not None]
    return float(np.mean(values)) if values else None


def write_mechanism_csv(path, records, pool_file, analysis_commit):
    columns = ['task_version', 'method', 'training_seed', 'states', 'actions_per_state', 'state_region',
        'reference_vn', 'reference_vt', 'action_vn', 'action_vt', 'gaussian_proxy_hn', 'gaussian_proxy_ht',
        'clipping_rate', 'local_risk_increment', 'normal_cosine', 'source_commit', 'status',
        'valid_common_normal_states', 'valid_geometry_normal_comparisons', 'checkpoint_steps',
        'checkpoint_safe_success', 'checkpoint_violation_rate', 'training_violation_count',
        'reference_eigenvalue_n_mean', 'reference_eigenvalue_t_mean',
        'oracle_backend', 'state_pool', 'analysis_commit']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for run in records:
            for region in ['all', 'safe_interior', 'near_boundary']:
                rows = [r for r in run['states'] if region == 'all' or r['region'] == region]
                valid = [r for r in rows if r['common_normal_valid']]
                comparisons = [r['geometry_normal_cosine_at_reference_action'] for r in rows
                               if r['geometry_normal_cosine_at_reference_action'] is not None]
                evaluation = run['final_evaluation']
                writer.writerow(dict(task_version=TaskConfig().version, method=run['method'],
                    training_seed=run['training_seed'], states=len(rows), actions_per_state=128,
                    state_region=region, reference_vn=mean_defined(valid, 'reference', 'vn'),
                    reference_vt=mean_defined(valid, 'reference', 'vt'),
                    action_vn=mean_defined(valid, 'endpoint', 'vn'), action_vt=mean_defined(valid, 'endpoint', 'vt'),
                    gaussian_proxy_hn=mean_defined(valid, 'endpoint', 'gaussian_proxy_hn'),
                    gaussian_proxy_ht=mean_defined(valid, 'endpoint', 'gaussian_proxy_ht'),
                    clipping_rate=float(np.mean([r['clipping_rate'] for r in rows])),
                    local_risk_increment=float(np.mean([r['local_risk_increment_m'] for r in rows])),
                    normal_cosine=float(np.mean(comparisons)) if comparisons else None,
                    source_commit=run['source_commit'], status='final_checkpoint_conditional_mechanism',
                    valid_common_normal_states=len(valid), valid_geometry_normal_comparisons=len(comparisons),
                    checkpoint_steps=run['training_steps'], checkpoint_safe_success=evaluation['safe_success'],
                    checkpoint_violation_rate=evaluation['violation'],
                    training_violation_count=evaluation['training_violation_count'],
                    **{key: float(np.mean([r[key] for r in rows if r[key] is not None]))
                       if any(r[key] is not None for r in rows) else None
                       for key in ['reference_eigenvalue_n_mean', 'reference_eigenvalue_t_mean']},
                    oracle_backend=run['oracle_backend'], state_pool=pool_file, analysis_commit=analysis_commit))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--final-mechanism', action='store_true')
    parser.add_argument('--checkpoint-steps', type=int,
                        help='Compare an explicit common milestone using policy_N.pt and its replay prefix, never the best checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    configs = [json.loads((run / 'run_config.json').read_text()) for run in args.runs]
    pairs = [(c['method'], c['training_seed']) for c in configs]
    if not 2 <= len(configs) <= 128 or len(set(pairs)) != len(pairs):
        parser.error('Use at least two distinct method/seed runs; never duplicate a selected checkpoint')
    if args.final_mechanism:
        if args.checkpoint_steps not in [None, 300000]:
            parser.error('Final mechanism mode cannot select an earlier checkpoint')
        required = {'flow_iso', 'flow_oracle', 'flow_learned', 'flow_iso_low', 'flow_random_aniso'}
        if not required <= {c['method'] for c in configs}:
            parser.error('Final mechanism mode includes the directional methods AND both required controls')
        for method in {c['method'] for c in configs}:
            if {c['training_seed'] for c in configs if c['method'] == method} != {2027, 2028}:
                parser.error(f'{method}: retain BOTH development seeds, not only the stronger seed')
        for run, config in zip(args.runs, configs):
            receipt = json.loads((run / 'receipt.json').read_text())
            if receipt['status'] != 'completed' or receipt['steps'] != 300000 or config['total_transition_budget'] != 300000:
                parser.error('Final mechanism comparison uses each fixed final300k checkpoint')
    if args.output.exists():
        raise FileExistsError('Never overwrite an earlier diagnostic')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    rng = np.random.default_rng(1337)
    torch.manual_seed(1337)
    oracle = SimulatorOracle(workers=args.workers)
    pool, policies, behavior_oracles = [], [], []
    seen_states = set()
    expected_config = asdict(TaskConfig())
    try:
        for run_index, (run, config) in enumerate(zip(args.runs, configs)):
            saved = torch.load(run / 'latest_full.pt', map_location='cpu', weights_only=False)
            if config['task_config'] != expected_config or saved['task_config'] != expected_config:
                raise ValueError('Use a matched current obstacle task, never pool v1 and v2')
            if args.final_mechanism and saved['steps'] != 300000:
                raise ValueError('Full replay and learner must be from the fixed final checkpoint')
            quota = 128 // len(args.runs) + int(run_index < 128 % len(args.runs))
            replay = saved['replay']
            n = replay['count']
            policy_steps = saved['steps']
            policy_state = saved['learner']
            if args.checkpoint_steps is not None:
                if args.checkpoint_steps <= 0 or saved['steps'] < args.checkpoint_steps:
                    raise ValueError('The explicit milestone has not yet been reached')
                if saved['steps'] > replay['capacity']:
                    raise ValueError('The requested early replay prefix has been overwritten; use an earlier native full snapshot')
                policy_steps = args.checkpoint_steps
                policy_state = torch.load(run / f'policy_{policy_steps}.pt', map_location='cpu', weights_only=False)
                if policy_state['updates'] != max(0, policy_steps - 2000):
                    raise ValueError('Policy update counters do not match the requested fixed milestone')
                n = min(n, policy_steps)
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
                unique_available, locally_seen = [], set(seen_states)
                for i in available:
                    state_key = states[i].tobytes()
                    if state_key not in locally_seen:
                        unique_available.append(i)
                        locally_seen.add(state_key)
                available = np.asarray(unique_available, dtype=int)
                if len(available) < quota:
                    raise RuntimeError(f'{run.name}: only {len(available)} genuine {region} states; do not fabricate balance')
                for index in rng.choice(available, quota, replace=False):
                    seen_states.add(states[index].tobytes())
                    native_index = int(candidates[index])
                    pool.append(dict(observation=states[index].tolist(),
                        reference_action=data['action'][native_index].tolist(),
                        state_clearance_m=float(clearance[index]), region=region,
                        source_run=str(run), source_replay_index=native_index,
                        source_method=config['method'], source_training_seed=config['training_seed'],
                        source_training_steps=policy_steps, replay_prefix_transitions=n,
                        source_full_snapshot_steps=saved['steps']))
            behavior_oracle = None
            if saved['method'] == 'flow_oracle':
                backend = config.get('oracle_backend', 'exact')
                behavior_oracle = (ControllerJacobianOracle(workers=args.workers) if backend == 'controller_jacobian'
                                   else SimulatorOracle(workers=args.workers))
                behavior_oracles.append(behavior_oracle)
            learner = ArmLearner(25, saved['method'], args.device,
                                 variance=config.get('reference_variance') or .04, oracle_geometry=behavior_oracle)
            learner.load_state_dict(policy_state)
            policies.append((learner, dict(run=str(run), method=saved['method'],
                source_commit=saved['source_commit'], training_steps=policy_steps,
                source_full_snapshot_steps=saved['steps'], replay_prefix_transitions=n,
                checkpoint_selection='explicit_common_milestone' if args.checkpoint_steps else 'native_latest_at_snapshot',
                training_seed=config['training_seed'], oracle_backend=config.get('oracle_backend'),
                reference_variance=config.get('reference_variance'),
                reference_match_run=config.get('reference_match_run'),
                native_evaluation_at_policy_step=json.loads((run / f'evaluation_{policy_steps}.json').read_text())
                    if (run / f'evaluation_{policy_steps}.json').exists() else None,
                final_evaluation=json.loads((run / 'evaluation_300000.json').read_text()) if args.final_mechanism else None)))
            del saved, replay, data
        assert len(pool) == 256
        assert len(seen_states) == 256, 'The shared pool must have distinct recorded states'
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
            if args.final_mechanism:
                torch.manual_seed(1337)
            geometry_cosines = [None] * len(pool)
            geometry_diagnostics = None
            shared_obs = torch.as_tensor(observations, device=learner.device)
            shared_actions = torch.as_tensor(reference_actions, device=learner.device)
            if learner.geometry is not None:
                if hasattr(learner, 'margin'):
                    normal, weight, factors = learner.margin.geometry(shared_obs, shared_actions, return_diagnostics=True)
                    geometry_diagnostics = {k: v.detach().cpu().tolist() for k, v in factors.items()}
                    geometry_diagnostics['active_q_at_reference_action'] = weight.cpu().tolist()
                    with torch.no_grad():
                        head_margins = learner.margin(shared_obs, shared_actions).cpu().numpy()
                    errors = head_margins.mean(0) - truth[:, 0]
                    geometry_diagnostics.update(predicted_head_margins_m=head_margins.tolist(),
                        exact_reference_margins_m=truth[:, 0].tolist(),
                        common_pool_margin_rmse_m=float(np.sqrt(np.mean(errors**2))),
                        margin_rmse_by_region={region:float(np.sqrt(np.mean(errors[
                            np.asarray([r['region']==region for r in pool])]**2)))
                            for region in ['near_boundary', 'safe_interior']})
                else:
                    normal, weight = learner.geometry(shared_obs, shared_actions)
                    geometry_diagnostics = dict(active_q_at_reference_action=weight.cpu().tolist())
                # Random-Aniso does NOT use its heads' normal as its direction.
                if learner.method != 'flow_random_aniso':
                    predicted = normal.detach().cpu().numpy()
                    for i in range(len(pool)):
                        if truth[i, 4] > 1e-7 and np.linalg.norm(predicted[i]) > 1e-7:
                            geometry_cosines[i] = float(predicted[i] @ truth[i, 1:4])
            if hasattr(learner, 'cost_q') and (args.final_mechanism or learner.geometry is not None):
                from scripts.star.arm_safe_entropy.geometry import old_cost_weight_diagnostics
                legacy = old_cost_weight_diagnostics(learner.cost_q, shared_obs, shared_actions)
                geometry_diagnostics = dict(geometry_diagnostics or {},
                    old_long_cost_factors_diagnostic_only={k: v.detach().cpu().tolist() for k, v in legacy.items()})
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
                    geometry_normal_cosine_at_reference_action=geometry_cosines[index],
                    reference_q_mean=float(trace['weights'].mean()) if learner.flow else None,
                    reference_eigenvalue_n_mean=float(trace['vn'].mean()) if learner.flow else None,
                    reference_eigenvalue_t_mean=float(trace['vt'].mean()) if learner.flow else None,
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
                region_rows = [r for r in per_state if r['region'] == region]
                normal_cosines = [r['geometry_normal_cosine_at_reference_action'] for r in region_rows
                                  if r['geometry_normal_cosine_at_reference_action'] is not None]
                summary[region] = dict(states=sum(r['region'] == region for r in per_state),
                    valid_normal_states=len(valid),
                    geometry_normal_comparisons=len(normal_cosines),
                    geometry_normal_cosine=float(np.mean(normal_cosines)) if normal_cosines else None,
                    reference_vn=mean_defined(valid, 'reference', 'vn'),
                    reference_vt=mean_defined(valid, 'reference', 'vt'),
                    **{key: float(np.mean([r['endpoint'][key] for r in valid])) if valid else None
                       for key in ['vn', 'vt', 'gaussian_proxy_hn', 'gaussian_proxy_ht']})
            if geometry_diagnostics and 'common_pool_margin_rmse_m' in geometry_diagnostics:
                summary['common_pool_margin_rmse_m'] = geometry_diagnostics['common_pool_margin_rmse_m']
            all_records.append(dict(**provenance, states=per_state, geometry_diagnostics=geometry_diagnostics))
            summaries.append(summary)
            print(json.dumps({k:v for k,v in summary.items()
                              if k not in ['native_evaluation_at_policy_step', 'final_evaluation']}), flush=True)
        if args.final_mechanism:
            write_mechanism_csv(args.output / 'mechanism_results.csv', all_records, 'state_pool.json', args.source_commit)
        payload = dict(kind='final_checkpoint_conditional_mechanism' if args.final_mechanism else
                           'interim_conditional_sampling_not_final_mechanism_result',
            source_commit=args.source_commit, task_config=expected_config, training_transitions=0,
            requested_checkpoint_steps=args.checkpoint_steps,
            state_pool='state_pool.json', states=256, actions_per_state=128,
            common_normal='exact interval-margin gradient at each source replay action, shared across methods',
            state_regions='128 states with current static clearance0..20mm above safety threshold;128 above20mm',
            entropy_definition='per-state Gaussian marginal covariance proxies; H_T is two-dimensional; H_N+H_T not claimed joint entropy',
            risk_definition='actual next-interval margin minus common replay-action margin, not episode risk',
            limitations=('Covariance/entropy proxies are conditional sample estimates, not guarantees. Normal cosine uses identical reference s,a; random control has no aligned normal. Oracle backend errors remain explicit.'
                         if args.final_mechanism else
                         'interim matched policy milestone and replay prefix; full snapshot ages may differ; not a completed directional experiment'
                         if args.checkpoint_steps is not None else
                         'interim different training ages; not an efficacy comparison or a completed directional experiment'),
            summaries=summaries, records=all_records, oracle_statistics=oracle.statistics())
        (args.output / 'diagnostic.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    finally:
        oracle.close()
        for backend in behavior_oracles:
            backend.close()


if __name__ == '__main__':
    main()

"""One fixed-budget offline ReLU/SiLU comparison; no actor or env training.

Train labels are actual replay interval margins. Exact midpoint teacher data
are evaluation-only. Both activations receive identical initialization, draws
and bootstrap masks, and all exact probe states are excluded from fitting.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.star.arm_safe_entropy.geometry import LocalMarginEnsemble


def evaluate(model, obs, action, truth):
    normal, weight, factors = model.geometry(obs, action, return_diagnostics=True)
    with torch.no_grad():
        predicted = model(obs, action).mean(0).cpu().numpy()
    normal, weight = normal.cpu().numpy(), weight.cpu().numpy()
    true_valid = np.asarray([r['exact_gradient_norm'] > 1e-7 for r in truth])
    true_normal = np.asarray([r['exact_normal'] for r in truth])
    valid = true_valid & (np.linalg.norm(normal, axis=1) > 1e-7)
    cosine = (normal * true_normal).sum(-1)
    return dict(points=len(truth), true_valid=int(true_valid.sum()), valid_comparisons=int(valid.sum()),
        signed_cosine=float(cosine[valid].mean()) if valid.any() else None,
        axis_cosine_squared=float((cosine[valid]**2).mean()) if valid.any() else None,
        margin_rmse_m=float(np.sqrt(np.mean((predicted - np.asarray([r['exact_margin_m'] for r in truth]))**2))),
        mean_q_true_valid=float(weight[true_valid].mean()) if true_valid.any() else None,
        mean_q_true_flat=float(weight[~true_valid].mean()) if (~true_valid).any() else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs=2, type=Path, required=True)
    parser.add_argument('--path-diagnostic', type=Path, required=True)
    parser.add_argument('--state-pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Never overwrite an offline comparison')
    configs = [json.loads((run / 'run_config.json').read_text()) for run in args.runs]
    if {c['training_seed'] for c in configs} != {2027, 2028} or any(c['method'] != 'flow_learned' for c in configs):
        parser.error('Retain both Learned development seeds')
    probe = json.loads(args.path_diagnostic.read_text())
    pool = json.loads(args.state_pool.read_text())
    excluded_states = {np.asarray(pool[i]['observation'], np.float32).tobytes()
                       for i in probe['selected_pool_indices']}
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    results = []
    started = time.monotonic()
    for run, config in zip(args.runs, configs):
        seed = config['training_seed']
        original = next(r for r in probe['records'] if r['summary']['training_seed'] == seed)
        saved = torch.load(run / 'latest_full.pt', map_location='cpu', weights_only=False)
        if saved['steps'] != 300000:
            raise ValueError('Offline comparison is based on fixed final300k replay')
        data, n = saved['replay']['data'], saved['replay']['count']
        eligible = np.asarray([i for i in range(n) if data['obs'][i].tobytes() not in excluded_states])
        tensors = {key:torch.as_tensor(data[key][:n], device=args.device) for key in ['obs', 'action', 'margin']}
        validation_obs = torch.as_tensor(np.asarray([pool[p['source_pool_index']]['observation']
            for p in original['points']], np.float32), device=args.device)
        validation_action = torch.as_tensor(np.asarray([p['midpoint_action'] for p in original['points']],
            np.float32), device=args.device)
        torch.manual_seed(seed)
        students = {name:LocalMarginEnsemble(activation=name).to(args.device) for name in ['relu', 'silu']}
        students['silu'].load_state_dict(students['relu'].state_dict())
        optimizers = {name:torch.optim.Adam(model.parameters(), lr=.0003) for name, model in students.items()}
        rng = np.random.default_rng(902027 + seed)
        # One predeclared comparison,5000 updates/model; never select an epoch
        # or modify targets using teacher validation results.
        for step in range(5000):
            indices = torch.as_tensor(rng.choice(eligible, 256), device=args.device)
            obs, action, labels = [tensors[key][indices] for key in ['obs', 'action', 'margin']]
            mask = torch.as_tensor((rng.random((2, 256)) < .8).astype(np.float32), device=args.device)
            for name, model in students.items():
                predicted = model(obs, action)
                loss = ((predicted - labels).square() * mask).sum() / mask.sum().clamp_min(1.)
                optimizers[name].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizers[name].step()
        for name, model in students.items():
            metrics = evaluate(model, validation_obs, validation_action, original['points'])
            result = dict(training_seed=seed, activation=name, fitting_updates=5000,
                eligible_replay_transitions=len(eligible), excluded_exact_state_rows=n-len(eligible),
                **metrics)
            results.append(result)
            # Offline auxiliary model only: not a policy checkpoint and never
            # installed in a native run or used to overwrite existing results.
            torch.save(dict(activation=name, margin=model.state_dict(), training_seed=seed,
                fitting_updates=5000, source_commit=args.source_commit),
                args.output / f'offline_margin_seed{seed}_{name}.pt')
            print(json.dumps(result), flush=True)
    payload = dict(kind='single_factor_offline_margin_activation_probe_not_policy_training',
        source_commit=args.source_commit, path_diagnostic=str(args.path_diagnostic),
        starting_models='fresh identical parameters per activation within each seed, not native actor modification',
        replay_source='actual collected interval-margin labels only; uniform offline sampling',
        teacher_use='evaluation only, never a fitting target', state_exclusion='all16 exact probe observations excluded; nearby/layout correlation remains',
        width=64, heads=2, bootstrap_probability=.8, lr=.0003, batch_size=256,
        fitting_updates_per_model=5000, training_transitions=0, actor_updates=0,
        elapsed_seconds=time.monotonic()-started, torch=str(torch.__version__), device=args.device,
        limitations='fixed small offline budget; no claim of online learning improvement or disjoint-layout generalization; no best-epoch selection',
        results=results)
    (args.output / 'comparison.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()

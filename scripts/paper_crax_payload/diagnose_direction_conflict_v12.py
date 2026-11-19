"""Read-only saved-replay actor-gradient diagnosis; no optimizer step or physics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from scripts.paper_crax_payload.train_matched_ablation_v12 import FORMAT, PLAN, build
from scripts.paper_crax_payload.direction_development import outward_loss
from scripts.paper_crax_payload.summarize_simulation_v12 import output
from scripts.star.arm_safe_entropy.endpoint_geometry import frozen_parameters


ROOT = Path('/storage_0/SAFEFLOW2027_lzqw_uid1007/simulation_v12')
ARMS = ('A-full-P5', 'A-no-direction')
STEPS = (100000, 250000)


def gradient(loss, parameters):
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(parameters, values)])


def comparison(left, right):
    left_norm = left.norm(); right_norm = right.norm()
    dot = torch.dot(left, right)
    return dict(left_norm=float(left_norm), right_norm=float(right_norm), dot=float(dot),
        cosine=float(dot/(left_norm*right_norm)) if left_norm > 1e-12 and right_norm > 1e-12 else None,
        left_loss_slope_under_negative_right_gradient=float(-dot))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--source-commit', required=True)
    p.add_argument('--direction-taper', action='store_true', help='Offline loss-only candidate;never change checkpoint or action')
    args = p.parse_args(); torch.set_num_threads(1)
    if not args.output.resolve().is_relative_to(ROOT.resolve(strict=True)):
        raise ValueError('Keep output under the admitted A100 data root')
    if shutil.disk_usage(ROOT).free < 41*1024**3:
        raise OSError('Require40GiB reserve plus1GiB diagnostic growth allowance')
    config = json.loads(PLAN.read_text()); args.output.mkdir(exist_ok=False)
    registration = dict(protocol='v12_saved_replay_gradient_conflict', source_commit=args.source_commit,
        direction_taper_diagnostic=args.direction_taper,
        arms=ARMS, checkpoints=STEPS, batch_size=256, batches_per_checkpoint=4,
        sampling='half uniform valid replay;half recent10K;fixed diagnostic RNG991700+batch',
        risk_groups='generated action EMA margin:near_or_unsafe>=-2mm;interior[-10,-2)mm;deep_safe<-10mm',
        actor='saved online actor;one stochastic trajectory per sampled state',
        new_training_steps=0, new_simulation_controls=0, optimizer_steps=0, independent_test_used=False,
        checkpoint_mutations=0, automatic_retry=False,
        interpretation='Gradient alignment on saved visited-state distribution;not true risk, causal proof, or inference policy')
    (args.output/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    rows = []; sources = []
    for arm in ARMS:
        for step in STEPS:
            path = ROOT/'runs-matched-ablation'/(arm+'-2034')/f'full_{step}.pt'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            saved = torch.load(path, map_location='cpu', weights_only=False)
            spec = next(a for a in config['arms'] if a['name'] == arm)
            if (saved['format'] != FORMAT or saved['configuration'] != config or saved['arm'] != spec
                    or saved['seed'] != 2034 or saved['steps'] != step or saved['source_commit'] != '2f09231b'):
                raise ValueError('Use the four registered original A100 native checkpoints')
            learner = build(config, spec, 2034, 'cpu'); learner.load_state_dict(saved['learner'])
            if args.direction_taper:
                from scripts.paper_crax_payload.direction_taper_v12 import DirectionTaperActor
                learner.actor.__class__ = DirectionTaperActor
            replay = saved['replay']; parameters = tuple(learner.actor.parameters())
            for batch in range(4):
                rng = np.random.default_rng(991700+batch)
                indices = rng.integers(replay['count'], size=256)
                indices[:128] = (replay['cursor']-1-rng.integers(min(replay['count'],10000), size=128)) % replay['capacity']
                obs = torch.as_tensor(replay['data']['obs'][indices])
                torch.manual_seed(991800+batch)
                with frozen_parameters(learner.q, learner.cost_q):
                    action, energy, clips, trace = learner.actor(obs, geometry=learner.geometry, return_reference=True)
                    task = -learner.q(obs, action).min(0).values.mean()
                    cost = learner.safety_multiplier*learner.cost_q(obs, action).mean()
                    entropy = learner.log_alpha.detach().exp()*energy.mean()
                    direction_weights = trace['direction_weights'] if args.direction_taper else trace['weights']
                    direction = outward_loss(trace['velocities'], trace['normals'], direction_weights, .25)
                    task_grad = gradient(task, parameters); dir_grad = gradient(direction, parameters)
                    base_grad = gradient(task+cost+entropy, parameters)
                    with torch.no_grad():
                        margin = learner.margin_geometry(obs, action).mean(0)
                        bins = {'near_or_unsafe': margin >= -.002,
                                'interior': (margin < -.002) & (margin >= -.01), 'deep_safe': margin < -.01}
                    speed_scale = (.25*trace['velocities'].square().sum(-1).sum(-1).mean()).detach()+1e-6
                    outward = (trace['velocities']*trace['normals']).sum(-1).clamp_min(0)
                    per_state = .25*(direction_weights*outward.square()).sum(-1)/speed_scale
                    groups = {}
                    for name, mask in bins.items():
                        contribution = (per_state*mask).mean()
                        group_grad = gradient(contribution, parameters)
                        groups[name] = dict(samples=int(mask.sum()), loss_contribution=float(contribution.detach()),
                            **comparison(task_grad, group_grad))
                    rows.append(dict(arm=arm, transitions=step, batch=batch, task_loss=float(task.detach()),
                        direction_loss=float(direction.detach()), active_direction_coefficient=spec['direction_weight'],
                        safety_multiplier=learner.safety_multiplier, alpha=float(learner.log_alpha.detach().exp()),
                        task_vs_direction=comparison(task_grad, dir_grad),
                        task_cost_entropy_vs_direction=comparison(base_grad, dir_grad), groups=groups,
                        mean_geometry_weight=float(trace['weights'].mean()),
                        mean_direction_loss_weight=float(direction_weights.mean()),
                        high20_weight_mean=float(trace['weights'].mean(-1).topk(52).values.mean()),
                        sampled_action_clipped_component_fraction=float((action.detach().abs()>=.99999).float().mean()),
                        per_state_loss_sum_matches=float((per_state.mean()-direction).detach().abs())))
            sources.append(dict(arm=arm, transitions=step, relative_path=str(path.relative_to(ROOT)), sha256=digest))
            del saved, learner, replay
    # Nested gradients are JSON: keep the exact per-batch quantities without a misleading pooled cosine.
    (args.output/'batches.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    summary = []
    for arm in ARMS:
        for step in STEPS:
            subset = [r for r in rows if r['arm'] == arm and r['transitions'] == step]
            total = sum(r['direction_loss'] for r in subset)
            summary.append(dict(arm=arm, transitions=step, batches=len(subset),
                task_direction_conflict_batches=sum(r['task_vs_direction']['dot']<0 for r in subset),
                base_direction_conflict_batches=sum(r['task_cost_entropy_vs_direction']['dot']<0 for r in subset),
                mean_task_direction_cosine=float(np.mean([r['task_vs_direction']['cosine'] for r in subset])),
                deep_safe_direction_loss_fraction=sum(r['groups']['deep_safe']['loss_contribution'] for r in subset)/total if total else None,
                deep_safe_sample_fraction=sum(r['groups']['deep_safe']['samples'] for r in subset)/1024,
                near_unsafe_sample_fraction=sum(r['groups']['near_or_unsafe']['samples'] for r in subset)/1024))
    output(args.output/'summary.csv', summary)
    (args.output/'receipt.json').write_text(json.dumps(dict(status='completed', sources=sources,
        diagnostic_batches=len(rows), sampled_states=256*len(rows), new_training_steps=0,
        new_simulation_controls=0, optimizer_steps=0, independent_test_used=False,
        counterfactual_policy_used=False, model_risk_only=True), indent=2)+'\n')


if __name__ == '__main__': main()

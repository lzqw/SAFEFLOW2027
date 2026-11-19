"""Same-action direction-loss probe on four pinned native Piper saved replays."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.paper_crax_payload.diagnose_direction_conflict_v12 import gradient, comparison
from scripts.paper_crax_payload.diagnose_piper_noise_v12 import PLAN, load
from scripts.paper_crax_payload.direction_development import outward_loss
from scripts.paper_crax_payload.direction_taper_v12 import DirectionTaperActor
from scripts.paper_crax_payload.summarize_simulation_v12 import output
from scripts.paper_crax_payload.train_native_p5_adaptation import sha256, storage_check
from scripts.star.arm_safe_entropy.endpoint_geometry import frozen_parameters


ROOT = Path('/home/carla/4T/SAFEFLOW2027_SIMULATION_V12')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--source-commit', required=True)
    args = p.parse_args(); torch.set_num_threads(1)
    if not args.output.resolve().is_relative_to(ROOT.resolve(strict=True)):
        raise ValueError('Keep output on4080 data disk')
    storage_check(ROOT, growth_gib=1)
    spec = json.loads(PLAN.read_text())
    args.output.mkdir(exist_ok=False)
    (args.output/'registration.json').write_text(json.dumps(dict(
        protocol='v12_native_piper_saved_replay_direction_conflict', source_commit=args.source_commit,
        source_cases=spec['cases'], selection='all_four_previously_pinned_native100K_endpoints',
        batches_per_case=4, batch_size=256, replay_seed=992700, action_seed=992800,
        sampling='half uniform valid replay;half recent10K',
        risk_group='per_flow_injection_point_original_EMA_margin_including_registered_activation_offset',
        CPU_only=True, new_training_steps=0,new_simulation_controls=0,independent_test_used=False,
        hypothesis='Check native Piper task gradient interference before assuming SafeReach mechanism transfers'),indent=2)+'\n')
    rows = []; summaries = []; sources = []
    for case in spec['cases']:
        learner, native, line = load(case)  # Existing strict native format/plan/arm/hash guards.
        saved = torch.load(case['checkpoint'],map_location='cpu',weights_only=False)
        replay = saved['replay']; del saved
        if replay['data']['obs'].shape[-1] != 34:
            raise ValueError('Require native Piper34 replay;no robot weight transfer')
        learner.actor.__class__ = DirectionTaperActor
        parameters = tuple(learner.actor.parameters())
        for batch in range(4):
            rng = np.random.default_rng(992700+batch)
            indices = rng.integers(replay['count'],size=256)
            indices[:128] = (replay['cursor']-1-rng.integers(min(replay['count'],10000),size=128)) % replay['capacity']
            obs = torch.as_tensor(replay['data']['obs'][indices]); torch.manual_seed(992800+batch)
            with frozen_parameters(learner.q,learner.cost_q):
                action,energy,clips,trace = learner.actor(obs,geometry=learner.geometry,return_reference=True)
                task = -learner.q(obs,action).min(0).values.mean()
                base = task+learner.safety_multiplier*learner.cost_q(obs,action).mean()+learner.log_alpha.detach().exp()*energy.mean()
                dt = 1./learner.actor.num_steps
                original = outward_loss(trace['velocities'],trace['normals'],trace['weights'],dt)
                tapered = outward_loss(trace['velocities'],trace['normals'],trace['direction_weights'],dt)
                task_grad = gradient(task,parameters); base_grad = gradient(base,parameters)
                old_grad = gradient(original,parameters); new_grad = gradient(tapered,parameters)
                with torch.no_grad():
                    contributions = trace['weights']*(trace['velocities']*trace['normals']).sum(-1).clamp_min(0).square()
                    deep = trace['direction_margin_m'] <= -.01
                    total = contributions.sum()
                    deep_fraction = float(contributions[deep].sum()/total) if total>0 else None
                rows.append(dict(case=case['name'],line=case['line'],batch=batch,
                    original_loss=float(original.detach()),tapered_loss=float(tapered.detach()),
                    task_vs_original=comparison(task_grad,old_grad),task_vs_taper=comparison(task_grad,new_grad),
                    base_vs_original=comparison(base_grad,old_grad),base_vs_taper=comparison(base_grad,new_grad),
                    deep_safe_flow_point_fraction=float(deep.float().mean()),
                    deep_safe_original_loss_fraction=deep_fraction,
                    mean_geometry_weight=float(trace['weights'].mean()),mean_direction_weight=float(trace['direction_weights'].mean()),
                    risk_group_is_model_estimate_not_true_margin=True))
        subset = [r for r in rows if r['case']==case['name']]
        summaries.append(dict(case=case['name'],line=case['line'],batches=4,
            task_original_conflict_batches=sum(r['task_vs_original']['dot']<0 for r in subset),
            base_original_conflict_batches=sum(r['base_vs_original']['dot']<0 for r in subset),
            mean_original_loss=float(np.mean([r['original_loss'] for r in subset])),
            mean_tapered_loss=float(np.mean([r['tapered_loss'] for r in subset])),
            mean_deep_safe_flow_point_fraction=float(np.mean([r['deep_safe_flow_point_fraction'] for r in subset])),
            mean_deep_safe_original_loss_fraction=float(np.mean([r['deep_safe_original_loss_fraction'] for r in subset])),
            original_task_adverse_slope=float(np.mean([r['task_vs_original']['left_loss_slope_under_negative_right_gradient'] for r in subset])),
            tapered_task_adverse_slope=float(np.mean([r['task_vs_taper']['left_loss_slope_under_negative_right_gradient'] for r in subset]))))
        sources.append(dict(case=case['name'],checkpoint_sha256=case['checkpoint_sha256'],
            receipt_sha256=sha256(case['receipt']),native_observation_dim=34,training_plan_sha256=sha256(case['plan'])))
        del learner,replay
    (args.output/'batches.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    output(args.output/'summary.csv',summaries)
    (args.output/'receipt.json').write_text(json.dumps(dict(status='completed',sources=sources,
        sampled_saved_states=4096,diagnostic_batches=16,new_training_steps=0,new_simulation_controls=0,
        optimizer_steps=0,checkpoint_mutations=0,physical_robot_interactions=0,independent_test_used=False,
        winner_confirmation=False,scope='Native training replay;not unseen57.5/60mm trajectory or causal proof'),indent=2)+'\n')


if __name__ == '__main__': main()

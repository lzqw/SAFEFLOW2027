"""Small retrospective diagnosis of the completed frozen cohort; never train.

600k weights are probed on explicitly labelled common final-replay batches,
NOT paired with that replay as a native600k resume. Cross evaluations are
post-hoc actor/margin interventions, never replacement primary scores.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.star.arm_safe_entropy.models import ArmLearner
from scripts.star.arm_safe_entropy.evaluate_main_robustness import physics_pool
from scripts.star.arm_safe_entropy.evaluate_obstacle_robustness import (
    atomic_json, load_policy, rollout_wave, summarize)


BASELINE = '1dc1f2945bb01ace08500147659a73623f30d977'


def native_run(project, method, seed):
    stem = {'sac_lag': 'sac-lag', 'flow_iso': 'flow-iso', 'flow_learned': 'flow-learned'}[method]
    suffix = '-v3' if seed != 2032 else ('-v3-a100-fresh' if method == 'flow_learned' else '-v3-4080-fresh')
    name = f'{stem}-seed{seed}{suffix}'
    candidates = [project/'analysis_inputs/formal_main_archive'/name, project/'runs/main'/name]
    found = [p for p in candidates if (p/'latest_full.pt').exists()]
    if not found:
        raise FileNotFoundError(f'Missing native full archive: {method},seed{seed}')
    run = found[0]
    receipt = json.loads((run/'receipt.json').read_text())
    if receipt['status'] != 'completed' or receipt['steps'] != 1000000:
        raise ValueError('Do not diagnose an incomplete/relabelled native main run')
    return run


def vector_gradient(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph,
                                    allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g,p in zip(gradients, parameters)])


def cosine(a, b):
    norms = a.norm()*b.norm()
    return float((a@b/norms).detach()) if float(norms.detach()) > 1e-12 else None


def probe(learner, batch):
    obs, replay_action = batch['obs'], batch['action']
    learner.q.requires_grad_(False)
    learner.cost_q.requires_grad_(False)
    if learner.flow:
        action, energy, clip = learner.actor(obs, geometry=learner.geometry)
        regularizer = learner.log_alpha.detach().exp()*energy
    else:
        action, logp = learner.actor(obs)
        regularizer, energy, clip = learner.log_alpha.detach().exp()*logp, None, None
    replay_q = learner.q(obs, replay_action).min(0).values
    policy_q = learner.q(obs, action).min(0).values
    cost_prediction = learner.cost_q(obs, action).mean(0)
    rloss = -policy_q.mean()
    closs = learner.safety_multiplier*cost_prediction.mean()
    eloss = regularizer.mean()
    parameters = list(learner.actor.parameters())
    gr = vector_gradient(rloss, parameters, True)
    gc = vector_gradient(closs, parameters, True)
    ge = vector_gradient(eloss, parameters, False)
    upper = 5*(obs[:,14:17]-obs[:,17:20]).norm(dim=-1).clamp_max(1.)+10
    output = dict(reward_gradient_norm=float(gr.norm()), cost_gradient_norm=float(gc.norm()),
        energy_or_entropy_gradient_norm=float(ge.norm()), reward_cost_gradient_cosine=cosine(gr,gc),
        reward_regularizer_gradient_cosine=cosine(gr,ge), cost_regularizer_gradient_cosine=cosine(gc,ge),
        q_replay_min_mean=float(replay_q.detach().mean()), q_replay_min_min=float(replay_q.detach().min()),
        q_replay_min_max=float(replay_q.detach().max()), q_policy_min_mean=float(policy_q.detach().mean()),
        q_policy_min_min=float(policy_q.detach().min()), q_policy_min_max=float(policy_q.detach().max()),
        q_replay_above_task_upper_fraction=float((replay_q.detach()>upper+.01).float().mean()),
        q_policy_above_task_upper_fraction=float((policy_q.detach()>upper+.01).float().mean()),
        regularizer_kind='path_energy' if learner.flow else 'SAC_log_probability',
        alpha=float(learner.log_alpha.detach().exp()), safety_multiplier=learner.safety_multiplier,
        path_energy=float(energy.detach().mean()) if energy is not None else None,
        clipping_rate=float(clip.detach().mean()) if clip is not None else None,
        policy_cost_prediction_mean=float(cost_prediction.detach().mean()))
    with torch.no_grad():
        replay_cost = learner.cost_q(obs, replay_action).mean(0)
        failed = batch['cost'] > .5
        output['replay_cost_prediction_on_recorded_failures'] = float(replay_cost[failed].mean()) if failed.any() else None
        output['recorded_failure_samples'] = int(failed.sum())
        if hasattr(learner,'margin'):
            prediction = learner.margin(obs,replay_action).mean(0)
            truth = batch['margin']
            near = truth.abs() <= .02
            predicted_safe = prediction < 0
            output.update(margin_replay_mae=float((prediction-truth).abs().mean()),
                near_boundary_samples=int(near.sum()),
                margin_near_boundary_replay_mae=float((prediction[near]-truth[near]).abs().mean()) if near.any() else None,
                false_safe_joint_fraction=float(((truth>0)&predicted_safe).float().mean()),
                false_safe_given_predicted_safe=float((truth[predicted_safe]>0).float().mean()) if predicted_safe.any() else None,
                margin_scope='retrospective_final_replay_fit_NOT_episode_isolated_generalization')
    return output


def critic_probes(project, output, device, frozen):
    rows=[]
    for method in frozen['methods']:
        for seed in frozen['training_seeds']:
            run = native_run(project,method,seed)
            full = torch.load(run/'latest_full.pt',map_location='cpu',weights_only=False)
            if full['steps'] != 1000000 or full['training_seed'] != seed:
                raise ValueError('Native full/replay identity mismatch')
            replay = full['replay']
            rng = np.random.default_rng(717000+seed)
            indices = [rng.integers(replay['count'],size=128) for _ in range(2)]
            for step in (600000,1000000):
                state = torch.load(run/f'policy_{step}.pt',map_location='cpu',weights_only=False)
                learner = ArmLearner(25,method,device)
                learner.load_state_dict(state)
                for index,selection in enumerate(indices):
                    batch = {k:torch.as_tensor(v[selection],device=device) for k,v in replay['data'].items()}
                    torch.manual_seed(819000+seed+index)
                    row=dict(method=method,seed=seed,checkpoint_steps=step,batch=index,batch_size=128,
                        run=str(run),replay_source_steps=1000000,learner_updates_performed=0,
                        interpretation='common_final_replay_retrospective_probe_not_native_600k_replay',
                        **probe(learner,batch))
                    rows.append(row)
                del learner,state
            del full,replay
            atomic_json(output/'critic_probes.json',dict(baseline_commit=BASELINE,records=rows,
                training_transitions=0,optimizer_steps=0,physical_queries=0))
            print(json.dumps(dict(event='critic_probe_completed',method=method,seed=seed)),flush=True)
    fields=sorted({k for r in rows for k in r})
    with (output/'critic_probes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return rows


def cross_evaluations(project, output, device, frozen):
    rows=[]
    plan=dict(task_version=frozen['task_version'],layout_seed_start=92027,
        action_sampling_seed=3970027,action_batch_size=20)
    conditions=[dict(name='nominal',radius_m=.06,lateral_offset_m=0.),
                dict(name='radius_080',radius_m=.08,lateral_offset_m=0.)]
    with physics_pool(20) as pipes:
        for seed in (2030,2032):
            run=native_run(project,'flow_learned',seed)
            for actor_step in (600000,1000000):
                actor,_,_=load_policy(run,device,actor_step,frozen)
                for margin_step in (600000,1000000):
                    _,margin,_=load_policy(run,device,margin_step,frozen)
                    for condition in conditions:
                        records=rollout_wave(pipes,condition,list(range(20)),plan,actor,margin,seed)
                        result=dict(kind='posthoc_actor_margin_cross_NOT_primary',seed=seed,
                            actor_steps=actor_step,margin_steps=margin_step,condition=condition['name'],
                            baseline_commit=BASELINE,training_transitions=0,adaptation_updates=0,
                            **summarize(records),records=records)
                        name=f'cross-seed{seed}-actor{actor_step}-margin{margin_step}-{condition["name"]}.json'
                        atomic_json(output/name,result)
                        rows.append({k:v for k,v in result.items() if k!='records'})
                        print(json.dumps(dict(event='cross_completed',seed=seed,actor=actor_step,
                            margin=margin_step,condition=condition['name'],safe_success=result['safe_success'],
                            violation=result['violation'])),flush=True)
    with (output/'cross_evaluations.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def plot_logs(project, output):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return 'plot_pending_host_matplotlib'
    fig,axes=plt.subplots(4,4,figsize=(15,11),squeeze=False)
    jobs=[('flow_iso',2030),('flow_iso',2031),('flow_learned',2030),('flow_learned',2032)]
    for panels,(method,seed) in zip(axes,jobs):
        with (native_run(project,method,seed)/'progress.csv').open() as stream:
            rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
        minimum=950000 if method=='flow_iso' else 600000
        rows=[r for r in rows if r['steps']>=minimum]
        x=[r['steps']/1000 for r in rows]
        for ax,field,title in zip(panels,['reward_q_abs_max','clipping_rate','energy','safety_multiplier'],
                                  ['Replay |Q| max','Clipping','Path energy','Cost multiplier']):
            ax.plot(x,[r[field] for r in rows]);ax.set_title(f'{method},{seed}: {title}')
            ax.set_xlabel('Native train transitions,k');ax.grid(alpha=.2)
        # Only valid at lambda=0; logged alpha is post-update, so approximate.
        panels[0].plot(x,[r['alpha']*r['energy']-r['actor_loss'] if r['safety_multiplier']==0 else np.nan for r in rows],
            label='Approx policy signed Q,lambda=0',alpha=.75)
        panels[0].axhline(15.,color='red',linestyle='--',label='Task upper ceiling15')
        panels[0].legend(fontsize=7)
    fig.suptitle('Historical frozen logs; signed-Q probes in separate table; no retraining')
    fig.tight_layout();fig.savefig(output/'late_native_logs.png',dpi=140);plt.close(fig)
    return 'late_native_logs.png'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True,help='4080 project containing all fixed native4T archives')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Never overwrite a prior diagnosis')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    frozen=json.loads((Path(__file__).resolve().parents[3]/'configs/star/arm_safe_entropy/main_plan.json').read_text())
    started=time.monotonic()
    probes=critic_probes(args.project,args.output,args.device,frozen)
    cross=cross_evaluations(args.project,args.output,args.device,frozen)
    plot=plot_logs(args.project,args.output)
    atomic_json(args.output/'receipt.json',dict(status='completed_critic_and_actor_margin_attribution',
        source_commit=args.source_commit,baseline_commit=BASELINE,critic_probe_batches=len(probes),
        cross_condition_rows=len(cross),cross_policy_episodes=sum(r['episodes'] for r in cross),
        physical_evaluation_transitions=sum(r['evaluation_transitions'] for r in cross),
        training_transitions=0,optimizer_steps=0,oracle_queries=0,plot=plot,
        elapsed_seconds=time.monotonic()-started,
        remaining='Episode-isolated64state geometry/direction/risk diagnostic is separate; replay fits are not its substitute'))


if __name__=='__main__':
    main()

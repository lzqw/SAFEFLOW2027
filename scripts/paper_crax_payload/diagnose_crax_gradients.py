"""ONE frozen256-state replay gradient probe; no environment, SGD or replay writes.

All three objective components use the same single generated action/noise path
and the actual saved observation transform. Norms/cosines are ACTOR-PARAMETER
loss gradients: -Q_R, lambda*Q_C, alpha_E*energy. These are conditional local
diagnostics, not proof that a hyperparameter intervention will improve training.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import torch

from scripts.paper_crax_payload.crax_models import CumulativeFlowLearner
from scripts.paper_crax_payload.observation_normalizer import ObservationNormalizer


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(tensors):
    digest=hashlib.sha256()
    for name,value in sorted(tensors.items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def stats(value):
    value=value.detach()
    return dict(mean=float(value.mean().cpu()),std=float(value.std(unbiased=False).cpu()),
                min=float(value.min().cpu()),max=float(value.max().cpu()),
                nonzero=int(torch.count_nonzero(value).cpu()))


def cosine(a,b):
    denominator=a.norm()*b.norm()
    return float((a@b/denominator).cpu()) if denominator>1e-20 else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--analysis-sha256',required=True)
    parser.add_argument('--learner-sha256',required=True)
    parser.add_argument('--normalizer-sha256',required=True)
    parser.add_argument('--seed',type=int,default=202611125)
    args=parser.parse_args()
    if args.output.exists():parser.error('Do not overwrite a previous frozen diagnostic')
    source_hashes=dict(analysis=file_sha(__file__),learner=file_sha(inspect.getfile(CumulativeFlowLearner)),
                       normalizer=file_sha(inspect.getfile(ObservationNormalizer)))
    if source_hashes!=dict(analysis=args.analysis_sha256,learner=args.learner_sha256,normalizer=args.normalizer_sha256):
        raise ValueError('Actual analysis/learner/normalizer source differs from the explicitly pinned hashes')
    started=time.monotonic();torch.set_num_threads(1)
    checkpoint_stat=args.checkpoint.stat()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    saved=state['learner'];settings=saved['settings']
    permitted=set(inspect.signature(CumulativeFlowLearner.__init__).parameters)-{'self','device'}
    constructor={key:value for key,value in settings.items() if key in permitted}
    learner=CumulativeFlowLearner(**constructor,device=args.device)
    # load_state_dict checks the complete settings, including non-constructor
    # metadata; don't bypass batching/config guards to load an incompatible run.
    learner.load_state_dict(saved)
    normalizer=ObservationNormalizer(settings['obs_dim'],learner.device,state['normalizer']['mode'])
    normalizer.load_state_dict(state['normalizer'])
    before_actor=tensor_digest(learner.actor.state_dict())
    before_normalizer=tensor_digest({k:v for k,v in normalizer.state_dict().items() if torch.is_tensor(v)})
    replay=state['replay'];replay_rng_before=replay['rng'].clone()
    # Do not instantiate DeviceReplay or touch its RNG. Index raw storage once
    # using an independent CPU diagnostic generator.
    index_generator=torch.Generator(device='cpu').manual_seed(args.seed)
    indices=torch.randint(int(replay['size']),(256,),generator=index_generator)
    batch={key:value[indices].to(learner.device) for key,value in replay['data'].items()}
    observations=normalizer.transform(batch['obs'])
    if not torch.isfinite(observations).all():raise FloatingPointError('Saved transform generated nonfinite observations')
    for module in (learner.q,learner.cost_q,learner.margin,learner.margin_geometry):module.requires_grad_(False)
    parameters=tuple(learner.actor.parameters())
    noise_generator=torch.Generator(device=learner.device).manual_seed(args.seed+1)
    actions,energy,diagnostic=learner.actor(observations,geometry=learner.geometry,generator=noise_generator)
    reward_q=learner.q(observations,actions).min(0).values
    cost_q=learner.cost_q(observations,actions).min(0).values
    alpha=learner.log_alpha.detach().exp();multiplier=learner.multiplier.detach()
    objectives=dict(reward=-reward_q.mean(),unweighted_cost=cost_q.mean(),unweighted_energy=energy.mean())
    gradients={}
    for index,(name,objective) in enumerate(objectives.items()):
        values=torch.autograd.grad(objective,parameters,retain_graph=index<2,allow_unused=False)
        gradients[name]=torch.cat([value.detach().flatten() for value in values])
    weighted=dict(reward=gradients['reward'],cost=multiplier*gradients['unweighted_cost'],
                  energy=alpha*gradients['unweighted_energy'])
    total=sum(weighted.values())
    norm={name:float(value.norm().cpu()) for name,value in weighted.items()}
    reward_norm=norm['reward']
    pairwise={f'{a}__{b}':cosine(weighted[a],weighted[b])
              for a,b in [('reward','cost'),('reward','energy'),('cost','energy')]}
    total_norm=float(total.norm().cpu())
    after_actor=tensor_digest(learner.actor.state_dict())
    after_normalizer=tensor_digest({k:v for k,v in normalizer.state_dict().items() if torch.is_tensor(v)})
    if before_actor!=after_actor or before_normalizer!=after_normalizer or not torch.equal(replay_rng_before,replay['rng']):
        raise AssertionError('Frozen diagnostic mutated actor/normalizer/replay RNG')
    if args.checkpoint.stat().st_mtime_ns!=checkpoint_stat.st_mtime_ns or args.checkpoint.stat().st_size!=checkpoint_stat.st_size:
        raise AssertionError('Input checkpoint changed during the probe')
    result=dict(status='completed_frozen_single_batch_probe',source_hashes=source_hashes,
        checkpoint=str(args.checkpoint),checkpoint_bytes=checkpoint_stat.st_size,checkpoint_source_commit=state['source_commit'],
        checkpoint_steps=state['steps'],seed=args.seed,batch_size=256,
        replay_size=int(replay['size']),replay_position=int(replay['position']),
        indices=indices.tolist(),indices_sha256=hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
        settings=settings,constructor_kwargs_used=constructor,
        normalizer=dict(mode=normalizer.mode,count=float(normalizer.count.cpu()),
            mean=normalizer.mean.cpu().tolist(),std=normalizer.std.cpu().tolist(),
            updated_during_probe=False,normalized_observations=stats(observations)),
        raw_batch={key:stats(batch[key]) for key in ('reward','cost','terminated','truncated','margin')},
        sampled_action=stats(actions),energy=stats(energy),reward_q=stats(reward_q),cost_q=stats(cost_q),
        scalar_coefficients=dict(lambda_lagr=float(multiplier.cpu()),alpha_energy=float(alpha.cpu()),
            episode_cost_ema=float(learner.episode_cost_ema.cpu()),dual_episodes=float(learner.dual_episodes.cpu())),
        objective_means=dict(reward_term=float(objectives['reward'].detach().cpu()),
            weighted_cost_term=float((multiplier*objectives['unweighted_cost']).detach().cpu()),
            weighted_energy_term=float((alpha*objectives['unweighted_energy']).detach().cpu()),
            total_actor_objective=float((objectives['reward']+multiplier*objectives['unweighted_cost']+alpha*objectives['unweighted_energy']).detach().cpu())),
        actor_parameter_loss_gradient_norms=norm,
        unweighted_gradient_norms={name:float(value.norm().cpu()) for name,value in gradients.items()},
        weighted_cost_over_reward_norm=norm['cost']/reward_norm if reward_norm>0 else None,
        weighted_energy_over_reward_norm=norm['energy']/reward_norm if reward_norm>0 else None,
        pairwise_loss_gradient_cosines=pairwise,total_gradient_norm=total_norm,
        total_gradient_cosine_to_components={name:cosine(total,value) for name,value in weighted.items()},
        actor_parameter_count=sum(p.numel() for p in parameters),
        learner_global_grad_clip_factor_if_updated=min(1.,10./total_norm) if total_norm>0 else 1.,
        sampled_action_clipping_fraction=float(diagnostic['clipping'].mean().detach().cpu()),
        geometry_weight=float(diagnostic['geometry_weight'].mean().detach().cpu()),
        actor_weights_unchanged=before_actor==after_actor,normalizer_unchanged=before_normalizer==after_normalizer,
        replay_rng_unchanged=True,optimizer_step_calls=0,new_training_transitions=0,
        environment_objects_created=0,environment_step_calls=0,simulator_queries=0,real_interactions=0,
        elapsed_seconds=time.monotonic()-started,
        interpretation='One off-policy replay batch and one shared stochastic Actor path;loss-gradient signs explicit. Does not prove a dual-unit intervention will improve learning or estimate on-policy population effects.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','actor_parameter_loss_gradient_norms','weighted_cost_over_reward_norm',
        'weighted_energy_over_reward_norm','pairwise_loss_gradient_cosines','objective_means','new_training_transitions','simulator_queries')}),flush=True)


if __name__=='__main__':main()

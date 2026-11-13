"""Shared EA-STAB learner. Historical ArmLearner and frozen loading are untouched.

Only declared changes: slower Actor, Huber reward TD, critic-only preheat,
slow geometry model and the shared actual-transition ramp. No target/Q clipping
or additional revision is silently activated. Logging stays device-side until
the runner explicitly requests a periodic metrics snapshot.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from scripts.star.arm_safe_entropy.models import ArmLearner
from scripts.star.arm_safe_entropy.endpoint_geometry import (
    EndpointGeometry, StableFlowActor, frozen_parameters, geometry_ramp)


class StableArmLearner(ArmLearner):
    MODES = dict(s_iso='none', s_local='local', s_ea='endpoint_pullback')
    METRIC_NAMES = ('actor_loss','energy','clipping_rate','q_policy_mean','q_policy_max',
        'policy_cost','geometry_zero_gradient_fraction','geometry_proposal_clipped_fraction',
        'mean_reference_vn','mean_reference_vt','mean_geometry_weight','alpha_before',
        'reward_td','reward_loss','reward_q_abs_max','q_replay_mean',
        'q_replay_above_upper_fraction','alpha','cost_loss','margin_loss')

    def __init__(self, obs_dim=25, variant='s_ea', device='cuda:0', width=128,
                 actor_lr=1e-4, critic_lr=3e-4, critic_only_updates=1000,
                 margin_ema_tau=.005, actor_interval=2, rho=.25):
        if variant not in self.MODES or critic_only_updates < 0 or actor_interval < 1:
            raise ValueError('Use a declared shared-stability variant and valid update schedule')
        if not 0 < margin_ema_tau <= 1:
            raise ValueError('Margin EMA tau must be in(0,1]')
        # All three controls retain the SAME margin capacity,label batch and
        # optimizer updates. Iso trains the head but never uses its geometry.
        super().__init__(obs_dim,'flow_learned',device,width,
                         actor_lr=actor_lr,critic_lr=critic_lr,rho=rho)
        self.method='flow_iso' if variant=='s_iso' else 'flow_learned'
        self.variant = variant
        self.settings = dict(actor_lr=actor_lr,critic_lr=critic_lr,
            critic_only_updates=critic_only_updates,margin_ema_tau=margin_ema_tau,
            actor_interval=actor_interval,rho=rho,reward_loss='huber_beta1',
            reward_updates_per_round=2,target_samples=4,target_polyak=.001)
        # Preserve the native initial network weights AND generator RNG state;
        # constructing a new sampler wrapper must not shift learning randomness.
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            actor = StableFlowActor(obs_dim,width,variance=.04,rho=rho).to(self.device)
            actor.load_state_dict(self.actor.state_dict())
        self.actor = actor
        self.target_actor = deepcopy(actor).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.transitions = self.warmup_safety_episodes = 0
        self._last_metrics = {name:torch.zeros((),device=self.device) for name in self.METRIC_NAMES}
        self._last_metrics['alpha'] = self.log_alpha.detach().exp()
        if hasattr(self,'margin'):
            self.margin_online = self.margin  # Existing optimizer owns online only.
            self.margin_geometry = deepcopy(self.margin_online).requires_grad_(False)
            self.geometry = EndpointGeometry(self.margin_geometry,self.MODES[variant])
        else:
            self.geometry = None

    def set_transitions(self, transitions):
        if transitions < self.transitions:
            raise ValueError('Do not move the real transition/ramp counter backwards')
        self.transitions = int(transitions)
        if self.geometry is not None:
            self.geometry.transitions = self.transitions

    def record_episode(self, episode_cost):
        # Neither the multiplier NOR its episode EMA may ingest random/preheat
        # episodes before any Actor learning. Global runner counts still include
        # ALL their steps,completed episodes and real violations.
        if self.actor_updates == 0:
            self.warmup_safety_episodes += 1
            return
        super().record_episode(episode_cost)

    def update(self, batch):
        obs,action,reward,next_obs,done = [batch[k] for k in ('obs','action','reward','next_obs','done')]
        alpha = self.log_alpha.detach().exp()
        margin_loss = torch.zeros((),device=self.device)
        # geometry is NEVER updated here: online margin training cannot change
        # the geometry used inside either reward update or the Actor update.
        if hasattr(self,'margin_online'):
            predicted = self.margin_online(obs,action)
            mask = (torch.rand_like(predicted) < .8).to(predicted.dtype)
            margin_loss = ((predicted-batch['margin']).square()*mask).sum()/mask.sum().clamp_min(1.)
            self.margin_opt.zero_grad(set_to_none=True)
            margin_loss.backward()
            nn.utils.clip_grad_norm_(self.margin_online.parameters(),10.)
            self.margin_opt.step()
        with torch.no_grad():
            cost_action,_,_ = self.actor(next_obs,geometry=self.geometry)
            cost_target = batch['cost']+(1.-done)*self.cost_qt(next_obs,cost_action).mean(0)
        cost_loss = (self.cost_q(obs,action)-cost_target).square().mean()
        self.cost_opt.zero_grad(set_to_none=True)
        cost_loss.backward()
        nn.utils.clip_grad_norm_(self.cost_q.parameters(),10.)
        self.cost_opt.step()
        with torch.no_grad():
            for p,pt in zip(self.cost_q.parameters(),self.cost_qt.parameters()):
                pt.lerp_(p,.005)
        # Opt-in Piper engineering path: both independent four-sample reward
        # groups share one frozen target-actor traversal. Keep Qt evaluation
        # inside each round because its Polyak state changes after round one.
        packed = None
        if getattr(self, 'batch_reward_targets', False):
            with torch.no_grad():
                packed = self.target_actor(next_obs.repeat(8, 1), geometry=self.geometry)
        for reward_index in range(2):
            with torch.no_grad():
                expanded = next_obs.repeat(4,1)
                # The sampler passes THIS target generator into its geometry
                # query; therefore the deterministic tail is also target-owned.
                if packed is None:
                    next_action,next_energy,_ = self.target_actor(expanded,geometry=self.geometry)
                else:
                    begin = reward_index * 4 * len(obs)
                    end = begin + 4 * len(obs)
                    next_action, next_energy = packed[0][begin:end], packed[1][begin:end]
                future = (self.qt(expanded,next_action).min(0).values-alpha*next_energy).reshape(4,-1).mean(0)
                target = reward+.99*(1.-done)*future
            prediction = self.q(obs,action)
            reward_loss = F.smooth_l1_loss(prediction,target.expand_as(prediction),beta=1.)
            self.q_opt.zero_grad(set_to_none=True)
            reward_loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(),10.)
            self.q_opt.step()
            self.reward_updates += 1
            with torch.no_grad():
                for p,pt in zip(self.q.parameters(),self.qt.parameters()):
                    pt.lerp_(p,.001)
        self.updates += 1
        actor_age = self.updates-self.settings['critic_only_updates']
        if actor_age > 0 and actor_age % self.settings['actor_interval'] == 0:
            with frozen_parameters(self.q,self.cost_q):
                generated,energy,clips,trace = self.actor(obs,geometry=self.geometry,return_reference=True)
                policy_q = self.q(obs,generated).min(0).values
                policy_cost = self.cost_q(obs,generated).mean(0)
                actor_loss = (alpha*energy-policy_q+self.safety_multiplier*policy_cost).mean()
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),10.)
                self.actor_opt.step()
            alpha_loss = self.log_alpha*(self.energy_target-energy.detach().mean())
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                self.log_alpha.clamp_(-12.,3.)
                for p,pt in zip(self.actor.parameters(),self.target_actor.parameters()):
                    pt.lerp_(p,.001)
            self.actor_updates += 1
            self._last_metrics.update(actor_loss=actor_loss.detach(),energy=energy.detach().mean(),
                clipping_rate=clips.detach().mean(),q_policy_mean=policy_q.detach().mean(),
                q_policy_max=policy_q.detach().max(),policy_cost=policy_cost.detach().mean(),
                geometry_zero_gradient_fraction=trace['geometry_zero_gradient'].float().mean(),
                geometry_proposal_clipped_fraction=trace['geometry_proposal_clipped'].float().mean(),
                mean_reference_vn=trace['vn'].mean(),mean_reference_vt=trace['vt'].mean(),
                mean_geometry_weight=trace['weights'].mean(),alpha_before=alpha)
        # One EMA at the END of the entire learner round, including both reward
        # updates and the optional Actor update. Never between those operations.
        if hasattr(self,'margin_geometry'):
            with torch.no_grad():
                for online,geometry in zip(self.margin_online.parameters(),self.margin_geometry.parameters()):
                    geometry.lerp_(online,self.settings['margin_ema_tau'])
        upper = 5*(obs[:,14:17]-obs[:,17:20]).norm(dim=-1).clamp_max(1.)+10
        self._last_metrics.update(reward_td=(prediction.detach()-target).abs().mean(),
            reward_loss=reward_loss.detach(),reward_q_abs_max=prediction.detach().abs().max(),
            q_replay_mean=prediction.detach().min(0).values.mean(),
            q_replay_above_upper_fraction=(prediction.detach().min(0).values>upper+.01).float().mean(),
            alpha=self.log_alpha.detach().exp(),cost_loss=cost_loss.detach(),margin_loss=margin_loss.detach())
        # No .item(),.cpu(),.tolist() synchronization in this per-update path.

    def metrics(self):
        keys=list(self._last_metrics)
        values=torch.stack([self._last_metrics[k] for k in keys]).detach().cpu().tolist() if keys else []
        return dict(zip(keys,values)) | dict(safety_multiplier=self.safety_multiplier,
            geometry_beta=geometry_ramp(self.transitions),learner_updates=self.updates,
            reward_updates=self.reward_updates,actor_updates=self.actor_updates,
            transitions=self.transitions,warmup_safety_episodes=self.warmup_safety_episodes)

    def state_dict(self):
        state=super().state_dict()
        state['ea_stability_v2']=dict(variant=self.variant,settings=deepcopy(self.settings),
            transitions=self.transitions,warmup_safety_episodes=self.warmup_safety_episodes,
            last_metrics={k:v.detach() for k,v in self._last_metrics.items()},
            margin_geometry=self.margin_geometry.state_dict() if hasattr(self,'margin_geometry') else None)
        return state

    def load_state_dict(self,state):
        stable=state.get('ea_stability_v2')
        if stable is None or stable['variant']!=self.variant or stable['settings']!=self.settings:
            raise ValueError('Native resume must retain its variant/settings; revisions need an explicit fork')
        super().load_state_dict(state)
        if hasattr(self,'margin_geometry'):
            self.margin_geometry.load_state_dict(stable['margin_geometry'])
        self.set_transitions(stable['transitions'])
        self.warmup_safety_episodes=stable['warmup_safety_episodes']
        self._last_metrics={k:v.to(self.device) for k,v in stable['last_metrics'].items()}

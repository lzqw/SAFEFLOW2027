"""Payload uses binary episodic cost, unlike cumulative CRAX cost units."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from scripts.star.arm_safe_entropy.models import ArmLearner
from scripts.star.arm_safe_entropy.stability_models import StableArmLearner
from scripts.star.arm_safe_entropy.endpoint_geometry import frozen_parameters


class PayloadFlowLearner(StableArmLearner):
    def update(self, batch):
        super().update(batch)
        # Preserve the original update exactly, but use this task's actual
        # payload-to-goal distance for the optional reward upper diagnostic.
        with torch.no_grad():
            upper = 5 * (batch['obs'][:, 25:28] - batch['obs'][:, 42:45]).norm(dim=-1).clamp_max(1.) + 10
            values = self.q(batch['obs'], batch['action']).min(0).values
            self._last_metrics['q_replay_above_upper_fraction'] = (values > upper + .01).float().mean()


class PayloadSACLag(ArmLearner):
    """Gaussian SAC-Lag with matched shared critic stabilization and preheat.

    Policy entropy and Flow reference energy are different objectives. Both
    use the same observation, task, episode budget, reward units, width, replay,
    random/preheat data and reward-critic update count in the default protocol.
    """

    def __init__(self, obs_dim=45, device='cuda:0', width=128, actor_lr=1e-4,
                 critic_lr=3e-4, critic_only_updates=1000, actor_interval=2):
        super().__init__(obs_dim, 'sac_lag', device, width, actor_lr=actor_lr, critic_lr=critic_lr)
        self.variant = 'sac_lag'
        self.transitions = self.warmup_safety_episodes = 0
        self.settings = dict(actor_lr=actor_lr, critic_lr=critic_lr,
            critic_only_updates=critic_only_updates, actor_interval=actor_interval,
            reward_loss='huber_beta1', reward_updates_per_round=2, target_polyak=.001)
        self._last_metrics = {}

    def set_transitions(self, transitions):
        if transitions < self.transitions:
            raise ValueError('Cannot rewind training counter')
        self.transitions = int(transitions)

    def record_episode(self, episode_cost):
        if self.actor_updates == 0:
            self.warmup_safety_episodes += 1
        else:
            super().record_episode(episode_cost)

    def update(self, batch):
        obs, action, reward, next_obs, done = [batch[k] for k in ('obs', 'action', 'reward', 'next_obs', 'done')]
        alpha = self.log_alpha.detach().exp()
        with torch.no_grad():
            cost_action, _ = self.actor(next_obs)
            cost_target = batch['cost'] + (1. - done) * self.cost_qt(next_obs, cost_action).mean(0)
        cost_loss = (self.cost_q(obs, action) - cost_target).square().mean()
        self.cost_opt.zero_grad(set_to_none=True)
        cost_loss.backward()
        nn.utils.clip_grad_norm_(self.cost_q.parameters(), 10.)
        self.cost_opt.step()
        with torch.no_grad():
            for p, pt in zip(self.cost_q.parameters(), self.cost_qt.parameters()):
                pt.lerp_(p, .005)
        for _ in range(2):
            with torch.no_grad():
                next_action, next_logp = self.actor(next_obs)
                future = self.qt(next_obs, next_action).min(0).values - alpha * next_logp
                target = reward + .99 * (1. - done) * future
            prediction = self.q(obs, action)
            reward_loss = F.smooth_l1_loss(prediction, target.expand_as(prediction), beta=1.)
            self.q_opt.zero_grad(set_to_none=True)
            reward_loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), 10.)
            self.q_opt.step()
            self.reward_updates += 1
            with torch.no_grad():
                for p, pt in zip(self.q.parameters(), self.qt.parameters()):
                    pt.lerp_(p, .001)
        self.updates += 1
        age = self.updates - self.settings['critic_only_updates']
        if age > 0 and age % self.settings['actor_interval'] == 0:
            with frozen_parameters(self.q, self.cost_q):
                generated, logp = self.actor(obs)
                policy_q = self.q(obs, generated).min(0).values
                policy_cost = self.cost_q(obs, generated).mean(0)
                actor_loss = (alpha * logp - policy_q + self.safety_multiplier * policy_cost).mean()
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
                self.actor_opt.step()
            alpha_loss = -(self.log_alpha * (logp.detach() - 3.)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                self.log_alpha.clamp_(-12., 3.)
                for p, pt in zip(self.actor.parameters(), self.target_actor.parameters()):
                    pt.lerp_(p, .001)
            self.actor_updates += 1
            self._last_metrics.update(actor_loss=actor_loss.detach(), policy_logp=logp.detach().mean(),
                q_policy_mean=policy_q.detach().mean(), policy_cost=policy_cost.detach().mean())
        self._last_metrics.update(reward_td=(prediction.detach() - target).abs().mean(),
            reward_loss=reward_loss.detach(), cost_loss=cost_loss.detach(), alpha=self.log_alpha.detach().exp())

    def metrics(self):
        keys = list(self._last_metrics)
        values = torch.stack([self._last_metrics[k] for k in keys]).detach().cpu().tolist() if keys else []
        return dict(zip(keys, values)) | dict(safety_multiplier=self.safety_multiplier,
            learner_updates=self.updates, reward_updates=self.reward_updates, actor_updates=self.actor_updates,
            transitions=self.transitions, warmup_safety_episodes=self.warmup_safety_episodes)

    def state_dict(self):
        return super().state_dict() | dict(payload_sac=dict(settings=deepcopy(self.settings),
            transitions=self.transitions, warmup_safety_episodes=self.warmup_safety_episodes,
            last_metrics={k: v.detach() for k, v in self._last_metrics.items()}))

    def load_state_dict(self, state):
        extra = state.get('payload_sac')
        if extra is None or extra['settings'] != self.settings:
            raise ValueError('Restore matching Payload SAC configuration only')
        super().load_state_dict(state)
        self.set_transitions(extra['transitions'])
        self.warmup_safety_episodes = extra['warmup_safety_episodes']
        self._last_metrics = {k: v.to(self.device) for k, v in extra['last_metrics'].items()}


def build_payload_learner(config, method, device):
    shared = config['shared_learner']
    kwargs = dict(obs_dim=45, device=device, width=shared['hidden_width'],
        actor_lr=shared['actor_lr'], critic_lr=shared['critic_lr'],
        critic_only_updates=shared['critic_only_updates'], actor_interval=shared['actor_update_interval'])
    if method == 'sac_lag':
        learner = PayloadSACLag(**kwargs)
    elif method in ('s_iso', 's_ea'):
        learner = PayloadFlowLearner(**kwargs, variant=method,
            margin_ema_tau=shared['margin_ema_tau'], rho=shared['rho'])
    else:
        raise ValueError('Payload main comparison is SAC-Lag / Stable-Iso / EA')
    learner.cost_limit = config['cost']['episode_budget']
    return learner

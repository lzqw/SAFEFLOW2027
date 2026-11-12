"""Dimension-general Flow and cumulative-cost learner; historical Arm code untouched.

Geometry reuses the actual injection-site frozen tail/Jacobian implementation.
The CRAX margin target is endpoint hazard clearance, not a future cost probability
or an asserted interval minimum. Native cumulative costs remain entirely separate.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from scripts.star.arm_safe_entropy.models import mlp
from scripts.star.arm_safe_entropy.endpoint_geometry import EndpointGeometry, frozen_parameters


class QEnsemble(nn.Module):
    def __init__(self, obs_dim, action_dim, width=256, probability=False):
        super().__init__()
        self.heads = nn.ModuleList([mlp(obs_dim + action_dim, 1, width) for _ in range(2)])
        self.probability = probability

    def forward(self, obs, action):
        inputs = torch.cat((obs, action), -1)
        heads = torch.stack([head(inputs).squeeze(-1) for head in self.heads])
        return heads.sigmoid() if self.probability else heads


class MarginEnsemble(QEnsemble):
    temperature = .05


class UpdateGeometry(EndpointGeometry):
    """1000–5000 *effective margin updates*, not sampled transitions."""
    profile_full_geometry = False

    def set_margin_updates(self, updates):
        self.margin_updates = int(updates)
        # Exact algebraic reparameterization of the existing 10k–30k ramp.
        self.transitions = 30000 if self.profile_full_geometry else max(0, 10000 + 5 * (self.margin_updates - 1000))


class FlowActor(nn.Module):
    def __init__(self, obs_dim, action_dim, width=256, num_steps=4, variance=.04, rho=.25):
        super().__init__()
        if action_dim < 2 or num_steps < 1 or variance <= 0 or not 0 < rho <= 1:
            raise ValueError('Invalid flow dimensions/covariance')
        self.action_dim, self.num_steps = action_dim, num_steps
        self.variance, self.rho = variance, rho
        self.net = mlp(obs_dim + action_dim + 1, action_dim, width)
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def velocity(self, obs, action, time):
        return 2 * torch.tanh(self.net(torch.cat((obs, action, time), -1)))

    def forward(self, obs, deterministic=False, geometry=None, generator=None):
        action = obs.new_zeros((*obs.shape[:-1], self.action_dim))
        energy, clips, weights = (obs.new_zeros(obs.shape[:-1]) for _ in range(3))
        dt = 1 / self.num_steps
        for index in range(self.num_steps):
            time = torch.full_like(action[..., :1], index * dt)
            midpoint = (action + .5 * dt * self.velocity(obs, action, time)).clamp(-1, 1)
            velocity = self.velocity(obs, midpoint, time + .5 * dt)
            injection = action + dt * velocity
            if geometry is None:
                normal, weight = torch.zeros_like(action), torch.zeros_like(energy)
            else:
                normal, weight = geometry.query(self, obs, midpoint, injection, index)
            vn = self.variance * (1 - weight.clamp(0, 1) * (1 - self.rho))
            vt = (self.action_dim * self.variance - vn) / (self.action_dim - 1)
            normal_v = (normal * velocity).sum(-1)
            tangent2 = (velocity.square().sum(-1) - normal_v.square()).clamp_min(0)
            energy = energy + .5 * dt * (normal_v.square() / vn + tangent2 / vt)
            noise = torch.zeros_like(action) if deterministic else torch.randn(
                action.shape, dtype=action.dtype, device=action.device, generator=generator)
            increment = vt.sqrt()[..., None] * noise + ((vn.sqrt() - vt.sqrt()) *
                (noise * normal).sum(-1))[..., None] * normal
            proposal = injection + math.sqrt(dt) * increment
            clips = clips + (proposal.abs() > 1).to(obs.dtype).mean(-1) * dt
            weights = weights + weight * dt
            action = proposal.clamp(-1, 1)
        return action, energy, dict(clipping=clips, geometry_weight=weights)


def cost_target(cost, next_value, terminated, discount, task_mode):
    if task_mode not in ('cumulative', 'episodic_binary'):
        raise ValueError('Explicit task_mode required')
    return cost + (1 - terminated) * (discount if task_mode == 'cumulative' else 1.) * next_value


class CumulativeFlowLearner:
    def __init__(self, obs_dim, action_dim, method='stable_iso', device='cuda:0',
                 width=256, actor_lr=1e-4, critic_lr=3e-4, variance=.04, rho=.25,
                 cost_budget=25., discount=.99, critic_only_updates=1000, flow_steps=4):
        if method not in ('stable_iso', 'ea', 'local'):
            raise ValueError('Declared Flow method required')
        self.device, self.method = torch.device(device), method
        self.settings = dict(obs_dim=obs_dim, action_dim=action_dim, method=method,
            width=width, actor_lr=actor_lr, critic_lr=critic_lr, variance=variance,
            rho=rho, cost_budget=cost_budget, discount=discount,
            critic_only_updates=critic_only_updates, flow_steps=flow_steps,
            task_mode='cumulative', reward_scaling=1., cost_scaling=1.)
        self.actor = FlowActor(obs_dim, action_dim, width, flow_steps, variance, rho).to(self.device)
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.q = QEnsemble(obs_dim, action_dim, width).to(self.device)
        self.qt = deepcopy(self.q).requires_grad_(False)
        self.cost_q = QEnsemble(obs_dim, action_dim, width, probability=False).to(self.device)
        self.cost_qt = deepcopy(self.cost_q).requires_grad_(False)
        self.margin = MarginEnsemble(obs_dim, action_dim, 128).to(self.device)
        self.margin_geometry = deepcopy(self.margin).requires_grad_(False)
        mode = dict(stable_iso='none', local='local', ea='endpoint_pullback')[method]
        self.geometry = UpdateGeometry(self.margin_geometry, mode)
        self.geometry.set_margin_updates(0)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=critic_lr)
        self.cost_opt = torch.optim.Adam(self.cost_q.parameters(), lr=critic_lr)
        self.margin_opt = torch.optim.Adam(self.margin.parameters(), lr=critic_lr)
        self.log_alpha = torch.tensor(math.log(.02 * variance), device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=1e-4)
        self.energy_target = action_dim * .25 / variance
        self.multiplier = torch.zeros((), device=self.device)
        self.episode_cost_ema = torch.zeros((), device=self.device)
        self.dual_episodes = torch.zeros((), device=self.device)
        self.updates = self.actor_updates = self.margin_updates = self.reward_updates = 0
        self.last_metrics = {}

    @torch.no_grad()
    def record_completed_episodes(self, costs, completed):
        """Masked full episodes only; no host sync or sampling-block pseudoepisode.

        A simultaneous completed batch contributes its mean once; lr.01 acts in
        raw episodic-cost units. Number of completed episodes is recorded. This
        intentionally differs from native SAC's per-step normalized dual rule.
        """
        if self.actor_updates == 0:
            return
        count = completed.to(costs.dtype).sum()
        mean = (costs * completed).sum() / count.clamp_min(1)
        ema = torch.where(self.dual_episodes == 0, mean, .95 * self.episode_cost_ema + .05 * mean)
        self.episode_cost_ema.copy_(torch.where(count > 0, ema, self.episode_cost_ema))
        candidate = (self.multiplier + .01 * (ema - self.settings['cost_budget'])).clamp(0, 100)
        self.multiplier.copy_(torch.where(count > 0, candidate, self.multiplier))
        self.dual_episodes.add_(count)

    def update(self, batch):
        obs, action, reward, nxt, terminated = (batch[k] for k in
            ('obs','action','reward','next_obs','terminated'))
        # Raw state observations and rewards; no undisclosed normalization/scaling.
        predicted = self.margin(obs, action)
        mask = (torch.rand_like(predicted) < .8).to(obs.dtype)
        margin_loss = ((predicted - batch['margin']).square() * mask).sum() / mask.sum().clamp_min(1)
        self.margin_opt.zero_grad(set_to_none=True)
        margin_loss.backward()
        nn.utils.clip_grad_norm_(self.margin.parameters(), 10)
        self.margin_opt.step()
        self.margin_updates += 1
        alpha = self.log_alpha.detach().exp()
        with torch.no_grad():
            next_action, _, _ = self.target_actor(nxt, geometry=self.geometry)
            target_c = cost_target(batch['cost'], self.cost_qt(nxt, next_action).min(0).values,
                terminated, self.settings['discount'], 'cumulative')
        loss_c = F.smooth_l1_loss(self.cost_q(obs, action), target_c.expand(2, -1))
        self.cost_opt.zero_grad(set_to_none=True)
        loss_c.backward()
        nn.utils.clip_grad_norm_(self.cost_q.parameters(), 10)
        self.cost_opt.step()
        with torch.no_grad():
            for p, t in zip(self.cost_q.parameters(), self.cost_qt.parameters()):
                t.lerp_(p, .005)
        for _ in range(2):
            with torch.no_grad():
                expanded = nxt.repeat(4, 1)
                sampled, energy, _ = self.target_actor(expanded, geometry=self.geometry)
                future = (self.qt(expanded, sampled).min(0).values - alpha * energy).reshape(4, -1).mean(0)
                target = reward + self.settings['discount'] * (1 - terminated) * future
            qloss = F.smooth_l1_loss(self.q(obs, action), target.expand(2, -1), beta=1.)
            self.q_opt.zero_grad(set_to_none=True)
            qloss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), 10)
            self.q_opt.step()
            self.reward_updates += 1
            with torch.no_grad():
                for p, t in zip(self.q.parameters(), self.qt.parameters()):
                    t.lerp_(p, .001)
        self.updates += 1
        age = self.updates - self.settings['critic_only_updates']
        if age > 0 and age % 2 == 0:
            with frozen_parameters(self.q, self.cost_q):
                sampled, energy, diag = self.actor(obs, geometry=self.geometry)
                objective = (alpha * energy - self.q(obs, sampled).min(0).values +
                    self.multiplier * self.cost_q(obs, sampled).min(0).values).mean()
                self.actor_opt.zero_grad(set_to_none=True)
                objective.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
                self.actor_opt.step()
            aloss = self.log_alpha * (self.energy_target - energy.detach().mean())
            self.alpha_opt.zero_grad(set_to_none=True)
            aloss.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                self.log_alpha.clamp_(-12, 3)
                for p, t in zip(self.actor.parameters(), self.target_actor.parameters()):
                    t.lerp_(p, .001)
            self.actor_updates += 1
            self.last_metrics.update(actor_loss=objective.detach(), energy=energy.detach().mean(),
                clipping=diag['clipping'].mean().detach(), geometry_weight=diag['geometry_weight'].mean().detach())
        with torch.no_grad():
            for p, t in zip(self.margin.parameters(), self.margin_geometry.parameters()):
                t.lerp_(p, .005)
        self.geometry.set_margin_updates(self.margin_updates)
        self.last_metrics.update(reward_loss=qloss.detach(), cost_loss=loss_c.detach(),
            margin_loss=margin_loss.detach(), alpha=self.log_alpha.detach().exp(),
            multiplier=self.multiplier.detach().clone())

    def metrics(self):
        keys = list(self.last_metrics)
        values = torch.stack([self.last_metrics[k] for k in keys]).detach().cpu().tolist() if keys else []
        return dict(zip(keys, values)) | dict(updates=self.updates, actor_updates=self.actor_updates,
            margin_updates=self.margin_updates, reward_updates=self.reward_updates,
            energy_target=self.energy_target)

    def state_dict(self):
        return dict(settings=deepcopy(self.settings), models={k:getattr(self,k).state_dict() for k in
            ('actor','target_actor','q','qt','cost_q','cost_qt','margin','margin_geometry')},
            optimizers={k:getattr(self,k).state_dict() for k in
            ('actor_opt','q_opt','cost_opt','margin_opt','alpha_opt')},
            log_alpha=self.log_alpha.detach(), multiplier=self.multiplier.detach(),
            episode_cost_ema=self.episode_cost_ema.detach(),dual_episodes=self.dual_episodes.detach(),
            updates=self.updates, actor_updates=self.actor_updates,
            margin_updates=self.margin_updates,reward_updates=self.reward_updates)

    def load_state_dict(self, state):
        if state['settings'] != self.settings:
            raise ValueError('Exact continuation requires identical configuration')
        for k, value in state['models'].items():
            getattr(self,k).load_state_dict(value)
        for k, value in state['optimizers'].items():
            getattr(self,k).load_state_dict(value)
        with torch.no_grad():
            for k in ('log_alpha','multiplier','episode_cost_ema','dual_episodes'):
                getattr(self,k).copy_(state[k])
        for k in ('updates','actor_updates','margin_updates','reward_updates'):
            setattr(self,k,state[k])
        self.geometry.set_margin_updates(self.margin_updates)

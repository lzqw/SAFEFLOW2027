"""Small single-policy SAC / flow backbones for the arm development task.

The flow midpoint discretization and inverse-reference energy follow the
existing FlowActor. Arm-v1 starts Brownian flow at zero: an independent
uniform base must not drown out the directional covariance being tested.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F


def mlp(inputs, outputs, width=128):
    return nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, outputs))


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, width=128):
        super().__init__()
        self.net = mlp(obs_dim, 6, width)

    def forward(self, obs, deterministic=False):
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        dist = torch.distributions.Normal(mean, log_std.clamp(-20, 2).exp())
        raw = mean if deterministic else dist.rsample()
        action = raw.tanh()
        logp = (dist.log_prob(raw) - 2. * (math.log(2.) - raw - F.softplus(-2. * raw))).sum(-1)
        return action, logp


class ArmFlowActor(nn.Module):
    def __init__(self, obs_dim, width=128, num_steps=4, variance=.04, rho=.25):
        super().__init__()
        self.net = mlp(obs_dim + 4, 3, width)
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)
        self.num_steps, self.variance, self.rho = num_steps, variance, rho

    def velocity(self, obs, action, time):
        return 2. * torch.tanh(self.net(torch.cat([obs, action, time], -1)))

    def forward(self, obs, deterministic=False, geometry=None):
        action = torch.zeros((*obs.shape[:-1], 3), device=obs.device, dtype=obs.dtype)
        energy = torch.zeros(obs.shape[:-1], device=obs.device)
        clips = torch.zeros_like(energy)
        dt = 1. / self.num_steps
        for index in range(self.num_steps):
            t = torch.full_like(action[..., :1], index * dt)
            midpoint = (action + .5 * dt * self.velocity(obs, action, t)).clamp(-1, 1)
            velocity = self.velocity(obs, midpoint, t + .5 * dt)
            if geometry is None:
                normal = torch.zeros_like(action)
                weight = torch.zeros_like(energy)
            else:
                normal, weight = geometry(obs, midpoint)
                normal, weight = normal.detach(), weight.detach()
            vn = self.variance * (1. - weight * (1. - self.rho))
            vt = (3. * self.variance - vn) / 2.
            vnormal = (velocity * normal).sum(-1)
            vtangent2 = (velocity.square().sum(-1) - vnormal.square()).clamp_min(0)
            energy = energy + .5 * dt * (vnormal.square() / vn + vtangent2 / vt)
            noise = torch.zeros_like(action) if deterministic else torch.randn_like(action)
            increment = vt.sqrt()[..., None] * noise + ((vn.sqrt() - vt.sqrt()) * (noise * normal).sum(-1))[..., None] * normal
            proposal = action + dt * velocity + math.sqrt(dt) * increment
            clips = clips + (proposal.abs() > 1).float().mean(-1) / self.num_steps
            action = proposal.clamp(-1, 1)
        return action, energy, clips


class TwinQ(nn.Module):
    def __init__(self, obs_dim, width=128, probability=False):
        super().__init__()
        self.heads = nn.ModuleList([mlp(obs_dim + 3, 1, width) for _ in range(2)])
        self.probability = probability

    def forward(self, obs, action):
        value = torch.stack([head(torch.cat([obs, action], -1)).squeeze(-1) for head in self.heads])
        return value.sigmoid() if self.probability else value


class ArmLearner:
    def __init__(self, obs_dim, method, device='cuda:0', width=128, variance=.04,
                 actor_lr=3e-4, critic_lr=3e-4, kinetic_alpha=.02, kinetic_target_per_action=.25,
                 oracle_geometry=None):
        self.method, self.device = method, torch.device(device)
        self.flow = method.startswith('flow')
        self.safety = method not in ['sac', 'flow']
        self.actor = (ArmFlowActor(obs_dim, width, variance=variance) if self.flow else GaussianActor(obs_dim, width)).to(self.device)
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.q = TwinQ(obs_dim, width).to(self.device)
        self.qt = deepcopy(self.q).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=critic_lr)
        # nu*alpha_kin and target_kin/nu preserve reference-energy units.
        self.log_alpha = torch.tensor(math.log(kinetic_alpha * variance if self.flow else .1), device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=1e-4 if self.flow else 3e-4)
        self.energy_target = 3. * kinetic_target_per_action / variance
        self.updates, self.reward_updates, self.actor_updates = 0, 0, 0
        self.geometry = None
        if method == 'flow_oracle':
            if oracle_geometry is None:
                raise ValueError('OracleDir requires actual controller-space oracle geometry')
            self.geometry = oracle_geometry
        self.cost_limit = .1
        self.safety_multiplier, self.episode_cost_ema = 0., None
        self.completed_safety_episodes = 0
        if self.safety:
            self.cost_q = TwinQ(obs_dim, width, probability=True).to(self.device)
            self.cost_qt = deepcopy(self.cost_q).requires_grad_(False)
            self.cost_opt = torch.optim.Adam(self.cost_q.parameters(), lr=critic_lr)
        if method == 'flow_learned':
            from scripts.star.arm_safe_entropy.geometry import LocalMarginEnsemble
            self.margin = LocalMarginEnsemble(obs_dim).to(self.device)
            self.margin_opt = torch.optim.Adam(self.margin.parameters(), lr=critic_lr)
            self.geometry = self.margin.geometry

    def record_episode(self, episode_cost):
        if not self.safety:
            return
        self.episode_cost_ema = float(episode_cost) if self.episode_cost_ema is None else .95 * self.episode_cost_ema + .05 * float(episode_cost)
        # Random warmup costs are counted but cannot drive the dual to its cap
        # before a learned behavior policy/critic has taken any update.
        if self.updates > 0:
            self.safety_multiplier = min(100., max(0., self.safety_multiplier + .05 * (self.episode_cost_ema - self.cost_limit)))
        self.completed_safety_episodes += 1

    @torch.no_grad()
    def action(self, obs, deterministic=False):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if self.flow:
            action, _, _ = self.actor(x, deterministic=deterministic, geometry=self.geometry)
        else:
            action, _ = self.actor(x, deterministic=deterministic)
        return action.cpu().numpy()

    def update(self, batch):
        obs, action, reward, next_obs, done = [batch[k] for k in ['obs', 'action', 'reward', 'next_obs', 'done']]
        alpha = self.log_alpha.detach().exp()
        q_loss = td = torch.zeros((), device=self.device)
        cost_loss = margin_loss = torch.zeros((), device=self.device)
        if hasattr(self, 'margin'):
            predicted_margin = self.margin(obs, action)
            # Independent bootstrap masks; labels are actual recorded interval minima.
            mask = (torch.rand_like(predicted_margin) < .8).float()
            margin_loss = ((predicted_margin - batch['margin']).square() * mask).sum() / mask.sum().clamp_min(1.)
            self.margin_opt.zero_grad()
            margin_loss.backward()
            nn.utils.clip_grad_norm_(self.margin.parameters(), 10.)
            self.margin_opt.step()
        if self.safety:
            with torch.no_grad():
                if self.flow:
                    a_cost, _, _ = self.actor(next_obs, geometry=self.geometry)
                else:
                    a_cost, _ = self.actor(next_obs)
                cost_target = batch['cost'] + (1. - done) * self.cost_qt(next_obs, a_cost).mean(0)
            cost_loss = (self.cost_q(obs, action) - cost_target).square().mean()
            self.cost_opt.zero_grad()
            cost_loss.backward()
            nn.utils.clip_grad_norm_(self.cost_q.parameters(), 10.)
            self.cost_opt.step()
            with torch.no_grad():
                for p, pt in zip(self.cost_q.parameters(), self.cost_qt.parameters()):
                    pt.lerp_(p, .005)
        for _ in range(2 if self.flow else 1):
            with torch.no_grad():
                if self.flow:
                    expanded = next_obs.repeat(4, 1)
                    a2, energy2, _ = self.target_actor(expanded, geometry=self.geometry)
                    future = (self.qt(expanded, a2).min(0).values - alpha * energy2).reshape(4, -1).mean(0)
                else:
                    a2, logp2 = self.actor(next_obs)
                    future = self.qt(next_obs, a2).min(0).values - alpha * logp2
                target = reward + .99 * (1. - done) * future
            prediction = self.q(obs, action)
            q_loss = (prediction - target).square().mean()
            self.q_opt.zero_grad()
            q_loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), 10.)
            self.q_opt.step()
            td = (prediction.detach() - target).abs().mean()
            self.reward_updates += 1
            with torch.no_grad():
                for p, pt in zip(self.q.parameters(), self.qt.parameters()):
                    pt.lerp_(p, .001 if self.flow else .005)
        self.updates += 1
        actor_loss = energy_mean = clipping = torch.zeros((), device=self.device)
        if not self.flow or self.updates % 2 == 0:
            self.q.requires_grad_(False)
            if self.safety:
                self.cost_q.requires_grad_(False)
            if self.flow:
                a, energy, clip = self.actor(obs, geometry=self.geometry)
                actor_objective = alpha * energy - self.q(obs, a).min(0).values
                alpha_loss = self.log_alpha * (self.energy_target - energy.detach().mean())
                energy_mean, clipping = energy.mean(), clip.mean()
            else:
                a, logp = self.actor(obs)
                actor_objective = alpha * logp - self.q(obs, a).min(0).values
                alpha_loss = -(self.log_alpha * (logp.detach() - 3.)).mean()
            if self.safety:
                # No legacy /(1+lambda) rescaling: exactly the arm-v1 objective.
                actor_objective = actor_objective + self.safety_multiplier * self.cost_q(obs, a).mean(0)
            actor_loss = actor_objective.mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
            self.actor_opt.step()
            self.q.requires_grad_(True)
            if self.safety:
                self.cost_q.requires_grad_(True)
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                self.log_alpha.clamp_(-12., 3.)
                for p, pt in zip(self.actor.parameters(), self.target_actor.parameters()):
                    pt.lerp_(p, .001)
            self.actor_updates += 1
        values = torch.stack([td, prediction.detach().abs().max(), actor_loss.detach(),
                              energy_mean.detach(), clipping.detach(), self.log_alpha.detach().exp(),
                              cost_loss.detach(), margin_loss.detach()]).cpu().tolist()
        return dict(zip(['reward_td', 'reward_q_abs_max', 'actor_loss', 'energy', 'clipping_rate', 'alpha',
                         'cost_loss', 'margin_loss'], values)) | dict(safety_multiplier=self.safety_multiplier)

    def state_dict(self):
        names = ['actor', 'target_actor', 'q', 'qt', 'actor_opt', 'q_opt', 'alpha_opt']
        if self.safety:
            names += ['cost_q', 'cost_qt', 'cost_opt']
        if hasattr(self, 'margin'):
            names += ['margin', 'margin_opt']
        return {key: getattr(self, key).state_dict() for key in names} | dict(
            log_alpha=self.log_alpha.detach(), updates=self.updates,
            reward_updates=self.reward_updates, actor_updates=self.actor_updates,
            safety_multiplier=self.safety_multiplier, episode_cost_ema=self.episode_cost_ema,
            completed_safety_episodes=self.completed_safety_episodes)

    def load_state_dict(self, state):
        names = ['actor', 'target_actor', 'q', 'qt', 'actor_opt', 'q_opt', 'alpha_opt']
        if self.safety:
            names += ['cost_q', 'cost_qt', 'cost_opt']
        if hasattr(self, 'margin'):
            names += ['margin', 'margin_opt']
        for key in names:
            getattr(self, key).load_state_dict(state[key])
        with torch.no_grad():
            self.log_alpha.copy_(state['log_alpha'])
        for key in ['updates', 'reward_updates', 'actor_updates']:
            setattr(self, key, state[key])
        for key in ['safety_multiplier', 'episode_cost_ema', 'completed_safety_episodes']:
            if key in state:
                setattr(self, key, state[key])

"""Shared CRAX/Acme-style running mean/std, float32, device-only statistics.

Population std bounds1e-6..1e6, no observation clipping, initialmean0/std1.
Replay remains raw. update only on real training observations; evaluation calls
transform only. Normalization is a registered training choice, not a hidden fix.
"""
import torch


class ObservationNormalizer:
    def __init__(self, dimension, device, mode='none'):
        if mode not in ('none','running_acme'):
            raise ValueError('Explicit none/running_acme normalization mode required')
        self.mode=mode
        self.count=torch.zeros((),device=device)
        self.mean=torch.zeros(dimension,device=device)
        self.summed_variance=torch.zeros_like(self.mean)
        self.std=torch.ones_like(self.mean)

    @torch.no_grad()
    def update(self, observations):
        if self.mode=='none':return
        flat=observations.detach().reshape(-1,self.mean.numel())
        n=flat.shape[0];batch_mean=flat.mean(0)
        batch_m2=(flat-batch_mean).square().sum(0)
        total=self.count+n;delta=batch_mean-self.mean
        self.summed_variance.add_(batch_m2+delta.square()*self.count*n/total)
        self.mean.add_(delta*n/total);self.count.copy_(total)
        self.std.copy_((self.summed_variance.clamp_min(0)/total).sqrt().clamp(1e-6,1e6))

    def transform(self, observations):
        return observations if self.mode=='none' else (observations-self.mean)/self.std

    def batch(self, raw):
        return raw|dict(obs=self.transform(raw['obs']),next_obs=self.transform(raw['next_obs']))

    def state_dict(self):
        return dict(mode=self.mode,count=self.count.detach().clone(),mean=self.mean.detach().clone(),
            summed_variance=self.summed_variance.detach().clone(),std=self.std.detach().clone(),
            definition='raw_training_observations_only;population_std_clip1e-6_1e6;no_obs_clipping')

    def load_state_dict(self,state):
        if self.mode!=state['mode'] or self.mean.shape!=state['mean'].shape:
            raise ValueError('Do not change observation units during native continuation')
        for name in ('count','mean','summed_variance','std'):
            getattr(self,name).copy_(state[name])

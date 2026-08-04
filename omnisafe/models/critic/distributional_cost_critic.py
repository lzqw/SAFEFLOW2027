# Copyright 2026 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Ensemble quantile cost critic for STAR route A."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from omnisafe.typing import Activation, InitFunction
from omnisafe.utils.model import build_mlp_network


class DistributionalCostCritic(nn.Module):
    """Predict non-negative cost-to-go quantiles with independent ensemble heads."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list[int],
        num_quantiles: int = 25,
        ensemble_size: int = 2,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        """Initialize the quantile ensemble and fixed quantile fractions."""
        super().__init__()
        if min(obs_dim, act_dim, num_quantiles, ensemble_size) <= 0:
            raise ValueError('critic dimensions, quantiles, and ensemble size must be positive')
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_quantiles = num_quantiles
        self.ensemble_size = ensemble_size
        self.heads = nn.ModuleList(
            [
                build_mlp_network(
                    sizes=[obs_dim + act_dim, *hidden_sizes, num_quantiles],
                    activation=activation,
                    output_activation='identity',
                    weight_initialization_mode=weight_initialization_mode,
                )
                for _ in range(ensemble_size)
            ],
        )
        fractions = (torch.arange(num_quantiles, dtype=torch.float32) + 0.5) / num_quantiles
        self.register_buffer('quantile_fractions', fractions)

    def forward(  # pylint: disable=not-callable
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Return quantiles with shape ``[ensemble, batch, quantile]``."""
        inputs = torch.cat((obs, action), dim=-1)
        return torch.stack([functional.softplus(head(inputs)) for head in self.heads], dim=0)

    def expected_cost(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return each ensemble head's expected future cost."""
        return self(obs, action).mean(dim=-1)

    def exceedance_probability(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        threshold: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Return smooth probability that future cost exceeds a per-state threshold."""
        if temperature <= 0:
            raise ValueError('risk temperature must be positive')
        if threshold.ndim == 1:
            threshold = threshold.unsqueeze(-1)
        quantiles = self(obs, action)
        return torch.sigmoid((quantiles - threshold.unsqueeze(0)) / temperature).mean(dim=-1)


@torch.no_grad()
def distributional_bellman_target(
    cost: torch.Tensor,
    done: torch.Tensor,
    next_quantiles: torch.Tensor,
    gamma_cost: float,
) -> torch.Tensor:
    """Construct the route-A quantile Bellman target without a target-network graph."""
    if cost.ndim == 1:
        cost = cost.unsqueeze(-1)
    if done.ndim == 1:
        done = done.unsqueeze(-1)
    return cost.unsqueeze(0) + gamma_cost * (1.0 - done.unsqueeze(0)) * next_quantiles


# pylint: disable-next=too-many-arguments
def quantile_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    fractions: torch.Tensor,
    importance_weights: torch.Tensor | None = None,
    bootstrap_mask: torch.Tensor | None = None,
    huber_kappa: float = 1.0,
) -> torch.Tensor:
    """Return pairwise quantile Huber regression loss for an ensemble batch."""
    if prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError('prediction and target must share [ensemble, batch, quantile] shape')
    if huber_kappa <= 0:
        raise ValueError('Huber threshold must be positive')
    error = target.unsqueeze(-2) - prediction.unsqueeze(-1)
    absolute = error.abs()
    huber = torch.where(
        absolute <= huber_kappa,
        0.5 * error.square(),
        huber_kappa * (absolute - 0.5 * huber_kappa),
    )
    tau = fractions.reshape(1, 1, -1, 1).to(prediction)
    asymmetric = (tau - (error.detach() < 0).to(prediction.dtype)).abs()
    per_head_sample = (asymmetric * huber / huber_kappa).mean(dim=(-1, -2))

    weights = torch.ones_like(per_head_sample)
    if importance_weights is not None:
        weights = weights * importance_weights.reshape(1, -1).to(prediction)
    if bootstrap_mask is not None:
        if bootstrap_mask.shape != per_head_sample.shape:
            raise ValueError('bootstrap mask must have [ensemble, batch] shape')
        weights = weights * bootstrap_mask.to(prediction)
    return (per_head_sample * weights).sum() / weights.sum().clamp_min(1.0)

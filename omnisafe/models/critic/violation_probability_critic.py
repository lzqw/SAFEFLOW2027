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
"""Independent ensemble probability critic for completed-episode budget violations."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from omnisafe.typing import Activation, InitFunction
from omnisafe.utils.model import build_mlp_network
from omnisafe.utils.star_validation import average_ranks


class ViolationProbabilityCritic(nn.Module):
    """Predict a Bernoulli budget-violation probability with independent heads."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list[int],
        ensemble_size: int = 2,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        """Initialize independent logit networks without a forced 0.5 cost threshold."""
        super().__init__()
        if min(obs_dim, act_dim, ensemble_size) <= 0:
            raise ValueError('probability critic dimensions and ensemble size must be positive')
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ensemble_size = ensemble_size
        self.heads = nn.ModuleList(
            [
                build_mlp_network(
                    sizes=[obs_dim + act_dim, *hidden_sizes, 1],
                    activation=activation,
                    output_activation='identity',
                    weight_initialization_mode=weight_initialization_mode,
                )
                for _ in range(ensemble_size)
            ],
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``[ensemble, batch]``."""
        if obs.ndim != 2 or action.ndim != 2 or obs.shape[0] != action.shape[0]:
            raise ValueError('probability critic expects matching rank-two batches')
        if obs.shape[-1] != self.obs_dim or action.shape[-1] != self.act_dim:
            raise ValueError('probability critic observation/action width is invalid')
        inputs = torch.cat((obs, action), dim=-1)
        return torch.stack([head(inputs).squeeze(-1) for head in self.heads], dim=0)

    def probabilities(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return head-wise Bernoulli probabilities."""
        return self(obs, action).sigmoid()

    def ucb(self, obs: torch.Tensor, action: torch.Tensor, beta: float) -> torch.Tensor:
        """Return a clipped ensemble upper-confidence probability."""
        if beta < 0:
            raise ValueError('probability UCB beta must be non-negative')
        probabilities = self.probabilities(obs, action)
        return (probabilities.mean(dim=0) + beta * probabilities.std(dim=0, unbiased=False)).clamp(
            0.0,
            1.0,
        )


def masked_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    labeled: torch.Tensor,
    importance_weights: torch.Tensor | None = None,
    bootstrap_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Train only finalized episode labels, with optional replay and bootstrap weights."""
    if logits.ndim != 2:
        raise ValueError('probability logits must have [ensemble, batch] shape')
    target = target.reshape(-1).to(logits)
    labeled = labeled.reshape(-1).bool().to(logits.device)
    if target.shape[0] != logits.shape[1] or labeled.shape[0] != logits.shape[1]:
        raise ValueError('probability targets and masks must match the batch dimension')
    expanded_target = target.unsqueeze(0).expand_as(logits)
    selected_target = target[labeled]
    if bool((~torch.isfinite(selected_target)).any()) or bool(
        ((selected_target < 0) | (selected_target > 1)).any(),
    ):
        raise ValueError('labeled probability targets must be finite and lie in [0, 1]')
    weights = labeled.to(logits).unsqueeze(0).expand_as(logits)
    if importance_weights is not None:
        weights = weights * importance_weights.reshape(1, -1).to(logits)
    if bootstrap_mask is not None:
        if bootstrap_mask.shape != logits.shape:
            raise ValueError('probability bootstrap mask must match [ensemble, batch]')
        weights = weights * bootstrap_mask.to(logits)
    losses = functional.binary_cross_entropy_with_logits(
        logits,
        expanded_target,
        reduction='none',
    )
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def binary_brier_score(probability: torch.Tensor, target: torch.Tensor) -> float:
    """Return mean squared Bernoulli probability error."""
    probability = probability.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    if probability.numel() != target.numel() or probability.numel() == 0:
        return float('nan')
    if not bool(torch.isfinite(probability).all()) or bool(
        ((probability < 0) | (probability > 1)).any(),
    ):
        raise ValueError('Bernoulli probabilities must be finite and lie in [0, 1]')
    return float((probability - target).square().mean().item())


def binary_brier_skill(probability: torch.Tensor, target: torch.Tensor) -> float:
    """Return Brier skill relative to the observed-label climatology baseline."""
    probability = probability.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    if probability.numel() != target.numel() or probability.numel() == 0:
        return float('nan')
    if not bool(torch.isfinite(probability).all()) or bool(
        ((probability < 0) | (probability > 1)).any(),
    ):
        raise ValueError('Bernoulli probabilities must be finite and lie in [0, 1]')
    reference = (target - target.mean()).square().mean()
    if float(reference.item()) <= torch.finfo(reference.dtype).eps:
        return float('nan')
    score = (probability - target).square().mean()
    return float((1.0 - score / reference).item())


def binary_auroc(probability: torch.Tensor, target: torch.Tensor) -> float:
    """Return tie-correct binary AUROC using the Mann-Whitney rank identity."""
    probability = probability.detach().float().reshape(-1)
    target = target.detach().bool().reshape(-1)
    if probability.numel() != target.numel() or probability.numel() == 0:
        return float('nan')
    if not bool(torch.isfinite(probability).all()) or bool(
        ((probability < 0) | (probability > 1)).any(),
    ):
        raise ValueError('Bernoulli probabilities must be finite and lie in [0, 1]')
    positives = int(target.sum().item())
    negatives = target.numel() - positives
    if positives == 0 or negatives == 0:
        return float('nan')
    ranks = average_ranks(probability)
    positive_rank_sum = ranks[target].sum()
    minimum_positive_rank_sum = positives * (positives - 1) / 2.0
    return float(
        ((positive_rank_sum - minimum_positive_rank_sum) / (positives * negatives)).item(),
    )

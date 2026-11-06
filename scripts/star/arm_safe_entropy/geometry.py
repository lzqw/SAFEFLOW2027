"""Detached local-margin directional geometry, independent of long-horizon cost Q."""
import torch
from torch import nn

from scripts.star.arm_safe_entropy.models import mlp


def strict_normals(gradient, minimum_norm=1e-7):
    """Use exact unit vectors when valid, zeros otherwise (no biased projection)."""
    norm = gradient.norm(dim=-1)
    valid = norm > minimum_norm
    denominator = torch.where(valid, norm, torch.ones_like(norm))
    return torch.where(valid[..., None], gradient / denominator[..., None], torch.zeros_like(gradient)), valid


class LocalMarginEnsemble(nn.Module):
    """Two small heads supervised on actual control-interval minimum distances."""
    def __init__(self, obs_dim=25, width=64, temperature=.02, activation='relu'):
        super().__init__()
        if activation not in ['relu', 'silu']:
            raise ValueError('Unknown local-margin activation')
        self.heads = nn.ModuleList([mlp(obs_dim + 3, 1, width) for _ in range(2)])
        if activation == 'silu':
            for head in self.heads:
                head[1], head[3] = nn.SiLU(), nn.SiLU()
        self.activation = activation
        self.temperature = temperature

    def forward(self, obs, action):
        inputs = torch.cat([obs, action], -1)
        return torch.stack([head(inputs).squeeze(-1) for head in self.heads])

    def geometry(self, obs, action, return_diagnostics=False):
        parameter_flags = [p.requires_grad for p in self.parameters()]
        self.requires_grad_(False)
        try:
            with torch.enable_grad():
                local_action = action.detach().requires_grad_(True)
                margins = self(obs.detach(), local_action)
                gradients = torch.stack([torch.autograd.grad(margins[j].sum(), local_action,
                                           retain_graph=j == 0)[0] for j in range(2)])
        finally:
            for parameter, flag in zip(self.parameters(), parameter_flags):
                parameter.requires_grad_(flag)
        units, head_valid = strict_normals(gradients)
        normal, mean_valid = strict_normals(gradients.mean(0))
        agreement = (units[0] * units[1]).sum(-1).clamp(0, 1)
        valid = head_valid.all(0) & mean_valid
        safety_weight = torch.sigmoid(margins.detach().mean(0) / self.temperature)
        q = (safety_weight * agreement * valid).detach()
        normal = normal.detach()
        if return_diagnostics:
            # Legacy factors are diagnostic ONLY; never multiply the active q.
            gnorm = gradients.detach().mean(0).norm(dim=-1)
            return normal, q, dict(safety_weight=safety_weight, agreement=agreement.detach(),
                gradient_valid=valid, gradient_norm=gnorm,
                margin_gradient_factor_if_old_rule=gnorm / (gnorm + .005),
                margin_disagreement_factor_if_old_rule=torch.exp(-margins.detach().std(0, unbiased=False) / .05))
        return normal, q


def reference_variances(weight, reference_variance=.04, rho=.25):
    if reference_variance <= 0 or not 0 < rho <= 1:
        raise ValueError('Invalid directional reference covariance')
    vn = reference_variance * (1. - weight.clamp(0, 1) * (1. - rho))
    vt = (3. * reference_variance - vn) / 2.
    return vn, vt


def old_cost_weight_diagnostics(cost_critic, obs, action, budget=.1,
                                cost_scale=1., temperature=.02):
    """Old-style factor products on the ARM long-horizon cost critic only.

    This is not Std(local margin), not a reconstruction of old navigation
    outcomes, and never the active LearnedDir weight. It permits the required
    same-state/action factor comparison without another behavior policy.
    """
    flags = [p.requires_grad for p in cost_critic.parameters()]
    cost_critic.requires_grad_(False)
    try:
        with torch.enable_grad():
            local_action = action.detach().requires_grad_(True)
            heads = cost_critic(obs.detach(), local_action) / cost_scale
            mean = heads.mean(0)
            gradient = torch.autograd.grad(mean.sum(), local_action)[0]
    finally:
        for parameter, flag in zip(cost_critic.parameters(), flags):
            parameter.requires_grad_(flag)
    margin = mean.detach() - budget / cost_scale
    gradient_factor = gradient.detach().norm(dim=-1)
    gradient_factor = gradient_factor / (gradient_factor + .005)
    numerical_factor = torch.exp(-heads.detach().std(0, unbiased=False) / .05)
    single_sided = torch.sigmoid(margin / temperature)
    gaussian = torch.exp(-.5 * (margin / temperature).square())
    return dict(cost_margin=margin, cost_gradient_factor=gradient_factor,
                cost_ensemble_numeric_factor=numerical_factor,
                single_sided_weight=single_sided, gaussian_band_weight=gaussian,
                old_style_one_sided_product=single_sided * gradient_factor * numerical_factor,
                old_style_gaussian_product=gaussian * gradient_factor * numerical_factor)

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
    def __init__(self, obs_dim=25, width=64, temperature=.02):
        super().__init__()
        self.heads = nn.ModuleList([mlp(obs_dim + 3, 1, width) for _ in range(2)])
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
                legacy_gradient_factor=gnorm / (gnorm + .005),
                legacy_numerical_disagreement_factor=torch.exp(-margins.detach().std(0, unbiased=False) / .05))
        return normal, q


def reference_variances(weight, reference_variance=.04, rho=.25):
    if reference_variance <= 0 or not 0 < rho <= 1:
        raise ValueError('Invalid directional reference covariance')
    vn = reference_variance * (1. - weight.clamp(0, 1) * (1. - rho))
    vt = (3. * reference_variance - vn) / 2.
    return vn, vt

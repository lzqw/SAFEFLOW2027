"""One narrow check of unit normals, paired energy/covariance and stop-gradient."""
import argparse
import json
from pathlib import Path

import torch

from scripts.star.arm_safe_entropy.geometry import LocalMarginEnsemble, strict_normals, reference_variances
from scripts.star.arm_safe_entropy.models import ArmFlowActor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    torch.manual_seed(2027)
    gradients = torch.tensor([[1e-12, 0., 0.], [3., 4., 0.]])
    units, valid = strict_normals(gradients)
    assert not valid[0] and valid[1]
    assert torch.allclose(units[1].norm(), torch.tensor(1.)) and units[0].norm() == 0.
    q = torch.linspace(0, 1, 13)
    vn, vt = reference_variances(q)
    assert torch.all(vn > 0) and torch.all(vt > 0)
    assert torch.allclose(vn + 2 * vt, torch.full_like(vn, .12), atol=1e-7)
    head = LocalMarginEnsemble()
    obs = torch.zeros(8, 25)
    action = torch.randn(8, 3, requires_grad=True)
    normal, weight = head.geometry(obs, action)
    assert not normal.requires_grad and not weight.requires_grad
    flow = ArmFlowActor(25)
    result, energy, clip = flow(obs, geometry=head.geometry)
    energy.mean().backward()
    assert any(p.grad is not None for p in flow.parameters())
    assert all(p.grad is None for p in head.parameters())
    # Initial policy only, at one common state. This is not trained-method evidence.
    flow.eval()
    with torch.no_grad():
        def constant_geometry(x, a):
            n = torch.zeros_like(a); n[..., 0] = 1.
            return n, torch.ones(a.shape[:-1])
        samples, _, _ = flow(torch.zeros(4096, 25), geometry=constant_geometry)
        covariance = torch.cov(samples.T)
    diagonal = covariance.diag().tolist()
    assert .007 < diagonal[0] < .014 and all(.045 < x < .066 for x in diagonal[1:])
    payload = dict(unit_normal_and_zero_gradient=True, positive_covariance_fixed_trace=True,
                   geometry_stop_gradient=True, initial_policy_action_covariance=covariance.tolist(),
                   initial_only_not_trained_mechanism_result=True, new_environment_transitions=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload))


if __name__ == '__main__':
    main()

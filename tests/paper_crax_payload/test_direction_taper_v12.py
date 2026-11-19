import torch

from scripts.paper_crax_payload.direction_development import NormalShrinkActor, outward_loss
from scripts.paper_crax_payload.direction_taper_v12 import DirectionTaperActor, direction_taper
from scripts.star.arm_safe_entropy.endpoint_geometry import EndpointGeometry
from scripts.star.arm_safe_entropy.geometry import LocalMarginEnsemble


def test_direction_only_taper_preserves_exact_sampler_energy_rng_and_loss_boundary():
    torch.set_num_threads(1); torch.manual_seed(2034)
    margin = torch.tensor([-.02, -.01, -.005, 0., .01], requires_grad=True)
    torch.testing.assert_close(direction_taper(margin), torch.tensor([0., 0., .5, 1., 1.]))
    assert not direction_taper(margin).requires_grad
    base = NormalShrinkActor(25, 128, variance=.04, rho=.25)
    candidate = DirectionTaperActor(25, 128, variance=.04, rho=.25)
    candidate.load_state_dict(base.state_dict())
    geometry = EndpointGeometry(LocalMarginEnsemble(25), transitions=100000)
    obs = torch.randn(12,25); noise = torch.randn(4,12,3)
    state = torch.random.get_rng_state()
    reference = base(obs, geometry=geometry, noise_increments=noise, return_reference=True)
    result = candidate(obs, geometry=geometry, noise_increments=noise, return_reference=True)
    assert torch.equal(torch.random.get_rng_state(), state)
    for left, right in zip(reference[:3], result[:3]):
        assert torch.equal(left, right)
    for key in reference[3]:
        assert torch.equal(reference[3][key], result[3][key])
    trace = result[3]
    assert (trace['direction_weights']<=trace['weights']).all()
    assert not trace['direction_weights'].requires_grad
    old = outward_loss(trace['velocities'], trace['normals'], trace['weights'], .25)
    new = outward_loss(trace['velocities'], trace['normals'], trace['direction_weights'], .25)
    assert new <= old
    new.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in candidate.parameters())

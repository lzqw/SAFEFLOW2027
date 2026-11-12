import torch

from scripts.paper_crax_payload.mechanism_sampling import (
    M1_MODES, M2_MODES, covariance_matching_description, common_noise, frozen_sample)
from scripts.star.arm_safe_entropy.endpoint_geometry import StableFlowActor


class LinearMargin(torch.nn.Module):
    temperature = .02

    def forward(self, obs, action):
        value = action[...,0]+.4*action[...,1]-.01
        return torch.stack([value,value])


def test_m1_spectra_declare_only_actual_matching():
    design = covariance_matching_description()
    assert abs(design['iso_high']['trace']-.12)<1e-9
    assert abs(design['directional_full']['trace']-.12)<1e-9
    for mode in ('iso_low','normal_shrink','directional_full'):
        assert abs(design[mode]['normal_variance']-.01)<1e-9
    assert design['normal_shrink']['trace']<design['directional_full']['trace']


def test_common_noise_random_axes_and_spectrum_without_oracle():
    torch.manual_seed(1)
    actor = StableFlowActor(5,width=8).requires_grad_(False)
    observations = torch.zeros(2,5)
    noise,axes = common_noise(actor,observations,draws=4)
    batch = observations.repeat_interleave(4,0)
    outputs = {}
    for mode in M1_MODES+M2_MODES:
        result = frozen_sample(actor,batch,LinearMargin(),mode,noise,axes)
        assert result[0].shape == (8,3)
        assert torch.isfinite(result[1]).all()
        assert torch.all(result[0].abs()<=1)
        outputs[mode] = result
    for mode in M2_MODES:
        assert torch.equal(outputs[mode][3]['vn'],outputs['local'][3]['vn'])
        assert torch.equal(outputs[mode][3]['vt'],outputs['local'][3]['vt'])
    random = outputs['random'][3]['normals']
    assert torch.equal(random[:,0],random[:,-1])
    assert not torch.allclose(random[:,0],outputs['local'][3]['normals'][:,0])
    repeated = frozen_sample(actor,batch,LinearMargin(),'endpoint_pullback',noise,axes)
    assert torch.equal(repeated[0],outputs['endpoint_pullback'][0])
    assert all(not p.requires_grad for p in actor.parameters())

import torch
from scripts.paper_crax_payload.observation_normalizer import ObservationNormalizer


def test_population_statistics_batching_and_frozen_transform_resume():
    data=torch.tensor([[1.,2.,4.],[3.,8.,4.],[7.,5.,4.],[2.,3.,4.]])
    norm=ObservationNormalizer(3,'cpu','running_acme')
    norm.update(data[:2]);norm.update(data[2:])
    torch.testing.assert_close(norm.mean,data.mean(0))
    torch.testing.assert_close(norm.std,data.std(0,unbiased=False).clamp(1e-6,1e6))
    saved=norm.state_dict();clone=ObservationNormalizer(3,'cpu','running_acme');clone.load_state_dict(saved)
    torch.testing.assert_close(norm.transform(data),clone.transform(data))
    assert norm.count==4 and clone.count==4
    raw={'obs':data,'next_obs':data+1.,'cost':torch.ones(4)}
    transformed=norm.batch(raw)
    assert torch.equal(raw['obs'],data) and transformed['cost'] is raw['cost']
    assert norm.count==4 # transform/evaluation is not an adaptation update

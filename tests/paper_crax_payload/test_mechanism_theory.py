import numpy as np
import pytest

from scripts.paper_crax_payload.mechanism_theory import covariance, constrained_logdet, same_covariance_kl


def test_linear_pullback_variance_and_sin_squared():
    tail = np.array([[2., .5], [0., 1.]])
    g = np.array([1., 0.]); b = tail.T@g; n = b/np.linalg.norm(b)
    matrix = covariance(n, .01, .04)
    assert g@tail@matrix@tail.T@g == pytest.approx(b@matrix@b)
    tangent = np.array([-n[1], n[0]])
    theta = .6; wrong = np.cos(theta)*n+np.sin(theta)*tangent
    assert b@covariance(wrong,.01,.04)@b == pytest.approx(np.dot(b,b)*(.01+.03*np.sin(theta)**2))


def test_constrained_logdet_active_and_inactive():
    n = np.array([1., 2., 3.]); n /= np.linalg.norm(n)
    active = constrained_logdet(n, .12, .01)
    assert np.trace(active['covariance']) == pytest.approx(.12)
    assert n@active['covariance']@n == pytest.approx(.01)
    assert np.linalg.eigvalsh(active['covariance']) == pytest.approx([.01,.055,.055])
    assert active['risk_multiplier'] > 0
    inactive = constrained_logdet(n, .12, .09)
    assert inactive['covariance'] == pytest.approx(.04*np.eye(3))
    assert inactive['risk_multiplier'] == 0


def test_same_covariance_gaussian_kl_and_validation():
    assert same_covariance_kl([0.,0.], [1.,2.], np.diag([2.,4.])) == pytest.approx(.75)
    with pytest.raises(ValueError):
        constrained_logdet([0.,0.], .12, .01)

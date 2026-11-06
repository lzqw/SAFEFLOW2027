"""Per-state covariance proxies. Never pool actions from different states."""
import math

import numpy as np


def projected_covariance(covariance, common_normal, variance_floor=1e-12):
    covariance = np.asarray(covariance, dtype=np.float64)
    normal = np.asarray(common_normal, dtype=np.float64)
    result = dict(trace=float(np.trace(covariance)), covariance=covariance.tolist(),
                  vn=None, vt=None, gaussian_proxy_hn=None, gaussian_proxy_ht=None,
                  entropy_variance_floor=variance_floor, entropy_floor_applied=False)
    norm = np.linalg.norm(normal)
    if norm <= 1e-7:
        return result  # Undefined geometry is retained, not assigned an arbitrary axis.
    if not np.isclose(norm, 1., atol=1e-5):
        raise ValueError('Shared projection normal must already have unit length')
    normal = normal / norm  # remove only floating-point roundoff
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    tangent1 = np.cross(normal, axis)
    tangent1 /= np.linalg.norm(tangent1)
    tangent = np.stack([tangent1, np.cross(normal, tangent1)], -1)
    vn = float(normal @ covariance @ normal)
    tangent_cov = tangent.T @ covariance @ tangent
    tangent_eigenvalues = np.linalg.eigvalsh(tangent_cov)
    result.update(vn=vn, vt=float(np.trace(tangent_cov) / 2),
        gaussian_proxy_hn=float(.5 * np.log(2 * math.pi * math.e * max(vn, variance_floor))),
        gaussian_proxy_ht=float(np.log(2 * math.pi * math.e) +
                               .5 * np.log(np.maximum(tangent_eigenvalues, variance_floor)).sum()),
        entropy_floor_applied=bool(vn < variance_floor or (tangent_eigenvalues < variance_floor).any()))
    return result


def endpoint_metrics(actions, common_normal):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 3 or len(actions) < 2:
        raise ValueError('Use at least two actions from ONE fixed state')
    result = projected_covariance(np.cov(actions, rowvar=False, ddof=1), common_normal)
    result['mean_action'] = actions.mean(0).tolist()
    return result


def reference_metrics(normals, vn, vt, common_normal):
    """Integrated reference covariance averaged over paths, before learned drift.

    Inputs are [actions, time_steps, ...]. Unit flow duration and equal dt imply
    sum_k D_k*dt = mean_k D_k. This is not a terminal-policy covariance.
    Project onto the SAME common safety normal used for all endpoint policies.
    """
    normals, vn, vt = map(lambda x: np.asarray(x, dtype=np.float64), (normals, vn, vt))
    covariance = vt[..., None, None] * np.eye(3) + (vn-vt)[..., None, None] * (
        normals[..., :, None] * normals[..., None, :])
    return projected_covariance(covariance.mean(axis=(0, 1)), common_normal)

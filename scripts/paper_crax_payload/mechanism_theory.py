"""Small, reproducible M4 experiments, not evidence about real robot safety.

Only synthetic Gaussian draws and linear maps are used.  No simulator,
checkpoint, learner, or physical device is imported by this module.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def covariance(normal, normal_variance, tangent_variance):
    normal = np.asarray(normal, dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm <= 0 or min(normal_variance, tangent_variance) <= 0:
        raise ValueError('A nonzero axis and positive variances are required')
    normal = normal / norm
    return tangent_variance * np.eye(len(normal)) + (normal_variance - tangent_variance) * np.outer(normal, normal)


def constrained_logdet(normal, trace, variance_cap):
    """Exact optimizer: max logdet(D), tr(D)=trace, n^T D n<=cap.

    `normal` is unit-normalized internally: cap is variance on that unit axis,
    not on an unnormalized risk gradient.  For risk gradient g and cap c,
    callers must pass variance_cap=c/||g||².
    """
    normal = np.asarray(normal, dtype=np.float64)
    d = len(normal)
    if d < 2 or trace <= 0 or variance_cap <= 0 or np.linalg.norm(normal) <= 0:
        raise ValueError('A feasible positive trace/cap and nonzero d>=2 normal are required')
    normal = normal / np.linalg.norm(normal)
    vn = min(variance_cap, trace / d)
    vt = (trace - vn) / (d - 1)
    matrix = covariance(normal, vn, vt)
    return dict(covariance=matrix, vn=vn, vt=vt, constraint_active=variance_cap < trace / d,
                trace_multiplier=1 / vt, risk_multiplier=max(0., 1 / vn - 1 / vt))


def same_covariance_kl(mean0, mean1, matrix):
    delta = np.asarray(mean0, dtype=float) - np.asarray(mean1, dtype=float)
    return .5 * float(delta @ np.linalg.solve(matrix, delta))


def run(seed=20261112, draws=200000, feasible_trials=20000):
    if draws < 1000 or feasible_trials < 100:
        raise ValueError('Use enough synthetic samples to resolve the configured tolerances')
    rng = np.random.default_rng(seed)
    # A nonorthogonal deterministic tail: direct and pulled-back axes differ.
    tail = np.array([[1., .8, -.3], [.1, .6, .2], [.2, -.4, 1.1]])
    terminal_gradient = np.array([.3, -.7, .8]); terminal_gradient /= np.linalg.norm(terminal_gradient)
    pulled_gradient = tail.T @ terminal_gradient
    pulled_axis = pulled_gradient / np.linalg.norm(pulled_gradient)
    vn, vt, dt = .01, .055, .25
    matrix = covariance(pulled_axis, vn, vt)
    analytic_endpoint = float(dt * terminal_gradient @ tail @ matrix @ tail.T @ terminal_gradient)
    analytic_pullback = float(dt * pulled_gradient @ matrix @ pulled_gradient)
    noise = rng.multivariate_normal(np.zeros(3), matrix, size=draws)
    scalar_risk_increment = np.sqrt(dt) * noise @ pulled_gradient
    empirical = float(np.var(scalar_risk_increment, ddof=1))
    endpoint = dict(tail=tail.tolist(), terminal_gradient=terminal_gradient.tolist(),
        pullback_gradient=pulled_gradient.tolist(), covariance=matrix.tolist(), dt=dt,
        endpoint_formula=analytic_endpoint, pullback_formula=analytic_pullback,
        empirical_variance=empirical, formula_absolute_error=abs(analytic_endpoint-analytic_pullback),
        empirical_relative_error=abs(empirical/analytic_endpoint-1),
        passed=bool(np.isclose(analytic_endpoint, analytic_pullback, rtol=1e-12) and abs(empirical/analytic_endpoint-1)<.02))
    # Construct an orthonormal tangent to the fixed true pulled-back axis.
    tangent = np.cross(pulled_axis, np.eye(3)[np.argmin(np.abs(pulled_axis))])
    tangent /= np.linalg.norm(tangent)
    misalignment = []
    for angle_deg in (0, 15, 30, 45, 60, 90):
        theta = np.deg2rad(angle_deg)
        proposed_axis = np.cos(theta)*pulled_axis + np.sin(theta)*tangent
        direct = float(pulled_gradient @ covariance(proposed_axis, vn, vt) @ pulled_gradient)
        formula = float(np.dot(pulled_gradient, pulled_gradient)*(vn+(vt-vn)*np.sin(theta)**2))
        misalignment.append(dict(angle_degrees=angle_deg, matrix_variance=direct,
            sin_squared_formula=formula, absolute_error=abs(direct-formula),
            passed=bool(np.isclose(direct, formula, atol=1e-14))))
    optimizers = []
    for cap in (.01, .06):
        solution = constrained_logdet(pulled_axis, .12, cap)
        optimum = solution['covariance']; optimum_logdet = float(np.linalg.slogdet(optimum)[1])
        # Uniformly generate SPD shapes, normalize trace, and retain only
        # actually feasible shapes.  This is a numerical cross-check; KKT
        # stationarity/complementarity are reported separately and exactly.
        feasible_logdets = []
        for _ in range(feasible_trials):
            shape = rng.normal(size=(3, 3)); trial = shape @ shape.T
            trial *= .12 / np.trace(trial)
            if pulled_axis @ trial @ pulled_axis <= cap:
                feasible_logdets.append(float(np.linalg.slogdet(trial)[1]))
        if not feasible_logdets:
            raise RuntimeError('No feasible numerical cross-check samples; do not mark this a pass')
        stationarity = np.linalg.inv(optimum) - solution['trace_multiplier']*np.eye(3) - solution['risk_multiplier']*np.outer(pulled_axis, pulled_axis)
        complementarity = solution['risk_multiplier']*(float(pulled_axis@optimum@pulled_axis)-cap)
        row = dict(cap=cap, trace=.12, vn=solution['vn'], vt=solution['vt'],
            active=solution['constraint_active'], optimum_logdet=optimum_logdet,
            feasible_random_matrices=len(feasible_logdets), max_feasible_sample_logdet=max(feasible_logdets),
            largest_numerical_advantage_over_optimum=max(feasible_logdets)-optimum_logdet,
            stationarity_max_abs=float(np.abs(stationarity).max()), complementarity=float(complementarity),
            covariance=optimum.tolist())
        row['passed'] = bool(row['stationarity_max_abs']<1e-10 and abs(complementarity)<1e-10
            and row['largest_numerical_advantage_over_optimum']<1e-10)
        optimizers.append(row)
    mean0 = np.array([.1, -.2, .05]); mean1 = np.array([-.03, .08, .2])
    samples = rng.multivariate_normal(mean0, matrix, size=draws)
    inverse = np.linalg.inv(matrix)
    delta0 = samples-mean0; delta1 = samples-mean1
    # log p0 - log p1; identical normalizer and logdet terms cancel.
    logratio = .5*(np.einsum('ni,ij,nj->n', delta1, inverse, delta1)
                     -np.einsum('ni,ij,nj->n', delta0, inverse, delta0))
    exact = same_covariance_kl(mean0, mean1, matrix)
    empirical_kl = float(logratio.mean()); se = float(logratio.std(ddof=1)/np.sqrt(draws))
    kl = dict(mean0=mean0.tolist(), mean1=mean1.tolist(), covariance=matrix.tolist(),
        analytic_kl=exact, empirical_mean_logratio=empirical_kl, empirical_standard_error=se,
        empirical_z_error=abs(empirical_kl-exact)/se,
        passed=bool(abs(empirical_kl-exact)<=6*se))
    return dict(status='completed', seed=seed, synthetic_draws_per_gaussian_experiment=draws,
        synthetic_total_gaussian_draws=2*draws, random_spd_proposals=2*feasible_trials,
        new_training_transitions=0, simulator_queries=0, real_interactions=0,
        endpoint_pullback=endpoint, direction_misalignment=misalignment,
        constrained_logdet=optimizers, same_covariance_relative_entropy=kl,
        all_checks_passed=bool(endpoint['passed'] and all(r['passed'] for r in misalignment)
            and all(r['passed'] for r in optimizers) and kl['passed']),
        interpretation='Linear Gaussian identities and constrained Gaussian entropy allocation only;not a theorem of arbitrary nonlinear dynamics safety or real-robot generalization')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261112)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite earlier numerical evidence')
    result = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','all_checks_passed','simulator_queries','new_training_transitions')}))
    if not result['all_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

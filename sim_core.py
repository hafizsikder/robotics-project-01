"""Shared world, models and filters for the Topic A experiments (02_*, 03_*, ...).

2D unicycle robot, known landmarks, range-bearing sensor. EKF and bootstrap
particle filter, both assuming the nominal noise Q_NOM.
"""

import numpy as np

# ------------------------------- settings ------------------------------------
DT = 0.1
STEPS = 600
SHIFT_STEP = STEPS // 2
SHIFT_STD_FACTOR = 2.0       # true motion-noise std multiplier after the shift
M_RUNS = 100
N_PARTICLES = 300
LANDMARKS = np.array([[5, 10], [-5, 10], [-8, 0], [0, -3], [8, 2], [3, 15]], float)
MAX_RANGE = 8.0
MIN_RANGE = 0.5            # real range sensors have a blind zone; also avoids
                           # the EKF bearing-Jacobian singularity (1/r^2) at r -> 0
Q_NOM = np.diag([0.1, np.deg2rad(5)]) ** 2      # assumed (v, omega) noise
R = np.diag([0.3, np.deg2rad(3)]) ** 2           # range-bearing noise
R_INV = np.linalg.inv(R)
STATE_DIM = 3
MEAS_DIM = 2
SEED = 2026


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def motion(x, u):
    v, w = u[..., 0], u[..., 1]
    th = x[..., 2]
    out = np.array(x, dtype=float, copy=True)
    out[..., 0] += v * np.cos(th) * DT
    out[..., 1] += v * np.sin(th) * DT
    out[..., 2] = wrap(th + w * DT)
    return out


def observe(x, lm):
    d = lm - x[..., :2]
    r = np.hypot(d[..., 0], d[..., 1])
    b = wrap(np.arctan2(d[..., 1], d[..., 0]) - x[..., 2])
    return np.stack([r, b], axis=-1)


def control(t):
    return np.array([1.0, 0.1 + 0.05 * np.sin(0.01 * t)])


def q_true(t, shift, shift_factor=SHIFT_STD_FACTOR):
    """True motion-noise covariance at step t."""
    factor = shift_factor if (shift and t >= SHIFT_STEP) else 1.0
    return Q_NOM * factor ** 2


def simulate_step(truth, t, shift, rng, R=R, shift_factor=SHIFT_STD_FACTOR):
    """Advance ground truth one step and return (truth, [(z, landmark), ...])."""
    u = control(t)
    truth = motion(truth, u + rng.multivariate_normal([0, 0], q_true(t, shift, shift_factor)))
    zs = []
    for lm in LANDMARKS:
        z_true = observe(truth, lm)
        if MIN_RANGE <= z_true[0] <= MAX_RANGE:
            zs.append((z_true + rng.multivariate_normal([0, 0], R), lm))
    return truth, zs


# ---------------------------------- EKF --------------------------------------
def ekf_predict(mu, S, u):
    G, V = ekf_jacobians(mu, u)
    return motion(mu, u), G @ S @ G.T + V @ Q_NOM @ V.T


def ekf_jacobians(mu, u):
    """Motion Jacobians G = df/dx, V = df/du at the estimate."""
    v, th = u[0], mu[2]
    G = np.array([[1, 0, -v * np.sin(th) * DT],
                  [0, 1, v * np.cos(th) * DT],
                  [0, 0, 1]])
    V = np.array([[np.cos(th) * DT, 0],
                  [np.sin(th) * DT, 0],
                  [0, DT]])
    return G, V


def ekf_update(mu, S, z, lm, R=R, return_gain=False):
    """Sequential EKF update. Returns (mu, S, nis) where nis = nu^T S_nu^-1 nu.

    With return_gain=True also returns (K, H, S_nu) for error-covariance analysis.
    """
    dx, dy = lm - mu[:2]
    q = dx ** 2 + dy ** 2
    r = np.sqrt(q)
    H = np.array([[-dx / r, -dy / r, 0],
                  [dy / q, -dx / q, -1]])
    innov = z - observe(mu, lm)
    innov[1] = wrap(innov[1])
    S_nu = H @ S @ H.T + R
    S_nu_inv = np.linalg.inv(S_nu)
    K = S @ H.T @ S_nu_inv
    mu = mu + K @ innov
    mu[2] = wrap(mu[2])
    I_KH = np.eye(3) - K @ H
    nis = float(innov @ S_nu_inv @ innov)
    S_new = I_KH @ S @ I_KH.T + K @ R @ K.T      # Joseph form: stays PSD
    if return_gain:
        return mu, S_new, nis, (K, H, S_nu)
    return mu, S_new, nis


# ---------------------------- particle filter --------------------------------
def pf_predict(P, u, rng):
    return motion(P, u + rng.multivariate_normal([0, 0], Q_NOM, len(P)))


def pf_correct(P, w, zs, rng):
    logw = np.log(np.maximum(w, 1e-300))    # underflowed weights stay finite
    for z, lm in zs:
        innov = z - observe(P, lm)
        innov[:, 1] = wrap(innov[:, 1])
        logw += -0.5 * np.einsum("ni,ij,nj->n", innov, R_INV, innov)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    if 1 / np.sum(w ** 2) < len(P) / 2:
        pos = (rng.random() + np.arange(len(P))) / len(P)
        idx = np.minimum(np.searchsorted(np.cumsum(w), pos), len(P) - 1)
        P, w = P[idx], np.full(len(P), 1 / len(P))
    return P, w


def pf_step(P, w, u, zs, rng):
    return pf_correct(pf_predict(P, u, rng), w, zs, rng)


def pf_predictive_nis(P, w, zs):
    """Joint NIS of all measurements this step under the PF's predictive cloud.

    Stacks all k observed landmarks: z_hat ~ observe(P_pred) + noise, with
    S_nu = weighted sample cov of observe(P_pred) + blockdiag(R).
    Returns (nis, dof) or (None, 0) if nothing was observed.
    """
    if not zs:
        return None, 0
    Zp = np.concatenate([observe(P, lm) for _, lm in zs], axis=1)   # (N, 2k)
    z = np.concatenate([zz for zz, _ in zs])
    k = len(zs)
    b = slice(1, 2 * k, 2)                                           # bearing columns
    zbar = w @ Zp
    zbar[b] = np.arctan2(w @ np.sin(Zp[:, b]), w @ np.cos(Zp[:, b]))
    D = Zp - zbar
    D[:, b] = wrap(D[:, b])
    S_nu = (w[:, None] * D).T @ D + np.kron(np.eye(k), R)
    nu = z - zbar
    nu[b] = wrap(nu[b])
    return float(nu @ np.linalg.solve(S_nu, nu)), 2 * k


def pf_moments(P, w):
    th = np.arctan2(w @ np.sin(P[:, 2]), w @ np.cos(P[:, 2]))
    mean = np.array([*(w @ P[:, :2]), th])
    d = P - mean
    d[:, 2] = wrap(d[:, 2])
    cov = (w[:, None] * d).T @ d + 1e-9 * np.eye(3)
    return mean, cov


def nees(est, cov, truth):
    e = truth - est
    e[2] = wrap(e[2])
    return float(e @ np.linalg.solve(cov, e))

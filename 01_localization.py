"""Lesson 1: robot localization with an EKF and a particle filter.

A differential-drive robot drives a loop among known landmarks. It knows its
commanded velocities (noisy odometry) and measures range + bearing to landmarks
(noisy sensor). Two filters estimate its pose (x, y, theta):

  * EKF             - keeps one Gaussian (mean + covariance), linearizes the models
  * Particle filter - keeps N weighted samples, no Gaussian assumption

Run:  uv run python 01_localization.py
Output: 01_localization.png  (trajectories, uncertainty ellipses, error plot)

Reference: Thrun, Burgard & Fox, "Probabilistic Robotics", ch. 5-8.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

rng = np.random.default_rng(0)

# ---------------- world & noise settings (try changing these) ----------------
DT = 0.1                     # time step [s]
STEPS = 600
LANDMARKS = np.array([[5, 10], [-5, 10], [-8, 0], [0, -3], [8, 2], [3, 15]], float)
MAX_RANGE = 8.0              # sensor only sees landmarks within this range [m]

MOTION_NOISE = np.diag([0.1, np.deg2rad(5)]) ** 2      # std of (v, omega)
SENSOR_NOISE = np.diag([0.3, np.deg2rad(3)]) ** 2      # std of (range, bearing)
N_PARTICLES = 500


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def motion(x, u):
    """Velocity motion model. x=[x,y,th], u=[v,w]. Works on (3,) or (N,3)."""
    v, w = u[..., 0], u[..., 1]
    th = x[..., 2]
    out = np.array(x, dtype=float, copy=True)
    out[..., 0] += v * np.cos(th) * DT
    out[..., 1] += v * np.sin(th) * DT
    out[..., 2] = wrap(th + w * DT)
    return out


def observe(x, lm):
    """Expected range/bearing from pose x to landmark lm."""
    d = lm - x[..., :2]
    r = np.hypot(d[..., 0], d[..., 1])
    b = wrap(np.arctan2(d[..., 1], d[..., 0]) - x[..., 2])
    return np.stack([r, b], axis=-1)


def control(t):
    return np.array([1.0, 0.1 + 0.05 * np.sin(0.01 * t)])  # gentle loop


# ------------------------------- EKF ----------------------------------------
def ekf_predict(mu, S, u):
    v, th = u[0], mu[2]
    G = np.array([[1, 0, -v * np.sin(th) * DT],
                  [0, 1,  v * np.cos(th) * DT],
                  [0, 0, 1]])
    V = np.array([[np.cos(th) * DT, 0],
                  [np.sin(th) * DT, 0],
                  [0, DT]])
    return motion(mu, u), G @ S @ G.T + V @ MOTION_NOISE @ V.T


def ekf_update(mu, S, z, lm):
    dx, dy = lm - mu[:2]
    q = dx ** 2 + dy ** 2
    r = np.sqrt(q)
    H = np.array([[-dx / r, -dy / r, 0],
                  [dy / q, -dx / q, -1]])
    innov = z - observe(mu, lm)
    innov[1] = wrap(innov[1])
    K = S @ H.T @ np.linalg.inv(H @ S @ H.T + SENSOR_NOISE)
    mu = mu + K @ innov
    mu[2] = wrap(mu[2])
    return mu, (np.eye(3) - K @ H) @ S


# --------------------------- particle filter ---------------------------------
def pf_predict(P, u):
    noisy_u = u + rng.multivariate_normal([0, 0], MOTION_NOISE, len(P))
    return motion(P, noisy_u)


def pf_weight(P, w, z, lm):
    innov = z - observe(P, lm)
    innov[:, 1] = wrap(innov[:, 1])
    inv = np.linalg.inv(SENSOR_NOISE)
    w = w * np.exp(-0.5 * np.einsum("ni,ij,nj->n", innov, inv, innov))
    return w / w.sum() if w.sum() > 0 else np.full(len(P), 1 / len(P))


def low_variance_resample(P, w):
    n = len(P)
    positions = (rng.random() + np.arange(n)) / n
    idx = np.minimum(np.searchsorted(np.cumsum(w), positions), n - 1)
    return P[idx], np.full(n, 1 / n)


def pf_estimate(P, w):
    xy = w @ P[:, :2]
    th = np.arctan2(w @ np.sin(P[:, 2]), w @ np.cos(P[:, 2]))
    return np.array([*xy, th])


# ------------------------------- simulation ----------------------------------
def cov_ellipse(ax, mu, S, color):
    vals, vecs = np.linalg.eigh(S[:2, :2])
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * 2 * np.sqrt(np.maximum(vals[::-1], 0))  # 2-sigma
    ax.add_patch(Ellipse(mu[:2], w, h, angle=ang, fill=False, color=color, lw=0.8))


def main():
    truth = np.zeros(3)
    dead = truth.copy()
    mu, S = truth.copy(), np.eye(3) * 0.01
    P = truth + rng.normal(0, 0.1, (N_PARTICLES, 3))
    w = np.full(N_PARTICLES, 1 / N_PARTICLES)

    hist = {k: [] for k in ("truth", "dead", "ekf", "pf")}
    ellipses = []

    for t in range(STEPS):
        u = control(t)
        truth = motion(truth, u + rng.multivariate_normal([0, 0], MOTION_NOISE))
        dead = motion(dead, u)                      # odometry only, no sensor

        mu, S = ekf_predict(mu, S, u)
        P = pf_predict(P, u)

        for lm in LANDMARKS:
            z_true = observe(truth, lm)
            if z_true[0] > MAX_RANGE:
                continue
            z = z_true + rng.multivariate_normal([0, 0], SENSOR_NOISE)
            mu, S = ekf_update(mu, S, z, lm)
            w = pf_weight(P, w, z, lm)

        if 1 / np.sum(w ** 2) < N_PARTICLES / 2:   # effective sample size check
            P, w = low_variance_resample(P, w)

        hist["truth"].append(truth)
        hist["dead"].append(dead)
        hist["ekf"].append(mu.copy())
        hist["pf"].append(pf_estimate(P, w))
        if t % 40 == 0:
            ellipses.append((mu.copy(), S.copy()))

    H = {k: np.array(v) for k, v in hist.items()}
    err = {k: np.hypot(*(H[k][:, :2] - H["truth"][:, :2]).T) for k in ("dead", "ekf", "pf")}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    a1.plot(*H["truth"][:, :2].T, "k-", lw=2, label="ground truth")
    a1.plot(*H["dead"][:, :2].T, "r--", label="dead reckoning")
    a1.plot(*H["ekf"][:, :2].T, "b-", label="EKF")
    a1.plot(*H["pf"][:, :2].T, "g-", label="particle filter")
    a1.scatter(P[:, 0], P[:, 1], s=2, c="g", alpha=0.3)
    a1.scatter(*LANDMARKS.T, marker="*", s=200, c="orange", edgecolors="k", label="landmarks")
    for m, s in ellipses:
        cov_ellipse(a1, m, s, "b")
    a1.set_aspect("equal"); a1.legend(); a1.set_title("Trajectories (blue ellipses = EKF 2σ)")

    tt = np.arange(STEPS) * DT
    for k, c in (("dead", "r"), ("ekf", "b"), ("pf", "g")):
        a2.plot(tt, err[k], c, label=f"{k}  (mean {err[k].mean():.2f} m)")
    a2.set_xlabel("time [s]"); a2.set_ylabel("position error [m]")
    a2.legend(); a2.set_title("Localization error")

    fig.tight_layout()
    fig.savefig("01_localization.png", dpi=120)
    print({k: round(float(v.mean()), 3) for k, v in err.items()}, "-> saved 01_localization.png")


if __name__ == "__main__":
    main()

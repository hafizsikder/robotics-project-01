"""Shared machinery for the online-calibration experiments (06_* onward).

* EKF score simulation (NEES per step), with on-disk caching
* a causal driver that replays any calibrator on stored score sequences with
  labels every L steps delivered D steps late
* evaluation metrics (coverage, recovery, region volume vs oracle)

05_aci_sparse_labels.py keeps its own copies of these helpers and is left
unchanged as the record of method v1.
"""

import os
from multiprocessing import Pool

import numpy as np

from sim_core import (DT, SHIFT_STEP, STEPS, control, ekf_predict, ekf_update, nees,
                      simulate_step)

ALPHA = 0.05
SMOOTH = 20                  # 2 s trailing average for coverage curves
RECOVERY_LEVEL = 0.92


# ------------------------------ simulation -----------------------------------
def ekf_scores(args):
    seed, shift, shift_factor = args
    rng = np.random.default_rng(seed)
    truth = np.zeros(3)
    mu, S = truth.copy(), np.eye(3) * 0.01
    s = np.zeros(STEPS)
    for t in range(STEPS):
        u = control(t)
        truth, zs = simulate_step(truth, t, shift, rng, shift_factor=shift_factor)
        mu, S = ekf_predict(mu, S, u)
        for z, lm in zs:
            mu, S, _ = ekf_update(mu, S, z, lm)
        s[t] = nees(mu, S, truth)
    return s


def cached_scores(path, seeds, shift, shift_factor):
    """NEES sequences (len(seeds), STEPS), simulated once and cached."""
    if os.path.exists(path):
        d = np.load(path)
        if d["scores"].shape[0] == len(seeds) and float(d["shift_factor"]) == shift_factor:
            return d["scores"]
    with Pool(4) as pool:
        out = np.array(pool.map(ekf_scores, [(s, shift, shift_factor) for s in seeds], chunksize=10))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, scores=out, shift_factor=shift_factor)
    return out


# --------------------------------- driver ------------------------------------
def drive(cal, scores, L, D):
    """Replay a calibrator causally.

    At each step t the calibrator first reports tau_t (region {NEES <= tau_t});
    then, if a label for step k = t - D exists (k % L == 0), it receives the
    score s_k together with the threshold tau_k it had reported at step k.
    Returns tau (M, T) and the boolean mask of labelled steps.
    """
    M, T = scores.shape
    tau = np.zeros((M, T))
    labelled = np.zeros(T, dtype=bool)
    labelled[::L] = True
    for t in range(T):
        tau[:, t] = cal.tau()
        k = t - D
        if k >= 0 and labelled[k]:
            cal.update(scores[:, k], tau[:, k])
    return tau, labelled


# -------------------------------- metrics ------------------------------------
def smooth(x, w=SMOOTH):
    """Trailing (causal) moving average."""
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    lo = np.maximum(0, np.arange(len(x)) - w + 1)
    return (c[1:] - c[lo]) / (np.arange(len(x)) + 1 - lo)


def recovery_time(sm):
    """Seconds after the shift until trailing coverage is >= RECOVERY_LEVEL and
    stays there. 0 if it never drops below; None if it never recovers."""
    post = sm[SHIFT_STEP:]
    below = np.where(post < RECOVERY_LEVEL)[0]
    if below.size == 0:
        return 0.0
    if below[-1] == post.size - 1:
        return None
    return float((below[-1] + 1) * DT)


def metrics(scores, tau, oracle, labelled=None):
    cov = scores <= tau
    curve = cov.mean(axis=0)
    pre, post = slice(SMOOTH, SHIFT_STEP), slice(SHIFT_STEP, None)
    vol = (tau / oracle[None, :]) ** 1.5           # ellipsoid volume ratio (3D), same P
    out = {
        "cov_pre": float(cov[:, pre].mean()),
        "cov_post": float(cov[:, post].mean()),
        "cov_late": float(cov[:, SHIFT_STEP + 150:].mean()),          # 45-60 s
        "recovery_s": recovery_time(smooth(curve)),
        # mean |coverage - target| over time (after filter warm-up): the tuning objective
        "cov_tracking_err": float(np.abs(curve[SMOOTH:] - (1 - ALPHA)).mean()),
        "vol_ratio_pre_median": float(np.median(vol[:, pre])),
        "vol_ratio_post_median": float(np.median(vol[:, post])),
    }
    if labelled is not None:
        out["miscov_on_labelled"] = float(1 - cov[:, labelled].mean())
    return out, curve

"""Experiment 5 (method v1): online conformal calibration with sparse, delayed labels.

The EKF's pose region at step t is the ellipsoid
    C_t = { x : (x - mu_t)^T P_t^-1 (x - mu_t) <= tau_t }          (heading wrapped)
whose score s_t = NEES_t is the Mahalanobis nonconformity score (the same score
as the "cgkf" variant of Weisman et al., arXiv 2609.27506).

Calibrators for tau_t
  nominal     tau = chi2_3(0.95), the filter's own claim
  offline CP  tau = empirical 95% quantile of scores pooled over separate
              nominal (no-shift) calibration runs; exchangeable split CP,
              approximating Weisman et al. 2026
  ACI (ours)  online quantile tracking on log tau (Gibbs & Candes 2021;
              Angelopoulos, Candes & Tibshirani 2023):
                  when the label for step k arrives (at step k + D):
                  log tau <- log tau + eta * (1{s_k > tau_k} - alpha)
              where tau_k is the threshold that was in force at step k.
              Labels (true poses) exist only every L steps, delivered D steps late.

Guarantee (quantile tracking, holds for ANY score sequence): over the n labelled
steps, |miscoverage - alpha| <= (|log tau_max - log tau_0| + eta) / (eta * n).
It says nothing about the unlabelled steps; measuring that gap is the point here.

The filter does not depend on the calibrator, so the EKF is simulated once and
all (L, D, eta) configurations are replayed on the stored score sequences.

Run:    uv run python 05_aci_sparse_labels.py    (~2 min on 4 cores)
Output: 05_aci_sparse_labels.png, 05_aci_sparse_labels.json, cache/05_scores.npz
"""

import itertools
import json
import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from sim_core import (DT, SEED, SHIFT_STEP, STATE_DIM, STEPS, control, ekf_predict,
                      ekf_update, nees, simulate_step)

ALPHA = 0.05
M_TEST = 200                 # shifted test runs
M_CAL = 100                  # nominal calibration runs for offline CP (disjoint seeds)
CAL_SEED_OFFSET = 10_000
LABEL_INTERVALS = [1, 2, 5, 10, 20, 50, 100]   # a true pose every L steps (L*0.1 s)
DELAYS = [0, 20]                               # label latency [steps]
ETAS = [0.01, 0.05, 0.2]                       # ACI step size on log tau
RECOVERY_LEVEL = 0.92        # coverage level counted as "recovered"
SMOOTH = 20                  # 2 s moving average for coverage curves
CACHE = os.path.join("cache", "05_scores.npz")


# ------------------------------ simulation -----------------------------------
def ekf_scores(args):
    seed, shift = args
    rng = np.random.default_rng(seed)
    truth = np.zeros(3)
    mu, S = truth.copy(), np.eye(3) * 0.01
    s = np.zeros(STEPS)
    for t in range(STEPS):
        u = control(t)
        truth, zs = simulate_step(truth, t, shift, rng)
        mu, S = ekf_predict(mu, S, u)
        for z, lm in zs:
            mu, S, _ = ekf_update(mu, S, z, lm)
        s[t] = nees(mu, S, truth)
    return s


def load_scores():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        if d["test"].shape[0] == M_TEST and d["cal"].shape[0] == M_CAL:
            return d["test"], d["cal"]
    jobs = [(SEED + i, True) for i in range(M_TEST)] + \
           [(SEED + CAL_SEED_OFFSET + i, False) for i in range(M_CAL)]
    with Pool(4) as pool:
        out = np.array(pool.map(ekf_scores, jobs, chunksize=10))
    test, cal = out[:M_TEST], out[M_TEST:]
    os.makedirs("cache", exist_ok=True)
    np.savez_compressed(CACHE, test=test, cal=cal)
    return test, cal


# ------------------------------ calibrators ----------------------------------
def aci(scores, L, D, eta, tau0):
    """Online quantile tracking with labels every L steps, delayed by D steps.
    Returns tau_t (M, T) used at each step, and a mask of labelled steps."""
    M, T = scores.shape
    log_tau = np.full(M, np.log(tau0))
    used = np.zeros((M, T))
    labelled = np.zeros(T, dtype=bool)
    labelled[::L] = True
    for t in range(T):
        used[:, t] = log_tau                            # region reported at step t ...
        k = t - D
        if k >= 0 and labelled[k]:                      # ... then the label for step k arrives
            err = scores[:, k] > np.exp(used[:, k])     # (k == t when D == 0: no look-ahead)
            log_tau += eta * (err - ALPHA)
    return np.exp(used), labelled


def smooth(x, w=SMOOTH):
    """Trailing (causal) moving average; no mixing across the shift or edge effects."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x, dtype=float)
    for t in range(len(x)):
        lo = max(0, t - w + 1)
        out[t] = (c[t + 1] - c[lo]) / (t + 1 - lo)
    return out


def recovery_time(sm):
    """Seconds after the shift until the trailing coverage is >= RECOVERY_LEVEL
    and stays there for the rest of the run. 0 if it never drops below; None if never recovers."""
    post = sm[SHIFT_STEP:]
    below = np.where(post < RECOVERY_LEVEL)[0]
    if below.size == 0:
        return 0.0
    if below[-1] == post.size - 1:
        return None
    return float((below[-1] + 1) * DT)


def metrics(scores, tau, oracle, labelled=None):
    cov = scores <= tau                                  # (M, T)
    curve = cov.mean(axis=0)
    out = {
        "cov_pre": float(cov[:, SMOOTH:SHIFT_STEP].mean()),
        "cov_post": float(cov[:, SHIFT_STEP:].mean()),
        "cov_late": float(cov[:, SHIFT_STEP + 150:].mean()),     # 45-60 s
        "recovery_s": recovery_time(smooth(curve)),
        # Region volume relative to the per-step oracle threshold (same P):
        "vol_ratio_post_median": float(np.median((tau[:, SHIFT_STEP:] / oracle[SHIFT_STEP:]) ** 1.5)),
    }
    if labelled is not None:
        out["miscov_on_labelled"] = float(1 - cov[:, labelled].mean())
        out["miscov_all_steps"] = float(1 - cov.mean())
    return out, curve


# ---------------------------------- main -------------------------------------
def main():
    test, cal = load_scores()
    print(f"scores: test {test.shape}, cal {cal.shape}")
    tau_nom = chi2.ppf(1 - ALPHA, STATE_DIM)
    tau_off = float(np.quantile(cal[:, SMOOTH:], 1 - ALPHA))       # skip filter warm-up
    oracle = np.quantile(test, 1 - ALPHA, axis=0)                  # unattainable reference
    M = test.shape[0]

    results, curves = {}, {}
    results["nominal"], curves["nominal"] = metrics(test, np.full_like(test, tau_nom), oracle)
    results["offline_CP"], curves["offline_CP"] = metrics(test, np.full_like(test, tau_off), oracle)
    for L, D, eta in itertools.product(LABEL_INTERVALS, DELAYS, ETAS):
        tau, lab = aci(test, L, D, eta, tau_nom)
        key = f"ACI L={L} D={D} eta={eta}"
        results[key], curves[key] = metrics(test, tau, oracle, lab)
    summary = {"tau_nominal": tau_nom, "tau_offline_CP": tau_off,
               "oracle_tau_post_median": float(np.median(oracle[SHIFT_STEP:])),
               "M_test": M, "M_cal": M_CAL, "alpha": ALPHA, "results": results}
    with open("05_aci_sparse_labels.json", "w") as fh:
        json.dump(summary, fh, indent=1)

    # ---------------------------------- plots ----------------------------------
    tt = np.arange(STEPS) * DT
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    a = ax[0, 0]
    a.plot(tt, smooth(curves["nominal"]), "k", lw=1.2, label="nominal χ² region")
    a.plot(tt, smooth(curves["offline_CP"]), color="gray", lw=1.2, ls="--", label="offline CP (nominal cal.)")
    cm = plt.cm.viridis
    for i, L in enumerate([1, 5, 20, 100]):
        a.plot(tt, smooth(curves[f"ACI L={L} D=0 eta=0.05"]), color=cm(i / 3), lw=1.2,
               label=f"ACI, label every {L * DT:g} s")
    a.axhline(1 - ALPHA, color="r", ls=":", lw=1)
    a.axvline(SHIFT_STEP * DT, color="r", ls="--", lw=1)
    a.set_ylim(0.3, 1.01); a.set_xlabel("time [s]"); a.set_ylabel("coverage (2 s avg, 200 runs)")
    a.set_title("(a) coverage over time, η = 0.05, no delay")
    a.legend(fontsize=7, loc="lower left")

    xs = np.array(LABEL_INTERVALS) * DT
    styles = {0: "-", 20: "--"}
    ecol = {0.01: "tab:orange", 0.05: "tab:blue", 0.2: "tab:purple"}
    for (D, eta) in itertools.product(DELAYS, ETAS):
        rows = [results[f"ACI L={L} D={D} eta={eta}"] for L in LABEL_INTERVALS]
        lab = f"η={eta}, delay {D * DT:g} s"
        rec = [r["recovery_s"] if r["recovery_s"] is not None else np.nan for r in rows]
        ax[0, 1].plot(xs, rec, styles[D], color=ecol[eta], marker="o", ms=3, label=lab)
        ax[1, 0].plot(xs, [r["cov_post"] for r in rows], styles[D], color=ecol[eta], marker="o", ms=3, label=lab)
        ax[1, 1].plot(xs, [r["vol_ratio_post_median"] for r in rows], styles[D], color=ecol[eta],
                      marker="o", ms=3, label=lab)
    ax[0, 1].set_title(f"(b) time to recover {RECOVERY_LEVEL:.2f} coverage after shift (missing = never)")
    ax[0, 1].set_ylabel("recovery time [s]")
    ax[1, 0].axhline(1 - ALPHA, color="r", ls=":", lw=1)
    ax[1, 0].axhline(results["nominal"]["cov_post"], color="k", lw=1, label="nominal")
    ax[1, 0].axhline(results["offline_CP"]["cov_post"], color="gray", ls="--", lw=1, label="offline CP")
    ax[1, 0].set_title("(c) coverage over 30–60 s (all steps, labelled or not)")
    ax[1, 0].set_ylabel("coverage")
    ax[1, 1].axhline(1, color="r", ls=":", lw=1)
    ax[1, 1].set_title("(d) region volume / oracle volume after shift (median)")
    ax[1, 1].set_ylabel("volume ratio (1 = oracle, <1 = too small)")
    for a in (ax[0, 1], ax[1, 0], ax[1, 1]):
        a.set_xscale("log"); a.set_xlabel("label interval [s]"); a.legend(fontsize=6)
        a.set_xlim(xs[0] * 0.8, xs[-1] * 1.25)
    fig.tight_layout()
    fig.savefig("05_aci_sparse_labels.png", dpi=130)

    print(f"tau nominal {tau_nom:.2f} | offline CP {tau_off:.2f} | "
          f"oracle median after shift {summary['oracle_tau_post_median']:.2f}")
    hdr = f"{'method':<28}{'cov_pre':>8}{'cov_post':>9}{'cov_late':>9}{'recov_s':>8}{'vol/orc':>8}{'mis_lab':>8}{'mis_all':>8}"
    print(hdr)
    for k, r in results.items():
        if k.startswith("ACI") and "eta=0.05" not in k:
            continue
        rs = f"{r['recovery_s']:.1f}" if r["recovery_s"] is not None else "never"
        print(f"{k:<28}{r['cov_pre']:>8.3f}{r['cov_post']:>9.3f}{r['cov_late']:>9.3f}{rs:>8}"
              f"{r['vol_ratio_post_median']:>8.2f}{r.get('miscov_on_labelled', np.nan):>8.3f}"
              f"{r.get('miscov_all_steps', np.nan):>8.3f}")


if __name__ == "__main__":
    main()

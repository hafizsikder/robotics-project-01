"""Experiment 2: filter consistency (NEES) under a motion-noise shift.

Motivation figure for Topic A. Shows that a filter tuned for nominal noise
becomes *overconfident* when the real noise changes, so its covariance-based
95% region no longer contains the true pose 95% of the time.

Setup
  * Same robot/landmark world as 01_localization.py.
  * Two conditions, M Monte Carlo runs each:
      nominal - true motion noise == noise assumed by the filters
      shift   - at t = T/2 the true motion-noise *std* doubles (variance x4),
                filters keep assuming the nominal value (e.g. wheel slip)
  * Filters: EKF and particle filter (PF covariance = weighted sample cov).

Metrics (Bar-Shalom, Li & Kirubarajan 2001, ch. 5.4)
  NEES_k  = e_k^T P_k^-1 e_k, with e = [dx, dy, wrap(dtheta)]
  ANEES_k = mean of NEES_k over M runs; consistent if inside the two-sided
            95% chi-square band chi2(M*3)/M
  Coverage_k = fraction of runs with NEES_k <= chi2_3(0.95), should be ~0.95

Run:    uv run python 02_nees_consistency.py
Output: 02_nees_consistency.png, 02_nees_consistency.json
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from sim_core import (M_RUNS, N_PARTICLES, SEED, SHIFT_STD_FACTOR, SHIFT_STEP,
                      STATE_DIM, STEPS, DT, control, ekf_predict, ekf_update, nees,
                      pf_moments, pf_step, simulate_step)


# ------------------------------ one MC run ------------------------------------
def run(seed, shift):
    rng = np.random.default_rng(seed)
    truth = np.zeros(3)
    mu, S = truth.copy(), np.eye(3) * 0.01
    P = truth + rng.multivariate_normal(np.zeros(3), S, N_PARTICLES)
    w = np.full(N_PARTICLES, 1 / N_PARTICLES)
    out = np.zeros((2, STEPS))           # rows: EKF, PF

    for t in range(STEPS):
        u = control(t)
        truth, zs = simulate_step(truth, t, shift, rng)

        mu, S = ekf_predict(mu, S, u)
        for z, lm in zs:
            mu, S, _ = ekf_update(mu, S, z, lm)
        P, w = pf_step(P, w, u, zs, rng)

        out[0, t] = nees(mu, S, truth)
        out[1, t] = nees(*pf_moments(P, w), truth)
    return out


# --------------------------------- main ---------------------------------------
def main():
    results = {}
    for cond in ("nominal", "shift"):
        # Same seeds in both conditions -> paired comparison.
        runs = np.stack([run(SEED + i, cond == "shift") for i in range(M_RUNS)])
        results[cond] = runs                         # (M, 2, STEPS)
        print(f"{cond}: done")

    lo, hi = chi2.ppf([0.025, 0.975], M_RUNS * STATE_DIM) / M_RUNS
    thr95 = chi2.ppf(0.95, STATE_DIM)
    tt = np.arange(STEPS) * DT
    names = ("EKF", "PF")
    colors = ("tab:blue", "tab:green")

    summary = {"band_95": [lo, hi], "M_runs": M_RUNS, "shift_time_s": SHIFT_STEP * DT,
               "shift_std_factor": SHIFT_STD_FACTOR}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for col, cond in enumerate(("nominal", "shift")):
        runs = results[cond]
        anees = runs.mean(axis=0)                     # (2, STEPS)
        cover = (runs <= thr95).mean(axis=0)
        ax_a, ax_c = axes[0, col], axes[1, col]
        for f, (name, c) in enumerate(zip(names, colors)):
            ax_a.semilogy(tt, anees[f], c, lw=1, label=name)
            ax_c.plot(tt, cover[f], c, lw=1, label=name)
            for phase, sl in (("before", slice(0, SHIFT_STEP)), ("after", slice(SHIFT_STEP, None))):
                summary[f"{cond}/{name}/{phase}"] = {
                    "mean_ANEES": round(float(anees[f, sl].mean()), 3),
                    "frac_steps_in_band": round(float(((anees[f, sl] >= lo) & (anees[f, sl] <= hi)).mean()), 3),
                    "mean_coverage_95": round(float(cover[f, sl].mean()), 3),
                }
        ax_a.axhspan(lo, hi, color="gray", alpha=0.3, label="95% consistency band")
        ax_a.axhline(STATE_DIM, color="k", ls=":", lw=0.8)
        ax_c.axhline(0.95, color="k", ls="--", lw=1, label="target 0.95")
        for ax in (ax_a, ax_c):
            if cond == "shift":
                ax.axvline(SHIFT_STEP * DT, color="r", ls="--", lw=1)
        ax_a.set_title(f"{cond}: ANEES over {M_RUNS} runs"
                       + (f" (noise std ×{SHIFT_STD_FACTOR:g} at red line)" if cond == "shift" else ""))
        ax_c.set_title(f"{cond}: empirical coverage of the filter's 95% region")
        ax_c.set_ylim(0, 1.02)
        ax_c.set_xlabel("time [s]")
        ax_a.legend(loc="upper left", fontsize=8)
        ax_c.legend(loc="lower left", fontsize=8)
    axes[0, 0].set_ylabel("ANEES (log)")
    axes[1, 0].set_ylabel("coverage")
    fig.tight_layout()
    fig.savefig("02_nees_consistency.png", dpi=130)

    with open("02_nees_consistency.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

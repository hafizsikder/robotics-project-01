"""Experiment 3: can innovations (NIS) reveal the noise shift without ground truth?

Experiment 2 showed the filters become overconfident after the motion-noise
shift, but it measured this with NEES, which needs the true pose. Online, the
only observable signal is the innovation nu = z - z_hat. Here we ask:

  Q1 (population)  Does ANIS leave its chi-square band after the shift,
                   and by how much compared with ANEES?
  Q2 (single run)  Can a windowed NIS test on ONE run detect the shift
                   quickly with few false alarms? (Online, there is only one run.)

NIS definitions
  EKF: sequential updates -> each landmark gives nu_j ~ N(0, S_nu,j) with 2 DOF
       when the filter is consistent; per step we sum NIS and DOF.
  PF : joint predictive NIS of all landmarks seen this step, using the
       predicted particle cloud (before reweighting), see sim_core.pf_predictive_nis.

Detector: sliding window of W steps, statistic = sum NIS, dof = sum DOF;
alarm when chi2.sf(stat, dof) < ALPHA.

Run:    uv run python 03_nis_detection.py
Output: 03_nis_detection.png, 03_nis_detection.json
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from sim_core import (DT, M_RUNS, N_PARTICLES, SEED, SHIFT_STD_FACTOR, SHIFT_STEP, STATE_DIM,
                      STEPS, control, ekf_predict, ekf_update, nees, pf_correct, pf_moments,
                      pf_predict, pf_predictive_nis, simulate_step)

WINDOW = 20          # detector window [steps] = 2 s
ALPHA = 0.01         # per-step false-alarm level of the window test
DETECT_HORIZON = 100 # count a detection only within 10 s after the shift


def run(seed, shift):
    rng = np.random.default_rng(seed)
    truth = np.zeros(3)
    mu, S = truth.copy(), np.eye(3) * 0.01
    P = truth + rng.multivariate_normal(np.zeros(3), S, N_PARTICLES)
    w = np.full(N_PARTICLES, 1 / N_PARTICLES)
    # [filter, quantity, t]; quantity: 0 = NIS sum, 1 = DOF, 2 = NEES
    out = np.zeros((2, 3, STEPS))

    for t in range(STEPS):
        u = control(t)
        truth, zs = simulate_step(truth, t, shift, rng)

        mu, S = ekf_predict(mu, S, u)
        for z, lm in zs:
            mu, S, nis = ekf_update(mu, S, z, lm)
            out[0, 0, t] += nis
            out[0, 1, t] += 2

        P = pf_predict(P, u, rng)
        nis, dof = pf_predictive_nis(P, w, zs)
        if dof:
            out[1, 0, t], out[1, 1, t] = nis, dof
        P, w = pf_correct(P, w, zs, rng)

        out[0, 2, t] = nees(mu, S, truth)
        out[1, 2, t] = nees(*pf_moments(P, w), truth)
    return out


def window_alarms(nis, dof):
    """(M, STEPS) -> boolean alarms; False where the window has no measurements."""
    k = np.ones(WINDOW)
    s = np.apply_along_axis(lambda x: np.convolve(x, k)[:STEPS], 1, nis)
    d = np.apply_along_axis(lambda x: np.convolve(x, k)[:STEPS], 1, dof)
    alarms = np.zeros_like(s, dtype=bool)
    ok = d > 0
    alarms[ok] = chi2.sf(s[ok], d[ok]) < ALPHA
    alarms[:, :WINDOW] = False            # window not yet full
    return alarms


def main():
    data = {}
    for cond in ("nominal", "shift"):
        data[cond] = np.stack([run(SEED + i, cond == "shift") for i in range(M_RUNS)])
        print(f"{cond}: done")                 # (M, filter, quantity, STEPS)

    names, colors = ("EKF", "PF"), ("tab:blue", "tab:green")
    tt = np.arange(STEPS) * DT
    pre, post = slice(WINDOW, SHIFT_STEP), slice(SHIFT_STEP, None)
    summary = {"window_steps": WINDOW, "alpha": ALPHA, "M_runs": M_RUNS,
               "shift_time_s": SHIFT_STEP * DT, "shift_std_factor": SHIFT_STD_FACTOR}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for col, cond in enumerate(("nominal", "shift")):
        D = data[cond]
        ax_r, ax_d = axes[0, col], axes[1, col]
        for f, (name, c) in enumerate(zip(names, colors)):
            nis, dof, ne = D[:, f, 0], D[:, f, 1], D[:, f, 2]
            tot_dof = dof.sum(axis=0)
            has = tot_dof > 0
            anis_ratio = np.full(STEPS, np.nan)        # 1.0 when consistent
            anis_ratio[has] = nis.sum(axis=0)[has] / tot_dof[has]
            anees_ratio = ne.mean(axis=0) / STATE_DIM  # 1.0 when consistent
            lo = chi2.ppf(0.025, tot_dof[has]) / tot_dof[has]
            hi = chi2.ppf(0.975, tot_dof[has]) / tot_dof[has]
            in_band = np.full(STEPS, np.nan)
            in_band[has] = (anis_ratio[has] >= lo) & (anis_ratio[has] <= hi)

            ax_r.semilogy(tt, anees_ratio, c, lw=1, ls="--", alpha=0.7, label=f"{name} ANEES/3 (needs truth)")
            ax_r.semilogy(tt, anis_ratio, c, lw=1.2, label=f"{name} ANIS/dof (observable)")

            alarms = window_alarms(nis, dof)
            ax_d.plot(tt, alarms.mean(axis=0), c, lw=1, label=f"{name} alarm rate")

            entry = {}
            for phase, sl in (("before", pre), ("after", post)):
                entry[f"{phase}/mean_ANIS_ratio"] = round(float(np.nanmean(anis_ratio[sl])), 3)
                entry[f"{phase}/mean_ANEES_ratio"] = round(float(np.nanmean(anees_ratio[sl])), 3)
                entry[f"{phase}/frac_ANIS_in_band"] = round(float(np.nanmean(in_band[sl])), 3)
                entry[f"{phase}/alarm_rate"] = round(float(alarms[:, sl].mean()), 4)
            entry["frac_steps_without_measurement"] = round(float((dof == 0).mean()), 3)
            # In the nominal condition this is the chance-level baseline:
            # probability of >= 1 (false) alarm in the same 10 s window.
            win = alarms[:, SHIFT_STEP:SHIFT_STEP + DETECT_HORIZON]
            hit = win.any(axis=1)
            delays = win.argmax(axis=1)[hit] * DT
            entry["any_alarm_within_10s_of_t30"] = round(float(hit.mean()), 3)
            entry["median_first_alarm_s"] = round(float(np.median(delays)), 2) if hit.any() else None
            if cond == "shift":
                # Is the observable ratio a good proxy for the unobservable one?
                m = ~np.isnan(anis_ratio[post])
                entry["corr_ANIS_ANEES_after"] = round(float(np.corrcoef(
                    anis_ratio[post][m], anees_ratio[post][m])[0, 1]), 3)
            summary[f"{cond}/{name}"] = entry

        ax_r.axhline(1, color="k", ls=":", lw=0.8)
        ax_d.axhline(ALPHA, color="k", ls="--", lw=1, label=f"nominal level α={ALPHA}")
        for ax in (ax_r, ax_d):
            if cond == "shift":
                ax.axvline(SHIFT_STEP * DT, color="r", ls="--", lw=1)
        ax_r.set_title(f"{cond}: normalized ANIS vs ANEES ({M_RUNS} runs)")
        ax_d.set_title(f"{cond}: single-run window test, fraction of runs alarming")
        ax_d.set_ylim(-0.02, 1.02)
        ax_d.set_xlabel("time [s]")
        ax_r.legend(fontsize=7, loc="upper left")
        ax_d.legend(fontsize=7, loc="upper left")
    axes[0, 0].set_ylabel("ratio to expected (1 = consistent)")
    axes[1, 0].set_ylabel("alarm rate")
    fig.tight_layout()
    fig.savefig("03_nis_detection.png", dpi=130)

    with open("03_nis_detection.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

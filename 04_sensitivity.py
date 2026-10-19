"""Experiment 4: when are innovations blind to a motion-noise shift? (EKF)

Sweeps sensor-noise scale x shift magnitude x detector window, and compares
Monte Carlo results with two theories (derivations in THEORY.md):

  T1  scalar closed form in rho = Q/R (random-walk Kalman filter):
        NEES ratio -> (1+c)/2,  NIS ratio -> 1 + (c-1)/2 * sqrt(rho)  as rho -> 0
      where c = (true/assumed motion-noise std)^2.
  T2  linearized error-covariance analysis along each EKF trajectory:
        P_true^- = G P_true^+ G^T + V Q_true V^T
        P_true^+ = (I-KH) P_true^- (I-KH)^T + K R K^T
        E[NEES] = tr(P^-1 P_true),  E[NIS_j] = tr(S_nu,j^-1 (H P_true^- H^T + R))

Effective SNR (observable online, no ground truth):
    g   = mean over updates of tr(R^-1 H P^- H^T) / 2   (filter's own prior)
    rho_eff = g^2 / (1 + g)        (inverts the scalar relation P^-/R = g(rho))

Run:    uv run python 04_sensitivity.py      (~5 min on 4 cores)
Output: 04_sensitivity.png, 04_sensitivity.json
"""

import itertools
import json
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from sim_core import (DT, M_RUNS, Q_NOM, R as R_NOM, SEED, SHIFT_STEP, STATE_DIM, STEPS,
                      control, ekf_jacobians, ekf_predict, ekf_update, nees, q_true,
                      simulate_step)

R_SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]     # sensor-noise std multiplier
SHIFTS = [1.5, 2.0, 3.0]                  # true motion-noise std multiplier after t=30 s
WINDOWS = [10, 20, 50, 100]               # detector window [steps]
ALPHA = 0.01
POST = slice(SHIFT_STEP, STEPS)


# ------------------------------ scalar theory --------------------------------
def scalar_theory(rho, c):
    """Steady-state NEES and NIS ratios of a random-walk KF (R = 1, Q = rho)
    whose true process noise is c * rho. Verified against MC to 3 decimals."""
    Pm = (rho + np.sqrt(rho ** 2 + 4 * rho)) / 2
    K = Pm / (Pm + 1)
    Ptm = (K ** 2 + c * rho) / (K * (2 - K))
    Ptp = (1 - K) ** 2 * Ptm + K ** 2
    return Ptp / K, (Ptm + 1) / (Pm + 1)


# --------------------------------- one run -----------------------------------
def run(args):
    seed, r_scale, shift_factor, shift = args
    Rm = R_NOM * r_scale ** 2
    rng = np.random.default_rng(seed)
    truth = np.zeros(3)
    mu, S = truth.copy(), np.eye(3) * 0.01
    Pt = np.zeros((3, 3))                 # true error covariance (filter starts exact)
    out = np.zeros((6, STEPS))            # nis, dof, nees, nees_pred, nis_pred, g_sum

    for t in range(STEPS):
        u = control(t)
        truth, zs = simulate_step(truth, t, shift, rng, R=Rm, shift_factor=shift_factor)
        G, V = ekf_jacobians(mu, u)
        Pt = G @ Pt @ G.T + V @ q_true(t, shift, shift_factor) @ V.T
        mu, S = ekf_predict(mu, S, u)
        for z, lm in zs:
            S_prior = S
            mu, S, nis, (K, H, S_nu) = ekf_update(mu, S, z, lm, R=Rm, return_gain=True)
            out[0, t] += nis
            out[1, t] += 2
            out[4, t] += np.trace(np.linalg.solve(S_nu, H @ Pt @ H.T + Rm))
            out[5, t] += np.trace(np.linalg.solve(Rm, H @ S_prior @ H.T)) / 2
            I_KH = np.eye(3) - K @ H
            Pt = I_KH @ Pt @ I_KH.T + K @ Rm @ K.T
        out[2, t] = nees(mu, S, truth)
        out[3, t] = np.trace(np.linalg.solve(S, Pt))
    return out


def window_alarms(nis, dof, W):
    k = np.ones(W)
    s = np.apply_along_axis(lambda x: np.convolve(x, k)[:STEPS], 1, nis)
    d = np.apply_along_axis(lambda x: np.convolve(x, k)[:STEPS], 1, dof)
    alarms = np.zeros_like(s, dtype=bool)
    ok = d > 0
    alarms[ok] = chi2.sf(s[ok], d[ok]) < ALPHA
    alarms[:, :W] = False
    return alarms


# ---------------------------------- main -------------------------------------
def main():
    conds = [(r, 1.0, False) for r in R_SCALES] + \
            [(r, s, True) for r, s in itertools.product(R_SCALES, SHIFTS)]
    jobs = [(SEED + i, r, s, sh) for (r, s, sh) in conds for i in range(M_RUNS)]
    with Pool(4) as pool:
        flat = pool.map(run, jobs, chunksize=10)
    data = {c: np.stack(flat[k * M_RUNS:(k + 1) * M_RUNS]) for k, c in enumerate(conds)}
    print("simulation done")

    rows = []
    for r in R_SCALES:
        nom = data[(r, 1.0, False)]
        fa = {W: float(window_alarms(nom[:, 0], nom[:, 1], W)[:, W:].mean()) for W in WINDOWS}
        for s in SHIFTS:
            D = data[(r, s, True)]
            nis, dof = D[:, 0, POST], D[:, 1, POST]
            row = {
                "r_scale": r, "shift_std": s, "c": s ** 2,
                "rho_eff": None,
                "NEES_ratio_mc": float(D[:, 2, POST].mean() / STATE_DIM),
                "NEES_ratio_T2": float(D[:, 3, POST].mean() / STATE_DIM),
                "NIS_ratio_mc": float(nis.sum() / dof.sum()),
                "NIS_ratio_T2": float(D[:, 4, POST].sum() / dof.sum()),
                "dof_per_step": float(dof.mean()),
                "power": {}, "power_pred": {}, "false_alarm": fa,
            }
            g = D[:, 5].sum() / (D[:, 1].sum() / 2)          # mean over all updates
            row["rho_eff"] = float(g ** 2 / (1 + g))
            row["NEES_ratio_T1"], row["NIS_ratio_T1"] = map(float, scalar_theory(row["rho_eff"], s ** 2))
            for W in WINDOWS:
                al = window_alarms(D[:, 0], D[:, 1], W)
                row["power"][W] = float(al[:, SHIFT_STEP + W:].mean())
                d = W * row["dof_per_step"]
                row["power_pred"][W] = float(chi2.sf(chi2.ppf(1 - ALPHA, d) / row["NIS_ratio_T2"], d))
            rows.append(row)

    # ---------------------------------- plots ----------------------------------
    cmap = {1.5: "tab:orange", 2.0: "tab:blue", 3.0: "tab:purple"}
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    a = ax[0, 0]
    rr = np.logspace(-4, 1.5, 300)
    for s in SHIFTS:
        ne, ni = scalar_theory(rr, s ** 2)
        a.semilogx(rr, ne, color=cmap[s], lw=1.5, label=f"NEES theory, std×{s:g}")
        a.semilogx(rr, ni, color=cmap[s], lw=1.5, ls="--", label=f"NIS theory, std×{s:g}")
        pts = [x for x in rows if x["shift_std"] == s]
        a.scatter([p["rho_eff"] for p in pts], [p["NEES_ratio_mc"] for p in pts], color=cmap[s], marker="o", zorder=3)
        a.scatter([p["rho_eff"] for p in pts], [p["NIS_ratio_mc"] for p in pts], facecolor="none",
                  edgecolor=cmap[s], marker="s", zorder=3)
    a.axhline(1, color="k", lw=0.6, ls=":")
    a.set_xlabel("effective SNR  ρ_eff = Q/R (observable, from filter's own prior)")
    a.set_ylabel("ratio to nominal (1 = consistent)")
    a.set_title("(a) EKF Monte Carlo (●NEES, □NIS) vs scalar closed form T1")
    a.legend(fontsize=7, ncol=2)

    b = ax[0, 1]
    for key, mk in (("NEES", "o"), ("NIS", "s")):
        b.scatter([x[f"{key}_ratio_T2"] for x in rows], [x[f"{key}_ratio_mc"] for x in rows],
                  marker=mk, c=[cmap[x["shift_std"]] for x in rows], label=key)
    lim = [0.9, max(max(x["NEES_ratio_mc"], x["NIS_ratio_mc"]) for x in rows) * 1.1]
    b.plot(lim, lim, "k--", lw=0.8)
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("predicted by linearized error-covariance analysis T2")
    b.set_ylabel("measured (100-run Monte Carlo)")
    b.set_title("(b) T2 prediction vs measurement (all 15 conditions)")
    b.legend()

    c = ax[1, 0]
    for i, r in enumerate(R_SCALES):
        row = next(x for x in rows if x["r_scale"] == r and x["shift_std"] == 2.0)
        col = plt.cm.viridis(i / (len(R_SCALES) - 1))
        c.plot(WINDOWS, [row["power"][W] for W in WINDOWS], "o-", color=col,
               label=f"R std×{r:g} (ρ_eff={row['rho_eff']:.3g})")
        c.plot(WINDOWS, [row["power_pred"][W] for W in WINDOWS], "--", color=col, lw=0.8)
    c.axhline(ALPHA, color="k", ls=":", lw=0.8)
    c.set_xscale("log"); c.set_xticks(WINDOWS, [f"{W*DT:g} s" for W in WINDOWS])
    c.set_xlabel("detector window"); c.set_ylabel("per-step detection power")
    c.set_title("(c) single-run window test, shift std×2 (solid MC, dashed predicted)")
    c.legend(fontsize=7)

    d = ax[1, 1]
    M = np.array([[next(x for x in rows if x["r_scale"] == r and x["shift_std"] == s)["power"][20]
                   for s in SHIFTS] for r in R_SCALES])
    im = d.imshow(M, cmap="magma", vmin=0, vmax=1, aspect="auto")
    for i in range(len(R_SCALES)):
        for j in range(len(SHIFTS)):
            d.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                   color="w" if M[i, j] < 0.5 else "k")
    d.set_xticks(range(len(SHIFTS)), [f"×{s:g}" for s in SHIFTS])
    d.set_yticks(range(len(R_SCALES)), [f"×{r:g}" for r in R_SCALES])
    d.set_xlabel("motion-noise std shift"); d.set_ylabel("sensor-noise std")
    d.set_title("(d) detection power, 2 s window, α=0.01")
    fig.colorbar(im, ax=d)

    fig.tight_layout()
    fig.savefig("04_sensitivity.png", dpi=130)
    with open("04_sensitivity.json", "w") as fh:
        json.dump(rows, fh, indent=1)

    print(f"{'R×':>5} {'std×':>5} {'rho_eff':>8} | {'NEES mc':>8} {'T2':>6} {'T1':>6} | "
          f"{'NIS mc':>7} {'T2':>6} {'T1':>6} | power W=10/20/50/100 (FA@20)")
    for x in rows:
        print(f"{x['r_scale']:>5g} {x['shift_std']:>5g} {x['rho_eff']:>8.4f} | "
              f"{x['NEES_ratio_mc']:>8.3f} {x['NEES_ratio_T2']:>6.3f} {x['NEES_ratio_T1']:>6.3f} | "
              f"{x['NIS_ratio_mc']:>7.3f} {x['NIS_ratio_T2']:>6.3f} {x['NIS_ratio_T1']:>6.3f} | "
              + "/".join(f"{x['power'][W]:.2f}" for W in WINDOWS) + f" ({x['false_alarm'][20]:.3f})")


if __name__ == "__main__":
    main()

"""Experiment 6 (method v2): calibrators that learn from score magnitudes, with sparse labels.

Exp 5 showed that plain ACI never recovers coverage when true poses arrive
every 1 s or less often: each label carries one bit (miss / no miss). Here we
compare calibrators that also use the size of the labelled scores.

Calibrators (all report tau_t; region = {x : NEES <= tau_t}):
  ACI        quantile tracking on log tau (Gibbs & Candes 2021). Bit only.
  PI         PI-style controller in the spirit of conformal PID (Angelopoulos,
             Candes & Tibshirani 2023): proportional quantile tracking with a
             scale-adaptive step lr * B_t (B_t = max of the last 20 labelled
             scores) plus a saturating integrator K_I * B_t * tanh(S_t / 5),
             S_t = sum of (miss - alpha). Uses magnitudes through B_t.
             This is our variant, not their exact code.
  RollQ      split-conformal quantile of the last K labelled scores
             (rank ceil((K+1)(1-alpha)); window warm-started with K scores
             drawn from nominal calibration runs).
  WQ         non-exchangeable weighted conformal quantile (Barber et al. 2023)
             over all labelled scores with weights 2^(-age/h) and the usual
             test-point mass at +inf; same warm start.
  DtACI      Gibbs & Candes (JMLR 2024): ACI experts with step sizes
             {0.01..0.8} on log tau, combined by exponential weights on the
             pinball loss, with the paper's default eta_w and sigma.
  nominal    tau = chi2_3(0.95);  offline CP: 95% quantile of nominal runs.

Protocol (avoids tuning on the test set):
  * tune every method's hyperparameters on a VALIDATION scenario: shift std x1.5,
    200 runs with disjoint seeds, objective = mean |coverage_t - 0.95|
    (chosen separately for each label interval L and delay D);
  * evaluate the chosen configuration on the TEST scenario: shift std x2,
    the same 200 runs as Exp 5 (cache/05_scores.npz).
Label intervals: 0.5, 1, 2, 5, 10 s. Delays: 0 and 2 s.

Run:    uv run python 06_magnitude_calibrators.py     (~3-5 min)
Output: 06_magnitude_calibrators.png, 06_magnitude_calibrators.json
"""

import itertools
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from calib_core import ALPHA, SMOOTH, cached_scores, drive, metrics, smooth
from sim_core import DT, SEED, SHIFT_STEP, STATE_DIM, STEPS

LABEL_INTERVALS = [5, 10, 20, 50, 100]          # steps (x 0.1 s)
DELAYS = [0, 20]
M_VAL = 200
VAL_SEED_OFFSET = 20_000
VAL_SHIFT = 1.5
TAU0 = chi2.ppf(1 - ALPHA, STATE_DIM)
PREFILL_SEED = 7


# ------------------------------- calibrators ---------------------------------
class ACI:
    def __init__(self, M, cal_pool, eta):
        self.eta = eta
        self.log_tau = np.full(M, np.log(TAU0))

    def tau(self):
        return np.exp(self.log_tau)

    def update(self, s, tau_k):
        self.log_tau += self.eta * ((s > tau_k) - ALPHA)


def _prefill(M, cal_pool, n):
    rng = np.random.default_rng(PREFILL_SEED)
    return rng.choice(cal_pool, size=(M, n), replace=True)


class PI:
    def __init__(self, M, cal_pool, lr, ki, window=20, csat=5.0):
        self.lr, self.ki, self.csat = lr, ki, csat
        self.buf = _prefill(M, cal_pool, window)          # recent labelled scores
        self.qp = np.full(M, TAU0)
        self.S = np.zeros(M)
        self.q = self.qp.copy()

    def tau(self):
        return np.maximum(self.q, 1e-3)

    def update(self, s, tau_k):
        g = (s > tau_k) - ALPHA
        self.buf = np.concatenate([self.buf[:, 1:], s[:, None]], axis=1)
        B = self.buf.max(axis=1)
        self.qp = self.qp + self.lr * B * g
        self.S += g
        self.q = self.qp + self.ki * B * np.tanh(self.S / self.csat)


def conformal_rank(n):
    return math.ceil((n + 1) * (1 - ALPHA))


class RollQ:
    def __init__(self, M, cal_pool, K):
        self.K = K
        self.buf = _prefill(M, cal_pool, K)
        self._set()

    def _set(self):
        r = min(conformal_rank(self.K), self.K)           # r > K only if K < 19
        self.q = np.sort(self.buf, axis=1)[:, r - 1]

    def tau(self):
        return self.q

    def update(self, s, tau_k):
        self.buf = np.concatenate([self.buf[:, 1:], s[:, None]], axis=1)
        self._set()


class WQ:
    def __init__(self, M, cal_pool, half_life, n_prefill=50):
        self.h = half_life
        self.vals = _prefill(M, cal_pool, n_prefill)
        self.age = np.full(n_prefill, 1.0)              # prefill ages like the oldest label
        self._set()

    def _set(self):
        w = 2.0 ** (-self.age / self.h)
        w_inf = 1.0                                       # test point (age 0) at +inf
        order = np.argsort(self.vals, axis=1)
        sv = np.take_along_axis(self.vals, order, axis=1)
        cw = np.cumsum(w[order], axis=1) / (w.sum() + w_inf)
        idx = (cw < 1 - ALPHA).sum(axis=1)                # first index reaching 1-alpha
        q = np.full(self.vals.shape[0], np.inf)
        ok = idx < sv.shape[1]
        q[ok] = sv[ok, idx[ok]]
        self.q = np.minimum(q, 1e4)                       # +inf -> very large region

    def tau(self):
        return self.q

    def update(self, s, tau_k):
        self.age += 1
        self.vals = np.concatenate([self.vals, s[:, None]], axis=1)
        self.age = np.append(self.age, 1.0)
        keep = self.age <= 12 * self.h                    # weights < 2^-12 dropped
        self.vals, self.age = self.vals[:, keep], self.age[keep]
        self._set()


class DtACI:
    ETAS = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8])

    def __init__(self, M, cal_pool, n_labels):
        k, I = len(self.ETAS), max(n_labels, 2)
        self.theta = np.full((M, k), np.log(TAU0))
        self.w = np.full((M, k), 1.0 / k)
        self.sigma = 1.0 / (2 * I)
        self.eta_w = math.sqrt(3 / I) * math.sqrt((math.log(k * I) + 2) /
                                                  ((1 - ALPHA) ** 2 * ALPHA ** 3))

    def tau(self):
        return np.exp((self.w * self.theta).sum(axis=1))

    def update(self, s, tau_k):
        b = np.log(np.maximum(s, 1e-12))[:, None]
        diff = b - self.theta
        loss = np.where(diff > 0, (1 - ALPHA) * diff, -ALPHA * diff)   # pinball, level 1-alpha
        lw = np.log(self.w) - self.eta_w * loss
        lw -= lw.max(axis=1, keepdims=True)
        w = np.exp(lw)
        w /= w.sum(axis=1, keepdims=True)
        self.w = (1 - self.sigma) * w + self.sigma / w.shape[1]
        self.theta += self.ETAS[None, :] * ((b > self.theta) - ALPHA)   # each expert's own miss


GRIDS = {
    # Grids widened after a first run selected edge values (ACI eta=1, PI lr=0.1, ki=2).
    "ACI": [dict(eta=e) for e in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)],
    "PI": [dict(lr=lr, ki=ki) for lr in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
           for ki in (0, 0.5, 1, 2, 4, 8)],
    "RollQ": [dict(K=K) for K in (20, 30, 50, 100)],
    # Weighted CP gives a finite threshold only if sum(label weights) >= (1-alpha)/alpha = 19
    # (test-point mass 1 at +inf); with half-life h that sum is ~1.44 h, so h < ~14 is infeasible.
    "WQ": [dict(half_life=h) for h in (15, 20, 30, 50, 100)],
    "DtACI": [dict()],
}
CLASSES = {"ACI": ACI, "PI": PI, "RollQ": RollQ, "WQ": WQ, "DtACI": DtACI}


def make(name, M, cal_pool, L, params):
    if name == "DtACI":
        return DtACI(M, cal_pool, n_labels=STEPS // L)
    return CLASSES[name](M, cal_pool, **params)


# ---------------------------------- main -------------------------------------
def main():
    d = np.load(os.path.join("cache", "05_scores.npz"))
    test, cal = d["test"], d["cal"]
    val = cached_scores(os.path.join("cache", "06_val_scores.npz"),
                        [SEED + VAL_SEED_OFFSET + i for i in range(M_VAL)], True, VAL_SHIFT)
    cal_pool = cal[:, SMOOTH:].ravel()
    oracle_test = np.quantile(test, 1 - ALPHA, axis=0)
    oracle_val = np.quantile(val, 1 - ALPHA, axis=0)
    M = test.shape[0]
    print(f"test {test.shape}, val {val.shape}, cal pool {cal_pool.size}")

    results, curves, chosen = {}, {}, {}
    for name, tau_c in (("nominal", TAU0), ("offline_CP", float(np.quantile(cal_pool, 1 - ALPHA)))):
        results[name], curves[name] = metrics(test, np.full_like(test, tau_c), oracle_test)

    for L, D in itertools.product(LABEL_INTERVALS, DELAYS):
        for name, grid in GRIDS.items():
            # 1) tune on validation
            best, best_p = np.inf, None
            for p in grid:
                tau_v, _ = drive(make(name, val.shape[0], cal_pool, L, p), val, L, D)
                err = metrics(val, tau_v, oracle_val)[0]["cov_tracking_err"]
                if err < best:
                    best, best_p = err, p
            # 2) evaluate once on test
            tau_t, lab = drive(make(name, M, cal_pool, L, best_p), test, L, D)
            key = f"{name} L={L} D={D}"
            results[key], curves[key] = metrics(test, tau_t, oracle_test, lab)
            results[key]["chosen_params"] = best_p
            results[key]["val_tracking_err"] = best
        print(f"L={L} D={D} done")

    with open("06_magnitude_calibrators.json", "w") as fh:
        json.dump({"alpha": ALPHA, "M_test": M, "M_val": int(val.shape[0]),
                   "val_shift_std": VAL_SHIFT, "results": results}, fh, indent=1)

    # ---------------------------------- plots ----------------------------------
    names = list(GRIDS)
    col = dict(zip(names, ("tab:orange", "tab:blue", "tab:green", "tab:red", "tab:purple")))
    tt = np.arange(STEPS) * DT
    xs = np.array(LABEL_INTERVALS) * DT
    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))
    for a, L in ((ax[0, 0], 10), (ax[0, 1], 50)):
        a.plot(tt, smooth(curves["nominal"]), "k", lw=1, label="nominal")
        a.plot(tt, smooth(curves["offline_CP"]), color="gray", ls="--", lw=1, label="offline CP")
        for n in names:
            a.plot(tt, smooth(curves[f"{n} L={L} D=20"]), color=col[n], lw=1.3, label=n)
        a.axhline(1 - ALPHA, color="r", ls=":", lw=1); a.axvline(SHIFT_STEP * DT, color="r", ls="--", lw=1)
        a.set_ylim(0.4, 1.01); a.set_xlabel("time [s]"); a.set_ylabel("coverage (2 s trailing avg)")
        a.set_title(f"({'a' if L == 10 else 'b'}) label every {L * DT:g} s, 2 s delay (test, std×2)")
        a.legend(fontsize=7, loc="lower left")

    panels = ((ax[0, 2], "cov_post", "(c) coverage 30–60 s", 1 - ALPHA),
              (ax[1, 0], "recovery_s", "(d) recovery time to 0.92 [s] (missing = never)", None),
              (ax[1, 1], "vol_ratio_post_median", "(e) volume / oracle after shift", 1.0),
              (ax[1, 2], "vol_ratio_pre_median", "(f) volume / oracle before shift (cost of caution)", 1.0))
    for a, m, title, ref in panels:
        for n in names:
            for D, ls in ((20, "-"), (0, ":")):
                ys = [results[f"{n} L={L} D={D}"][m] for L in LABEL_INTERVALS]
                ys = [np.nan if y is None else y for y in ys]
                a.plot(xs, ys, ls, color=col[n], marker="o", ms=3,
                       label=f"{n}" + (" (2 s delay)" if D == 20 else " (no delay)"))
        if m == "cov_post":
            a.axhline(results["nominal"]["cov_post"], color="k", lw=1, label="nominal")
            a.axhline(results["offline_CP"]["cov_post"], color="gray", ls="--", lw=1, label="offline CP")
        if ref is not None:
            a.axhline(ref, color="r", ls=":", lw=1)
        a.set_xscale("log"); a.set_xlim(xs[0] * 0.8, xs[-1] * 1.25)
        a.set_xlabel("label interval [s]"); a.set_title(title)
    ax[0, 2].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig("06_magnitude_calibrators.png", dpi=125)

    print(f"\n{'method':<20}{'params':<26}{'cov_pre':>8}{'cov_post':>9}{'cov_late':>9}"
          f"{'recov':>7}{'vol_pre':>8}{'vol_post':>9}")
    for k, r in results.items():
        if k not in ("nominal", "offline_CP") and "D=20" not in k:
            continue
        rs = "never" if r["recovery_s"] is None else f"{r['recovery_s']:.1f}"
        ps = json.dumps(r.get("chosen_params", ""))
        print(f"{k:<20}{ps:<26}{r['cov_pre']:>8.3f}{r['cov_post']:>9.3f}{r['cov_late']:>9.3f}"
              f"{rs:>7}{r['vol_ratio_pre_median']:>8.2f}{r['vol_ratio_post_median']:>9.2f}")


if __name__ == "__main__":
    main()

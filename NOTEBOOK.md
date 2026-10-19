# Research notebook: Topic A (online conformal calibration of recursive estimators)

Every entry: date, script, the exact numbers it produced, and what we concluded.

---

## 2026-10-18: Exp 2, NEES consistency under a motion-noise shift

**Script:** `02_nees_consistency.py` → `02_nees_consistency.png`, `02_nees_consistency.json`
**Setup:** 100 Monte Carlo runs × 60 s, paired seeds. At t = 30 s the true motion-noise std doubles (variance ×4). Both filters keep the nominal noise model. Sensor range 0.5–8 m. PF uses 300 particles.
95% ANEES consistency band for M = 100, 3 DOF: [2.54, 3.50].

| Condition / filter | Phase | mean ANEES | steps inside band | coverage of nominal 95% region |
|---|---|---|---|---|
| nominal / EKF | 0–30 s | 3.03 | 88% | 0.948 |
| nominal / EKF | 30–60 s | 3.02 | 98% | 0.952 |
| nominal / PF  | 0–30 s | 3.75 | 18% | 0.898 |
| nominal / PF  | 30–60 s | 4.12 | 15% | 0.887 |
| shift / EKF   | 30–60 s | 7.24 | **0%** | **0.655** |
| shift / PF    | 30–60 s | 114.6 | **0%** | **0.399** |

**Findings**
1. A correctly tuned EKF is consistent: ANEES ≈ 3, coverage ≈ 0.95. This validates the pipeline.
2. After the shift, the "95%" region of the EKF covers the truth only **66%** of the time, and the PF's only **40%**. This is the core motivation for Topic A.
3. Even without any shift, the PF is mildly overconfident (coverage ~0.89). Sample impoverishment with 300 particles and a sharp likelihood is the likely cause. So covariance-based regions can be miscalibrated even under nominal conditions, which strengthens the case for distribution-free calibration.
4. The miscalibration is **time-varying** (EKF coverage drifts from ~0.8 to ~0.5 after the shift). A single offline inflation factor cannot fix this, which argues for an *online/adaptive* method rather than split conformal.

**Pitfall found and fixed (worth a sentence in the paper's experimental setup)**
The first run showed ANEES spikes up to ~2400 in the *nominal* case. Every spike came from a run where the robot passed 3–5 cm from a landmark, where the bearing Jacobian (∝ 1/r²) is singular and the EKF linearization fails. A 0.5 m minimum sensing range (physically realistic) removed it. Lesson: always inspect per-run outliers before averaging NEES.
A PF log-weight underflow (`log(0)`) was also fixed.

**Open questions for the method**
- Ground truth is unavailable online. Does the **NIS** (normalized innovation squared, computed from measurements only) detect the shift as clearly as NEES does? If yes, innovations are a usable calibration signal. → next experiment.
- How should a measurement-space miscoverage be mapped to a state-space (SE(2)) set?

---

## 2026-10-19: Exp 3, do innovations (NIS) reveal the shift without ground truth?

**Script:** `03_nis_detection.py` → `03_nis_detection.png`, `03_nis_detection.json`
**Refactor:** world, models and filters moved to `sim_core.py`. Exp 2 rerun after the refactor gives **bit-identical** JSON (regression check passed).
**Setup:** same 100 paired runs as Exp 2. EKF NIS from sequential updates (2 DOF per landmark). PF NIS is the joint predictive NIS from the predicted particle cloud. Single-run detector: 2 s sliding window, chi-square test at α = 0.01.

### Q1: population level (ratio to expected value, 1 = consistent)

| Condition / filter | Phase | ANIS ratio (observable) | ANEES ratio (needs truth) | ANIS steps in 95% band |
|---|---|---|---|---|
| nominal / EKF | 30–60 s | 0.98 | 1.01 | 96% |
| nominal / PF  | 30–60 s | 0.99 | **1.37** | 96% |
| shift / EKF   | 30–60 s | **1.13** | **2.41** | 75% |
| shift / PF    | 30–60 s | **1.19** | **38.2** | 49% |

Time correlation between the ANIS and ANEES ratios after the shift: 0.06 (EKF), 0.14 (PF).

### Q2: single-run detection (the online setting)

| | P(≥1 alarm within 10 s of t = 30 s) | median first alarm |
|---|---|---|
| nominal EKF / PF (chance baseline) | 0.18 / 0.20 | 4.7 / 4.8 s |
| shift EKF / PF | 0.50 / 0.67 | 3.4 / 4.4 s |

Per-step alarm rate after the shift is only 5% (EKF) and 10% (PF), against 1% before.

### Findings (important for the method design)
1. **Innovations do detect the shift, but weakly.** ANIS moves only 13–19% above nominal while ANEES more than doubles. A single 2 s window on one run detects the shift in at most 50–67% of runs, against an 18–20% chance baseline. Without that baseline the detector would look much better than it is.
2. **Innovation consistency is necessary but not sufficient for state consistency.** The nominal PF has a perfect ANIS (0.99) yet ANEES of 1.37: its posterior collapse is invisible in measurement space. After the shift, the PF's ANEES is 38× nominal while its ANIS is only 1.19×.
3. **Likely mechanism:** S_nu = H P H^T + R. With frequent updates, the sensor noise R dominates S_nu, so extra motion noise barely changes the innovation statistics. It does, however, accumulate in the unobserved pose error. This is a signal-to-noise and observability issue, and it should be derivable analytically.
4. **Consequence for Topic A.** A naive "conformalize the innovations" layer would inflate uncertainty by about 1.1×, when about 2.4× is needed. It would *under-correct*. The method must either
   (a) map measurement-space miscalibration to state space through the filter's sensitivity (e.g. how NIS responds to Q), or
   (b) use **sparse delayed labels** (loop closures, GNSS fixes) for state-space calibration, with innovations only as a fast shift detector.
   This gap is a sharper research question than we started with, and could become the paper's central claim.

### Next experiments
- **Sensitivity map:** sweep sensor noise R × shift magnitude × detector window W. Where is NIS informative and where is it blind? This could make a paper figure, and it checks finding 3.
- Baseline: classical adaptive noise estimation (Mehra covariance matching, Sage–Husa). Does it recover the ~2.4× needed? It is a mandatory baseline for reviewers.

---

## 2026-10-19: Exp 4, sensitivity sweep and analytic mechanism

**Script:** `04_sensitivity.py` → `04_sensitivity.png`, `04_sensitivity.json`. **Theory:** `THEORY.md`.
**Setup:** EKF only, 100 paired runs per condition. Sensor-noise std ×{0.25, 0.5, 1, 2, 4} × motion-noise shift std ×{1.5, 2, 3} × detector window {1, 2, 5, 10} s, plus a nominal (no-shift) run set per R for false alarms. 2000 runs in total, about 5 min on 4 cores.
**Regression:** after parametrizing `sim_core.py`, Exp 2 and Exp 3 both reproduce **bit-identical** JSON. (One check was killed by the session interruption and then rerun cleanly.)

### Theory vs Monte Carlo (post-shift ratios, 1 = consistent)

| R std× | shift std× | ρ_eff | NEES MC | T2 | T1 | NIS MC | T2 | T1 | power at 2 s / 10 s |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 2 | 0.133 | 2.29 | 2.27 | 2.23 | 1.46 | 1.45 | 1.54 | 0.41 / 0.95 |
| 0.5 | 2 | 0.031 | 2.36 | 2.34 | 2.37 | 1.25 | 1.24 | 1.26 | 0.16 / 0.65 |
| **1** | **2** | **0.0077** | **2.42** | 2.38 | 2.43 | **1.13** | 1.12 | 1.13 | **0.06 / 0.23** |
| 2 | 2 | 0.0021 | 2.50 | 2.39 | 2.47 | 1.07 | 1.06 | 1.07 | 0.03 / 0.08 |
| 4 | 2 | 0.0006 | 2.58 | 2.38 | 2.48 | 1.04 | 1.03 | 1.04 | 0.02 / 0.04 |
| 1 | 3 | 0.0077 | 4.84 | 4.77 | 4.83 | 1.33 | 1.32 | 1.35 | 0.24 / 0.79 |

Full 15-row table in the JSON. False-alarm rate in the nominal runs is 0.8% (α = 1%) for every R.

### Findings
1. **The mechanism is confirmed.** As SNR drops, the state miscalibration rises toward (1+c)/2 (2.5 for std×2, 5.0 for std×3) while the innovation excess falls like √ρ. At the lowest SNR the EKF is 2.6× overconfident, yet its NIS is only 4% above nominal.
2. **A single observable number predicts both.** Using only ρ_eff (from the filter's own prior, no ground truth), the scalar closed form T1 predicts the 2D EKF's NIS ratio within 1–8% and its NEES ratio within 1–10% across all 15 conditions (panel a). This is a strong, simple result.
3. **The linearized error-covariance analysis T2 is even closer for NIS:** within 1.5% everywhere. For NEES it is within about 1% for large shifts.
4. **Systematic deviation (report it honestly):** the measured NEES exceeds T2 by an additive ~0.1–0.25, growing with sensor noise. Relative to the ratio, this matters most for the smallest shift (std×1.5: +7–11%). The likely cause is EKF linearization error, which T2 ignores by construction and which grows as estimation errors grow. Even the nominal EKF had ANEES ratio 1.03 in Exp 2. It should be checked with an iterated EKF or an InEKF.
5. **Detection cost follows §1.5 of THEORY.md.** At our default sensor noise, a std×2 shift is caught by a 2 s window only 6% of the time per step (23% at 10 s). With 4× smaller sensor noise std (ρ ≈ 17× higher), the same window reaches 41% / 95%. The predicted power (dashed in panel c) tracks the MC but underestimates it slightly at mid SNR.
6. **Limitation of this sweep:** ρ_eff only spans 6e-4 to 0.13, so we have not tested the high-SNR branch, where theory says NIS becomes the *more* sensitive signal (crossover near ρ ≈ 1). Reaching it needs sensor std ×0.05 or less, or a larger Q.

### What this means for the paper
- We now have a candidate **theoretical contribution**: a closed-form, ground-truth-free predictor of how much a filter's state uncertainty is miscalibrated versus how much its innovations reveal. It is validated on a nonlinear 2D EKF.
- The method idea is sharper: estimate c from NIS, map it to the required state inflation via r_NEES(c, ρ_eff), and calibrate with conformal prediction. Sparse labels take over where ρ_eff says the innovations are uninformative.
- **Must do before claiming novelty:** literature check for prior statements of this duality (Bar-Shalom ch. 5; Mehra 1970/1972; Sage–Husa; work on detectability of model mismatch).

### Next
- Literature check above, which is critical before investing further.
- Extend the sweep to the high-SNR branch (ρ_eff up to ~10).
- Repeat for the PF; per Exp 3, its NEES can be wildly off while NIS stays near nominal, so the theory likely needs a sample-impoverishment term.
- Then build the method v1 and the adaptive-noise baselines (Mehra, Sage–Husa).

---

## 2026-10-19: Literature check 1 (see LIT_REVIEW_01.md)
- The theory's ingredients are classical (Heffes 1966, Sangsuk-Iam & Bullock 1990, Kalata 1984, Bar-Shalom & Birmiwal 1983). The explicit NEES/NIS duality was not found, but it is a short corollary, so it is a supporting lemma and not the headline.
- **New close competitor:** Weisman, Shlezinger & Laufer-Goldshtein, *Conformalized Kalman Filters*, arXiv 2609.27506 (Sep 2026). It is offline, needs labelled trajectories, assumes exchangeability, and has no Lie groups or real data. Online/adaptive is their stated future work, so we must move fast. Their code is our offline baseline.
- Repositioned claim: online conformal calibration under shift with **sparse delayed labels** on Lie groups, plus the duality lemma explaining why labels cannot be dropped.

---

## 2026-10-19: Exp 5, method v1: ACI with sparse, delayed labels

**Script:** `05_aci_sparse_labels.py` → `05_aci_sparse_labels.png`, `05_aci_sparse_labels.json` (scores cached in `cache/05_scores.npz`, git-ignored).
**Setup:** EKF, shift std×2 at 30 s, 200 test runs plus 100 disjoint nominal calibration runs. Region = Mahalanobis ellipsoid {NEES ≤ τ_t}, the same score as Weisman et al.'s cgkf. ACI = online quantile tracking on log τ, updated only when a true pose arrives (every L steps, D steps late). The filter does not depend on τ, so all 84 (L, D, η) configurations are replayed on the same stored scores.
**Reference values:** nominal τ = 7.81 (χ²₃, 95%); offline CP τ = 7.94; per-step oracle τ after the shift ≈ 19.9 (2.5× larger, consistent with the theory's r_NEES ≈ 2.4).

**Bugs found and fixed before interpreting (worth remembering)**
1. Look-ahead: with D = 0, the label for step t was processed before τ_t was recorded, so it was compared against an uninitialized value. Coverage was 100% and the volume ratio 10⁴⁵. Fix: report the region first, then receive feedback.
2. The centred moving average mixed pre-shift steps into the "recovery" metric (showing 0 s recovery) and produced an artificial drop at 60 s. Fix: trailing average; recovery = back above 0.92 *and staying there*.

**Results (coverage over 30–60 s, all steps; target 0.95)**

| label interval | η = 0.05, D = 0 | η = 0.2, D = 0 | η = 0.2, D = 2 s | recovery (η = 0.2, D = 0) |
|---|---|---|---|---|
| 0.1 s (every step) | 0.89 | **0.94** | 0.93 | 5.3 s |
| 0.2 s | 0.85 | 0.92 | 0.92 | 6.9 s |
| 0.5 s | 0.77 | 0.89 | 0.87 | 23.3 s |
| 1 s | 0.72 | 0.84 | 0.82 | never |
| 2 s | 0.69 | 0.77 | 0.76 | never |
| 5 s | 0.67 | 0.70 | 0.70 | never |
| 10 s | 0.66 | 0.67 | 0.67 | never |
| *nominal / offline CP* | *0.65 / 0.66* | | | *never* |

**Guarantee check:** the quantile-tracking bound |miscov_labelled − α| ≤ (B + η)/(η n) held in **every run** (0 violations across 5 configurations × 200 runs). It is tight with dense labels (bound 0.08 at L = 1) and **vacuous with sparse labels** (0.20 at 1 s, 0.62 at 10 s).

**Findings**
1. **Offline CP is no better than the nominal filter under shift** (0.66 vs 0.65). Calibrating on nominal data gives τ ≈ χ² quantile. This is the failure we expected for the Weisman et al. setting and is a clean motivating result.
2. *(Superseded by Exp 6: with η up to 1–2, plain ACI does recover at 1 s labels.)* **Plain ACI works only with dense labels.** With a true pose every 0.1–0.2 s and a large step (η = 0.2), coverage returns to ~0.93 within 5–7 s. **At one label per second or sparser, it never recovers**, and at 5–10 s it is indistinguishable from doing nothing. Realistic loop-closure or GNSS rates are 1–10 s, so plain ACI is **not usable in the regime that matters**. This is exactly the gap the method must fill.
3. **Step-size trade-off:** small η is stable but far too slow (η = 0.01 barely moves). Large η recovers faster but, with a 2 s delay, overshoots: the region becomes 1.5× the oracle volume at L = 1, a typical delayed-feedback oscillation.
4. **Regions stay too small after the shift** in every sparse-label setting (median volume 0.25–0.45 × oracle). Under-coverage is the dominant failure, not over-conservatism.
5. Delay of 2 s costs only ~0.01–0.02 coverage. **Label sparsity, not latency, is the bottleneck.**

**Why plain ACI is slow (diagnosis):** each label contributes only one bit (covered or not), and only ~1 in 3 post-shift labels is a miss. Going from τ = 7.8 to τ ≈ 20 needs Δlog τ ≈ 0.93, i.e. ≈ 5 misses at η = 0.2 or ≈ 19 at η = 0.05. At one label per second that takes 15–60 s, longer than the 30 s post-shift window.

**Method v2 directions (to test next, in this order)**
1. **Use the score magnitude, not just the miss bit.** When a label shows NEES = 25 against τ = 7.8, that one label already says "inflate ~3×". Options: conformal PID with a proportional term (Angelopoulos et al. 2023), or a rolling quantile of recent labelled scores with weighting.
2. **Strongly adaptive / expert-weighted step size:** DtACI (Gibbs & Candès, JMLR 2024) runs several η in parallel. It is a standard baseline that reviewers will expect.
3. **Fill the gaps between labels with the innovation signal:** use the theory (r_NIS → c → r_NEES via ρ_eff) as a prior inflation between labels. Exp 4 predicts this is weak at our ρ_eff ≈ 0.008 but useful at higher SNR. Measure it rather than assume.
4. Make the label process realistic: labels that arrive when the robot revisits places (correlated with the state) instead of periodically. This may break the "labelled steps ≈ all steps" property we saw here.

---

## 2026-10-19: Project moved to its permanent folder
The scratch workspace was copied here by the app. The Python environment was rebuilt with `uv sync`; the cached scores came across intact. **This folder is not a git repository** (the old one had no commits).

## 2026-10-19: Exp 6, method v2: magnitude-aware calibrators vs ACI and DtACI

**Script:** `06_magnitude_calibrators.py` (+ shared `calib_core.py`) → `06_magnitude_calibrators.png/.json`. Validation scores in `cache/06_val_scores.npz`.
**Protocol (new):** each method's hyperparameters are **tuned on a validation scenario** (shift std×1.5, 200 runs, disjoint seeds; objective = mean |coverage_t − 0.95|), separately per label interval and delay. The chosen configuration is then **evaluated once on the test scenario** (std×2, the Exp 5 runs).
**Checks:** `calib_core` reproduces the cached Exp 5 scores bit-for-bit, and the new driver reproduces Exp 5's ACI exactly.
**Methods:** ACI (log-τ quantile tracking); PI (our conformal-PID-style variant with scale-adaptive step and saturating integrator); RollQ (split-CP quantile of the last K labelled scores); WQ (weighted non-exchangeable CP, Barber et al. 2023); DtACI (Gibbs & Candès 2024, default hyperparameters).

### Test results, 2 s label delay (coverage target 0.95)

| label every | ACI | PI | RollQ | WQ | DtACI | nominal |
|---|---|---|---|---|---|---|
| 0.5 s | 0.92 (rec 7 s) | **0.93** (7 s) | 0.90 (29 s) | 0.92 (8 s) | 0.81 | 0.65 |
| 1 s | **0.92** (8 s) | 0.91 (15 s) | 0.88 (29 s) | 0.89 (29 s) | 0.79 | 0.65 |
| 2 s | 0.90 (15 s) | **0.91** (15 s) | 0.85 | 0.86 | 0.76 | 0.65 |
| 5 s | **0.91**\* (9 s) | 0.86\* | 0.80 | 0.77 | 0.69 | 0.65 |
| 10 s | 0.69 | 0.68 | **0.76** | 0.73 | 0.65 | 0.65 |

\* At 5 s, the tuned ACI (η = 2) and PI buy post-shift coverage with **pre-shift under-coverage** (0.90 and 0.91; regions 0.72–0.77 × oracle volume). That is an oscillating, bang-bang regime, not a real gain.

### Findings
1. **Correction to Exp 5.** "Plain ACI never recovers at ≥ 1 s labels" was an artifact of my step-size grid stopping at η = 0.2. With η ≈ 1 on log τ (chosen on validation), plain ACI recovers in ~8 s at 1 s labels with 0.92 post-shift coverage. Exp 5's mechanism explanation (one bit per label) still holds, but its numeric conclusion was too pessimistic.
2. **Using score magnitudes did *not* beat well-tuned ACI on coverage.** PI matches ACI and is more efficient at 1–2 s (post-shift volume 0.93–0.98 × oracle vs ACI's 1.28 overshoot at 1 s). RollQ and WQ are slow (≈ 29 s recovery: the window must turn over) and **over-conservative before the shift** (1.3–1.5 × oracle volume).
3. **Validity forces long memory in weighted CP.** A finite WQ threshold needs label weights summing to ≥ (1−α)/α = 19, so the half-life must be ≥ ~14 labels. This is a hard floor on adaptation speed for any finite-sample-valid weighted CP, and worth stating in the paper.
4. **DtACI is feedback-starved, not broken.** With ~60 labels per run its expert weights never concentrate (slow experts keep 30–40% weight after the shift). Caveat: its default η_w was derived for α-scale losses, and we apply it to log τ.
5. **Information floor.** At 10 s labels (3 post-shift labels in 30 s), *every* label-only method stays at 0.65–0.76. No amount of tuning can fix this. Labels alone cannot do it; a between-label signal is needed. **This is the opening for our contribution.**
6. **Protocol flaw found (fix next time).** The validation scenario *always* contains a shift, so tuning rewards aggressive step sizes and ignores stationary periods (see the 5 s row). The next protocol should tune on a **mixture of shift and no-shift runs** and report pre-shift efficiency alongside.
7. Chosen PI parameters at 5 s (lr = 0.5) still sit on the grid edge. Treat the PI numbers at 5 s as provisional.

### Next (method v3)
Add the **innovation channel between labels**: estimate c from a windowed NIS ratio, map it to the needed state inflation via r_NEES(c, ρ_eff) (THEORY.md), and use it to pre-inflate τ, with ACI correcting from labels. Evaluate across ρ_eff (sensor noise ×0.25–4). Exp 4 predicts little help at our default ρ_eff ≈ 0.008 and real help at ρ_eff ≳ 0.03. Use the mixed shift/no-shift validation protocol.

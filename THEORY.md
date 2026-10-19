# Why innovations are blind to process-noise shifts (and when they are not)

Working note for Topic A. Explains the Exp 3 observation that NIS rises about 1.13× while NEES rises about 2.4×, and predicts when an innovation-based method can work.

---

## 1. Scalar model

Random-walk state, direct measurement:

$$x_{k+1} = x_k + w_k,\quad w_k \sim \mathcal N(0, cQ), \qquad z_k = x_k + v_k,\quad v_k \sim \mathcal N(0, R).$$

The filter **assumes** process noise $Q$ (true value $cQ$; $c$ = squared std-shift factor, so std×2 means $c = 4$). It knows $R$ correctly. Without loss of generality set $R = 1$ and define the **signal-to-noise ratio**

$$\rho = Q/R .$$

### 1.1 What the filter believes (steady state)
Riccati fixed point with $P^- = P^+ + \rho$ and $P^+ = P^-/(P^- + 1)$:

$$P^- = \tfrac12\left(\rho + \sqrt{\rho^2 + 4\rho}\right), \qquad K = \frac{P^-}{P^- + 1}, \qquad P^+ = K .$$

### 1.2 What actually happens (fixed gain $K$, true noise $c\rho$)
Error recursion $e^-_{k+1} = e^+_k + w_k$, $e^+_k = (1-K)e^-_k - K v_k$. The stationary true variances are

$$P_t^- = \frac{K^2 + c\rho}{K(2-K)}, \qquad P_t^+ = (1-K)^2 P_t^- + K^2 .$$

### 1.3 The two consistency ratios
Expected NEES and NIS, relative to their nominal values:

$$\boxed{\;r_{\text{NEES}} = \frac{P_t^+}{P^+},\qquad r_{\text{NIS}} = \frac{P_t^- + 1}{P^- + 1}\;}$$

### 1.4 Asymptotics: the key result
**Low SNR ($\rho \to 0$; sensor noise dominates, $K \approx \sqrt\rho$):**

$$r_{\text{NEES}} \to \frac{1 + c}{2}, \qquad r_{\text{NIS}} = 1 + \frac{c-1}{2}\sqrt{\rho} + O(\rho) \to 1 .$$

**High SNR ($\rho \to \infty$; $K \to 1$):**

$$r_{\text{NEES}} \to 1, \qquad r_{\text{NIS}} \to c .$$

**Interpretation.** The two ratios move in *opposite* directions as SNR changes.
- When the filter leans on its motion model (low $\rho$), the state error grows by up to $(1+c)/2$. The innovation, however, is dominated by fresh sensor noise $R$, so its excess shrinks like $\sqrt\rho$. **The state is miscalibrated exactly where the innovations cannot see it.**
- When the filter leans on measurements (high $\rho$), innovations reveal the shift fully, but the state estimate is barely affected because each new measurement resets it.

For std×2 ($c = 4$), the low-SNR limit gives $r_{\text{NEES}} = 2.5$ and $r_{\text{NIS}} \approx 1 + 1.5\sqrt\rho$. At $\rho = 0.01$ this predicts 2.43 and 1.15. The 2D EKF in Exp 3 measured **2.41 and 1.13**.

### 1.5 Detectability cost
A window of $W$ steps with $d$ innovation DOF per step contains $Wd$ chi-square DOF scaled by $r_{\text{NIS}}$. Using the KL divergence between scaled and unscaled $\chi^2$ (≈ $\tfrac{d}{4}(r-1)^2$ per step for small $r - 1$), the window needed for reliable detection scales as

$$W^\star \;\propto\; \frac{1}{d\,(r_{\text{NIS}} - 1)^2} \;\approx\; \frac{4}{d\,(c-1)^2\,\rho}.$$

So halving the sensor-noise std (ρ ×4) shortens the required window 4×. A shift that is twice as large (in $c - 1$) shortens it 4× as well.

**Verified numerically** (scalar MC, 4000 runs, R = 1, c = 4):

| ρ | NEES MC / theory | NIS MC / theory |
|---|---|---|
| 0.01 | 2.416 / 2.425 | 1.149 / 1.150 |
| 0.1 | 2.268 / 2.266 | 1.467 / 1.469 |
| 1 | 1.830 / 1.829 | 2.334 / 2.342 |
| 10 | 1.233 / 1.232 | 3.542 / 3.535 |

---

## 2. Multivariate EKF: linearized error-covariance analysis (T2)

For the actual 2D EKF, propagate the **true** error covariance alongside the filter, using the same Jacobians and gains:

$$P_t^- = G P_t^+ G^\top + V Q_{\text{true}} V^\top, \qquad P_t^+ = (I - KH) P_t^- (I - KH)^\top + K R K^\top,$$

$$\mathbb E[\text{NEES}] = \operatorname{tr}(P^{-1} P_t), \qquad \mathbb E[\text{NIS}_j] = \operatorname{tr}\!\left(S_{\nu,j}^{-1}(H P_t^- H^\top + R)\right).$$

This is exact for a linear system with these gains. For the EKF it is a first-order approximation that is accurate while linearization errors are small.

### Effective SNR (observable online)
The filter's own prior gives a dimensionless gain-like quantity per update:

$$g = \tfrac{1}{m}\operatorname{tr}\!\left(R^{-1} H P^- H^\top\right), \qquad \rho_{\text{eff}} = \frac{g^2}{1 + g},$$

which inverts the scalar relation $P^-/R = g(\rho)$. **It needs no ground truth.** If the multivariate results collapse onto the scalar curves when plotted against $\rho_{\text{eff}}$, then one observable number predicts both how miscalibrated the state is and how visible that is in the innovations. → **Tested in Exp 4: confirmed.** Across 15 conditions (sensor std ×0.25–4, shift std ×1.5–3), T1 evaluated at the observable ρ_eff predicts the 2D EKF's NIS ratio within 1–8% and its NEES ratio within 1–10%. T2 predicts NIS within 1.5%. Measured NEES exceeds T2 by an additive ~0.1–0.25 (EKF linearization error, not modelled). Only ρ_eff ∈ [6e-4, 0.13] has been tested so far; the high-SNR branch is still open. See NOTEBOOK.md, Exp 4.

---

## 3. Implications for Topic A

1. **Innovation-only calibration is fundamentally limited in the low-SNR regime.** Conformalizing innovations gives a correction of about $r_{\text{NIS}}$ when about $r_{\text{NEES}}$ is needed. The worst case ratio of the two, $(1+c)/2$ vs ≈ 1, is reached exactly when the filter trusts its motion model.
2. **The fix is a known map, not a guess.** Given $\rho_{\text{eff}}$, both ratios are functions of the single unknown $c$. So a measured $r_{\text{NIS}}$ can be *inverted* to estimate $c$, and then mapped to the needed state-space inflation $r_{\text{NEES}}(c, \rho_{\text{eff}})$. The price: the noise in that estimate scales like $1/(\sqrt{Wd}\,\sqrt\rho)$, which (§1.5) is large at low SNR.
3. That suggests the method: **innovation-based inversion where it is informative, sparse ground-truth labels (loop closures, GNSS) where it is not**, with $\rho_{\text{eff}}$ deciding the weighting. A conformal layer on top supplies the finite-sample coverage guarantee.
4. Relation to classical adaptive filtering (Mehra 1970/72 covariance matching, Sage–Husa): those estimators also learn $Q$ from innovations, so they inherit the same $\sqrt\rho$ blindness. Our analysis quantifies *how badly* they fail in the low-SNR regime, which supports including them as baselines.

**Prior work (from LIT_REVIEW_01.md, 2026-10-19).** §2's true-error recursion is Heffes (IEEE TAC 1966) and Sangsuk-Iam & Bullock (IEEE TAC 1990). §1.1's steady state is Kalata's tracking-index theory (IEEE TAES 1984), and ρ is essentially the squared tracking index. §1.5's detection is classical NIS maneuver detection (Bar-Shalom & Birmiwal 1983). The explicit opposite-direction limits in §1.4, their reading as a limit on label-free calibration, and ρ_eff as a diagnostic were **not found in the literature checked**. Bar-Shalom et al. (2001) ch. 5.4/11 and Mehra (1972) still need checking. Present this as a lemma built on the cited results, not as a headline contribution.

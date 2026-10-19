# Literature check 1: is the NEES/NIS duality already published? Where does Topic A stand?

Date: 2026-10-19. Scope: about 14 targeted searches plus full abstracts or text of the key papers. **This is a targeted check, not yet a systematic review.** Items marked *unverified* need reading in full.

---

## Verdict in one paragraph

**The building blocks of our theory are classical, and we must cite them rather than claim them.** The true-error-covariance recursion under wrong Q/R (our "T2") is Heffes 1966 and Sangsuk-Iam & Bullock 1990. The steady-state random-walk gain as a function of Q/R is the tracking-index theory of Kalata 1984 (our ρ is essentially the squared tracking index). Detecting increased process noise from NIS with sliding or fading windows is classical maneuver detection (Bar-Shalom & Birmiwal 1983). The difficulty of estimating a small Q relative to R from innovations has been observed numerically (Zhang et al., IEEE Access 2020), and "passing chi-square tests does not prove correct tuning" is stated by Chen et al. 2023. **What I did *not* find stated anywhere** is the explicit opposite-direction duality: as SNR falls, state overconfidence tends to (1+c)/2 while the innovation excess vanishes like √ρ. I also did not find its use as a *fundamental limit on label-free calibration* or the ground-truth-free predictor ρ_eff. However, it is a short corollary of classical results, so a reviewer may call it folklore. **It should be a supporting lemma and motivation in the paper, not the headline contribution.**

---

## 1. Prior work that covers parts of our theory

| Our element | Prior work | Status |
|---|---|---|
| True error covariance of a filter with wrong Q, R (T2) | Heffes, *The effect of erroneous models on the Kalman filter response*, IEEE TAC 11(3), 1966. Sangsuk-Iam & Bullock, *Analysis of discrete-time Kalman filtering under incorrect noise covariances*, IEEE TAC 35(12), 1990. Also Gelb (ed.), *Applied Optimal Estimation*, 1974, on sensitivity analysis | **Classical. Cite; do not claim** |
| Closed-form steady state of the random-walk KF vs Q/R (T1, §1.1) | Kalata, *The tracking index: a generalized parameter for α-β and α-β-γ target trackers*, IEEE TAES 1984; Urbano, Kalata & Kam 2012 | **Classical. Cite** |
| Detecting a Q increase from windowed NIS | Bar-Shalom & Birmiwal 1983 (fading-memory NIS maneuver detection); Kadan et al., AMOS 2023 (comparison of NIS detectors) | **Classical. Our detection-power analysis is a restatement** |
| Q poorly identifiable when Q ≪ R | Zhang, Pattipati et al., *On the identification of noise covariances and adaptive Kalman filtering: a new look at a 50-year-old problem*, IEEE Access 2020 (Q estimate varies widely when small relative to R; ill-conditioned LS). Mehra 1970; Odelson, Rajamani & Rawlings 2006 (ALS identifiability) | Observed numerically; **no closed-form SNR law found** |
| NIS/NEES chi-square tests are insufficient for correct tuning | Chen, Biggie, Ahmed, Julier, Heckman, arXiv 2306.07225 (2023, submitted to IEEE TAES). Their 2018 FUSION paper "Weak in the NEES?" treats NIS as the no-ground-truth substitute for NEES, with no warning that it is blind to Q (checked in the full text) | General statement; **no mechanism or SNR dependence** |
| Credibility metrics under noise/model mismatch | Li, Zhao & Jilkov 2002/2006 (noncredibility index); Yan, Zhan, Sun & Hsu, arXiv 2509.03311 (2025–26, NCI + NLL + energy score to diagnose noise vs model mismatch) | Diagnostic; needs truth; **does not treat NIS vs NEES** |
| NIS consistency ≠ NEES consistency | Or, arXiv 2512.18508 (2025–26): innovation statistics contract under *gating*. A different mechanism (selection), not process-noise mismatch | Adjacent; cite as another route to the same warning |

**Not checked yet (must read):** Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking and Navigation* (2001), ch. 5.4 and ch. 11. A textbook exercise on mismatched steady-state filters may already contain the duality. Also Mehra 1972, *Approaches to adaptive filtering*.

---

## 2. Competitors to Topic A itself (conformal calibration of filters)

### ⚠ Closest: Weisman, Shlezinger & Laufer-Goldshtein, *Conformalized Kalman Filters for State Estimation with Trustworthy Confidence Regions*, arXiv 2609.27506 (23 Sep 2026; parts at ICASSP 2026)
- Conformal calibration built on the KF/EKF/UKF posterior moments, with finite-sample sample-wise and trajectory-wise coverage.
- **Needs ground-truth state trajectories for calibration.** Assumes **exchangeability** between calibration and test trajectories.
- **No online adaptation and no distribution shift.** The authors list adaptive/weighted conformal methods as *future work*.
- No Lie groups or SE(2)/SE(3), no robotics localization, simulation only (pendulum, Lorenz). No NIS/NEES or label-free discussion.
- Code on GitHub. **This must be a baseline in our paper.**

**Implication:** the offline, labelled, exchangeable version of Topic A is now published. Our remaining space is exactly their stated future work, plus what they do not touch: **online, under shift, with scarce or no labels, on Lie groups, on real robot data.** They will likely pursue the adaptive extension, so **speed matters**. Target a fast venue (RA-L) and post to arXiv early.

### Others
- Marques, Ghaffari & Berenson, *CLAPS*, arXiv 2512.10294: conformal sets on Lie groups for **action/dynamics prediction**, not recursive state estimation. Cite for the SE(2) score.
- Pathak & Seifi, *Adaptive Geodesic Conformal Prediction for Egocentric Camera Pose*, arXiv 2605.00233: geodesic SE(3) score. "Adaptive" here means difficulty-conditioned, not online under shift. VO wrapper, not a filter. Cite for the SE(3) score.
- AdaptNC (ICLR 2026): online adaptation of nonconformity *score and threshold*; generic, needs online labels.
- *Adaptive conformal inference by particle filtering under HMMs*, arXiv 2411.01558: ACI with hidden states, using particles as pseudo-labels. **This is the closest method to the label-free idea.** Read in full.
- Gibbs & Candès 2021 (ACI): the core online tool. It needs a label every step, which is exactly what localization lacks.

---

## 3. Revised positioning (recommendation)

**Paper claim (draft v2):**
> *Online conformal calibration of recursive state estimators under distribution shift with sparse, delayed ground truth, on Lie groups.* We show analytically that innovation-only self-assessment fails exactly when the filter trusts its motion model (supporting lemma, building on Heffes 1966 and Kalata 1984). We then propose a calibrator that adapts quickly using innovations where they are informative (high ρ_eff) and anchors to sparse labels (loop closures, GNSS/landmark fixes) where they are not, with a long-run coverage guarantee that degrades gracefully with label sparsity.

**Contribution ranking**
1. **Method plus guarantee** (headline): ACI-style online calibration with *sparse delayed* labels, with innovation-informed adaptation, on SE(2)/SE(3).
2. **Analysis** (supporting): the NEES/NIS duality as a quantitative limit on label-free calibration, with ρ_eff as a ground-truth-free diagnostic. Present it as an explicit, verified corollary of classical results and cite them prominently.
3. **Evaluation:** real public datasets plus the Weisman et al. baseline (offline CP) plus classical adaptive noise estimation (Mehra/ALS, Sage–Husa).

**What reviewers will ask, and our answer**
- *"Isn't this just Weisman et al. 2026 made online?"* They need exchangeable labelled trajectories. We handle shift with sparse labels and prove why labels cannot be dropped entirely.
- *"Isn't the duality known?"* The ingredients are, and we cite them. The explicit limit and its use for calibration were not found. Re-check Bar-Shalom before submission.
- *"Why not just adapt Q (Mehra, Sage–Husa)?"* Our lemma predicts those estimators inherit the √ρ blindness at low SNR. We must demonstrate this experimentally.

---

## 4. Action items
1. **Read in full:** Weisman et al. 2609.27506; the ACI-PF HMM paper 2411.01558; Bar-Shalom ch. 5.4 and 11; Zhang/Pattipati IEEE Access 2020.
2. Clone the Weisman et al. code and reproduce one result. It becomes our offline baseline.
3. Update `THEORY.md` to cite Heffes, Sangsuk-Iam & Bullock, and Kalata at the points where we use their results.
4. Then start method v1: ACI with sparse delayed labels in our 2D simulator. First question: how fast does coverage recover after the t = 30 s shift as label rate goes from every step to every 100 steps?

---

## Sources
- [Heffes 1966 (Semantic Scholar record)](https://www.semanticscholar.org/paper/The-effect-of-erroneous-models-on-the-Kalman-filter-Heffes/6d05ab2b09a41735c211fc9257e2ede0bd7d6ae3) · [NASA JPL TR 32-1319, Modeling Errors in Kalman Filters](https://ntrs.nasa.gov/api/citations/19700018935/downloads/19700018935.pdf) · [Sangsuk-Iam & Bullock 1990](https://iaorifors.com/paper/4569)
- [Kalata, The Tracking Index](https://researchdiscovery.drexel.edu/esploro/outputs/journalArticle/The-Tracking-Index-A-Generalized-Parameter/991019173672304721) · [Urbano, Kalata & Kam 2012](https://researchwith.njit.edu/en/publications/optimal-tracking-index-relationship-for-random-and-deterministic-/)
- [Zhang/Pattipati et al., IEEE Access 2020 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8638515/) · [Odelson et al. ALS tech report](https://sites.engineering.ucsb.edu/~jbraw/jbrweb-archives/tech-reports/twmcc-2003-04.pdf)
- [Chen et al. 2018, Weak in the NEES?](https://arxiv.org/pdf/1807.08855) · [Chen et al. 2023, chi-squared auto-tuning](https://arxiv.org/abs/2306.07225)
- [Yan et al., Credible UQ under noise and model mismatch](https://arxiv.org/abs/2509.03311) · [Or, Selection-induced contraction of innovation statistics](https://arxiv.org/abs/2512.18508)
- [Kadan et al., AMOS 2023 (NIS maneuver detectors)](https://amostech.com/TechnicalPapers/2023/SDA/Kadan.pdf) · [METU lecture on fading-memory NIS](https://users.metu.edu.tr/home405/umut/wwwhome/ee793/files/METULecture4.pdf)
- [Weisman et al., Conformalized Kalman Filters (2026)](https://arxiv.org/abs/2609.27506)
- [Marques et al., CLAPS](https://arxiv.org/pdf/2512.10294) · [Pathak & Seifi, Adaptive Geodesic CP](https://arxiv.org/abs/2605.00233) · [AdaptNC, ICLR 2026](https://iclr.cc/virtual/2026/10019200) · [ACI by particle filtering under HMMs](https://arxiv.org/pdf/2411.01558) · [Gibbs & Candès 2021](https://arxiv.org/pdf/2106.00170)

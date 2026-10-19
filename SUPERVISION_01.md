# Supervision Meeting 1: Choosing a Q1-worthy topic

Date: 2026-10-18
Goal: a first-author paper in a Q1 robotics journal, done entirely on this laptop.

---

## 1. Honest assessment of our constraints

| Constraint | Consequence | How we work around it |
|---|---|---|
| No GPU (Iris Xe, 16 GB RAM) | Large learned models (big neural odometry, foundation models) are out | Pick a **method/theory contribution**, not a "bigger network" contribution |
| No robot hardware | Reviewers at T-RO/RA-L distrust simulation-only results | Use **public real-world datasets** (all run on CPU): KITTI, EuRoC MAV, TUM-VI, Intel Lab / Freiburg 2D laser logs, UTIAS multi-robot dataset |
| Working alone | No co-author to catch errors | Strict reproducibility: every figure produced by a script in this repo, fixed seeds |

**Rule of thumb from experience:** a Q1 paper from a small setup wins on *a sharp idea plus a guarantee plus convincing evaluation on real datasets*. It does not win on scale.

### Target venues (check current quartile in SJR/JCR before submitting)
- **IEEE Robotics and Automation Letters (RA-L)**: 6–8 pages, fast, good first target
- **Robotics and Autonomous Systems (Elsevier)**: welcomes careful methodological work
- **Autonomous Robots (Springer)**
- **IEEE Transactions on Robotics (T-RO)**: the stretch goal; needs a strong theory plus real-data story
- Also relevant: *IEEE Transactions on Automation Science and Engineering*, *IEEE Transactions on Aerospace and Electronic Systems* (filtering-heavy work)

---

## 2. What the recent literature says (quick scan, Oct 2026)

Recurring open problems in probabilistic state estimation:
1. **Learned and classical estimators are overconfident.** Their covariances are not calibrated, so downstream planners trust them too much (e.g. MUSE, arXiv 2605.17421).
2. **Conformal prediction is entering robotics fast**, but mostly for *dynamics/action* prediction or *offline* calibration:
   - Marques, Ghaffari, Berenson: CLAPS, conformal prediction sets on Lie groups for actions (arXiv 2512.10294)
   - Local conformal calibration of dynamics uncertainty (arXiv 2409.08249)
   - Task-aware conformal state estimation for planning (NeurIPS 2026, "CMBS-α")
3. **Adaptive/online conformal inference** (Gibbs & Candès 2021) handles distribution shift, but has barely been combined with recursive filters. The closest is ACI with particle filters for HMMs (arXiv 2411.01558). My scan found **no Kalman/invariant-filter + online conformal treatment for robot localization.**
4. Invariant/equivariant filtering on Lie groups is mature and mathematically crowded (strong groups at Mines Paris, ANU, Michigan).
5. Noise-covariance adaptation is active but crowded (e.g. arXiv 2510.24508).

> Caveat: this is a 1-hour scan, not a systematic review. Gap claims must be verified (Section 5, task 1) before we commit.

---

## 3. Candidate topics, ranked

### ★ Topic A (recommended): Online conformal calibration of recursive state estimators under distribution shift
**Idea.** Wrap an EKF / invariant EKF (and optionally a particle filter) with an *adaptive conformal* layer that outputs pose **sets on SE(2)/SE(3)** with a guaranteed long-run coverage rate. The rate must hold even when the noise statistics drift (wheel slip, sensor degradation, entering a featureless corridor).

**The hard, publishable part.** In localization the true state is *never observed online*, so standard conformal calibration has no labels. Candidate solutions to study:
- calibrate in **measurement space** using innovations (observable!) and map coverage back to state space through the filter's geometry;
- use **delayed/partial ground truth** (loop closures, GNSS fixes, landmark re-observations) as sparse labels;
- prove a coverage bound that degrades gracefully with label sparsity.

**Why it fits us.** Pure CPU work. It builds directly on the EKF/PF code from Lesson 1. It has a theory component (a coverage guarantee) and evaluates on public datasets.

**Contribution statement (draft):** *"The first recursive state estimator with distribution-free, online coverage guarantees on Lie groups that requires no online ground truth."*

**Risks.** (i) The NeurIPS 2026 CMBS-α paper or the Marques/Berenson line may already cover part of this, so we must read them first. (ii) The guarantee may need assumptions we must state honestly.

### Topic B: Robust non-Gaussian filtering with heavy-tailed / multimodal measurement models
Student-t or Gaussian-mixture likelihoods inside an invariant filter, plus consistency analysis. Feasible, but incremental unless we find a sharp theoretical angle.

### Topic C: Differentiable particle filters with calibrated uncertainty
Interesting, but training needs a GPU to be competitive, and the area is dominated by groups with large compute. **Not recommended** for this setup.

### Topic D: Invariant-filter consistency theory
Elegant, but hard to beat established groups on their own ground in a first paper. **Not recommended** as a first project.

---

## 4. Proposed timeline for Topic A (about 9–12 months to submission)

| Months | Milestone |
|---|---|
| 0–1 | Systematic literature review (≥40 papers), confirm the gap, write a 2-page related-work draft |
| 1–2 | Reproduce baselines: EKF, InEKF, PF on 2D simulation; measure NEES/ANEES consistency |
| 2–4 | Method v1: innovation-based adaptive conformal layer; simulation with injected noise shift |
| 3–5 | Theory: coverage bound (with sparse/delayed labels); sanity-check it numerically |
| 5–7 | Real data: UTIAS / Intel Lab (2D), then KITTI or EuRoC (3D); compare against baselines (adaptive EKF, covariance inflation, offline split conformal, CLAPS-style) |
| 7–9 | Ablations, failure cases, write the paper (RA-L format) |
| 9–12 | Internal review, revise, submit; release the code |

---

## 5. Your tasks before our next meeting (2 weeks)

1. **Read closely and summarize (1 page each, template below):**
   - Gibbs & Candès, *Adaptive Conformal Inference Under Distribution Shift* (NeurIPS 2021)
   - Marques et al., *Lies We Can Trust* (CLAPS), arXiv 2512.10294
   - *Adaptive Conformal Inference by Particle Filtering under HMMs*, arXiv 2411.01558
   - The NeurIPS 2026 *Task-Aware Conformal State Estimation* paper (CMBS-α)
   - Barrau & Bonnabel, *The Invariant Extended Kalman Filter as a Stable Observer* (IEEE TAC 2017)
2. **Search arXiv + Google Scholar** for: `conformal` AND (`Kalman` OR `state estimation` OR `localization` OR `SLAM`), 2023–2026. Log every hit in a spreadsheet: title, venue, does it handle *online* + *no ground truth* + *Lie group*?
3. **Code:** extend `01_localization.py` to compute **NEES** (normalized estimation error squared) and show that the EKF becomes *inconsistent* when you double the motion noise halfway through the run. This plot becomes Figure 1 of our motivation.
4. **Refresh the math** if needed: Probabilistic Robotics ch. 3 and 7; Solà et al., *A micro Lie theory for state estimation in robotics* (arXiv 1812.01537); Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* (arXiv 2107.07511).

### Paper summary template
```
Problem:            What exactly do they solve?
Assumptions:        Exchangeability? Ground truth available? Gaussian?
Method (3 lines):
Guarantee:          What is proven, under what conditions?
Experiments:        Datasets, baselines, metrics
Limitation we can exploit:
```

---

## 6. Ground rules
- **Integrity.** I am an AI acting in a supervisor role. I can guide, critique, derive and code with you, but I cannot be an author, and most journals (IEEE, Elsevier, Springer) require you to **disclose AI assistance**. Try to get a human co-advisor or collaborator at your institution to review the work before submission.
- Never claim novelty without a documented search.
- Every number in the paper must come from a script in this repository.

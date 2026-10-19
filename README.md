# robotics-project-01

Online, distribution-free calibration of probabilistic robot state estimators (EKF / particle filter) under distribution shift, with sparse and delayed ground truth.
Simulation-only, CPU-only research code (2D range-bearing localization).

## Status
Work in progress toward a journal paper. The current state of findings is in [NOTEBOOK.md](NOTEBOOK.md).

## Layout
| File | What it is |
|---|---|
| `sim_core.py` | World, motion and sensor models, EKF and particle filter |
| `calib_core.py` | Score simulation, causal calibration driver, metrics |
| `01_localization.py` | Lesson: EKF vs particle filter localization |
| `02_nees_consistency.py` | Filter consistency (NEES) under a motion-noise shift |
| `03_nis_detection.py` | Can innovations (NIS) detect the shift without ground truth? |
| `04_sensitivity.py` | SNR sweep; theory (THEORY.md) vs Monte Carlo |
| `05_aci_sparse_labels.py` | Method v1: adaptive conformal inference with sparse, delayed labels |
| `06_magnitude_calibrators.py` | Method v2: PI, rolling/weighted CP, DtACI vs ACI (validation-tuned) |
| `NOTEBOOK.md` | Lab notebook: setup, numbers and conclusions for every experiment |
| `THEORY.md` | NEES/NIS duality under process-noise mismatch |
| `LIT_REVIEW_01.md` | Literature check and positioning |
| `SUPERVISION_01.md` | Initial research plan |

## Reproduce
Requires [uv](https://docs.astral.sh/uv/).
```bash
uv sync
uv run python 02_nees_consistency.py
```
Every script writes its figure (`.png`) and numbers (`.json`) next to itself. Seeds are fixed. `cache/` holds regenerable simulation scores and is not tracked.

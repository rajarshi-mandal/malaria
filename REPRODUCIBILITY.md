# Reproducibility Guide

**A Real-Data-Calibrated Benchmark for Insecticide-Treated-Net (ITN) Allocation**

This repository reproduces every table and figure in the accompanying paper. All
experiments run on a **compartmental transmission simulator calibrated to public
malaria surveillance data**. The released region tables ship with the fitted
parameters, so the headline results reproduce **without redownloading the raw
data sources**. The code is **CPU only** so no GPU or CUDA is required.

---

## 1. Requirements

- **Python 3.13**
- Dependencies are pinned in [`requirements.txt`](requirements.txt) to the exact
  versions used to produce the reported results:

  | Package | Version | Role |
  |---|---|---|
  | numpy | 2.4.x | arrays / numerics |
  | scipy | 1.16.x | optimization, statistics |
  | pandas | 3.0.x | data handling |
  | scikit-learn | 1.7.x | decision-tree distillation |
  | numba | 0.65.x | JIT-compiled simulator |
  | torch | 2.8.x (CPU) | policy networks |
  | gymnasium | 1.3.x | RL environment API |
  | stable-baselines3 | 2.9.x | PPO / DQN |
  | SALib | — | Sobol / LHS–PRCC sensitivity analysis |
  | shap, matplotlib | — | interpretability, plotting |

### Setup

```bash
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Single-threaded execution (important)

Run all heavy jobs single-threaded. This is both **faster** for this workload
(many small operations; numba provides the speedup) and
**avoids an OpenMP/MKL deadlock** observed on some CPUs. Each script also calls
`torch.set_num_threads(1)` internally.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1     # PowerShell: $env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1
```

To use additional cores, run several scripts **as separate single-threaded
processes** rather than increasing the thread count within one process.

---

## 2. Repository layout

```
malaria/                                  # repository root
├── code/                                 # all analysis and experiment scripts
├── malaria-data-for-modeling-dynamics/   # cleaned public source data
├── malaria-gan-outputs/                  # generated resistance-trajectory library
├── results/                              # reference result tables (.csv) and arrays (.npz)
├── figures/                              # reference figures (.png)
├── requirements.txt
└── REPRODUCIBILITY.md                    # this file
```

**Working directory.** Run all commands below from the repository root (scripts
reference data and released tables by relative path). Scripts write their outputs
to the working directory. The `results/` and `figures/` folders contain reference
copies of those outputs for comparison.

---

## 3. Data sources

All inputs are public. Cleaning scripts convert each raw source into the tidy form
used by the simulator.

| Source | Use | Cleaning script |
|---|---|---|
| Malaria Atlas Project (PfPR, incidence) | calibration targets | `nt-malaria-atlas-project-cleaning.py` |
| WHO Global Health Observatory (ITN access, suspected cases) | coverage, case seeding | `nt-the-global-health-observatory-cleaning.py` |
| NASA Giovanni (vegetation / evapotranspiration) | seasonal biting rate | `nt-earthdata-giovanni-cleaning.py` |
| UNICEF MICS / DHS | net age, covariates | `nt-multiple-indicator-cluster-surveys-cleaning.py`, `nt-demographic-and-health-surveys-cleaning.py` |
| IR-Mapper (pyrethroid bioassays) | insecticide-resistance trajectories | `nt-ir-mapper-anopheles-cleaning.py` |

**Released calibrated benchmark tables** (in `results/`):

| Table | Description |
|---|---|
| `calibrated_regions.csv` | 8 real admin-1 regions, SEITR calibration |
| `extended_calibrated_regions.csv` | 8 real admin-1 regions, SEITAR calibration |
| `real_regions_n50.csv` | 50 real admin-1 regions, calibrated |

---

## 4. Core simulator

- **`extended_ode.py`** — the SEITR / SEITAR compartmental model with a
  fourth-order Runge–Kutta integrator.
- **`fast_sim.py`** — a numba-JIT rollout that is **bit-identical** to the reference
  model (verified by `verify_fast.py`, relative error 0) and roughly **288× faster**.
  This is what makes the rollout planners tractable at scale.
- **`fast_sim_endo.py`** — a metapopulation extension adding human mobility, net
  attrition, and endogenous insecticide resistance. It **reduces bit-identically**
  to the base SEITAR model when the extensions are disabled (verified by
  `verify_endo.py`, absolute error 0).

---

## 5. Result → script map

Every paper table and figure maps to a script and its output artifact.

| Paper element | Script(s) | Output |
|---|---|---|
| SEITR / SEITAR calibration (figure) | `calibrate.py`, `plot_calibration.py`, `plot_extended.py` | `fig_calibration.png`, `fig_extended_calibration.png` |
| R₀ and forward bifurcation (figure) | `t2b_stability.py` | `fig_bifurcation.png` |
| Global sensitivity of R₀ (figure) | `t2d_sensitivity.py` | `fig_sensitivity.png` |
| Learned method ladder, SEITR / SEITAR (table) | `calibrated_experiment.py`, `extended_experiment.py` | `calibrated_results.npz`, `extended_results.npz` |
| Heuristic + planner ladder (table) | `baselines_heuristic.py`, `baselines_planner.py` | `ws1_heuristics_*.npz`, `ws234_planners_*.npz` |
| Environment-component ablations (table) | `c_ablations.py` | `ablation_results.csv` |
| 50-region real-data benchmark | `build_real_regions.py`, `experiment_scaled.py` | `ws56_scaled_n50.npz` |
| Calibration-uncertainty robustness (figure) | `t4_calibration_uncertainty.py` | `fig_calib_uncertainty.png` |
| Cost-effectiveness / cost-per-QALY (figure) | `t3_cost_qaly.py` | `fig_cost_qaly.png` |
| External effect-size + temporal validation (figure) | `external_validation.py`, `temporal_validation.py` | `fig_external_validation.png`, `fig_temporal_validation.png` |
| Temporal change-direction subgroup | `temporal_subgroup.py` | `temporal_subgroup.csv` |
| ARMOR-Adapt surveillance-driven adaptation (table/figure) | `meta_adaptive.py` | `meta_adaptive_n40.npz`, `fig_adaptive.png` |
| ARMOR-IDA dual control (table/figure) | `meta_bayesian.py` | `meta_bayesian_n60.npz`, `fig_bayesian.png` |
| λ-necessity ablation | `meta_bayesian_ablation.py` | `meta_bayesian_ablation.npz` |
| Real-data panel / confounding analysis | `real_env.py`, `real_env_alloc.py` | `real_env_fit.npz`, `real_env_alloc.npz` |
| Pooled effect sizes, 95% bootstrap CIs, Holm-corrected tests | `stats_hardening.py` | (stdout) |

---

## 6. One command reproduction of the headline results

Run from the repository root with the single-threaded environment set (Section 1):

```bash
# 1. Calibration and the learned method ladder
python calibrated_experiment.py && python extended_experiment.py

# 2. Non-learned baselines, simulator-based planners, and the 50-region benchmark
python baselines_heuristic.py --model seitr
python baselines_planner.py  --model seitr
python build_real_regions.py --n-regions 50 && python experiment_scaled.py --n-regions 50

# 3. Beyond-simulator validation
python external_validation.py && python temporal_validation.py && python temporal_subgroup.py

# 4. Adaptive allocation and dual-control results
#    (headline regime: sigma_obs=0.6, kappa_0=0.05, lambda=8, K=8 campaigns)
python meta_adaptive.py --n-worlds 40 --out meta_adaptive_n40.npz
python meta_bayesian.py  --n-worlds 60 --campaigns 8 --sigma-obs 0.6 --kap0 0.05 --lam 8 --out meta_bayesian_n60.npz
python meta_bayesian_ablation.py --n-worlds 40

# 5. Pooled statistics (effect sizes, bootstrap CIs, Holm-corrected paired tests)
python stats_hardening.py

# 6. Figures
python meta_plots.py
```

---

## 7. Determinism and expected outputs

- Every script fixes its random seeds internally, so results are reproducible run
  to run.
- **Held-out evaluation** uses environment seeds that are disjoint from the training
  seeds; reported numbers are held-out unless stated otherwise.
- Each script prints its result table to standard output and writes a `.npz` or
  `.csv` artifact. Compare against the reference copies in `results/` (and figures
  in `figures/`). Small numerical differences across platforms are expected in
  floating-point summaries but do not affect the reported conclusions.

---

## 8. Data availability, licensing, and citation

- **Data.** All source datasets are publicly available from the providers listed in
  Section 3 and remain subject to their respective terms of use. This repository
  redistributes only derived, cleaned tables and the fitted benchmark parameters.
- **Code license.** See the `LICENSE` file in the repository root.
- **Citation.** If you use this benchmark, simulator, or code, please cite the
  accompanying paper (full citation to be added upon publication).

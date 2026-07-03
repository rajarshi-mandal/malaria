"""
WS7 -- calibration uncertainty.

The SEITAR calibration is a central contribution (it matches prevalence AND
incidence). We show it is not brittle:

 1. Bootstrap the observed calibration TARGETS (parasite prevalence + clinical
    incidence) per region with realistic observational noise, refit (biting rate a,
    symptomatic fraction p) each time (fast njit steady), and report per-region 95%
    ranges of the fitted parameters.

 2. Robustness of the allocation result: draw K calibrated parameter sets from those
    bootstrap distributions, build the SEITAR env for each, and evaluate the
    allocation advantage of simulator-based optimization (greedy / MPC) over the
    population-proportional standard of practice. Since greedy/MPC ~ PPO ~ oracle at
    this scale (WS2-4), the persistence of this advantage under calibration
    uncertainty implies the learned advantage is robust too (no per-draw retraining).

Run: python t4_calibration_uncertainty.py [--boot 200] [--draws 30]
Outputs: console + calib_uncertainty.csv + fig_calib_uncertainty.png
"""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fast_sim as fs
import baselines_planner as BP
from experiment_scaled import ScaledSeitarEnv, run_episode

RNG = np.random.default_rng(0)


def fit_ap(prev_obs, inc_obs, cov, mult, treat=10):
    rho = 1.0 / treat

    def resid(theta):
        a, p = theta
        prev, inc = fs.seitar_steady(a, p, cov, mult, rho)
        return [(prev - prev_obs) / max(prev_obs, 1e-3),
                (inc - inc_obs) / max(inc_obs, 1e-3)]
    sol = least_squares(resid, x0=[1.0, 0.3], bounds=([0.05, 0.02], [15.0, 0.98]),
                        diff_step=0.05, xtol=1e-3, ftol=1e-3, max_nfev=80)
    return sol.x[0], sol.x[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="extended_calibrated_regions.csv")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--rel-noise", type=float, default=0.15)
    args = ap.parse_args()

    base = pd.read_csv(args.regions)
    n = len(base)
    print(f"Calibration uncertainty on {n} regions | {args.boot} bootstraps | "
          f"obs noise {args.rel_noise:.0%}", flush=True)

    # ---- 1. bootstrap the fitted parameters ----
    boot_a = np.zeros((n, args.boot))
    boot_p = np.zeros((n, args.boot))
    for j, (_, r) in enumerate(base.iterrows()):
        cov, mult = r.ITN_coverage, r.pop_multiplier
        for bt in range(args.boot):
            prev = r.PfPR_obs * np.exp(RNG.normal(0, args.rel_noise))
            inc = r.incidence_obs * np.exp(RNG.normal(0, args.rel_noise))
            a, p = fit_ap(prev, inc, cov, mult, int(r.treatment_seeking))
            boot_a[j, bt] = a; boot_p[j, bt] = p
    rows = []
    for j, (_, r) in enumerate(base.iterrows()):
        rows.append(dict(region=r.region,
                         a_fit=r.biting_rate, a_lo=np.percentile(boot_a[j], 2.5),
                         a_hi=np.percentile(boot_a[j], 97.5),
                         p_fit=r.p_sympt, p_lo=np.percentile(boot_p[j], 2.5),
                         p_hi=np.percentile(boot_p[j], 97.5)))
    tab = pd.DataFrame(rows)
    tab.to_csv("calib_uncertainty.csv", index=False)
    print("\nPer-region fitted parameters (95% bootstrap ranges):")
    print(tab.round(3).to_string(index=False))
    a_cv = (np.std(boot_a, axis=1) / np.mean(boot_a, axis=1)).mean()
    p_cv = (np.std(boot_p, axis=1) / np.mean(boot_p, axis=1)).mean()
    print(f"\nMean coefficient of variation: biting rate {a_cv:.1%}, symptomatic fraction {p_cv:.1%}")

    # ---- 2. robustness of the allocation advantage ----
    print(f"\nEvaluating static vs greedy/MPC over {args.draws} sampled calibrations "
          f"({args.n_eval} envs each)...", flush=True)
    plan_rng = np.random.default_rng(1)
    adv_greedy, adv_mpc = [], []
    for d in range(args.draws):
        reg = base.copy()
        reg["biting_rate"] = [boot_a[j, RNG.integers(args.boot)] for j in range(n)]
        reg["p_sympt"] = [boot_p[j, RNG.integers(args.boot)] for j in range(n)]
        budget = 20000.0
        pop = reg.pop_multiplier.to_numpy()
        POP_STATIC = (pop / pop.sum() * budget).astype(np.float32)
        sred, gred, mred = [], [], []
        for i in range(args.n_eval):
            seed = 20000 + d * 1000 + i
            e = ScaledSeitarEnv(reg, budget, np.random.default_rng(seed))
            base_b = run_episode(e, lambda o, k: np.zeros(n, np.float32))
            e = ScaledSeitarEnv(reg, budget, np.random.default_rng(seed))
            s = BP.run_planner_episode(e, lambda env, k: POP_STATIC.copy(), n)
            e = ScaledSeitarEnv(reg, budget, np.random.default_rng(seed))
            g = BP.run_planner_episode(e, lambda env, k: BP.greedy_alloc(env, "seitar", budget, n), n)
            e = ScaledSeitarEnv(reg, budget, np.random.default_rng(seed))
            mp = BP.run_planner_episode(e, lambda env, k: BP.mpc_alloc(env, "seitar", budget, n, plan_rng), n)
            sred.append((base_b - s) / base_b * 100)
            gred.append((base_b - g) / base_b * 100)
            mred.append((base_b - mp) / base_b * 100)
        adv_greedy.append(np.mean(gred) - np.mean(sred))
        adv_mpc.append(np.mean(mred) - np.mean(sred))
        if (d + 1) % 5 == 0:
            print(f"  ...{d+1}/{args.draws} draws", flush=True)

    adv_greedy = np.array(adv_greedy); adv_mpc = np.array(adv_mpc)
    print(f"\nAllocation advantage over static across calibration uncertainty:")
    print(f"  greedy: {adv_greedy.mean():.2f} pp  95% CI "
          f"[{np.percentile(adv_greedy,2.5):.2f}, {np.percentile(adv_greedy,97.5):.2f}]  "
          f"min {adv_greedy.min():.2f}")
    print(f"  MPC:    {adv_mpc.mean():.2f} pp  95% CI "
          f"[{np.percentile(adv_mpc,2.5):.2f}, {np.percentile(adv_mpc,97.5):.2f}]  "
          f"min {adv_mpc.min():.2f}")
    print(f"  advantage > 0 in {(adv_greedy>0).mean()*100:.0f}% (greedy) / "
          f"{(adv_mpc>0).mean()*100:.0f}% (MPC) of sampled calibrations")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(n)
    ax[0].errorbar(x, tab.a_fit, yerr=[tab.a_fit - tab.a_lo, tab.a_hi - tab.a_fit],
                   fmt="o", capsize=3)
    ax[0].set_xlabel("region"); ax[0].set_ylabel(r"fitted biting rate $\bar{a}$")
    ax[0].set_title("(a) Calibration uncertainty (95% bootstrap)")
    ax[1].hist(adv_greedy, bins=15, alpha=0.7, label="greedy", color="#4477aa")
    ax[1].hist(adv_mpc, bins=15, alpha=0.7, label="MPC", color="#cc6677")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set_xlabel("advantage over static (pp)"); ax[1].set_ylabel("sampled calibrations")
    ax[1].set_title("(b) Advantage is robust to calibration"); ax[1].legend()
    plt.tight_layout()
    plt.savefig("fig_calib_uncertainty.png", dpi=150)
    print("\nsaved -> calib_uncertainty.csv, fig_calib_uncertainty.png")


if __name__ == "__main__":
    main()

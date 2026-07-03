"""
Tier 3.2 -- Probabilistic cost-effectiveness (Monte Carlo PSA).

Replaces the manuscript's fragile point estimate ($60.44 / QALY, built from
best-of-training 11.2% vs 5.6%) with a probabilistic sensitivity analysis that
(a) propagates uncertainty in every input assumption and (b) anchors the
allocation benefit to the HELD-OUT, multi-seed PPO-vs-static reductions (with
their per-environment uncertainty) instead of a single cherry-picked number.

Model (same structure as the manuscript, so the numbers stay comparable):
  YLL  = CFR * (life_exp - age_at_death)
  YLD  = duration_yrs * disability_weight
  DALY = YLL + YLD                        (~= QALY lost per case)
  combined_reduction = 1 - (1-g) * (1 - r_ppo)/(1 - r_static)
        g       = real-world static-ITN incidence reduction (GiveWell ~50%)
        r_ppo   = simulated held-out reduction under optimized (PPO) allocation
        r_static= simulated held-out reduction under population-proportional
  cases_averted_per_net = protected_per_net * years_per_net * annual_incidence
                          * combined_reduction
  cost_per_QALY = (ITN_cost + dist_cost) / (DALY * cases_averted_per_net)

The (r_static, r_ppo) pair is drawn by PAIRED BOOTSTRAP from the 50 held-out
evaluation environments in calibrated_results.npz, so the policy effect carries
its real statistical uncertainty. Static-allocation cost/QALY (factor = 1) is
reported alongside for context.

Run: python t3_cost_qaly.py
Outputs: console summary + fig_cost_qaly.png + t3_cost_qaly_results.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)
N = 200_000


def triangular(lo, mode, hi, n):
    return RNG.triangular(lo, mode, hi, n)


def normal_clip(mean, sd, lo, hi, n):
    return np.clip(RNG.normal(mean, sd, n), lo, hi)


def lognormal_from_ci(median, hi95, n):
    """Lognormal with given median and 97.5th percentile."""
    sigma = np.log(hi95 / median) / 1.959963985
    return np.exp(RNG.normal(np.log(median), sigma, n))


def load_policy_pairs():
    d = np.load("calibrated_results.npz", allow_pickle=True)
    le = d["last_eval"].item()
    b = np.asarray(le["baseline"], float)
    r_static = (b - np.asarray(le["static"], float)) / b
    r_ppo = (b - np.asarray(le["PPO"], float)) / b
    return r_static, r_ppo


def sample_inputs():
    rs_env, rp_env = load_policy_pairs()
    idx = RNG.integers(0, len(rs_env), N)          # paired bootstrap over held-out envs
    inp = {
        "disability_weight": np.clip(RNG.normal(0.191, 0.04, N), 0.05, 0.45),
        "case_fatality":     lognormal_from_ci(0.002, 0.0045, N),
        "life_exp":          normal_clip(65, 4, 55, 75, N),
        "age_at_death":      triangular(1, 4, 12, N),
        "duration_yrs":      triangular(0.027, 0.038, 0.055, N),     # ~10-20 days
        "annual_incidence":  normal_clip(250e6 / 1550e6, 0.025, 0.10, 0.25, N),
        "givewell_effect":   triangular(0.40, 0.50, 0.60, N),        # static ITN real-world effect
        "protected_per_net": triangular(1.5, 2.0, 2.0, N),
        "years_per_net":     triangular(2.0, 3.0, 3.0, N),           # LLIN effective life
        "itn_cost":          triangular(2.0, 3.0, 5.0, N),
        "dist_cost":         triangular(0.5, 1.0, 2.0, N),
        "r_static":          rs_env[idx],
        "r_ppo":             rp_env[idx],
    }
    return inp


def cost_per_qaly(inp, optimized=True):
    yll = inp["case_fatality"] * np.maximum(inp["life_exp"] - inp["age_at_death"], 0.0)
    yld = inp["duration_yrs"] * inp["disability_weight"]
    daly = yll + yld
    if optimized:
        factor = (1 - inp["r_ppo"]) / (1 - inp["r_static"])
    else:
        factor = 1.0                                   # static allocation
    combined = 1 - (1 - inp["givewell_effect"]) * factor
    cases_averted = (inp["protected_per_net"] * inp["years_per_net"]
                     * inp["annual_incidence"] * combined)
    cost = inp["itn_cost"] + inp["dist_cost"]
    qaly_per_dollar = daly * cases_averted / cost
    return cost / (daly * cases_averted), qaly_per_dollar, combined


def prcc(inp, y, keys):
    """Partial rank correlation coefficient of each input with output y."""
    def rank(a):
        return pd.Series(a).rank().to_numpy()
    Xr = np.column_stack([rank(inp[k]) for k in keys])
    Xr = (Xr - Xr.mean(0)) / Xr.std(0)
    yr = rank(y); yr = (yr - yr.mean()) / yr.std()
    out = {}
    for i, k in enumerate(keys):
        others = np.delete(Xr, i, axis=1)
        A = np.column_stack([np.ones(len(yr)), others])
        # residuals of X_i and y after regressing out the other inputs
        bx, *_ = np.linalg.lstsq(A, Xr[:, i], rcond=None)
        by, *_ = np.linalg.lstsq(A, yr, rcond=None)
        rx = Xr[:, i] - A @ bx
        ry = yr - A @ by
        out[k] = float(np.corrcoef(rx, ry)[0, 1])
    return out


def summarize(name, cpq):
    finite = cpq[np.isfinite(cpq) & (cpq > 0)]
    q = np.percentile(finite, [2.5, 25, 50, 75, 97.5])
    print(f"\n{name}")
    print(f"  median ${q[2]:.2f} / QALY   95% CI [${q[0]:.2f}, ${q[4]:.2f}]"
          f"   IQR [${q[1]:.2f}, ${q[3]:.2f}]   mean ${finite.mean():.2f}")
    for thr in (500, 1000, 1500, 3000):
        print(f"  P(cost/QALY < ${thr}) = {(finite < thr).mean()*100:.1f}%")
    return q, finite


def main():
    inp = sample_inputs()
    cpq_opt, qpd_opt, comb_opt = cost_per_qaly(inp, optimized=True)
    cpq_sta, _, comb_sta = cost_per_qaly(inp, optimized=False)

    print("=" * 66)
    print("Tier 3.2 -- Probabilistic cost-effectiveness (Monte Carlo, N=%d)" % N)
    print("=" * 66)
    print(f"\nHeld-out reductions used (paired bootstrap over 50 envs):")
    print(f"  static  r = {inp['r_static'].mean():.3f} +/- {inp['r_static'].std():.3f}")
    print(f"  PPO     r = {inp['r_ppo'].mean():.3f} +/- {inp['r_ppo'].std():.3f}")
    print(f"  combined program reduction: optimized {comb_opt.mean():.3f} | "
          f"static {comb_sta.mean():.3f}")

    q_opt, fin_opt = summarize("OPTIMIZED (PPO) allocation -- cost per QALY:", cpq_opt)
    q_sta, fin_sta = summarize("STATIC (population-proportional) allocation -- cost per QALY:", cpq_sta)

    # incremental: ratio of static to optimized cost-per-QALY (program-level improvement)
    impr = (1 - cpq_opt / cpq_sta)
    impr = impr[np.isfinite(impr)]
    print(f"\nOptimized allocation lowers cost/QALY by a median "
          f"{np.median(impr)*100:.1f}% [{np.percentile(impr,2.5)*100:.1f}, "
          f"{np.percentile(impr,97.5)*100:.1f}] vs static (same nets).")

    keys = ["disability_weight", "case_fatality", "life_exp", "age_at_death",
            "duration_yrs", "annual_incidence", "givewell_effect",
            "protected_per_net", "years_per_net", "itn_cost", "dist_cost",
            "r_static", "r_ppo"]
    pr = prcc(inp, cpq_opt, keys)
    pr_sorted = sorted(pr.items(), key=lambda kv: -abs(kv[1]))
    print("\nPRCC (drivers of cost/QALY, optimized):")
    for k, v in pr_sorted:
        print(f"  {k:18s} {v:+.3f}")

    # ----- figure -----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    clip_hi = np.percentile(fin_opt, 99)
    ax[0].hist(fin_opt[fin_opt < clip_hi], bins=80, color="#4477aa", alpha=0.85)
    ax[0].axvline(q_opt[2], color="k", lw=2, label=f"median ${q_opt[2]:.0f}")
    ax[0].axvline(q_opt[0], color="k", ls="--", lw=1)
    ax[0].axvline(q_opt[4], color="k", ls="--", lw=1, label=f"95% CI [{q_opt[0]:.0f}, {q_opt[4]:.0f}]")
    ax[0].set_xlabel("Cost per QALY gained (USD)")
    ax[0].set_ylabel("Monte Carlo draws")
    ax[0].set_title("(a) Cost-effectiveness distribution\n(optimized ITN allocation)")
    ax[0].legend(fontsize=9)

    labels = [k for k, _ in pr_sorted]
    vals = [v for _, v in pr_sorted]
    colors = ["#cc6677" if v > 0 else "#4477aa" for v in vals]
    ax[1].barh(range(len(vals))[::-1], vals, color=colors)
    ax[1].set_yticks(range(len(vals))[::-1])
    ax[1].set_yticklabels(labels, fontsize=8)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_xlabel("Partial rank correlation with cost/QALY")
    ax[1].set_title("(b) Sensitivity drivers (PRCC)")
    plt.tight_layout()
    plt.savefig("fig_cost_qaly.png", dpi=150)
    print("\nsaved -> fig_cost_qaly.png")

    pd.DataFrame({
        "quantile": ["2.5%", "25%", "50%", "75%", "97.5%"],
        "optimized_cost_per_qaly": q_opt,
        "static_cost_per_qaly": q_sta,
    }).to_csv("t3_cost_qaly_results.csv", index=False)
    pd.DataFrame(pr_sorted, columns=["input", "prcc"]).to_csv(
        "t3_cost_qaly_prcc.csv", index=False)
    print("saved -> t3_cost_qaly_results.csv, t3_cost_qaly_prcc.csv")


if __name__ == "__main__":
    main()

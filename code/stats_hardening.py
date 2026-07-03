"""Disciplined statistical reporting across every results ladder.

Addresses the reviewer point that the paper reports tiny p-values and "fraction of
environments won" but is light on effect-size confidence intervals and multiple-
comparison correction. For each ladder we:
  * convert each method's per-environment cumulative infection ratio to a % reduction
    vs the paired no-ITN baseline (lower ratio = better);
  * report the mean reduction;
  * for each non-reference method, report the paired effect size vs the reference
    (population-proportional "static") with a 95% bootstrap CI (10k resamples) and a
    paired Wilcoxon signed-rank p-value;
  * Holm-Bonferroni-correct the Wilcoxon p-values within each ladder's family of
    comparisons, and flag significance at corrected alpha=0.05.

Pure post-hoc analysis of saved .npz arrays; no simulation is re-run here.
"""
import numpy as np
from scipy import stats

RNG = np.random.default_rng(20260626)
NBOOT = 10000


def boot_ci(d, nboot=NBOOT, alpha=0.05):
    """Bootstrap CI for the mean of paired differences d."""
    idx = RNG.integers(0, len(d), size=(nboot, len(d)))
    means = d[idx].mean(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def analyze(npz, key, methods, ref="static", title=""):
    import os
    if not os.path.exists(npz):
        print(f"\n### {title} ###\n  [skipped: {npz} not ready yet]")
        return None
    d = np.load(npz, allow_pickle=True)[key].item()
    base = np.asarray(d["baseline"], float)
    red = {m: (base - np.asarray(d[m], float)) / base * 100 for m in methods}
    ref_red = red[ref]
    comp = [m for m in methods if m != ref and m != "baseline"]
    deltas, ps = {}, {}
    for m in comp:
        dd = red[m] - ref_red                      # paired effect size vs static
        deltas[m] = dd
        try:
            ps[m] = stats.wilcoxon(red[m], ref_red, zero_method="wilcox").pvalue
        except ValueError:
            ps[m] = 1.0
    padj = holm(np.array([ps[m] for m in comp])) if comp else np.array([])

    n = len(base)
    print(f"\n### {title}  (n={n} held-out environments) ###")
    print(f"{'method':30s} {'mean red%':>9s} {'vs static (pp)':>15s} "
          f"{'95% CI':>16s} {'p(Holm)':>9s} {'win%':>6s}")
    print(f"{ref:30s} {ref_red.mean():9.2f} {'-- (ref)':>15s} {'':>16s} {'':>9s} {'':>6s}")
    for m, pa in zip(comp, padj):
        dd = deltas[m]
        lo, hi = boot_ci(dd)
        win = np.mean(dd > 0) * 100
        sig = "*" if pa < 0.05 else " "
        print(f"{m:30s} {red[m].mean():9.2f} {dd.mean():+15.2f} "
              f"[{lo:+.2f},{hi:+.2f}]".rjust(0).ljust(0) +
              f"  {pa:8.1e}{sig} {win:5.0f}")
    return red


print("=" * 78)
print("STATISTICAL HARDENING: effect sizes, 95% bootstrap CIs, Holm-corrected tests")
print("=" * 78)

analyze("calibrated_results.npz", "last_eval",
        ["baseline", "static", "DQN", "bandit", "QT-Opt", "PPO"],
        title="Real-calibrated SEITR (learned ladder)")

analyze("extended_results.npz", "last_eval",
        ["baseline", "static", "DQN", "bandit", "QT-Opt", "PPO"],
        title="Real-calibrated SEITAR (learned ladder)")

analyze("ws234_planners_seitr.npz", "results",
        ["baseline", "static", "greedy", "mpc", "oracle"],
        title="SEITR planner ladder (greedy / MPC / oracle)")

analyze("ws56_scaled_n50.npz", "results",
        ["baseline", "static", "incidence", "prevalence", "greedy", "mpc", "oracle", "PPO"],
        title="N=50 real-region benchmark")

analyze("meta_adaptive_n40.npz", "red",
        ["baseline", "static", "open", "mpc", "adapt", "oracle"],
        title="ARMOR-Adapt: surveillance-driven adaptation (n=40)")

analyze("meta_bayesian_n60.npz", "red",
        ["baseline", "static", "greedy-obs", "ce", "thompson", "ucb", "ida", "oracle"],
        title="ARMOR-IDA: dual control under noisy surveillance (n=60)")

print("\n(* = significant at Holm-corrected alpha=0.05; win% = fraction of paired "
      "environments where the method beats static)")

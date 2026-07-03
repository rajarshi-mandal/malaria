"""Analyze Item B results: table, paired significance tests, and figures.

Reads baselines_b_results.npz (written by baselines_b.py) and produces:
  * a console table (held-out infection ratio + % reduction, mean +/- std)
  * paired Wilcoxon signed-rank tests vs QT-Opt on the held-out eval set
  * fig_baselines_bar.png   -- % reduction by method (error bars across eval envs)
  * fig_learning_curves.png -- QT-Opt vs contextual-bandit training curves
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

METHODS = ["baseline", "static", "bandit", "DQN", "PPO", "QT-Opt"]
PRETTY = {"baseline": "Baseline\n(no ITNs)", "static": "Static\n(pop-prop)",
          "bandit": "Contextual\nbandit", "DQN": "DQN", "PPO": "PPO",
          "QT-Opt": "QT-Opt\n(ours)"}


def load(path="baselines_b_results.npz"):
    d = np.load(path, allow_pickle=True)
    return (d["ratios"].item(), d["reductions"].item(),
            d["last_eval"].item(), d["curves"].item(), d["seeds"])


def main():
    ratios, reductions, last_eval, curves, seeds = load()
    n_seeds = len(seeds)

    print("================ ITEM B: HELD-OUT COMPARISON ================")
    print(f"(mean over {n_seeds} training seed(s); held-out eval envs)\n")
    print(f"{'Method':12s} | {'Inf. ratio':>14s} | {'% reduction':>16s}")
    print("-" * 50)
    for m in METHODS:
        r, rd = np.array(ratios[m]), np.array(reductions[m])
        print(f"{m:12s} | {r.mean():7.3f} +/- {r.std():.3f} | "
              f"{rd.mean():7.2f} +/- {rd.std():.2f}%")

    # ---- paired significance vs QT-Opt on the held-out eval set (last seed) ----
    print("\n--- Paired Wilcoxon signed-rank vs QT-Opt (held-out envs, last seed) ---")
    qt = last_eval["QT-Opt"]
    for m in METHODS:
        if m == "QT-Opt":
            continue
        other = last_eval[m]
        # lower infection ratio is better; positive delta => QT-Opt better
        delta = other - qt
        try:
            stat, p = stats.wilcoxon(qt, other)
        except ValueError:
            p = float("nan")
        better = (delta > 0).mean() * 100
        print(f"QT-Opt vs {m:10s}: median d(ratio)={np.median(delta):+.3f} "
              f"| QT-Opt better in {better:.0f}% of envs | p={p:.2e}")

    # ---- Figure 1: % reduction bar chart ----
    fig, ax = plt.subplots(figsize=(7, 4))
    means = [np.array(reductions[m]).mean() for m in METHODS]
    # error bar: std across held-out eval envs (within last seed), in reduction units
    base_mean = last_eval["baseline"].mean()
    errs = [ (last_eval[m] / base_mean * 100).std() if m != "baseline" else 0
             for m in METHODS]
    colors = ["#999999", "#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    bars = ax.bar([PRETTY[m] for m in METHODS], means, yerr=errs, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.6)
    for b, mny in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, mny + 0.15, f"{mny:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Reduction in cumulative infection ratio\nvs. no-ITN baseline (%)")
    ax.set_title("Held-out ITN-allocation performance by method")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_baselines_bar.png", dpi=150)

    # ---- Figure 2: QT-Opt vs bandit learning curves ----
    fig2, ax2 = plt.subplots(figsize=(6.5, 4))
    for name, color in [("QT-Opt", "#CCB974"), ("bandit", "#55A868")]:
        c = np.array(curves[name])
        w = 25
        if len(c) >= w:
            sm = np.convolve(c, np.ones(w) / w, mode="valid")
            ax2.plot(np.arange(len(sm)) + w, sm,
                     label=("QT-Opt (sequential)" if name == "QT-Opt"
                            else "Contextual bandit (one-step)"),
                     color=color, linewidth=1.8)
    ax2.set_xlabel("Training episode")
    ax2.set_ylabel("Cumulative infection ratio\n(training, lower is better)")
    ax2.set_title("Sequential bootstrapping vs. one-step regression")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("fig_learning_curves.png", dpi=150)

    print("\nsaved -> fig_baselines_bar.png, fig_learning_curves.png")


if __name__ == "__main__":
    main()

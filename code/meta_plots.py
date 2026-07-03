"""Figures for the novelty results: (1) the efficiency-equity Pareto frontier,
(2) the robustness (mean vs worst-case) comparison. Run after meta_equity.py and
meta_robust.py. Produces fig_equity_frontier.png and fig_robust.png."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def equity_fig():
    d = np.load("meta_equity.npz", allow_pickle=True)
    weights = list(d["weights"])
    agg = d["agg"].item()
    static = d["static"].item()
    tot = [np.mean(agg[w]["total"]) for w in weights]
    wor = [np.mean(agg[w]["worst"]) for w in weights]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot(wor, tot, "o-", color="#4477aa")
    for w, x, y in zip(weights, wor, tot):
        ax[0].annotate(f"w={w}", (x, y), fontsize=7, xytext=(4, 4),
                       textcoords="offset points")
    ax[0].scatter([np.mean(static["worst"])], [np.mean(static["total"])],
                  color="#cc6677", zorder=5, label="static (WHO)")
    ax[0].set_xlabel("worst-region per-capita reduction (%)")
    ax[0].set_ylabel("total reduction (%)")
    ax[0].set_title("(a) Efficiency-equity frontier"); ax[0].legend(fontsize=8)
    disp = [np.mean(agg[w]["disp"]) for w in weights]
    ax[1].plot(weights, disp, "s-", color="#228833")
    ax[1].set_xlabel("equity weight w"); ax[1].set_ylabel("between-region disparity (pp)")
    ax[1].set_title("(b) Disparity shrinks with equity weight")
    plt.tight_layout(); plt.savefig("fig_equity_frontier.png", dpi=150)
    print("saved -> fig_equity_frontier.png")


def robust_fig():
    d = np.load("meta_robust.npz", allow_pickle=True)
    rows = d["rows"].item(); tails = d["tails"].item()
    methods = ["static", "mean", "robust"]
    labels = ["static\n(WHO)", "mean-optimal", "robust\n(CVaR)"]
    x = np.arange(len(methods)); wdt = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.bar(x - wdt / 2, [np.mean(rows[m]) for m in methods], wdt,
           label="mean reduction", color="#88ccee")
    ax.bar(x + wdt / 2, [np.mean(tails[m]) for m in methods], wdt,
           label="worst-case (CVaR) reduction", color="#cc6677")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("infection reduction (%)")
    ax.set_title("Distributionally-robust allocation under uncertainty")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("fig_robust.png", dpi=150)
    print("saved -> fig_robust.png")


def adaptive_fig():
    d = np.load("meta_adaptive.npz", allow_pickle=True)
    red = d["red"].item()
    b = np.asarray(red["baseline"], float)               # per-world paired baseline
    order = ["static", "open", "mpc", "adapt", "oracle"]
    labels = ["static\n(WHO)", "open-loop", "MPC\n(state fb)", "ARMOR-Adapt\n(learns)", "oracle\n(knows truth)"]
    order = [m for m in order if m in red]
    means = [((b - np.asarray(red[m], float)) / b * 100).mean() for m in order]
    cvars = []
    for m in order:
        r = np.sort((b - np.asarray(red[m], float)) / b * 100)
        n = max(1, int(np.ceil(0.2 * len(r))))
        cvars.append(r[:n].mean())
    x = np.arange(len(order)); wdt = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = ["#bbbbbb", "#88ccee", "#44aa99", "#cc6677", "#332288"]
    ax.bar(x - wdt / 2, means, wdt, label="mean reduction",
           color=[colors[i] for i in range(len(order))])
    ax.bar(x + wdt / 2, cvars, wdt, label="worst-case (CVaR) reduction",
           color=[colors[i] for i in range(len(order))], alpha=0.55)
    ax.set_xticks(x); ax.set_xticklabels([labels[["static","open","mpc","adapt","oracle"].index(m)] for m in order], fontsize=8)
    ax.set_ylabel("infection reduction (%)")
    ax.set_title("Value of surveillance-driven adaptation under unknown transmission")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("fig_adaptive.png", dpi=150)
    print("saved -> fig_adaptive.png")


def bayesian_fig():
    d = np.load("meta_bayesian.npz", allow_pickle=True)
    red = d["red"].item(); traces = d["traces"].item()
    b = np.asarray(red["baseline"], float)               # per-world paired baseline
    order = [a for a in ["static", "greedy-obs", "ce", "thompson", "ucb", "ida", "oracle"]
             if a in red]
    labels = {"static": "static", "greedy-obs": "greedy\n(obs)", "ce": "CE-MPC",
              "thompson": "Thompson", "ucb": "UCB", "ida": "ARMOR-IDA\n(novel)",
              "oracle": "oracle"}
    means = [((b - np.asarray(red[a], float)) / b * 100).mean() for a in order]
    cvars = []
    for a in order:
        r = np.sort((b - np.asarray(red[a], float)) / b * 100)
        n = max(1, int(np.ceil(0.2 * len(r)))); cvars.append(r[:n].mean())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(order)); wdt = 0.4
    cols = ["#bbbbbb", "#999999", "#88ccee", "#44aa99", "#ddcc77", "#cc6677", "#332288"]
    ax[0].bar(x - wdt / 2, means, wdt, label="mean", color=[cols[i] for i in range(len(order))])
    ax[0].bar(x + wdt / 2, cvars, wdt, label="worst-case (CVaR)",
              color=[cols[i] for i in range(len(order))], alpha=0.55)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xticks(x); ax[0].set_xticklabels([labels[a] for a in order], fontsize=7)
    ax[0].set_ylabel("infection reduction (%)")
    ax[0].set_title("(a) Adaptive-control ladder under noisy surveillance")
    ax[0].legend(fontsize=8)
    for a, col in (("ce", "#88ccee"), ("ida", "#cc6677")):
        if a in traces and len(traces[a]):
            m = np.array(traces[a]).mean(0)
            ax[1].plot(np.arange(1, len(m) + 1), m, "o-", color=col,
                       label="CE-MPC" if a == "ce" else "ARMOR-IDA")
    ax[1].set_xlabel("campaign")
    ax[1].set_ylabel("decision-relevant posterior error\n(burden-weighted |log-transmission err|)")
    ax[1].set_title("(b) Incomplete learning: CE plateaus, IDA corrects")
    ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("fig_bayesian.png", dpi=150)
    print("saved -> fig_bayesian.png")


if __name__ == "__main__":
    import os
    if os.path.exists("meta_equity.npz"):
        equity_fig()
    if os.path.exists("meta_robust.npz"):
        robust_fig()
    if os.path.exists("meta_adaptive.npz"):
        adaptive_fig()
    if os.path.exists("meta_bayesian.npz"):
        bayesian_fig()

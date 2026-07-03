"""Figures for the extended SEITAR model: calibration (prevalence now matched)
and the allocation comparison (if results present)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- calibration: SEITR vs SEITAR prevalence match ----
ext = pd.read_csv("extended_calibrated_regions.csv").sort_values("PfPR_obs").reset_index(drop=True)
old = pd.read_csv("calibrated_regions.csv").set_index("region")
labels = [f"{r.region}\n({r.ISO3})" for _, r in ext.iterrows()]
x = np.arange(len(ext)); w = 0.27

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
ax1.bar(x - w, ext.PfPR_obs*100, w, label="Observed PfPR (MAP)", color="#C44E52", edgecolor="black", lw=0.5)
ax1.bar(x, [old.loc[r, "PfPR_sim"]*100 for r in ext.region], w, label="SEITR (old)", color="#999999", edgecolor="black", lw=0.5)
ax1.bar(x + w, ext.PfPR_sim*100, w, label="SEITAR (extended)", color="#55A868", edgecolor="black", lw=0.5)
ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax1.set_ylabel("Parasite prevalence (%)")
ax1.set_title("Asymptomatic compartment closes the prevalence gap")
ax1.legend(); ax1.grid(axis="y", alpha=0.3)

ax2.bar(x - w/2, ext.incidence_obs, w*1.5, label="Observed (MAP)", color="#4C72B0", edgecolor="black", lw=0.5)
ax2.bar(x + w/2, ext.incidence_sim, w*1.5, label="SEITAR sim", color="#CCB974", edgecolor="black", lw=0.5)
ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax2.set_ylabel("Clinical incidence (cases / 1,000 / yr)")
ax2.set_title("Incidence still matched (joint calibration)")
ax2.legend(); ax2.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig("fig_extended_calibration.png", dpi=150)
print("saved -> fig_extended_calibration.png")

# ---- allocation comparison (if available) ----
if os.path.exists("extended_results.npz"):
    d = np.load("extended_results.npz", allow_pickle=True)
    ratio = d["ratio"].item(); last = d["last_eval"].item()
    methods = ["baseline", "static", "bandit", "DQN", "PPO", "QT-Opt"]
    pretty = {"baseline": "Baseline", "static": "Static\n(pop-prop)", "bandit": "Bandit\n(one-step)",
              "DQN": "DQN", "PPO": "PPO", "QT-Opt": "QT-Opt\n(ours)"}
    colors = ["#999999", "#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    base = np.mean(ratio["baseline"])
    means = [(base-np.mean(ratio[m]))/base*100 for m in methods]
    errs = [((base-last[m])/base*100).std() if m != "baseline" else 0 for m in methods]
    fig2, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar([pretty[m] for m in methods], means, yerr=errs, capsize=4,
                  color=colors, edgecolor="black", lw=0.6)
    for b, m in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, m+0.3, f"{m:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Reduction vs no-ITN baseline (%)")
    ax.set_title("Allocation performance on calibrated SEITAR environment")
    ax.grid(axis="y", alpha=0.3)
    fig2.tight_layout(); fig2.savefig("fig_extended_bar.png", dpi=150)
    print("saved -> fig_extended_bar.png")

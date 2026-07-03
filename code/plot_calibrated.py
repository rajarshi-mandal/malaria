"""Figure for the calibrated (Item D) method comparison."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("calibrated_results.npz", allow_pickle=True)
ratio = d["ratio"].item()
last = d["last_eval"].item()
methods = ["baseline", "static", "bandit", "DQN", "PPO", "QT-Opt"]
pretty = {"baseline": "Baseline\n(no ITNs)", "static": "Static\n(pop-prop)",
          "bandit": "Bandit\n(one-step)", "DQN": "DQN", "PPO": "PPO",
          "QT-Opt": "QT-Opt\n(ours)"}
colors = ["#999999", "#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]

base = np.mean(ratio["baseline"])
means = [(base - np.mean(ratio[m])) / base * 100 for m in methods]
errs = [((base - last[m]) / base * 100).std() if m != "baseline" else 0 for m in methods]

fig, ax = plt.subplots(figsize=(7.5, 4.2))
bars = ax.bar([pretty[m] for m in methods], means, yerr=errs, capsize=4,
              color=colors, edgecolor="black", linewidth=0.6)
for b, m in zip(bars, means):
    ax.text(b.get_x() + b.get_width() / 2, m + 0.4, f"{m:.1f}%",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Reduction in cumulative infection ratio\nvs. no-ITN baseline (%)")
ax.set_title("Held-out allocation performance on 8 real calibrated regions")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("fig_calibrated_bar.png", dpi=150)
print("saved -> fig_calibrated_bar.png")

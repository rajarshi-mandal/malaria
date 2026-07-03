"""Figure for the shared-budget (sequential) experiment."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("shared_budget_results.npz", allow_pickle=True)
last = d["last_eval"].item()
curves = d["curves"].item()
methods = ["even-thirds", "spend-early", "bandit", "QT-Opt"]
pretty = {"even-thirds": "Even thirds\n(heuristic)", "spend-early": "Spend early\n(heuristic)",
          "bandit": "Bandit\n(myopic, g=0)", "QT-Opt": "QT-Opt\n(sequential)"}
colors = ["#BBBBBB", "#888888", "#55A868", "#CCB974"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

means = [last[m].mean() for m in methods]
errs = [last[m].std() for m in methods]
bars = ax1.bar([pretty[m] for m in methods], means, yerr=errs, capsize=4,
               color=colors, edgecolor="black", linewidth=0.6)
for b, m in zip(bars, means):
    ax1.text(b.get_x() + b.get_width() / 2, m + 0.004, f"{m:.3f}",
             ha="center", va="bottom", fontsize=9)
ax1.set_ylabel("Cumulative infection ratio (lower is better)")
ax1.set_title("Shared seasonal budget + ITN decay")
ax1.set_ylim(min(means) - 0.05, max(means) + 0.05)
ax1.grid(axis="y", alpha=0.3)

for name, color in [("QT-Opt", "#CCB974"), ("bandit", "#55A868")]:
    c = np.array(curves[name]); w = 25
    if len(c) >= w:
        sm = np.convolve(c, np.ones(w) / w, mode="valid")
        ax2.plot(np.arange(len(sm)) + w, sm, color=color, linewidth=1.8,
                 label=("QT-Opt (sequential)" if name == "QT-Opt" else "Bandit (myopic)"))
ax2.set_xlabel("Training episode")
ax2.set_ylabel("Cumulative infection ratio (training)")
ax2.set_title("Sequential vs. myopic (shared budget)")
ax2.legend(); ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("fig_shared_budget.png", dpi=150)
print("saved -> fig_shared_budget.png")

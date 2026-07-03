"""Calibration-quality figures: simulated vs observed incidence, and the
prevalence gap that motivates the extended ODE."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("calibrated_regions.csv").sort_values("incidence_obs").reset_index(drop=True)
labels = [f"{r.region}\n({r.ISO3})" for _, r in df.iterrows()]
x = np.arange(len(df))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# Panel 1: observed vs simulated incidence (calibration target)
w = 0.4
ax1.bar(x - w/2, df.incidence_obs, w, label="Observed (MAP)", color="#4C72B0", edgecolor="black", lw=0.5)
ax1.bar(x + w/2, df.incidence_sim, w, label="Simulated (calibrated)", color="#CCB974", edgecolor="black", lw=0.5)
ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax1.set_ylabel("Clinical incidence (cases / 1,000 / yr)")
ax1.set_title("Calibration target: incidence matched per region")
ax1.legend(); ax1.grid(axis="y", alpha=0.3)

# Panel 2: prevalence gap (observed PfPR vs model) -> motivates extended ODE
ax2.bar(x - w/2, df.PfPR_obs * 100, w, label="Observed PfPR (MAP)", color="#C44E52", edgecolor="black", lw=0.5)
ax2.bar(x + w/2, df.PfPR_sim * 100, w, label="Model prevalence (SEITR)", color="#999999", edgecolor="black", lw=0.5)
ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax2.set_ylabel("Parasite prevalence (%)")
ax2.set_title("Prevalence gap: SEITR caps ~10% (motivates asymptomatic compartment)")
ax2.legend(); ax2.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("fig_calibration.png", dpi=150)
print("saved -> fig_calibration.png")

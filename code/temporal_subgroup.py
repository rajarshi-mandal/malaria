"""Subgroup analysis of the out-of-sample temporal validation.

The headline temporal result (temporal_validation.py) is that the model predicts
held-out prevalence LEVELS well (Pearson r~0.77) but year-to-year CHANGE direction
is near chance. A reviewer reasonably asks whether that means the model misses
intervention dynamics. This script tests an honest, mechanistic prediction: the
change signal should be recoverable precisely where the ITN-coverage signal is
large (high signal-to-noise), and absent where coverage barely moved (the bulk of
the panel, where observed change is dominated by ACT/IRS/SMC scale-up and noise the
ITN-only model omits).

We compute, per region, the baseline(2015)->final change in observed and predicted
prevalence and in ITN coverage, then stratify by |Delta coverage|. Reports
change-direction agreement and Pearson r of (Delta obs, Delta pred) overall and in
the large-coverage-change subgroup. No model refit; purely a stratified read of the
existing held-out predictions.
"""
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("temporal_validation.csv")

# Per region: baseline (earliest, =2015) and final (latest) year.
rows = []
for (iso, reg), g in df.groupby(["ISO3", "region"]):
    g = g.sort_values("year")
    base, fin = g.iloc[0], g.iloc[-1]
    if fin["year"] <= base["year"]:
        continue
    rows.append(dict(iso=iso, region=reg,
                     d_obs=fin["obs"] - base["obs"],
                     d_pred=fin["pred"] - base["pred"],
                     d_itn=fin["itn"] - base["itn"],
                     years=fin["year"] - base["year"]))
R = pd.DataFrame(rows)
print(f"regions with >=2 years: {len(R)}")


def report(sub, label):
    n = len(sub)
    if n < 5:
        print(f"  {label:34s} n={n:4d}  (too few)")
        return
    # direction agreement: do obs and pred move the same way?
    hits = int(np.sum(np.sign(sub["d_obs"]) == np.sign(sub["d_pred"])))
    agree = hits / n
    # binomial test: is direction agreement above chance (0.5)?
    pb = stats.binomtest(hits, n, 0.5, alternative="greater").pvalue
    r, p = stats.pearsonr(sub["d_obs"], sub["d_pred"])
    rho, _ = stats.spearmanr(sub["d_obs"], sub["d_pred"])
    print(f"  {label:34s} n={n:4d}  dir-agree={agree*100:5.1f}% "
          f"(binom p={pb:.1e})  r={r:+.3f}  rho={rho:+.3f}")


print("\n=== change-direction skill, stratified by |Delta ITN coverage| ===")
report(R, "all regions")
absitn = R["d_itn"].abs()
for q, lab in [(0.50, "top 50% coverage movers"),
               (0.67, "top 33% coverage movers"),
               (0.75, "top 25% coverage movers")]:
    thr = absitn.quantile(q)
    report(R[absitn >= thr], f"{lab} (|dITN|>={thr:.3f})")
# absolute thresholds (interpretable)
for t in (0.05, 0.10, 0.15):
    report(R[absitn >= t], f"|dITN| >= {t*100:.0f} pp")

# Complementary: the flat-coverage bulk where we EXPECT no recoverable signal.
report(R[absitn < 0.05], "|dITN| < 5 pp (flat, the bulk)")

print("\n=== levels skill (context, held-out region-years) ===")
ho = df[df["held_out"]]
r, p = stats.pearsonr(ho["obs"], ho["pred"])
print(f"  held-out PfPR level: n={len(ho)}  Pearson r={r:.3f}  "
      f"median|err|={np.median(np.abs(ho['obs']-ho['pred']))*100:.1f} pp")

R.to_csv("temporal_subgroup.csv", index=False)
print("\nsaved -> temporal_subgroup.csv")

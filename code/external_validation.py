"""
BEYOND-SIMULATOR VALIDATION 2: external effect-size check.

A low-risk credibility floor: does the simulator's implied ITN protective effect
match the published trial/meta-analytic evidence? We compare the model's
community-level reduction in clinical incidence and parasite prevalence (no-ITN
vs realistic coverage) against well-established literature anchors:

  * Clinical malaria incidence: ITNs reduce episodes by ~50% (Lengeler 2004
    Cochrane; the GiveWell program estimate the paper already uses ~50%).
  * Parasite prevalence: ITNs reduce prevalence by ~13 percentage points /
    ~24% relative at high coverage (Lengeler 2004; Bhatt 2015 Nature estimated
    ITNs averted ~68% of cases 2000-2015 alongside scale-up).

If the model -- calibrated only to cross-sectional prevalence+incidence, with NO
tuning to trial effect sizes -- independently reproduces these reductions, that
is external support that its ITN mechanism is realistic.

Run: python external_validation.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from extended_ode import ExtendedCounty

# literature anchors (community-level, high coverage)
LIT_INC_REDUCTION = (40.0, 60.0)      # % reduction in clinical incidence (~50%)
LIT_PREV_REL_REDUCTION = (18.0, 35.0)  # % relative reduction in prevalence (~24%)


def regions_table():
    for fn in ("real_regions_n50.csv", "extended_calibrated_regions.csv"):
        if os.path.exists(fn):
            return pd.read_csv(fn), fn
    raise FileNotFoundError("no calibrated region table found")


def sustained_outcome(a, p, mult, cov, R=0.4, replace=365, warm_years=12):
    """Periodic steady state under SUSTAINED coverage with realistic annual net
    replacement (tau reset each year), NOT decayed-to-zero nets. Returns
    (mean prevalence, annual clinical incidence /1000)."""
    c = ExtendedCounty(biting=a, coverage=max(cov, 0.0), multiplier=mult,
                       treat=10, p_sympt=p, resistance=R)
    for _ in range(warm_years):
        c.C = cov if cov > 0 else 0.0
        c.tau = 0.0
        for _ in range(replace):
            c.step(0.0, seasonal=False)
    c.C = cov if cov > 0 else 0.0
    c.tau = 0.0
    prev, inc = [], 0.0
    for _ in range(365):
        inc += c.step(0.0, seasonal=False)
        prev.append(c.prevalence())
    return float(np.mean(prev)), inc / c.N_H * 1000.0


def reductions(a, p, mult, cov_target):
    prev0, inc0 = sustained_outcome(a, p, mult, 0.0)            # no ITN
    prevc, incc = sustained_outcome(a, p, mult, cov_target)      # sustained coverage
    inc_red = (inc0 - incc) / max(inc0, 1e-9) * 100
    prev_red_rel = (prev0 - prevc) / max(prev0, 1e-9) * 100
    prev_red_abs = (prev0 - prevc) * 100
    return inc_red, prev_red_rel, prev_red_abs


def main():
    df, fn = regions_table()
    print(f"external effect-size validation on {len(df)} regions ({fn})", flush=True)
    rows = []
    for _, r in df.iterrows():
        mult = r.get("pop_multiplier", 1.0)
        for cov in (0.5, 0.8):
            ir, prr, pra = reductions(r.biting_rate, r.p_sympt, mult, cov)
            rows.append(dict(region=r.region, coverage=cov, inc_red=ir,
                             prev_red_rel=prr, prev_red_abs=pra))
    res = pd.DataFrame(rows)
    res.to_csv("external_validation.csv", index=False)

    print("\n===== MODEL ITN EFFECT SIZE vs LITERATURE =====")
    for cov in (0.5, 0.8):
        s = res[res.coverage == cov]
        print(f"\n-- community ITN coverage {cov:.0%} --")
        print(f"  clinical incidence reduction : median {s.inc_red.median():.1f}% "
              f"(IQR {s.inc_red.quantile(.25):.1f}-{s.inc_red.quantile(.75):.1f}%)  "
              f"[lit ~50%]")
        print(f"  prevalence reduction (relative): median {s.prev_red_rel.median():.1f}% "
              f"(IQR {s.prev_red_rel.quantile(.25):.1f}-{s.prev_red_rel.quantile(.75):.1f}%)  "
              f"[lit ~24%]")
        print(f"  prevalence reduction (absolute): median {s.prev_red_abs.median():.1f} pp")

    s80 = res[res.coverage == 0.8]
    in_band_inc = ((s80.inc_red >= LIT_INC_REDUCTION[0]) &
                   (s80.inc_red <= LIT_INC_REDUCTION[1])).mean() * 100
    print(f"\nfraction of regions with incidence reduction in literature band "
          f"{LIT_INC_REDUCTION} at 80% coverage: {in_band_inc:.0f}%")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for cov, col in ((0.5, "#88ccee"), (0.8, "#4477aa")):
        s = res[res.coverage == cov]
        ax[0].hist(s.inc_red, bins=15, alpha=0.6, color=col, label=f"{cov:.0%} coverage")
    ax[0].axvspan(*LIT_INC_REDUCTION, color="gray", alpha=0.25, label="literature ~50%")
    ax[0].set_xlabel("simulated clinical incidence reduction (%)")
    ax[0].set_ylabel("regions"); ax[0].set_title("(a) Incidence effect vs trials")
    ax[0].legend(fontsize=8)
    for cov, col in ((0.5, "#ee9988"), (0.8, "#cc6677")):
        s = res[res.coverage == cov]
        ax[1].hist(s.prev_red_rel, bins=15, alpha=0.6, color=col, label=f"{cov:.0%} coverage")
    ax[1].axvspan(*LIT_PREV_REL_REDUCTION, color="gray", alpha=0.25, label="literature ~24%")
    ax[1].set_xlabel("simulated prevalence reduction (relative %)")
    ax[1].set_ylabel("regions"); ax[1].set_title("(b) Prevalence effect vs trials")
    ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("fig_external_validation.png", dpi=150)
    print("saved -> external_validation.csv, fig_external_validation.png")


if __name__ == "__main__":
    main()

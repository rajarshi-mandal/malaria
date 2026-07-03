"""
BEYOND-SIMULATOR VALIDATION 1: out-of-sample temporal prediction.

Reviewer concern: "the central results are still simulator-only; is the calibrated
model predictive of actual ITN-campaign outcomes?" We test exactly that against
historical surveillance trends, using data already on disk:

  * Malaria Atlas Project Pf Subnational: observed PfPR + incidence, per admin-1
    region, 2010-2022.
  * WHO GHO: modelled ITN access (%) per country, 2015-2022.

Protocol (genuine out-of-sample):
  1. For each region, calibrate the SEITAR transmission parameters (biting rate a,
     symptomatic fraction p) ONCE at the 2015 baseline -- to the 2015 observed
     prevalence + incidence at the 2015 observed ITN access.
  2. Holding (a,p) fixed, DRIVE the model forward year by year with the ACTUAL
     observed ITN access trajectory (2016-2022) and predict PfPR each year.
  3. Compare predicted vs observed PfPR in the HELD-OUT years 2016-2022.

We report pooled prediction skill (level + change), trend-direction agreement,
and median error. Honest framing: ITN access is the only time-varying driver in
the model, while real declines also reflect ACT/IRS/SMC scale-up, so the model is
expected to capture the DIRECTION and a fraction of the magnitude, not match
exactly. A positive, well-correlated prediction is meaningful external support.

Run: python temporal_validation.py
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from extended_ode import ExtendedCounty
import fast_sim as fs

DATA = "malaria-data-for-modeling-dynamics"
BASE_YEAR = 2015
END_YEAR = 2022
R_CONST = 0.4


def load_panel():
    sub = pd.read_csv(f"{DATA}/Malaria Atlas Project/Pf Subnational.csv")
    piv = sub.pivot_table(index=["ISO3", "Name", "Year"], columns="Metric",
                          values="Value").reset_index()
    piv = piv.rename(columns={"Infection Prevalence": "PfPR", "Incidence Rate": "inc"})
    piv = piv.dropna(subset=["PfPR", "inc"])
    itn = pd.read_csv(f"{DATA}/The Global Health Observatory/ITN Access.csv")
    itn = itn[itn.IndicatorCode == "MALARIA_ITN_COVERAGE"][
        ["SpatialDimValueCode", "Period", "FactValueNumeric"]].rename(
        columns={"SpatialDimValueCode": "ISO3", "Period": "Year", "FactValueNumeric": "ITN"})
    df = piv.merge(itn, on=["ISO3", "Year"], how="inner")
    df = df[(df.Year >= BASE_YEAR) & (df.Year <= END_YEAR)]
    df["PfPR"] = df.PfPR / 100.0
    df["ITN"] = df.ITN / 100.0
    return df


def calibrate(prev0, inc0, cov0):
    def resid(theta):
        a, p = theta
        pr, ic = fs.seitar_steady(a, p, cov0, 1.0, 1.0 / 10)
        return [(pr - prev0) / max(prev0, 1e-3), (ic - inc0) / max(inc0, 1e-3)]
    sol = least_squares(resid, x0=[1.0, 0.3], bounds=([0.05, 0.02], [15.0, 0.98]),
                        diff_step=0.05, xtol=1e-3, ftol=1e-3, max_nfev=60)
    return sol.x


def predict_trajectory(a, p, cov_by_year, years):
    """Calibrate-then-drive: start at the baseline-year steady state, then each
    year set ITN coverage to the observed value and integrate 365 days; record
    annual mean prevalence."""
    c = ExtendedCounty(biting=a, coverage=cov_by_year[years[0]], multiplier=1.0,
                       treat=10, p_sympt=p, resistance=R_CONST)
    for _ in range(3000):                      # burn in to baseline equilibrium
        c.step(0.0, seasonal=False)
    pred = {}
    for yr in years:
        c.C = float(cov_by_year[yr]); c.tau = 0.0     # drive observed access (annual)
        prev = []
        for _ in range(365):
            c.step(0.0, seasonal=True)
            prev.append(c.prevalence())
        pred[yr] = float(np.mean(prev))
    return pred


def main():
    df = load_panel()
    # keep regions with a full 2015-2022 panel and meaningful endemicity
    years = list(range(BASE_YEAR, END_YEAR + 1))
    g = df.groupby(["ISO3", "Name"])
    regions = []
    for (iso, name), sub in g:
        sub = sub.set_index("Year")
        if not all(y in sub.index for y in years):
            continue
        if sub.loc[BASE_YEAR, "PfPR"] < 0.02:   # skip near-zero (uninformative)
            continue
        regions.append((iso, name, sub))
    print(f"{len(regions)} regions with full {BASE_YEAR}-{END_YEAR} panel", flush=True)

    rows = []
    for iso, name, sub in regions:
        cov = {y: sub.loc[y, "ITN"] for y in years}
        a, p = calibrate(sub.loc[BASE_YEAR, "PfPR"], sub.loc[BASE_YEAR, "inc"], cov[BASE_YEAR])
        pred = predict_trajectory(a, p, cov, years)
        for y in years:
            rows.append(dict(ISO3=iso, region=name, year=y,
                             obs=sub.loc[y, "PfPR"], pred=pred[y],
                             itn=cov[y], held_out=(y > BASE_YEAR)))
    res = pd.DataFrame(rows)
    res.to_csv("temporal_validation.csv", index=False)

    ho = res[res.held_out]
    # level skill
    r_level = stats.pearsonr(ho.obs, ho.pred)[0]
    mae = float((ho.obs - ho.pred).abs().median())
    # change skill: predicted vs observed delta from baseline to 2022
    last = res[res.year == END_YEAR].set_index(["ISO3", "region"])
    base = res[res.year == BASE_YEAR].set_index(["ISO3", "region"])
    d_obs = (last.obs - base.obs)
    d_pred = (last.pred - base.pred)
    r_change = stats.pearsonr(d_obs, d_pred)[0]
    sign_agree = float((np.sign(d_obs) == np.sign(d_pred)).mean())
    # fraction of observed change explained (pooled regression through origin)
    beta = float((d_obs * d_pred).sum() / (d_pred ** 2).sum())

    print("\n===== OUT-OF-SAMPLE TEMPORAL VALIDATION (held-out 2016-2022) =====")
    print(f"regions: {len(regions)}   held-out region-years: {len(ho)}")
    print(f"PfPR level   : Pearson r = {r_level:.3f}   median |err| = {mae*100:.2f} pp")
    print(f"PfPR change  : Pearson r = {r_change:.3f}   (2015->2022 delta)")
    print(f"trend-direction agreement: {sign_agree*100:.1f}% of regions")
    print(f"obs change vs pred change slope (beta): {beta:.2f} "
          f"(1=model fully explains; <1 expected, other interventions too)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(ho.obs * 100, ho.pred * 100, s=10, alpha=0.35, color="#4477aa")
    lim = max(ho.obs.max(), ho.pred.max()) * 100 * 1.05
    ax[0].plot([0, lim], [0, lim], "k--", lw=1)
    ax[0].set_xlabel("observed PfPR (%)"); ax[0].set_ylabel("predicted PfPR (%)")
    ax[0].set_title(f"(a) Held-out PfPR (r={r_level:.2f}, {len(ho)} region-years)")
    ax[1].scatter(d_obs * 100, d_pred * 100, s=12, alpha=0.4, color="#cc6677")
    lo = min(d_obs.min(), d_pred.min()) * 100 * 1.05
    hi = max(d_obs.max(), d_pred.max()) * 100 * 1.05
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[1].axhline(0, color="gray", lw=0.5); ax[1].axvline(0, color="gray", lw=0.5)
    ax[1].set_xlabel("observed dPfPR 2015->2022 (pp)")
    ax[1].set_ylabel("predicted dPfPR (pp)")
    ax[1].set_title(f"(b) Prevalence change (r={r_change:.2f}, sign {sign_agree*100:.0f}%)")
    plt.tight_layout(); plt.savefig("fig_temporal_validation.png", dpi=150)
    print("saved -> temporal_validation.csv, fig_temporal_validation.png")


if __name__ == "__main__":
    main()

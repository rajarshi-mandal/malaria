"""
Calibrate the endogenous deployment->resistance selection rate to REAL data.

The endogenous-resistance environment (endo_experiment.py) models the ITN
resistance-impact R_i as a STATE that rises under local insecticide selection
pressure:

    dR/dt = k_sel * coverage * (1 - R)  -  k_rev * R          (per day)

This script grounds k_sel in the IR-Mapper Anopheles pyrethroid susceptibility
record (2010-2017), the same library the manuscript already uses. Pyrethroid is
the ITN insecticide class, so its mortality decline IS the resistance rise that
net deployment drives. We:

  1. count-weight mean pyrethroid % mortality by year (the manuscript's weighting),
  2. map mortality -> resistance-impact R via the manuscript's sigmoid pm_to_R,
  3. linear-fit dR/dyr over 2010-2017 (the historical net-scale-up window),
  4. solve dR/dt = k_sel * Cbar * (1-Rbar) for k_sel at the period-mean ITN
     coverage Cbar (WHO GHO), giving a data-grounded per-day selection rate.

Output: endo_resistance_calibration.csv + endo_resistance_calibration.png + a
printed k_sel (per day) used as the env default.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "malaria-data-for-modeling-dynamics"


def pm_to_R(pm_percent):
    """Manuscript map: percent mosquito mortality -> resistance impact in [0,1].
    (sar_sequential.pm_to_R; input here is in PERCENT, converted to fraction)."""
    pm = pm_percent / 100.0
    return 1.0 / (1.0 + np.exp(36.0 * (pm - 0.63)))


def mean_itn_coverage(years):
    """Period-mean modelled ITN access across African countries (WHO GHO),
    as a fraction. Used as the average selection-pressure coverage Cbar."""
    itn = pd.read_csv(f"{DATA}/The Global Health Observatory/ITN Access.csv")
    itn = itn[itn.IndicatorCode == "MALARIA_ITN_COVERAGE"]
    itn = itn[(itn.Period >= years[0]) & (itn.Period <= years[1])]
    # African region only (ITNs are deployed where transmission is)
    itn = itn[itn.ParentLocationCode == "AFR"]
    return float(itn.FactValueNumeric.mean()) / 100.0


def main():
    df = pd.read_csv(f"{DATA}/IR Mapper (Anopheles)/insecticide_resistance.csv")
    # ITN insecticide class only
    pyr = df[df["Insecticide class"] == "pyrethroid"].copy()
    pyr = pyr.dropna(subset=["Percent mortality", "No. mosquitoes tested", "Year"])
    pyr = pyr[(pyr.Year >= 2010) & (pyr.Year <= 2017)]

    # count-weighted mean mortality by year (manuscript weighting)
    rows = []
    for yr, g in pyr.groupby("Year"):
        w = g["No. mosquitoes tested"].to_numpy(float)
        m = g["Percent mortality"].to_numpy(float)
        wm = float((w * m).sum() / w.sum())
        rows.append(dict(year=int(yr), mortality=wm, R=float(pm_to_R(wm)),
                         n_assays=len(g), n_mosq=int(w.sum())))
    cal = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

    # linear fits over the window
    yr = cal.year.to_numpy(float)
    mort_slope, mort_int = np.polyfit(yr, cal.mortality.to_numpy(float), 1)
    R_slope, R_int = np.polyfit(yr, cal.R.to_numpy(float), 1)   # dR per YEAR
    Rbar = float(cal.R.mean())

    Cbar = mean_itn_coverage((2010, 2017))
    # dR/dt = k_sel * Cbar * (1 - Rbar)  ->  per-day k_sel
    dR_per_day = R_slope / 365.0
    k_sel = dR_per_day / (Cbar * (1.0 - Rbar))

    cal.to_csv("endo_resistance_calibration.csv", index=False)

    print("=== IR-Mapper pyrethroid resistance trend (2010-2017) ===")
    print(cal.to_string(index=False))
    print(f"\nmortality trend : {mort_slope:+.2f} %/yr "
          f"(from {cal.mortality.iloc[0]:.1f}% to {cal.mortality.iloc[-1]:.1f}%)")
    print(f"R (impact) trend: {R_slope:+.4f} /yr  (mean R={Rbar:.3f})")
    print(f"mean ITN coverage 2010-2017 (AFR, GHO): Cbar={Cbar:.3f}")
    print(f"\n-> endogenous selection rate  k_sel = {k_sel:.3e} /day")
    print(f"   (dR/dt = k_sel * C * (1-R); at C={Cbar:.2f} gives "
          f"dR/yr={k_sel*Cbar*(1-Rbar)*365:.4f})")
    # a plausible slow reversion: resistance reverts ~ a few %/yr with no nets
    # (fitness cost of resistance; conservative). Report a default.
    k_rev = 0.10 / 365.0   # ~10%/yr relative reversion when C=0
    print(f"   reversion (fitness cost, default) k_rev = {k_rev:.3e} /day "
          f"(~10%/yr at C=0)")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(cal.year, cal.mortality, "o-", color="#cc6677")
    ax[0].plot(yr, mort_int + mort_slope * yr, "k--", lw=1,
               label=f"{mort_slope:+.2f} %/yr")
    ax[0].set_xlabel("year"); ax[0].set_ylabel("pyrethroid % mortality (count-wtd)")
    ax[0].set_title("(a) IR-Mapper resistance rise"); ax[0].legend()
    ax[1].plot(cal.year, cal.R, "o-", color="#4477aa")
    ax[1].plot(yr, R_int + R_slope * yr, "k--", lw=1, label=f"{R_slope:+.4f}/yr")
    ax[1].set_xlabel("year"); ax[1].set_ylabel("resistance impact R")
    ax[1].set_title("(b) mapped to ITN resistance impact"); ax[1].legend()
    plt.tight_layout()
    plt.savefig("endo_resistance_calibration.png", dpi=150)
    print("\nsaved -> endo_resistance_calibration.csv, endo_resistance_calibration.png")


if __name__ == "__main__":
    main()

"""
WS6 -- expand the real-data benchmark to N real admin-1 regions (default 50).

Selects N real admin-1 regions spanning the P. falciparum prevalence gradient from
the Malaria Atlas Project subnational data (joined with WHO GHO ITN access), and
calibrates the SEITAR model per region by jointly fitting the biting rate `a` and
symptomatic fraction `p` to match BOTH observed parasite prevalence and clinical
incidence (fast njit steady-state, fs.seitar_steady -- bit-identical to the scalar
calibrator). All regions are REAL (named admin-1 units); none are synthetic.

Output: real_regions_n{N}.csv  (released benchmark table) + a diagnostic plot.

Run: python build_real_regions.py --n-regions 50
"""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fast_sim as fs

DATA = "malaria-data-for-modeling-dynamics"
YEAR = 2019


def select_regions(n, year=YEAR):
    sub = pd.read_csv(f"{DATA}/Malaria Atlas Project/Pf Subnational.csv")
    cnt = pd.read_csv(f"{DATA}/Malaria Atlas Project/Pf Subnational Counts.csv")
    itn = pd.read_csv(f"{DATA}/The Global Health Observatory/ITN Access.csv")
    sub = sub[sub.Year == year]
    piv = sub.pivot_table(index=["ISO3", "National Unit", "Name"],
                          columns="Metric", values="Value").reset_index()
    piv = piv.rename(columns={"Infection Prevalence": "PfPR",
                              "Incidence Rate": "inc", "Mortality Rate": "mortality"})
    cases = cnt[(cnt.Year == year) & (cnt.Metric == "Clinical Cases")][
        ["ISO3", "Name", "Value"]].rename(columns={"Value": "cases"})
    df = piv.merge(cases, on=["ISO3", "Name"], how="left")
    df["popn"] = np.where(df.inc > 1.0, df.cases / (df.inc / 1000.0), np.nan)
    itn_y = itn[itn.IndicatorCode == "MALARIA_ITN_COVERAGE"].copy()
    itn_y["d"] = (itn_y.Period - year).abs()
    itn_b = (itn_y.sort_values("d").groupby("SpatialDimValueCode").first().reset_index()
             [["SpatialDimValueCode", "FactValueNumeric"]]
             .rename(columns={"SpatialDimValueCode": "ISO3", "FactValueNumeric": "ITN"}))
    df = df.merge(itn_b, on="ISO3", how="left")
    df = df.dropna(subset=["PfPR", "inc", "popn", "ITN"])
    df = df[(df.PfPR > 1.0) & (df.popn > 5e4)].sort_values("PfPR").reset_index(drop=True)
    idx = np.linspace(0, len(df) - 1, n).round().astype(int)
    idx = np.unique(idx)
    sel = df.iloc[idx].reset_index(drop=True)
    print(f"{len(df)} candidate real regions; selected {len(sel)} across "
          f"PfPR {sel.PfPR.min():.1f}-{sel.PfPR.max():.1f}%", flush=True)
    return sel


def calibrate_seitar(prev_obs, inc_obs, cov, mult, treat=10):
    rho = 1.0 / treat

    def resid(theta):
        a, p = theta
        prev, inc = fs.seitar_steady(a, p, cov, mult, rho)
        return [(prev - prev_obs) / max(prev_obs, 1e-3),
                (inc - inc_obs) / max(inc_obs, 1e-3)]

    sol = least_squares(resid, x0=[1.0, 0.3], bounds=([0.05, 0.02], [15.0, 0.98]),
                        diff_step=0.05, xtol=1e-3, ftol=1e-3, max_nfev=80)
    a, p = sol.x
    prev_sim, inc_sim = fs.seitar_steady(a, p, cov, mult, rho)
    return a, p, prev_sim, inc_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-regions", type=int, default=50)
    args = ap.parse_args()

    sel = select_regions(args.n_regions)
    mean_pop = sel.popn.mean()
    rows = []
    for _, r in sel.iterrows():
        mult = r.popn / mean_pop
        cov = r.ITN / 100.0
        a, p, ps, isim = calibrate_seitar(r.PfPR / 100.0, r.inc, cov, mult)
        rows.append(dict(ISO3=r.ISO3, region=r.Name, admin_unit=r["National Unit"],
                         PfPR_obs=r.PfPR / 100.0, PfPR_sim=ps,
                         incidence_obs=r.inc, incidence_sim=isim,
                         mortality=r.mortality, biting_rate=a, p_sympt=p,
                         ITN_coverage=cov, population=r.popn, pop_multiplier=mult,
                         treatment_seeking=10))
    out = pd.DataFrame(rows)
    fn = f"real_regions_n{args.n_regions}.csv"
    out.to_csv(fn, index=False)
    pe = (out.PfPR_sim - out.PfPR_obs).abs().mean()
    ie = (out.incidence_sim - out.incidence_obs).abs().mean()
    print(f"\nCalibrated {len(out)} real regions. "
          f"Mean |error|: prevalence {pe*100:.2f}pp, incidence {ie:.1f}/1000")
    print(f"PfPR_sim range {out.PfPR_sim.min()*100:.1f}-{out.PfPR_sim.max()*100:.1f}%")
    print(f"saved -> {fn}")

    # diagnostic obs-vs-sim plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(out.PfPR_obs * 100, out.PfPR_sim * 100, s=18, alpha=0.7)
    lim = max(out.PfPR_obs.max(), out.PfPR_sim.max()) * 100 * 1.05
    ax[0].plot([0, lim], [0, lim], "k--", lw=1)
    ax[0].set_xlabel("observed PfPR (%)"); ax[0].set_ylabel("simulated PfPR (%)")
    ax[0].set_title(f"(a) Prevalence fit ({len(out)} real regions)")
    ax[1].scatter(out.incidence_obs, out.incidence_sim, s=18, alpha=0.7, color="#cc6677")
    lim2 = max(out.incidence_obs.max(), out.incidence_sim.max()) * 1.05
    ax[1].plot([0, lim2], [0, lim2], "k--", lw=1)
    ax[1].set_xlabel("observed incidence (/1000/yr)"); ax[1].set_ylabel("simulated incidence")
    ax[1].set_title("(b) Incidence fit")
    plt.tight_layout()
    plt.savefig(f"fig_real_regions_n{args.n_regions}.png", dpi=150)
    print(f"saved -> fig_real_regions_n{args.n_regions}.png")


if __name__ == "__main__":
    main()

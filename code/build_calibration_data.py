"""
Item D: build a real-data calibration table for admin1 regions.

Joins Malaria Atlas Project subnational metrics (Pf infection prevalence,
incidence rate, mortality) and clinical-case counts with WHO GHO national ITN
coverage. Population per region is derived self-consistently from
    population = clinical_cases / (incidence_per_1000 / 1000).

We then select a heterogeneous set of real high-burden admin1 regions spanning
a transmission gradient (low -> high PfPR) to serve as the "counties" of the
calibrated allocation environment.

Output: calibration_regions.csv
"""
import numpy as np
import pandas as pd

DATA = "malaria-data-for-modeling-dynamics"
YEAR = 2019
# High-burden countries with rich subnational data AND GHO ITN coverage.
FOCUS_ISO3 = ["COD", "NGA", "MOZ", "UGA", "TZA", "BFA", "MLI", "CIV", "CMR", "AGO"]
N_REGIONS = 8


def main():
    sub = pd.read_csv(f"{DATA}/Malaria Atlas Project/Pf Subnational.csv")
    cnt = pd.read_csv(f"{DATA}/Malaria Atlas Project/Pf Subnational Counts.csv")
    itn = pd.read_csv(f"{DATA}/The Global Health Observatory/ITN Access.csv")

    sub = sub[sub.Year == YEAR]
    piv = sub.pivot_table(index=["ISO3", "National Unit", "Name"],
                          columns="Metric", values="Value").reset_index()
    piv = piv.rename(columns={"Infection Prevalence": "PfPR_pct",
                              "Incidence Rate": "incidence_per_1000",
                              "Mortality Rate": "mortality"})

    cases = cnt[(cnt.Year == YEAR) & (cnt.Metric == "Clinical Cases")][
        ["ISO3", "Name", "Value"]].rename(columns={"Value": "clinical_cases"})
    df = piv.merge(cases, on=["ISO3", "Name"], how="left")

    # derive population from cases / incidence
    df["population"] = np.where(df["incidence_per_1000"] > 1.0,
                                df["clinical_cases"] / (df["incidence_per_1000"] / 1000.0),
                                np.nan)

    # national ITN coverage (nearest year <= YEAR, else nearest)
    itn_y = itn[itn.IndicatorCode == "MALARIA_ITN_COVERAGE"].copy()
    itn_y["dist"] = (itn_y.Period - YEAR).abs()
    itn_best = (itn_y.sort_values("dist").groupby("SpatialDimValueCode")
                .first().reset_index()[["SpatialDimValueCode", "FactValueNumeric", "Period"]])
    itn_best = itn_best.rename(columns={"SpatialDimValueCode": "ISO3",
                                        "FactValueNumeric": "ITN_coverage_pct",
                                        "Period": "ITN_year"})
    df = df.merge(itn_best, on="ISO3", how="left")

    # keep focus countries with complete, endemic data
    df = df[df.ISO3.isin(FOCUS_ISO3)]
    df = df.dropna(subset=["PfPR_pct", "incidence_per_1000", "population", "ITN_coverage_pct"])
    df = df[(df.PfPR_pct > 1.0) & (df.population > 5e4)]

    print(f"{len(df)} candidate regions across {df.ISO3.nunique()} countries "
          f"(year {YEAR}).")
    print("PfPR range:", round(df.PfPR_pct.min(), 1), "-", round(df.PfPR_pct.max(), 1))

    # select N regions spanning the PfPR gradient (evenly spaced quantiles)
    df = df.sort_values("PfPR_pct").reset_index(drop=True)
    idx = np.linspace(0, len(df) - 1, N_REGIONS).round().astype(int)
    sel = df.iloc[idx].copy().reset_index(drop=True)

    cols = ["ISO3", "National Unit", "Name", "PfPR_pct", "incidence_per_1000",
            "mortality", "clinical_cases", "population", "ITN_coverage_pct", "ITN_year"]
    sel = sel[cols]
    sel.to_csv("calibration_regions.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nSelected calibration regions:")
    print(sel.to_string())
    print("\nsaved -> calibration_regions.csv")


if __name__ == "__main__":
    main()

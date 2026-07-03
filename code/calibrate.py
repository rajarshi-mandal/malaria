"""
Item D, step 3: calibrate per-region transmission to observed clinical INCIDENCE.

For each real admin1 region we fit a single transmission parameter -- the
baseline mosquito biting rate `a` -- so that the compartmental ODE's endemic
annual clinical incidence (cases per 1,000 population) matches the Malaria Atlas
Project observed incidence, GIVEN that region's observed baseline ITN coverage.
Everything else (populations, ITN coverage, mortality) is read from data.

We calibrate to incidence (a *flow*: the E->I transition sigma_H*E_H integrated
over a year) rather than to PfPR. The current SEITR model -- with ~10-day
treatment removal and no asymptomatic-carriage reservoir -- structurally caps
parasite prevalence near ~10%, so it cannot reproduce observed PfPR of 16-61%.
Incidence is both model-compatible and the quantity the allocation objective
(cases averted) actually targets. We still REPORT modeled prevalence as a
diagnostic; closing the prevalence gap motivates the extended ODE (asymptomatic
compartment) explored separately.

Output: calibrated_regions.csv  (data + fitted `a` + sim-vs-observed check)
"""
import numpy as np
import pandas as pd

from sar_sequential import MalariaCounty

R_CONST = 0.4          # representative insecticide-resistance impact
SEED_PER_CAPITA = 4e-5 # daily importation rate (fraction of N_H) -> low floor
T_SS = 1500            # steady-state horizon (days)
AVG_WINDOW = 200       # average prevalence over the final window
MEAN_POP = None        # set at runtime


def steady_metrics(a, C, multiplier, treat=10):
    """Run the ODE to steady state; return (annual_incidence_per_1000, prevalence)."""
    rs = np.full(T_SS + 400, R_CONST)
    c = MalariaCounty(rs, pop_multiply=multiplier, vegetation_biting=a,
                      init_coverage=C, treatment_seeking=treat)
    c.external_seeding = SEED_PER_CAPITA * c.N_H
    c.delta = 0.001
    for _ in range(T_SS):                       # burn-in to steady state
        c.step(0.0)
    new_cases, prev = 0.0, []                    # measure over one final year
    for _ in range(365):
        new_cases += c.sigma_H * c.E_H           # E -> I clinical incidence flow
        c.step(0.0)
        prev.append((c.I_H + c.T_H) / c.N_H)
    incidence = new_cases / c.N_H * 1000.0
    return incidence, float(np.mean(prev))


def fit_biting_rate(target_inc, C, multiplier, treat=10,
                    lo=0.02, hi=3.0, iters=38):
    """Bisection on `a` so steady-state annual incidence/1000 == target."""
    i_lo = steady_metrics(lo, C, multiplier, treat)[0]
    i_hi = steady_metrics(hi, C, multiplier, treat)[0]
    if target_inc <= i_lo:
        return lo, *steady_metrics(lo, C, multiplier, treat)
    if target_inc >= i_hi:
        return hi, *steady_metrics(hi, C, multiplier, treat)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        inc = steady_metrics(mid, C, multiplier, treat)[0]
        if inc < target_inc:
            lo = mid
        else:
            hi = mid
    a = 0.5 * (lo + hi)
    inc, prev = steady_metrics(a, C, multiplier, treat)
    return a, inc, prev


def main():
    df = pd.read_csv("calibration_regions.csv")
    mean_pop = df.population.mean()
    rows = []
    print(f"Calibrating {len(df)} regions to observed incidence (mean pop {mean_pop:,.0f})...\n")
    for _, r in df.iterrows():
        target_inc = r.incidence_per_1000
        C = r.ITN_coverage_pct / 100.0
        mult = r.population / mean_pop
        a, sim_inc, sim_prev = fit_biting_rate(target_inc, C, mult)
        rows.append({
            "ISO3": r.ISO3, "region": r.Name, "country": r["National Unit"],
            "incidence_obs": target_inc, "incidence_sim": sim_inc,
            "PfPR_obs": r.PfPR_pct / 100.0, "PfPR_sim": sim_prev,
            "biting_rate": a, "ITN_coverage": C,
            "pop_multiplier": mult, "population": r.population,
            "mortality": r.mortality, "treatment_seeking": 10,
        })
        print(f"{r.Name:16s} ({r.ISO3}) | inc obs {target_inc:6.1f} "
              f"sim {sim_inc:6.1f} | a*={a:.3f} | C={C:.2f} | mult={mult:.2f} "
              f"| PfPR sim {sim_prev:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv("calibrated_regions.csv", index=False)
    err = (out.incidence_sim - out.incidence_obs).abs()
    rel = (err / out.incidence_obs)
    print(f"\nIncidence calibration error: mean abs {err.mean():.1f}/1000 "
          f"({rel.mean()*100:.1f}% rel), max {err.max():.1f}")
    print("saved -> calibrated_regions.csv")


if __name__ == "__main__":
    main()

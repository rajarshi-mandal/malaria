"""
Extended SEITAR-SEI malaria model with an asymptomatic-carriage reservoir.

Motivation (from Item D): the SEITR model cannot reproduce observed parasite
prevalence (caps ~3% vs observed 1.8-61%) because it lacks asymptomatic carriage
-- the dominant, slowly-clearing P. falciparum reservoir in endemic settings.

Human compartments: S, E, I (clinical), T (treated), A (asymptomatic), R.
  E -> I  at rate p * sigma_H       (symptomatic fraction p)
  E -> A  at rate (1-p) * sigma_H   (asymptomatic)
  A clears slowly (gamma_A ~ 1/200 d) and is infectious to mosquitoes at reduced
  relative infectiousness kappa_A. This sustains realistic prevalence while
  clinical incidence (p * sigma_H * E) stays moderate.

Calibration per region (joint, alternating):
  * fit biting rate `a`            -> match observed parasite prevalence (I+T+A)/N
  * fit symptomatic fraction `p`   -> match observed clinical incidence
iterated to a fixed point. Mosquito block (SEI) is unchanged.

Output: extended_calibrated_regions.csv
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

R_CONST = 0.4
SEED_PER_CAPITA = 4e-5
T_SS = 3000          # burn-in: asymptomatic reservoir is slow (~200d time constant)
AVG = 365


class ExtendedCounty:
    """SEITAR-SEI county with forward-Euler (dt=1 day)."""

    def __init__(self, biting, coverage, multiplier, treat=10, p_sympt=0.5,
                 gamma_A=1/300, kappa_A=0.5, resistance=R_CONST, seeding_pc=SEED_PER_CAPITA):
        self.mu_H = 1/(60*365); self.mu_M = 1/12
        self.gamma = 1/50; self.gamma_T = 1/21
        self.rho = 1/treat
        self.delta_H = 0.01; self.delta_T = 0.001
        self.sigma_H = 1/14; self.sigma_M = 1/10
        self.kappa = 0.3                       # treated relative infectiousness
        self.kappa_A = kappa_A                 # asymptomatic relative infectiousness
        self.p = p_sympt                       # symptomatic fraction
        self.gamma_A = gamma_A                 # asymptomatic clearance (slow)
        self.a = biting; self.b = 0.2; self.c = 0.2
        self.C = coverage; self.delta = 0.001
        self.R = resistance
        self.multiplier = multiplier
        self.N_H = 10000*multiplier; self.N_M = 30000*multiplier
        self.external_seeding = seeding_pc*self.N_H
        self.tau = 0
        self.reset()

    def reset(self):
        m = self.multiplier
        self.S_H = (self.N_H-1400); self.E_H = 1000*m; self.I_H = 200*m
        self.T_H = 100*m; self.A_H = 500*m; self.R_H = 100*m
        self.S_M = (self.N_M-6000); self.E_M = 3000*m; self.I_M = 3000*m
        self.t = 0; self.tau = 0
        self.a_t = self.a

    def step(self, new_itns=0.0, seasonal=False):
        curr = self.C*self.N_H
        new = float(np.clip(new_itns, 0, self.N_H))
        tot = curr + new
        self.C = min(tot/self.N_H, 1.0)
        self.tau = self.tau*curr/tot if tot > 0 else self.tau
        self.a_t = self.a*(1+0.2*np.sin(2*np.pi*self.t/180)) if seasonal else self.a
        theta = self.C*np.exp(-self.delta*self.tau)*(1-0.5*self.R)
        Lam_H = self.a_t*self.c*self.I_M/self.N_M*max(1-theta, 0)
        Lam_M = self.a_t*self.b*(self.I_H + self.kappa*self.T_H + self.kappa_A*self.A_H)/self.N_H

        dS = self.mu_H*self.N_H - Lam_H*self.S_H + self.gamma*self.R_H - self.mu_H*self.S_H
        dE = Lam_H*self.S_H - self.sigma_H*self.E_H - self.mu_H*self.E_H + self.external_seeding
        dI = self.p*self.sigma_H*self.E_H - (self.rho+self.mu_H+self.delta_H)*self.I_H
        dA = (1-self.p)*self.sigma_H*self.E_H - (self.gamma_A+self.mu_H)*self.A_H
        dT = self.rho*self.I_H - (self.mu_H+self.delta_T+self.gamma_T)*self.T_H
        dR = self.gamma_T*self.T_H + self.gamma_A*self.A_H - self.gamma*self.R_H - self.mu_H*self.R_H
        dSM = self.mu_M*self.N_M - Lam_M*self.S_M - self.mu_M*self.S_M
        dEM = Lam_M*self.S_M - self.sigma_M*self.E_M - self.mu_M*self.E_M
        dIM = self.sigma_M*self.E_M - self.mu_M*self.I_M

        self.S_H += dS; self.E_H += dE; self.I_H += dI; self.A_H += dA
        self.T_H += dT; self.R_H += dR
        self.S_M += dSM; self.E_M += dEM; self.I_M += dIM
        self.t += 1; self.tau += 1
        new_clinical = self.p*self.sigma_H*self.E_H
        return new_clinical

    def prevalence(self):
        return (self.I_H + self.T_H + self.A_H)/self.N_H


def steady(biting, coverage, multiplier, p, treat=10):
    c = ExtendedCounty(biting, coverage, multiplier, treat=treat, p_sympt=p)
    for _ in range(T_SS):
        c.step(0.0)
    prev, inc = [], 0.0
    for _ in range(AVG):
        inc += c.step(0.0)
        prev.append(c.prevalence())
    return float(np.mean(prev)), inc/c.N_H*1000.0


def calibrate_region(prev_obs, inc_obs, coverage, multiplier, treat=10):
    """Joint 2-D fit of (biting rate a, symptomatic fraction p) to match BOTH
    observed parasite prevalence and clinical incidence (relative residuals)."""
    def resid(theta):
        a, p = theta
        prev, inc = steady(a, coverage, multiplier, p, treat)
        return [(prev - prev_obs) / max(prev_obs, 1e-3),
                (inc - inc_obs) / max(inc_obs, 1e-3)]
    sol = least_squares(resid, x0=[1.0, 0.3], bounds=([0.05, 0.02], [15.0, 0.98]),
                        diff_step=0.05, xtol=1e-3, ftol=1e-3, max_nfev=80)
    a, p = sol.x
    prev_sim, inc_sim = steady(a, coverage, multiplier, p, treat)
    return a, p, prev_sim, inc_sim


def main():
    df = pd.read_csv("calibrated_regions.csv")
    rows = []
    print("Calibrating SEITAR to BOTH prevalence and incidence...\n")
    for _, r in df.iterrows():
        a, p, ps, is_ = calibrate_region(r.PfPR_obs, r.incidence_obs,
                                         r.ITN_coverage, r.pop_multiplier)
        rows.append({"ISO3": r.ISO3, "region": r.region, "country": r.country,
                     "PfPR_obs": r.PfPR_obs, "PfPR_sim": ps,
                     "incidence_obs": r.incidence_obs, "incidence_sim": is_,
                     "biting_rate": a, "p_sympt": p, "ITN_coverage": r.ITN_coverage,
                     "pop_multiplier": r.pop_multiplier, "treatment_seeking": 10})
        print(f"{r.region:16s} | PfPR obs {r.PfPR_obs:.3f} sim {ps:.3f} | "
              f"inc obs {r.incidence_obs:6.1f} sim {is_:6.1f} | a*={a:.3f} p*={p:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv("extended_calibrated_regions.csv", index=False)
    pe = (out.PfPR_sim-out.PfPR_obs).abs().mean()
    ie = (out.incidence_sim-out.incidence_obs).abs().mean()
    print(f"\nMean |error|: prevalence {pe:.4f}, incidence {ie:.1f}/1000")
    print("saved -> extended_calibrated_regions.csv")


if __name__ == "__main__":
    main()

"""
Tier 2a: numerical-integration validation.

The simulator uses forward Euler with dt = 1 day. Reviewers will (rightly) ask
whether that is accurate. We validate against an adaptive high-accuracy reference
(scipy solve_ivp, RK45, tight tolerances) and a fixed-step RK4, for BOTH the
SEITR and the extended SEITAR vector fields, with seasonal forcing and fixed ITN
coverage. We report a step-size convergence study (dt = 1, 0.5, 0.25, 0.1) and the
max relative trajectory error vs the reference.

Output: console table + fig_numerics.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---- shared epidemiological parameters (match MalariaCounty / ExtendedCounty) ----
P = dict(mu_H=1/(60*365), mu_M=1/12, gamma=1/50, gamma_T=1/21, rho=1/10,
         delta_H=0.01, delta_T=0.001, sigma_H=1/14, sigma_M=1/10,
         kappa=0.3, b=0.2, c=0.2, a_bar=0.5, C=0.4, Rres=0.4, delta_itn=0.001,
         N_H=10000.0, N_M=30000.0, seeding=2.0, tau_itn=30.0,
         kappa_A=0.5, gamma_A=1/300, p_sympt=0.45)


def a_of_t(t):
    return P["a_bar"] * (1 + 0.2*np.sin(2*np.pi*t/180))


def theta_itn():
    return P["C"]*np.exp(-P["delta_itn"]*P["tau_itn"])*(1-0.5*P["Rres"])


def rhs_seitr(t, y):
    S_H, E_H, I_H, T_H, R_H, S_M, E_M, I_M = y
    a = a_of_t(t); th = theta_itn()
    Lam_H = a*P["c"]*I_M/P["N_M"]*max(1-th, 0)
    Lam_M = a*P["b"]*(I_H+P["kappa"]*T_H)/P["N_H"]
    dS_H = P["mu_H"]*P["N_H"] - Lam_H*S_H + P["gamma"]*R_H - P["mu_H"]*S_H
    dE_H = Lam_H*S_H - P["sigma_H"]*E_H - P["mu_H"]*E_H + P["seeding"]
    dI_H = P["sigma_H"]*E_H - (P["rho"]+P["mu_H"]+P["delta_H"])*I_H
    dT_H = P["rho"]*I_H - (P["mu_H"]+P["delta_T"]+P["gamma_T"])*T_H
    dR_H = P["gamma_T"]*T_H - P["gamma"]*R_H - P["mu_H"]*R_H
    dS_M = P["mu_M"]*P["N_M"] - Lam_M*S_M - P["mu_M"]*S_M
    dE_M = Lam_M*S_M - P["sigma_M"]*E_M - P["mu_M"]*E_M
    dI_M = P["sigma_M"]*E_M - P["mu_M"]*I_M
    return np.array([dS_H, dE_H, dI_H, dT_H, dR_H, dS_M, dE_M, dI_M])


def rhs_seitar(t, y):
    S_H, E_H, I_H, T_H, A_H, R_H, S_M, E_M, I_M = y
    a = a_of_t(t); th = theta_itn(); p = P["p_sympt"]
    Lam_H = a*P["c"]*I_M/P["N_M"]*max(1-th, 0)
    Lam_M = a*P["b"]*(I_H+P["kappa"]*T_H+P["kappa_A"]*A_H)/P["N_H"]
    dS_H = P["mu_H"]*P["N_H"] - Lam_H*S_H + P["gamma"]*R_H - P["mu_H"]*S_H
    dE_H = Lam_H*S_H - P["sigma_H"]*E_H - P["mu_H"]*E_H + P["seeding"]
    dI_H = p*P["sigma_H"]*E_H - (P["rho"]+P["mu_H"]+P["delta_H"])*I_H
    dA_H = (1-p)*P["sigma_H"]*E_H - (P["gamma_A"]+P["mu_H"])*A_H
    dT_H = P["rho"]*I_H - (P["mu_H"]+P["delta_T"]+P["gamma_T"])*T_H
    dR_H = P["gamma_T"]*T_H + P["gamma_A"]*A_H - P["gamma"]*R_H - P["mu_H"]*R_H
    dS_M = P["mu_M"]*P["N_M"] - Lam_M*S_M - P["mu_M"]*S_M
    dE_M = Lam_M*S_M - P["sigma_M"]*E_M - P["mu_M"]*E_M
    dI_M = P["sigma_M"]*E_M - P["mu_M"]*I_M
    return np.array([dS_H, dE_H, dI_H, dT_H, dA_H, dR_H, dS_M, dE_M, dI_M])


def euler(rhs, y0, T, dt):
    n = int(T/dt); y = np.array(y0, float); traj = [y.copy()]
    t = 0.0
    for _ in range(n):
        y = y + dt*rhs(t, y); t += dt
        traj.append(y.copy())
    return np.array(traj), np.linspace(0, T, n+1)


def rk4(rhs, y0, T, dt):
    n = int(T/dt); y = np.array(y0, float); traj = [y.copy()]; t = 0.0
    for _ in range(n):
        k1 = rhs(t, y); k2 = rhs(t+dt/2, y+dt/2*k1)
        k3 = rhs(t+dt/2, y+dt/2*k2); k4 = rhs(t+dt, y+dt*k3)
        y = y + dt/6*(k1+2*k2+2*k3+k4); t += dt
        traj.append(y.copy())
    return np.array(traj), np.linspace(0, T, n+1)


def max_rel_err(traj, tt, ref_sol):
    ref = ref_sol.sol(tt).T
    denom = np.maximum(np.abs(ref).max(axis=0), 1.0)
    return np.max(np.abs(traj - ref) / denom)


def validate(name, rhs, y0, T=180):
    ref = solve_ivp(rhs, [0, T], y0, method="RK45", rtol=1e-9, atol=1e-9,
                    dense_output=True, max_step=0.5)
    print(f"\n=== {name} (reference: solve_ivp RK45, rtol=1e-9) ===")
    rows = {}
    for dt in [1.0, 0.5, 0.25, 0.1]:
        te, tt = euler(rhs, y0, T, dt)
        rows[dt] = max_rel_err(te, tt, ref)
        print(f"  Euler dt={dt:<4} | max rel err vs ref: {rows[dt]:.3e}")
    tr, ttr = rk4(rhs, y0, T, 1.0)
    rk4_err = max_rel_err(tr, ttr, ref)
    print(f"  RK4   dt=1.0  | max rel err vs ref: {rk4_err:.3e}")
    return rows, rk4_err, ref


def main():
    y0_seitr = [8600, 1000, 200, 100, 100, 24000, 3000, 3000]
    y0_seitar = [8600, 1000, 200, 100, 500, 100, 24000, 3000, 3000]
    e1, r1, ref1 = validate("SEITR", rhs_seitr, y0_seitr)
    e2, r2, ref2 = validate("SEITAR", rhs_seitar, y0_seitar)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    dts = [1.0, 0.5, 0.25, 0.1]
    ax1.loglog(dts, [e1[d] for d in dts], "o-", label="SEITR (Euler)")
    ax1.loglog(dts, [e2[d] for d in dts], "s-", label="SEITAR (Euler)")
    ax1.loglog(dts, [dts[i]/dts[0]*e1[1.0] for i in range(len(dts))], "k--",
               alpha=0.5, label="O(dt) reference slope")
    ax1.set_xlabel("Euler step size dt (days)"); ax1.set_ylabel("Max relative error vs RK45")
    ax1.set_title("Step-size convergence (first-order, as expected)")
    ax1.legend(); ax1.grid(alpha=0.3, which="both")

    T = 180
    for rhs, y0, lab, c in [(rhs_seitr, y0_seitr, "I_H SEITR", "#4C72B0"),
                            (rhs_seitar, y0_seitar, "I_H SEITAR", "#C44E52")]:
        ref = solve_ivp(rhs, [0, T], y0, method="RK45", rtol=1e-9, atol=1e-9,
                        dense_output=True, max_step=0.5)
        tt = np.linspace(0, T, 181)
        idx = 2  # I_H index (same in both layouts)
        ax2.plot(tt, ref.sol(tt)[idx], color=c, lw=2, label=f"{lab} (RK45 ref)")
        te, tte = euler(rhs, y0, T, 1.0)
        ax2.plot(tte, te[:, idx], "--", color=c, lw=1, alpha=0.8, label=f"{lab} (Euler dt=1)")
    ax2.set_xlabel("Day"); ax2.set_ylabel("Infectious humans I_H")
    ax2.set_title("Euler dt=1 vs RK45 reference (overlap)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("fig_numerics.png", dpi=150)
    print("\nsaved -> fig_numerics.png")
    print(f"\nSUMMARY: Euler dt=1 max rel err -> SEITR {e1[1.0]:.2e}, SEITAR {e2[1.0]:.2e}")


if __name__ == "__main__":
    main()

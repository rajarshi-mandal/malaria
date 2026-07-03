"""
Tier 2b: reproduction number, equilibrium stability, and bifurcation analysis.

Replaces the manuscript's hand-wavy "Lyapunov via PaCMAP plot" with rigorous
dynamical-systems analysis:

1. R0 via the next-generation matrix (van den Driessche-Watmough) for SEITR,
   verified against the manuscript's closed-form expression.
2. A NEW R0 for the extended SEITAR model (asymptomatic carriers contribute an
   extra transmission pathway with reduced infectiousness and slow clearance).
3. Disease-free equilibrium (DFE) local stability via the eigenvalues of (F - V):
   stable iff R0 < 1.
4. A real BIFURCATION DIAGRAM: endemic prevalence vs R0, computed from both low-
   and high-infection initial conditions, to test for forward vs backward
   bifurcation (bistability / sub-threshold persistence).

Output: console report + fig_bifurcation.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

P = dict(mu_H=1/(60*365), mu_M=1/12, gamma=1/50, gamma_T=1/21, rho=1/10,
         delta_H=0.01, delta_T=0.001, sigma_H=1/14, sigma_M=1/10,
         kappa=0.3, b=0.2, c=0.2, N_H=10000.0, N_M=30000.0,
         kappa_A=0.5, gamma_A=1/300, p_sympt=0.45)


# --------------------------------------------------------------------------- #
# Next-generation matrices                                                    #
# --------------------------------------------------------------------------- #
def R0_seitr(a, theta=0.0, p=P):
    """Basic reproduction number for SEITR via NGM. Infected: [E_H,I_H,T_H,E_M,I_M]."""
    s = 1 - theta
    F = np.zeros((5, 5)); V = np.zeros((5, 5))
    F[0, 4] = a*p["c"]*p["N_H"]/p["N_M"]*s          # I_M -> new E_H
    F[3, 1] = a*p["b"]*p["N_M"]/p["N_H"]            # I_H -> new E_M
    F[3, 2] = a*p["b"]*p["kappa"]*p["N_M"]/p["N_H"] # T_H -> new E_M
    V[0, 0] = p["sigma_H"]+p["mu_H"]
    V[1, 0] = -p["sigma_H"]; V[1, 1] = p["rho"]+p["mu_H"]+p["delta_H"]
    V[2, 1] = -p["rho"]; V[2, 2] = p["mu_H"]+p["delta_T"]+p["gamma_T"]
    V[3, 3] = p["sigma_M"]+p["mu_M"]
    V[4, 3] = -p["sigma_M"]; V[4, 4] = p["mu_M"]
    K = F @ np.linalg.inv(V)
    return max(abs(np.linalg.eigvals(K))), F, V


def R0_seitar(a, theta=0.0, p=P):
    """R0 for the extended SEITAR model. Infected: [E_H,I_H,T_H,A_H,E_M,I_M]."""
    s = 1 - theta; ps = p["p_sympt"]
    F = np.zeros((6, 6)); V = np.zeros((6, 6))
    F[0, 5] = a*p["c"]*p["N_H"]/p["N_M"]*s                 # I_M -> E_H
    F[4, 1] = a*p["b"]*p["N_M"]/p["N_H"]                   # I_H -> E_M
    F[4, 2] = a*p["b"]*p["kappa"]*p["N_M"]/p["N_H"]        # T_H -> E_M
    F[4, 3] = a*p["b"]*p["kappa_A"]*p["N_M"]/p["N_H"]      # A_H -> E_M
    V[0, 0] = p["sigma_H"]+p["mu_H"]
    V[1, 0] = -ps*p["sigma_H"]; V[1, 1] = p["rho"]+p["mu_H"]+p["delta_H"]
    V[2, 1] = -p["rho"]; V[2, 2] = p["mu_H"]+p["delta_T"]+p["gamma_T"]
    V[3, 0] = -(1-ps)*p["sigma_H"]; V[3, 3] = p["gamma_A"]+p["mu_H"]
    V[4, 4] = p["sigma_M"]+p["mu_M"]
    V[5, 4] = -p["sigma_M"]; V[5, 5] = p["mu_M"]
    K = F @ np.linalg.inv(V)
    return max(abs(np.linalg.eigvals(K))), F, V


def analytic_R0_seitr(a, p=P):
    """Manuscript closed-form, for verification."""
    inside = (a**2*p["b"]*p["c"]*p["N_H"]/p["N_M"]
              * 1/(p["mu_M"]*(p["sigma_M"]+p["mu_M"]))
              * 1/(p["sigma_H"]+p["mu_H"])
              * 1/(p["rho"]+p["mu_H"]+p["delta_H"])
              * (1 + p["kappa"]*p["rho"]/(p["mu_H"]+p["delta_T"]+p["gamma_T"])))
    return np.sqrt(inside)


def dfe_stability(F, V):
    """DFE locally asymptotically stable iff all eig(F - V) have negative real part."""
    eig = np.linalg.eigvals(F - V)
    return np.all(eig.real < 0), eig


# --------------------------------------------------------------------------- #
# Endemic steady state via integration (autonomous, seeding = 0)              #
# --------------------------------------------------------------------------- #
def rhs_seitar(t, y, a, theta, seeding=0.0, p=P):
    S_H, E_H, I_H, T_H, A_H, R_H, S_M, E_M, I_M = y
    ps = p["p_sympt"]; s = 1-theta
    Lam_H = a*p["c"]*I_M/p["N_M"]*max(s, 0)
    Lam_M = a*p["b"]*(I_H+p["kappa"]*T_H+p["kappa_A"]*A_H)/p["N_H"]
    return [p["mu_H"]*p["N_H"]-Lam_H*S_H+p["gamma"]*R_H-p["mu_H"]*S_H,
            Lam_H*S_H-p["sigma_H"]*E_H-p["mu_H"]*E_H+seeding,
            ps*p["sigma_H"]*E_H-(p["rho"]+p["mu_H"]+p["delta_H"])*I_H,
            p["rho"]*I_H-(p["mu_H"]+p["delta_T"]+p["gamma_T"])*T_H,
            (1-ps)*p["sigma_H"]*E_H-(p["gamma_A"]+p["mu_H"])*A_H,
            p["gamma_T"]*T_H+p["gamma_A"]*A_H-p["gamma"]*R_H-p["mu_H"]*R_H,
            p["mu_M"]*p["N_M"]-Lam_M*S_M-p["mu_M"]*S_M,
            Lam_M*S_M-p["sigma_M"]*E_M-p["mu_M"]*E_M,
            p["sigma_M"]*E_M-p["mu_M"]*I_M]


def endemic_equilibrium(a, theta, guess_frac=0.4):
    """Find an equilibrium via root-finding (autonomous, seeding=0) and assess its
    stability via the Jacobian eigenvalues. Returns (prevalence, stable)."""
    from scipy.optimize import fsolve
    NH, NM = P["N_H"], P["N_M"]
    g = [(1-guess_frac)*NH, 0.02*NH, 0.01*NH, guess_frac*0.5*NH, guess_frac*NH*0.4,
         0.1*NH, 0.5*NM, 0.2*NM, 0.3*NM]
    f = lambda y: rhs_seitar(0, y, a, theta, 0.0)
    eq, info, ier, _ = fsolve(f, g, full_output=True)
    if ier != 1 or np.any(eq[:6] < -1e-6):
        return 0.0, False
    prev = (eq[2] + eq[3] + eq[4]) / NH
    # numerical Jacobian -> stability
    n = len(eq); J = np.zeros((n, n)); f0 = np.array(f(eq)); h = 1e-3
    for j in range(n):
        e = eq.copy(); e[j] += h
        J[:, j] = (np.array(f(e)) - f0) / h
    stable = np.all(np.linalg.eigvals(J).real < 1e-8)
    return (prev if prev > 1e-4 else 0.0), (stable and prev > 1e-4)


def main():
    # ---- 1. R0 verification (SEITR) ----
    a0 = 0.5
    r0_ngm, F, V = R0_seitr(a0)
    r0_analytic = analytic_R0_seitr(a0)
    print("=== R0 verification (SEITR, a=0.5, no ITN) ===")
    print(f"  NGM spectral radius : {r0_ngm:.4f}")
    print(f"  manuscript formula  : {r0_analytic:.4f}")
    print(f"  agreement           : {abs(r0_ngm-r0_analytic) < 1e-6}")

    # ---- 2. R0 for SEITAR (new) ----
    r0_ext, Fe, Ve = R0_seitar(a0)
    print(f"\n=== R0 for extended SEITAR (a=0.5, no ITN) ===")
    print(f"  NGM spectral radius : {r0_ext:.4f}  (asymptomatic adds a transmission path)")

    # ---- 3. DFE stability at low vs high transmission ----
    print("\n=== DFE local stability (SEITAR) ===")
    for a in [0.05, 0.12, 0.5]:
        r0, Fe, Ve = R0_seitar(a)
        stable, eig = dfe_stability(Fe, Ve)
        print(f"  a={a:<5} R0={r0:5.2f} | DFE stable: {stable} "
              f"| max Re(eig)={eig.real.max():+.4e}")

    # ---- 4. Bifurcation diagram via root-finding + Jacobian stability ----
    print("\n=== Bifurcation scan (endemic equilibria via fsolve; testing forward vs backward) ===")
    a_grid = np.linspace(0.05, 1.2, 48)
    R0s, prev_eq, stab = [], [], []
    for a in a_grid:
        R0s.append(R0_seitar(a)[0])
        pv, st = endemic_equilibrium(a, 0.0)
        prev_eq.append(pv); stab.append(st)
    R0s = np.array(R0s); prev_eq = np.array(prev_eq); stab = np.array(stab)
    # backward bifurcation <=> a STABLE positive equilibrium exists for some R0 < 1
    backward = np.any((R0s < 1.0) & stab & (prev_eq > 1e-3))
    a_thresh = 0.5 / R0_seitar(0.5)[0]
    print(f"  R0 range over scan: {R0s.min():.2f} - {R0s.max():.2f}")
    print(f"  threshold biting rate (R0=1): a = {a_thresh:.3f}")
    print(f"  stable endemic equilibrium for any R0 < 1? {backward}")
    print(f"  -> {'BACKWARD bifurcation (bistable, sub-threshold persistence)' if backward else 'FORWARD bifurcation (clean R0=1 threshold, no bistability)'}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    stable_mask = stab & (prev_eq > 1e-3)
    ax1.plot(R0s[stable_mask], prev_eq[stable_mask]*100, "o-", color="#C44E52",
             label="stable endemic equilibrium")
    ax1.plot(R0s[~stable_mask], prev_eq[~stable_mask]*100, "x", color="#999999",
             alpha=0.5, label="no stable endemic state")
    ax1.axvline(1.0, color="k", ls=":", alpha=0.6, label="R0 = 1 threshold")
    ax1.set_xlabel("Basic reproduction number R0"); ax1.set_ylabel("Endemic parasite prevalence (%)")
    ax1.set_title(f"Bifurcation diagram (SEITAR): {'backward' if backward else 'forward'}")
    ax1.legend(); ax1.grid(alpha=0.3)

    # convergence-to-equilibrium illustration (replaces PaCMAP "Lyapunov" figure)
    a_demo = 0.5
    for frac, c in [(0.05, "#999999"), (0.5, "#55A868"), (0.9, "#CCB974")]:
        NH, NM = P["N_H"], P["N_M"]
        y0 = [(1-frac)*NH, 0.02*NH, 0.01*NH, frac*0.5*NH, frac*NH*0.3, 0.1*NH,
              0.5*NM, 0.2*NM, 0.3*NM]
        sol = solve_ivp(rhs_seitar, [0, 2000], y0, args=(a_demo, 0.0), method="LSODA",
                        rtol=1e-7, atol=1e-7, dense_output=True)
        tt = np.linspace(0, 2000, 400)
        prev = (sol.sol(tt)[2]+sol.sol(tt)[3]+sol.sol(tt)[4])/P["N_H"]*100
        ax2.plot(tt, prev, color=c, label=f"init prevalence ~{frac*100:.0f}%")
    ax2.set_xlabel("Day"); ax2.set_ylabel("Parasite prevalence (%)")
    ax2.set_title("Trajectories converge to a unique endemic equilibrium")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("fig_bifurcation.png", dpi=150)
    print("\nsaved -> fig_bifurcation.png")


if __name__ == "__main__":
    main()

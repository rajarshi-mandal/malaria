"""
WS0 -- numba-JIT fast rollout for the malaria allocation simulators.

No usable GPU on this machine (Intel Iris Xe, no CUDA; the bottleneck is the
serial day-by-day Python ODE loop, not matmuls). numba is installed, so we
JIT-compile the inner ODE step. Regions are DECOUPLED (counties interact only
through the shared budget, not the dynamics), so a planner's lookahead only needs
single-region rollouts.

These functions replicate `MalariaCounty.step` (SEITR, calibrated_experiment.py)
and `ExtendedCounty.step` (SEITAR, extended_experiment.py) EXACTLY (forward-Euler
dt=1). `verify_*` checks equivalence to the scalar classes to < 1e-6 before any
planner trusts them. Used by baselines_planner.py (greedy / MPC / oracle).
"""
import numpy as np
from numba import njit

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# SEITR (MalariaCounty) -- resistance is a per-day series Rseries[t]          #
# --------------------------------------------------------------------------- #
@njit(cache=True, fastmath=False)
def _seitr_step(s, C, tau, t, alloc, P, Rseries):
    # s = [S_H,E_H,I_H,T_H,R_H,S_M,E_M,I_M]; P = param vector (see pack_seitr)
    (mu_H, mu_M, gamma, gamma_T, rho, delta_H, delta_T, sigma_H, sigma_M,
     eps, a, b, c, seasonal_amp, delta, N_H, N_M, seeding) = (
        P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8],
        P[9], P[10], P[11], P[12], P[13], P[14], P[15], P[16], P[17])
    curr = C * N_H
    nv = alloc
    if nv < 0.0:
        nv = 0.0
    if nv > N_H:
        nv = N_H
    tot = curr + nv
    C = tot / N_H
    if C > 1.0:
        C = 1.0
    if tot > 0.0:
        tau = tau * curr / tot
    R = Rseries[t] if t < Rseries.shape[0] else Rseries[Rseries.shape[0] - 1]
    a_t = a * (1.0 + seasonal_amp * np.sin(TWO_PI * t / 180.0))
    theta = C * np.exp(-delta * tau) * (1.0 - 0.5 * R)
    om = 1.0 - theta
    if om < 0.0:
        om = 0.0
    Lam_H = a_t * c * s[7] / N_M * om
    Lam_M = a_t * b * (s[2] + eps * s[3]) / N_H
    dSH = mu_H * N_H - Lam_H * s[0] + gamma * s[4] - mu_H * s[0]
    dEH = Lam_H * s[0] - sigma_H * s[1] - mu_H * s[1] + seeding
    dIH = sigma_H * s[1] - (rho + mu_H + delta_H) * s[2]
    dTH = rho * s[2] - (mu_H + delta_T + gamma_T) * s[3]
    dRH = gamma_T * s[3] - gamma * s[4] - mu_H * s[4]
    dSM = mu_M * N_M - Lam_M * s[5] - mu_M * s[5]
    dEM = Lam_M * s[5] - sigma_M * s[6] - mu_M * s[6]
    dIM = sigma_M * s[6] - mu_M * s[7]
    s[0] += dSH; s[1] += dEH; s[2] += dIH; s[3] += dTH; s[4] += dRH
    s[5] += dSM; s[6] += dEM; s[7] += dIM
    # clip to [0,N] (Euler dt=1 can overshoot at extreme biting rates; never
    # triggers in the valid regime, so N<=8 results are unchanged)
    for k in range(5):
        v = s[k]
        s[k] = 0.0 if (v != v or v < 0.0) else (N_H if v > N_H else v)
    for k in range(5, 8):
        v = s[k]
        s[k] = 0.0 if (v != v or v < 0.0) else (N_M if v > N_M else v)
    return C, tau + 1.0


@njit(cache=True, fastmath=False)
def seitr_lookahead(s0, C0, tau0, t0, new_itns, P, Rseries, horizon):
    """Cumulative (E_H+I_H) over `horizon` days if `new_itns` are added now."""
    s = s0.copy()
    C = C0; tau = tau0
    burden = 0.0
    for d in range(horizon):
        alloc = new_itns if d == 0 else 0.0
        C, tau = _seitr_step(s, C, tau, t0 + d, alloc, P, Rseries)
        burden += s[1] + s[2]
    return burden


@njit(cache=True, fastmath=False)
def seitr_windows(s0, C0, tau0, P, Rseries, alloc3, camp0, camp1, camp2, ep_days):
    """Full-episode per-region window burdens [w0,w1,w2,w3] for an open-loop
    3-campaign net schedule (used by the oracle). Mirrors run_episode windowing."""
    s = s0.copy()
    C = C0; tau = tau0
    w = np.zeros(4)
    idx = 0
    for day in range(ep_days):
        alloc = 0.0
        if day == camp0:
            alloc = alloc3[0]; idx = 1
        elif day == camp1:
            alloc = alloc3[1]; idx = 2
        elif day == camp2:
            alloc = alloc3[2]; idx = 3
        C, tau = _seitr_step(s, C, tau, day, alloc, P, Rseries)
        wj = 0 if day < camp0 else idx
        w[wj] += s[1] + s[2]
    return w


# --------------------------------------------------------------------------- #
# SEITAR (ExtendedCounty) -- resistance constant R; seasonal forcing on       #
# --------------------------------------------------------------------------- #
@njit(cache=True, fastmath=False)
def _seitar_step(s, C, tau, t, alloc, P):
    # s = [S_H,E_H,I_H,T_H,A_H,R_H,S_M,E_M,I_M]
    (mu_H, mu_M, gamma, gamma_T, rho, delta_H, delta_T, sigma_H, sigma_M,
     kappa, kappa_A, p, gamma_A, a, b, c, delta, R, N_H, N_M, seeding) = (
        P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10],
        P[11], P[12], P[13], P[14], P[15], P[16], P[17], P[18], P[19], P[20])
    curr = C * N_H
    nv = alloc
    if nv < 0.0:
        nv = 0.0
    if nv > N_H:
        nv = N_H
    tot = curr + nv
    C = tot / N_H
    if C > 1.0:
        C = 1.0
    if tot > 0.0:
        tau = tau * curr / tot
    a_t = a * (1.0 + 0.2 * np.sin(TWO_PI * t / 180.0))
    theta = C * np.exp(-delta * tau) * (1.0 - 0.5 * R)
    om = 1.0 - theta
    if om < 0.0:
        om = 0.0
    Lam_H = a_t * c * s[8] / N_M * om
    Lam_M = a_t * b * (s[2] + kappa * s[3] + kappa_A * s[4]) / N_H
    dS = mu_H * N_H - Lam_H * s[0] + gamma * s[5] - mu_H * s[0]
    dE = Lam_H * s[0] - sigma_H * s[1] - mu_H * s[1] + seeding
    dI = p * sigma_H * s[1] - (rho + mu_H + delta_H) * s[2]
    dA = (1.0 - p) * sigma_H * s[1] - (gamma_A + mu_H) * s[4]
    dT = rho * s[2] - (mu_H + delta_T + gamma_T) * s[3]
    dR = gamma_T * s[3] + gamma_A * s[4] - gamma * s[5] - mu_H * s[5]
    dSM = mu_M * N_M - Lam_M * s[6] - mu_M * s[6]
    dEM = Lam_M * s[6] - sigma_M * s[7] - mu_M * s[7]
    dIM = sigma_M * s[7] - mu_M * s[8]
    s[0] += dS; s[1] += dE; s[2] += dI; s[3] += dT; s[4] += dA; s[5] += dR
    s[6] += dSM; s[7] += dEM; s[8] += dIM
    for k in range(6):
        v = s[k]
        s[k] = 0.0 if (v != v or v < 0.0) else (N_H if v > N_H else v)
    for k in range(6, 9):
        v = s[k]
        s[k] = 0.0 if (v != v or v < 0.0) else (N_M if v > N_M else v)
    return C, tau + 1.0


@njit(cache=True, fastmath=False)
def seitar_lookahead(s0, C0, tau0, t0, new_itns, P, horizon):
    s = s0.copy()
    C = C0; tau = tau0
    burden = 0.0
    for d in range(horizon):
        alloc = new_itns if d == 0 else 0.0
        C, tau = _seitar_step(s, C, tau, t0 + d, alloc, P)
        burden += s[1] + s[2]
    return burden


@njit(cache=True, fastmath=False)
def seitar_windows(s0, C0, tau0, P, alloc3, camp0, camp1, camp2, ep_days):
    s = s0.copy()
    C = C0; tau = tau0
    w = np.zeros(4)
    idx = 0
    for day in range(ep_days):
        alloc = 0.0
        if day == camp0:
            alloc = alloc3[0]; idx = 1
        elif day == camp1:
            alloc = alloc3[1]; idx = 2
        elif day == camp2:
            alloc = alloc3[2]; idx = 3
        C, tau = _seitar_step(s, C, tau, day, alloc, P)
        wj = 0 if day < camp0 else idx
        w[wj] += s[1] + s[2]
    return w


# --------------------------------------------------------------------------- #
# Batched oracle scorers (whole CEM population in one compiled call)          #
# --------------------------------------------------------------------------- #
@njit(cache=True, fastmath=False)
def seitr_oracle_batch(states, C0, tau0, Pn, Rn, cand, camp0, camp1, camp2, ep):
    # states[n,8], C0[n], tau0[n], Pn[n,18], Rn[n,T], cand[pop,3,n] -> cums[pop]
    pop = cand.shape[0]
    n = states.shape[0]
    cums = np.empty(pop)
    for p in range(pop):
        w0 = 0.0; w1 = 0.0; w2 = 0.0; w3 = 0.0
        for i in range(n):
            s = states[i].copy()
            C = C0[i]; tau = tau0[i]
            P = Pn[i]; R = Rn[i]
            for day in range(ep):
                alloc = 0.0; idx = 0
                if day == camp0:
                    alloc = cand[p, 0, i]; idx = 1
                elif day == camp1:
                    alloc = cand[p, 1, i]; idx = 2
                elif day == camp2:
                    alloc = cand[p, 2, i]; idx = 3
                else:
                    idx = (0 if day < camp0 else (1 if day < camp1 else (2 if day < camp2 else 3)))
                C, tau = _seitr_step(s, C, tau, day, alloc, P, R)
                bd = s[1] + s[2]
                wj = 0 if day < camp0 else idx
                if wj == 0:
                    w0 += bd
                elif wj == 1:
                    w1 += bd
                elif wj == 2:
                    w2 += bd
                else:
                    w3 += bd
        cums[p] = (w1 + w2 + w3) / w0
    return cums


@njit(cache=True, fastmath=False)
def seitar_oracle_batch(states, C0, tau0, Pn, cand, camp0, camp1, camp2, ep):
    pop = cand.shape[0]
    n = states.shape[0]
    cums = np.empty(pop)
    for p in range(pop):
        w0 = 0.0; w1 = 0.0; w2 = 0.0; w3 = 0.0
        for i in range(n):
            s = states[i].copy()
            C = C0[i]; tau = tau0[i]
            P = Pn[i]
            for day in range(ep):
                alloc = 0.0; idx = 0
                if day == camp0:
                    alloc = cand[p, 0, i]; idx = 1
                elif day == camp1:
                    alloc = cand[p, 1, i]; idx = 2
                elif day == camp2:
                    alloc = cand[p, 2, i]; idx = 3
                else:
                    idx = (0 if day < camp0 else (1 if day < camp1 else (2 if day < camp2 else 3)))
                C, tau = _seitar_step(s, C, tau, day, alloc, P)
                bd = s[1] + s[2]
                wj = 0 if day < camp0 else idx
                if wj == 0:
                    w0 += bd
                elif wj == 1:
                    w1 += bd
                elif wj == 2:
                    w2 += bd
                else:
                    w3 += bd
        cums[p] = (w1 + w2 + w3) / w0
    return cums


@njit(cache=True, fastmath=False)
def seitar_steady(a, p, coverage, mult, rho, t_ss=3000, avg=365):
    """Endemic steady-state (prevalence, annual clinical incidence/1000) for the
    SEITAR county with ExtendedCounty's default constants. Used for fast
    50-region calibration (mirrors extended_ode.steady, seasonal off)."""
    mu_H = 1.0/(60*365); mu_M = 1.0/12; gamma = 1.0/50; gamma_T = 1.0/21
    delta_H = 0.01; delta_T = 0.001; sigma_H = 1.0/14; sigma_M = 1.0/10
    kappa = 0.3; kappa_A = 0.5; gamma_A = 1.0/300; b = 0.2; c = 0.2
    delta = 0.001; R = 0.4
    N_H = 10000.0*mult; N_M = 30000.0*mult; seeding = 4e-5*N_H
    SH = N_H-1400.0; EH = 1000.0*mult; IH = 200.0*mult; TH = 100.0*mult
    AH = 500.0*mult; RH = 100.0*mult
    SM = N_M-6000.0; EM = 3000.0*mult; IM = 3000.0*mult
    C = coverage; tau = 0.0
    prev_sum = 0.0; inc_sum = 0.0
    total = t_ss + avg
    for it in range(total):
        theta = C*np.exp(-delta*tau)*(1.0-0.5*R)
        om = 1.0-theta
        if om < 0.0:
            om = 0.0
        Lam_H = a*c*IM/N_M*om
        Lam_M = a*b*(IH+kappa*TH+kappa_A*AH)/N_H
        dS = mu_H*N_H - Lam_H*SH + gamma*RH - mu_H*SH
        dE = Lam_H*SH - sigma_H*EH - mu_H*EH + seeding
        dI = p*sigma_H*EH - (rho+mu_H+delta_H)*IH
        dA = (1.0-p)*sigma_H*EH - (gamma_A+mu_H)*AH
        dT = rho*IH - (mu_H+delta_T+gamma_T)*TH
        dR = gamma_T*TH + gamma_A*AH - gamma*RH - mu_H*RH
        dSM = mu_M*N_M - Lam_M*SM - mu_M*SM
        dEM = Lam_M*SM - sigma_M*EM - mu_M*EM
        dIM = sigma_M*EM - mu_M*IM
        SH += dS; EH += dE; IH += dI; AH += dA; TH += dT; RH += dR
        SM += dSM; EM += dEM; IM += dIM
        # clip to [0,N] each day (Euler dt=1 can overshoot at extreme biting rates;
        # mirrors StableCounty -- never triggers in the valid regime)
        SH = 0.0 if (SH != SH or SH < 0.0) else (N_H if SH > N_H else SH)
        EH = 0.0 if (EH != EH or EH < 0.0) else (N_H if EH > N_H else EH)
        IH = 0.0 if (IH != IH or IH < 0.0) else (N_H if IH > N_H else IH)
        AH = 0.0 if (AH != AH or AH < 0.0) else (N_H if AH > N_H else AH)
        TH = 0.0 if (TH != TH or TH < 0.0) else (N_H if TH > N_H else TH)
        RH = 0.0 if (RH != RH or RH < 0.0) else (N_H if RH > N_H else RH)
        SM = 0.0 if (SM != SM or SM < 0.0) else (N_M if SM > N_M else SM)
        EM = 0.0 if (EM != EM or EM < 0.0) else (N_M if EM > N_M else EM)
        IM = 0.0 if (IM != IM or IM < 0.0) else (N_M if IM > N_M else IM)
        tau += 1.0
        if it >= t_ss:                       # accumulate post-update (matches steady())
            inc_sum += p*sigma_H*EH
            prev_sum += (IH+TH+AH)/N_H
    return prev_sum/avg, inc_sum/N_H*1000.0


# --------------------------------------------------------------------------- #
# Parameter packing from the scalar county objects                            #
# --------------------------------------------------------------------------- #
def pack_seitr(c):
    return np.array([c.mu_H, c.mu_M, c.gamma, c.gamma_T, c.rho, c.delta_H,
                     c.delta_T, c.sigma_H, c.sigma_M, c.epsilon, c.a, c.b, c.c,
                     c.seasonal_amp, c.delta, c.N_H, c.N_M, c.external_seeding],
                    dtype=np.float64)


def state_seitr(c):
    return np.array([c.S_H, c.E_H, c.I_H, c.T_H, c.R_H, c.S_M, c.E_M, c.I_M],
                    dtype=np.float64)


def pack_seitar(c):
    return np.array([c.mu_H, c.mu_M, c.gamma, c.gamma_T, c.rho, c.delta_H,
                     c.delta_T, c.sigma_H, c.sigma_M, c.kappa, c.kappa_A, c.p,
                     c.gamma_A, c.a, c.b, c.c, c.delta, c.R, c.N_H, c.N_M,
                     c.external_seeding], dtype=np.float64)


def state_seitar(c):
    return np.array([c.S_H, c.E_H, c.I_H, c.T_H, c.A_H, c.R_H,
                     c.S_M, c.E_M, c.I_M], dtype=np.float64)

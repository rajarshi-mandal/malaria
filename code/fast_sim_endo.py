"""
Endogenous-resistance SEITAR core (numba-JIT), for the harder, more realistic
allocation environment that breaks the near-myopia of the base benchmark.

WHAT IS NEW vs fast_sim.py
--------------------------
In the base benchmark, the insecticide-resistance impact R is EXOGENOUS (a fixed
per-episode constant in SEITAR). That makes an allocation's cost purely
instantaneous, so a one-step greedy rule already matches the open-loop oracle
(the paper's near-myopia result).

Here R_i is an evolving STATE driven by LOCAL insecticide selection pressure --
the well-documented phenomenon that deploying pyrethroid ITNs accelerates local
pyrethroid resistance (Barbosa 2018; WHO GPIRM). Selection pressure is the
fraction of bites meeting an *active* (fresh, undecayed) net, i.e. the effective
coverage eff = C * exp(-delta * tau):

    R_i(t+1) = R_i(t) + k_sel * eff_i(t) * (1 - R_i(t)) - k_rev * R_i(t)

  * k_sel grounded in IR-Mapper (endo_calibrate_resistance.py): ~9.6e-4/day
  * k_rev a slow fitness-cost reversion when a region is rested (default ~10%/yr)
  * logistic (1-R) -> saturates at fixation R=1; reversion -> resting restores
    susceptibility. This couples campaigns: over-deploying in a region now raises
    its resistance and degrades ALL its future campaigns -> a genuine
    exploit-vs-conserve tradeoff that myopic allocation gets wrong.

BACKWARD COMPATIBILITY: with k_sel=k_rev=0 and R held at the SEITAR constant, the
per-day update is bit-identical to fast_sim._seitar_step (verified in
verify_endo.py). So the extension is a strict generalization.

Parameter vector Pe (per region, 20 elts; R removed since it is now state):
 0 mu_H 1 mu_M 2 gamma 3 gamma_T 4 rho 5 delta_H 6 delta_T 7 sigma_H 8 sigma_M
 9 kappa 10 kappa_A 11 p 12 gamma_A 13 a 14 b 15 c 16 delta 17 N_H 18 N_M 19 seeding
State s (9): [S_H,E_H,I_H,T_H,A_H,R_H, S_M,E_M,I_M]
"""
import numpy as np
from numba import njit

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# Single-region one-day step (R endogenous)                                   #
# --------------------------------------------------------------------------- #
@njit(cache=True, fastmath=False)
def _endo_step(s, C, tau, R, t, alloc, P, k_sel, k_rev):
    mu_H = P[0]; mu_M = P[1]; gamma = P[2]; gamma_T = P[3]; rho = P[4]
    delta_H = P[5]; delta_T = P[6]; sigma_H = P[7]; sigma_M = P[8]; kappa = P[9]
    kappa_A = P[10]; p = P[11]; gamma_A = P[12]; a = P[13]; b = P[14]; c = P[15]
    delta = P[16]; N_H = P[17]; N_M = P[18]; seeding = P[19]
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
    eff = C * np.exp(-delta * tau)            # effective (fresh) net coverage
    theta = eff * (1.0 - 0.5 * R)
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
    # endogenous resistance update (selection by effective coverage, slow reversion)
    R = R + k_sel * eff * (1.0 - R) - k_rev * R
    if R < 0.0:
        R = 0.0
    if R > 1.0:
        R = 1.0
    return C, tau + 1.0, R


# --------------------------------------------------------------------------- #
# Vectorized helpers used by the Python env (all regions together)            #
# --------------------------------------------------------------------------- #
@njit(cache=True, fastmath=False)
def endo_step_vec(S, C, tau, R, t, alloc, P, k_sel, k_rev):
    """One day for all N regions; mutates S,C,tau,R in place. Returns sum E+I."""
    N = S.shape[0]
    burden = 0.0
    for i in range(N):
        Ci, taui, Ri = _endo_step(S[i], C[i], tau[i], R[i], t, alloc[i],
                                   P[i], k_sel, k_rev)
        C[i] = Ci; tau[i] = taui; R[i] = Ri
        burden += S[i][1] + S[i][2]
    return burden


@njit(cache=True, fastmath=False)
def endo_interval_vec(S, C, tau, R, t0, alloc, P, k_sel, k_rev, ndays):
    """Deploy `alloc` at day t0, then run ndays; mutate in place; return sum E+I."""
    N = S.shape[0]
    burden = 0.0
    for d in range(ndays):
        t = t0 + d
        for i in range(N):
            ai = alloc[i] if d == 0 else 0.0
            Ci, taui, Ri = _endo_step(S[i], C[i], tau[i], R[i], t, ai,
                                      P[i], k_sel, k_rev)
            C[i] = Ci; tau[i] = taui; R[i] = Ri
            burden += S[i][1] + S[i][2]
    return burden


# --------------------------------------------------------------------------- #
# Single-region lookahead (for greedy / MPC / the resistance-aware planner)    #
# --------------------------------------------------------------------------- #
@njit(cache=True, fastmath=False)
def endo_lookahead(s0, C0, tau0, R0, t0, new_itns, P, k_sel, k_rev, horizon):
    """Cumulative (E+I) over `horizon` days if `new_itns` deployed now (one region)."""
    s = s0.copy()
    C = C0; tau = tau0; R = R0
    burden = 0.0
    for d in range(horizon):
        alloc = new_itns if d == 0 else 0.0
        C, tau, R = _endo_step(s, C, tau, R, t0 + d, alloc, P, k_sel, k_rev)
        burden += s[1] + s[2]
    return burden


@njit(cache=True, fastmath=False)
def endo_lookahead_g(s0, C0, tau0, R0, t0, new_itns, P, k_sel, k_rev, k_cov, horizon):
    """Single-region (DECOUPLED) lookahead with attrition but NO mobility -- the
    naive planner that a decoupled greedy uses; it cannot see network spillovers."""
    s = s0.copy()
    C = C0; tau = tau0; R = R0
    burden = 0.0
    for d in range(horizon):
        alloc = new_itns if d == 0 else 0.0
        hinf = (s[2] + P[9] * s[3] + P[10] * s[4]) / P[17]
        C, tau, R = _endo_step_g(s, C, tau, R, t0 + d, alloc, P, k_sel, k_rev, k_cov, hinf)
        burden += s[1] + s[2]
    return burden


@njit(cache=True, fastmath=False)
def endo_multistage_region(s0, C0, tau0, R0, t0, allocs, gaps, P, k_sel, k_rev):
    """One region, MULTI-campaign open-loop lookahead. `allocs[k]` deployed at the
    start of stage k; stage k lasts gaps[k] days. Returns total (E+I) burden over
    all stages. This is what a resistance-aware planner uses to see the future
    efficacy cost of deploying now (greedy uses only horizon=gaps[0])."""
    s = s0.copy()
    C = C0; tau = tau0; R = R0
    t = t0
    burden = 0.0
    K = allocs.shape[0]
    for k in range(K):
        nd = gaps[k]
        for d in range(nd):
            alloc = allocs[k] if d == 0 else 0.0
            C, tau, R = _endo_step(s, C, tau, R, t, alloc, P, k_sel, k_rev)
            burden += s[1] + s[2]
            t += 1
    return burden


# --------------------------------------------------------------------------- #
# Full-episode ratio + batched oracle scorer (open-loop schedule allocs[K,N])  #
# --------------------------------------------------------------------------- #
@njit(cache=True, fastmath=False)
def endo_episode_ratio(states, C0, tau0, R0, P, allocs, camp, ep, k_sel, k_rev, pre_end):
    """Cumulative infection ratio for an open-loop schedule allocs[K,N].
    ratio = (post-first-campaign burden) / (pre-window burden), summed over regions."""
    N = states.shape[0]; K = camp.shape[0]
    pre = 0.0; post = 0.0
    for i in range(N):
        s = states[i].copy(); C = C0[i]; tau = tau0[i]; R = R0[i]
        ci = 0
        for day in range(ep):
            alloc = 0.0
            if ci < K and day == camp[ci]:
                alloc = allocs[ci, i]; ci += 1
            C, tau, R = _endo_step(s, C, tau, R, day, alloc, P[i], k_sel, k_rev)
            bd = s[1] + s[2]
            if day < pre_end:
                pre += bd
            else:
                post += bd
    return post / pre


@njit(cache=True, fastmath=False)
def endo_oracle_batch(states, C0, tau0, R0, P, cand, camp, ep, k_sel, k_rev, pre_end):
    """Score a CEM population of schedules cand[pop,K,N] -> ratios[pop]."""
    pop = cand.shape[0]; N = states.shape[0]; K = camp.shape[0]
    out = np.empty(pop)
    for pp in range(pop):
        pre = 0.0; post = 0.0
        for i in range(N):
            s = states[i].copy(); C = C0[i]; tau = tau0[i]; R = R0[i]
            ci = 0
            for day in range(ep):
                alloc = 0.0
                if ci < K and day == camp[ci]:
                    alloc = cand[pp, ci, i]; ci += 1
                C, tau, R = _endo_step(s, C, tau, R, day, alloc, P[i], k_sel, k_rev)
                bd = s[1] + s[2]
                if day < pre_end:
                    pre += bd
                else:
                    post += bd
        out[pp] = post / pre
    return out


@njit(cache=True, fastmath=False)
def endo_oracle_batch_cvar(states, C0, tau0, R0s, Ps, ksels, krevs, cand,
                           camp, ep, pre_end, cvar_q):
    """Risk-sensitive oracle scorer: score each schedule cand[pop,K,N] across a
    SCENARIO ENSEMBLE (R0s[M,N], Ps[M,N,20], ksels[M], krevs[M]) and return, per
    candidate, the CVaR (mean of worst cvar_q-fraction) of the infection ratio.
    Lower is better, so the worst tail = the LARGEST ratios."""
    pop = cand.shape[0]; N = states.shape[0]; K = camp.shape[0]; M = R0s.shape[0]
    out = np.empty(pop)
    ntail = int(np.ceil(cvar_q * M))
    if ntail < 1:
        ntail = 1
    scen = np.empty(M)
    for pp in range(pop):
        for m in range(M):
            pre = 0.0; post = 0.0
            for i in range(N):
                s = states[i].copy(); C = C0[i]; tau = tau0[i]; R = R0s[m, i]
                ci = 0
                for day in range(ep):
                    alloc = 0.0
                    if ci < K and day == camp[ci]:
                        alloc = cand[pp, ci, i]; ci += 1
                    C, tau, R = _endo_step(s, C, tau, R, day, alloc, Ps[m, i],
                                           ksels[m], krevs[m])
                    bd = s[1] + s[2]
                    if day < pre_end:
                        pre += bd
                    else:
                        post += bd
            scen[m] = post / pre
        # CVaR over the worst (largest-ratio) tail
        order = np.argsort(scen)
        acc = 0.0
        for j in range(ntail):
            acc += scen[order[M - 1 - j]]
        out[pp] = acc / ntail
    return out


# --------------------------------------------------------------------------- #
# Receding-horizon planning scorers (absolute-day offset for seasonality)      #
# Used by ARMOR: at campaign k, re-optimize the REMAINING horizon from the      #
# realized state. day0 = absolute start day so seasonal forcing stays correct.  #
# --------------------------------------------------------------------------- #
@njit(cache=True, fastmath=False)
def endo_plan_batch(states, C0, tau0, R0, P, cand, camp_rel, ep_rem, day0, k_sel, k_rev):
    """Total remaining (E+I) burden for each candidate schedule cand[pop,Kr,N]
    over ep_rem days, campaigns at relative days camp_rel[Kr]. Minimize -> plan."""
    pop = cand.shape[0]; N = states.shape[0]; Kr = camp_rel.shape[0]
    out = np.empty(pop)
    for pp in range(pop):
        tot = 0.0
        for i in range(N):
            s = states[i].copy(); C = C0[i]; tau = tau0[i]; R = R0[i]
            ci = 0
            for d in range(ep_rem):
                alloc = 0.0
                if ci < Kr and d == camp_rel[ci]:
                    alloc = cand[pp, ci, i]; ci += 1
                C, tau, R = _endo_step(s, C, tau, R, day0 + d, alloc, P[i], k_sel, k_rev)
                tot += s[1] + s[2]
        out[pp] = tot
    return out


@njit(cache=True, fastmath=False)
def endo_plan_batch_cvar(states, C0, tau0, R0s, Ps, ksels, krevs, cand,
                         camp_rel, ep_rem, day0, cvar_q):
    """Risk-sensitive planning score: CVaR (worst cvar_q-fraction) of total
    remaining burden across a scenario ensemble (R0s[M,N], Ps[M,N,20],
    ksels[M], krevs[M]). Returns CVaR[pop]; minimize -> robust plan."""
    pop = cand.shape[0]; N = states.shape[0]; Kr = camp_rel.shape[0]; M = R0s.shape[0]
    out = np.empty(pop)
    ntail = int(np.ceil(cvar_q * M))
    if ntail < 1:
        ntail = 1
    scen = np.empty(M)
    for pp in range(pop):
        for m in range(M):
            tot = 0.0
            for i in range(N):
                s = states[i].copy(); C = C0[i]; tau = tau0[i]; R = R0s[m, i]
                ci = 0
                for d in range(ep_rem):
                    alloc = 0.0
                    if ci < Kr and d == camp_rel[ci]:
                        alloc = cand[pp, ci, i]; ci += 1
                    C, tau, R = _endo_step(s, C, tau, R, day0 + d, alloc, Ps[m, i],
                                           ksels[m], krevs[m])
                    tot += s[1] + s[2]
            scen[m] = tot
        order = np.argsort(scen)
        acc = 0.0
        for j in range(ntail):
            acc += scen[order[M - 1 - j]]
        out[pp] = acc / ntail
    return out


# =========================================================================== #
# METAPOPULATION coupling + net attrition (the gap-opening extensions)        #
# Mobility: infectious humans travel, so mosquitoes in region i feed on a      #
# MIXED human reservoir (local + network), W row-stochastic, m = mobility.     #
# Attrition: coverage C decays at k_cov/day (nets lost/worn), so allocation is #
# an ONGOING triage and resistance selection becomes spatially differential.   #
# _endo_step_g reduces EXACTLY to _endo_step when k_cov=0 and hinf=local.       #
# =========================================================================== #
@njit(cache=True, fastmath=False)
def _endo_step_g(s, C, tau, R, t, alloc, P, k_sel, k_rev, k_cov, hinf):
    mu_H = P[0]; mu_M = P[1]; gamma = P[2]; gamma_T = P[3]; rho = P[4]
    delta_H = P[5]; delta_T = P[6]; sigma_H = P[7]; sigma_M = P[8]; kappa = P[9]
    kappa_A = P[10]; p = P[11]; gamma_A = P[12]; a = P[13]; b = P[14]; c = P[15]
    delta = P[16]; N_H = P[17]; N_M = P[18]; seeding = P[19]
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
    eff = C * np.exp(-delta * tau)
    theta = eff * (1.0 - 0.5 * R)
    om = 1.0 - theta
    if om < 0.0:
        om = 0.0
    Lam_H = a_t * c * s[8] / N_M * om
    Lam_M = a_t * b * hinf                     # hinf = (mixed) infectious-human frac
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
    R = R + k_sel * eff * (1.0 - R) - k_rev * R
    if R < 0.0:
        R = 0.0
    if R > 1.0:
        R = 1.0
    C = C * (1.0 - k_cov)                        # net attrition
    if C < 0.0:
        C = 0.0
    return C, tau + 1.0, R


@njit(cache=True, fastmath=False)
def _mixed_hinf(S, P, W, m, out):
    """Mixed infectious-human fraction per region: (1-m)*local + m*sum_j W_ij*frac_j."""
    N = S.shape[0]
    loc = np.empty(N)
    for j in range(N):
        loc[j] = (S[j][2] + P[j][9] * S[j][3] + P[j][10] * S[j][4]) / P[j][17]
    for i in range(N):
        if m <= 0.0:
            out[i] = loc[i]
        else:
            acc = 0.0
            for j in range(N):
                acc += W[i, j] * loc[j]
            out[i] = (1.0 - m) * loc[i] + m * acc
    return out


@njit(cache=True, fastmath=False)
def meta_interval_vec(S, C, tau, R, t0, alloc, P, k_sel, k_rev, k_cov, W, m, ndays):
    """Deploy alloc at t0, run ndays with mobility coupling + attrition; mutate
    in place; return sum E+I over the interval."""
    N = S.shape[0]
    burden = 0.0
    hinf = np.empty(N)
    for d in range(ndays):
        t = t0 + d
        _mixed_hinf(S, P, W, m, hinf)
        for i in range(N):
            ai = alloc[i] if d == 0 else 0.0
            Ci, taui, Ri = _endo_step_g(S[i], C[i], tau[i], R[i], t, ai, P[i],
                                        k_sel, k_rev, k_cov, hinf[i])
            C[i] = Ci; tau[i] = taui; R[i] = Ri
            burden += S[i][1] + S[i][2]
    return burden


@njit(cache=True, fastmath=False)
def meta_interval_vec_regional(S, C, tau, R, t0, alloc, P, k_sel, k_rev, k_cov,
                               W, m, ndays, out):
    """Deploy alloc at t0, run ndays with coupling; mutate state; fill out[N] with
    PER-REGION accumulated (E+I) burden (the observable surveillance signal used by
    the adaptive planner to recalibrate). Returns total burden."""
    N = S.shape[0]
    for i in range(N):
        out[i] = 0.0
    hinf = np.empty(N)
    tot = 0.0
    for d in range(ndays):
        t = t0 + d
        _mixed_hinf(S, P, W, m, hinf)
        for i in range(N):
            ai = alloc[i] if d == 0 else 0.0
            Ci, taui, Ri = _endo_step_g(S[i], C[i], tau[i], R[i], t, ai, P[i],
                                        k_sel, k_rev, k_cov, hinf[i])
            C[i] = Ci; tau[i] = taui; R[i] = Ri
            bd = S[i][1] + S[i][2]
            out[i] += bd; tot += bd
    return tot


@njit(cache=True, fastmath=False)
def meta_plan_batch(states, C0, tau0, R0, P, cand, camp_rel, ep_rem, day0,
                    k_sel, k_rev, k_cov, W, m):
    """Coupled full-network rollout: total remaining burden per candidate
    schedule cand[pop,Kr,N]. Steps ALL regions together each day (cannot be
    decoupled once mobility couples them). Minimize -> network-aware plan."""
    pop = cand.shape[0]; N = states.shape[0]; Kr = camp_rel.shape[0]
    out = np.empty(pop)
    hinf = np.empty(N)
    for pp in range(pop):
        S = states.copy(); C = C0.copy(); tau = tau0.copy(); R = R0.copy()
        ci = 0
        tot = 0.0
        for d in range(ep_rem):
            deploy = (ci < Kr and d == camp_rel[ci])
            _mixed_hinf(S, P, W, m, hinf)
            for i in range(N):
                ai = cand[pp, ci, i] if deploy else 0.0
                Ci, taui, Ri = _endo_step_g(S[i], C[i], tau[i], R[i], day0 + d, ai,
                                            P[i], k_sel, k_rev, k_cov, hinf[i])
                C[i] = Ci; tau[i] = taui; R[i] = Ri
                tot += S[i][1] + S[i][2]
            if deploy:
                ci += 1
        out[pp] = tot
    return out


@njit(cache=True, fastmath=False)
def meta_plan_batch_regional(states, C0, tau0, R0, P, cand, camp_rel, ep_rem, day0,
                             k_sel, k_rev, k_cov, W, m):
    """Like meta_plan_batch but returns PER-REGION burden out[pop,N] (for equity /
    worst-region objectives, scalarized in Python)."""
    pop = cand.shape[0]; N = states.shape[0]; Kr = camp_rel.shape[0]
    out = np.zeros((pop, N))
    hinf = np.empty(N)
    for pp in range(pop):
        S = states.copy(); C = C0.copy(); tau = tau0.copy(); R = R0.copy()
        ci = 0
        for d in range(ep_rem):
            deploy = (ci < Kr and d == camp_rel[ci])
            _mixed_hinf(S, P, W, m, hinf)
            for i in range(N):
                ai = cand[pp, ci, i] if deploy else 0.0
                Ci, taui, Ri = _endo_step_g(S[i], C[i], tau[i], R[i], day0 + d, ai,
                                            P[i], k_sel, k_rev, k_cov, hinf[i])
                C[i] = Ci; tau[i] = taui; R[i] = Ri
                out[pp, i] += S[i][1] + S[i][2]
            if deploy:
                ci += 1
    return out


@njit(cache=True, fastmath=False)
def meta_plan_batch_cvar(states, C0, tau0, R0s, Ps, ksels, krevs, kcovs, cand,
                         camp_rel, ep_rem, day0, W, m, cvar_q):
    """Risk-sensitive coupled planner score: CVaR of total burden across a
    scenario ensemble (R0s[M,N], Ps[M,N,20], ksels[M], krevs[M], kcovs[M])."""
    pop = cand.shape[0]; N = states.shape[0]; Kr = camp_rel.shape[0]; M = R0s.shape[0]
    out = np.empty(pop)
    ntail = int(np.ceil(cvar_q * M))
    if ntail < 1:
        ntail = 1
    scen = np.empty(M)
    hinf = np.empty(N)
    for pp in range(pop):
        for mm in range(M):
            S = states.copy(); C = C0.copy(); tau = tau0.copy(); R = R0s[mm].copy()
            ci = 0
            tot = 0.0
            for d in range(ep_rem):
                deploy = (ci < Kr and d == camp_rel[ci])
                _mixed_hinf(S, Ps[mm], W, m, hinf)
                for i in range(N):
                    ai = cand[pp, ci, i] if deploy else 0.0
                    Ci, taui, Ri = _endo_step_g(S[i], C[i], tau[i], R[i], day0 + d, ai,
                                                Ps[mm, i], ksels[mm], krevs[mm],
                                                kcovs[mm], hinf[i])
                    C[i] = Ci; tau[i] = taui; R[i] = Ri
                    tot += S[i][1] + S[i][2]
                if deploy:
                    ci += 1
            scen[mm] = tot
        order = np.argsort(scen)
        acc = 0.0
        for j in range(ntail):
            acc += scen[order[M - 1 - j]]
        out[pp] = acc / ntail
    return out


# --------------------------------------------------------------------------- #
# Packing from the scalar ExtendedCounty (R dropped -> state)                  #
# --------------------------------------------------------------------------- #
def pack_endo(c):
    return np.array([c.mu_H, c.mu_M, c.gamma, c.gamma_T, c.rho, c.delta_H,
                     c.delta_T, c.sigma_H, c.sigma_M, c.kappa, c.kappa_A, c.p,
                     c.gamma_A, c.a, c.b, c.c, c.delta, c.N_H, c.N_M,
                     c.external_seeding], dtype=np.float64)


def state_endo(c):
    return np.array([c.S_H, c.E_H, c.I_H, c.T_H, c.A_H, c.R_H,
                     c.S_M, c.E_M, c.I_M], dtype=np.float64)

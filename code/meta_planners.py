"""
Metapopulation ITN-allocation environment + planner ladder (the gap-opening env).

Two realistic couplings are added to the calibrated SEITAR benchmark, each of
which BREAKS a simplification the near-myopia result depended on:
  * MOBILITY (spatial): infectious humans travel, so mosquitoes in a region feed
    on a network-mixed human reservoir (gravity matrix W, mobility m). Protecting
    a high-connectivity region now suppresses transmission in the regions coupled
    to it -- a spillover a DECOUPLED greedy (per-region lookahead) cannot see.
  * ATTRITION + ENDOGENOUS RESISTANCE (temporal): coverage decays (nets lost),
    so allocation is an ongoing triage, and deploying selects local resistance
    (fast_sim_endo) -- a delayed cost.

Ladder:
  static    population-proportional (WHO standard of practice)
  dgreedy   DECOUPLED greedy -- per-region marginal benefit, ignores coupling
            (this is the base paper's near-optimal myopic rule)
  ngreedy   network-aware greedy -- marginal benefit scored on the FULL network
  ARMOR     NOVEL: network- & resistance-aware receding-horizon planner; at each
            campaign re-optimizes the remaining schedule on coupled rollouts
  oracle    open-loop network optimum from t=0 (upper bound)

If dgreedy << oracle (unlike the base benchmark where greedy==oracle), the
couplings have made the problem genuinely non-myopic, and a coordinated planner
is required. Run: python meta_planners.py --n-eval 24
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats

import fast_sim_endo as fe
from extended_ode import ExtendedCounty

REGIONS = pd.read_csv("extended_calibrated_regions.csv")
N = len(REGIONS)
BUDGET = 12000.0                # scarcer than base (0.15x pop) so triage matters
FIRST = 30
K_SEL = 9.6e-4
K_REV = 1.0e-3
K_COV = 7.6e-4                  # net attrition ~ halve in 2.5 yr (no replenishment)
MOBILITY = 0.35                # fraction of transmission that is network-mixed
R0_MEAN, R0_SD = 0.40, 0.08
_mult = REGIONS.pop_multiplier.to_numpy()
POP_STATIC = (_mult / _mult.sum() * BUDGET).astype(np.float64)
TOTAL_POP = float((10000 * _mult).sum())

# gravity mobility matrix: travel attractiveness ~ destination population.
_pop = (10000 * _mult).astype(np.float64)
_W = np.tile(_pop, (N, 1))
np.fill_diagonal(_W, 0.0)
_W = _W / _W.sum(1, keepdims=True)      # row-stochastic (destination shares)

_STEADY = None


def get_steady():
    global _STEADY
    if _STEADY is None:
        _STEADY = []
        for _, r in REGIONS.iterrows():
            c = ExtendedCounty(r.biting_rate, r.ITN_coverage, r.pop_multiplier,
                               int(r.treatment_seeking), r.p_sympt)
            for _ in range(2500):
                c.step(0.0, seasonal=False)
            _STEADY.append(fe.state_endo(c))
    return _STEADY


class MetaEnv:
    def __init__(self, rng, k=6, gap=365, k_sel=K_SEL, k_rev=K_REV, k_cov=K_COV,
                 mobility=MOBILITY, r0_mean=R0_MEAN, r0_sd=R0_SD,
                 budget=BUDGET, seeding_scale=1.0):
        self.rng = rng
        self.K = k; self.gap = gap
        self.k_sel = float(k_sel); self.k_rev = float(k_rev); self.k_cov = float(k_cov)
        self.m = float(mobility)
        self.W = _W
        self.budget = float(budget)
        self.pop_static = (_mult / _mult.sum() * self.budget).astype(np.float64)
        self.ep = FIRST + k * gap
        self.camp = np.array([FIRST + j * gap for j in range(k)], dtype=np.int64)
        self.P = np.array([self._pack(r) for _, r in REGIONS.iterrows()], dtype=np.float64)
        self.P[:, 19] *= float(seeding_scale)          # tune reseeding (elimination regime)
        self.R0 = np.clip(rng.normal(r0_mean, r0_sd, N), 0.10, 0.80)
        self.reset()

    @staticmethod
    def _pack(r):
        c = ExtendedCounty(r.biting_rate, r.ITN_coverage, r.pop_multiplier,
                           int(r.treatment_seeking), r.p_sympt)
        return fe.pack_endo(c)

    def reset(self):
        self.S = np.array([s.copy() for s in get_steady()], dtype=np.float64)
        self.C = REGIONS.ITN_coverage.to_numpy(float).copy()
        self.tau = np.zeros(N)
        self.R = self.R0.copy()
        self.t = 0
        return self._obs()

    def _obs(self):
        eff = self.C * np.exp(-self.P[:, 16] * self.tau)
        a_t = self.P[:, 13] * (1.0 + 0.2 * np.sin(2 * np.pi * self.t / 180.0))
        feats = []
        for i in range(N):
            feats.extend([self.P[i, 17] / 1e4, self.P[i, 13], self.C[i],
                          1.0 / self.P[i, 4], *self.S[i], a_t[i], self.R[i], eff[i]])
        camp_left = (self.camp >= self.t).sum() / self.K
        return np.array(feats + [camp_left], dtype=np.float32)

    def snapshot(self):
        return (self.S.copy(), self.C.copy(), self.tau.copy(), self.R.copy(), self.t)

    def run_interval(self, alloc, ndays):
        b = fe.meta_interval_vec(self.S, self.C, self.tau, self.R, self.t,
                                 np.asarray(alloc, float), self.P, self.k_sel,
                                 self.k_rev, self.k_cov, self.W, self.m, ndays)
        self.t += ndays
        return b


def run_episode(env, policy, collect=False, center=1.0, scale=5.0):
    env.reset()
    pre = env.run_interval(np.zeros(N), FIRST)
    obs = env._obs()
    post = 0.0
    trans = []
    last = obs
    for k in range(env.K):
        a = policy(last, k)
        b = env.run_interval(np.asarray(a, float), env.gap)
        post += b
        nxt = env._obs()
        if collect:
            trans.append((last, np.asarray(a, np.float32), scale * (center - b / pre),
                          nxt, k == env.K - 1))
        last = nxt
    return post / pre, trans


# --------------------------------------------------------------------------- #
# Planners                                                                     #
# --------------------------------------------------------------------------- #
def static_pol(env):
    return lambda obs, k: env.pop_static.copy()


def dgreedy_pol(env, horizon=120, chunk_frac=0.05):
    """DECOUPLED greedy: per-region marginal benefit, ignores mobility."""
    def pol(obs, k):
        S, C, tau, R, t = env.snapshot()
        alloc = np.zeros(N)
        cur = np.array([fe.endo_lookahead_g(S[i], C[i], tau[i], R[i], t, 0.0,
                        env.P[i], env.k_sel, env.k_rev, env.k_cov, horizon)
                        for i in range(N)])
        chunk = env.budget * chunk_frac
        rem = env.budget
        while rem > 1e-6:
            cc = min(chunk, rem)
            bi, bg, bn = -1, -np.inf, 0.0
            for i in range(N):
                bnew = fe.endo_lookahead_g(S[i], C[i], tau[i], R[i], t, alloc[i] + cc,
                                           env.P[i], env.k_sel, env.k_rev, env.k_cov, horizon)
                g = cur[i] - bnew
                if g > bg:
                    bg, bi, bn = g, i, bnew
            alloc[bi] += cc; cur[bi] = bn; rem -= cc
        return alloc
    return pol


def _meta_eval(env, S, C, tau, R, t, alloc, horizon):
    """Total network burden over `horizon` if `alloc` deployed now (coupled)."""
    cand = alloc.reshape(1, 1, N).astype(np.float64)
    return fe.meta_plan_batch(np.asarray(S, np.float64), C, tau, R, env.P, cand,
                              np.array([0], np.int64), horizon, t,
                              env.k_sel, env.k_rev, env.k_cov, env.W, env.m)[0]


def ngreedy_pol(env, horizon=120, chunk_frac=0.05):
    """NETWORK-aware greedy: marginal benefit scored on the full coupled network."""
    def pol(obs, k):
        S, C, tau, R, t = env.snapshot()
        alloc = np.zeros(N)
        cur = _meta_eval(env, S, C, tau, R, t, alloc, horizon)
        chunk = env.budget * chunk_frac
        rem = env.budget
        while rem > 1e-6:
            cc = min(chunk, rem)
            bi, bg = -1, -np.inf
            for i in range(N):
                trial = alloc.copy(); trial[i] += cc
                val = _meta_eval(env, S, C, tau, R, t, trial, horizon)
                g = cur - val
                if g > bg:
                    bg, bi = g, i
            alloc[bi] += cc
            cur = _meta_eval(env, S, C, tau, R, t, alloc, horizon)
            rem -= cc
        return alloc
    return pol


def cem_plan(env, S, C, tau, R, camp_rel, ep_rem, day0, pop, iters, elite, rng):
    Kr = len(camp_rel)
    alphas = [np.ones(N) for _ in range(Kr)]
    best_sched, best = None, np.inf
    camp_rel = np.asarray(camp_rel, np.int64)
    states = np.asarray(S, np.float64)
    for it in range(iters):
        cand = np.empty((pop, Kr, N))
        for j in range(Kr):
            cand[:, j, :] = rng.dirichlet(alphas[j], size=pop) * env.budget
        sc = fe.meta_plan_batch(states, C, tau, R, env.P, cand, camp_rel, ep_rem,
                                day0, env.k_sel, env.k_rev, env.k_cov, env.W, env.m)
        el = np.argsort(sc)[:elite]
        for j in range(Kr):
            alphas[j] = np.clip((cand[el, j, :] / env.budget).mean(0) * 10 * (it + 1), 1e-2, None)
        if sc[el[0]] < best:
            best = sc[el[0]]; best_sched = cand[el[0]].copy()
    return best_sched


def armor_pol(env, rng, pop=400, iters=12, elite=40):
    """NOVEL deployable: network- & resistance-aware receding-horizon planner."""
    def pol(obs, k):
        S, C, tau, R, t = env.snapshot()
        kr = env.K - k
        camp_rel = [j * env.gap for j in range(kr)]
        sched = cem_plan(env, S, C, tau, R, camp_rel, kr * env.gap, t,
                         pop, iters, elite, rng)
        return sched[0]
    return pol


def oracle_schedule(env, rng, pop=2500, iters=35, elite=150):
    env.reset()
    env.run_interval(np.zeros(N), FIRST)
    S, C, tau, R, t = env.snapshot()
    camp_rel = [j * env.gap for j in range(env.K)]
    return cem_plan(env, S, C, tau, R, camp_rel, env.K * env.gap, t, pop, iters, elite, rng)


def run_oracle(env, sched):
    env.reset()
    pre = env.run_interval(np.zeros(N), FIRST)
    post = 0.0
    for k in range(env.K):
        post += env.run_interval(sched[k], env.gap)
    return post / pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--gap", type=int, default=365)
    ap.add_argument("--mobility", type=float, default=MOBILITY)
    ap.add_argument("--k-cov", type=float, default=K_COV)
    ap.add_argument("--budget", type=float, default=BUDGET)
    ap.add_argument("--seeding-scale", type=float, default=1.0)
    ap.add_argument("--oracle-pop", type=int, default=2500)
    ap.add_argument("--oracle-iters", type=int, default=35)
    ap.add_argument("--base-seed", type=int, default=10000)
    ap.add_argument("--methods", nargs="+",
                    default=["static", "dgreedy", "ngreedy", "armor", "oracle"])
    ap.add_argument("--out", type=str, default="meta_planners.npz")
    args = ap.parse_args()

    def make_env(seed):
        return MetaEnv(np.random.default_rng(seed), k=args.k, gap=args.gap,
                       mobility=args.mobility, k_cov=args.k_cov,
                       budget=args.budget, seeding_scale=args.seeding_scale)

    plan_rng = np.random.default_rng(0)
    res = {m: [] for m in ["baseline"] + args.methods}
    print(f"Meta env | N={N} | K={args.k}x{args.gap}d | budget={args.budget:.0f} | "
          f"mobility={args.mobility} k_cov={args.k_cov:.1e} k_sel={K_SEL:.1e} "
          f"seeding_scale={args.seeding_scale}", flush=True)
    for i in range(args.n_eval):
        seed = args.base_seed + i
        e = make_env(seed); res["baseline"].append(run_episode(e, lambda o, k: np.zeros(N))[0])
        if "static" in args.methods:
            e = make_env(seed); res["static"].append(run_episode(e, static_pol(e))[0])
        if "dgreedy" in args.methods:
            e = make_env(seed); res["dgreedy"].append(run_episode(e, dgreedy_pol(e))[0])
        if "ngreedy" in args.methods:
            e = make_env(seed); res["ngreedy"].append(run_episode(e, ngreedy_pol(e))[0])
        if "armor" in args.methods:
            e = make_env(seed); res["armor"].append(run_episode(e, armor_pol(e, plan_rng))[0])
        if "oracle" in args.methods:
            e = make_env(seed)
            sched = oracle_schedule(e, plan_rng, pop=args.oracle_pop, iters=args.oracle_iters)
            res["oracle"].append(run_oracle(e, sched))
        print(f"  env {i+1}/{args.n_eval} done", flush=True)

    res = {k: np.array(v) for k, v in res.items()}
    b = res["baseline"].mean(); st = res["static"]
    print(f"\n===== METAPOPULATION LADDER ({args.n_eval} held-out envs) =====")
    print(f"{'method':10s} | {'inf.ratio':>10s} | {'%reduction':>10s} | "
          f"{'beats static':>12s} | {'p vs static':>11s}")
    print("-" * 64)
    for m in ["baseline"] + args.methods:
        arr = res[m]; red = (b - arr.mean()) / b * 100
        if m in ("baseline", "static"):
            print(f"{m:10s} | {arr.mean():10.3f} | {red:9.2f}% | {'--':>12s} | {'--':>11s}")
            continue
        beats = (arr < st).mean() * 100
        try:
            _, pv = stats.wilcoxon(arr, st)
        except ValueError:
            pv = float("nan")
        print(f"{m:10s} | {arr.mean():10.3f} | {red:9.2f}% | {beats:10.0f}% | {pv:11.2e}")
    if "dgreedy" in res and "oracle" in res:
        dg = (b - res["dgreedy"].mean()) / b * 100
        orc = (b - res["oracle"].mean()) / b * 100
        print(f"\nDECOUPLED-GREEDY={dg:.2f}%  ORACLE={orc:.2f}%  ->  GAP = {orc-dg:.2f} pp "
              f"(base benchmark gap ~0 pp)")
        if "armor" in res:
            ar = (b - res["armor"].mean()) / b * 100
            print(f"ARMOR={ar:.2f}%  recovers {(ar-dg)/max(orc-dg,1e-9)*100:.0f}% of the gap")
    np.savez(args.out, results=res, mobility=args.mobility, k_cov=args.k_cov,
             k=args.k, gap=args.gap)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

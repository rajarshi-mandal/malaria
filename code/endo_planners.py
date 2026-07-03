"""
Endogenous-resistance allocation environment + planner ladder.

This is the harder, more realistic benchmark where deploying ITNs accelerates
local pyrethroid resistance (fast_sim_endo). The point: in the BASE benchmark a
one-step greedy rule already equals the open-loop oracle (near-myopia). Here the
deployment->resistance feedback couples campaigns, so we expect myopic methods to
fall well short of the oracle, and a horizon-aware resistance-management planner
to recover the gap.

Method ladder (all model-based, decoupled-region lookaheads via numba):
  static    population-proportional (WHO standard of practice)
  greedy    myopic marginal-benefit, horizon = one inter-campaign gap
  mpc1      myopic CEM, horizon = one gap (the base paper's "MPC")
  ARMOR     NOVEL: resistance-aware receding-horizon planner. At each campaign it
            re-optimizes the FULL remaining multi-campaign schedule from the
            realized (observed-resistance) state, so it prices in the future
            efficacy cost of selecting resistance now. Deployable closed-loop MPC.
  oracle    open-loop joint optimum from t=0 (full model access) -- upper bound.

Run: python endo_planners.py --n-eval 24 [--k-sel 9.6e-4 --gap 365 --k 6]
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats

import fast_sim_endo as fe
from extended_ode import ExtendedCounty

# ---- env config ----------------------------------------------------------- #
REGIONS = pd.read_csv("extended_calibrated_regions.csv")
N = len(REGIONS)
BUDGET = 20000.0
FIRST = 30                      # first campaign day (pre-window = days 0..29)
K_SEL = 9.6e-4                  # IR-Mapper-grounded selection rate (per day)
K_REV = 2.74e-4                # fitness-cost reversion (~10%/yr at C=0)
R0_MEAN, R0_SD = 0.40, 0.08    # per-env initial resistance draw
_mult = REGIONS.pop_multiplier.to_numpy()
POP_STATIC = (_mult / _mult.sum() * BUDGET).astype(np.float64)
TOTAL_POP = float((10000 * _mult).sum())

_STEADY = None


def get_steady():
    """Per-region endemic steady state at the SEITAR baseline R (cached once)."""
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


class EndoEnv:
    """Vectorized endogenous-resistance SEITAR env (numba-backed).

    K campaigns spaced `gap` days apart, first at day FIRST. Resistance R_i is a
    per-region STATE evolving under local effective-coverage selection (k_sel)
    and slow reversion (k_rev). Per env, R0_i and (optionally) k_sel/k_rev are
    drawn so held-out evaluation spans calibration/resistance uncertainty."""

    def __init__(self, rng, k=6, gap=365, k_sel=K_SEL, k_rev=K_REV,
                 r0_mean=R0_MEAN, r0_sd=R0_SD):
        self.rng = rng
        self.K = k
        self.gap = gap
        self.k_sel = float(k_sel)
        self.k_rev = float(k_rev)
        self.ep = FIRST + k * gap
        self.camp = np.array([FIRST + j * gap for j in range(k)], dtype=np.int64)
        self.P = np.array([self._pack(r) for _, r in REGIONS.iterrows()], dtype=np.float64)
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
        # per region: [mult, a, C, 1/rho, S,E,I,T,A,R_H, S_M,E_M,I_M, a_t, R_res, eff]
        eff = self.C * np.exp(-self.P[:, 16] * self.tau)
        a_t = self.P[:, 13] * (1.0 + 0.2 * np.sin(2 * np.pi * self.t / 180.0))
        feats = []
        for i in range(N):
            feats.extend([self.P[i, 17] / 1e4, self.P[i, 13], self.C[i],
                          1.0 / self.P[i, 4], *self.S[i], a_t[i], self.R[i], eff[i]])
        # also append remaining-campaign fraction (finite-horizon signal)
        camp_left = (self.camp >= self.t).sum() / self.K
        return np.array(feats + [camp_left], dtype=np.float32)

    def snapshot(self):
        return (self.S.copy(), self.C.copy(), self.tau.copy(), self.R.copy(), self.t)

    def run_interval(self, alloc, ndays):
        """Deploy alloc at the current day, advance ndays, return E+I burden."""
        b = fe.endo_interval_vec(self.S, self.C, self.tau, self.R, self.t,
                                 np.asarray(alloc, float), self.P,
                                 self.k_sel, self.k_rev, ndays)
        self.t += ndays
        return b


def run_episode(env, policy, collect=False, center=1.0, scale=5.0):
    """Closed-loop episode. Returns (cumulative infection ratio, transitions)."""
    env.reset()
    pre = env.run_interval(np.zeros(N), FIRST)          # pre-window burden
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
            r = scale * (center - b / pre)
            trans.append((last, np.asarray(a, np.float32), r, nxt, k == env.K - 1))
        last = nxt
    return post / pre, trans


# --------------------------------------------------------------------------- #
# Planners                                                                     #
# --------------------------------------------------------------------------- #
def static_pol(env):
    return lambda obs, k: POP_STATIC.copy()


def greedy_pol(env, horizon=60, chunk_frac=0.05):
    def pol(obs, k):
        S, C, tau, R, t = env.snapshot()
        alloc = np.zeros(N)
        cur = np.array([fe.endo_lookahead(S[i], C[i], tau[i], R[i], t, 0.0,
                                          env.P[i], env.k_sel, env.k_rev, horizon)
                        for i in range(N)])
        chunk = BUDGET * chunk_frac
        remaining = BUDGET
        while remaining > 1e-6:
            cc = min(chunk, remaining)
            best_i, best_gain, best_new = -1, -np.inf, 0.0
            for i in range(N):
                bnew = fe.endo_lookahead(S[i], C[i], tau[i], R[i], t, alloc[i] + cc,
                                         env.P[i], env.k_sel, env.k_rev, horizon)
                gain = cur[i] - bnew
                if gain > best_gain:
                    best_gain, best_i, best_new = gain, i, bnew
            alloc[best_i] += cc
            cur[best_i] = best_new
            remaining -= cc
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
            cand[:, j, :] = rng.dirichlet(alphas[j], size=pop) * BUDGET
        sc = fe.endo_plan_batch(states, C, tau, R, env.P, cand, camp_rel,
                                ep_rem, day0, env.k_sel, env.k_rev)
        el = np.argsort(sc)[:elite]
        for j in range(Kr):
            alphas[j] = np.clip((cand[el, j, :] / BUDGET).mean(0) * 10 * (it + 1), 1e-2, None)
        if sc[el[0]] < best:
            best = sc[el[0]]; best_sched = cand[el[0]].copy()
    return best_sched


def mpc1_pol(env, rng, horizon=60, pop=128, iters=5, elite=16):
    """Myopic CEM: optimize THIS campaign for a short horizon only."""
    def pol(obs, k):
        S, C, tau, R, t = env.snapshot()
        sched = cem_plan(env, S, C, tau, R, [0], horizon, t, pop, iters, elite, rng)
        return sched[0]
    return pol


def armor_pol(env, rng, pop=400, iters=12, elite=40):
    """NOVEL: resistance-aware receding-horizon. Re-optimize ALL remaining
    campaigns from the realized state; deploy the first. Prices in future
    resistance because the rollout carries R forward over the whole horizon."""
    def pol(obs, k):
        S, C, tau, R, t = env.snapshot()
        kr = env.K - k                              # remaining campaigns
        camp_rel = [j * env.gap for j in range(kr)]
        ep_rem = kr * env.gap
        sched = cem_plan(env, S, C, tau, R, camp_rel, ep_rem, t, pop, iters, elite, rng)
        return sched[0]
    return pol


def oracle_schedule(env, rng, pop=2500, iters=35, elite=150):
    """Open-loop joint optimum over all K campaigns from t=0 (upper bound)."""
    env.reset()
    env.run_interval(np.zeros(N), FIRST)            # advance to first campaign
    S, C, tau, R, t = env.snapshot()
    camp_rel = [j * env.gap for j in range(env.K)]
    ep_rem = env.K * env.gap
    sched = cem_plan(env, S, C, tau, R, camp_rel, ep_rem, t, pop, iters, elite, rng)
    return sched


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
    ap.add_argument("--k-sel", type=float, default=K_SEL)
    ap.add_argument("--k-rev", type=float, default=K_REV)
    ap.add_argument("--base-seed", type=int, default=10000)
    ap.add_argument("--methods", nargs="+",
                    default=["static", "greedy", "mpc1", "armor", "oracle"])
    ap.add_argument("--out", type=str, default="endo_planners.npz")
    args = ap.parse_args()

    def make_env(seed):
        return EndoEnv(np.random.default_rng(seed), k=args.k, gap=args.gap,
                       k_sel=args.k_sel, k_rev=args.k_rev)

    plan_rng = np.random.default_rng(0)
    res = {m: [] for m in ["baseline"] + args.methods}
    print(f"Endo env | N={N} | K={args.k} campaigns x gap {args.gap}d | "
          f"ep={FIRST + args.k * args.gap}d | k_sel={args.k_sel:.2e} k_rev={args.k_rev:.2e}",
          flush=True)
    for i in range(args.n_eval):
        seed = args.base_seed + i
        e = make_env(seed)
        res["baseline"].append(run_episode(e, lambda o, k: np.zeros(N))[0])
        if "static" in args.methods:
            e = make_env(seed); res["static"].append(run_episode(e, static_pol(e))[0])
        if "greedy" in args.methods:
            e = make_env(seed); res["greedy"].append(run_episode(e, greedy_pol(e))[0])
        if "mpc1" in args.methods:
            e = make_env(seed); res["mpc1"].append(run_episode(e, mpc1_pol(e, plan_rng))[0])
        if "armor" in args.methods:
            e = make_env(seed); res["armor"].append(run_episode(e, armor_pol(e, plan_rng))[0])
        if "oracle" in args.methods:
            e = make_env(seed)
            sched = oracle_schedule(e, plan_rng)
            res["oracle"].append(run_oracle(e, sched))
        print(f"  env {i+1}/{args.n_eval} done", flush=True)

    res = {k: np.array(v) for k, v in res.items()}
    b = res["baseline"].mean()
    st = res["static"]
    print(f"\n===== ENDOGENOUS-RESISTANCE LADDER ({args.n_eval} held-out envs) =====")
    print(f"{'method':10s} | {'inf.ratio':>10s} | {'%reduction':>10s} | "
          f"{'beats static':>12s} | {'p vs static':>11s}")
    print("-" * 64)
    for m in ["baseline"] + args.methods:
        arr = res[m]
        red = (b - arr.mean()) / b * 100
        if m in ("baseline", "static"):
            print(f"{m:10s} | {arr.mean():10.3f} | {red:9.2f}% | {'--':>12s} | {'--':>11s}")
            continue
        beats = (arr < st).mean() * 100
        try:
            _, pv = stats.wilcoxon(arr, st)
        except ValueError:
            pv = float("nan")
        print(f"{m:10s} | {arr.mean():10.3f} | {red:9.2f}% | {beats:10.0f}% | {pv:11.2e}")

    # the key near-myopia-break diagnostic
    if "greedy" in res and "oracle" in res:
        gr = (b - res["greedy"].mean()) / b * 100
        orc = (b - res["oracle"].mean()) / b * 100
        print(f"\nGREEDY={gr:.2f}%  ORACLE={orc:.2f}%  ->  oracle-minus-greedy GAP = "
              f"{orc - gr:.2f} pp  (base benchmark: ~0 pp)")
    np.savez(args.out, results=res, k=args.k, gap=args.gap,
             k_sel=args.k_sel, k_rev=args.k_rev)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

"""
NOVELTY (robustness axis): a deployer does not know the true transmission
calibration, the future insecticide-resistance selection rate, or net attrition
-- the paper itself quantifies this calibration uncertainty. A schedule optimized
for the MEAN (nominal) parameters can fail badly on adverse draws (high
transmission, fast resistance). We formulate ITN allocation as a
distributionally-robust optimization and optimize the CVaR (mean of the worst
tail) of the infection ratio over an uncertainty ensemble.

We compare two open-loop planners on HELD-OUT scenario draws:
  mean    minimize the mean infection ratio over the ensemble (risk-neutral)
  robust  minimize CVaR_{20%} (the worst-20%-tail) -- our risk-sensitive method

and report mean reduction AND worst-case (CVaR) reduction. A robust planner that
gives up little mean reduction to substantially improve the worst case is the
contribution -- mean-optimal methods (greedy/PPO/oracle) do not provide this.

Run: python meta_robust.py --n-worlds 12 --m-train 24 --m-test 40
"""
import argparse
import numpy as np

import fast_sim_endo as fe
import meta_planners as mp


def build_scenarios(env, M, rng, spatial_only=False):
    """Sample an uncertainty ensemble over calibration (a,p), initial resistance,
    and the resistance/attrition rates a real deployer cannot know exactly.

    spatial_only=True isolates the ALLOCATION-relevant uncertainty: it shuffles
    WHICH regions are highest-transmission while holding the mean biting rate fixed
    (we are far more uncertain about the spatial pattern of transmission than its
    aggregate level). This is the uncertainty a robust ALLOCATION should hedge."""
    N = mp.N
    baseP = env.P.copy()
    base_mean_a = baseP[:, 13].mean()
    R0s = np.empty((M, N)); Ps = np.empty((M, N, 20))
    ksels = np.empty(M); krevs = np.empty(M); kcovs = np.empty(M)
    for mmi in range(M):
        P = baseP.copy()
        # per-region biting-rate uncertainty at the CV~35% the paper's calibration-
        # uncertainty analysis (WS7) measured -> the spatial ranking of the
        # worst-transmission region genuinely varies across scenarios.
        mult = rng.lognormal(0.0, 0.35, N)
        P[:, 13] *= mult
        if spatial_only:                              # renormalize: fix mean biting
            P[:, 13] *= base_mean_a / P[:, 13].mean()
        P[:, 11] = np.clip(P[:, 11] * rng.lognormal(0.0, 0.20, N), 0.02, 0.98)  # p
        Ps[mmi] = P
        R0s[mmi] = np.clip(rng.normal(0.45, 0.12, N), 0.10, 0.85)
        if spatial_only:
            ksels[mmi] = 9.6e-4; krevs[mmi] = 1.0e-3; kcovs[mmi] = 7.6e-4
        else:
            ksels[mmi] = np.clip(rng.normal(9.6e-4, 4e-4), 1e-4, 2.5e-3)
            krevs[mmi] = np.clip(rng.normal(1.0e-3, 4e-4), 1e-4, 2.0e-3)
            kcovs[mmi] = np.clip(rng.normal(7.6e-4, 3e-4), 1e-4, 1.5e-3)
    return R0s, Ps, ksels, krevs, kcovs


def cem_robust(env, states, C0, tau0, scen, camp_rel, ep_rem, day0, cvar_q,
               pop=300, iters=10, elite=30, rng=None):
    R0s, Ps, ksels, krevs, kcovs = scen
    Kr = len(camp_rel)
    cr = np.asarray(camp_rel, np.int64)
    alphas = [np.ones(mp.N) for _ in range(Kr)]
    best, best_sched = np.inf, None
    for it in range(iters):
        cand = np.empty((pop, Kr, mp.N))
        for j in range(Kr):
            cand[:, j, :] = rng.dirichlet(alphas[j], size=pop) * env.budget
        sc = fe.meta_plan_batch_cvar(states, C0, tau0, R0s, Ps, ksels, krevs, kcovs,
                                     cand, cr, ep_rem, day0, env.W, env.m, cvar_q)
        el = np.argsort(sc)[:elite]
        for j in range(Kr):
            alphas[j] = np.clip((cand[el, j, :] / env.budget).mean(0) * 10 * (it + 1),
                                1e-2, None)
        if sc[el[0]] < best:
            best = sc[el[0]]; best_sched = cand[el[0]].copy()
    return best_sched


def eval_schedule(env, states, C0, tau0, scen, sched, camp_rel, ep_rem, day0):
    """Per-scenario % reduction vs that scenario's own no-ITN burden."""
    R0s, Ps, ksels, krevs, kcovs = scen
    M = R0s.shape[0]
    cr = np.asarray(camp_rel, np.int64)
    zero = np.zeros((1, len(camp_rel), mp.N))
    one = np.asarray(sched, np.float64).reshape(1, len(camp_rel), mp.N)
    reds = np.empty(M)
    for mmi in range(M):
        b0 = fe.meta_plan_batch(states, C0, tau0, R0s[mmi], Ps[mmi], zero, cr,
                                ep_rem, day0, ksels[mmi], krevs[mmi], kcovs[mmi],
                                env.W, env.m)[0]
        bm = fe.meta_plan_batch(states, C0, tau0, R0s[mmi], Ps[mmi], one, cr,
                                ep_rem, day0, ksels[mmi], krevs[mmi], kcovs[mmi],
                                env.W, env.m)[0]
        reds[mmi] = (b0 - bm) / b0 * 100
    return reds


def cvar_low(x, q=0.2):
    """Mean of the worst (lowest) q-fraction of reductions (the adverse tail)."""
    x = np.sort(np.asarray(x))
    n = max(1, int(np.ceil(q * len(x))))
    return float(x[:n].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-worlds", type=int, default=12)
    ap.add_argument("--m-train", type=int, default=24)
    ap.add_argument("--m-test", type=int, default=40)
    ap.add_argument("--cvar-q", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--gap", type=int, default=365)
    ap.add_argument("--spatial-only", action="store_true",
                    help="isolate allocation-relevant (spatial) uncertainty")
    ap.add_argument("--out", type=str, default="meta_robust.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rows = {"static": [], "mean": [], "robust": []}
    tails = {"static": [], "mean": [], "robust": []}
    print(f"Robust allocation | budget={mp.BUDGET:.0f} mobility={mp.MOBILITY} | "
          f"{args.n_worlds} worlds, train M={args.m_train}, test M={args.m_test}, "
          f"CVaR q={args.cvar_q}", flush=True)

    for wld in range(args.n_worlds):
        env = mp.MetaEnv(np.random.default_rng(20000 + wld), k=args.k, gap=args.gap)
        env.reset(); env.run_interval(np.zeros(mp.N), mp.FIRST)
        S, C, tau, R, t = env.snapshot()
        camp_rel = [j * env.gap for j in range(env.K)]
        ep_rem = env.K * env.gap
        states = np.asarray(S, np.float64)

        scen_tr = build_scenarios(env, args.m_train, rng, args.spatial_only)
        scen_te = build_scenarios(env, args.m_test, rng, args.spatial_only)

        sched_mean = cem_robust(env, states, C, tau, scen_tr, camp_rel, ep_rem, t,
                                cvar_q=1.0, rng=rng)
        sched_rob = cem_robust(env, states, C, tau, scen_tr, camp_rel, ep_rem, t,
                               cvar_q=args.cvar_q, rng=rng)
        sched_static = np.tile(env.pop_static, (env.K, 1))

        for name, sched in (("static", sched_static), ("mean", sched_mean),
                            ("robust", sched_rob)):
            reds = eval_schedule(env, states, C, tau, scen_te, sched, camp_rel, ep_rem, t)
            rows[name].append(reds.mean())
            tails[name].append(cvar_low(reds, args.cvar_q))
        print(f"  world {wld+1}/{args.n_worlds} done", flush=True)

    print("\n===== DISTRIBUTIONALLY-ROBUST ALLOCATION (held-out scenarios) =====")
    print(f"{'method':10s} | {'mean red%':>10s} | {'worst-case (CVaR) red%':>22s}")
    print("-" * 48)
    for name in ("static", "mean", "robust"):
        print(f"{name:10s} | {np.mean(rows[name]):10.2f} | {np.mean(tails[name]):22.2f}")
    dmean = np.mean(rows["mean"]) - np.mean(rows["robust"])
    dtail = np.mean(tails["robust"]) - np.mean(tails["mean"])
    print(f"\nrobust vs mean-optimal: worst-case reduction +{dtail:.2f} pp "
          f"for only {dmean:.2f} pp less mean reduction")
    np.savez(args.out, rows=rows, tails=tails, cvar_q=args.cvar_q)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

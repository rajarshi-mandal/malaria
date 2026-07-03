"""
HEADLINE NOVELTY: surveillance-driven ADAPTIVE ITN allocation (ARMOR-Adapt).

The expected-burden allocation problem is near-myopic (greedy approx oracle)
GIVEN the transmission model. But the oracle's ~6.6 pp advantage over the WHO
standard of practice (static) comes entirely from KNOWING each region's
transmission intensity. A program entering a new setting does NOT know the
per-region transmission a priori; it learns it from surveillance over successive
campaigns. We therefore pose allocation as closed-loop control + online
calibration: start from an UNINFORMATIVE prior (assume uniform transmission),
recalibrate per-region biting rates from observed incidence after each campaign,
and re-plan the remaining horizon.

On identical true environments (deployer never sees true params):
  static     population-proportional (WHO practice) -- no model
  open       optimize once under the uninformative prior, then commit
  mpc        re-plan each campaign from observed state, prior model FIXED
             (state feedback only, no learning)
  adapt      re-plan + recalibrate from surveillance each campaign (NOVEL)
  oracle     re-plan with the TRUE model -- upper bound

The mean reduction ARMOR-Adapt recovers over open-loop/static -- approaching the
oracle purely by learning -- is the contribution: the algorithmic leverage is
surveillance-driven adaptation, not one-shot optimization.

Run: python meta_adaptive.py --n-worlds 12
"""
import argparse
import numpy as np

import fast_sim_endo as fe
import meta_planners as mp

POP, ITERS, ELITE = 300, 10, 30


def cem_from(plan_env, S, C, tau, R, t, kr, gap, rng):
    camp_rel = [j * gap for j in range(kr)]
    return mp.cem_plan(plan_env, S, C, tau, R, camp_rel, kr * gap, t, POP, ITERS, ELITE, rng)


def model_predict_region(plan_env, S, C, tau, R, t, alloc, ndays):
    """Per-region burden the planner's CURRENT model expects over the interval."""
    Sc = np.array([s.copy() for s in S]); Cc = C.copy(); tauc = tau.copy(); Rc = R.copy()
    out = np.empty(mp.N)
    fe.meta_interval_vec_regional(Sc, Cc, tauc, Rc, t, np.asarray(alloc, float),
                                  plan_env.P, plan_env.k_sel, plan_env.k_rev,
                                  plan_env.k_cov, plan_env.W, plan_env.m, ndays, out)
    return out


def run_closed_loop(true_env, plan_env, mode, rng):
    """plan_env carries the planner's model (P[:,13] = biting-rate estimate);
    true_env runs the unknown truth. Returns the cumulative infection ratio."""
    true_env.reset()
    pre = true_env.run_interval(np.zeros(mp.N), mp.FIRST)
    if mode == "oracle":
        plan_env.P = true_env.P.copy(); plan_env.k_sel = true_env.k_sel
    post = 0.0
    obs = np.empty(mp.N)
    fixed_sched = None
    for k in range(true_env.K):
        S, C, tau, R, t = true_env.snapshot()           # observed surveillance state
        kr = true_env.K - k
        if mode == "open":
            if k == 0:
                fixed_sched = cem_from(plan_env, S, C, tau, R, t, kr, true_env.gap, rng)
            alloc = fixed_sched[k]
        else:
            alloc = cem_from(plan_env, S, C, tau, R, t, kr, true_env.gap, rng)[0]
        # realized per-region burden on the TRUE environment (surveillance signal)
        tot = fe.meta_interval_vec_regional(true_env.S, true_env.C, true_env.tau,
                                            true_env.R, true_env.t, np.asarray(alloc, float),
                                            true_env.P, true_env.k_sel, true_env.k_rev,
                                            true_env.k_cov, true_env.W, true_env.m,
                                            true_env.gap, obs)
        true_env.t += true_env.gap
        post += tot
        if mode == "adapt":
            # online recalibration: fixed-point fit of per-region biting rates so the
            # model reproduces the OBSERVED per-region burden (under the deployed
            # allocation + coupling). A few iterations converge the estimate.
            for _ in range(6):
                pr = model_predict_region(plan_env, S, C, tau, R, t, alloc, true_env.gap)
                ratio = np.clip(obs / np.maximum(pr, 1e-9), 0.3, 3.0)
                plan_env.P[:, 13] = np.clip(plan_env.P[:, 13] * ratio, 0.05, 20.0)
    return post / pre


def true_biting(seed, rng_p):
    """The unknown true per-region biting rates (calibrated + per-world variation)."""
    e = mp.MetaEnv(np.random.default_rng(seed))
    return e.P[:, 13] * rng_p.lognormal(0.0, 0.25, mp.N)


def cvar_low(x, q=0.2):
    x = np.sort(np.asarray(x)); n = max(1, int(np.ceil(q * len(x))))
    return float(x[:n].mean())


def build(seed, a_true, prior=False):
    e = mp.MetaEnv(np.random.default_rng(seed))
    e.P[:, 13] = a_true.mean() if prior else a_true
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-worlds", type=int, default=12)
    ap.add_argument("--methods", nargs="+",
                    default=["static", "open", "mpc", "adapt", "oracle"])
    ap.add_argument("--out", type=str, default="meta_adaptive.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(0); rng_p = np.random.default_rng(123)
    red = {m: [] for m in ["baseline", "static"] +
           [x for x in args.methods if x != "static"]}
    print(f"Adaptive allocation | N={mp.N} | budget={mp.BUDGET:.0f} | "
          f"{args.n_worlds} worlds", flush=True)
    for wld in range(args.n_worlds):
        seed = 10000 + wld
        a_true = true_biting(seed, rng_p)
        red["baseline"].append(mp.run_episode(build(seed, a_true),
                                              lambda o, k: np.zeros(mp.N))[0])
        red["static"].append(mp.run_episode(build(seed, a_true),
                                            mp.static_pol(build(seed, a_true)))[0])
        for mode in [m for m in args.methods if m != "static"]:
            te = build(seed, a_true)
            pe = build(seed, a_true, prior=True)
            red[mode].append(run_closed_loop(te, pe, mode, rng))
        print(f"  world {wld+1}/{args.n_worlds} done", flush=True)

    bmean = np.mean(red["baseline"])
    order = ["static"] + [m for m in args.methods if m != "static"]
    print(f"\n===== SURVEILLANCE-DRIVEN ADAPTIVE ALLOCATION ({args.n_worlds} worlds) =====")
    print(f"{'method':12s} | {'mean red%':>10s} | {'worst-case (CVaR) red%':>22s}")
    print("-" * 50)
    ratios = {}
    for m in order:
        arr = np.array(red[m]); r = (bmean - arr) / bmean * 100
        ratios[m] = r
        print(f"{m:12s} | {r.mean():10.2f} | {cvar_low(r):22.2f}")
    if all(k in ratios for k in ("adapt", "open", "oracle")):
        gain = ratios["adapt"].mean() - ratios["open"].mean()
        avail = ratios["oracle"].mean() - ratios["open"].mean()
        print(f"\nARMOR-Adapt vs open-loop: +{gain:.2f} pp mean reduction "
              f"(recovers {gain/max(avail,1e-9)*100:.0f}% of the oracle's "
              f"information advantage of {avail:.2f} pp)")
    np.savez(args.out, red={m: np.array(red[m]) for m in red})
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

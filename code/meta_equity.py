"""
NOVELTY (equity axis): the standard utilitarian objective (minimize TOTAL
infection burden) -- what every method in the benchmark optimizes, and on which
the problem is near-myopic -- yields a spatially UNEQUAL solution: it concentrates
nets where they avert the most cases, leaving the highest-transmission regions
with high residual per-capita burden. That disparity is invisible to total-burden
metrics yet central to deployment ethics ("leave no one behind").

We formulate ITN allocation as an EQUITY-AWARE optimization and trace the
efficiency-equity Pareto frontier. The planner minimizes

    (1-w) * total_burden_norm  +  w * worst_region_percapita_norm

where worst_region_percapita = max_i (burden_i / N_H_i) is the per-capita infection
burden of the WORST-OFF region. w=0 reproduces the utilitarian optimum; w>0
protects the worst-suffering region. We report total %reduction (cases averted),
the worst-region per-capita reduction, and the disparity, across held-out
metapopulation envs. Lifting the worst region a lot at small total cost is the
contribution -- no mean-optimal method (greedy/PPO/oracle) provides it.

Run: python meta_equity.py --n-eval 20
"""
import argparse
import numpy as np

import fast_sim_endo as fe
import meta_planners as mp

WEIGHTS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def regional(env, S, C, tau, R, t, sched, camp_rel, ep_rem):
    cand = np.asarray(sched, np.float64).reshape(1, len(camp_rel), mp.N)
    out = fe.meta_plan_batch_regional(np.asarray(S, np.float64), C, tau, R, env.P,
                                      cand, np.asarray(camp_rel, np.int64), ep_rem, t,
                                      env.k_sel, env.k_rev, env.k_cov, env.W, env.m)
    return out[0]


def cem_equity(env, S, C, tau, R, t, camp_rel, ep_rem, b0, NH, w,
               pop=500, iters=18, elite=50, rng=None):
    """Minimize (1-w)*total_norm + w*worst_percapita_norm."""
    Kr = len(camp_rel)
    ref_pc = (b0 / NH).max()                     # worst no-ITN per-capita burden
    b0sum = b0.sum()
    alphas = [np.ones(mp.N) for _ in range(Kr)]
    best, best_sched = np.inf, None
    cr = np.asarray(camp_rel, np.int64)
    states = np.asarray(S, np.float64)
    for it in range(iters):
        cand = np.empty((pop, Kr, mp.N))
        for j in range(Kr):
            cand[:, j, :] = rng.dirichlet(alphas[j], size=pop) * env.budget
        reg = fe.meta_plan_batch_regional(states, C, tau, R, env.P, cand, cr,
                                          ep_rem, t, env.k_sel, env.k_rev, env.k_cov,
                                          env.W, env.m)
        total_norm = reg.sum(1) / b0sum
        worst_norm = (reg / NH).max(1) / ref_pc
        score = (1 - w) * total_norm + w * worst_norm
        el = np.argsort(score)[:elite]
        for j in range(Kr):
            alphas[j] = np.clip((cand[el, j, :] / env.budget).mean(0) * 10 * (it + 1),
                                1e-2, None)
        if score[el[0]] < best:
            best = score[el[0]]; best_sched = cand[el[0]].copy()
    return best_sched


def metrics(reg, b0, NH):
    total_red = (b0.sum() - reg.sum()) / b0.sum() * 100
    pc = reg / NH; pc0 = b0 / NH
    worst_idx = int(pc0.argmax())                # intrinsically worst-hit region
    worst_red = (pc0[worst_idx] - pc[worst_idx]) / pc0[worst_idx] * 100
    red_i = (b0 - reg) / np.maximum(b0, 1e-9) * 100
    disparity = float(red_i.max() - red_i.min())
    return total_red, worst_red, disparity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--gap", type=int, default=365)
    ap.add_argument("--base-seed", type=int, default=10000)
    ap.add_argument("--out", type=str, default="meta_equity.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    agg = {w: {"total": [], "worst": [], "disp": []} for w in WEIGHTS}
    agg_static = {"total": [], "worst": [], "disp": []}
    print(f"Equity frontier | N={mp.N} | budget={mp.BUDGET:.0f} | "
          f"mobility={mp.MOBILITY} | {args.n_eval} held-out envs", flush=True)

    for i in range(args.n_eval):
        seed = args.base_seed + i
        env = mp.MetaEnv(np.random.default_rng(seed), k=args.k, gap=args.gap)
        env.reset(); env.run_interval(np.zeros(mp.N), mp.FIRST)
        S, C, tau, R, t = env.snapshot()
        NH = env.P[:, 17].copy()
        camp_rel = [j * env.gap for j in range(env.K)]
        ep_rem = env.K * env.gap
        b0 = regional(env, S, C, tau, R, t, np.zeros((env.K, mp.N)), camp_rel, ep_rem)

        st_sched = np.tile(env.pop_static, (env.K, 1))
        reg_st = regional(env, S, C, tau, R, t, st_sched, camp_rel, ep_rem)
        tr, wr, dp = metrics(reg_st, b0, NH)
        agg_static["total"].append(tr); agg_static["worst"].append(wr); agg_static["disp"].append(dp)

        for w in WEIGHTS:
            sched = cem_equity(env, S, C, tau, R, t, camp_rel, ep_rem, b0, NH, w, rng=rng)
            reg = regional(env, S, C, tau, R, t, sched, camp_rel, ep_rem)
            tr, wr, dp = metrics(reg, b0, NH)
            agg[w]["total"].append(tr); agg[w]["worst"].append(wr); agg[w]["disp"].append(dp)
        if i == 0:
            pc0 = b0 / NH
            print(f"  [diag env0] no-ITN per-capita burden by region: "
                  f"{np.round(pc0, 3)}  worst region idx={pc0.argmax()}", flush=True)
        print(f"  env {i+1}/{args.n_eval} done", flush=True)

    print("\n===== EFFICIENCY-EQUITY FRONTIER (metapopulation, held-out) =====")
    print(f"{'method':18s} | {'total red%':>10s} | {'worst-region red%':>17s} | "
          f"{'disparity pp':>12s}")
    print("-" * 66)
    print(f"{'static (WHO)':18s} | {np.mean(agg_static['total']):10.2f} | "
          f"{np.mean(agg_static['worst']):17.2f} | {np.mean(agg_static['disp']):12.2f}")
    for w in WEIGHTS:
        tag = "utilitarian" if w == 0 else ("maximin" if w == 1 else f"w={w}")
        print(f"{('equity ' + tag):18s} | {np.mean(agg[w]['total']):10.2f} | "
              f"{np.mean(agg[w]['worst']):17.2f} | {np.mean(agg[w]['disp']):12.2f}")

    u, e = agg[0.0], agg[1.0]
    print(f"\nutilitarian -> maximin: worst-region reduction "
          f"{np.mean(u['worst']):.1f}% -> {np.mean(e['worst']):.1f}% "
          f"(+{np.mean(e['worst'])-np.mean(u['worst']):.1f} pp) at total cost "
          f"{np.mean(u['total'])-np.mean(e['total']):.1f} pp")
    np.savez(args.out, weights=WEIGHTS, agg={w: agg[w] for w in WEIGHTS}, static=agg_static)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

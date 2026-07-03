"""
WS2/WS3/WS4 -- non-learned, simulator-based allocation planners.

All three use the simulator at decision time (model-based control), which is
exactly the comparison reviewers asked for. Regions are decoupled, so lookaheads
are single-region and fast (numba, fast_sim.py).

  greedy  (WS2): receding-horizon greedy marginal-benefit. Assign the budget in
                 chunks; each chunk goes to the region with the largest simulated
                 reduction in its own 50-day post-campaign burden.
  mpc     (WS3): receding-horizon CEM on the simulator. At each campaign sample
                 Dirichlet allocations, score by simulated 50-day burden, refit
                 the elite, iterate, deploy the best. (= QT-Opt's action search
                 WITHOUT a learned Q -> tests whether learning is even needed.)
  oracle  (WS4): open-loop joint optimization of all three campaigns (3N vars) by
                 heavy CEM with full simulator access and full-episode rollouts.
                 Non-deployable upper bound that frames effect sizes. (Dynamics are
                 deterministic per env, so open-loop optimal = closed-loop optimal.)

Evaluated on the SAME held-out envs (base_seed=10000) as every other method.

Run:  python baselines_planner.py --model seitr  [--methods greedy mpc oracle]
      python baselines_planner.py --model seitar
"""
import argparse
import numpy as np
from scipy import stats

import fast_sim as fs

CAMPAIGN_DAYS = (30, 80, 130)
EPISODE_DAYS = 180
HORIZON = 50


# --------------------------------------------------------------------------- #
# Per-region snapshot helpers                                                 #
# --------------------------------------------------------------------------- #
def snapshot(env, model):
    """Return per-region (state, params, C, tau, t, Rseries) lists."""
    S, P, C, TAU, T, R = [], [], [], [], [], []
    for c in env.counties:
        if model == "seitr":
            S.append(fs.state_seitr(c)); P.append(fs.pack_seitr(c))
            R.append(np.asarray(c.Rs, dtype=np.float64))
        else:
            S.append(fs.state_seitar(c)); P.append(fs.pack_seitar(c))
            R.append(None)
        C.append(float(c.C)); TAU.append(float(c.tau)); T.append(int(c.t))
    return S, P, C, TAU, T, R


def lookahead(model, s, C, tau, t, nets, P, Rser, horizon):
    if model == "seitr":
        return fs.seitr_lookahead(s, C, tau, t, nets, P, Rser, horizon)
    return fs.seitar_lookahead(s, C, tau, t, nets, P, horizon)


def windows(model, s, C, tau, P, Rser, alloc3):
    if model == "seitr":
        return fs.seitr_windows(s, C, tau, P, Rser, alloc3, *CAMPAIGN_DAYS, EPISODE_DAYS)
    return fs.seitar_windows(s, C, tau, P, alloc3, *CAMPAIGN_DAYS, EPISODE_DAYS)


# --------------------------------------------------------------------------- #
# Planners (return an allocation vector for the current campaign)             #
# --------------------------------------------------------------------------- #
def greedy_alloc(env, model, budget, n, horizon=HORIZON, chunk_frac=0.05):
    S, P, C, TAU, T, R = snapshot(env, model)
    alloc = np.zeros(n)
    cur = np.array([lookahead(model, S[i], C[i], TAU[i], T[i], 0.0, P[i], R[i], horizon)
                    for i in range(n)])
    chunk = budget * chunk_frac
    remaining = budget
    while remaining > 1e-6:
        c = min(chunk, remaining)
        best_i, best_gain, best_new = -1, -np.inf, 0.0
        for i in range(n):
            bnew = lookahead(model, S[i], C[i], TAU[i], T[i], alloc[i] + c, P[i], R[i], horizon)
            gain = cur[i] - bnew
            if gain > best_gain:
                best_gain, best_i, best_new = gain, i, bnew
        alloc[best_i] += c
        cur[best_i] = best_new
        remaining -= c
    return alloc.astype(np.float32)


def mpc_alloc(env, model, budget, n, rng, horizon=HORIZON, pop=128, iters=5, elite=16):
    S, P, C, TAU, T, R = snapshot(env, model)
    alpha = np.ones(n)
    best_a, best_score = None, np.inf
    for it in range(iters):
        cand = rng.dirichlet(alpha, size=pop) * budget
        score = np.empty(pop)
        for p in range(pop):
            tot = 0.0
            for i in range(n):
                tot += lookahead(model, S[i], C[i], TAU[i], T[i], cand[p, i], P[i], R[i], horizon)
            score[p] = tot
        el = np.argsort(score)[:elite]
        alpha = np.clip((cand[el] / budget).mean(0) * 10 * (it + 1), 1e-2, None)
        if score[el[0]] < best_score:
            best_score, best_a = score[el[0]], cand[el[0]]
    return best_a.astype(np.float32)


def oracle_schedule(env, model, budget, n, rng, pop=1500, iters=30, elite=120):
    """Open-loop 3-campaign joint optimum (full simulator access), batched njit."""
    env.reset()
    S, P, C, TAU, T, R = snapshot(env, model)   # t=0, tau=0, C=init
    states = np.array(S, dtype=np.float64)
    C0 = np.array(C); tau0 = np.array(TAU); Pn = np.array(P, dtype=np.float64)
    if model == "seitr":
        Rn = np.array([np.asarray(r, dtype=np.float64) for r in R])
    alphas = [np.ones(n) for _ in range(3)]
    best_sched, best_cum = None, np.inf
    for it in range(iters):
        cand = np.empty((pop, 3, n))
        for j in range(3):
            cand[:, j, :] = rng.dirichlet(alphas[j], size=pop) * budget
        if model == "seitr":
            cums = fs.seitr_oracle_batch(states, C0, tau0, Pn, Rn, cand,
                                         *CAMPAIGN_DAYS, EPISODE_DAYS)
        else:
            cums = fs.seitar_oracle_batch(states, C0, tau0, Pn, cand,
                                          *CAMPAIGN_DAYS, EPISODE_DAYS)
        el = np.argsort(cums)[:elite]
        for j in range(3):
            alphas[j] = np.clip((cand[el, j, :] / budget).mean(0) * 10 * (it + 1), 1e-2, None)
        if cums[el[0]] < best_cum:
            best_cum = cums[el[0]]
            best_sched = cand[el[0]].copy()  # (3, n)
    return best_sched


# --------------------------------------------------------------------------- #
# Closed-loop episode driver (planner decides at each campaign)               #
# --------------------------------------------------------------------------- #
def run_planner_episode(env, decide, n):
    env.reset()
    window = [0.0, 0.0, 0.0, 0.0]
    idx = 0
    for day in range(EPISODE_DAYS):
        if day in CAMPAIGN_DAYS:
            alloc = decide(env, idx)
            idx += 1
        else:
            alloc = np.zeros(n, np.float32)
        _, burden = env.step(alloc)
        window[0 if day < CAMPAIGN_DAYS[0] else idx] += burden
    sc = window[0]
    return float(sum(window[k + 1] / sc for k in range(3)))


def run_oracle_episode(env, sched, n):
    """Deploy a precomputed open-loop 3-campaign schedule."""
    env.reset()
    window = [0.0, 0.0, 0.0, 0.0]
    idx = 0
    for day in range(EPISODE_DAYS):
        if day in CAMPAIGN_DAYS:
            alloc = sched[idx].astype(np.float32)
            idx += 1
        else:
            alloc = np.zeros(n, np.float32)
        _, burden = env.step(alloc)
        window[0 if day < CAMPAIGN_DAYS[0] else idx] += burden
    sc = window[0]
    return float(sum(window[k + 1] / sc for k in range(3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["seitr", "seitar"], default="seitr")
    ap.add_argument("--methods", nargs="+", default=["greedy", "mpc", "oracle"])
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--base-seed", type=int, default=10000)
    args = ap.parse_args()

    if args.model == "seitr":
        import calibrated_experiment as M
        rdf = M.load_resistance_trajectories()
        def make_env(seed):
            return M.CalibratedEnv(rdf, np.random.default_rng(seed))
        npz = "calibrated_results.npz"
    else:
        import extended_experiment as M
        def make_env(seed):
            return M.ExtEnv(np.random.default_rng(seed))
        npz = "extended_results.npz"

    n, budget = M.ACTION_DIM, M.BUDGET
    plan_rng = np.random.default_rng(0)

    res = {m: [] for m in (["baseline", "static"] + args.methods)}
    for i in range(args.n_eval):
        seed = args.base_seed + i
        # baseline + static for reference (recompute on identical envs)
        e = make_env(seed); res["baseline"].append(run_planner_episode(e, lambda env, k: np.zeros(n, np.float32), n))
        e = make_env(seed); res["static"].append(run_planner_episode(e, lambda env, k: M.POP_STATIC.copy(), n))
        if "greedy" in args.methods:
            e = make_env(seed)
            res["greedy"].append(run_planner_episode(e, lambda env, k: greedy_alloc(env, args.model, budget, n), n))
        if "mpc" in args.methods:
            e = make_env(seed)
            res["mpc"].append(run_planner_episode(e, lambda env, k: mpc_alloc(env, args.model, budget, n, plan_rng), n))
        if "oracle" in args.methods:
            e = make_env(seed)
            sched = oracle_schedule(e, args.model, budget, n, plan_rng)
            res["oracle"].append(run_oracle_episode(e, sched, n))
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{args.n_eval} envs", flush=True)

    res = {k: np.array(v) for k, v in res.items()}
    b = res["baseline"].mean()
    static_arr = res["static"]

    # merge learned + heuristics for a full table
    extra = {}
    nev = args.n_eval
    try:
        d = np.load(npz, allow_pickle=True)
        le = d["last_eval"].item()
        for m in ["DQN", "bandit", "QT-Opt", "PPO"]:
            if m in le and len(np.asarray(le[m])) == nev:
                extra[m] = np.asarray(le[m], float)
    except Exception:
        pass
    try:
        h = np.load(f"ws1_heuristics_{args.model}.npz", allow_pickle=True)["results"].item()
        for m in ["incidence", "prevalence", "burden_transmit", "resistance_aware"]:
            if m in h and len(np.asarray(h[m])) == nev:
                extra["heur:" + m] = np.asarray(h[m], float)
    except Exception:
        pass

    allres = dict(res); allres.update(extra)
    order = (["baseline", "static"] + [k for k in ["heur:incidence", "heur:prevalence",
             "heur:burden_transmit", "heur:resistance_aware"] if k in allres]
             + args.methods + [k for k in ["DQN", "bandit", "QT-Opt", "PPO"] if k in allres])

    print(f"\n===== WS2-4 planners + full comparison on {args.model.upper()} "
          f"({args.n_eval} held-out envs) =====")
    print(f"{'method':18s} | {'inf.ratio':>10s} | {'%reduction':>10s} | "
          f"{'beats static':>12s} | {'p vs static':>11s}")
    print("-" * 72)
    for m in order:
        arr = allres[m]
        red = (b - arr.mean()) / b * 100
        if m in ("baseline", "static"):
            print(f"{m:18s} | {arr.mean():10.3f} | {red:9.2f}% | {'--':>12s} | {'--':>11s}")
            continue
        beats = (arr < static_arr).mean() * 100
        try:
            _, pv = stats.wilcoxon(arr, static_arr)
        except ValueError:
            pv = float("nan")
        print(f"{m:18s} | {arr.mean():10.3f} | {red:9.2f}% | {beats:10.0f}% | {pv:11.2e}")

    np.savez(f"ws234_planners_{args.model}.npz", results=res)
    print(f"\nsaved -> ws234_planners_{args.model}.npz")


if __name__ == "__main__":
    main()

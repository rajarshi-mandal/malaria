"""
WS1 -- non-RL heuristic allocation baselines.

Six interpretable, deterministic, budget-feasible, N-agnostic rules that read the
observed per-region state and allocate the budget proportional to a weight:

  population        w_i = pop_i                         (= the existing static rule)
  incidence         w_i = E_H + I_H                     (current infections)
  prevalence        w_i = parasite stock (I+T[+A])
  coverage_gap      w_i = (1 - C_i) * pop_i             (fill low-coverage regions)
  burden_transmit   w_i = (E_H+I_H) * a_t               (burden x transmission)
  resistance_aware  w_i = (E_H+I_H) * a_t * (1-0.5 R)   (x expected ITN effectiveness)

These directly answer the reviewer question: would a simple epidemiology-weighted
rule do what PPO does? Evaluated on the SAME held-out envs (base_seed=10000) as the
learned methods, so the per-env arrays are directly comparable to the saved npz.

Run:  python baselines_heuristic.py --model seitr
      python baselines_heuristic.py --model seitar
"""
import argparse
import numpy as np
from scipy import stats


def feature_index(model):
    # per-region obs layout
    if model == "seitr":   # [mult,a,C,1/rho, S,E,I,T,R, S_M,E_M,I_M, a_t, Rresist] (14)
        return dict(FEATS=14, mult=0, C=2, E=5, I=6, T=7, R=8, prev=(6, 7), a_t=12, res=13)
    else:                  # SEITAR [mult,a,C,1/rho, S,E,I,T,A,R, S_M,E_M,I_M, a_t, Rresist] (15)
        return dict(FEATS=15, mult=0, C=2, E=5, I=6, T=7, A=8, R=9, prev=(6, 7, 8), a_t=13, res=14)


def make_heuristics(model, budget, action_dim):
    ix = feature_index(model)
    F = ix["FEATS"]

    def alloc_from_weights(w):
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s <= 0:
            return np.full(action_dim, budget / action_dim, np.float32)
        return (w / s * budget).astype(np.float32)

    def grid(state):
        return np.asarray(state, np.float32).reshape(action_dim, F)

    def population(s, k):
        g = grid(s); return alloc_from_weights(g[:, ix["mult"]])

    def incidence(s, k):
        g = grid(s); return alloc_from_weights(g[:, ix["E"]] + g[:, ix["I"]])

    def prevalence(s, k):
        g = grid(s); return alloc_from_weights(g[:, list(ix["prev"])].sum(1))

    def coverage_gap(s, k):
        g = grid(s); return alloc_from_weights((1.0 - g[:, ix["C"]]) * g[:, ix["mult"]])

    def burden_transmit(s, k):
        g = grid(s); return alloc_from_weights((g[:, ix["E"]] + g[:, ix["I"]]) * g[:, ix["a_t"]])

    def resistance_aware(s, k):
        g = grid(s)
        w = (g[:, ix["E"]] + g[:, ix["I"]]) * g[:, ix["a_t"]] * (1.0 - 0.5 * g[:, ix["res"]])
        return alloc_from_weights(w)

    return {"incidence": incidence, "prevalence": prevalence,
            "coverage_gap": coverage_gap, "burden_transmit": burden_transmit,
            "resistance_aware": resistance_aware}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["seitr", "seitar"], default="seitr")
    ap.add_argument("--n-eval", type=int, default=50)
    args = ap.parse_args()

    if args.model == "seitr":
        import calibrated_experiment as M
        rdf = M.load_resistance_trajectories()
        def ev(pols):
            return M.evaluate(pols, rdf, args.n_eval)
        npz = "calibrated_results.npz"
    else:
        import extended_experiment as M
        def ev(pols):
            return M.evaluate(pols, args.n_eval)
        npz = "extended_results.npz"

    heur = make_heuristics(args.model, M.BUDGET, M.ACTION_DIM)
    pols = {"baseline": M.baseline_policy, "static": M.static_policy}
    pols.update(heur)

    res = ev(pols)
    b = res["baseline"].mean()
    static_arr = res["static"]

    # bring in the learned methods from the saved npz (same held-out envs)
    learned = {}
    try:
        d = np.load(npz, allow_pickle=True)
        le = d["last_eval"].item()
        for m in ["DQN", "bandit", "QT-Opt", "PPO"]:
            if m in le:
                learned[m] = np.asarray(le[m], float)
    except Exception as e:
        print(f"(could not load {npz}: {e})")

    order = ["baseline", "static", "incidence", "prevalence", "coverage_gap",
             "burden_transmit", "resistance_aware"] + list(learned.keys())
    allres = dict(res); allres.update(learned)

    print(f"\n===== WS1 heuristic baselines on {args.model.upper()} "
          f"({args.n_eval} held-out envs) =====")
    print(f"{'method':16s} | {'inf.ratio':>10s} | {'%reduction':>10s} | "
          f"{'beats static':>12s} | {'p vs static':>11s}")
    print("-" * 70)
    for m in order:
        if m not in allres:
            continue
        arr = allres[m]
        red = (b - arr.mean()) / b * 100
        if m in ("baseline", "static"):
            print(f"{m:16s} | {arr.mean():10.3f} | {red:9.2f}% | {'--':>12s} | {'--':>11s}")
            continue
        beats = (arr < static_arr).mean() * 100
        try:
            _, pv = stats.wilcoxon(arr, static_arr)
        except ValueError:
            pv = float("nan")
        print(f"{m:16s} | {arr.mean():10.3f} | {red:9.2f}% | {beats:10.0f}% | {pv:11.2e}")

    np.savez(f"ws1_heuristics_{args.model}.npz",
             results={k: np.asarray(v) for k, v in res.items()})
    print(f"\nsaved -> ws1_heuristics_{args.model}.npz")


if __name__ == "__main__":
    main()

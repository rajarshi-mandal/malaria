"""Necessity ablation for the value-of-information exploration term. Sweeps the
exploration weight lambda for ARMOR-IDA over identical worlds (lambda=0 is exactly
certainty-equivalent MPC). Shows the exploration term is necessary and where the
sweet spot is. Run: python meta_bayesian_ablation.py"""
import argparse
import numpy as np

import meta_bayesian as mb
import meta_planners as mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-worlds", type=int, default=8)
    ap.add_argument("--campaigns", type=int, default=8)
    ap.add_argument("--sigma-obs", type=float, default=0.6)
    ap.add_argument("--kap0", type=float, default=0.05)
    ap.add_argument("--lams", type=float, nargs="+", default=[0, 2, 4, 8, 16, 32])
    args = ap.parse_args()
    mb.KAP0 = args.kap0
    rng = np.random.default_rng(0); rng_p = np.random.default_rng(7)
    base = []; res = {l: [] for l in args.lams}

    def te(seed, a_true):
        e = mb.build(seed, a_true); e.K = args.campaigns
        e.camp = np.array([mp.FIRST + j * e.gap for j in range(e.K)], np.int64)
        e.ep = mp.FIRST + e.K * e.gap
        return e

    print(f"VoI exploration-weight ablation | K={args.campaigns} sigma={args.sigma_obs} "
          f"kap0={args.kap0} | {args.n_worlds} worlds", flush=True)
    for w in range(args.n_worlds):
        seed = 10000 + w; a_true = mb.true_biting(seed, rng_p)
        base.append(mp.run_episode(te(seed, a_true), lambda o, k: np.zeros(mp.N))[0])
        for l in args.lams:
            r, _ = mb.run_world(te(seed, a_true), "ida", seed, args.sigma_obs, l, 1.0, rng)
            res[l].append(r)
        print(f"  world {w+1}/{args.n_worlds} done", flush=True)
    b = np.mean(base)
    print("\n===== EXPLORATION-WEIGHT (lambda) ABLATION =====")
    print(f"{'lambda':>8s} | {'mean red%':>10s}   (lambda=0 is certainty-equivalent MPC)")
    print("-" * 34)
    for l in args.lams:
        print(f"{l:8.0f} | {(b-np.mean(res[l]))/b*100:10.2f}")
    np.savez("meta_bayesian_ablation.npz", lams=args.lams,
             red={float(l): np.array(res[l]) for l in args.lams}, base=np.array(base))
    print("saved -> meta_bayesian_ablation.npz")


if __name__ == "__main__":
    main()

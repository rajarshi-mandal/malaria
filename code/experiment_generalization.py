"""
WS5 -- geographic generalization for the LEARNED policy.

The held-out protocol used elsewhere holds out stochastic realizations of a fixed
region set. This script tests the stronger claim: does a PPO policy trained on one set
of real regions generalize to a DISJOINT set of real regions (new geography)?

From the 50-real-region pool we take two disjoint 8-region sets, A and B, each spanning
the PfPR gradient. We train PPO on A and evaluate it on B (out-of-distribution
geography), comparing to static and to the training-free greedy planner on B. PPO
learns a mapping from region features to allocation, so transfer is meaningful.

Run: python experiment_generalization.py [--ppo-steps 60000]
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats

import baselines_planner as BP
from experiment_scaled import ScaledSeitarEnv, run_episode, train_eval_ppo


def eval_methods(regions, ppo_model_fnorm, budget, n, n_eval, base_seed=15000):
    pop = regions.pop_multiplier.to_numpy()
    POP_STATIC = (pop / pop.sum() * budget).astype(np.float32)
    model, FNORM = ppo_model_fnorm

    def ppo_pol(obs, k):
        a, _ = model.predict((np.asarray(obs) / FNORM).astype(np.float32), deterministic=True)
        e = np.exp(a - a.max())
        return ((e / e.sum()) * budget).astype(np.float32)

    out = {"baseline": [], "static": [], "greedy": [], "PPO": []}
    for i in range(n_eval):
        seed = base_seed + i
        e = ScaledSeitarEnv(regions, budget, np.random.default_rng(seed))
        out["baseline"].append(run_episode(e, lambda o, k: np.zeros(n, np.float32)))
        e = ScaledSeitarEnv(regions, budget, np.random.default_rng(seed))
        out["static"].append(BP.run_planner_episode(e, lambda env, k: POP_STATIC.copy(), n))
        e = ScaledSeitarEnv(regions, budget, np.random.default_rng(seed))
        out["greedy"].append(BP.run_planner_episode(e, lambda env, k: BP.greedy_alloc(env, "seitar", budget, n), n))
        e = ScaledSeitarEnv(regions, budget, np.random.default_rng(seed))
        out["PPO"].append(run_episode(e, ppo_pol))
    return {k: np.array(v) for k, v in out.items()}


def train_ppo_model(regions, budget, n, steps):
    """Train PPO on `regions`; return (model, FNORM) for reuse on other region sets."""
    from experiment_scaled import ScaledSeitarEnv as Env
    import torch
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    torch.set_num_threads(1)
    RS = 5.0
    FNORM = np.tile(np.array([1, 1, 1, 15, 1e4, 1e3, 1e3, 1e3, 5e3, 1e3,
                              3e4, 3e3, 3e3, 1, 1], np.float32), n)

    def decode(a):
        a = np.asarray(a, np.float64); e = np.exp(a - a.max())
        return ((e / e.sum()) * budget).astype(np.float32)
    base = [run_episode(Env(regions, budget, np.random.default_rng(777 + i)),
                        lambda o, k: np.zeros(n, np.float32)) for i in range(5)]
    center = float(np.mean(base) / 3.0)

    class GW(gym.Env):
        def __init__(self, seed):
            super().__init__()
            self._rng = np.random.default_rng(seed)
            self.observation_space = spaces.Box(-np.inf, np.inf, (n * 15,), np.float32)
            self.action_space = spaces.Box(-5.0, 5.0, (n,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.env = Env(regions, budget, self._rng); self.env.reset()
            self.scale = 0.0; obs = None
            for _ in range(30):
                obs, b = self.env.step(np.zeros(n, np.float32)); self.scale += b
            self.k = 0
            return (obs / FNORM).astype(np.float32), {}

        def step(self, action):
            alloc = decode(action); burden = 0.0; obs = None
            for d in range(50):
                obs, b = self.env.step(alloc if d == 0 else np.zeros(n, np.float32)); burden += b
            ratio = burden / self.scale; self.k += 1
            return (obs / FNORM).astype(np.float32), float(RS * (center - ratio)), self.k >= 3, False, {}

    m = PPO("MlpPolicy", Monitor(GW(0)), seed=0, verbose=0, n_steps=600, batch_size=200,
            gamma=0.99, ent_coef=0.0, learning_rate=3e-4, policy_kwargs=dict(net_arch=[128, 128]))
    m.learn(steps)
    return m, FNORM


def report(name, res):
    b = res["baseline"].mean(); s = res["static"]
    print(f"\n--- {name} ---")
    print(f"{'method':10s} | {'%reduction':>10s} | {'beats static':>12s} | {'p':>10s}")
    for m in ["static", "greedy", "PPO"]:
        arr = res[m]; red = (b - arr.mean()) / b * 100
        if m == "static":
            print(f"{m:10s} | {red:9.2f}% | {'--':>12s} | {'--':>10s}")
            continue
        beats = (arr < s).mean() * 100
        try:
            _, pv = stats.wilcoxon(arr, s)
        except ValueError:
            pv = float("nan")
        print(f"{m:10s} | {red:9.2f}% | {beats:10.0f}% | {pv:10.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="real_regions_n50.csv")
    ap.add_argument("--ppo-steps", type=int, default=60000)
    ap.add_argument("--n-eval", type=int, default=40)
    args = ap.parse_args()

    pool = pd.read_csv(args.pool).sort_values("PfPR_obs").reset_index(drop=True)
    A_idx = [0, 7, 14, 21, 28, 35, 42, 49]
    B_idx = [3, 10, 17, 24, 31, 38, 44, 47]
    A = pool.iloc[A_idx].reset_index(drop=True)
    B = pool.iloc[B_idx].reset_index(drop=True)
    n = len(A); budget = 20000.0
    print(f"Geographic generalization | train set A ({n} regions), "
          f"test set B ({len(B)} disjoint regions)", flush=True)
    print(f"A PfPR {A.PfPR_obs.min()*100:.1f}-{A.PfPR_obs.max()*100:.1f}% | "
          f"B PfPR {B.PfPR_obs.min()*100:.1f}-{B.PfPR_obs.max()*100:.1f}%", flush=True)

    mA = train_ppo_model(A, budget, n, args.ppo_steps)
    print("PPO trained on A", flush=True)

    res_A = eval_methods(A, mA, budget, n, args.n_eval)   # in-distribution
    res_B = eval_methods(B, mA, budget, n, args.n_eval)   # out-of-distribution geography
    report("PPO(A) on A  [in-distribution]", res_A)
    report("PPO(A) on B  [unseen geography]", res_B)

    np.savez("ws5_generalization.npz", res_A=res_A, res_B=res_B)
    print("\nsaved -> ws5_generalization.npz")


if __name__ == "__main__":
    main()

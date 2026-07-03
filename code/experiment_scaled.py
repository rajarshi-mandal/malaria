"""
WS6 (scale) + WS5 (generalization) -- the full allocation ladder on N real regions.

Runs the complete method ladder on the SEITAR environment built from
real_regions_n{N}.csv (default 50 real admin-1 regions):

  static  -> incidence/prevalence heuristics -> greedy / MPC / oracle (simulator
  -based, fast_sim) -> PPO (learned).

Two reviewer concerns at once:
 * SCALE (WS6): does the benchmark/pipeline work at 50 real regions?
 * GENERALIZATION (WS5): the simulator-based methods (heuristics, greedy, MPC,
   oracle) require NO training, so beating static on 50 *unseen* real regions is a
   direct test that the *finding* (learning-free optimization beats standard
   practice) generalizes to new geography, not just new stochastic seeds.

PPO is trained here too (fixed budget); per the Tier-2 scale test, the learned
advantage needs training budget to grow with N, so PPO at large N with a modest
budget is reported honestly as a lower bound on what learning can reach.

Run: python experiment_scaled.py --n-regions 50 [--ppo-steps 40000] [--no-ppo]
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats

from extended_ode import ExtendedCounty
import baselines_planner as BP
import baselines_heuristic as BH

CAMPAIGN_DAYS = (30, 80, 130)
EPISODE_DAYS = 180
HORIZON = 50


class StableExtendedCounty(ExtendedCounty):
    """ExtendedCounty with per-day [0,N] clipping. Forward Euler dt=1 can overshoot
    at the extreme biting rates fitted for very-high-incidence regions (T2a); the
    calibration (fast_sim.seitar_steady) is already clipped, so the env must clip too
    for internal consistency. Clipping never triggers in the valid regime."""

    def step(self, new_itns=0.0, seasonal=False):
        out = super().step(new_itns, seasonal)
        nh, nm = self.N_H, self.N_M

        def cl(v, cap):
            if v != v:
                return 0.0
            return 0.0 if v < 0.0 else (cap if v > cap else v)
        self.S_H = cl(self.S_H, nh); self.E_H = cl(self.E_H, nh); self.I_H = cl(self.I_H, nh)
        self.T_H = cl(self.T_H, nh); self.A_H = cl(self.A_H, nh); self.R_H = cl(self.R_H, nh)
        self.S_M = cl(self.S_M, nm); self.E_M = cl(self.E_M, nm); self.I_M = cl(self.I_M, nm)
        return out


class ScaledSeitarEnv:
    """SEITAR allocation env over an arbitrary set of real regions (steady-init)."""
    _steady_cache = {}

    def __init__(self, regions, budget, rng):
        self.regions = regions
        self.budget = budget
        self.N = len(regions)
        self.rng = rng
        self.total_pop = float((10000 * regions.pop_multiplier).sum())
        self.counties = [StableExtendedCounty(r.biting_rate, r.ITN_coverage, r.pop_multiplier,
                                              int(r.treatment_seeking), r.p_sympt)
                         for _, r in regions.iterrows()]
        self._steady = self._get_steady()

    def _get_steady(self):
        key = id(self.regions)
        if key not in ScaledSeitarEnv._steady_cache:
            st = []
            for _, r in self.regions.iterrows():
                c = StableExtendedCounty(r.biting_rate, r.ITN_coverage, r.pop_multiplier,
                                         int(r.treatment_seeking), r.p_sympt)
                for _ in range(2500):
                    c.step(0.0, seasonal=False)
                st.append((c.S_H, c.E_H, c.I_H, c.T_H, c.A_H, c.R_H, c.S_M, c.E_M, c.I_M))
            ScaledSeitarEnv._steady_cache[key] = st
        return ScaledSeitarEnv._steady_cache[key]

    def reset(self):
        for c, s in zip(self.counties, self._steady):
            c.reset()
            (c.S_H, c.E_H, c.I_H, c.T_H, c.A_H, c.R_H, c.S_M, c.E_M, c.I_M) = s
        return self._obs()

    def _obs(self):
        o = []
        for c in self.counties:
            o.extend([c.multiplier, c.a, c.C, 1.0 / c.rho,
                      c.S_H, c.E_H, c.I_H, c.T_H, c.A_H, c.R_H,
                      c.S_M, c.E_M, c.I_M, c.a_t, c.R])
        return np.array(o, dtype=np.float32)

    def step(self, action):
        burden = 0.0
        for i, c in enumerate(self.counties):
            c.step(action[i], seasonal=True)
            burden += c.E_H + c.I_H
        return self._obs(), burden / self.total_pop


def run_episode(env, policy):
    obs = env.reset()
    window = [0.0, 0.0, 0.0, 0.0]
    idx = 0
    for day in range(EPISODE_DAYS):
        if day in CAMPAIGN_DAYS:
            a = policy(obs, idx)
            idx += 1
        else:
            a = np.zeros(env.N, np.float32)
        obs, burden = env.step(a)
        window[0 if day < CAMPAIGN_DAYS[0] else idx] += burden
    sc = window[0]
    return float(sum(window[k + 1] / sc for k in range(3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-regions", type=int, default=50)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=10000)
    ap.add_argument("--ppo-steps", type=int, default=40000)
    ap.add_argument("--no-ppo", action="store_true")
    ap.add_argument("--oracle-pop", type=int, default=800)
    ap.add_argument("--oracle-iters", type=int, default=18)
    args = ap.parse_args()

    regions = pd.read_csv(f"real_regions_n{args.n_regions}.csv")
    N = len(regions)
    budget = 2500.0 * N
    pop = regions.pop_multiplier.to_numpy()
    POP_STATIC = (pop / pop.sum() * budget).astype(np.float32)
    print(f"SEITAR ladder | N={N} real regions | budget={budget:.0f} | "
          f"{args.n_eval} held-out envs", flush=True)

    def make_env(seed):
        return ScaledSeitarEnv(regions, budget, np.random.default_rng(seed))

    heur = BH.make_heuristics("seitar", budget, N)
    plan_rng = np.random.default_rng(0)

    methods = {
        "static": lambda env, k: POP_STATIC.copy(),
        "incidence": lambda env, k, f=heur["incidence"]: f(env._obs(), k),
        "prevalence": lambda env, k, f=heur["prevalence"]: f(env._obs(), k),
        "greedy": lambda env, k: BP.greedy_alloc(env, "seitar", budget, N),
        "mpc": lambda env, k: BP.mpc_alloc(env, "seitar", budget, N, plan_rng),
    }

    res = {"baseline": [], "static": [], "incidence": [], "prevalence": [],
           "greedy": [], "mpc": [], "oracle": []}
    for i in range(args.n_eval):
        seed = args.base_seed + i
        e = make_env(seed); res["baseline"].append(run_episode(e, lambda o, k: np.zeros(N, np.float32)))
        for name in ["static", "incidence", "prevalence", "greedy", "mpc"]:
            e = make_env(seed)
            res[name].append(BP.run_planner_episode(e, methods[name], N))
        e = make_env(seed)
        sched = BP.oracle_schedule(e, "seitar", budget, N, plan_rng,
                                   pop=args.oracle_pop, iters=args.oracle_iters,
                                   elite=max(10, args.oracle_pop // 12))
        res["oracle"].append(BP.run_oracle_episode(e, sched, N))
        if (i + 1) % 5 == 0:
            print(f"  ...{i+1}/{args.n_eval}", flush=True)

    # checkpoint the (expensive) non-learned ladder before training PPO
    np.savez(f"ws56_scaled_n{N}.npz", results={k: np.array(v) for k, v in res.items()})
    print(f"  [checkpoint] non-learned ladder saved -> ws56_scaled_n{N}.npz", flush=True)

    # optional PPO (learned) -- honest lower bound at large N
    if not args.no_ppo:
        try:
            res["PPO"] = train_eval_ppo(regions, budget, N, args)
        except Exception as ex:
            print(f"  [warn] PPO step failed ({ex}); reporting non-learned ladder only", flush=True)

    res = {k: np.array(v) for k, v in res.items()}
    b = res["baseline"].mean()
    static_arr = res["static"]
    np.savez(f"ws56_scaled_n{N}.npz", results=res)

    print(f"\n===== SEITAR ladder on N={N} real regions ({args.n_eval} held-out envs) =====")
    print(f"{'method':12s} | {'inf.ratio':>10s} | {'%reduction':>10s} | "
          f"{'beats static':>12s} | {'p vs static':>11s}")
    print("-" * 66)
    order = ["baseline", "static", "incidence", "prevalence", "greedy", "mpc", "oracle"]
    if "PPO" in res:
        order.append("PPO")
    for m in order:
        arr = res[m]; red = (b - arr.mean()) / b * 100
        if m in ("baseline", "static"):
            print(f"{m:12s} | {arr.mean():10.3f} | {red:9.2f}% | {'--':>12s} | {'--':>11s}")
            continue
        beats = (arr < static_arr).mean() * 100
        try:
            _, pv = stats.wilcoxon(arr, static_arr)
        except ValueError:
            pv = float("nan")
        print(f"{m:12s} | {arr.mean():10.3f} | {red:9.2f}% | {beats:10.0f}% | {pv:11.2e}")
    print(f"\nsaved -> ws56_scaled_n{N}.npz")


def train_eval_ppo(regions, budget, N, args):
    import torch
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    torch.set_num_threads(1)
    REWARD_SCALE = 5.0
    FNORM = np.tile(np.array([1, 1, 1, 15, 1e4, 1e3, 1e3, 1e3, 5e3, 1e3,
                              3e4, 3e3, 3e3, 1, 1], np.float32), N)

    def decode(a):
        a = np.asarray(a, np.float64); e = np.exp(a - a.max())
        return ((e / e.sum()) * budget).astype(np.float32)

    # measure center
    base = []
    for i in range(5):
        e = ScaledSeitarEnv(regions, budget, np.random.default_rng(777 + i))
        base.append(run_episode(e, lambda o, k: np.zeros(N, np.float32)))
    center = float(np.mean(base) / 3.0)

    class GW(gym.Env):
        def __init__(self, seed):
            super().__init__()
            self._rng = np.random.default_rng(seed)
            self.observation_space = spaces.Box(-np.inf, np.inf, (N * 15,), np.float32)
            self.action_space = spaces.Box(-5.0, 5.0, (N,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.env = ScaledSeitarEnv(regions, budget, self._rng)
            self.env.reset(); self.scale = 0.0; obs = None
            for _ in range(CAMPAIGN_DAYS[0]):
                obs, b = self.env.step(np.zeros(N, np.float32)); self.scale += b
            self.k = 0
            return (obs / FNORM).astype(np.float32), {}

        def step(self, action):
            alloc = decode(action); burden = 0.0; obs = None
            for d in range(HORIZON):
                obs, b = self.env.step(alloc if d == 0 else np.zeros(N, np.float32)); burden += b
            ratio = burden / self.scale; self.k += 1
            return (obs / FNORM).astype(np.float32), float(REWARD_SCALE * (center - ratio)), self.k >= 3, False, {}

    m = PPO("MlpPolicy", Monitor(GW(0)), seed=0, verbose=0, n_steps=600, batch_size=200,
            gamma=0.99, ent_coef=0.0, learning_rate=3e-4, policy_kwargs=dict(net_arch=[128, 128]))
    m.learn(args.ppo_steps)
    print("  PPO trained", flush=True)

    def ppo_pol(obs, k):
        a, _ = m.predict((np.asarray(obs) / FNORM).astype(np.float32), deterministic=True)
        return decode(a)

    out = []
    for i in range(args.n_eval):
        e = ScaledSeitarEnv(regions, budget, np.random.default_rng(args.base_seed + i))
        out.append(run_episode(e, ppo_pol))
    return out


if __name__ == "__main__":
    main()

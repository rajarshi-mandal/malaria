"""
Tier 2c: scale test. The manuscript claims the method "extends to 50 or 500
counties ... without changing the methodology." We test that honestly by
calibrating and running the allocation pipeline at N = 8, 20, 50 real admin1
regions, reporting (a) whether learned allocation still beats population-
proportional, and (b) how training wall-clock scales with N.

PPO (strongest learner from Items D/E) on a generalized N-region SEITR env, vs
the population-proportional static policy and the no-ITN baseline. A raw-allocation
core lets baseline (zeros), static (pop-proportional), and PPO (softmax of logits)
all be scored through the identical rollout.

Output: console table + scale_results.csv
"""
import time
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from sar_sequential import MalariaCounty, load_resistance_trajectories

torch.set_num_threads(1)
R_CONST = 0.4; T_SS = 1200; AVG = 365


class StableCounty(MalariaCounty):
    """MalariaCounty with per-day non-negativity/upper-bound clipping. Forward
    Euler dt=1 can overshoot at the extreme biting rates that arise when fitting
    very-high-incidence regions (T2a flagged this); clipping each compartment to
    [0, N] keeps the integration numerically stable without altering valid-regime
    dynamics (clipping never triggers when 0 <= compartments <= N already)."""

    def step(self, new_itns):
        super().step(new_itns)
        nh, nm = self.N_H, self.N_M
        # cheap scalar clamp (handles inf via comparisons; nan -> 0). Clipping each
        # day prevents inf from ever cascading into nan.
        def cl(v, cap):
            if v != v:        # nan
                return 0.0
            return 0.0 if v < 0.0 else (cap if v > cap else v)
        self.S_H = cl(self.S_H, nh); self.E_H = cl(self.E_H, nh); self.I_H = cl(self.I_H, nh)
        self.T_H = cl(self.T_H, nh); self.R_H = cl(self.R_H, nh)
        self.S_M = cl(self.S_M, nm); self.E_M = cl(self.E_M, nm); self.I_M = cl(self.I_M, nm)
DATA = "malaria-data-for-modeling-dynamics"
CAMPAIGN_DAYS = (30, 80, 130); HORIZON = 50; EPISODE_DAYS = 180
SEED_PER_CAPITA = 4e-5; REWARD_SCALE = 5.0
_CN = np.array([1, 1, 1, 15, 1e4, 1e3, 1e3, 1e3, 1e3, 3e4, 3e3, 3e3, 1, 1], np.float32)


def _steady_inc(a, C, mult):
    """Annual clinical incidence per 1000 at steady state, via the stable (clipped)
    integrator so the bisection never diverges at high biting rates."""
    rs = np.full(T_SS + AVG + 5, R_CONST)
    c = StableCounty(rs, pop_multiply=mult, vegetation_biting=a, init_coverage=C,
                     treatment_seeking=10)
    c.external_seeding = SEED_PER_CAPITA * c.N_H
    for _ in range(T_SS):
        c.step(0.0)
    inc = 0.0
    for _ in range(AVG):
        inc += c.sigma_H * c.E_H
        c.step(0.0)
    return inc / c.N_H * 1000.0


def _fit_biting(target_inc, C, mult, lo=0.02, hi=1.2, iters=30):
    if target_inc <= _steady_inc(lo, C, mult):
        return lo
    if target_inc >= _steady_inc(hi, C, mult):
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _steady_inc(mid, C, mult) < target_inc:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def select_and_calibrate(N, year=2019):
    sub = pd.read_csv(f"{DATA}/Malaria Atlas Project/Pf Subnational.csv")
    cnt = pd.read_csv(f"{DATA}/Malaria Atlas Project/Pf Subnational Counts.csv")
    itn = pd.read_csv(f"{DATA}/The Global Health Observatory/ITN Access.csv")
    sub = sub[sub.Year == year]
    piv = sub.pivot_table(index=["ISO3", "Name"], columns="Metric", values="Value").reset_index()
    piv = piv.rename(columns={"Incidence Rate": "inc"})
    cases = cnt[(cnt.Year == year) & (cnt.Metric == "Clinical Cases")][["ISO3", "Name", "Value"]]
    cases = cases.rename(columns={"Value": "cases"})
    df = piv.merge(cases, on=["ISO3", "Name"], how="left")
    df["pop"] = np.where(df.inc > 1, df.cases / (df.inc / 1000), np.nan)
    itn_y = itn[itn.IndicatorCode == "MALARIA_ITN_COVERAGE"].copy()
    itn_y["d"] = (itn_y.Period - year).abs()
    itn_b = itn_y.sort_values("d").groupby("SpatialDimValueCode").first().reset_index()
    itn_b = itn_b[["SpatialDimValueCode", "FactValueNumeric"]].rename(
        columns={"SpatialDimValueCode": "ISO3", "FactValueNumeric": "ITN"})
    df = df.merge(itn_b, on="ISO3", how="left").dropna(subset=["inc", "pop", "ITN"])
    df = df[(df.inc > 5) & (df["pop"] > 5e4)].sort_values("inc").reset_index(drop=True)
    idx = np.linspace(0, len(df) - 1, N).round().astype(int)
    sel = df.iloc[idx].reset_index(drop=True)
    mean_pop = sel["pop"].mean()
    rows = []
    for _, r in sel.iterrows():
        mult = r["pop"] / mean_pop
        a = _fit_biting(r.inc, r.ITN / 100, mult)
        rows.append(dict(biting=a, ITN=r.ITN / 100, mult=mult))
    return pd.DataFrame(rows)


class ScaleEnv(gym.Env):
    """Generalized N-region SEITR allocation env. SB3 step() takes logits; the
    raw-allocation core (step_core) is shared by all evaluation policies."""

    def __init__(self, regions, rdf, budget, seed=None):
        super().__init__()
        self.regions, self.rdf, self.budget = regions, rdf, budget
        self.N = len(regions)
        self.total_pop = float((10000 * regions.mult).sum())
        self._rng = np.random.default_rng(seed)
        self.snorm = np.tile(_CN, self.N).astype(np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.N * 14,), np.float32)
        self.action_space = spaces.Box(-5.0, 5.0, (self.N,), np.float32)
        self.center = 1.5

    def decode(self, logits):
        a = np.asarray(logits, np.float64); e = np.exp(a - a.max())
        return ((e / e.sum()) * self.budget).astype(np.float32)

    def _obs(self):
        o = []
        for c in self.counties:
            o.extend([c.multiplier, c.a, c.C, 1/c.rho, c.S_H, c.E_H, c.I_H, c.T_H, c.R_H,
                      c.S_M, c.E_M, c.I_M, c.a_t, c.R])
        return (np.array(o, np.float32) / self.snorm).astype(np.float32)

    def _day(self, alloc):
        burden = 0.0
        for i, c in enumerate(self.counties):
            c.step(alloc[i]); burden += c.E_H + c.I_H
        return burden / self.total_pop

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.counties = []
        for _, r in self.regions.iterrows():
            row = self.rdf.iloc[self._rng.integers(0, len(self.rdf))]
            c = StableCounty(row.to_numpy(), pop_multiply=r.mult, vegetation_biting=r.biting,
                             init_coverage=r.ITN, treatment_seeking=10)
            c.external_seeding = SEED_PER_CAPITA * c.N_H
            c.reset()
            self.counties.append(c)
        self.scale = sum(self._day(np.zeros(self.N, np.float32)) for _ in range(CAMPAIGN_DAYS[0]))
        self.k = 0
        return self._obs(), {}

    def step_core(self, raw_alloc):
        """Advance one campaign with a RAW net-allocation vector."""
        burden = 0.0
        for d in range(HORIZON):
            burden += self._day(raw_alloc if d == 0 else np.zeros(self.N, np.float32))
        ratio = burden / self.scale; self.k += 1
        return self._obs(), float(REWARD_SCALE * (self.center - ratio)), self.k >= 3, False, {"ratio": ratio}

    def step(self, action):
        return self.step_core(self.decode(action))


def rollout(env, raw_alloc_fn):
    obs, _ = env.reset(); tot = 0.0
    for _ in range(3):
        obs, r, done, _, info = env.step_core(raw_alloc_fn(obs, env))
        tot += info["ratio"]
    return tot


def eval_raw(env_fn, raw_fn, n=30, base=10000):
    return np.array([rollout(env_fn(base + i), raw_fn) for i in range(n)])


def main():
    rdf = load_resistance_trajectories()
    results = []
    for N in [8, 20, 50]:
        print(f"\n===== N = {N} regions =====", flush=True)
        t_sel = time.time()
        regions = select_and_calibrate(N)
        budget = 2500.0 * N
        def env_fn(seed):
            return ScaleEnv(regions, rdf, budget, seed=seed)

        baseline = eval_raw(env_fn, lambda o, e: np.zeros(e.N, np.float32))
        b = baseline.mean()
        center = float(b / 3)
        pop = regions.mult.to_numpy()
        static_raw = (pop / pop.sum() * budget).astype(np.float32)
        static = eval_raw(env_fn, lambda o, e: static_raw)

        t0 = time.time()
        tenv = ScaleEnv(regions, rdf, budget, seed=0); tenv.center = center
        model = PPO("MlpPolicy", Monitor(tenv), seed=0, verbose=0, n_steps=600, batch_size=200,
                    gamma=0.99, ent_coef=0.0, learning_rate=3e-4,
                    policy_kwargs=dict(net_arch=[128, 128]))
        model.learn(18000)
        ttime = time.time() - t0

        ppo = eval_raw(env_fn, lambda o, e: e.decode(model.predict(o, deterministic=True)[0]))
        rs = (b - static.mean()) / b * 100
        rp = (b - ppo.mean()) / b * 100
        print(f"  baseline {b:.3f} | static {rs:.1f}% | PPO {rp:.1f}% "
              f"| PPO beats static in {(ppo < static).mean()*100:.0f}% "
              f"| train {ttime:.0f}s | calib {t0-t_sel:.0f}s", flush=True)
        results.append(dict(N=N, baseline=round(b, 3), static_red=round(rs, 1),
                            ppo_red=round(rp, 1), ppo_beats_static_pct=round((ppo < static).mean()*100),
                            train_time_s=round(ttime), calib_time_s=round(t0 - t_sel)))
    out = pd.DataFrame(results)
    out.to_csv("scale_results.csv", index=False)
    print("\n" + out.to_string())
    print("saved -> scale_results.csv")


if __name__ == "__main__":
    main()

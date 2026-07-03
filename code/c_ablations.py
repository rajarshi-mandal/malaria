"""
Item C: real environment-component ablation study (replaces the manuscript's
unverifiable Table IV).

On the calibrated 8-region env we toggle each modeling component and measure, with
multi-seed held-out evaluation, how it affects (a) the achievable reduction and
(b) the learned-vs-static ADVANTAGE (PPO reduction minus static reduction). This
shows which components matter and that the core finding is robust to them.

Ablations: full | no insecticide resistance | no seasonality | no ITN decay |
deterministic resistance (no stochastic draw).

Output: console table + ablation_results.csv
"""
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from sar_sequential import MalariaCounty, load_resistance_trajectories

torch.set_num_threads(1)
REGIONS = pd.read_csv("calibrated_regions.csv")
N = len(REGIONS)
CAMPAIGN_DAYS = (30, 80, 130); HORIZON = 50; SEED_PER_CAPITA = 4e-5; REWARD_SCALE = 5.0
BUDGET = 20000.0
_CN = np.array([1, 1, 1, 15, 1e4, 1e3, 1e3, 1e3, 1e3, 3e4, 3e3, 3e3, 1, 1], np.float32)
_SN = np.tile(_CN, N).astype(np.float32)
_mult = REGIONS.pop_multiplier.to_numpy()
POP_STATIC = (_mult / _mult.sum() * BUDGET).astype(np.float32)
TOTAL_POP = float((10000 * _mult).sum())

ABLATIONS = {
    "full": {},
    "no_resistance": {"resistance": False},
    "no_seasonality": {"seasonality": False},
    "no_itn_decay": {"decay": False},
    "deterministic": {"deterministic": True},
}


class AblEnv(gym.Env):
    def __init__(self, toggles, rdf, seed=None):
        super().__init__()
        self.tg, self.rdf = toggles, rdf
        self._rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(-np.inf, np.inf, (N * 14,), np.float32)
        self.action_space = spaces.Box(-5.0, 5.0, (N,), np.float32)
        self.center = 1.5

    def decode(self, logits):
        a = np.asarray(logits, np.float64); e = np.exp(a - a.max())
        return ((e / e.sum()) * BUDGET).astype(np.float32)

    def _obs(self):
        o = []
        for c in self.counties:
            o.extend([c.multiplier, c.a, c.C, 1/c.rho, c.S_H, c.E_H, c.I_H, c.T_H, c.R_H,
                      c.S_M, c.E_M, c.I_M, c.a_t, c.R])
        return (np.array(o, np.float32) / _SN).astype(np.float32)

    def _day(self, alloc):
        burden = 0.0
        for i, c in enumerate(self.counties):
            c.step(alloc[i]); burden += c.E_H + c.I_H
        return burden / TOTAL_POP

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.counties = []
        det_row = self.rdf.iloc[0].to_numpy()
        for _, r in REGIONS.iterrows():
            if self.tg.get("deterministic"):
                row = det_row
            else:
                row = self.rdf.iloc[self._rng.integers(0, len(self.rdf))].to_numpy()
            c = MalariaCounty(row, pop_multiply=r.pop_multiplier, vegetation_biting=r.biting_rate,
                              init_coverage=r.ITN_coverage, treatment_seeking=10)
            c.external_seeding = SEED_PER_CAPITA * c.N_H
            if self.tg.get("resistance", True) is False:
                c.Rs = np.zeros(len(row))
            if self.tg.get("seasonality", True) is False:
                c.seasonal_amp = 0.0
            if self.tg.get("decay", True) is False:
                c.delta = 0.0
            c.reset()
            self.counties.append(c)
        self.scale = sum(self._day(np.zeros(N, np.float32)) for _ in range(CAMPAIGN_DAYS[0]))
        self.k = 0
        return self._obs(), {}

    def step_core(self, raw):
        burden = 0.0
        for d in range(HORIZON):
            burden += self._day(raw if d == 0 else np.zeros(N, np.float32))
        ratio = burden / self.scale; self.k += 1
        return self._obs(), float(REWARD_SCALE*(self.center-ratio)), self.k >= 3, False, {"ratio": ratio}

    def step(self, action):
        return self.step_core(self.decode(action))


def rollout(env, raw_fn):
    obs, _ = env.reset(); tot = 0.0
    for _ in range(3):
        obs, r, done, _, info = env.step_core(raw_fn(obs, env)); tot += info["ratio"]
    return tot


def eval_raw(toggles, rdf, raw_fn, n=40, base=10000):
    return np.array([rollout(AblEnv(toggles, rdf, seed=base+i), raw_fn) for i in range(n)])


def main():
    rdf = load_resistance_trajectories()
    seeds = [0, 1, 2]
    rows = []
    for name, tg in ABLATIONS.items():
        red_static, red_ppo, adv = [], [], []
        baseline = eval_raw(tg, rdf, lambda o, e: np.zeros(N, np.float32))
        b = baseline.mean()
        center = float(b / 3)
        static = eval_raw(tg, rdf, lambda o, e: POP_STATIC)
        rs = (b - static.mean()) / b * 100
        for sd in seeds:
            tenv = AblEnv(tg, rdf, seed=sd); tenv.center = center
            m = PPO("MlpPolicy", Monitor(tenv), seed=sd, verbose=0, n_steps=600, batch_size=200,
                    gamma=0.99, ent_coef=0.0, learning_rate=3e-4, policy_kwargs=dict(net_arch=[128, 128]))
            m.learn(35000)
            ppo = eval_raw(tg, rdf, lambda o, e: e.decode(m.predict(o, deterministic=True)[0]))
            rp = (b - ppo.mean()) / b * 100
            red_ppo.append(rp); adv.append(rp - rs)
        rows.append(dict(ablation=name, baseline=round(b, 3), static_red=round(rs, 2),
                         ppo_red_mean=round(np.mean(red_ppo), 2), ppo_red_std=round(np.std(red_ppo), 2),
                         advantage_mean=round(np.mean(adv), 2), advantage_std=round(np.std(adv), 2)))
        print(f"{name:16s} | static {rs:5.1f}% | PPO {np.mean(red_ppo):5.1f}+/-{np.std(red_ppo):.1f}% "
              f"| advantage {np.mean(adv):+5.2f}+/-{np.std(adv):.2f} pp", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("ablation_results.csv", index=False)
    print("\n" + out.to_string())
    print("saved -> ablation_results.csv")


if __name__ == "__main__":
    main()

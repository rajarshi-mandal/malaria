"""
Item D, steps 4-5: real-data-calibrated allocation environment + full comparison.

The five fictional counties are replaced by EIGHT real admin1 regions spanning a
transmission gradient (Dar-es-Salaam 36 -> Tshuapa DRC 455 cases/1000/yr), each
parameterized from data: biting rate fitted to observed incidence (calibrate.py),
ITN coverage from WHO GHO, population derived from MAP case counts.

We re-run the Item B comparison (baseline / static / contextual-bandit / DQN /
PPO / QT-Opt) on this calibrated environment, with the same honest held-out
protocol, multi-seed training, and paired significance tests.

Run: python calibrated_experiment.py --seeds 0 1 2 --episodes 1000
"""
import argparse
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from scipy import stats

from sar_sequential import MalariaCounty, load_resistance_trajectories, INPUT_NORM as _N5

torch.set_num_threads(1)

REGIONS = pd.read_csv("calibrated_regions.csv")
N = len(REGIONS)
FEATS = 14
STATE_DIM = N * FEATS
ACTION_DIM = N
INPUT_DIM = STATE_DIM + ACTION_DIM
CAMPAIGN_DAYS = (30, 80, 130)
HORIZON = 50
EPISODE_DAYS = 180
SEED_PER_CAPITA = 4e-5
N_CAND = 128
N_MAX = 32
REWARD_SCALE = 5.0

# per-region normalizer (same 14-feature template used for the 5-county model)
_COUNTY_NORM = _N5[:FEATS]
BUDGET = 20000.0                       # ITNs per campaign (~0.25 x total pop)
_mult = REGIONS.pop_multiplier.to_numpy()
POP_STATIC = (_mult / _mult.sum() * BUDGET).astype(np.float32)   # population-proportional
TOTAL_POP = float((10000 * _mult).sum())


# --------------------------------------------------------------------------- #
# Calibrated environment                                                      #
# --------------------------------------------------------------------------- #
class CalibratedEnv:
    def __init__(self, resistance_df, rng):
        self.counties = []
        for _, r in REGIONS.iterrows():
            row = resistance_df.iloc[rng.integers(0, len(resistance_df))]
            c = MalariaCounty(row.to_numpy(), pop_multiply=r.pop_multiplier,
                              vegetation_biting=r.biting_rate, init_coverage=r.ITN_coverage,
                              treatment_seeking=int(r.treatment_seeking))
            c.external_seeding = SEED_PER_CAPITA * c.N_H
            self.counties.append(c)

    def reset(self):
        for c in self.counties:
            c.reset()
        return self._obs()

    def _obs(self):
        obs = []
        for c in self.counties:
            obs.extend(c.static_features() +
                       [c.S_H, c.E_H, c.I_H, c.T_H, c.R_H,
                        c.S_M, c.E_M, c.I_M, c.a_t, c.R])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        burden = 0.0
        for i, c in enumerate(self.counties):
            c.step(action[i])
            burden += c.E_H + c.I_H
        return self._obs(), burden / TOTAL_POP


def run_episode(env, policy, collect=False, center=1.6, scale=REWARD_SCALE):
    obs = env.reset()
    window = [0.0, 0.0, 0.0, 0.0]
    idx = 0
    dstates = [None] * 3
    dacts = [None] * 3
    last = obs
    for day in range(EPISODE_DAYS):
        if day in CAMPAIGN_DAYS:
            dstates[idx] = last
            a = policy(last, idx)
            dacts[idx] = a
            idx += 1
        else:
            a = np.zeros(ACTION_DIM, dtype=np.float32)
        obs, burden = env.step(a)
        window[0 if day < CAMPAIGN_DAYS[0] else idx] += burden
        last = obs
    sc = window[0]
    ratios = [window[k + 1] / sc for k in range(3)]
    cum = float(sum(ratios))
    trans = []
    if collect:
        for k in range(3):
            r = scale * (center - ratios[k])
            trans.append((dstates[k], dacts[k], r, dstates[k + 1] if k < 2 else dstates[k], k == 2))
    return cum, trans, dacts


def baseline_policy(s, k):
    return np.zeros(ACTION_DIM, dtype=np.float32)


def static_policy(s, k):
    return POP_STATIC.copy()


# --------------------------------------------------------------------------- #
# QT-Opt / bandit agent (N-region, CEM action search)                         #
# --------------------------------------------------------------------------- #
class QNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        norm = np.concatenate([np.tile(_COUNTY_NORM, N), [BUDGET] * ACTION_DIM]).astype(np.float32)
        self.register_buffer("norm", torch.tensor(norm))
        self.fc1 = nn.Linear(INPUT_DIM, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x / self.norm
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return self.out(x)


class QTOpt:
    def __init__(self, gamma=0.99, lr=1e-3, tau=0.01, batch=128, seed=0,
                 cem_pop=96, cem_iters=3, cem_elite=12, alpha=1.0):
        self.gamma, self.tau, self.batch = gamma, tau, batch
        self.cem_pop, self.cem_iters, self.cem_elite, self.alpha = cem_pop, cem_iters, cem_elite, alpha
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.online, self.target = QNet(), QNet()
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.mem = deque(maxlen=10000)

    def _q(self, obs, cands):
        o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).expand(len(cands), -1)
        c = torch.tensor(cands, dtype=torch.float32)
        with torch.no_grad():
            return self.online(torch.cat([o, c], 1)).squeeze(1).numpy()

    def _cem(self, obs):
        alpha = np.full(ACTION_DIM, self.alpha)
        best_a, best_q = None, -np.inf
        for it in range(self.cem_iters):
            cand = (self.rng.dirichlet(alpha, size=self.cem_pop) * BUDGET).astype(np.float32)
            q = self._q(obs, cand)
            elite = np.argsort(q)[-self.cem_elite:]
            mean_p = (cand[elite] / BUDGET).mean(0)
            alpha = np.clip(mean_p * 10.0 * (it + 1), 1e-2, None)
            if q[elite[-1]] > best_q:
                best_q, best_a = q[elite[-1]], cand[elite[-1]]
        return best_a

    def select(self, obs, eps):
        if self.rng.random() < eps:
            return (self.rng.dirichlet(np.ones(ACTION_DIM)) * BUDGET).astype(np.float32)
        return self._cem(obs)

    def remember(self, *tr):
        self.mem.append(tr)

    def _max_next(self, ns):
        B = ns.shape[0]
        cand = self.rng.dirichlet(np.ones(ACTION_DIM), size=(B, N_MAX)) * BUDGET
        ct = torch.tensor(cand, dtype=torch.float32)
        no = ns.unsqueeze(1).expand(B, N_MAX, STATE_DIM)
        inp = torch.cat([no, ct], 2).reshape(B * N_MAX, INPUT_DIM)
        with torch.no_grad():
            qt = self.target(inp).reshape(B, N_MAX)
            qo = self.online(inp).reshape(B, N_MAX)
            return qt[torch.arange(B), qo.argmax(1)]

    def train_step(self):
        if len(self.mem) < self.batch:
            return
        s, a, r, s2, d = zip(*random.sample(self.mem, self.batch))
        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(np.array(a), dtype=torch.float32)
        r = torch.tensor(np.array(r), dtype=torch.float32)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32)
        d = torch.tensor(np.array(d), dtype=torch.float32)
        tgt = r + self.gamma * (1 - d) * self._max_next(s2)
        pred = self.online(torch.cat([s, a], 1)).squeeze(1)
        loss = self.loss_fn(pred, tgt)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        with torch.no_grad():
            for tp, op in zip(self.target.parameters(), self.online.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * op)


def train_qtopt(rdf, gamma, episodes, seed, center):
    rng = np.random.default_rng(seed)
    agent = QTOpt(gamma=gamma, seed=seed)
    eps, emin, edec = 1.0, 0.05, 0.996
    hist = []
    for ep in range(episodes):
        env = CalibratedEnv(rdf, rng)
        cum, trans, _ = run_episode(env, lambda o, k: agent.select(o, eps),
                                    collect=True, center=center)
        for tr in trans:
            agent.remember(*tr); agent.train_step()
        if eps > emin:
            eps *= edec
        hist.append(cum)
    agent.history = hist
    return agent


# --------------------------------------------------------------------------- #
# SB3 wrapper (3-step) + DQN action library                                   #
# --------------------------------------------------------------------------- #
_SNORM = np.tile(_COUNTY_NORM, N).astype(np.float32)


def norm_state(s):
    return (np.asarray(s, np.float32) / _SNORM).astype(np.float32)


def decode_cont(a):
    a = np.asarray(a, np.float64); e = np.exp(a - a.max())
    return ((e / e.sum()) * BUDGET).astype(np.float32)


def action_library(k=80, seed=0):
    rng = np.random.default_rng(seed)
    lib = [np.eye(ACTION_DIM, dtype=np.float32)[i] * BUDGET for i in range(ACTION_DIM)]
    lib.append(np.full(ACTION_DIM, BUDGET / ACTION_DIM, np.float32))
    lib.append(POP_STATIC.astype(np.float32))
    while len(lib) < k:
        al = rng.choice([0.3, 0.5, 1.0])
        lib.append((rng.dirichlet(np.full(ACTION_DIM, al)) * BUDGET).astype(np.float32))
    return np.array(lib[:k], np.float32)


class GymWrap(gym.Env):
    def __init__(self, rdf, center, discrete=False, lib=None, seed=None):
        super().__init__()
        self.rdf, self.center, self.discrete, self.lib = rdf, center, discrete, lib
        self._rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32)
        self.action_space = spaces.Discrete(len(lib)) if discrete else \
            spaces.Box(-5.0, 5.0, (ACTION_DIM,), np.float32)

    def _dec(self, a):
        return self.lib[int(a)].astype(np.float32) if self.discrete else decode_cont(a)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env = CalibratedEnv(self.rdf, self._rng)
        self.env.reset()
        self.scale = 0.0
        obs = None
        for _ in range(CAMPAIGN_DAYS[0]):
            obs, b = self.env.step(np.zeros(ACTION_DIM, np.float32))
            self.scale += b
        self.k = 0
        return norm_state(obs), {}

    def step(self, action):
        alloc = self._dec(action)
        burden, obs = 0.0, None
        for d in range(HORIZON):
            obs, b = self.env.step(alloc if d == 0 else np.zeros(ACTION_DIM, np.float32))
            burden += b
        ratio = burden / self.scale
        self.k += 1
        return norm_state(obs), float(REWARD_SCALE * (self.center - ratio)), self.k >= 3, False, {}


def sb3_pol(model, discrete, lib=None):
    def pol(s, k):
        a, _ = model.predict(norm_state(s), deterministic=True)
        return lib[int(a)].astype(np.float32) if discrete else decode_cont(a)
    return pol


def train_ppo(rdf, center, steps, seed):
    env = Monitor(GymWrap(rdf, center, discrete=False, seed=seed))
    m = PPO("MlpPolicy", env, seed=seed, verbose=0, n_steps=600, batch_size=200,
            gamma=0.99, ent_coef=0.0, learning_rate=3e-4,
            policy_kwargs=dict(net_arch=[128, 128]))
    m.learn(steps); return m


def train_dqn(rdf, center, steps, seed, lib):
    env = Monitor(GymWrap(rdf, center, discrete=True, lib=lib, seed=seed))
    m = DQN("MlpPolicy", env, seed=seed, verbose=0, learning_rate=1e-3, buffer_size=50000,
            learning_starts=1000, batch_size=128, gamma=0.99, train_freq=1,
            target_update_interval=500, exploration_fraction=0.5,
            exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[128, 128]))
    m.learn(steps); return m


# --------------------------------------------------------------------------- #
# Evaluation + driver                                                         #
# --------------------------------------------------------------------------- #
def evaluate(policies, rdf, n_eval, base_seed=10000):
    out = {n: [] for n in policies}
    for i in range(n_eval):
        seed = base_seed + i
        for n_, pol in policies.items():
            env = CalibratedEnv(rdf, np.random.default_rng(seed))
            cum, _, _ = run_episode(env, pol)
            out[n_].append(cum)
    return {k: np.array(v) for k, v in out.items()}


def measure_center(rdf, n=10):
    rng = np.random.default_rng(777)
    vals = []
    for _ in range(n):
        env = CalibratedEnv(rdf, rng)
        cum, _, _ = run_episode(env, baseline_policy)
        vals.append(cum)
    return float(np.mean(vals) / 3.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--ppo-steps", type=int, default=60000)
    p.add_argument("--dqn-steps", type=int, default=60000)
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--out", type=str, default="calibrated_results.npz")
    args = p.parse_args()

    rdf = load_resistance_trajectories()
    center = measure_center(rdf)
    lib = action_library()
    print(f"N={N} regions | BUDGET={BUDGET:.0f}/campaign | reward_center={center:.3f}", flush=True)

    methods = ["baseline", "static", "bandit", "DQN", "PPO", "QT-Opt"]
    ratio = {m: [] for m in methods}
    last_eval, last_curves = None, {}
    for seed in args.seeds:
        print(f"\n--- seed {seed} ---", flush=True)
        qt = train_qtopt(rdf, 0.99, args.episodes, seed, center); print("  QT-Opt done", flush=True)
        bd = train_qtopt(rdf, 0.0, args.episodes, seed, center); print("  bandit done", flush=True)
        ppo = train_ppo(rdf, center, args.ppo_steps, seed); print("  PPO done", flush=True)
        dqn = train_dqn(rdf, center, args.dqn_steps, seed, lib); print("  DQN done", flush=True)
        pols = {"baseline": baseline_policy, "static": static_policy,
                "bandit": lambda s, k: bd.select(s, 0.0),
                "DQN": sb3_pol(dqn, True, lib), "PPO": sb3_pol(ppo, False),
                "QT-Opt": lambda s, k: qt.select(s, 0.0)}
        res = evaluate(pols, rdf, args.n_eval)
        for m in methods:
            ratio[m].append(res[m].mean())
        last_eval = res
        last_curves = {"QT-Opt": np.array(qt.history), "bandit": np.array(bd.history)}
        b = res["baseline"].mean()
        print("  " + " | ".join(f"{m} {(b-res[m].mean())/b*100:.1f}%" for m in methods), flush=True)

    b = np.mean(ratio["baseline"])
    print("\n========= ITEM D: CALIBRATED HELD-OUT COMPARISON =========")
    print(f"({len(args.seeds)} seeds, {args.n_eval} held-out envs, {N} real regions)\n")
    print(f"{'Method':10s} | {'Inf.ratio':>14s} | {'% reduction':>14s}")
    print("-" * 44)
    for m in methods:
        r = np.array(ratio[m]); red = (b - r.mean()) / b * 100
        print(f"{m:10s} | {r.mean():7.3f}+/-{r.std():.3f} | {red:7.2f}%")
    # significance vs static on held-out envs (last seed)
    print("\n--- Wilcoxon vs static (held-out, last seed) ---")
    for m in ["bandit", "DQN", "PPO", "QT-Opt"]:
        try:
            _, pv = stats.wilcoxon(last_eval[m], last_eval["static"])
        except ValueError:
            pv = float("nan")
        better = (last_eval[m] < last_eval["static"]).mean() * 100
        print(f"{m:8s} beats static in {better:.0f}% of envs | p={pv:.2e}")
    np.savez(args.out, ratio={m: np.array(ratio[m]) for m in methods},
             last_eval=last_eval, curves=last_curves, center=center,
             regions=REGIONS.region.tolist())
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()

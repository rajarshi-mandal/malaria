"""
Item E, step 2: allocation comparison on the calibrated SEITAR environment.

Same six allocators as Item B/D, now on the extended model with an asymptomatic
reservoir (300-day carriage). Central question: does the slow reservoir -- which
makes an ITN allocation's benefit persist for many months beyond the 50-day
reward window -- finally create delayed dynamics where the SEQUENTIAL learner
(QT-Opt, bootstrapped) beats the MYOPIC one-step bandit?

Run: python extended_experiment.py --seeds 0 1 2 --episodes 1000
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

from extended_ode import ExtendedCounty

torch.set_num_threads(1)

REGIONS = pd.read_csv("extended_calibrated_regions.csv")
N = len(REGIONS)
FEATS = 15
STATE_DIM = N * FEATS
ACTION_DIM = N
INPUT_DIM = STATE_DIM + ACTION_DIM
CAMPAIGN_DAYS = (30, 80, 130)
HORIZON = 50
EPISODE_DAYS = 180
N_CAND = 128
N_MAX = 32
REWARD_SCALE = 5.0
BUDGET = 20000.0

# per-region feature normalizer (15 feats)
# [pop_mult, biting, ITN_cov, treat, S,E,I,T,A,R, S_M,E_M,I_M, a_t, Rresist]
_FNORM = np.array([1, 1, 1, 15, 1e4, 1e3, 1e3, 1e3, 5e3, 1e3,
                   3e4, 3e3, 3e3, 1, 1], dtype=np.float32)
_mult = REGIONS.pop_multiplier.to_numpy()
POP_STATIC = (_mult / _mult.sum() * BUDGET).astype(np.float32)
TOTAL_POP = float((10000 * _mult).sum())


_STEADY = None


def get_steady():
    """Per-region endemic steady-state compartments (computed once, cached).

    Episodes are initialized here so the slow asymptomatic reservoir starts at its
    realistic calibrated level rather than at arbitrary values it could never reach
    within a 180-day episode."""
    global _STEADY
    if _STEADY is None:
        _STEADY = []
        for _, r in REGIONS.iterrows():
            c = ExtendedCounty(r.biting_rate, r.ITN_coverage, r.pop_multiplier,
                               int(r.treatment_seeking), r.p_sympt)
            for _ in range(2500):
                c.step(0.0, seasonal=False)
            _STEADY.append((c.S_H, c.E_H, c.I_H, c.T_H, c.A_H, c.R_H,
                            c.S_M, c.E_M, c.I_M))
    return _STEADY


class ExtEnv:
    def __init__(self, rng):
        self.counties = []
        for _, r in REGIONS.iterrows():
            self.counties.append(ExtendedCounty(
                biting=r.biting_rate, coverage=r.ITN_coverage,
                multiplier=r.pop_multiplier, treat=int(r.treatment_seeking),
                p_sympt=r.p_sympt))
        self.rng = rng

    def reset(self):
        for c, st in zip(self.counties, get_steady()):
            c.reset()                       # zero t/tau, baseline coverage
            (c.S_H, c.E_H, c.I_H, c.T_H, c.A_H, c.R_H,
             c.S_M, c.E_M, c.I_M) = st       # start at calibrated endemic state
        return self._obs()

    def _obs(self):
        obs = []
        for c in self.counties:
            obs.extend([c.multiplier, c.a, c.C, 1.0 / c.rho,
                        c.S_H, c.E_H, c.I_H, c.T_H, c.A_H, c.R_H,
                        c.S_M, c.E_M, c.I_M, c.a_t, c.R])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        burden = 0.0
        for i, c in enumerate(self.counties):
            c.step(action[i], seasonal=True)
            burden += c.E_H + c.I_H
        return self._obs(), burden / TOTAL_POP


def run_episode(env, policy, collect=False, center=1.6, scale=REWARD_SCALE):
    obs = env.reset()
    window = [0.0, 0.0, 0.0, 0.0]
    idx = 0
    ds, da = [None]*3, [None]*3
    last = obs
    for day in range(EPISODE_DAYS):
        if day in CAMPAIGN_DAYS:
            ds[idx] = last
            a = policy(last, idx)
            da[idx] = a
            idx += 1
        else:
            a = np.zeros(ACTION_DIM, dtype=np.float32)
        obs, burden = env.step(a)
        window[0 if day < CAMPAIGN_DAYS[0] else idx] += burden
        last = obs
    sc = window[0]
    ratios = [window[k+1]/sc for k in range(3)]
    cum = float(sum(ratios))
    trans = []
    if collect:
        for k in range(3):
            r = scale*(center - ratios[k])
            trans.append((ds[k], da[k], r, ds[k+1] if k < 2 else ds[k], k == 2))
    return cum, trans, da


def baseline_policy(s, k):
    return np.zeros(ACTION_DIM, dtype=np.float32)


def static_policy(s, k):
    return POP_STATIC.copy()


# ---- QT-Opt / bandit (CEM) ----
class QNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        norm = np.concatenate([np.tile(_FNORM, N), [BUDGET]*ACTION_DIM]).astype(np.float32)
        self.register_buffer("norm", torch.tensor(norm))
        self.fc1 = nn.Linear(INPUT_DIM, hidden); self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x / self.norm
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return self.out(x)


class QTOpt:
    def __init__(self, gamma=0.99, lr=1e-3, tau=0.01, batch=128, seed=0,
                 cem_pop=96, cem_iters=3, cem_elite=12):
        self.gamma, self.tau, self.batch = gamma, tau, batch
        self.cp, self.ci, self.ce = cem_pop, cem_iters, cem_elite
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.online, self.target = QNet(), QNet()
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(); self.mem = deque(maxlen=10000)

    def _q(self, obs, cands):
        o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).expand(len(cands), -1)
        c = torch.tensor(cands, dtype=torch.float32)
        with torch.no_grad():
            return self.online(torch.cat([o, c], 1)).squeeze(1).numpy()

    def _cem(self, obs):
        alpha = np.ones(ACTION_DIM)
        ba, bq = None, -np.inf
        for it in range(self.ci):
            cand = (self.rng.dirichlet(alpha, size=self.cp) * BUDGET).astype(np.float32)
            q = self._q(obs, cand)
            el = np.argsort(q)[-self.ce:]
            alpha = np.clip((cand[el]/BUDGET).mean(0)*10*(it+1), 1e-2, None)
            if q[el[-1]] > bq:
                bq, ba = q[el[-1]], cand[el[-1]]
        return ba

    def select(self, obs, eps):
        if self.rng.random() < eps:
            return (self.rng.dirichlet(np.ones(ACTION_DIM))*BUDGET).astype(np.float32)
        return self._cem(obs)

    def remember(self, *tr):
        self.mem.append(tr)

    def _max_next(self, ns):
        B = ns.shape[0]
        cand = self.rng.dirichlet(np.ones(ACTION_DIM), size=(B, N_MAX))*BUDGET
        ct = torch.tensor(cand, dtype=torch.float32)
        no = ns.unsqueeze(1).expand(B, N_MAX, STATE_DIM)
        inp = torch.cat([no, ct], 2).reshape(B*N_MAX, INPUT_DIM)
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
        tgt = r + self.gamma*(1-d)*self._max_next(s2)
        pred = self.online(torch.cat([s, a], 1)).squeeze(1)
        loss = self.loss_fn(pred, tgt)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        with torch.no_grad():
            for tp, op in zip(self.target.parameters(), self.online.parameters()):
                tp.mul_(1-self.tau).add_(self.tau*op)


def train_qtopt(gamma, episodes, seed, center):
    rng = np.random.default_rng(seed)
    agent = QTOpt(gamma=gamma, seed=seed)
    eps = 1.0
    hist = []
    for ep in range(episodes):
        env = ExtEnv(rng)
        cum, trans, _ = run_episode(env, lambda o, k: agent.select(o, eps),
                                    collect=True, center=center)
        for tr in trans:
            agent.remember(*tr); agent.train_step()
        if eps > 0.05:
            eps *= 0.996
        hist.append(cum)
    agent.history = hist
    return agent


# ---- SB3 wrapper ----
_SNORM = np.tile(_FNORM, N).astype(np.float32)


def norm_state(s):
    return (np.asarray(s, np.float32)/_SNORM).astype(np.float32)


def decode_cont(a):
    a = np.asarray(a, np.float64); e = np.exp(a-a.max())
    return ((e/e.sum())*BUDGET).astype(np.float32)


def action_library(k=80, seed=0):
    rng = np.random.default_rng(seed)
    lib = [np.eye(ACTION_DIM, dtype=np.float32)[i]*BUDGET for i in range(ACTION_DIM)]
    lib.append(np.full(ACTION_DIM, BUDGET/ACTION_DIM, np.float32))
    lib.append(POP_STATIC.astype(np.float32))
    while len(lib) < k:
        lib.append((rng.dirichlet(np.full(ACTION_DIM, rng.choice([0.3, 0.5, 1.0])))*BUDGET).astype(np.float32))
    return np.array(lib[:k], np.float32)


class GymWrap(gym.Env):
    def __init__(self, center, discrete=False, lib=None, seed=None):
        super().__init__()
        self.center, self.discrete, self.lib = center, discrete, lib
        self._rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32)
        self.action_space = spaces.Discrete(len(lib)) if discrete else \
            spaces.Box(-5.0, 5.0, (ACTION_DIM,), np.float32)

    def _dec(self, a):
        return self.lib[int(a)].astype(np.float32) if self.discrete else decode_cont(a)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env = ExtEnv(self._rng); self.env.reset()
        self.scale = 0.0; obs = None
        for _ in range(CAMPAIGN_DAYS[0]):
            obs, b = self.env.step(np.zeros(ACTION_DIM, np.float32)); self.scale += b
        self.k = 0
        return norm_state(obs), {}

    def step(self, action):
        alloc = self._dec(action); burden, obs = 0.0, None
        for d in range(HORIZON):
            obs, b = self.env.step(alloc if d == 0 else np.zeros(ACTION_DIM, np.float32)); burden += b
        ratio = burden/self.scale; self.k += 1
        return norm_state(obs), float(REWARD_SCALE*(self.center-ratio)), self.k >= 3, False, {}


def sb3_pol(model, discrete, lib=None):
    def pol(s, k):
        a, _ = model.predict(norm_state(s), deterministic=True)
        return lib[int(a)].astype(np.float32) if discrete else decode_cont(a)
    return pol


def train_ppo(center, steps, seed):
    env = Monitor(GymWrap(center, discrete=False, seed=seed))
    m = PPO("MlpPolicy", env, seed=seed, verbose=0, n_steps=600, batch_size=200,
            gamma=0.99, ent_coef=0.0, learning_rate=3e-4, policy_kwargs=dict(net_arch=[128, 128]))
    m.learn(steps); return m


def train_dqn(center, steps, seed, lib):
    env = Monitor(GymWrap(center, discrete=True, lib=lib, seed=seed))
    m = DQN("MlpPolicy", env, seed=seed, verbose=0, learning_rate=1e-3, buffer_size=50000,
            learning_starts=1000, batch_size=128, gamma=0.99, train_freq=1,
            target_update_interval=500, exploration_fraction=0.5, exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[128, 128]))
    m.learn(steps); return m


def evaluate(policies, n_eval, base_seed=10000):
    out = {n: [] for n in policies}
    for i in range(n_eval):
        seed = base_seed + i
        for n_, pol in policies.items():
            env = ExtEnv(np.random.default_rng(seed))
            cum, _, _ = run_episode(env, pol)
            out[n_].append(cum)
    return {k: np.array(v) for k, v in out.items()}


def measure_center(n=10):
    rng = np.random.default_rng(777); vals = []
    for _ in range(n):
        cum, _, _ = run_episode(ExtEnv(rng), baseline_policy)
        vals.append(cum)
    return float(np.mean(vals)/3.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--ppo-steps", type=int, default=60000)
    p.add_argument("--dqn-steps", type=int, default=60000)
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--out", type=str, default="extended_results.npz")
    args = p.parse_args()

    center = measure_center()
    lib = action_library()
    print(f"SEITAR env | N={N} | BUDGET={BUDGET:.0f} | center={center:.3f}", flush=True)
    methods = ["baseline", "static", "bandit", "DQN", "PPO", "QT-Opt"]
    ratio = {m: [] for m in methods}
    last_eval, last_curves = None, {}
    for seed in args.seeds:
        print(f"\n--- seed {seed} ---", flush=True)
        qt = train_qtopt(0.99, args.episodes, seed, center); print("  QT-Opt done", flush=True)
        bd = train_qtopt(0.0, args.episodes, seed, center); print("  bandit done", flush=True)
        ppo = train_ppo(center, args.ppo_steps, seed); print("  PPO done", flush=True)
        dqn = train_dqn(center, args.dqn_steps, seed, lib); print("  DQN done", flush=True)
        pols = {"baseline": baseline_policy, "static": static_policy,
                "bandit": lambda s, k: bd.select(s, 0.0),
                "DQN": sb3_pol(dqn, True, lib), "PPO": sb3_pol(ppo, False),
                "QT-Opt": lambda s, k: qt.select(s, 0.0)}
        res = evaluate(pols, args.n_eval)
        for m in methods:
            ratio[m].append(res[m].mean())
        last_eval = res
        last_curves = {"QT-Opt": np.array(qt.history), "bandit": np.array(bd.history)}
        b = res["baseline"].mean()
        print("  " + " | ".join(f"{m} {(b-res[m].mean())/b*100:.1f}%" for m in methods), flush=True)

    b = np.mean(ratio["baseline"])
    print("\n===== ITEM E: SEITAR (asymptomatic reservoir) COMPARISON =====")
    print(f"({len(args.seeds)} seeds, {args.n_eval} held-out envs)\n")
    print(f"{'Method':10s} | {'Inf.ratio':>14s} | {'% reduction':>12s}")
    print("-" * 42)
    for m in methods:
        r = np.array(ratio[m])
        print(f"{m:10s} | {r.mean():7.3f}+/-{r.std():.3f} | {(b-r.mean())/b*100:7.2f}%")
    print("\n--- Sequential vs myopic (key test) ---")
    qa, ba_ = last_eval["QT-Opt"], last_eval["bandit"]
    try:
        _, pv = stats.wilcoxon(qa, ba_)
    except ValueError:
        pv = float("nan")
    print(f"QT-Opt beats bandit in {(qa < ba_).mean()*100:.0f}% of envs | "
          f"mean ratio QT-Opt {qa.mean():.3f} vs bandit {ba_.mean():.3f} | p={pv:.2e}")
    for m in ["bandit", "PPO", "QT-Opt"]:
        try:
            _, pv = stats.wilcoxon(last_eval[m], last_eval["static"])
        except ValueError:
            pv = float("nan")
        print(f"{m:8s} beats static in {(last_eval[m] < last_eval['static']).mean()*100:.0f}% | p={pv:.2e}")
    np.savez(args.out, ratio={m: np.array(ratio[m]) for m in methods},
             last_eval=last_eval, curves=last_curves, center=center,
             regions=REGIONS.region.tolist())
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()

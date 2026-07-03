"""
Genuinely-sequential reformulation: a SHARED seasonal ITN budget.

Motivation (from Item B): with an independent fixed budget per campaign, the
allocation problem is near-myopic -- a one-step contextual bandit matches full
sequential QT-Opt, and standard PPO/DQN match it too. To test whether sequential
credit assignment ever matters here, we change the problem so that *timing* is a
real decision:

  * One shared budget B_total is rationed across the 3 campaigns AND 5 counties.
  * ITNs DECAY meaningfully (half-life ~70 days, cf. Wheldrake 2021 on net
    physical integrity), so nets deployed early are worn out by late season.

Now spending the whole budget early (myopically optimal for the first window,
because more nets always lower the immediate burden) leaves the late season both
under-funded AND with decayed nets. A sequential agent must learn to ration
budget over time. We compare:
  * spend-early   : deploy all budget at campaign 1 (pop-proportional)
  * even-thirds   : deploy B_total/3 per campaign (pop-proportional)
  * bandit (g=0)  : Q-learner with no bootstrapping (myopic)
  * QT-Opt (g=.99): sequential Q-learner (bootstrapped) -- our method

Run:  python shared_budget.py --episodes 1500 --seeds 0 1 2 --decay 0.01 --budget 26000
"""

import argparse
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Pin to a single thread: the per-row candidate scoring issues many small tensor
# ops, and CPU OpenMP/MKL intra-op threading can deadlock on that pattern. One
# thread is also faster here (no thread-pool overhead for tiny matmuls).
torch.set_num_threads(1)

from sar_sequential import (
    MalariaCounty, load_resistance_trajectories, COUNTY_SPECS, POP_STATIC,
    ACTION_DIM, CAMPAIGN_DAYS, HORIZON, TOTAL_POP, STATE_DIM,
)

REWARD_CENTER, REWARD_SCALE = 1.63, 5.0
N_CAND = 200          # candidate allocations scored per decision
N_MAX = 64            # candidates for the bootstrap max


# --------------------------------------------------------------------------- #
# Shared-budget environment                                                   #
# --------------------------------------------------------------------------- #
class SharedBudgetEnv:
    def __init__(self, resistance_df, rng, budget=26000.0, decay=0.01):
        self.budget = float(budget)
        self.counties = []
        for spec in COUNTY_SPECS:
            row = resistance_df.iloc[rng.integers(0, len(resistance_df))]
            c = MalariaCounty(row.to_numpy(), **spec)
            c.delta = decay                      # realistic net decay (timing matters)
            self.counties.append(c)

    def reset(self):
        for c in self.counties:
            c.reset()
        self.remaining = self.budget
        self._scale = 0.0
        obs = None
        for _ in range(CAMPAIGN_DAYS[0]):        # days 0..29 -> scale
            obs, burden = self._day_step(np.zeros(ACTION_DIM, dtype=np.float32))
            self._scale += burden
        self.k = 0
        return self._obs(obs)

    def _day_step(self, action):
        burden = 0.0
        state = []
        for i, c in enumerate(self.counties):
            c.step(action[i])
            burden += c.E_H + c.I_H
        return self._raw_state(), burden / TOTAL_POP

    def _raw_state(self):
        obs = []
        for c in self.counties:
            obs.extend(c.static_features() +
                       [c.S_H, c.E_H, c.I_H, c.T_H, c.R_H,
                        c.S_M, c.E_M, c.I_M, c.a_t, c.R])
        return np.array(obs, dtype=np.float32)

    def _obs(self, raw):
        # augment county state with remaining-budget fraction
        return np.concatenate([raw, [self.remaining / self.budget]]).astype(np.float32)

    def step(self, deploy_vec):
        """deploy_vec: 5 nonneg net counts requested this campaign."""
        deploy = np.maximum(np.asarray(deploy_vec, dtype=np.float64), 0.0)
        if self.k == 2:                          # last campaign: flush remaining budget
            s = deploy.sum()
            props = deploy / s if s > 1e-9 else np.full(ACTION_DIM, 1.0 / ACTION_DIM)
            deploy = props * self.remaining
        else:
            if deploy.sum() > self.remaining:    # cannot exceed remaining budget
                deploy *= self.remaining / (deploy.sum() + 1e-9)
        self.remaining = max(self.remaining - deploy.sum(), 0.0)

        burden = 0.0
        raw = None
        for d in range(HORIZON):
            act = deploy.astype(np.float32) if d == 0 else np.zeros(ACTION_DIM, np.float32)
            raw, b = self._day_step(act)
            burden += b
        ratio = burden / self._scale
        reward = REWARD_SCALE * (REWARD_CENTER - ratio)
        self.k += 1
        done = self.k >= 3
        return self._obs(raw), reward, ratio, done


OBS_DIM = STATE_DIM + 1            # 70 county features + remaining-budget fraction
INPUT_DIM = OBS_DIM + ACTION_DIM   # + 5 action


# --------------------------------------------------------------------------- #
# Q-network                                                                   #
# --------------------------------------------------------------------------- #
from sar_sequential import INPUT_NORM as _BASE_NORM
_STATE_NORM = _BASE_NORM[:STATE_DIM]


class QNet(nn.Module):
    def __init__(self, budget, hidden=128):
        super().__init__()
        norm = np.concatenate([_STATE_NORM, [1.0], [budget] * ACTION_DIM]).astype(np.float32)
        self.register_buffer("norm", torch.tensor(norm))
        self.fc1 = nn.Linear(INPUT_DIM, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x / self.norm
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


# --------------------------------------------------------------------------- #
# Sequential / myopic Q-agent over the shared-budget action space             #
# --------------------------------------------------------------------------- #
class SharedAgent:
    def __init__(self, budget, gamma=0.99, lr=1e-3, tau=0.01, batch=128, seed=0):
        self.budget = budget
        self.gamma = gamma
        self.tau = tau
        self.batch = batch
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.online = QNet(budget)
        self.target = QNet(budget)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.mem = deque(maxlen=10000)

    def _candidates(self, remaining, n):
        """Sample candidate deployments: fraction-of-remaining x spatial simplex.

        Fractions mix {explicit 0, explicit 1, U(0,1)} so 'spend nothing now' and
        'spend everything now' are always in the candidate set."""
        fr = self.rng.random(n)
        fr[0] = 1.0
        fr[1] = 0.0
        alphas = self.rng.choice([0.3, 0.5, 1.0], size=n)
        spatial = np.array([self.rng.dirichlet(np.full(ACTION_DIM, a)) for a in alphas])
        return (fr[:, None] * remaining * spatial).astype(np.float32)

    def _q(self, obs, cands):
        o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).expand(len(cands), -1)
        c = torch.tensor(cands, dtype=torch.float32)
        with torch.no_grad():
            return self.online(torch.cat([o, c], 1)).squeeze(1).numpy()

    def select(self, obs, epsilon):
        remaining = float(obs[-1]) * self.budget
        cands = self._candidates(remaining, N_CAND)
        if self.rng.random() < epsilon:
            return cands[self.rng.integers(0, len(cands))]
        q = self._q(obs, cands)
        return cands[int(np.argmax(q))]

    def remember(self, s, a, r, s2, done):
        self.mem.append((s, a, r, s2, done))

    def _max_next(self, next_obs):
        # next_obs: [B, OBS_DIM]; build candidates per-row using each row's remaining
        B = next_obs.shape[0]
        rem = next_obs[:, -1].numpy() * self.budget
        cand = np.stack([self._candidates(max(rem[i], 0.0), N_MAX) for i in range(B)])  # [B,N,5]
        ct = torch.tensor(cand, dtype=torch.float32)
        no = next_obs.unsqueeze(1).expand(B, N_MAX, OBS_DIM)
        inp = torch.cat([no, ct], 2).reshape(B * N_MAX, INPUT_DIM)
        with torch.no_grad():
            qt = self.target(inp).reshape(B, N_MAX)
            qo = self.online(inp).reshape(B, N_MAX)
            best = qo.argmax(1)
            return qt[torch.arange(B), best]

    def train_step(self):
        if len(self.mem) < self.batch:
            return
        batch = random.sample(self.mem, self.batch)
        s, a, r, s2, d = zip(*batch)
        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(np.array(a), dtype=torch.float32)
        r = torch.tensor(np.array(r), dtype=torch.float32)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32)
        d = torch.tensor(np.array(d), dtype=torch.float32)
        maxn = self._max_next(s2)
        target = r + self.gamma * (1 - d) * maxn
        pred = self.online(torch.cat([s, a], 1)).squeeze(1)
        loss = self.loss_fn(pred, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        with torch.no_grad():
            for tp, op in zip(self.target.parameters(), self.online.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * op)


# --------------------------------------------------------------------------- #
# Rollouts                                                                    #
# --------------------------------------------------------------------------- #
def run_episode(env, policy, collect=False):
    obs = env.reset()
    ratios, trans, acts = [], [], []
    states = []
    for k in range(3):
        states.append(obs)
        a = policy(obs, k)
        acts.append(a)
        obs2, r, ratio, done = env.step(a)
        ratios.append(ratio)
        if collect:
            trans.append([obs, a, r, obs2, done])
        obs = obs2
    return float(sum(ratios)), trans, acts


def heuristic_spend_early(budget):
    def pol(obs, k):
        return POP_STATIC / POP_STATIC.sum() * budget if k == 0 else np.zeros(ACTION_DIM, np.float32)
    return pol


def heuristic_even(budget):
    def pol(obs, k):
        return POP_STATIC / POP_STATIC.sum() * (budget / 3.0)
    return pol


# --------------------------------------------------------------------------- #
# Train + evaluate                                                            #
# --------------------------------------------------------------------------- #
def train_agent(resistance_df, gamma, episodes, seed, budget, decay):
    rng = np.random.default_rng(seed)
    agent = SharedAgent(budget, gamma=gamma, seed=seed)
    eps, eps_min, eps_decay = 1.0, 0.05, 0.997
    hist = []
    for ep in range(episodes):
        env = SharedBudgetEnv(resistance_df, rng, budget=budget, decay=decay)
        cum, trans, _ = run_episode(env, lambda o, k: agent.select(o, eps), collect=True)
        for tr in trans:
            agent.remember(*tr)
            agent.train_step()
        if eps > eps_min:
            eps *= eps_decay
        hist.append(cum)
    agent.history = hist
    return agent


def evaluate(policies, resistance_df, budget, decay, n_eval=50, base_seed=10000):
    out = {n: [] for n in policies}
    for i in range(n_eval):
        seed = base_seed + i
        for n, pol in policies.items():
            env = SharedBudgetEnv(resistance_df, np.random.default_rng(seed),
                                  budget=budget, decay=decay)
            cum, _, _ = run_episode(env, pol)
            out[n].append(cum)
    return {k: np.array(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--budget", type=float, default=26000.0)
    p.add_argument("--decay", type=float, default=0.01)
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--out", type=str, default="shared_budget_results.npz")
    args = p.parse_args()

    rdf = load_resistance_trajectories()
    methods = ["spend-early", "even-thirds", "bandit", "QT-Opt"]
    per_ratio = {m: [] for m in methods}
    last_eval, last_curves = None, {}

    for seed in args.seeds:
        print(f"\n--- seed {seed} (budget={args.budget:.0f}, decay={args.decay}) ---",
              flush=True)
        print("  training bandit (g=0)...", flush=True)
        bandit = train_agent(rdf, 0.0, args.episodes, seed, args.budget, args.decay)
        print("  training QT-Opt (g=0.99)...", flush=True)
        qt = train_agent(rdf, 0.99, args.episodes, seed, args.budget, args.decay)
        policies = {
            "spend-early": heuristic_spend_early(args.budget),
            "even-thirds": heuristic_even(args.budget),
            "bandit": lambda o, k: bandit.select(o, 0.0),
            "QT-Opt": lambda o, k: qt.select(o, 0.0),
        }
        res = evaluate(policies, rdf, args.budget, args.decay, args.n_eval)
        for m in methods:
            per_ratio[m].append(res[m].mean())
        last_eval = res
        last_curves = {"bandit": np.array(bandit.history), "QT-Opt": np.array(qt.history)}
        print("  " + " | ".join(f"{m} {res[m].mean():.3f}" for m in methods), flush=True)

    print("\n========= SHARED-BUDGET (sequential) COMPARISON =========")
    print(f"(mean over {len(args.seeds)} seeds; {args.n_eval} held-out envs; "
          f"lower infection ratio is better)\n")
    base = per_ratio["spend-early"]
    for m in methods:
        r = np.array(per_ratio[m])
        print(f"{m:12s} | infection ratio {r.mean():.3f} +/- {r.std():.3f}")
    # how much does sequential beat myopic?
    bandit_r = np.array(per_ratio["bandit"]).mean()
    qt_r = np.array(per_ratio["QT-Opt"]).mean()
    print(f"\nQT-Opt vs bandit (myopic): {(bandit_r - qt_r) / bandit_r * 100:+.2f}% "
          f"lower infection ratio for QT-Opt")
    np.savez(args.out, per_ratio={m: np.array(per_ratio[m]) for m in methods},
             last_eval=last_eval, curves=last_curves, seeds=np.array(args.seeds),
             budget=args.budget, decay=args.decay)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

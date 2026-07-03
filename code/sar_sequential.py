"""
Sequential State-Action Regression (SAR) for ITN allocation.

This module re-implements the SAR allocator as a *genuinely sequential* value
function with Bellman bootstrapping across the three mass-distribution campaigns
(days 30, 80, 130), replacing the original one-step reward regressor (contextual
bandit) in `mathematical-modelling.py`.

Why this matters
----------------
The manuscript formalizes the problem as a constrained MDP with discounted return
  max_pi  E[ sum_t gamma^t r_t ].
The original SAR regressed only the *immediate* per-campaign infection ratio onto
(state, action) with MSE -- no bootstrapping, and `gamma`, the target network, and
the soft-update `tau` were defined but never used. This module makes SAR a Q-value
function Q(s, a) trained with a Bellman backup, so credit propagates across
campaigns and the implementation matches the paper's formalism.

Episode structure (short 3-step MDP)
------------------------------------
  * decision epochs k = 0,1,2 at days 30, 80, 130
  * state  s_k : 70-dim observation just before campaign k (5 counties x 14 feats)
  * action a_k : 5-dim allocation, sum = BUDGET (Dirichlet x BUDGET => constraint holds)
  * reward r_k : -(burden over the 50 days after campaign k) / scale
                 scale = pre-campaign burden over days 0..29
  * target y_k = r_k + gamma*(1-done)*max_{a'} Q_target(s_{k+1}, a')
                 (terminal at k = 2)
The max over a' is approximated over the same Dirichlet candidate set used for
action selection. Double-DQN targeting (argmax by online net, value by target net)
counters value overestimation.

Run:  python sar_sequential.py --episodes 1000 --seed 0
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


# --------------------------------------------------------------------------- #
# Constants for the allocation problem                                        #
# --------------------------------------------------------------------------- #
BUDGET = 13000          # ITNs distributed per campaign
CAMPAIGN_DAYS = (30, 80, 130)
HORIZON = 50            # days of burden accumulated after each campaign
EPISODE_DAYS = 180
TOTAL_POP = 52000.0     # sum of N_H across the five counties
N_CANDIDATES = 25       # Dirichlet candidates scored per decision
STATE_DIM = 70          # 5 counties x 14 features
ACTION_DIM = 5
INPUT_DIM = STATE_DIM + ACTION_DIM  # 75


# --------------------------------------------------------------------------- #
# Synthetic insecticide-resistance trajectories                              #
# (same pipeline as mathematical-modelling.py)                                #
# --------------------------------------------------------------------------- #
def pm_to_R(pm):
    """Convert percent mosquito mortality to a resistance-impact value in [0,1]."""
    pm = pm / 100.0
    return 1.0 / (1.0 + np.exp(36.0 * (pm - 0.63)))


def load_resistance_trajectories(path="malaria-gan-outputs/random_pm.csv"):
    """Interpolate the 8 yearly mortality values into 180-day resistance series."""
    pm = pd.read_csv(path)
    series = []
    for row in range(len(pm)):
        data = pm.iloc[row].tolist()
        interp = []
        for i in range(len(data) - 1):
            seg = np.linspace(data[i], data[i + 1], 26, endpoint=False)
            interp.extend(seg)
        interp.append(data[-1])
        interp = interp[:EPISODE_DAYS]
        series.append([pm_to_R(v) for v in interp])
    return pd.DataFrame(series)


# --------------------------------------------------------------------------- #
# County compartmental model (human SEITR + mosquito SEI)                     #
# --------------------------------------------------------------------------- #
class MalariaCounty:
    """SEITR-SEI ODE county with seasonal biting, ITN decay, and resistance.

    Integration is forward-Euler with dt = 1 day, matching the original code.
    All epidemiological constants are identical to mathematical-modelling.py so
    that baseline / static numbers stay comparable.
    """

    def __init__(self, resistance_series, pop_multiply=1.0, vegetation_biting=0.3,
                 init_coverage=0.4, treatment_seeking=10):
        self.Rs = resistance_series
        self.multiplier = pop_multiply
        # --- constants ---
        self.mu_H = 1 / (60 * 365)
        self.mu_M = 1 / 12
        self.gamma = 1 / 50
        self.gamma_T = 1 / 21
        self.gamma_eta = 1 / 365
        self.rho = 1 / treatment_seeking
        self.delta_H = 0.01
        self.delta_T = 0.001
        self.sigma_H = 1 / 14
        self.sigma_M = 1 / 10
        self.epsilon = 0.3
        self.a = vegetation_biting
        self.seasonal_amp = 0.2          # seasonal biting amplitude (0 disables; for ablations)
        self.b = 0.2
        self.c = 0.2
        self.external_seeding = 60 * self.multiplier
        self.C = init_coverage
        self.delta = 0.001
        self.N_H = 10000 * self.multiplier
        self.N_M = 30000 * self.multiplier
        self.reset()

    def reset(self):
        m = self.multiplier
        self.S_H = (self.N_H - 1400) * 1.0
        self.E_H = 1000 * m
        self.I_H = 200 * m
        self.T_H = 100 * m
        self.R_H = 100 * m
        self.S_M = (self.N_M - 6000) * 1.0
        self.E_M = 3000 * m
        self.I_M = 3000 * m
        self.t = 0
        self.tau = 0
        self.R = self.Rs[self.t]
        self.a_t = self.a * (1 + self.seasonal_amp * np.sin(2 * np.pi * self.t / 180))

    def step(self, new_itns):
        curr_itns = self.C * self.N_H
        new_itns = float(np.clip(new_itns, 0, self.N_H))
        total_itns = curr_itns + new_itns
        self.C = min(float(total_itns / self.N_H), 1.0)
        self.tau = self.tau * curr_itns / total_itns if total_itns > 0 else self.tau
        self.R = self.Rs[self.t]
        self.a_t = self.a * (1 + self.seasonal_amp * np.sin(2 * np.pi * self.t / 180))

        theta_ITN = self.C * np.exp(-self.delta * self.tau) * (1 - 0.5 * self.R)
        Lambda_H = self.a_t * self.c * self.I_M / self.N_M * max(1 - theta_ITN, 0)
        Lambda_M = self.a_t * self.b * (self.I_H + self.epsilon * self.T_H) / self.N_H

        dS_H = self.mu_H * self.N_H - Lambda_H * self.S_H + self.gamma * self.R_H - self.mu_H * self.S_H
        dE_H = Lambda_H * self.S_H - self.sigma_H * self.E_H - self.mu_H * self.E_H + self.external_seeding
        dI_H = self.sigma_H * self.E_H - (self.rho + self.mu_H + self.delta_H) * self.I_H
        dT_H = self.rho * self.I_H - (self.mu_H + self.delta_T + self.gamma_T) * self.T_H
        dR_H = self.gamma_T * self.T_H - self.gamma * self.R_H - self.mu_H * self.R_H
        dS_M = self.mu_M * self.N_M - Lambda_M * self.S_M - self.mu_M * self.S_M
        dE_M = Lambda_M * self.S_M - self.sigma_M * self.E_M - self.mu_M * self.E_M
        dI_M = self.sigma_M * self.E_M - self.mu_M * self.I_M

        self.S_H += dS_H; self.E_H += dE_H; self.I_H += dI_H
        self.T_H += dT_H; self.R_H += dR_H
        self.S_M += dS_M; self.E_M += dE_M; self.I_M += dI_M
        self.t += 1
        self.tau += 1

        return np.array([self.S_H, self.E_H, self.I_H, self.T_H, self.R_H,
                         self.S_M, self.E_M, self.I_M, self.a_t, self.R],
                        dtype=np.float32)

    def static_features(self):
        return [self.multiplier, self.a, self.C, 1.0 / self.rho]


# --------------------------------------------------------------------------- #
# Five-county allocation environment                                          #
# --------------------------------------------------------------------------- #
COUNTY_SPECS = [
    dict(pop_multiply=1.5, vegetation_biting=0.1, init_coverage=0.50, treatment_seeking=5),
    dict(pop_multiply=1.0, vegetation_biting=0.3, init_coverage=0.40, treatment_seeking=10),
    dict(pop_multiply=0.7, vegetation_biting=0.5, init_coverage=0.30, treatment_seeking=15),
    dict(pop_multiply=1.2, vegetation_biting=0.2, init_coverage=0.35, treatment_seeking=7),
    dict(pop_multiply=0.8, vegetation_biting=0.4, init_coverage=0.45, treatment_seeking=12),
]
# Population-proportional split of BUDGET (sum of multipliers = 5.2).
POP_STATIC = np.array([1.5, 1.0, 0.7, 1.2, 0.8], dtype=np.float32)
POP_STATIC = POP_STATIC / POP_STATIC.sum() * BUDGET  # -> [3750,2500,1750,3000,2000]


class CountyAllocation:
    def __init__(self, resistance_df, rng):
        self.counties = []
        for spec in COUNTY_SPECS:
            row = resistance_df.iloc[rng.integers(0, len(resistance_df))]
            self.counties.append(MalariaCounty(row.to_numpy(), **spec))

    def reset(self):
        for c in self.counties:
            c.reset()
        return self._obs()

    def _obs(self):
        obs = []
        for c in self.counties:
            state = [c.S_H, c.E_H, c.I_H, c.T_H, c.R_H,
                     c.S_M, c.E_M, c.I_M, c.a_t, c.R]
            obs.extend(c.static_features() + state)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Advance one day. `action` is a 5-vector of ITNs (zeros off-campaign)."""
        burden = 0.0
        for i, c in enumerate(self.counties):
            c.step(action[i])
            burden += c.E_H + c.I_H
        return self._obs(), burden / TOTAL_POP


# --------------------------------------------------------------------------- #
# SAR Q-network                                                               #
# --------------------------------------------------------------------------- #
# Per-county feature scales used to normalize network inputs to O(1).
# Order: [pop_mult, veg, cov, treat_seek, S_H,E_H,I_H,T_H,R_H, S_M,E_M,I_M, a_t, R]
_COUNTY_NORM = [1.0, 1.0, 1.0, 15.0,
                1e4, 1e3, 1e3, 1e3, 1e3,
                3e4, 3e3, 3e3, 1.0, 1.0]
INPUT_NORM = np.array(_COUNTY_NORM * 5 + [float(BUDGET)] * ACTION_DIM, dtype=np.float32)


class SAR(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=128, dropout_rate=0.01):
        super().__init__()
        self.register_buffer("norm", torch.tensor(INPUT_NORM))
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x / self.norm                       # bring all inputs to O(1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        return self.output(x)


def sample_candidates(n, rng, alpha=1.0):
    """n Dirichlet allocations on the budget simplex: shape [n, 5].

    alpha < 1 concentrates mass on simplex corners (sparse allocations that pour
    nets into a few counties); alpha = 1 is uniform; alpha > 1 favors even splits.
    """
    return (rng.dirichlet(np.full(ACTION_DIM, alpha), size=n) * BUDGET).astype(np.float32)


# --------------------------------------------------------------------------- #
# Agent                                                                       #
# --------------------------------------------------------------------------- #
class SequentialSAR:
    def __init__(self, gamma=0.99, lr=1e-3, tau=0.01, batch_size=128,
                 dev_penalty=0.0, double=True, device="cpu", seed=0,
                 n_candidates=25, alpha=1.0, cem=False,
                 cem_iters=3, cem_pop=64, cem_elite=8, n_max=25,
                 hidden_dim=128, loss="mse", updates_per_step=1):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.dev_penalty = dev_penalty   # off by default (was always-on uniform bias)
        self.double = double
        self.device = device
        self.n_candidates = n_candidates
        self.alpha = alpha
        self.cem = cem
        self.cem_iters = cem_iters
        self.cem_pop = cem_pop
        self.cem_elite = cem_elite
        self.n_max = n_max               # candidates for the bootstrap max (kept modest)
        self.updates_per_step = updates_per_step
        self.rng = np.random.default_rng(seed)

        torch.manual_seed(seed)
        self.online = SAR(hidden_dim=hidden_dim).to(device)
        self.target = SAR(hidden_dim=hidden_dim).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss() if loss == "huber" else nn.MSELoss()
        self.memory = deque(maxlen=10000)

    def _score(self, state, candidates):
        """Q(state, candidate) for a [n,5] candidate array -> [n] numpy."""
        self.online.eval()
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device)
            s = s.unsqueeze(0).expand(len(candidates), -1)
            cand = torch.tensor(candidates, dtype=torch.float32, device=self.device)
            q = self.online(torch.cat([s, cand], dim=1)).squeeze(1)
        return q.cpu().numpy()

    # -- action selection: epsilon-greedy, optionally CEM-refined ------------ #
    def select_action(self, state, epsilon):
        if self.rng.random() < epsilon:
            return sample_candidates(1, self.rng, self.alpha)[0]
        if self.cem:
            return self._cem_action(state)
        candidates = sample_candidates(self.n_candidates, self.rng, self.alpha)
        q = self._score(state, candidates)
        return candidates[int(np.argmax(q))]

    def _cem_action(self, state):
        """Cross-entropy-method maximization of Q over the budget simplex.

        Iteratively samples Dirichlet candidates, keeps the elite by Q, and
        refits a more concentrated Dirichlet around the elite mean (QT-Opt style).
        """
        alpha = np.full(ACTION_DIM, self.alpha)
        best_a, best_q = None, -np.inf
        for it in range(self.cem_iters):
            cand = (self.rng.dirichlet(alpha, size=self.cem_pop) * BUDGET).astype(np.float32)
            q = self._score(state, cand)
            elite_idx = np.argsort(q)[-self.cem_elite:]
            mean_p = (cand[elite_idx] / BUDGET).mean(axis=0)
            conc = 10.0 * (it + 1)                       # tighten each iteration
            alpha = np.clip(mean_p * conc, 1e-2, None)
            top = elite_idx[-1]
            if q[top] > best_q:
                best_q, best_a = q[top], cand[top]
        return best_a

    def remember(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    # -- vectorized max_a' Q(s', a') over fresh Dirichlet candidates --------- #
    def _max_next_q(self, next_states):
        B = next_states.shape[0]
        m = self.n_max
        cand = self.rng.dirichlet(np.full(ACTION_DIM, self.alpha), size=(B, m)) * BUDGET
        cand_t = torch.tensor(cand, dtype=torch.float32, device=self.device)
        ns = next_states.unsqueeze(1).expand(B, m, STATE_DIM)
        inp = torch.cat([ns, cand_t], dim=2).reshape(B * m, INPUT_DIM)
        self.online.eval(); self.target.eval()
        with torch.no_grad():
            q_t = self.target(inp).reshape(B, m)
            if self.double:
                q_o = self.online(inp).reshape(B, m)
                best = q_o.argmax(dim=1)
                return q_t[torch.arange(B), best]
            return q_t.max(dim=1).values

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)

        max_next = self._max_next_q(next_states)
        target = rewards + self.gamma * (1.0 - dones) * max_next

        self.online.train()
        pred = self.online(torch.cat([states, actions], dim=1)).squeeze(1)
        loss = self.loss_fn(pred, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # soft (Polyak) target update
        with torch.no_grad():
            for tp, op in zip(self.target.parameters(), self.online.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * op)
        return float(loss.item())


# --------------------------------------------------------------------------- #
# Episode rollout (shared by training and evaluation)                         #
# --------------------------------------------------------------------------- #
def run_episode(env, policy, epsilon=0.0, collect=False, agent=None,
                reward_center=0.0, reward_scale=1.0):
    """Roll out one 180-day episode.

    `policy(state, k)` returns the 5-vector allocation for campaign k.
    Returns (cumulative_infection_ratio, per_campaign_ratios, transitions).
    Transitions (for training) are (s_k, a_k, r_k, s_{k+1}, done).

    Reward shaping: r_k = reward_scale * (reward_center - ratio_k). The constant
    `reward_center` only shifts the value baseline (optimal policy unchanged) but
    lets the regressor focus on allocation-driven differences instead of the large
    common offset; `reward_scale` amplifies the gradient signal.
    """
    obs = env.reset()
    window = [0.0, 0.0, 0.0, 0.0]   # [scale, r-window0, r-window1, r-window2]
    idx = 0
    decision_states = [None, None, None]
    decision_actions = [None, None, None]
    last_obs = obs
    for day in range(EPISODE_DAYS):
        if day in CAMPAIGN_DAYS:
            decision_states[idx] = last_obs
            action = policy(last_obs, idx)
            decision_actions[idx] = action
            idx += 1
        else:
            action = np.zeros(ACTION_DIM, dtype=np.float32)
        obs, burden = env.step(action)
        # window 0 = days 0..29 (scale); windows 1..3 = post-campaign 50-day blocks
        cur = 0 if day < CAMPAIGN_DAYS[0] else idx
        window[cur] += burden
        last_obs = obs

    scale = window[0]
    ratios = [window[k + 1] / scale for k in range(3)]
    cum_ratio = float(sum(ratios))

    transitions = []
    if collect:
        for k in range(3):
            r = reward_scale * (reward_center - ratios[k])  # max return == min ratio
            done = (k == 2)
            s_next = decision_states[k + 1] if k < 2 else decision_states[k]
            transitions.append((decision_states[k], decision_actions[k], r, s_next, done))
    return cum_ratio, ratios, transitions, decision_actions


# --------------------------------------------------------------------------- #
# Baselines                                                                   #
# --------------------------------------------------------------------------- #
def baseline_policy(state, k):
    return np.zeros(ACTION_DIM, dtype=np.float32)


def static_policy(state, k):
    return POP_STATIC.copy()


# --------------------------------------------------------------------------- #
# Training / evaluation driver                                                #
# --------------------------------------------------------------------------- #
def train(episodes=1000, seed=0, dev_penalty=0.0, double=True, verbose=True,
          reward_center=1.63, reward_scale=5.0, agent_kwargs=None):
    rng = np.random.default_rng(seed)
    resistance_df = load_resistance_trajectories()
    agent = SequentialSAR(dev_penalty=dev_penalty, double=double, seed=seed,
                          **(agent_kwargs or {}))

    epsilon, eps_min, eps_decay = 1.0, 0.05, 0.996
    history = []
    for ep in range(episodes):
        env = CountyAllocation(resistance_df, rng)

        def policy(state, k):
            return agent.select_action(state, epsilon)

        cum, ratios, transitions, _ = run_episode(
            env, policy, epsilon=epsilon, collect=True, agent=agent,
            reward_center=reward_center, reward_scale=reward_scale)
        for tr in transitions:
            agent.remember(*tr)
            for _ in range(agent.updates_per_step):
                agent.train_step()

        if epsilon > eps_min:
            epsilon *= eps_decay
        history.append(cum)
        if verbose and (ep + 1) % 100 == 0:
            recent = np.mean(history[-100:])
            print(f"ep {ep+1:4d} | eps {epsilon:.3f} | cum_ratio {cum:.3f} "
                  f"| mean(last100) {recent:.3f}")
    agent.history = history          # learning curve (cumulative infection ratio/episode)
    return agent, resistance_df


def evaluate(agent, resistance_df, n_eval=50, base_seed=10000):
    """Evaluate SAR (greedy), static, and baseline on identical environments."""
    sar_v, static_v, base_v = [], [], []
    for i in range(n_eval):
        seed = base_seed + i
        def make_env():
            return CountyAllocation(resistance_df, np.random.default_rng(seed))

        cum_sar, _, _, _ = run_episode(make_env(), lambda s, k: agent.select_action(s, 0.0))
        cum_static, _, _, _ = run_episode(make_env(), static_policy)
        cum_base, _, _, _ = run_episode(make_env(), baseline_policy)
        sar_v.append(cum_sar); static_v.append(cum_static); base_v.append(cum_base)
    return np.array(sar_v), np.array(static_v), np.array(base_v)


def summarize(sar_v, static_v, base_v):
    b = base_v.mean()
    def red(x):  # percent reduction vs baseline
        return (b - x.mean()) / b * 100
    print("\n===== Evaluation (mean +/- std over "
          f"{len(sar_v)} held-out episodes) =====")
    print(f"Baseline (no ITNs)      : {base_v.mean():.3f} +/- {base_v.std():.3f}")
    print(f"Static (pop-proportional): {static_v.mean():.3f} +/- {static_v.std():.3f}"
          f"   ({red(static_v):.1f}% reduction)")
    print(f"Sequential SAR          : {sar_v.mean():.3f} +/- {sar_v.std():.3f}"
          f"   ({red(sar_v):.1f}% reduction)")
    print(f"SAR improvement factor vs static: {red(sar_v) / max(red(static_v), 1e-9):.2f}x")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--dev-penalty", type=float, default=0.0)
    p.add_argument("--no-double", action="store_true")
    # action-optimization levers (Item A tuning)
    p.add_argument("--candidates", type=int, default=25)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--cem", action="store_true")
    p.add_argument("--cem-iters", type=int, default=3)
    p.add_argument("--cem-pop", type=int, default=64)
    p.add_argument("--cem-elite", type=int, default=8)
    # use a separate seed block for tuning decisions; keep 10000+ for final eval
    p.add_argument("--eval-base-seed", type=int, default=10000)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    agent_kwargs = dict(n_candidates=args.candidates, alpha=args.alpha, cem=args.cem,
                        cem_iters=args.cem_iters, cem_pop=args.cem_pop,
                        cem_elite=args.cem_elite)
    agent, resistance_df = train(episodes=args.episodes, seed=args.seed,
                                 dev_penalty=args.dev_penalty,
                                 double=not args.no_double,
                                 agent_kwargs=agent_kwargs)
    sar_v, static_v, base_v = evaluate(agent, resistance_df, n_eval=args.n_eval,
                                       base_seed=args.eval_base_seed)
    summarize(sar_v, static_v, base_v)


if __name__ == "__main__":
    main()

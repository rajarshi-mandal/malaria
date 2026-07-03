"""
Item B: learned baselines for the ITN-allocation problem.

Compares, on identical held-out environments and the same honest greedy
protocol, six allocators:
  * baseline   - no ITNs
  * static     - population-proportional
  * bandit     - QT-Opt with gamma = 0  (no Bellman backup; the "old SAR"
                 idea, i.e. one-step reward regression, but evaluated honestly)
  * DQN        - value-based RL over a fixed library of allocation templates
  * PPO        - policy-gradient RL, continuous action (softmax -> simplex)
  * QT-Opt     - our sequential method (Bellman backup + CEM action search)

Key fairness guarantees
-----------------------
* ALL methods are scored by the SAME canonical rollout (`run_episode` from
  sar_sequential) on the SAME seeded environments, so dynamics/resistance draws
  are identical across methods.
* DQN/PPO are given a *generous* training budget (more environment interaction
  than QT-Opt), so the comparison is conservative -- it does not strawman them.
* The training reward shaping (center/scale) is identical for every learned
  method; the reported metric (cumulative infection ratio) is unshaped.

Run:  python baselines_b.py --seeds 0 1 2 --ppo-steps 60000 --dqn-steps 60000
"""

import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

from sar_sequential import (
    CountyAllocation, run_episode, static_policy, baseline_policy,
    load_resistance_trajectories, train as train_qtopt,
    BUDGET, ACTION_DIM, STATE_DIM, INPUT_NORM, CAMPAIGN_DAYS, HORIZON, POP_STATIC,
)

STATE_NORM = INPUT_NORM[:STATE_DIM]          # 70-dim observation normalizer
REWARD_CENTER, REWARD_SCALE = 1.63, 5.0      # identical to QT-Opt training

# QT-Opt config locked in Item A (CEM winner). dev set may refine; this is the base.
QTOPT_CFG = dict(cem=True, cem_pop=96, cem_iters=4, cem_elite=12, alpha=1.0)


def normalize_state(s):
    return (np.asarray(s, dtype=np.float32) / STATE_NORM).astype(np.float32)


def decode_continuous(action):
    """Real vector -> budget simplex via softmax."""
    a = np.asarray(action, dtype=np.float64)
    e = np.exp(a - a.max())
    return ((e / e.sum()) * BUDGET).astype(np.float32)


def build_action_library(k=64, seed=0):
    """Fixed discrete allocation templates for DQN.

    Includes structured anchors (corners, uniform, population-proportional) plus
    random Dirichlet draws at mixed concentrations for coverage of the simplex.
    """
    rng = np.random.default_rng(seed)
    lib = []
    for i in range(ACTION_DIM):                      # 5 corners (all to one county)
        v = np.zeros(ACTION_DIM, dtype=np.float32); v[i] = BUDGET; lib.append(v)
    lib.append(np.full(ACTION_DIM, BUDGET / ACTION_DIM, dtype=np.float32))   # uniform
    lib.append(POP_STATIC.astype(np.float32))                               # pop-prop
    while len(lib) < k:
        alpha = rng.choice([0.3, 0.5, 1.0])
        lib.append((rng.dirichlet(np.full(ACTION_DIM, alpha)) * BUDGET).astype(np.float32))
    return np.array(lib[:k], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Gymnasium-compliant 3-step allocation env (for SB3 training)                #
# --------------------------------------------------------------------------- #
class AllocationGymEnv(gym.Env):
    """One episode = the three campaign decisions (days 30, 80, 130).

    reset(): rolls days 0..29 to fix the per-episode scale, returns the
             normalized pre-campaign state.
    step(a): applies allocation on the current campaign day, rolls 50 days,
             returns shaped reward = scale*(center - ratio_k).
    Dynamics are byte-for-byte the same as sar_sequential.run_episode.
    """

    metadata = {"render_modes": []}

    def __init__(self, resistance_df, discrete=False, action_library=None, seed=None):
        super().__init__()
        self.resistance_df = resistance_df
        self.discrete = discrete
        self.action_library = action_library
        self._rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(STATE_DIM,), dtype=np.float32)
        if discrete:
            self.action_space = spaces.Discrete(len(action_library))
        else:
            self.action_space = spaces.Box(low=-5.0, high=5.0,
                                           shape=(ACTION_DIM,), dtype=np.float32)

    def _decode(self, action):
        if self.discrete:
            return self.action_library[int(action)].astype(np.float32)
        return decode_continuous(action)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.env = CountyAllocation(self.resistance_df, self._rng)
        self.env.reset()
        self._scale = 0.0
        obs = None
        for _ in range(CAMPAIGN_DAYS[0]):                 # days 0..29
            obs, burden = self.env.step(np.zeros(ACTION_DIM, dtype=np.float32))
            self._scale += burden
        self._k = 0
        return normalize_state(obs), {}

    def step(self, action):
        alloc = self._decode(action)
        burden = 0.0
        obs = None
        for d in range(HORIZON):                          # 50-day post-campaign block
            act = alloc if d == 0 else np.zeros(ACTION_DIM, dtype=np.float32)
            obs, b = self.env.step(act)
            burden += b
        ratio = burden / self._scale
        reward = REWARD_SCALE * (REWARD_CENTER - ratio)
        self._k += 1
        terminated = self._k >= 3
        return normalize_state(obs), float(reward), bool(terminated), False, {"ratio": ratio}


# --------------------------------------------------------------------------- #
# Policy wrappers -> policy(state, k) for the canonical evaluator              #
# --------------------------------------------------------------------------- #
def make_sb3_policy(model, discrete, action_library=None):
    def pol(state, k):
        obs = normalize_state(state)
        action, _ = model.predict(obs, deterministic=True)
        if discrete:
            return action_library[int(action)].astype(np.float32)
        return decode_continuous(action)
    return pol


def qtopt_policy(agent):
    return lambda state, k: agent.select_action(state, 0.0)


# --------------------------------------------------------------------------- #
# Unified held-out evaluation (identical environments for every policy)       #
# --------------------------------------------------------------------------- #
def evaluate_policies(policies, resistance_df, n_eval=50, base_seed=10000):
    results = {name: [] for name in policies}
    for i in range(n_eval):
        seed = base_seed + i
        for name, pol in policies.items():
            env = CountyAllocation(resistance_df, np.random.default_rng(seed))
            cum, _, _, _ = run_episode(env, pol)
            results[name].append(cum)
    return {k: np.array(v) for k, v in results.items()}


# --------------------------------------------------------------------------- #
# Training wrappers                                                           #
# --------------------------------------------------------------------------- #
def train_ppo(resistance_df, steps, seed):
    env = Monitor(AllocationGymEnv(resistance_df, discrete=False, seed=seed))
    model = PPO("MlpPolicy", env, seed=seed, verbose=0,
                n_steps=600, batch_size=200, gae_lambda=0.95, gamma=0.99,
                ent_coef=0.0, learning_rate=3e-4,
                policy_kwargs=dict(net_arch=[128, 128]))
    model.learn(total_timesteps=steps)
    return model


def train_dqn(resistance_df, steps, seed, action_library):
    env = Monitor(AllocationGymEnv(resistance_df, discrete=True,
                                   action_library=action_library, seed=seed))
    model = DQN("MlpPolicy", env, seed=seed, verbose=0,
                learning_rate=1e-3, buffer_size=50000, learning_starts=1000,
                batch_size=128, gamma=0.99, train_freq=1, target_update_interval=500,
                exploration_fraction=0.5, exploration_final_eps=0.05,
                policy_kwargs=dict(net_arch=[128, 128]))
    model.learn(total_timesteps=steps)
    return model


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def reductions(results):
    b = results["baseline"].mean()
    return {k: (b - v.mean()) / b * 100 for k, v in results.items()}


def run_one_seed(resistance_df, seed, qtopt_episodes, ppo_steps, dqn_steps,
                 bandit_episodes, action_library, n_eval, eval_base_seed):
    print(f"\n--- seed {seed} ---", flush=True)
    print("  training QT-Opt...", flush=True)
    qt_agent, _ = train_qtopt(episodes=qtopt_episodes, seed=seed, verbose=False,
                              agent_kwargs=QTOPT_CFG)
    print("  training contextual bandit (gamma=0)...", flush=True)
    bandit_agent, _ = train_qtopt(episodes=bandit_episodes, seed=seed, verbose=False,
                                  agent_kwargs={**QTOPT_CFG, "gamma": 0.0})
    print("  training PPO...", flush=True)
    ppo = train_ppo(resistance_df, ppo_steps, seed)
    print("  training DQN...", flush=True)
    dqn = train_dqn(resistance_df, dqn_steps, seed, action_library)

    policies = {
        "baseline": baseline_policy,
        "static": static_policy,
        "bandit": qtopt_policy(bandit_agent),
        "DQN": make_sb3_policy(dqn, discrete=True, action_library=action_library),
        "PPO": make_sb3_policy(ppo, discrete=False),
        "QT-Opt": qtopt_policy(qt_agent),
    }
    res = evaluate_policies(policies, resistance_df, n_eval=n_eval,
                            base_seed=eval_base_seed)
    curves = {"QT-Opt": np.array(qt_agent.history),
              "bandit": np.array(bandit_agent.history)}
    return res, curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--qtopt-episodes", type=int, default=1000)
    p.add_argument("--bandit-episodes", type=int, default=1000)
    p.add_argument("--ppo-steps", type=int, default=60000)
    p.add_argument("--dqn-steps", type=int, default=60000)
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--eval-base-seed", type=int, default=10000)
    p.add_argument("--out", type=str, default="baselines_b_results.npz")
    args = p.parse_args()

    resistance_df = load_resistance_trajectories()
    action_library = build_action_library(k=64, seed=0)

    methods = ["baseline", "static", "bandit", "DQN", "PPO", "QT-Opt"]
    per_seed_red = {m: [] for m in methods}
    per_seed_ratio = {m: [] for m in methods}
    last_res = None
    last_curves = None

    for seed in args.seeds:
        res, curves = run_one_seed(resistance_df, seed, args.qtopt_episodes,
                                   args.ppo_steps, args.dqn_steps, args.bandit_episodes,
                                   action_library, args.n_eval, args.eval_base_seed)
        red = reductions(res)
        for m in methods:
            per_seed_red[m].append(red[m])
            per_seed_ratio[m].append(res[m].mean())
        last_res = res
        last_curves = curves
        print(f"  seed {seed} reductions: "
              + " | ".join(f"{m} {red[m]:.1f}%" for m in methods), flush=True)

    print("\n================ ITEM B: HELD-OUT COMPARISON ================")
    print(f"(mean over {len(args.seeds)} training seeds; eval on {args.n_eval} "
          f"held-out envs, seeds {args.eval_base_seed}+)\n")
    print(f"{'Method':10s} | {'Inf. ratio':>12s} | {'% reduction':>16s}")
    print("-" * 46)
    for m in methods:
        r = np.array(per_seed_ratio[m]); rd = np.array(per_seed_red[m])
        print(f"{m:10s} | {r.mean():6.3f}+/-{r.std():.3f} | "
              f"{rd.mean():6.2f}+/-{rd.std():4.2f}%")

    np.savez(args.out,
             ratios={m: np.array(per_seed_ratio[m]) for m in methods},
             reductions={m: np.array(per_seed_red[m]) for m in methods},
             last_eval={m: last_res[m] for m in methods},
             curves=last_curves,
             seeds=np.array(args.seeds))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()

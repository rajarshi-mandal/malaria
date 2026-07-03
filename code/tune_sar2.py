"""Round-2 QT-Opt tweaks, building on the CEM winner.

All configs start from CEM (pop=96,iters=4,elite=12) and vary one axis.
Tuned on validation seed block 20000; test seeds (10000) stay clean.
"""
import numpy as np
from sar_sequential import train, evaluate

VAL_SEED, EPISODES, N_EVAL, TRAIN_SEED = 20000, 600, 30, 0
CEM = dict(cem=True, cem_pop=96, cem_iters=4, cem_elite=12, alpha=1.0)

CONFIGS = [
    ("cem-base",                {**CEM}),
    ("cem+bigger",              {**CEM, "cem_pop": 128, "cem_iters": 5, "cem_elite": 16}),
    ("cem+sparse-init(a=0.5)",  {**CEM, "alpha": 0.5}),
    ("cem+huber",               {**CEM, "loss": "huber"}),
    ("cem+wide(256)",           {**CEM, "hidden_dim": 256}),
    ("cem+2updates",            {**CEM, "updates_per_step": 2}),
    ("cem+nmax64",              {**CEM, "n_max": 64}),
]


def pct(x, b):
    return (b - x.mean()) / b * 100


def main():
    rows = []
    for name, kw in CONFIGS:
        agent, rdf = train(episodes=EPISODES, seed=TRAIN_SEED, verbose=False,
                           agent_kwargs=kw)
        s, st, ba = evaluate(agent, rdf, n_eval=N_EVAL, base_seed=VAL_SEED)
        b = ba.mean()
        rs, rst = pct(s, b), pct(st, b)
        rows.append((name, s.mean(), rs, rs / max(rst, 1e-9)))
        print(f"{name:24s} | SAR {s.mean():.3f} | red {rs:5.2f}% "
              f"| static {rst:4.1f}% | {rs/max(rst,1e-9):.2f}x", flush=True)

    rows.sort(key=lambda r: r[1])
    print("\n===== ROUND-2 SUMMARY (val seed 20000) =====")
    for name, sar, rs, fac in rows:
        print(f"{name:24s} | ratio {sar:.3f} | {rs:5.2f}% | {fac:.2f}x")
    print(f"\nBest: {rows[0][0]}")


if __name__ == "__main__":
    main()

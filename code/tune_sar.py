"""Validation sweep for SAR action-optimization levers (Item A tuning).

Tunes on a VALIDATION seed block (20000+) so the final held-out eval seeds
(10000+) stay clean. Each config is trained from the same seed for a fair
comparison; we report % reduction vs baseline and the factor over static.
"""
import numpy as np
from sar_sequential import train, evaluate

VAL_SEED = 20000          # validation block (distinct from final eval 10000+)
EPISODES = 500            # converged by ~500; fast enough to sweep
N_EVAL = 30
TRAIN_SEED = 0

CONFIGS = [
    ("current  (cand=25, a=1.0)",        dict(n_candidates=25,  alpha=1.0)),
    ("more     (cand=100, a=1.0)",       dict(n_candidates=100, alpha=1.0)),
    ("sparse   (cand=100, a=0.5)",       dict(n_candidates=100, alpha=0.5)),
    ("verysparse(cand=100, a=0.3)",      dict(n_candidates=100, alpha=0.3)),
    ("cem      (pop=96,it=4,el=12)",     dict(cem=True, cem_pop=96, cem_iters=4,
                                             cem_elite=12, alpha=1.0)),
]


def pct_red(x, b):
    return (b - x.mean()) / b * 100


def main():
    rows = []
    for name, kw in CONFIGS:
        agent, rdf = train(episodes=EPISODES, seed=TRAIN_SEED, verbose=False,
                           agent_kwargs=kw)
        sar_v, static_v, base_v = evaluate(agent, rdf, n_eval=N_EVAL,
                                           base_seed=VAL_SEED)
        b = base_v.mean()
        rs, rst = pct_red(sar_v, b), pct_red(static_v, b)
        rows.append((name, sar_v.mean(), rs, rst, rs / max(rst, 1e-9)))
        print(f"{name:28s} | SAR {sar_v.mean():.3f} "
              f"| SAR red {rs:5.1f}% | static red {rst:4.1f}% | {rs/max(rst,1e-9):.2f}x",
              flush=True)

    print("\n===== VALIDATION SWEEP SUMMARY (seed block 20000) =====")
    rows.sort(key=lambda r: r[1])  # lower SAR ratio is better
    for name, sar, rs, rst, fac in rows:
        print(f"{name:28s} | SAR ratio {sar:.3f} | {rs:5.1f}% red | {fac:.2f}x static")
    print(f"\nBest config: {rows[0][0]}")


if __name__ == "__main__":
    main()

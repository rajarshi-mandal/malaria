"""Diagnose the endo-resistance regime: does coverage saturate? does R move
enough to matter? how concentrated is the static vs optimal allocation? This
tells us what to tune so the deployment->resistance feedback actually creates a
greedy-vs-oracle gap (genuine sequential structure)."""
import numpy as np
import endo_planners as ep
import fast_sim_endo as fe


def trace(k_sel, k_rev, budget=None, k=6, gap=365):
    if budget is not None:
        ep.BUDGET = budget
        ep.POP_STATIC = (ep._mult / ep._mult.sum() * budget).astype(np.float64)
    e = ep.EndoEnv(np.random.default_rng(10000), k=k, gap=gap, k_sel=k_sel, k_rev=k_rev)
    e.reset()
    e.run_interval(np.zeros(ep.N), ep.FIRST)
    print(f"\nk_sel={k_sel:.1e} k_rev={k_rev:.1e} budget={ep.BUDGET:.0f}")
    print(f"  {'camp':>4} {'meanC':>6} {'meanEff':>7} {'meanR':>6} {'maxR':>6} {'minR':>6}")
    for c in range(k):
        e.run_interval(ep.POP_STATIC, gap)
        eff = e.C * np.exp(-e.P[:, 16] * e.tau)
        print(f"  {c:>4} {e.C.mean():6.3f} {eff.mean():7.3f} {e.R.mean():6.3f} "
              f"{e.R.max():6.3f} {e.R.min():6.3f}")


def gap_proxy(k_sel, k_rev, horizon, budget, n=4, k=6, gap=365):
    """ARMOR(full-horizon) minus greedy(short-horizon) %reduction over n envs."""
    ep.BUDGET = budget
    ep.POP_STATIC = (ep._mult / ep._mult.sum() * budget).astype(np.float64)
    rng = np.random.default_rng(0)
    gr, ar, st, bl = [], [], [], []
    for i in range(n):
        seed = 10000 + i
        e = ep.EndoEnv(np.random.default_rng(seed), k=k, gap=gap, k_sel=k_sel, k_rev=k_rev)
        bl.append(ep.run_episode(e, lambda o, kk: np.zeros(ep.N))[0])
        e = ep.EndoEnv(np.random.default_rng(seed), k=k, gap=gap, k_sel=k_sel, k_rev=k_rev)
        st.append(ep.run_episode(e, ep.static_pol(e))[0])
        e = ep.EndoEnv(np.random.default_rng(seed), k=k, gap=gap, k_sel=k_sel, k_rev=k_rev)
        gr.append(ep.run_episode(e, ep.greedy_pol(e, horizon=horizon))[0])
        e = ep.EndoEnv(np.random.default_rng(seed), k=k, gap=gap, k_sel=k_sel, k_rev=k_rev)
        ar.append(ep.run_episode(e, ep.armor_pol(e, rng))[0])
    b = np.mean(bl)
    rg = (b - np.mean(gr)) / b * 100
    ra = (b - np.mean(ar)) / b * 100
    rs = (b - np.mean(st)) / b * 100
    print(f"k_sel={k_sel:.1e} k_rev={k_rev:.1e} hz={horizon:>3} budget={budget:.0f} "
          f"| static={rs:5.2f}% greedy={rg:5.2f}% ARMOR={ra:5.2f}% "
          f"| ARMOR-greedy={ra-rg:+5.2f}pp")
    return ra - rg


if __name__ == "__main__":
    print("=== coverage/resistance trajectory under static allocation ===")
    trace(9.6e-4, 2.74e-4)
    trace(3.0e-3, 1.0e-3)
    trace(3.0e-3, 1.0e-3, budget=8000)
    print("\n=== ARMOR(full) vs greedy(short horizon) gap proxy ===")
    for hz in (50, 120):
        for ks in (9.6e-4, 2e-3, 4e-3):
            gap_proxy(ks, 1.0e-3, hz, 8000)

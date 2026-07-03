"""Find a regime where the allocation problem is genuinely non-myopic, i.e. a
network-aware horizon planner (ARMOR) clearly beats a decoupled greedy. Sweeps
budget scarcity, reseeding (elimination regime), and mobility coupling. ARMOR is
the cheap planning upper-bound proxy (skip the costly oracle during search)."""
import numpy as np
import meta_planners as mp


def evalcfg(budget, seeding, mob, n=3, k=6, gap=365):
    rng = np.random.default_rng(0)
    out = {m: [] for m in ["baseline", "static", "dgreedy", "armor"]}
    for i in range(n):
        seed = 10000 + i
        def mk():
            return mp.MetaEnv(np.random.default_rng(seed), k=k, gap=gap,
                              mobility=mob, budget=budget, seeding_scale=seeding)
        e = mk(); out["baseline"].append(mp.run_episode(e, lambda o, kk: np.zeros(mp.N))[0])
        e = mk(); out["static"].append(mp.run_episode(e, mp.static_pol(e))[0])
        e = mk(); out["dgreedy"].append(mp.run_episode(e, mp.dgreedy_pol(e))[0])
        e = mk(); out["armor"].append(mp.run_episode(e, mp.armor_pol(e, rng, pop=250, iters=8, elite=25))[0])
    b = np.mean(out["baseline"])
    red = {m: (b - np.mean(out[m])) / b * 100 for m in out}
    gap = red["armor"] - red["dgreedy"]
    print(f"budget={budget:>5.0f} seed={seeding:>3} mob={mob:>3} | "
          f"static={red['static']:5.1f}% dgreedy={red['dgreedy']:5.1f}% "
          f"armor={red['armor']:5.1f}% | ARMOR-dgreedy={gap:+5.2f}pp", flush=True)
    return gap


if __name__ == "__main__":
    print("=== regime sweep: looking for ARMOR >> dgreedy (non-myopic structure) ===")
    best = (-1e9, None)
    for budget in (3000.0, 6000.0, 12000.0):
        for seeding in (0.0, 0.1, 1.0):
            for mob in (0.0, 0.5):
                g = evalcfg(budget, seeding, mob)
                if g > best[0]:
                    best = (g, (budget, seeding, mob))
    print(f"\nBEST gap {best[0]:+.2f}pp at budget/seed/mob = {best[1]}")

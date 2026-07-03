"""Verify fast_sim.py matches the scalar simulators to floating-point tolerance."""
import numpy as np
import fast_sim as fs
from sar_sequential import MalariaCounty, load_resistance_trajectories
from extended_ode import ExtendedCounty

EP = 180
C0d, C1d, C2d = 30, 80, 130


def scalar_seitr_windows(c, alloc3):
    w = np.zeros(4); idx = 0
    for day in range(EP):
        a = 0.0
        if day == C0d:
            a = alloc3[0]; idx = 1
        elif day == C1d:
            a = alloc3[1]; idx = 2
        elif day == C2d:
            a = alloc3[2]; idx = 3
        c.step(a)
        wj = 0 if day < C0d else idx
        w[wj] += c.E_H + c.I_H
    return w


def scalar_seitar_windows(c, alloc3):
    w = np.zeros(4); idx = 0
    for day in range(EP):
        a = 0.0
        if day == C0d:
            a = alloc3[0]; idx = 1
        elif day == C1d:
            a = alloc3[1]; idx = 2
        elif day == C2d:
            a = alloc3[2]; idx = 3
        c.step(a, seasonal=True)
        wj = 0 if day < C0d else idx
        w[wj] += c.E_H + c.I_H
    return w


def main():
    rdf = load_resistance_trajectories()
    rng = np.random.default_rng(123)
    alloc3 = np.array([5000.0, 3000.0, 4000.0])

    # ---- SEITR full-episode windows ----
    print("=== SEITR ===")
    maxerr = 0.0
    for trial in range(5):
        row = rdf.iloc[rng.integers(0, len(rdf))].to_numpy()
        spec = dict(pop_multiply=float(rng.uniform(0.5, 2.0)),
                    vegetation_biting=float(rng.uniform(0.1, 1.5)),
                    init_coverage=float(rng.uniform(0.2, 0.6)),
                    treatment_seeking=int(rng.integers(5, 16)))
        c_fast = MalariaCounty(row, **spec)
        c_fast.external_seeding = 4e-5 * c_fast.N_H
        P = fs.pack_seitr(c_fast); s0 = fs.state_seitr(c_fast)
        Rser = np.asarray(c_fast.Rs, dtype=np.float64)
        wf = fs.seitr_windows(s0, c_fast.C, float(c_fast.tau), P, Rser,
                              alloc3, C0d, C1d, C2d, EP)
        c_sc = MalariaCounty(row, **spec); c_sc.external_seeding = 4e-5 * c_sc.N_H
        ws = scalar_seitr_windows(c_sc, alloc3)
        rel = np.abs(wf - ws) / (np.abs(ws) + 1e-9)
        maxerr = max(maxerr, rel.max())
        print(f"  trial {trial}: fast {wf.round(2)} scalar {ws.round(2)} relerr {rel.max():.2e}")
    print(f"SEITR max rel err = {maxerr:.2e}\n")

    # ---- SEITR mid-episode lookahead (greedy/MPC use this) ----
    print("=== SEITR lookahead (from day 30) ===")
    row = rdf.iloc[0].to_numpy()
    spec = dict(pop_multiply=1.2, vegetation_biting=0.8, init_coverage=0.4, treatment_seeking=10)
    c = MalariaCounty(row, **spec); c.external_seeding = 4e-5 * c.N_H
    for _ in range(C0d):           # advance to campaign day 30
        c.step(0.0)
    P = fs.pack_seitr(c); s0 = fs.state_seitr(c)
    Rser = np.asarray(c.Rs, dtype=np.float64)
    bf = fs.seitr_lookahead(s0, c.C, float(c.tau), c.t, 6000.0, P, Rser, 50)
    # scalar equivalent
    cs = MalariaCounty(row, **spec); cs.external_seeding = 4e-5 * cs.N_H
    for _ in range(C0d):
        cs.step(0.0)
    bs = 0.0
    for d in range(50):
        cs.step(6000.0 if d == 0 else 0.0)
        bs += cs.E_H + cs.I_H
    print(f"  fast {bf:.3f} scalar {bs:.3f} relerr {abs(bf-bs)/abs(bs):.2e}\n")

    # ---- SEITAR full-episode windows ----
    print("=== SEITAR ===")
    maxerr = 0.0
    for trial in range(5):
        biting = float(rng.uniform(0.2, 6.0)); cov = float(rng.uniform(0.2, 0.6))
        mult = float(rng.uniform(0.5, 2.0)); p = float(rng.uniform(0.1, 0.8))
        c_fast = ExtendedCounty(biting, cov, mult, treat=10, p_sympt=p)
        for _ in range(200):
            c_fast.step(0.0, seasonal=False)   # some burn-in to a nontrivial state
        P = fs.pack_seitar(c_fast); s0 = fs.state_seitar(c_fast)
        C0v, tau0v, t0v = c_fast.C, float(c_fast.tau), c_fast.t
        wf = fs.seitar_windows(s0, C0v, tau0v, P, alloc3, C0d, C1d, C2d, EP)
        # scalar: rebuild identical state by replaying the same burn-in
        c_sc = ExtendedCounty(biting, cov, mult, treat=10, p_sympt=p)
        for _ in range(200):
            c_sc.step(0.0, seasonal=False)
        c_sc.t = t0v   # windows() uses absolute day for seasonal; align time origin
        ws = scalar_seitar_windows_from(c_sc, alloc3, t0v)
        rel = np.abs(wf - ws) / (np.abs(ws) + 1e-9)
        maxerr = max(maxerr, rel.max())
        print(f"  trial {trial}: fast {wf.round(2)} scalar {ws.round(2)} relerr {rel.max():.2e}")
    print(f"SEITAR max rel err = {maxerr:.2e}")


def scalar_seitar_windows_from(c, alloc3, t0):
    """Scalar SEITAR windows where day index starts at t0 (to match seasonal phase
    used in seitar_windows, which iterates day=0..EP using absolute day)."""
    w = np.zeros(4); idx = 0
    c.t = 0  # seitar_windows uses day 0..EP for seasonal; replicate that origin
    for day in range(EP):
        a = 0.0
        if day == C0d:
            a = alloc3[0]; idx = 1
        elif day == C1d:
            a = alloc3[1]; idx = 2
        elif day == C2d:
            a = alloc3[2]; idx = 3
        c.t = day                      # force absolute-day seasonal forcing
        c.step(a, seasonal=True)
        wj = 0 if day < C0d else idx
        w[wj] += c.E_H + c.I_H
    return w


if __name__ == "__main__":
    main()

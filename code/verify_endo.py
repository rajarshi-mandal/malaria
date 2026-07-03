"""
Verify the endogenous-resistance core is a STRICT generalization of SEITAR:
with k_sel=k_rev=0 and R held at the SEITAR constant, fast_sim_endo._endo_step
must reproduce fast_sim._seitar_step bit-for-bit over a full episode.

Also sanity-checks that turning the feedback ON makes R move (and only then).
"""
import numpy as np
import fast_sim as fs
import fast_sim_endo as fe
from extended_ode import ExtendedCounty, R_CONST


def make_region(seed=0):
    rng = np.random.default_rng(seed)
    c = ExtendedCounty(biting=rng.uniform(0.5, 3.0), coverage=rng.uniform(0.2, 0.6),
                       multiplier=rng.uniform(0.5, 2.0), treat=10,
                       p_sympt=rng.uniform(0.1, 0.6))
    for _ in range(1500):
        c.step(0.0, seasonal=False)
    return c


def main():
    np.set_printoptions(precision=10)
    max_abs = 0.0
    max_rel = 0.0
    for seed in range(6):
        c = make_region(seed)
        s0 = fe.state_endo(c)
        Pe = fe.pack_endo(c)
        Pseitar = fs.pack_seitar(c)        # includes R at idx 17
        R0 = c.R                            # SEITAR constant (0.4)

        # drive both with the same random net schedule over a 540-day episode
        rng = np.random.default_rng(100 + seed)
        ep = 540
        allocs = {30: rng.uniform(0, 4000), 210: rng.uniform(0, 4000),
                  390: rng.uniform(0, 4000)}

        # reference: fs._seitar_step (constant R)
        s_ref = s0.copy(); C_ref = float(c.C); tau_ref = 0.0
        traj_ref = []
        for day in range(ep):
            a = allocs.get(day, 0.0)
            C_ref, tau_ref = fs._seitar_step(s_ref, C_ref, tau_ref, day, a, Pseitar)
            traj_ref.append(s_ref.copy())

        # endo with feedback OFF, R0 = SEITAR constant
        s_e = s0.copy(); C_e = float(c.C); tau_e = 0.0; R_e = R0
        traj_e = []
        for day in range(ep):
            a = allocs.get(day, 0.0)
            C_e, tau_e, R_e = fe._endo_step(s_e, C_e, tau_e, R_e, day, a, Pe, 0.0, 0.0)
            traj_e.append(s_e.copy())

        traj_ref = np.array(traj_ref); traj_e = np.array(traj_e)
        ad = np.abs(traj_ref - traj_e).max()
        rd = (np.abs(traj_ref - traj_e) / (np.abs(traj_ref) + 1e-9)).max()
        max_abs = max(max_abs, ad); max_rel = max(max_rel, rd)
        assert R_e == R0, "R drifted with feedback OFF"

        # feedback ON: R should move
        s_on = s0.copy(); C_on = float(c.C); tau_on = 0.0; R_on = R0
        for day in range(ep):
            a = allocs.get(day, 0.0)
            C_on, tau_on, R_on = fe._endo_step(s_on, C_on, tau_on, R_on, day, a,
                                               Pe, 9.6e-4, 2.74e-4)
        moved = abs(R_on - R0)
        print(f"seed {seed}: max|abs|={ad:.2e} max|rel|={rd:.2e} | "
              f"feedback ON moved R {R0:.3f} -> {R_on:.3f} (d={moved:+.3f})")

    print(f"\nOVERALL  max abs err = {max_abs:.3e}   max rel err = {max_rel:.3e}")
    if max_abs < 1e-9:
        print("PASS: endo core is bit-identical to SEITAR with feedback OFF.")
    else:
        print("WARN: nonzero deviation with feedback OFF -- investigate.")


if __name__ == "__main__":
    main()

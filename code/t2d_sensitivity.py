"""
Tier 2d: global sensitivity analysis of the reproduction number.

Two complementary global methods over biologically-plausible parameter ranges:
  * LHS-PRCC  (Latin-hypercube sampling + partial rank correlation) -- monotonic
    sensitivity, the standard in mathematical epidemiology.
  * Sobol variance decomposition (SALib) -- first-order + total-effect indices.

Target: the basic reproduction number R0 of the extended SEITAR model (and SEITR
for comparison), via the verified next-generation matrix. Shows which parameters
drive transmission and confirms the result is not an artifact of one parameter.

Output: console report + fig_sensitivity.png  (sensitivity_results.csv)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from t2b_stability import R0_seitr, R0_seitar, P as P0

# parameter ranges (biologically plausible); a is the biting rate
PARAMS = {
    "a": (0.05, 1.0), "b": (0.05, 0.5), "c": (0.05, 0.5),
    "mu_M": (1/20, 1/7), "sigma_M": (1/14, 1/8), "sigma_H": (1/21, 1/7),
    "rho": (1/20, 1/5), "delta_H": (0.001, 0.02), "gamma_T": (1/30, 1/14),
    "kappa": (0.1, 0.6), "kappa_A": (0.2, 0.8), "gamma_A": (1/400, 1/150),
    "p_sympt": (0.2, 0.7),
}
NAMES = list(PARAMS)


def make_P(vals):
    p = dict(P0)
    for k, v in zip(NAMES, vals):
        p[k] = v
    return p


def r0_of(vals, model="seitar"):
    p = make_P(vals)
    a = p["a"]
    return (R0_seitar(a, 0.0, p)[0] if model == "seitar" else R0_seitr(a, 0.0, p)[0])


def lhs(n, seed=0):
    rng = np.random.default_rng(seed)
    d = len(NAMES); X = np.zeros((n, d))
    for j, k in enumerate(NAMES):
        lo, hi = PARAMS[k]
        cuts = (np.arange(n) + rng.random(n)) / n
        X[:, j] = lo + cuts[rng.permutation(n)] * (hi - lo)
    return X


def prcc(X, y):
    n, d = X.shape
    Xr = np.column_stack([rankdata(X[:, j]) for j in range(d)])
    yr = rankdata(y)
    out = []
    for i in range(d):
        others = [j for j in range(d) if j != i]
        Z = np.column_stack([np.ones(n), Xr[:, others]])
        bi, *_ = np.linalg.lstsq(Z, Xr[:, i], rcond=None); ri = Xr[:, i] - Z @ bi
        by, *_ = np.linalg.lstsq(Z, yr, rcond=None); ry = yr - Z @ by
        out.append(np.corrcoef(ri, ry)[0, 1])
    return np.array(out)


def main():
    # ---- LHS-PRCC ----
    n = 3000
    X = lhs(n)
    y_ext = np.array([r0_of(X[i], "seitar") for i in range(n)])
    y_seitr = np.array([r0_of(X[i], "seitr") for i in range(n)])
    prcc_ext = prcc(X, y_ext)
    prcc_seitr = prcc(X[:, :10], y_seitr)  # SEITR ignores kappa_A,gamma_A,p_sympt

    print("=== LHS-PRCC on R0 (SEITAR), n=%d ===" % n)
    order = np.argsort(-np.abs(prcc_ext))
    for i in order:
        print(f"  {NAMES[i]:10s} PRCC = {prcc_ext[i]:+.3f}")
    print(f"  R0_SEITAR range over LHS: {y_ext.min():.2f} - {y_ext.max():.2f}, "
          f"median {np.median(y_ext):.2f}")

    # ---- Sobol (SALib) ----
    problem = {"num_vars": len(NAMES), "names": NAMES,
               "bounds": [list(PARAMS[k]) for k in NAMES]}
    Xs = sobol_sample.sample(problem, 1024)
    ys = np.array([r0_of(Xs[i], "seitar") for i in range(len(Xs))])
    Si = sobol_analyze.analyze(problem, ys, print_to_console=False)
    print("\n=== Sobol indices on R0 (SEITAR) ===")
    for i in np.argsort(-Si["ST"]):
        print(f"  {NAMES[i]:10s} S1={Si['S1'][i]:+.3f}  ST={Si['ST'][i]:+.3f}")

    pd.DataFrame({"param": NAMES, "PRCC_seitar": prcc_ext,
                  "Sobol_S1": Si["S1"], "Sobol_ST": Si["ST"]}).to_csv(
        "sensitivity_results.csv", index=False)

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    o = np.argsort(prcc_ext)
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in prcc_ext[o]]
    ax1.barh([NAMES[i] for i in o], prcc_ext[o], color=colors, edgecolor="black", lw=0.5)
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_xlabel("PRCC with R0"); ax1.set_title("LHS-PRCC sensitivity of R0 (SEITAR)")
    ax1.grid(axis="x", alpha=0.3)

    o2 = np.argsort(Si["ST"])
    ax2.barh([NAMES[i] for i in o2], Si["ST"][o2], color="#8172B3", edgecolor="black",
             lw=0.5, label="total-effect ST")
    ax2.barh([NAMES[i] for i in o2], Si["S1"][o2], color="#CCB974", edgecolor="black",
             lw=0.5, alpha=0.8, label="first-order S1")
    ax2.set_xlabel("Sobol index"); ax2.set_title("Sobol variance decomposition of R0")
    ax2.legend(); ax2.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig("fig_sensitivity.png", dpi=150)
    print("\nsaved -> fig_sensitivity.png, sensitivity_results.csv")


if __name__ == "__main__":
    main()

"""
ALGORITHMIC NOVELTY: ITN allocation as constrained Bayesian dual control, and a
value-of-information (VoI) allocation rule that beats standard adaptive baselines.

Motivation (addressing the "system- vs algorithm-novelty" critique). Under NOISY
surveillance, the certainty-equivalent adaptive planner (online point-estimate +
receding-horizon re-planning -- our ARMOR-Adapt, and the standard adaptive-MPC
recipe) suffers a classic dual-control failure, INCOMPLETE LEARNING: if it
underestimates a region's transmission it allocates little there, surveillance
quality stays low there (ITN campaigns bundle case detection), the estimate is
never corrected, and the policy locks onto a suboptimal allocation. The fix is a
NEW objective, not a new fitter: allocate to jointly minimize expected burden AND
maximize decision-relevant information gain, under the simplex budget constraint.

We maintain a Gaussian posterior over each region's log-transmission and compare,
under identical noisy true environments:
  static        population-proportional (WHO practice)
  greedy-obs    allocate proportional to observed incidence (passive adaptation)
  CE-MPC        certainty-equivalent: plan on the posterior MEAN (= ARMOR-Adapt)
  Thompson      plan on a posterior SAMPLE (randomized exploration)
  UCB           plan on an optimistic estimate mu+beta*sd
  ARMOR-IDA     NOVEL: minimize E[burden] - lambda * VoI (decision-aware
                exploration under the budget simplex)
  oracle        plan on the TRUE transmission (upper bound)

Surveillance precision is allocation-coupled: meas_var_i = sigma^2/(k0+k1*effcov_i)
(better case detection where ITN programs operate), which is what makes active
exploration valuable. Reports mean/CVaR infection reduction AND posterior error
over campaigns (the incomplete-learning diagnostic). lambda=0 recovers CE-MPC
(necessity ablation).

Run: python meta_bayesian.py --n-worlds 12 --campaigns 8
"""
import argparse
import numpy as np

import fast_sim_endo as fe
import meta_planners as mp

KAP0, KAP1 = 0.25, 2.0          # surveillance precision: sigma^2/(k0+k1*effcov)
POP, ITERS, ELITE = 240, 8, 24


def true_biting(seed, rng_p):
    e = mp.MetaEnv(np.random.default_rng(seed))
    return e.P[:, 13] * rng_p.lognormal(0.0, 0.30, mp.N)


def build(seed, a, prior_uniform=False):
    e = mp.MetaEnv(np.random.default_rng(seed))
    e.P[:, 13] = a.mean() if prior_uniform else a
    return e


def per_region_pred(plan_env, S, C, tau, R, t, alloc, ndays):
    Sc = np.array([s.copy() for s in S]); Cc = C.copy(); tauc = tau.copy(); Rc = R.copy()
    out = np.empty(mp.N)
    fe.meta_interval_vec_regional(Sc, Cc, tauc, Rc, t, np.asarray(alloc, float),
                                  plan_env.P, plan_env.k_sel, plan_env.k_rev,
                                  plan_env.k_cov, plan_env.W, plan_env.m, ndays, out)
    return out


def plan_campaign(plan_env, S, C, tau, R, t, ndays, theta_plan, mu, v, lam,
                  curcov, NH, sigma_obs, rng):
    """One-campaign budget allocation minimizing total burden - lambda*VoI under
    theta_plan. VoI_i = burden_i * (v_i - v_post_i(alloc_i)); v_post shrinks as the
    allocation raises region i's effective coverage (hence surveillance precision)."""
    plan_env.P[:, 13] = theta_plan
    states = np.asarray(S, np.float64)
    cr = np.array([0], np.int64)
    alphas = np.ones(mp.N)
    best, best_a = np.inf, None
    for it in range(ITERS):
        cand = (rng.dirichlet(alphas, size=POP) * plan_env.budget)
        reg = fe.meta_plan_batch_regional(states, C, tau, R, plan_env.P,
                                          cand.reshape(POP, 1, mp.N), cr, ndays, t,
                                          plan_env.k_sel, plan_env.k_rev, plan_env.k_cov,
                                          plan_env.W, plan_env.m)
        total = reg.sum(1)
        if lam > 0.0:
            effcov = np.minimum(curcov[None, :] + cand / NH[None, :], 1.0)
            meas_var = sigma_obs ** 2 / (KAP0 + KAP1 * effcov)
            v_post = 1.0 / (1.0 / v[None, :] + 1.0 / meas_var)
            voi = (reg * (v[None, :] - v_post)).sum(1)        # decision-weighted info gain
            score = total - lam * voi
        else:
            score = total
        el = np.argsort(score)[:ELITE]
        alphas = np.clip((cand[el] / plan_env.budget).mean(0) * 10 * (it + 1), 1e-2, None)
        if score[el[0]] < best:
            best = score[el[0]]; best_a = cand[el[0]].copy()
    return best_a


def run_world(true_env, algo, seed, sigma_obs, lam, beta, rng, track=False):
    """Closed-loop episode under noisy, allocation-coupled surveillance.
    Returns (cumulative infection ratio, mean |posterior log-a error| trace)."""
    true_env.reset()
    pre = true_env.run_interval(np.zeros(mp.N), mp.FIRST)
    plan_env = build(seed, true_env.P[:, 13])             # structure; P[:,13] overwritten
    NH = true_env.P[:, 17].copy()
    log_true = np.log(true_env.P[:, 13])
    # prior: uninformative, centred on the cross-region mean
    mu = np.full(mp.N, np.log(true_env.P[:, 13].mean()))
    v = np.full(mp.N, 0.30 ** 2 * 4)                      # broad prior
    post = 0.0
    obs = np.empty(mp.N)
    err_trace = []
    for k in range(true_env.K):
        S, C, tau, R, t = true_env.snapshot()
        curcov = C.copy()
        if algo == "static":
            alloc = true_env.pop_static.copy()
        elif algo == "greedy-obs":
            w = np.maximum(np.array([S[i][1] + S[i][2] for i in range(mp.N)]), 1e-6)
            alloc = w / w.sum() * true_env.budget
        else:
            if algo == "oracle":
                theta = true_env.P[:, 13].copy(); use_lam = 0.0
            elif algo == "thompson":
                theta = np.exp(rng.normal(mu, np.sqrt(v))); use_lam = 0.0
            elif algo == "ucb":
                theta = np.exp(mu + beta * np.sqrt(v)); use_lam = 0.0
            else:                                          # CE-MPC and ARMOR-IDA
                theta = np.exp(mu); use_lam = lam if algo == "ida" else 0.0
            alloc = plan_campaign(plan_env, S, C, tau, R, t, true_env.gap, theta,
                                  mu, v, use_lam, curcov, NH, sigma_obs, rng)
        # realized per-region burden on the TRUE env
        tot = fe.meta_interval_vec_regional(true_env.S, true_env.C, true_env.tau,
                                            true_env.R, true_env.t, np.asarray(alloc, float),
                                            true_env.P, true_env.k_sel, true_env.k_rev,
                                            true_env.k_cov, true_env.W, true_env.m,
                                            true_env.gap, obs)
        true_env.t += true_env.gap
        post += tot
        # ---- noisy, allocation-coupled surveillance + Bayesian update ----
        if algo not in ("static", "greedy-obs", "oracle"):
            effcov = np.minimum(curcov + np.asarray(alloc) / NH, 1.0)
            meas_sd = sigma_obs / np.sqrt(KAP0 + KAP1 * effcov)
            obs_noisy = obs * np.exp(rng.normal(0.0, meas_sd))
            # fixed-point fit of a measurement that reproduces the noisy observation
            a_fit = np.exp(mu).copy()
            pe = build(seed, a_fit)
            for _ in range(5):
                pe.P[:, 13] = a_fit
                pr = per_region_pred(pe, S, C, tau, R, t, alloc, true_env.gap)
                a_fit = np.clip(a_fit * (obs_noisy / np.maximum(pr, 1e-9)) ** 0.6, 0.03, 25.0)
            meas_var = meas_sd ** 2
            prec0 = 1.0 / v; precm = 1.0 / meas_var
            v = 1.0 / (prec0 + precm)
            mu = v * (prec0 * mu + precm * np.log(a_fit))
        if track:
            wdec = true_env.P[:, 13] / true_env.P[:, 13].sum()   # decision relevance
            err_trace.append(float(np.sum(wdec * np.abs(mu - log_true))))
    return post / pre, err_trace


def cvar_low(x, q=0.2):
    x = np.sort(np.asarray(x)); n = max(1, int(np.ceil(q * len(x))))
    return float(x[:n].mean())


def main():
    global KAP0, KAP1
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-worlds", type=int, default=12)
    ap.add_argument("--campaigns", type=int, default=8)
    ap.add_argument("--sigma-obs", type=float, default=0.55)
    ap.add_argument("--lam", type=float, default=8.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--kap0", type=float, default=KAP0,
                    help="baseline surveillance precision in unallocated regions (low = sparse)")
    ap.add_argument("--kap1", type=float, default=KAP1)
    ap.add_argument("--algos", nargs="+",
                    default=["static", "greedy-obs", "ce", "thompson", "ucb", "ida", "oracle"])
    ap.add_argument("--out", type=str, default="meta_bayesian.npz")
    args = ap.parse_args()
    KAP0, KAP1 = args.kap0, args.kap1

    rng = np.random.default_rng(0); rng_p = np.random.default_rng(7)
    red = {a: [] for a in ["baseline"] + args.algos}
    traces = {a: [] for a in args.algos}
    print(f"Bayesian dual-control | N={mp.N} | K={args.campaigns} | "
          f"sigma_obs={args.sigma_obs} lam={args.lam} | {args.n_worlds} worlds", flush=True)
    for wld in range(args.n_worlds):
        seed = 10000 + wld
        a_true = true_biting(seed, rng_p)

        def te():
            e = build(seed, a_true); e.K = args.campaigns
            e.camp = np.array([mp.FIRST + j * e.gap for j in range(e.K)], np.int64)
            e.ep = mp.FIRST + e.K * e.gap
            return e
        red["baseline"].append(mp.run_episode(te(), lambda o, k: np.zeros(mp.N))[0])
        for algo in args.algos:
            r, tr = run_world(te(), algo, seed, args.sigma_obs, args.lam, args.beta,
                              rng, track=(algo in ("ce", "ida")))
            red[algo].append(r)
            if tr:
                traces[algo].append(tr)
        print(f"  world {wld+1}/{args.n_worlds} done", flush=True)

    b = np.mean(red["baseline"])
    print(f"\n===== BAYESIAN DUAL-CONTROL ALLOCATION ({args.n_worlds} worlds, "
          f"noisy surveillance) =====")
    print(f"{'method':12s} | {'mean red%':>10s} | {'worst-case (CVaR) red%':>22s}")
    print("-" * 50)
    ratios = {}
    for a in args.algos:
        rr = (b - np.array(red[a])) / b * 100
        ratios[a] = rr
        print(f"{a:12s} | {rr.mean():10.2f} | {cvar_low(rr):22.2f}")
    if all(k in ratios for k in ("ida", "ce", "oracle")):
        adv = ratios["ida"].mean() - ratios["ce"].mean()
        gap = ratios["oracle"].mean() - ratios["ce"].mean()
        print(f"\nARMOR-IDA vs CE-MPC: +{adv:.2f} pp mean (closes "
              f"{adv/max(gap,1e-9)*100:.0f}% of the residual CE->oracle gap)")
    np.savez(args.out, red={a: np.array(red[a]) for a in red},
             traces={a: np.array(traces[a]) for a in traces if traces[a]},
             sigma_obs=args.sigma_obs, lam=args.lam)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

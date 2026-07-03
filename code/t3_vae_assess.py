"""
Tier 3.1 -- Does the VAE earn its place?

Provenance (verified from code, not the manuscript narrative):
  * The resistance trajectories come from IR-Mapper Anopheles WHO susceptibility
    bioassays (pyrethroid percent mosquito mortality, 2010-2017).
  * vae-percent-mortality.py augments these real series (pairwise linear
    interpolation + 9x Gaussian jitter, ~1020 rows) and trains a 3-d-latent VAE
    on the AUGMENTED REAL DATA -- *not* on GAN outputs. The GAN was an abandoned
    earlier attempt (mode collapse). The VAE samples are written to
    `malaria-gan-outputs/random_pm.csv` (10000 x 8) -- a misnamed folder.
  * The simulator (load_resistance_trajectories) interpolates each 8-year row to
    a 180-day series and maps mortality -> resistance impact R in [0,1] via a
    sigmoid; R attenuates ITN efficacy by up to 50%.

So the VAE is effectively a *resampler/smoother* of jittered real data. This
script tests, honestly, whether it changes any allocation conclusion by comparing
three resistance SOURCES under the same calibrated environment and protocol:

  (A) VAE      -- the current pipeline (random_pm.csv).
  (B) RealAug  -- the SAME augmented real IR-Mapper data the VAE was trained on,
                  used directly (drop the VAE entirely).
  (C) Fixed    -- a single mean resistance trajectory (no variability at all),
                  to test whether resistance *variability* matters at all.

For each source we train PPO (best method, Item D) multi-seed and evaluate PPO and
static on held-out envs, with paired Wilcoxon tests. We also compare the R-value
distributions of (A) vs (B) directly (KS test + moments).

Run (single-threaded; background + log):
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python t3_vae_assess.py --seeds 0 1 2 --ppo-steps 60000
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

import calibrated_experiment as CE
from sar_sequential import pm_to_R, EPISODE_DAYS


# --------------------------------------------------------------------------- #
# Build the three resistance libraries (180-day R-impact DataFrames)          #
# --------------------------------------------------------------------------- #
YEARS = list(range(2010, 2018))


def _pm_rows_to_R(pm_rows):
    """Replicate load_resistance_trajectories() from an in-memory array of 8
    yearly percent-mortality values per row -> 180-day resistance-impact df."""
    series = []
    for data in pm_rows:
        data = list(map(float, data))
        interp = []
        for i in range(len(data) - 1):
            seg = np.linspace(data[i], data[i + 1], 26, endpoint=False)
            interp.extend(seg)
        interp.append(data[-1])
        interp = interp[:EPISODE_DAYS]
        series.append([pm_to_R(v) for v in interp])
    return pd.DataFrame(series)


def build_real_pm():
    """Reproduce the real (un-augmented) IR-Mapper pyrethroid mortality series,
    using the same grouping logic as vae-percent-mortality.py."""
    ir = pd.read_csv("malaria-data-for-modeling-dynamics/IR Mapper (Anopheles)/"
                     "insecticide_resistance.csv")
    itn_ir = ir[ir["Insecticide class"] == "pyrethroid"].drop(
        ["Insecticide tested", "Insecticide class"], axis=1)
    itn_ir = itn_ir.map(lambda x: x.replace("\x92", "'") if isinstance(x, str) else x)
    itn_ir["Percent mortality"] = pd.to_numeric(itn_ir["Percent mortality"], errors="coerce")
    itn_ir["No. mosquitoes tested"] = pd.to_numeric(itn_ir["No. mosquitoes tested"], errors="coerce")

    rows = []
    for country in itn_ir["Country"].value_counts().index:
        ircountry = itn_ir[itn_ir["Country"] == country]
        for ircon in ircountry["Concentration (%)"].value_counts().index:
            full = ircountry[ircountry["Concentration (%)"] == ircon].dropna(
                subset=["Percent mortality", "No. mosquitoes tested"])
            if full.empty:
                continue
            mort_yr = full.groupby("Year").apply(
                lambda g: np.average(g["Percent mortality"], weights=g["No. mosquitoes tested"]))
            if len(mort_yr) >= 4:
                rows.append([country] + mort_yr.reindex(range(2010, 2018), fill_value=np.nan).tolist())

    df = pd.DataFrame(rows, columns=["Country"] + YEARS)
    # fill NaNs: country-mean transform, then row/col-mean blend (same as VAE script)
    df[YEARS] = df[YEARS].fillna(df.groupby("Country")[YEARS].transform("mean"))
    row_means = df[YEARS].mean(axis=1)
    col_means = df[YEARS].mean()
    for i in df.index:
        for y in YEARS:
            if pd.isna(df.at[i, y]):
                df.at[i, y] = (row_means[i] + col_means[y]) / 2
    return df


def augment_pm(df, seed=0):
    """Pairwise linear interpolation (alpha .25/.5/.75) + 9x Gaussian jitter,
    exactly mirroring vae-percent-mortality.py -- but with NO VAE."""
    rng = np.random.default_rng(seed)
    aug = []
    for _, group in df.groupby("Country"):
        rws = group[YEARS].values
        if len(rws) < 2:
            continue
        for i, j in itertools.combinations(range(len(rws)), 2):
            for alpha in (0.25, 0.5, 0.75):
                aug.append((alpha * rws[i] + (1 - alpha) * rws[j]).tolist())
    base = df[YEARS].values.tolist() + aug
    jit = []
    for row in base:
        for _ in range(9):
            jit.append(np.clip(np.array(row) + rng.normal(0, 5, size=8), 0, 100).tolist())
    return np.array(base + jit)


def build_libraries(seed=0):
    vae = CE.load_resistance_trajectories("malaria-gan-outputs/random_pm.csv")
    real_pm = build_real_pm()
    realaug = _pm_rows_to_R(augment_pm(real_pm, seed=seed))
    fixed_row = real_pm[YEARS].mean().to_numpy()
    fixed = _pm_rows_to_R(np.tile(fixed_row, (200, 1)))
    return {"VAE": vae, "RealAug": realaug, "Fixed": fixed}, real_pm


# --------------------------------------------------------------------------- #
# Distribution comparison: VAE vs RealAug resistance impact                   #
# --------------------------------------------------------------------------- #
def compare_distributions(libs):
    vae = libs["VAE"].to_numpy().ravel()
    rea = libs["RealAug"].to_numpy().ravel()
    ks, p = stats.ks_2samp(vae, rea)
    print("\n--- Resistance-impact R distribution: VAE vs RealAug ---", flush=True)
    print(f"VAE     R: mean {vae.mean():.3f}  sd {vae.std():.3f}  "
          f"[{np.percentile(vae,2.5):.3f}, {np.percentile(vae,97.5):.3f}]", flush=True)
    print(f"RealAug R: mean {rea.mean():.3f}  sd {rea.std():.3f}  "
          f"[{np.percentile(rea,2.5):.3f}, {np.percentile(rea,97.5):.3f}]", flush=True)
    print(f"KS statistic {ks:.3f}  (p={p:.2e})", flush=True)
    return dict(ks=float(ks), ks_p=float(p),
                vae_mean=float(vae.mean()), realaug_mean=float(rea.mean()),
                vae_sd=float(vae.std()), realaug_sd=float(rea.std()))


# --------------------------------------------------------------------------- #
# Allocation comparison under each resistance source                          #
# --------------------------------------------------------------------------- #
def run_source(name, rdf, seeds, ppo_steps, n_eval):
    print(f"\n========== source: {name} ==========", flush=True)
    center = CE.measure_center(rdf)
    static_red, ppo_red, adv, beats = [], [], [], []
    last = None
    for seed in seeds:
        ppo = CE.train_ppo(rdf, center, ppo_steps, seed)
        pols = {"static": CE.static_policy, "PPO": CE.sb3_pol(ppo, False),
                "baseline": CE.baseline_policy}
        res = CE.evaluate(pols, rdf, n_eval)
        b = res["baseline"].mean()
        sred = (b - res["static"].mean()) / b * 100
        pred = (b - res["PPO"].mean()) / b * 100
        static_red.append(sred); ppo_red.append(pred); adv.append(pred - sred)
        beats.append(float((res["PPO"] < res["static"]).mean() * 100))
        last = res
        print(f"  seed {seed}: static {sred:.2f}%  PPO {pred:.2f}%  adv +{pred-sred:.2f}pp", flush=True)
    _, pv = stats.wilcoxon(last["PPO"], last["static"])
    print(f"  MEAN: static {np.mean(static_red):.2f}%  PPO {np.mean(ppo_red):.2f}%  "
          f"adv +{np.mean(adv):.2f}+/-{np.std(adv):.2f}pp  "
          f"PPO beats static {np.mean(beats):.0f}% envs  p={pv:.2e}", flush=True)
    return dict(static=float(np.mean(static_red)), ppo=float(np.mean(ppo_red)),
                adv=float(np.mean(adv)), adv_sd=float(np.std(adv)),
                beats=float(np.mean(beats)), wilcoxon_p=float(pv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ppo-steps", type=int, default=60000)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--out", type=str, default="t3_vae_results.npz")
    args = ap.parse_args()

    libs, real_pm = build_libraries(seed=0)
    print(f"libraries: VAE {len(libs['VAE'])} | RealAug {len(libs['RealAug'])} "
          f"(from {len(real_pm)} real IR-Mapper series) | Fixed {len(libs['Fixed'])}", flush=True)

    dist = compare_distributions(libs)

    results = {}
    for name in ("VAE", "RealAug", "Fixed"):
        results[name] = run_source(name, libs[name], args.seeds, args.ppo_steps, args.n_eval)

    print("\n========= T3.1 SUMMARY: does the VAE earn its place? =========", flush=True)
    print(f"{'Source':9s} | {'static%':>8s} | {'PPO%':>8s} | {'adv(pp)':>10s} | {'beats%':>7s} | {'p':>9s}",
          flush=True)
    print("-" * 64, flush=True)
    for name in ("VAE", "RealAug", "Fixed"):
        r = results[name]
        print(f"{name:9s} | {r['static']:8.2f} | {r['ppo']:8.2f} | "
              f"+{r['adv']:.2f}+/-{r['adv_sd']:.2f} | {r['beats']:6.0f} | {r['wilcoxon_p']:.2e}",
              flush=True)
    print(f"\nVAE-vs-RealAug PPO advantage diff: "
          f"{abs(results['VAE']['adv']-results['RealAug']['adv']):.2f} pp "
          f"(KS {dist['ks']:.3f}). If small, the VAE adds nothing over direct "
          f"augmentation of the real data.", flush=True)

    np.savez(args.out, results=results, dist=dist)
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

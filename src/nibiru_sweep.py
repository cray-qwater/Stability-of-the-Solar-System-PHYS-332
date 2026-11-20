"""
Nibiru stability study.

  1. Baseline "myth" Nibiru (5 M_J, P ~ 3600 yr, q = 2.5 AU): one detailed run, compared
     against the unperturbed system. Saves the trajectory for animate.py.
  2. Sweep over Nibiru mass x perihelion distance, with K random orbit orientations per
     cell (the outcome depends on where the planets are during the passage). Reports the
     fraction of runs that end unstable (ejection / Sun collision) or with a close encounter.

Usage:
  python nibiru_sweep.py                 # baseline + full sweep
  python nibiru_sweep.py --skip-sweep    # baseline only
  python nibiru_sweep.py --quick         # small sweep for testing
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
from paths import DATA, FIGURES
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solar_system import nibiru_params, M_JUPITER
from stability import run_scenario, print_summary

OUTCOMES = ["stable", "encounter", "unstable", "unresolved"]   # code = index
MASSES_MJ = [0.01, 0.1, 1.0, 5.0, 20.0]
Q_AU = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
INC_MAX_DEG = 20.0     # sampled inclinations are uniform in [0, INC_MAX_DEG]


# -----------------------------------------------------------------------------
# 1. Baseline myth run
# -----------------------------------------------------------------------------
def baseline(dt, extra_years):
    print("=== Unperturbed solar system ===")
    ref = run_scenario(None, dt=dt, extra_years=extra_years + 77.0)
    print_summary(ref)

    print("\n=== Myth Nibiru ===")
    p = nibiru_params()
    res = run_scenario(p, dt=dt, extra_years=extra_years, record_every=20, keep_history=True)
    print_summary(res)
    print("  change in a vs unperturbed run:  " +
          ", ".join(f"{k} {100 * (o['a'] - ref['orbits'][k]['a']) / ref['orbits'][k]['a']:+.2f}%"
                    for k, o in res["orbits"].items()))

    h = res["history"]
    np.savez(DATA / "nibiru_myth_run.npz", t=h["t"], r=h["r"], names=res["names"], masses=res["masses"],
             E=h["E"], nibiru_q=p["q"], nibiru_a=p["a"], nibiru_mass=p["mass"])
    print("  saved data/nibiru_myth_run.npz (for animate.py)")
    plot_baseline(h, res["names"], p)


def plot_baseline(h, names, p):
    t, r = h["t"], h["r"]
    rel = r - r[:, [0]]                                  # heliocentric
    nib = len(names) - 1
    colors = plt.cm.tab10(np.arange(10))

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), layout="constrained")
    for ax, bodies, lim, title in [(axes[0], range(1, 5), 3, "Inner planets"),
                                   (axes[1], range(5, 9), 35, "Outer planets"),
                                   (axes[2], range(1, 9), 65, "Whole passage")]:
        ax.plot(0, 0, "*", color="gold", markersize=12)
        for i in bodies:
            ax.plot(rel[:, i, 0], rel[:, i, 1], lw=0.6, color=colors[i - 1], label=names[i])
        ax.plot(rel[:, nib, 0], rel[:, nib, 1], lw=1.8, color="black", ls="--", label="Nibiru")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x (AU, heliocentric)")
        ax.set_ylabel("y (AU)")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Myth Nibiru: {p['mass'] / M_JUPITER:.0f} M_J, q = {p['q']} AU, a = {p['a']} AU, "
                 f"{t[-1]:.0f} yr")
    plt.savefig(FIGURES / "nibiru_myth_orbits.png", dpi=150)
    plt.close(fig)
    print("  saved outputs/figures/nibiru_myth_orbits.png")


# -----------------------------------------------------------------------------
# 2. Sweep
# -----------------------------------------------------------------------------
def _worker(job):
    (im, iq, k), p, dt, extra_years = job
    res = run_scenario(p, dt=dt, extra_years=extra_years)
    first = next((e for e in res["events"] if e[1] in ("ejection", "sun_collision", "unresolved")), None)
    return (im, iq, k), OUTCOMES.index(res["outcome"]), res["max_dE"], \
        min(res["min_dist"].values()), (f"{first[1]}:{first[2]}" if first else "")


def sweep(masses_mj, qs, K, dt, extra_years, workers, seed=332):
    rng = np.random.default_rng(seed)
    jobs, orient = [], np.zeros((len(masses_mj), len(qs), K, 3))
    for im, mj in enumerate(masses_mj):
        for iq, q in enumerate(qs):
            for k in range(K):
                inc = np.radians(rng.uniform(0, INC_MAX_DEG))
                Om, om = rng.uniform(0, 2 * np.pi, 2)
                orient[im, iq, k] = inc, Om, om
                p = nibiru_params(mass=mj * M_JUPITER, q=q, inc=inc, Omega=Om, omega=om)
                jobs.append(((im, iq, k), p, dt, extra_years))

    shape = (len(masses_mj), len(qs), K)
    code = np.zeros(shape, int)
    max_dE = np.zeros(shape)
    min_d = np.zeros(shape)
    first_event = np.empty(shape, dtype=object)

    print(f"\n=== Sweep: {len(jobs)} runs on {workers} workers ===")
    t0 = time.perf_counter()
    with Pool(workers) as pool:
        for n, (idx, c, dE, d, ev) in enumerate(pool.imap_unordered(_worker, jobs), 1):
            code[idx], max_dE[idx], min_d[idx], first_event[idx] = c, dE, d, ev
            if n % max(1, len(jobs) // 20) == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}  ({time.perf_counter() - t0:.0f} s)")

    np.savez(DATA / "nibiru_sweep.npz", masses_mj=masses_mj, qs=qs, code=code, outcomes=OUTCOMES,
             max_dE=max_dE, min_dist=min_d, first_event=first_event.astype(str), orient=orient,
             dt=dt, extra_years=extra_years)
    print("  saved data/nibiru_sweep.npz")
    report(masses_mj, qs, code, first_event)
    plot_heatmap(masses_mj, qs, code, dt, extra_years)


def report(masses_mj, qs, code, first_event):
    K = code.shape[2]
    print(f"\n  Fraction unstable (ejection/collision) per cell, K = {K}; [u = unresolved runs]")
    print("  M_J \\ q " + "".join(f"{q:>9.1f}" for q in qs))
    for im, mj in enumerate(masses_mj):
        row = ""
        for iq in range(len(qs)):
            c = code[im, iq]
            cell = f"{np.mean(c == 2):.2f}"
            nu = np.sum(c == 3)
            row += f"{cell + (f'[{nu}u]' if nu else ''):>9s}"
        print(f"  {mj:7.2f} " + row)
    ev = [e for e in first_event.ravel() if e]
    if ev:
        kinds, counts = np.unique(ev, return_counts=True)
        print("  first terminating events: " + ", ".join(f"{k} x{c}" for k, c in zip(kinds, counts)))


def plot_heatmap(masses_mj, qs, code, dt, extra_years):
    K = code.shape[2]
    resolved = np.maximum(np.sum(code != 3, axis=2), 1)
    panels = [("P(unstable): ejection or Sun collision", np.sum(code == 2, axis=2) / resolved),
              ("P(close encounter or worse)", np.sum((code == 1) | (code == 2), axis=2) / resolved)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for ax, (title, frac) in zip(axes, panels):
        im = ax.imshow(frac, cmap="Reds", vmin=0, vmax=1, origin="lower", aspect="auto")
        for i in range(len(masses_mj)):
            for j in range(len(qs)):
                nu = np.sum(code[i, j] == 3)
                txt = f"{frac[i, j]:.2f}" + (f"\n({nu} unres.)" if nu else "")
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if frac[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(qs)), [f"{q:g}" for q in qs])
        ax.set_yticks(range(len(masses_mj)), [f"{m:g}" for m in masses_mj])
        ax.set_xlabel("Nibiru perihelion q (AU)")
        ax.set_ylabel("Nibiru mass (M_Jupiter)")
        ax.set_title(title)
    fig.colorbar(im, ax=axes, label="fraction of resolved runs")
    fig.suptitle(f"Solar-system stability after one Nibiru passage (a = 235 AU, K = {K} orientations "
                 f"per cell, dt = {dt} yr, +{extra_years:g} yr after passage)")
    plt.savefig(FIGURES / "nibiru_stability_heatmap.png", dpi=150)
    plt.close(fig)
    print("  saved outputs/figures/nibiru_stability_heatmap.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dt", type=float, default=0.001, help="time step (yr)")
    ap.add_argument("--extra-years", type=float, default=300.0, help="years simulated after the passage")
    ap.add_argument("--K", type=int, default=8, help="random orientations per grid cell")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--quick", action="store_true", help="3x3 grid, K=2, 50 extra years")
    args = ap.parse_args()

    masses_mj, qs = MASSES_MJ, Q_AU
    if args.quick:
        masses_mj, qs, args.K, args.extra_years = [0.1, 5.0, 20.0], [1.0, 5.0, 20.0], 2, 50.0

    if not args.skip_baseline:
        baseline(args.dt, args.extra_years)
    if not args.skip_sweep:
        sweep(masses_mj, qs, args.K, args.dt, args.extra_years, args.workers)

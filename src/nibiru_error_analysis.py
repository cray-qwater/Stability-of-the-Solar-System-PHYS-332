"""
Error analysis for the Nibiru simulation (myth Nibiru: 5 M_J, q = 2.5 AU, a = 235 AU).

  1. Conservation: |dE/E0| and |dL|/|L0| over the run, with perihelion marked.
  2. dt convergence: the same run at several dt. Energy error vs dt (expect slope ~4 for
     Yoshida 4) and planet position error vs the finest-dt reference.
  3. (--sweep-check) Re-run a sample of sweep runs at dt/2 and see whether the
     stable / encounter / unstable labels change.

Usage:
  python nibiru_error_analysis.py                        # parts 1 + 2
  python nibiru_error_analysis.py --extra-years 50       # faster
  python nibiru_error_analysis.py --sweep-check          # part 3 (needs nibiru_sweep.npz)
  python nibiru_error_analysis.py --sweep-check --n-check 60
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import DATA, FIGURES
from solar_system import nibiru_params, passage_duration, M_JUPITER
from stability import run_scenario

OUTCOMES = ["stable", "encounter", "unstable", "unresolved"]   # same codes as nibiru_sweep.py

DTS = [0.004, 0.002, 0.001, 0.0005, 0.00025]   # last entry = reference run
DT_USED = 0.001                                # dt used in the sweep
SAMPLE_DT = 0.04                               # yr between recorded samples (same for every dt)
MONITOR_DT = 0.02                              # yr between stability-monitor checks


# -----------------------------------------------------------------------------
# Parts 1 + 2: conservation and dt convergence on the myth run
# -----------------------------------------------------------------------------
def convergence(extra_years):
    p = nibiru_params()
    t_peri = passage_duration(p) / 2.0          # Nibiru reaches perihelion mid-passage

    runs = {}
    for dt in DTS:
        t0 = time.perf_counter()
        runs[dt] = run_scenario(p, dt=dt, extra_years=extra_years,
                                record_every=int(round(SAMPLE_DT / dt)),
                                monitor_every=int(round(MONITOR_DT / dt)),
                                keep_history=True)
        runs[dt]["wall"] = time.perf_counter() - t0
        print(f"dt = {dt:<8g} finished in {runs[dt]['wall']:6.0f} s  -> {runs[dt]['outcome']}")

    ref = runs[DTS[-1]]
    h_ref = ref["history"]
    rel_ref = h_ref["r"] - h_ref["r"][:, [0]]                      # heliocentric positions
    scale = np.linalg.norm(rel_ref[:, 1:9], axis=-1).mean(axis=0)  # mean orbital radius of each planet

    # ---- table + data for the plots ----
    print(f"\nReference: dt = {DTS[-1]}.  Planets only (Mercury..Neptune).")
    print(f"{'dt (yr)':>9s} {'outcome':>10s} {'max|dE/E0|':>12s} {'final pos err':>14s} {'max |da|/a':>12s}")
    max_dE, pos_err = {}, {}
    for dt, res in runs.items():
        h = res["history"]
        E = h["E"]
        max_dE[dt] = np.max(np.abs((E - E[0]) / E[0]))

        n = min(len(h["t"]), len(h_ref["t"]))                      # same sample times, so compare index by index
        rel = h["r"][:n] - h["r"][:n, [0]]
        err = np.linalg.norm(rel[:, 1:9] - rel_ref[:n, 1:9], axis=-1) / scale   # fraction of orbital radius
        pos_err[dt] = (h["t"][:n], err.max(axis=1))                # worst planet at each time

        da = max(abs(o["a"] - ref["orbits"][k]["a"]) / abs(ref["orbits"][k]["a"])
                 for k, o in res["orbits"].items())
        print(f"{dt:>9g} {res['outcome']:>10s} {max_dE[dt]:>12.2e} {pos_err[dt][1][-1]:>14.2e} {da:>12.2e}")

    # ---- figures: three separate files ----
    def save(fig, name):
        fig.savefig(FIGURES / name, dpi=150)
        plt.close(fig)
        print(f"saved outputs/figures/{name}")

    sub = f"Myth Nibiru, Yoshida 4, passage + {extra_years:g} yr"

    # (1) Conservation for the dt used in the sweep
    h = runs[DT_USED]["history"]
    dE = np.abs((h["E"] - h["E"][0]) / h["E"][0])
    dL = np.linalg.norm(h["L"] - h["L"][0], axis=1) / np.linalg.norm(h["L"][0])

    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.semilogy(h["t"][1:], np.maximum(dE[1:], 1e-17), label="|ΔE/E0|")
    ax.semilogy(h["t"][1:], np.maximum(dL[1:], 1e-17), label="|ΔL|/|L0|")
    ax.axvline(t_peri, color="gray", ls=":", label="Nibiru perihelion")
    ax.set_xlabel("t (yr)")
    ax.set_ylabel("relative error")
    ax.set_title(f"Conservation during Nibiru passage (dt = {DT_USED})\n{sub}")
    ax.legend()
    save(fig, "nibiru_conservation.png")

    # (2) Energy error vs dt, with ideal 4th-order slope
    dts = np.array(DTS)
    errs = np.array([max_dE[dt] for dt in DTS])
    slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]

    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.loglog(dts, errs, "o-", label=f"measured (slope {slope:.1f})")
    ax.loglog(dts, errs[0] * (dts / dts[0])**4, "k--", alpha=0.4, label="ideal slope 4")
    ax.set_xlabel("dt (yr)")
    ax.set_ylabel("max |ΔE/E0| over run")
    ax.set_title(f"Convergence of energy error\n{sub}")
    ax.legend()
    save(fig, "nibiru_energy_convergence.png")
    print(f"\nMeasured energy-error order ≈ {slope:.2f} (Yoshida 4: expect ~4)")

    # (3) Position error of each dt vs the reference
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    for dt in DTS[:-1]:
        t, e = pos_err[dt]
        ax.semilogy(t[1:], np.maximum(e[1:], 1e-17), label=f"dt = {dt:g}")
    ax.axvline(t_peri, color="gray", ls=":", label="Nibiru perihelion")
    ax.set_xlabel("t (yr)")
    ax.set_ylabel("worst planet: position error / orbital radius")
    ax.set_title(f"Position error vs reference (dt = {DTS[-1]:g})\n{sub}")
    ax.legend()
    save(fig, "nibiru_position_error.png")


# -----------------------------------------------------------------------------
# Part 3: do the sweep's outcome labels survive a smaller dt?
# -----------------------------------------------------------------------------
def _worker(job):
    idx, p, dt, extra_years = job
    res = run_scenario(p, dt=dt, extra_years=extra_years, monitor_every=int(round(0.01 / dt)))
    return idx, OUTCOMES.index(res["outcome"])


def sweep_check(n_check, workers, seed=1):
    d = np.load(DATA / "nibiru_sweep.npz")
    masses_mj, qs, code, orient = d["masses_mj"], d["qs"], d["code"], d["orient"]
    dt_old, extra_years = float(d["dt"]), float(d["extra_years"])
    dt_new = dt_old / 2.0

    # Half the sample from non-stable runs (where the labels matter), half from stable ones
    rng = np.random.default_rng(seed)
    flat = np.arange(code.size)
    nonstable = flat[code.ravel() != 0]
    stable = flat[code.ravel() == 0]
    pick = np.concatenate([
        rng.choice(nonstable, min(len(nonstable), n_check // 2), replace=False),
        rng.choice(stable, min(len(stable), n_check - min(len(nonstable), n_check // 2)), replace=False)])

    jobs = []
    for f in pick:
        im, iq, k = np.unravel_index(f, code.shape)
        inc, Om, om = orient[im, iq, k]
        p = nibiru_params(mass=masses_mj[im] * M_JUPITER, q=qs[iq], inc=inc, Omega=Om, omega=om)
        jobs.append(((im, iq, k), p, dt_new, extra_years))

    print(f"Re-running {len(jobs)} sweep runs at dt = {dt_new:g} (sweep used {dt_old:g}) on {workers} workers")
    t0 = time.perf_counter()
    with Pool(workers) as pool:
        results = pool.map(_worker, jobs)
    print(f"done in {time.perf_counter() - t0:.0f} s")

    M = np.zeros((4, 4), int)                       # rows: original label, cols: label at dt/2
    changed = []
    for idx, new in results:
        old = code[idx]
        M[old, new] += 1
        if old != new:
            changed.append((idx, old, new))

    print(f"\nLabel at dt = {dt_old:g} (rows) vs label at dt = {dt_new:g} (columns):")
    print(" " * 12 + "".join(f"{o:>12s}" for o in OUTCOMES))
    for i, o in enumerate(OUTCOMES):
        print(f"{o:>12s}" + "".join(f"{M[i, j]:>12d}" for j in range(4)))
    agree = np.trace(M) / M.sum()
    print(f"\nAgreement: {100 * agree:.1f}% ({np.trace(M)}/{M.sum()})")
    for (im, iq, k), old, new in changed:
        print(f"  changed: M = {masses_mj[im]:g} M_J, q = {qs[iq]:g} AU, orientation {k}: "
              f"{OUTCOMES[old]} -> {OUTCOMES[new]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra-years", type=float, default=100.0, help="years simulated after the passage")
    ap.add_argument("--sweep-check", action="store_true", help="run part 3 instead of parts 1 + 2")
    ap.add_argument("--n-check", type=int, default=40, help="number of sweep runs to re-check")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    if args.sweep_check:
        sweep_check(args.n_check, args.workers)
    else:
        convergence(args.extra_years)
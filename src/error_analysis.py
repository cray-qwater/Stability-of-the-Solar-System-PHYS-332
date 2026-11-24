# Error Analysis for the main solar system simulation
import numpy as np
from paths import FIGURES
import matplotlib.pyplot as plt
from engine import NBodySystem as nbs
from solar_system import build_system, NAMES as names

# Always build a fresh system:
#   run() mutates the state in place
def make_system():
    return build_system()[0]


def run_quiet(dt, years, method=nbs.step_yoshida4, sample_dt=0.04):
    """
    Run and record at fixed PHYSICAL sample times (every sample_dt years), so runs with
    different dt produce arrays of the same shape and can be compared sample by sample.
    sample_dt must be a multiple of dt.
    """
    every = int(round(sample_dt / dt))
    assert abs(every * dt - sample_dt) < 1e-12, "sample_dt must be a multiple of dt"
    steps = int(round(years / dt))
    return make_system().run(dt, steps, method=method, record_every=every, verbose=False)

# Max Energy Error of
def max_energy_error(out):
    E = out["E"]
    return np.max(np.abs((E - E[0]) / E[0]))


dts = np.array([0.008, 0.004, 0.002, 0.001])   # all divide 0.04 evenly
years = 20                                       # short on purpose: 4 dts x 2 methods


fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
for label, method, p in [("Leapfrog", nbs.leapfrog, 2), ("Yoshida 4", nbs.step_yoshida4, 4)]:
    errs = np.array([max_energy_error(run_quiet(dt, years, method)) for dt in dts])
    slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    print(f"{label:10s} measured order ≈ {slope:.2f} (expected {p})")
    line, = ax.loglog(dts, errs, "o-", label=f"{label} (slope {slope:.1f})")
    ax.loglog(dts, errs[0] * (dts / dts[0])**p, "--", color=line.get_color(), alpha=0.4)  # reference slope
ax.set_xlabel("dt (yr)")
ax.set_ylabel("max |ΔE/E0| over run")
ax.set_title(f"Convergence test, {years} yr (dashed = ideal slope)")
ax.legend()
plt.savefig(FIGURES / "convergence.png", dpi=150)


years  = 20
dt     = 0.001
dt_ref = 0.00025


out     = run_quiet(dt, years)
out_ref = run_quiet(dt_ref, years)


pos_err = np.linalg.norm(out["r"] - out_ref["r"], axis=-1)            # (M, N) in AU
a_mean  = np.linalg.norm(out_ref["r"] - out_ref["r"][:, [0]], axis=-1).mean(axis=0)  # mean distance from Sun
rel_err = pos_err / np.where(a_mean == 0, 1, a_mean)                  # fraction of orbit size


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained", sharex=True)
for i in range(1, len(names)):                                         # skip the Sun
    ax1.semilogy(out["t"][1:], pos_err[1:, i], label=names[i])
    ax2.semilogy(out["t"][1:], rel_err[1:, i], label=names[i])
ax1.set_ylabel("position error (AU)")
ax2.set_ylabel("position error / orbital radius")
for ax in (ax1, ax2):
    ax.set_xlabel("t (yr)")
ax1.set_title(f"Error vs reference (dt={dt} vs {dt_ref})")
ax1.legend(ncol=2)
plt.savefig(FIGURES / "position_error.png", dpi=150)
from engine import NBodySystem as nbs
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 9-body solar system from JPL Horizons (run Initial_Conditions.py first)
# -----------------------------------------------------------------------------
ic = np.load("initial_conditions.npz")   # barycentric, ecliptic frame
names = list(ic["names"])
masses = [1.0,        # sun
          1.66e-7,    # mercury
          2.45e-6,    # venus
          3.04e-6,    # earth + moon
          3.23e-7,    # mars
          9.55e-4,    # jupiter
          2.86e-4,    # saturn
          4.37e-5,    # uranus
          5.15e-5,    # neptune
]                     # in solar masses

# Positions in AU, velocities in AU/yr
sys_9 = nbs(masses=masses, positions=ic["r"], velocities=ic["v"])

dt    = 0.001   # yr (about 240 steps per Mercury orbit)
years = 200     # longer than one Neptune orbit (165 yr)
out = sys_9.run(dt, int(years / dt), method=nbs.step_yoshida4, record_every=37)   # 37 so samples don't repeat at the same spots each orbit

# ---- Checks ----
E0, L0 = out["E"][0], out["L"][0]
rel_dE = np.abs((out["E"] - E0) / E0)
rel_dL = np.linalg.norm(out["L"] - L0, axis=1) / np.linalg.norm(L0)
print(f"Max relative energy error:           {rel_dE.max():.3e}")
print(f"Max relative angular momentum error: {rel_dL.max():.3e}")

# Average period = total time / number of orbits completed around the Sun
T_known = [0.241, 0.615, 1.000, 1.881, 11.86, 29.46, 84.01, 164.8]   # yr
for i in range(1, 9):
    rel = out["r"][:, i] - out["r"][:, 0]                  # position relative to Sun
    angle = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))    # total angle travelled
    T_sim = 2 * np.pi * out["t"][-1] / abs(angle[-1] - angle[0])
    print(f"{names[i]:8s} period {T_sim:8.3f} yr (known: {T_known[i-1]:7.3f} yr)")

# Conservation Error Checks (It adds total linear momentum, which should stay constant in the barycentric frame)
def conservation_errors(out, m):
    """Relative drift of energy, angular momentum, and linear momentum."""
    E, L = out["E"], out["L"]
    dE = np.abs((E - E[0]) / E[0])
    dL = np.linalg.norm(L - L[0], axis=1) / np.linalg.norm(L[0])

    P = np.einsum("i,tij->tj", m, out["v"])                    # total momentum (M, 3)
    P_scale = np.sum(m * np.linalg.norm(out["v"][0], axis=1))  # sum of |p_i| at t=0
    dP = np.linalg.norm(P - P[0], axis=1) / P_scale
    return {"Energy |ΔE/E0|": dE, "Ang. momentum |ΔL|/|L0|": dL, "Lin. momentum |ΔP|/Σ|p|": dP}

errs = conservation_errors(out, np.array(masses))
for name, e in errs.items():
    print(f"Max {name:28s} {e.max():.3e}")

fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
for name, e in errs.items():
    ax.semilogy(out["t"][1:], np.maximum(e[1:], 1e-17), label=name)   # floor avoids log(0)
ax.axhline(np.finfo(float).eps, color="gray", ls=":", label="machine epsilon")
ax.set_xlabel("t (yr)")
ax.set_ylabel("relative error")
ax.set_title("Conservation errors (Yoshida 4)")
ax.legend()
plt.savefig("conservation_errors.png", dpi=150)

# ---- Plots ----
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), layout="constrained")

# Orbits colored by time so precession shows up; inner and outer planets on separate panels
for ax, bodies, lim, title in [(ax1, range(1, 5), 2, "Inner planets"),
                               (ax2, range(5, 9), 35, "Outer planets")]:
    ax.plot(out["r"][:, 0, 0], out["r"][:, 0, 1], "*", color="gold", markersize=10)  # Sun
    for i in bodies:
        sc = ax.scatter(out["r"][:, i, 0], out["r"][:, i, 1], c=out["t"], cmap="viridis", s=1)
        ax.annotate(names[i], out["r"][-1, i, :2], ha="center", va="bottom")      # label at final position
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.set_title(f"{title}, {years} yr")
fig.colorbar(sc, ax=[ax1, ax2], label="t (yr)")

ax3.semilogy(out["t"][1:], rel_dE[1:])
ax3.set_xlabel("t (yr)")
ax3.set_ylabel("|ΔE / E0|")
ax3.set_title("Energy error (Yoshida 4)")

plt.savefig("solar_system_9body.png", dpi=150)

# ----Ian Personal notes----
# use jpl horizons state vectors
# Make test plots with few objects
# Make full plots
# Include animation
# slides

# ---- Caiden Workspace ----

# Error Analysis

# Result Analysis with Plots

# Validation

# Computation times/Simulation time

# 3d probability sampling with mass/velocity/radial distance

# Find most unstable values(within bounds?).

# Eccentricity Stability analysis
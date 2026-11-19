from engine import NBodySystem as nbs

# -----------------------------------------------------------------------------
# NBodySystem layout
# -----------------------------------------------------------------------------
# __init__(masses, positions, velocities)
#   - store masses (N,), positions (N,3), velocities (N,3) as numpy arrays
#   - store N = number of bodies
#
# get_accelerations(r_pos=None)
#   - default to the current positions if none are given
#   - for each body, sum the gravitational pull from every other body
#   - skip self-interaction
#   - return accelerations (N,3)
#
# total_energy()
#   - compute kinetic energy from masses and velocities
#   - compute potential energy from all pairwise distances (no double counting)
#   - return T + V
#
# total_angular_momentum()
#   - sum m * (r x v) over all bodies
#   - return the L vector (3,)
#
# leapfrog(dt)
#   - half kick velocities
#   - full drift positions
#   - half kick velocities with the new accelerations
#
# step_yoshida4(dt)
#   - build the Yoshida weights w0, w1 from 2^(1/3)
#   - build the drift coefficients c (4) and kick coefficients d (4, last = 0)
#   - loop 4 sub-steps:
#       - drift positions by c[i] * dt
#       - if d[i] != 0: kick velocities by d[i] * dt using new accelerations
#
# Usage: make a system -> record the starting energy -> call step_yoshida4(dt)
#        in a loop -> record positions and energy drift along the way
# -----------------------------------------------------------------------------



import numpy as np
import matplotlib.pyplot as plt
from engine import G

# -----------------------------------------------------------------------------
# Test case: Sun + Mercury, Venus, Earth on circular, coplanar orbits
# -----------------------------------------------------------------------------
# Each planet starts on the +x axis at its semi-major axis a (AU), moving in +y
# with circular speed v = sqrt(G * M_sun / a). Expected periods from Kepler's
# third law: T = a^1.5 yr. Swap in JPL Horizons state vectors later.

names  = ["Sun", "Mercury", "Venus", "Earth"]
masses = [1.0,         #sun
          1.66e-7,   #mercury
          2.45e-6,   #venus
          3.04e-6,   #earth
#          3.23e-7,   #mars
#          9.55e-4,   #jupiter
#          2.86e-4,   #saturn
#          4.37e-5,   #uranus
#          5.15e-5    #neptune
#          xxxxxxx    #nibiru
]
a = np.array([0.0, 0.387, 0.723, 1.0])   # semi-major axes in AU

positions  = np.zeros((4, 3))
velocities = np.zeros((4, 3))
positions[1:, 0]  = a[1:]
velocities[1:, 1] = np.sqrt(G * masses[0] / a[1:])

# Give the Sun the opposite momentum so the system's center of mass stays put
velocities[0] = -np.sum(np.array(masses)[1:, None] * velocities[1:], axis=0) / masses[0]

sys_4 = nbs(masses=masses, positions=positions, velocities=velocities)

dt    = .01  # yr (about 240 steps per Mercury orbit)
years = 1e3
out = sys_4.run(dt, int(years / dt), method=nbs.step_yoshida4, record_every=10)

# ---- Checks ----
E0, L0 = out["E"][0], out["L"][0]
rel_dE = np.abs((out["E"] - E0) / E0)
rel_dL = np.linalg.norm(out["L"] - L0, axis=1) / np.linalg.norm(L0)
print(f"Max relative energy error:           {rel_dE.max():.3e}")
print(f"Max relative angular momentum error: {rel_dL.max():.3e}")

# Orbital radius (relative to the Sun) should stay near a for circular orbits,
# and the period found from the first return to +x should match a^1.5
for i in range(1, 4):
    rel = out["r"][:, i] - out["r"][:, 0]
    radius = np.linalg.norm(rel, axis=1)
    angle = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    T_sim = np.interp(2 * np.pi, angle, out["t"])
    print(f"{names[i]:8s} radius {radius.min():.4f}-{radius.max():.4f} AU, "
          f"period {T_sim:.4f} yr (Kepler: {a[i]**1.5:.4f} yr)")

# ---- Plots ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# Color each planet's recorded positions by time so orbit shifts (precession) show up
ax1.plot(out["r"][:, 0, 0], out["r"][:, 0, 1], "*", color="gold", markersize=10)
for i in range(1, 4):
    sc = ax1.scatter(out["r"][:, i, 0], out["r"][:, i, 1], c=out["t"], cmap="viridis", s=1)
    ax1.annotate(names[i], (0, a[i]), ha="center", va="bottom")
fig.colorbar(sc, ax=ax1, label="t (yr)")
ax1.set_aspect("equal")
ax1.set_xlabel("x (AU)")
ax1.set_ylabel("y (AU)")
ax1.set_title(f"Inner solar system, {years} yr")

ax2.semilogy(out["t"][1:], rel_dE[1:])
ax2.set_xlabel("t (yr)")
ax2.set_ylabel("|ΔE / E0|")
ax2.set_title("Energy error (Yoshida 4)")

plt.tight_layout()
plt.savefig("test_inner_planets.png", dpi=150)

# ----Personal notes----
# use jpl horizons state vectors
# Make test plots with few objects
# Make full plots
# Include animation
# Caiden: error analysis, result analysis with plots, validation, computation times/simulation time, 3d probability sampling with mass/velocity/radial distance? Find most unstable values(within bounds?).
# eccentricity stability analysis
# slides
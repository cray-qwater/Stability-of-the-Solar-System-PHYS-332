import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from integrators import NBodySystem, G

# 2-body Sun-Earth initial conditions (e = 0.2, a = 1.0 AU)
e = 0.2
mu = G * (1.0 + 3.04e-6)
r_peri = 1.0 * (1.0 - e)
v_peri = np.sqrt(mu * (1.0 + e) / r_peri)

# Log every 23 steps (0.23 yr) so we sample multiple phases per orbit instead of once per year
results = {}
for method in ['rk4', 'leapfrog', 'yoshida4']:
    sys = NBodySystem([1.0, 3.04e-6], [[0, 0, 0], [r_peri, 0, 0]], [[0, 0, 0], [0, v_peri, 0]])
    sys.com_frame()
    t, r, E, L = sys.integrate(t_max=2000.0, dt=0.01, method=method, log_interval=23)
    results[method] = (t[1:], np.abs((E[1:] - E[0]) / E[0]))

# Larger figure size and cleaner font sizes for presentation slides
plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.2, 2000)
ax.set_ylim(1e-15, 1e-2)
ax.set_title('Two-Body Kepler Test (e = 0.2)')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Relative Energy Error')
ax.grid(True, which='major', alpha=0.6)

line_rk4, = ax.plot([], [], color='black', linewidth=2, label='RK4')
line_lf, = ax.plot([], [], color='tab:blue', linewidth=1.2, alpha=0.85, label='Leapfrog')
line_y4, = ax.plot([], [], color='tab:orange', linewidth=2, label='Yoshida4')
ax.legend(loc=0, fontsize=12)
plt.tight_layout()

n_points = len(results['rk4'][0])
frames = np.linspace(2, n_points, 100, dtype=int)

def update(idx):
    for line, m in [(line_rk4, 'rk4'), (line_lf, 'leapfrog'), (line_y4, 'yoshida4')]:
        t_arr, err_arr = results[m]
        line.set_data(t_arr[:idx], err_arr[:idx])
    return line_rk4, line_lf, line_y4

anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
anim.save('kepler_test.gif', writer=PillowWriter(fps=20), dpi=200)
print("Saved kepler_test.gif!")


formulas = {
    "formula1.png": r"$F(\mathbf{p}, \mathbf{q}) = K(\mathbf{p}) + U(\mathbf{q})$",
    "formula2.png": r"$\hat{H}_{dt} = H + \mathcal{O}(dt^p)$",
}

for filename, tex in formulas.items():
    fig, ax = plt.subplots(figsize=(5, 1.2))
    ax.text(0.5, 0.5, tex, fontsize=24, ha="center", va="center", color="black")
    ax.axis("off")
    plt.savefig(filename, transparent=True, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close()

print("Saved formula1.png and formula2.png!")
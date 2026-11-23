import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from integrators import NBodySystem, G

# 2-body Sun-Earth initial conditions (e = 0.2, a = 1.0 AU, 2000 revolutions)
e = 0.2
mu = G * (1.0 + 3.04e-6)
r_peri = 1.0 * (1.0 - e)
v_peri = np.sqrt(mu * (1.0 + e) / r_peri)

# Run all 3 methods with log_interval=23 so it matches your energy GIF
results = {}
for method in ['rk4', 'leapfrog', 'yoshida4']:
    sys = NBodySystem([1.0, 3.04e-6], [[0, 0, 0], [r_peri, 0, 0]], [[0, 0, 0], [0, v_peri, 0]])
    sys.com_frame()
    t, r, E, L = sys.integrate(t_max=2000.0, dt=0.01, method=method, log_interval=23)
    dL = np.linalg.norm(L[1:] - L[0], axis=1)
    results[method] = (t[1:], dL)

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.2, 2000)
ax.set_ylim(1e-20, 1e-7)
ax.set_title('Two-Body Kepler Test: Angular Momentum Error')
ax.set_xlabel('Time (years)')
ax.set_ylabel(r'Angular Momentum Error $|\Delta \mathbf{L}|$')
ax.grid(True, alpha=0.6)

line_rk4, = ax.plot([], [], color='black', linewidth=1.4, label='RK4')
line_lf, = ax.plot([], [], color='tab:blue', linewidth=1.4, label='Leapfrog')
line_y4, = ax.plot([], [], color='tab:orange', linewidth=1.4, label='Yoshida4')
ax.legend(loc='upper left')

n_points = len(results['rk4'][0])
frames = np.linspace(2, n_points, 90, dtype=int)  # ~4.5 seconds at 20 fps

def update(idx):
    for line, m in [(line_rk4, 'rk4'), (line_lf, 'leapfrog'), (line_y4, 'yoshida4')]:
        t_arr, err_arr = results[m]
        line.set_data(t_arr[:idx], err_arr[:idx])
    return line_rk4, line_lf, line_y4

plt.tight_layout()
anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
anim.save('kepler_dL.gif', writer=PillowWriter(fps=20), dpi=200)
plt.close()
print("Saved kepler_dL.gif!")
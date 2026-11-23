import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from integrators import NBodySystem, G

e = 0.2
mu = G * (1.0 + 3.04e-6)
r_peri = 1.0 * (1.0 - e)
v_peri = np.sqrt(mu * (1.0 + e) / r_peri)

methods = [
    ('rk4', 'black'),
    ('leapfrog', 'tab:blue'),
    ('yoshida4', 'tab:orange'),
]

results = {}
for method, _ in methods:
    sys = NBodySystem([1.0, 3.04e-6], [[0, 0, 0], [r_peri, 0, 0]], [[0, 0, 0], [0, v_peri, 0]])
    sys.com_frame()
    t, r, E, L = sys.integrate(t_max=2000.0, dt=0.01, method=method, log_interval=23)
    rel_dE = np.abs((E[1:] - E[0]) / E[0])
    dL = np.linalg.norm(L[1:] - L[0], axis=1)
    results[method] = (t[1:], rel_dE, dL)

n_points = len(results['rk4'][0])
frames = np.linspace(2, n_points, 90, dtype=int)

# 1. Energy Error GIF (matches kepler_test.png)
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.2, 2000)
ax.set_ylim(1e-15, 1e-2)
ax.set_title('Two-Body Kepler Test: Energy')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Relative Energy Error')
ax.grid(True)

lines_E = [ax.plot([], [], color=c, label=m)[0] for m, c in methods]
ax.legend()

def update_E(idx):
    for line, (m, _) in zip(lines_E, methods):
        line.set_data(results[m][0][:idx], results[m][1][:idx])
    return lines_E

anim_E = FuncAnimation(fig, update_E, frames=frames, interval=50, blit=True)
anim_E.save('kepler_energy.gif', writer=PillowWriter(fps=20), dpi=200)
plt.close()

# 2. Angular Momentum Error GIF (matches kepler_dL.png)
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.2, 2000)
ax.set_ylim(1e-20, 1e-7)
ax.set_title('Two-Body Kepler Test: Angular Momentum Error')
ax.set_xlabel('Time (years)')
ax.set_ylabel(r'Error $|\Delta \mathbf{L}|$')
ax.grid(True)

lines_L = [ax.plot([], [], color=c, label=m)[0] for m, c in methods]
ax.legend()

def update_L(idx):
    for line, (m, _) in zip(lines_L, methods):
        line.set_data(results[m][0][:idx], results[m][2][:idx])
    return lines_L

anim_L = FuncAnimation(fig, update_L, frames=frames, interval=50, blit=True)
anim_L.save('kepler_dL.gif', writer=PillowWriter(fps=20), dpi=200)
plt.close()
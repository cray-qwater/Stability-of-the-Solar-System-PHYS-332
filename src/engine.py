import numpy as np

# Gravitational constant in astronomical units: AU^3 / (M_sun * yr^2)
G = 4.0 * np.pi**2

class NBodySystem:
    """
    Stores state (masses, positions, velocities) of an N-body gravitational system
    and provides symplectic integration methods (Leapfrog KDK and 4th-order Yoshida).
    """
    def __init__(self, masses, positions, velocities):
        # Masses in Solar Masses (M_sun)
        self.m = np.array(masses, dtype=np.float64)
        # Positions shape (N, 3) in AU
        self.r = np.array(positions, dtype=np.float64)
        # Velocities shape (N, 3) in AU/yr
        self.v = np.array(velocities, dtype=np.float64)
        self.N = len(masses)

    def get_accelerations(self, r_pos=None):
        """
        Calculates gravitational acceleration vectors for all bodies:
        a_i = G * sum_{j != i} m_j * (r_j - r_i) / |r_j - r_i|^3
        """
        if r_pos is None:
            r_pos = self.r

        # Displacement vectors dr[i, j] = r_j - r_i, shape (N, N, 3)
        dr = r_pos[np.newaxis, :, :] - r_pos[:, np.newaxis, :]
        dist_sq = np.sum(dr**2, axis=-1)

        # Set self-distance to infinity to ignore self-interaction (i == j)
        np.fill_diagonal(dist_sq, np.inf)

        # Weight w[i, j] = m_j / |r_j - r_i|^3
        w = self.m[np.newaxis, :] * dist_sq**(-1.5)

        # a_i = G * sum_j w[i, j] * dr[i, j]
        acc = G * np.einsum("ij,ijk->ik", w, dr)
        return acc

    def total_energy(self):
        """
        Calculates total mechanical energy E = T + V using fully vectorized
        broadcasting across all N bodies.
        """
        # Kinetic energy: T = 0.5 * sum(m_i * |v_i|^2)
        T = 0.5 * np.sum(self.m * np.sum(self.v**2, axis=1))
        
        # 3D displacement vectors between all pairs: shape (N, N, 3)
        diff = self.r[:, np.newaxis, :] - self.r[np.newaxis, :, :]
        
        # Pairwise scalar distances: shape (N, N)
        dist = np.linalg.norm(diff, axis=-1)
        
        # Set self-distance to infinity to zero out diagonal entries (i == j)
        np.fill_diagonal(dist, np.inf)
        
        # Outer product of mass vector: shape (N, N) where entry (i, j) = m_i * m_j
        m_matrix = np.outer(self.m, self.m)
        
        # Potential energy: factor of 0.5 corrects for double-counting pairs (i, j) and (j, i)
        V = -0.5 * G * np.sum(m_matrix / dist)

        return T + V
    
    def total_angular_momentum(self):
        """
        Calculates total angular momentum vector L = sum(m_i * (r_i x v_i)).
        Symplectic integrators conserve this vector to machine precision.
        """
        # np.cross operates along axis=-1 (the 3D Cartesian coordinates)
        L_individual = self.m[:, np.newaxis] * np.cross(self.r, self.v)
        return np.sum(L_individual, axis=0)
    
    def leapfrog(self, dt):
        """
        2nd-order Symplectic Leapfrog (Kick-Drift-Kick).
        Advances system positions and velocities by step size dt.
        """
        # 1. Half Kick: v(t + dt/2) = v(t) + (dt/2) * a(r(t))
        acc = self.get_accelerations()
        self.v += 0.5 * dt * acc
        
        # 2. Full Drift: r(t + dt) = r(t) + dt * v(t + dt/2)
        self.r += dt * self.v
        
        # 3. Half Kick: v(t + dt) = v(t + dt/2) + (dt/2) * a(r(t + dt))
        acc_new = self.get_accelerations()
        self.v += 0.5 * dt * acc_new

    def step_yoshida4(self, dt):
        """
        4th-order Explicit Symplectic Integrator (Yoshida 1990).
        Composes sub-steps using exact coefficients derived from 2^(1/3).
        """
        # Base constant from 2^(1/3)
        w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
        w0 = - (2.0**(1.0/3.0)) / (2.0 - 2.0**(1.0/3.0))
        
        # Drift coefficients c_i and kick coefficients d_i
        c = np.array([w1 / 2.0, (w0 + w1) / 2.0, (w0 + w1) / 2.0, w1 / 2.0])
        d = np.array([w1, w0, w1, 0.0])
        
        for i in range(4):
            self.r += c[i] * dt * self.v
            if d[i] != 0.0:
                acc = self.get_accelerations()
                self.v += d[i] * dt * acc
   
   
    #--------- Ian Run Loop ---------#
    def run(self, dt, steps, method=step_yoshida4, record_every=1,
            monitor=None, monitor_every=10, verbose=True):
        """
        Advances the system by `steps` steps of size dt using `method`
        (e.g. NBodySystem.step_yoshida4 or NBodySystem.leapfrog).
        Records the state every `record_every` steps, including the initial state.

        monitor: optional callable monitor(system, t) -> bool, called every
                 `monitor_every` steps. Returning True stops the run early
                 (the stopping state is recorded and the outputs are trimmed).
        verbose: print progress.

        Returns a dict of numpy arrays:
            t (M,), r (M, N, 3), v (M, N, 3), E (M,), L (M, 3)
        """
        n_records = steps // record_every + 2   # +1 spare slot for an early-stop record
        t = np.zeros(n_records)
        r_hist = np.zeros((n_records, self.N, 3))
        v_hist = np.zeros((n_records, self.N, 3))
        E_hist = np.zeros(n_records)
        L_hist = np.zeros((n_records, 3))

        def record(k, time):
            t[k] = time
            r_hist[k] = self.r
            v_hist[k] = self.v
            E_hist[k] = self.total_energy()
            L_hist[k] = self.total_angular_momentum()

        record(0, 0.0)
        k = 1
        for step in range(1, steps + 1):
            method(self, dt)
            if verbose and step % max(1, steps // 100) == 0:
                print(f"{step}/{steps}")
            recorded = False
            if step % record_every == 0:
                record(k, step * dt)
                k += 1
                recorded = True
            if monitor is not None and step % monitor_every == 0 and monitor(self, step * dt):
                if not recorded:
                    record(k, step * dt)
                    k += 1
                break

        return {"t": t[:k], "r": r_hist[:k], "v": v_hist[:k], "E": E_hist[:k], "L": L_hist[:k]}

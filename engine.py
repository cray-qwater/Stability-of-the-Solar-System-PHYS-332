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
            
        acc = np.zeros_like(r_pos)
        
        for i in range(self.N):
            # Displacement vectors pointing from body i to all other bodies
            dr = r_pos - r_pos[i]
            dist_sq = np.sum(dr**2, axis=1)
            
            # Set self-distance to infinity to ignore self-interaction (i == j)
            dist_sq[i] = np.inf
            
            # Inverse cube of distance: 1 / r^3
            inv_dist_cube = dist_sq**(-1.5)
            
            # Sum gravitational accelerations from all other bodies
            acc[i] = G * np.sum((self.m[:, np.newaxis] * dr) * inv_dist_cube[:, np.newaxis], axis=0)
            
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

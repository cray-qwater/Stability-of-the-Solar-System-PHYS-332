import numpy as np
import matplotlib.pyplot as plt
#Gravitational constant in astronomical units: AU^3 / (M_sun * yr^2)
G = 4.0 * np.pi**2

class NBodySystem:
    """
    Stores state (masses, positions, velocities) of an N-body gravitational system
    and provides symplectic integration methods (Leapfrog KDK and 4th-order Yoshida).
    """

    def __init__(self, masses, positions, velocities):
        #Masses in Solar Masses (M_sun)
        self.m = np.array(masses, dtype=np.float64)
        #Positions shape (N, 3) in AU
        self.r = np.array(positions, dtype=np.float64)
        #Velocities shape (N, 3) in AU/yr
        self.v = np.array(velocities, dtype=np.float64)
        self.N = len(masses)
        self.acc = None

    def accelerations(self, r_pos=None):
        """
        Calculates gravitational acceleration vectors for all bodies:
        a_i = G * sum_{j != i} m_j * (r_j - r_i) / |r_j - r_i|^3
        """
        if r_pos is None:
            r_pos = self.r
            
        acc = np.zeros_like(r_pos)
        
        for i in range(self.N):
            #Displacement vectors pointing from body i to all other bodies
            dr = r_pos - r_pos[i]
            dist_sq = np.sum(dr**2, axis=1)
            
            #Set self-distance to infinity to ignore self-interaction (i == j)
            dist_sq[i] = np.inf
            
            #Inverse cube of distance: 1 / r^3
            inv_dist_cube = dist_sq**(-1.5)
            
            #Sum gravitational accelerations from all other bodies
            acc[i] = G * np.sum((self.m[:, np.newaxis] * dr) * inv_dist_cube[:, np.newaxis], axis=0)
            
        return acc

    def energy(self):
        """
        Calculates total mechanical energy E = T + V using fully vectorized
        broadcasting across all N bodies.
        """
        #Kinetic energy: T = 0.5 * sum(m_i * |v_i|^2)
        T = 0.5 * np.sum(self.m * np.sum(self.v**2, axis=1))
        
        #3D displacement vectors between all pairs: shape (N, N, 3)
        diff = self.r[:, np.newaxis, :] - self.r[np.newaxis, :, :]
        
        #Pairwise scalar distances: shape (N, N)
        dist = np.linalg.norm(diff, axis=-1)
        
        #Set self-distance to infinity to zero out diagonal entries (i == j)
        np.fill_diagonal(dist, np.inf)
        
        #Outer product of mass vector: shape (N, N) where entry (i, j) = m_i * m_j
        m_matrix = np.outer(self.m, self.m)
        
        #Potential energy: factor of 0.5 corrects for double-counting pairs (i, j) and (j, i)
        V = -0.5 * G * np.sum(m_matrix / dist)

        return T + V
    
    def get_L(self):
        """
        Calculates total angular momentum vector L = sum(m_i * (r_i x v_i)).
        Symplectic integrators conserve this vector to machine precision.
        """
        #np.cross operates along axis=-1 (the 3D Cartesian coordinates)
        L_individual = self.m[:, np.newaxis] * np.cross(self.r, self.v)
        return np.sum(L_individual, axis=0)
    
    def leapfrog(self, dt):
        if self.acc is None:
            self.acc = self.accelerations()

        self.v += 0.5*dt*self.acc
        self.r += dt*self.v
        self.acc = self.accelerations()
        self.v += 0.5*dt*self.acc

    def yoshida4(self, dt):
        """
        4th-order Explicit Symplectic Integrator (Yoshida 1990).
        Composes sub-steps using exact coefficients derived from 2^(1/3).
        """
        #Base constant from 2^(1/3)
        w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
        w0 = - (2.0**(1.0/3.0)) / (2.0 - 2.0**(1.0/3.0))
        
        #Drift coefficients c_i and kick coefficients d_i
        c = np.array([w1 / 2.0, (w0 + w1) / 2.0, (w0 + w1) / 2.0, w1 / 2.0])
        d = np.array([w1, w0, w1, 0.0])
        
        for i in range(4):
            self.r += c[i] * dt * self.v
            if d[i] != 0.0:
                acc = self.accelerations()
                self.v += d[i] * dt * acc
        self.acc = None

    def rk4step(self,dt):
        r0 = self.r.copy()
        v0 = self.v.copy()
        self.acc = None

        #k1=dt*f(y)
        kr1 = dt*v0
        kv1 = dt*self.accelerations(r0)
        #k2 = dt*f(y+k1/2)
        kr2 = dt*(v0 + kv1/2.0)
        kv2 = dt*self.accelerations(r0 + kr1/2.0)
        #k3= dt*f(y+k2/2)
        kr3 = dt*(v0 + kv2/2.0)
        kv3 = dt*self.accelerations(r0 + kr2/2.0)
        #k4 = dt*f(y+k3)
        kr4 = dt*(v0 + kv3)
        kv4 = dt*self.accelerations(r0 + kr3)

        #ynew = y+(k1+k4)/6.0 + (k2+k3)/3.0
        self.r = r0 +(kr1+kr4)/6.0 + (kr2+kr3)/3.0
        self.v = v0 +(kv1+kv4)/6.0 + (kv2+kv3)/3.0



    #Helper Methods
    def com_frame(self):
        #Center of mass position and velocity (P = 0 frame)
        M_tot = np.sum(self.m)
        r_cm = np.sum(self.m[:, np.newaxis] * self.r, axis=0) / M_tot
        v_cm = np.sum(self.m[:, np.newaxis] * self.v, axis=0) / M_tot
        
        self.r -= r_cm
        self.v -= v_cm
        self.acc = None

    def integrate(self, t_max, dt, method="yoshida4", log_interval=10):
        n_steps = int(t_max / dt)
        t_list, r_list, E_list, L_list = [], [], [], []
        
        for step in range(n_steps + 1):
            if step % log_interval == 0:
                t_list.append(step * dt)
                r_list.append(self.r.copy())
                E_list.append(self.energy())
                L_list.append(self.get_L())
                
            if step < n_steps:
                if method == 'leapfrog':
                    self.leapfrog(dt)
                elif method == 'yoshida4':
                    self.yoshida4(dt)
                elif method == 'rk4':
                    self.rk4step(dt)
                    
        return np.array(t_list), np.array(r_list), np.array(E_list), np.array(L_list)


if __name__ == "__main__":
    #2-body Sun-Earth test (e = 0.2, a = 1.0 AU, 2000 revolutions)
    e = 0.2
    mu = G * (1.0 + 3.04e-6)
    r_peri = 1.0 * (1.0 - e)
    v_peri = np.sqrt(mu * (1.0 + e) / r_peri)

    #just setting up the output 
    print('Two-Body Kepler Test:')
    print('Method\t\tdt\tMax dE/E0\tmax dL')

    plt.figure()
    for method in ['leapfrog', 'yoshida4', 'rk4']:
        for dt in [0.01, 0.005]:
            sys = NBodySystem([1.0, 3.04e-6], [[0, 0, 0], [r_peri, 0, 0]], [[0, 0, 0], [0, v_peri, 0]])
            sys.com_frame()
            t, r, E, L = sys.integrate(t_max=2000.0, dt=dt, method=method, log_interval=100)

            rel_dE = np.abs((E - E[0]) / E[0])
            max_dE = np.max(rel_dE)
            max_dL = np.max(np.linalg.norm(L - L[0], axis=1))
            print(f"{method:8s}\t{dt}\t{max_dE:.3e}\t{max_dL:.3e}")

            if dt == 0.01:
                plt.loglog(t[1:], rel_dE[1:], label=method)

    plt.title('Two-Body Kepler Test')
    plt.xlabel('Time (years)')
    plt.ylabel('Fractional Energy Error')
    plt.grid(True)
    plt.legend()
    plt.savefig('kepler_test.png')
    print('\nNibiru Flyby Step-Size Sweep:')
    print('dt\tMax dE/E0')
    m_earth = 3.04e-6
    for dt in [0.01, 0.0075, 0.005, 0.0025, 0.001]:
        sys = NBodySystem(
            [1.0, m_earth, 10.0 * m_earth],
            [[0, 0, 0], [1.0, 0, 0], [-15.0, 0.5, 0]],
            [[0, 0, 0], [0, 2 * np.pi, 0], [6.0, 0, 0]]
        )
        sys.com_frame()
        t, r, E, L = sys.integrate(t_max=6.0, dt=dt, method='yoshida4', log_interval=10)
        print(f"{dt}\t{np.max(np.abs((E - E[0]) / E[0])):.3e}")
    plt.show()
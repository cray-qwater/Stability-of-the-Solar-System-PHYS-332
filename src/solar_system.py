"""
Single source of truth for the solar-system setup: body names, masses, JPL initial
conditions, and helpers to insert a hypothetical extra body ("Nibiru") from orbital
elements. Units: AU, M_sun, yr (G = 4 pi^2).
"""
import numpy as np
from paths import DATA
from engine import NBodySystem, G

NAMES = ["Sun", "Mercury", "Venus", "Earth", "Mars",
         "Jupiter", "Saturn", "Uranus", "Neptune"]
MASSES = [1.0,        # sun
          1.66e-7,    # mercury
          2.45e-6,    # venus
          3.04e-6,    # earth + moon
          3.23e-7,    # mars
          9.55e-4,    # jupiter
          2.86e-4,    # saturn
          4.37e-5,    # uranus
          5.15e-5,    # neptune
]                     # in solar masses
M_JUPITER = 9.55e-4   # M_sun


def load_baseline(path=None):
    """Return (names, masses, r, v) for the 9-body JPL system (barycentric, ecliptic)."""
    ic = np.load(path or DATA / "initial_conditions.npz")
    return [str(n) for n in ic["names"]], np.array(MASSES), ic["r"].copy(), ic["v"].copy()


# -----------------------------------------------------------------------------
# Orbital elements <-> Cartesian state (two-body, heliocentric)
# -----------------------------------------------------------------------------
def _rotation(inc, Omega, omega):
    """Perifocal -> ecliptic rotation matrix R3(-Omega) R1(-inc) R3(-omega)."""
    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(omega), np.sin(omega)
    return np.array([
        [cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci,  sO*si],
        [sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci, -cO*si],
        [sw*si,             cw*si,             ci   ],
    ])


def elements_to_state(q, e, inc, Omega, omega, f, mu):
    """
    Heliocentric position/velocity from perihelion distance q, eccentricity e,
    inclination inc, longitude of ascending node Omega, argument of perihelion
    omega, and true anomaly f (all angles in radians). Works for e < 1 and e >= 1.
    mu = G * (M_central + m_body).
    """
    p = q * (1.0 + e)                                   # semi-latus rectum
    r = p / (1.0 + e * np.cos(f))
    r_pf = r * np.array([np.cos(f), np.sin(f), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(f), e + np.cos(f), 0.0])
    R = _rotation(inc, Omega, omega)
    return R @ r_pf, R @ v_pf


def state_to_elements(r, v, mu):
    """Return dict with a, e, q, inc (rad) from a heliocentric state (for checks/diagnostics)."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    rn = np.linalg.norm(r)
    h = np.cross(r, v)
    e_vec = np.cross(v, h) / mu - r / rn
    e = np.linalg.norm(e_vec)
    energy = 0.5 * v @ v - mu / rn
    a = -mu / (2 * energy) if energy != 0 else np.inf
    q = (h @ h) / mu / (1 + e)
    inc = np.arccos(h[2] / np.linalg.norm(h))
    return {"a": a, "e": e, "q": q, "inc": inc, "energy": energy}


def true_anomaly_at_radius(r, q, e):
    """|f| at which the orbit reaches heliocentric distance r (r >= q)."""
    p = q * (1.0 + e)
    cos_f = (p / r - 1.0) / e
    if cos_f < -1.0:
        raise ValueError(f"Orbit never reaches r = {r} AU (aphelion is smaller)")
    return np.arccos(np.clip(cos_f, -1.0, 1.0))


def time_from_perihelion(r, q, e, mu):
    """Time (yr) to travel between perihelion and heliocentric distance r (elliptic or hyperbolic)."""
    if e < 1.0:
        a = q / (1.0 - e)
        E = np.arccos(np.clip((1.0 - r / a) / e, -1.0, 1.0))
        return (E - e * np.sin(E)) / np.sqrt(mu / a**3)
    if e > 1.0:
        a = q / (e - 1.0)                                # |a|
        F = np.arccosh((1.0 + r / a) / e)
        return (e * np.sinh(F) - F) / np.sqrt(mu / a**3)
    raise ValueError("Parabolic orbits (e == 1) not supported")


# -----------------------------------------------------------------------------
# Nibiru
# -----------------------------------------------------------------------------
# Popular-lore "myth" Nibiru: ~5 Jupiter masses, ~3600 yr period, perihelion in the asteroid belt
NIBIRU_MYTH = dict(mass=5 * M_JUPITER, q=2.5, a=235.0,
                   inc=np.radians(10.0), Omega=0.0, omega=0.0, r_start=60.0)


def nibiru_params(mass=None, q=None, a=None, inc=None, Omega=None, omega=None, r_start=None):
    """Fill in any unspecified parameter from NIBIRU_MYTH and add e = 1 - q/a."""
    p = dict(NIBIRU_MYTH)
    for k, val in dict(mass=mass, q=q, a=a, inc=inc, Omega=Omega, omega=omega, r_start=r_start).items():
        if val is not None:
            p[k] = val
    p["e"] = 1.0 - p["q"] / p["a"]
    return p


def nibiru_state(p):
    """
    Heliocentric (r, v) for Nibiru, placed *inbound* at distance r_start so the run
    begins before perihelion passage.
    """
    mu = G * (1.0 + p["mass"])
    f0 = -true_anomaly_at_radius(p["r_start"], p["q"], p["e"])   # negative f = approaching perihelion
    return elements_to_state(p["q"], p["e"], p["inc"], p["Omega"], p["omega"], f0, mu)


def passage_duration(p):
    """Time (yr) for Nibiru to go from r_start inbound, through perihelion, back out to r_start."""
    mu = G * (1.0 + p["mass"])
    return 2.0 * time_from_perihelion(p["r_start"], p["q"], p["e"], mu)


# -----------------------------------------------------------------------------
# System builder
# -----------------------------------------------------------------------------
def to_barycentric(m, r, v):
    """Shift positions and velocities so the center of mass is at rest at the origin."""
    m = np.asarray(m)
    r = r - (m @ r) / m.sum()
    v = v - (m @ v) / m.sum()
    return r, v


def build_system(nibiru=None):
    """
    Build an NBodySystem for the 9-body solar system, optionally with Nibiru appended
    as the last body. `nibiru` is a dict from nibiru_params() (or None).
    Returns (system, names, masses).
    """
    names, m, r, v = load_baseline()
    if nibiru is not None:
        r_hel, v_hel = nibiru_state(nibiru)
        sun = names.index("Sun")
        r = np.vstack([r, r[sun] + r_hel])
        v = np.vstack([v, v[sun] + v_hel])
        m = np.append(m, nibiru["mass"])
        names = names + ["Nibiru"]
        r, v = to_barycentric(m, r, v)
    return NBodySystem(masses=m, positions=r, velocities=v), names, m


if __name__ == "__main__":
    # Self-checks: element round trip and barycentric correction
    p = nibiru_params()
    mu = G * (1.0 + p["mass"])
    r0, v0 = nibiru_state(p)
    el = state_to_elements(r0, v0, mu)
    print(f"Nibiru myth: e = {p['e']:.4f}, P = {p['a']**1.5 / np.sqrt(1 + p['mass']):.0f} yr")
    print(f"  round trip: a {el['a']:.3f} (want {p['a']}), q {el['q']:.4f} (want {p['q']}), "
          f"inc {np.degrees(el['inc']):.3f} deg, |r0| {np.linalg.norm(r0):.3f} AU, "
          f"inbound: {r0 @ v0 < 0}")
    assert abs(el["a"] - p["a"]) < 1e-8 * p["a"] and abs(el["q"] - p["q"]) < 1e-10
    print(f"  passage r_start -> perihelion -> r_start: {passage_duration(p):.1f} yr")

    sysN, names, m = build_system(p)
    P = m @ sysN.v
    com = m @ sysN.r / m.sum()
    print(f"  bodies: {names}")
    print(f"  |total momentum| = {np.linalg.norm(P):.2e}, |COM| = {np.linalg.norm(com):.2e}")
    assert np.linalg.norm(P) < 1e-14 and np.linalg.norm(com) < 1e-14
    print("OK")

"""
Stability flags for a solar system perturbed by Nibiru.

Flags (checked inside the run every `monitor_every` steps):
  ejection        planet heliocentric specific energy > 0 AND distance > eject_r      (stops run)
  sun_collision   planet-Sun distance < sun_r                                         (stops run)
  close_encounter Nibiru-planet (or planet-planet) distance < hill_k * mutual Hill
                  radius ((m_i+m_j)/3M_sun)^(1/3) * mean heliocentric distance    (recorded)
  unresolved      |dE/E0| > E_tol: fixed-step integrator failed during a deep
                  encounter, so the outcome is numerical, not physical                (stops run)
"""
import numpy as np
from engine import NBodySystem, G
from solar_system import build_system, passage_duration, state_to_elements


class StabilityMonitor:
    def __init__(self, system, names, nibiru_index=None,
                 eject_r=100.0, sun_r=0.005, hill_k=3.0, E_tol=1e-4):
        self.names = list(names)
        self.sun = self.names.index("Sun")
        self.nib = nibiru_index
        self.planets = [i for i in range(system.N) if i not in (self.sun, nibiru_index)]
        self.eject_r, self.sun_r, self.hill_k, self.E_tol = eject_r, sun_r, hill_k, E_tol
        self.m = system.m.copy()
        self.M_sun = self.m[self.sun]

        # Initial heliocentric semi-major axes (for reporting orbit changes)
        rel_r = system.r - system.r[self.sun]
        rel_v = system.v - system.v[self.sun]
        self.a0 = np.full(system.N, np.nan)
        for i in self.planets:
            self.a0[i] = state_to_elements(rel_r[i], rel_v[i], G * (self.m[self.sun] + self.m[i]))["a"]

        self.E0 = system.total_energy()
        self.max_dE = 0.0
        self.min_dist = {i: np.inf for i in self.planets}     # closest Nibiru approach per planet
        self.min_dist_t = {i: np.nan for i in self.planets}
        self.events = []                                        # (t, type, body, detail)
        self._encountered = set()                               # report each encounter pair once
        self.stop_reason = None

    def hill(self, i, j, d_sun):
        """Mutual Hill radius of bodies i, j at their mean current heliocentric distance."""
        return 0.5 * (d_sun[i] + d_sun[j]) * ((self.m[i] + self.m[j]) / (3.0 * self.M_sun))**(1.0 / 3.0)

    def _event(self, t, kind, body, detail=""):
        self.events.append((t, kind, body, detail))

    def __call__(self, system, t):
        rel = system.r - system.r[self.sun]
        vel = system.v - system.v[self.sun]
        d_sun = np.linalg.norm(rel, axis=1)

        # Energy guard
        dE = abs((system.total_energy() - self.E0) / self.E0)
        self.max_dE = max(self.max_dE, dE)
        if dE > self.E_tol:
            self._event(t, "unresolved", None, f"|dE/E0| = {dE:.1e}")
            self.stop_reason = "unresolved"
            return True

        for i in self.planets:
            name = self.names[i]
            # Sun collision
            if d_sun[i] < self.sun_r:
                self._event(t, "sun_collision", name, f"r = {d_sun[i]:.4f} AU")
                self.stop_reason = "sun_collision"
                return True
            # Ejection: unbound from the Sun and far away
            eps = 0.5 * vel[i] @ vel[i] - G * (self.m[self.sun] + self.m[i]) / d_sun[i]
            if eps > 0 and d_sun[i] > self.eject_r:
                self._event(t, "ejection", name, f"r = {d_sun[i]:.1f} AU")
                self.stop_reason = "ejection"
                return True

        # Close encounters with Nibiru
        if self.nib is not None:
            d_nib = np.linalg.norm(system.r[self.planets] - system.r[self.nib], axis=1)
            for d, i in zip(d_nib, self.planets):
                if d < self.min_dist[i]:
                    self.min_dist[i], self.min_dist_t[i] = d, t
                rh = self.hill(i, self.nib, d_sun)
                if d < self.hill_k * rh and ("Nibiru", i) not in self._encountered:
                    self._encountered.add(("Nibiru", i))
                    self._event(t, "close_encounter", self.names[i],
                                f"Nibiru at {d:.4f} AU ({d / rh:.2f} mutual R_Hill)")

        # Planet-planet close encounters (sign of orbit crossing after the passage)
        P = self.planets
        for a in range(len(P)):
            for b in range(a + 1, len(P)):
                i, j = P[a], P[b]
                if (i, j) in self._encountered:
                    continue
                d = np.linalg.norm(system.r[i] - system.r[j])
                rh = self.hill(i, j, d_sun)
                if d < self.hill_k * rh:
                    self._encountered.add((i, j))
                    self._event(t, "close_encounter", f"{self.names[i]}-{self.names[j]}",
                                f"{d:.4f} AU ({d / rh:.2f} mutual R_Hill)")
        return False

    def summary(self, system):
        """Classify the run and report final orbital changes of each planet."""
        rel_r = system.r - system.r[self.sun]
        rel_v = system.v - system.v[self.sun]
        changes = {}
        for i in self.planets:
            el = state_to_elements(rel_r[i], rel_v[i], G * (self.m[self.sun] + self.m[i]))
            changes[self.names[i]] = {"a": el["a"], "e": el["e"], "da_rel": (el["a"] - self.a0[i]) / self.a0[i]}
        kinds = {e[1] for e in self.events}
        if self.stop_reason == "unresolved":
            outcome = "unresolved"
        elif kinds & {"ejection", "sun_collision"}:
            outcome = "unstable"
        elif "close_encounter" in kinds:
            outcome = "encounter"
        else:
            outcome = "stable"
        return {
            "outcome": outcome,
            "events": list(self.events),
            "max_dE": self.max_dE,
            "min_dist": {self.names[i]: self.min_dist[i] for i in self.planets},
            "min_dist_t": {self.names[i]: self.min_dist_t[i] for i in self.planets},
            "orbits": changes,
        }


def run_scenario(nibiru, dt=0.001, extra_years=300.0, record_every=50,
                 monitor_every=10, keep_history=False, verbose=False, **monitor_kw):
    """
    Simulate one Nibiru passage: from r_start inbound, through perihelion, back out to
    r_start, then `extra_years` more to see whether the perturbed planets stay bound.
    `nibiru=None` runs the unperturbed system for the same default duration.
    Returns the monitor summary dict (+ 'history', 'names' if keep_history).
    """
    system, names, m = build_system(nibiru)
    years = (passage_duration(nibiru) if nibiru else 0.0) + extra_years
    steps = int(round(years / dt))
    mon = StabilityMonitor(system, names, nibiru_index=(len(names) - 1 if nibiru else None), **monitor_kw)
    out = system.run(dt, steps, method=NBodySystem.step_yoshida4, record_every=record_every,
                     monitor=mon, monitor_every=monitor_every, verbose=verbose)
    res = mon.summary(system)
    res.update(years=years, t_end=out["t"][-1], nibiru=nibiru)
    if keep_history:
        res.update(history=out, names=names, masses=m)
    return res


def print_summary(res):
    nib = res["nibiru"]
    if nib:
        print(f"Nibiru: {nib['mass'] / 9.55e-4:.2f} M_J, q = {nib['q']} AU, a = {nib['a']} AU, "
              f"i = {np.degrees(nib['inc']):.1f} deg")
    print(f"  outcome: {res['outcome'].upper()}  (ran {res['t_end']:.1f} / {res['years']:.1f} yr, "
          f"max |dE/E0| = {res['max_dE']:.1e})")
    for t, kind, body, detail in res["events"]:
        print(f"    t = {t:8.2f} yr  {kind:16s} {body or '':18s} {detail}")
    if nib:
        print("  closest Nibiru approach:  " +
              ", ".join(f"{k} {v:.3f} AU" for k, v in res["min_dist"].items()))
    print("  final a (AU), e:          " +
          ", ".join(f"{k} {o['a']:.3f}/{o['e']:.3f}" for k, o in res["orbits"].items()))

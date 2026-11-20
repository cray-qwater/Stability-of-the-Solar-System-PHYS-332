# fetch_ic.py — run once
import numpy as np
from astroquery.jplhorizons import Horizons

names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
ids   = ["10",  "199",     "299",   "3",     "4",    "5",       "6",      "7",      "8"]
epoch = 2461000.5   # Julian date of the starting snapshot (Sep 2025)

r, v = [], []
for body in ids:
    vec = Horizons(id=body, location="@0", epochs=epoch).vectors(refplane="ecliptic")
    r.append([vec["x"][0],  vec["y"][0],  vec["z"][0]])            # AU
    v.append([vec["vx"][0], vec["vy"][0], vec["vz"][0]])           # AU/day
np.savez("initial_conditions.npz", names=names, r=np.array(r), v=np.array(v) * 365.25)  # convert to AU/yr

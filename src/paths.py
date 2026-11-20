"""Project folder locations, so scripts work no matter which directory they are run from."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"                       # .npz inputs and simulation results
FIGURES = ROOT / "outputs" / "figures"     # .png plots
ANIMATIONS = ROOT / "outputs" / "animations"

for _d in (DATA, FIGURES, ANIMATIONS):
    _d.mkdir(parents=True, exist_ok=True)

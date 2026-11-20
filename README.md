# Stability-of-the-Solar-System-PHYS-332
Group project analyzing the stability of the solar system for the PHYS 332. 

## Layout
```
src/                  Python code (run any script from any directory)
  engine.py             N-body integrator (leapfrog, Yoshida 4)
  solar_system.py       planet names/masses, JPL initial conditions, Nibiru setup
  stability.py          ejection / collision / close-encounter flags
  paths.py              folder locations used by all scripts
  Initial_Conditions.py fetch JPL Horizons state vectors -> data/initial_conditions.npz
  Implementation.py     200 yr 9-body run, conservation + period checks
  error_analysis.py     convergence order and position-error tests
  nibiru_sweep.py       myth Nibiru run + mass x perihelion stability sweep
  animate.py            3D animation of a saved run
data/                 .npz inputs and simulation results
outputs/figures/      .png plots
outputs/animations/   .mp4 animations
```

## Running
```
python src/Implementation.py
python src/error_analysis.py
python src/nibiru_sweep.py          # ~20 min; --quick for a short test
python src/animate.py               # needs data/nibiru_myth_run.npz from nibiru_sweep.py
```

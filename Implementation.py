from engine import NBodySystem as nbs

# -----------------------------------------------------------------------------
# NBodySystem layout
# -----------------------------------------------------------------------------
# __init__(masses, positions, velocities)
#   - store masses (N,), positions (N,3), velocities (N,3) as numpy arrays
#   - store N = number of bodies
#
# get_accelerations(r_pos=None)
#   - default to the current positions if none are given
#   - for each body, sum the gravitational pull from every other body
#   - skip self-interaction
#   - return accelerations (N,3)
#
# total_energy()
#   - compute kinetic energy from masses and velocities
#   - compute potential energy from all pairwise distances (no double counting)
#   - return T + V
#
# total_angular_momentum()
#   - sum m * (r x v) over all bodies
#   - return the L vector (3,)
#
# leapfrog(dt)
#   - half kick velocities
#   - full drift positions
#   - half kick velocities with the new accelerations
#
# step_yoshida4(dt)
#   - build the Yoshida weights w0, w1 from 2^(1/3)
#   - build the drift coefficients c (4) and kick coefficients d (4, last = 0)
#   - loop 4 sub-steps:
#       - drift positions by c[i] * dt
#       - if d[i] != 0: kick velocities by d[i] * dt using new accelerations
#
# Usage: make a system -> record the starting energy -> call step_yoshida4(dt)
#        in a loop -> record positions and energy drift along the way
# -----------------------------------------------------------------------------



nbs("Test Sim")
#def system(sun,mercury,venus,earth,mars,jupiter,saturn,uranus,):

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



sys_4 = nbs(
    masses= [1.0,         #sun
             1.66e-7,   #mercury
             2.45e-6,   #venus
             3.04e-6,   #earth
#             3.23e-7,   #mars
#             9.55e-4,   #jupiter
#             2.86e-4,   #saturn
#             4.37e-5,   #uranus  
#             5.15e-5    #neptune
#             xxxxxxx    #nibiru
    ],
    positions=[[0,0,0],[.5,.5,.5],[1,1,1],[2,2,2]],
    velocities=[[0,0,0],[7,7,7],[6.5,6.5,6.5],[6,6,6]]
)
#def system(sun,mercury,venus,earth,mars,jupiter,saturn,uranus,):
print(sys_4.m)
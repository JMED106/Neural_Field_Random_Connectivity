import numpy as np

from nflib import Data, Connectivity
from tools import Perturbation

__author__ = 'jm'

""" Minimal version of the Neural Field equations. It does not neither save or plot
    any data.
"""

###################################################################################
# 0) PREPARE FOR CALCULATIONS
# 0.1) Load data object:
d = Data(l=100, eta0=4.0, delta=0.5, tfinal=20.0)
# 0.2) Create connectivity matrix and extract eigenmodes
c = Connectivity(d.l, profile='mex-hat', amplitude=10.0, data=d)
# 0.3) Load initial conditions
d.load_ic(c.modes[0])
# 0.5) Set perturbation configuration
p = Perturbation(data=d, modes=[1])

###################################################################################
# 1) Simulation (Integrate the system)
print('Simulating ...')
tstep = 0
temps = 0

# Time loop
while temps < d.tfinal:
    # ######################## - PERTURBATION  - ##
    if p.pbool and not d.new_ic:
        if temps >= p.t0:
            p.timeevo(temps)
    p.it[tstep % d.nsteps, :] = p.input
    # ######################## -  INTEGRATION  - ##
    # We compute the Mean-field vector S ( 1.0/(2.0*np.pi)*dx = 1.0/l )
    d.sphi[tstep % d.nsteps] = (1.0 / d.l) * (
        np.dot(c.cnt, d.r[(tstep + d.nsteps - 1) % d.nsteps]))

    # -- Integration -- #
    d.r[tstep % d.nsteps] = d.r[(tstep + d.nsteps - 1) % d.nsteps] + d.dt * (
        d.delta / np.pi + 2.0 * d.r[(tstep + d.nsteps - 1) % d.nsteps] * d.v[
            (tstep + d.nsteps - 1) % d.nsteps])
    d.v[tstep % d.nsteps] = d.v[(tstep + d.nsteps - 1) % d.nsteps] + d.dt * (
        d.v[(tstep + d.nsteps - 1) % d.nsteps] * d.v[
            (tstep + d.nsteps - 1) % d.nsteps] +
        d.eta0 + d.sphi[tstep % d.nsteps] -
        np.pi * np.pi * d.r[(tstep + d.nsteps - 1) % d.nsteps] * d.r[
            (tstep + d.nsteps - 1) % d.nsteps] +
        p.input)

    # Perturbation at certain time
    if int(p.t0 / d.dt) == tstep:
        p.pbool = True

    # Time evolution
    temps += d.dt
    tstep += 1

import Gnuplot
import getopt
import sys
from timeit import default_timer as timer

import numpy as np
from numpy import pi
import progressbar as pb
import yaml

from nflib import Data, Connectivity, FiringRate
from tools import Perturbation, qifint, qifint_noise, noise, SaveResults, TheoreticalComputations, DictToObj

__author__ = 'jm'


def main(argv, options):
    try:
        optis, args = getopt.getopt(argv, "hm:a:s:c:N:n:e:d:t:D:f:",
                                    ["mode=", "amp=", "system=", "connec=", "neurons=", "lenght=", "extcurr=",
                                     "delta=", "tfinal=", "Distr=", "file="])
    except getopt.GetoptError:
        print 'main.py [-m <mode> -a <amplitude> -s <system> -c <connectivity> ' \
              '-N <number-of-neurons> -n <lenght-of-ring-e <external-current> ' \
              '-d <widt-of-dist> -t <final-t> -D <type-of-distr> -f <config-file>]'
        sys.exit(2)

    for opt, arg in optis:
        if len(opt) > 2:
            opt = opt[1:3]
        opt = opt[1]
        # Check type and cast
        if isinstance(options[opt], int):
            options[opt] = int(float(arg))
        elif isinstance(options[opt], float):
            options[opt] = float(arg)
        else:
            options[opt] = arg

    return options


opts = {"m": 0, "a": 1.0, "s": 'both', "c": 'mex-hat',
        "N": int(2E5), "n": 100, "e": 4.0, "d": 0.5, "t": 20,
        "D": 'lorentz', "f": "conf.txt"}
extopts = {"dt": 1E-3, "t0": 0.0, "ftau": 20.0E-3, "modes": [10, 7.5, -2.5]}
pertopts = {"dt": 0.5, "attack": 'exponential', "release": 'instantaneous'}

if __name__ == '__main__':
    opts2 = main(sys.argv[1:], opts)
else:
    opts2 = opts
try:
    (opts, extopts, pertopts) = yaml.load(file(opts2['f']))
    if __name__ == '__main__':
        opts = main(sys.argv[1:], opts)
except IOError:
    print "The configuration file %s is missing, using inbuilt configuration." % (opts2['f'])
except ValueError:
    print "Configuration file has bad format."
    exit(-1)

print opts
# Gather all parameters in a single dictionary (for saving)
parameters = {'opts': opts, 'extopts': extopts, 'pertopts': pertopts}
# Convert them to dictionaries for easier access
opts = DictToObj(opts)
extopts = DictToObj(extopts)
pertopts = DictToObj(pertopts)

###################################################################################
# 0) PREPARE FOR CALCULATIONS
# 0.1) Load data object:
d = Data(l=opts.n, N=opts.N, eta0=opts.e, delta=opts.d, tfinal=opts.t, system=opts.s, fp=opts.D)

# 0.2) Create connectivity matrix and extract eigenmodes
c = Connectivity(d.l, profile=opts.c, fsmodes=extopts.modes, amplitude=10.0, data=d)
print "Modes: ", c.modes

# 0.3) Load initial conditions
d.load_ic(c.modes[0], system=d.system)
# Override initial conditions generator:
if opts.a != 0.0:
    extopts.ic = False
else:
    extopts.ic = True
if extopts.ic:
    print "Overriding initial conditions."
    d.new_ic = True

# 0.4) Load Firing rate class in case qif network is simulated
if d.system != 'nf':
    fr = FiringRate(data=d, swindow=0.5, sampling=0.05)

# 0.5) Set perturbation configuration
p = Perturbation(data=d, dt=pertopts.dt, modes=[int(opts.m)], amplitude=float(opts.a), attack=pertopts.attack)

# 0.6) Define saving paths:
sr = SaveResults(data=d, cnt=c, pert=p, system=d.system, parameters=parameters)

# 0.7) Other theoretical tools:
th = TheoreticalComputations(d, c, p)

# Progress-bar configuration
widgets = ['Progress: ', pb.Percentage(), ' ',
           pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA(), ' ']

###################################################################################
# 1) Simulation (Integrate the system)
print('Simulating ...')
pbar = pb.ProgressBar(widgets=widgets, maxval=10 * (d.nsteps + 1)).start()
time1 = timer()
tstep = 0
temps = 0
kp = 0

# Time loop
while temps < d.tfinal:
    # Time step variables
    kp = tstep % d.nsteps
    k = (tstep + d.nsteps - 1) % d.nsteps
    # ######################## - PERTURBATION  - ##
    if p.pbool and not d.new_ic:
        if temps >= p.t0:
            p.timeevo(temps)
    p.it[kp, :] = p.input

    # ######################## -  INTEGRATION  - ##
    # ######################## -      qif      - ##
    if d.system == 'qif' or d.system == 'both':
        tsyp = tstep % d.T_syn
        tskp = tstep % d.spiketime
        tsk = (tstep + d.spiketime - 1) % d.spiketime
        # We compute the Mean-field vector s_j
        s = (1.0 / d.N) * np.dot(c.cnt, np.dot(d.auxMat, np.dot(d.spikes, d.a_tau[:, tsyp])))

        if d.fp == 'noise':
            noiseinput = np.sqrt(2.0 * d.dt / d.tau * d.delta) * noise(d.N)
            # Excitatory
            d.matrix = qifint_noise(d.matrix, d.matrix[:, 0], d.matrix[:, 1], d.eta0, s + p.input,
                                    noiseinput, temps, d.N,
                                    d.dN, d.dt, d.tau, d.vpeak, d.refr_tau, d.tau_peak)
        else:
            # Excitatory
            d.matrix = qifint(d.matrix, d.matrix[:, 0], d.matrix[:, 1], d.eta, s + p.input, temps, d.N,
                              d.dN, d.dt, d.tau, d.vpeak, d.refr_tau, d.tau_peak)

        # Prepare spike matrices for Mean-Field computation and firing rate measure
        # Excitatory
        d.spikes_mod[:, tsk] = 1 * d.matrix[:, 2]  # We store the spikes
        d.spikes[:, tsyp] = 1 * d.spikes_mod[:, tskp]

        # If we are just obtaining the initial conditions (a steady state) we don't need to
        # compute the firing rate.
        if not d.new_ic:
            # Voltage measure:
            vma = (d.matrix[:, 1] <= temps)  # Neurons which are not in the refractory period
            fr.vavg0[vma] += d.matrix[vma, 0]
            fr.vavg += 1

            # ######################## -- FIRING RATE MEASURE -- ##
            fr.frspikes[:, tstep % fr.wsteps] = 1 * d.spikes[:, tsyp]
            fr.firingrate(tstep)
            # Distribution of Firing Rates
            if tstep > 0:
                fr.tspikes2 += d.matrix[:, 2]
                fr.ravg2 += 1  # Counter for the "instantaneous" distribution
                fr.ravg += 1  # Counter for the "total time average" distribution

    # ######################## -  INTEGRATION  - ##
    # ######################## --   FR EQS.   -- ##
    if d.system == 'nf' or d.system == 'both':
        # We compute the Mean-field vector S ( 1.0/(2.0*pi)*dx = 1.0/l )
        d.sphi[kp] = (1.0 / d.l) * np.dot(c.cnt, d.r[k])
        d.d[kp] = d.d[k] + d.dt * ((1.0 - d.d[k]) / d.taud - d.u * d.r[k] * d.d[k])

        # -- Integration -- #
        d.r[kp] = d.r[k] + d.dt * (d.delta / pi + 2.0 * d.r[k] * d.v[k])
        d.v[kp] = d.v[k] + d.dt * (d.v[k] * d.v[k] + d.eta0 + d.sphi[kp] * d.d[kp] - pi * pi * d.r[k] * d.r[k] + p.input)

    # Perturbation at certain time
    if int(p.t0 / d.dt) == tstep:
        p.pbool = True

    # Time evolution
    pbar.update(10 * tstep + 1)
    temps += d.dt
    tstep += 1

# Finish pbar
pbar.finish()
# Stop the timer
print 'Total time: {}.'.format(timer() - time1)

# Compute distribution of firing rates of neurons
tstep -= 1
temps -= d.dt
th.thdist = th.theor_distrb(d.sphi[kp])

# Save initial conditions
if d.new_ic:
    d.save_ic(temps)
    exit(0)

# Register data to a dictionary
if 'qif' in d.systems:
    # Distribution of firing rates over all time
    fr.frqif0 = fr.tspikes / (fr.ravg * d.dt) / d.faketau

    if 'nf' in d.systems:
        d.register_ts(fr, th)
    else:
        d.register_ts(fr)
else:
    d.register_ts(th=th)

# Save results
sr.create_dict(phi0=[d.l / 2, d.l / 4, d.l / 20], t0=int(d.total_time / 10) * np.array([2, 4, 6, 8]))
sr.results['perturbation']['It'] = p.it
sr.save()

# Preliminar plotting with gnuplot
gp = Gnuplot.Gnuplot(persist=1)
p1 = Gnuplot.PlotItems.Data(np.c_[d.tpoints * d.faketau, d.r[:, d.l / 2] / d.faketau], with_='lines')
if opts.s != 'nf':
    p2 = Gnuplot.PlotItems.Data(np.c_[np.array(fr.tempsfr) * d.faketau, np.array(fr.r)[:, d.l / 2] / d.faketau],
                                with_='lines')
else:
    p2 = Gnuplot.PlotItems.Data(np.c_[d.tpoints * d.faketau, p.it[:, d.l / 2] + d.r0 / d.faketau], with_='lines')
gp.plot(p1, p2)

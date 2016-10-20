#!/usr/bin/python2.7

import argparse
import yaml
import sys
from timeit import default_timer as timer
import progressbar as pb
import Gnuplot

import numpy as np
from nflib import Data, FiringRate, Connectivity
from tools import qifint, qifint_noise, TheoreticalComputations, SaveResults, Perturbation, noise

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__author__ = 'jm'


# Empty class to manage external parameters
# noinspection PyClassHasNoInit
class Options:
    pass


options = None
ops = Options()
pi = np.pi
pi2 = np.pi * np.pi

# We first try to parse optional configuration files:
fparser = argparse.ArgumentParser(add_help=False)
fparser.add_argument('-f', '--file', default="conf.txt", dest='-f', metavar='<file>')
farg = fparser.parse_known_args()
conffile = vars(farg[0])['-f']

# We open the configuration file to load parameters (not optional)
try:
    options = yaml.load(file(conffile, 'rstored'))
except IOError:
    print "The configuration file '%s' is missing" % conffile
    exit(-1)
except yaml.YAMLError, exc:
    print "Error in configuration file:", exc
    exit(-1)

# We load parameters from the dictionary of the conf file and add command line options (2nd parsing)
parser = argparse.ArgumentParser(
    description='Simulator of a network of ensembles of all-to-all QIF neurons.',
    usage='python %s [-O <options>]' % sys.argv[0])

for group in options:
    gr = parser.add_argument_group(group)
    for key in options[group]:
        flags = key.split()
        args = options[group]
        gr.add_argument(*flags, default=args[key]['default'], help=args[key]['description'], dest=flags[0][1:],
                        metavar=args[key]['name'], type=type(args[key]['default']),
                        choices=args[key]['choices'])

# We parse command line arguments:
opts = parser.parse_args(farg[1])
args = parser.parse_args(farg[1], namespace=ops)

###################################################################################
# 0) PREPARE FOR CALCULATIONS
# 0.1) Load data object:
d = Data(l=args.n, n=args.N, eta0=args.e, j0=args.j, delta=args.d, tfinal=args.T, system=args.s, fp=args.D)

# 0.2) Create connectivity matrix and extract eigenmodes
c = Connectivity(d.l, profile=args.c, fsmodes=args.jk, amplitude=10.0, data=d, degree=args.dg, saved=True)

# 0.3) Load initial conditions
if args.oic is False:
    d.load_ic(0.0, system=d.system)
else:
    # Override initial conditions generator:
    pass
if args.ic:
    print "Forcing initial conditions generation..."
    d.new_ic = True

# 0.4) Load Firing rate class in case qif network is simulated
if d.system != 'nf':
    fr = FiringRate(data=d, swindow=0.5, sampling=0.05)

# 0.5) Set perturbation configuration
p = Perturbation(data=d, dt=args.pt, modes=args.m, amplitude=float(args.a), attack=args.A, cntmodes=c.modes[1],
                 t0=args.pt0)

# 0.6) Define saving paths:
sr = SaveResults(data=d, cnt=c, pert=p, system=d.system, parameters=opts)

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
nois = 0.0
noiseinput = 0.0
kp = k = 0

# Time loop
while temps < d.tfinal:
    # Time step variables
    kp = tstep % d.nsteps
    k = (tstep + d.nsteps - 1) % d.nsteps
    # ######################## - PERTURBATION  - ##
    if p.pbool and not d.new_ic:
        if temps >= p.t0:
            p.timeevo(temps)
    if args.ns and not d.new_ic:
        nois = np.sqrt(2.0 * args.nD) * np.random.randn(d.l)
    p.it[kp, :] = p.input + d.tau / d.dt * noise

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
        d.matrix = qifint(d.matrix, d.matrix[:, 0], d.matrix[:, 1], d.eta, s + p.input + d.tau / d.dt * noiseinput,
                          temps, d.N, d.dN, d.dt, d.tau, d.vpeak, d.refr_tau, d.tau_peak)

        # Prepare spike matrices for Mean-Field computation and firing rate measure
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
        # -- Integration -- #
        d.r[kp] = d.r[k] + d.dt * (d.delta / pi + 2.0 * d.r[k] * d.v[k])
        d.v[kp] = d.v[k] + d.dt * (
            d.v[k] * d.v[k] + d.eta0 + d.sphi[kp] * d.d[kp] - pi * pi * d.r[k] * d.r[k] + p.input)

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
if args.s != 'nf':
    p2 = Gnuplot.PlotItems.Data(np.c_[np.array(fr.tempsfr) * d.faketau, np.array(fr.r)[:, d.l / 2] / d.faketau],
                                with_='lines')
else:
    p2 = Gnuplot.PlotItems.Data(np.c_[d.tpoints * d.faketau, p.it[:, d.l / 2] + d.r0 / d.faketau], with_='lines')
gp.plot(p1, p2)

np.savetxt("p%d.dat" % args.m, np.c_[d.tpoints * d.faketau, d.r[:, d.l / 2] / d.faketau])

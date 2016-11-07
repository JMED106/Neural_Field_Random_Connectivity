#!/usr/bin/python2.7

import argparse
import yaml
import sys
from timeit import default_timer as timer
import progressbar as pb
import numpy as np
from nflib import Data, FiringRate, Connectivity
from tools import qifint, TheoreticalComputations, SaveResults, Perturbation, noise, FrequencySpectrum

import logging

import Gnuplot

# Use this option to turn off fifo if you get warnings like:
# line 0: warning: Skipping unreadable file "/tmp/tmpakexra.gnuplot/fifo"
Gnuplot.GnuplotOpts.prefer_fifo_data = 0


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

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
        logger.debug("flag: %4s\t default: %13s\t %s" % (
            flags[0][1:], str(args[key]['default']), str(type(args[key]['default']))))
        if isinstance(args[key]['default'], bool):
            logger.debug(args[key]['default'])
            gr.add_argument(*flags, default=args[key]['default'], help=args[key]['description'], dest=flags[0][1:],
                            action='store_true')
        else:
            gr.add_argument(*flags, default=args[key]['default'], help=args[key]['description'], dest=flags[0][1:],
                            metavar=args[key]['name'], type=type(args[key]['default']),
                            choices=args[key]['choices'])

# We parse command line arguments:
opts = parser.parse_args(farg[1])
args = parser.parse_args(farg[1], namespace=ops)

# ####### Debugging #########
logger.setLevel(args.db)

# ##################################################################################
# 0) PREPARE FOR CALCULATIONS
# 0.1) Load data object:
d = Data(l=args.n, n=args.N, eta0=args.e, j0=args.j, delta=args.d, tfinal=args.T, system=args.s, fp=args.D,
         debug=args.db, dt=args.dt)

# 0.2) Create connectivity matrix and extract eigenmodes
c = Connectivity(d.l, profile=args.c, fsmodes=args.jk, amplitude=10.0, data=d, degree=args.dg, saved=True)

# 0.3) Load initial conditions
if args.oic is False:
    d.load_ic(0.0, system=d.system)
else:
    # Override initial conditions generator:
    pass
if args.ic:
    logger.info("Forcing initial conditions generation...")
    d.new_ic = True

# 0.4) Load Firing rate class in case qif network is simulated
if d.system != 'nf':
    fr = FiringRate(data=d, swindow=0.5, sampling=0.05)

# 0.5) Set perturbation configuration
p = Perturbation(data=d, dt=args.pt, modes=args.m, amplitude=float(args.a), attack=args.A, cntmodes=c.eigenvectors,
                 t0=args.pt0, debug=args.db)

# 0.6) Define saving paths:
sr = SaveResults(data=d, cnt=c, pert=p, system=d.system, parameters=opts)

# 0.7) Other theoretical tools:
th = TheoreticalComputations(d, c, p)
F = FrequencySpectrum()

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
    k2p = tstep % 2
    k2 = (tstep + 2 - 1) % 2
    # ######################## - PERTURBATION  - ##
    if p.pbool and not d.new_ic:
        if temps >= p.t0:
            p.timeevo(temps)
            pt0step = tstep * 1
    # Noisy perturbation
    if args.ns and not d.new_ic:
        if tstep % 1 == 0:
            nois = np.sqrt(2.0 * d.dt / d.tau * args.nD) * np.random.randn(d.l)
        else:
            nois = 0.0

    # Another perturbation (directly changing mean potentials)
    if tstep == p.t0step:
        d.v_ex[k] += 0.0
        d.v_in[k] += 0.0

    p.it[kp, :] = p.input + d.tau / d.dt * nois

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
        d.sphi[k2p] = (1.0 / d.l * np.dot(c.cnt_ex, d.r_ex[k]) + 1.0 / d.l * np.dot(c.cnt_in, d.r_in[k]))
        # -- Integration -- #
        d.r_ex[kp] = d.r_ex[k] + d.dt * (d.delta / pi + 2.0 * d.r_ex[k] * d.v_ex[k])
        d.v_ex[kp] = d.v_ex[k] + d.dt * (d.v_ex[k] ** 2 + d.eta0 + d.sphi[k2p] - pi2 * d.r_ex[k] ** 2 + p.it[kp])
        d.r_in[kp] = d.r_in[k] + d.dt * (d.delta / pi + 2.0 * d.r_in[k] * d.v_in[k])
        d.v_in[kp] = d.v_in[k] + d.dt * (d.v_in[k] ** 2 + d.eta0 + d.sphi[k2p] - pi2 * d.r_in[k] ** 2 + args.sym * p.it[kp])

    # Perturbation at certain time
    if int(p.t0 / d.dt) == tstep:
        p.pbool = True

    # Compute the frequency by hand (for a given node, typically at the center)

    # Time evolution
    pbar.update(10 * tstep + 1)
    temps += d.dt
    tstep += 1

# Finish pbar
pbar.finish()
# Stop the timer
print 'Total time: {}.'.format(timer() - time1)

logger.info("Stationary firing rate (excitatory and inhibitory): %f, %f" % (d.r_ex[kp, 0], d.r_in[kp, 0]))
logger.info("Stationary mean membrane potential (excitatory and inhibitory): %f, %f" % (d.v_ex[kp, 0], d.v_in[kp, 0]))

# Compute distribution of firing rates of neurons
tstep -= 1
temps -= d.dt
# th.thdist = th.theor_distrb(d.sphi[kp])

# Save initial conditions
if d.new_ic:
    d.save_ic(temps)
    exit(0)

# Save data to dictionary
if not args.nos:
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

# Save just some data and plot
if args.pl:
    pass

# # # Preliminary plotting with gnuplot
if args.gpl:
    if d.nsteps > 10E6:
        points = d.nsteps/10E6
        if points <= 1:
            points = 10
    else:
        points = 1
    logger.info(points)
    gp = Gnuplot.Gnuplot(persist=1)
    p1_in = Gnuplot.PlotItems.Data(np.c_[d.tpoints[::points] * d.faketau, d.r_ex[::points, d.l / 2] / d.faketau], with_='lines')
    # p1_ex = Gnuplot.PlotItems.Data(np.c_[d.tpoints[::points] * d.faketau, d.r_in[::points, d.l / 2] / d.faketau], with_='lines')
    # p1_exin = Gnuplot.PlotItems.Data(
    #     np.c_[d.tpoints[::points] * d.faketau, (d.r_in[::points, d.l / 2] + d.r_ex[::points, d.l / 2]) / 2.0 / d.faketau], with_='lines')
    # p1v_in = Gnuplot.PlotItems.Data(np.c_[d.tpoints[::points] * d.faketau, d.v_ex[::points, d.l / 2] / d.faketau], with_='lines')
    # p1v_ex = Gnuplot.PlotItems.Data(np.c_[d.tpoints[::points] * d.faketau, d.v_in[::points, d.l / 2] / d.faketau], with_='lines')
    # p11_in = Gnuplot.PlotItems.Data(np.c_[d.tpoints[::points] * d.faketau, d.r_ex[::points, d.l / 4] / d.faketau], with_='lines')
    # p11_ex = Gnuplot.PlotItems.Data(np.c_[d.tpoints[::points] * d.faketau, d.r_in[::points, d.l / 4] / d.faketau], with_='lines')
    # p11_exin = Gnuplot.PlotItems.Data(
    #     np.c_[d.tpoints[::points] * d.faketau, (d.r_in[::points, d.l / 4] + d.r_ex[::points, d.l / 4]) / 2.0 / d.faketau], with_='lines')
    if args.s != 'nf':
        p2 = Gnuplot.PlotItems.Data(np.c_[np.array(fr.tempsfr) * d.faketau, np.array(fr.r)[::points, d.l / 2] / d.faketau],
                                    with_='lines')
    else:
        p2 = Gnuplot.PlotItems.Data(np.c_[d.tpoints[::points] * d.faketau, p.it[::points, d.l / 2] + d.r0 / d.faketau], with_='lines')
    # gp.plot(p1_ex, p1_in, p1_exin)
    gp.plot(p1_in, p2)
    # gp2 = Gnuplot.Gnuplot(persist=1)
    # gp2.plot(p1v_ex, p1v_in)
    gp3 = Gnuplot.Gnuplot(persist=1)
    # gp3.plot(p11_ex, p11_in, p11_exin)
    raw_input("Enter to exit ...")

    # np.savetxt("p%d.dat" % args.m, np.c_[d.tpoints[::points] * d.faketau, d.r[::points, d.l / 2] / d.faketau])

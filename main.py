#!/usr/bin/python2.7

import argparse
import yaml
import sys

import logging.config
from colorlog import ColoredFormatter
from timeit import default_timer as timer
import progressbar as pb
import numpy as np
from nflib import Data, FiringRate, Connectivity
from tools import qifint, TheoreticalComputations, SaveResults, Perturbation, noise, FrequencySpectrum, ColorPlot, \
    LinearStability, decay

import Gnuplot

# Use this option to turn off fifo if you get warnings like:
# line 0: warning: Skipping unreadable file "/tmp/tmpakexra.gnuplot/fifo"
Gnuplot.GnuplotOpts.prefer_fifo_data = 0

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
fparser.add_argument('-db', '--debug', default="DEBUG", dest='db', metavar='<debug>',
                     choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
farg = fparser.parse_known_args()
conffile = vars(farg[0])['-f']
# ####### Debugging #########
debug = getattr(logging, vars(farg[0])['db'].upper(), None)
if not isinstance(debug, int):
    raise ValueError('Invalid log level: %s' % vars(farg[0])['db'])

logformat = "%(log_color)s[%(levelname)-7.8s]%(reset)s %(name)-12.12s:%(funcName)-8.8s: " \
            "%(log_color)s%(message)s%(reset)s"
formatter = ColoredFormatter(logformat, log_colors={
    'DEBUG': 'cyan',
    'INFO': 'white',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
})

logging.config.dictConfig(yaml.load(file('logging.conf', 'rstored')))
handler = logging.root.handlers[0]
handler.setLevel(debug)
handler.setFormatter(formatter)
logger = logging.getLogger('simulation')

# We open the configuration file to load parameters (not optional)
try:
    options = yaml.load(file(conffile, 'rstored'))
except IOError:
    logger.error("The configuration file '%s' is missing" % conffile)
    exit(-1)
except yaml.YAMLError, exc:
    logger.error("Error in configuration file:", exc)
    exit(-1)

# We load parameters from the dictionary of the conf file and add command line options (2nd parsing)
parser = argparse.ArgumentParser(
    description='Simulator of a network of ensembles of all-to-all QIF neurons.',
    usage='python %s [-O <options>]' % sys.argv[0])
print "\n******************************************************************"
logger.info('Simulator of a network of ensembles of all-to-all QIF neurons.')
for group in options:
    gr = parser.add_argument_group(group)
    for key in options[group]:
        flags = key.split()
        args = options[group]
        if isinstance(args[key]['default'], bool):
            gr.add_argument(*flags, default=args[key]['default'], help=args[key]['description'], dest=flags[0][1:],
                            action='store_true')
        elif isinstance(args[key]['default'], list):
            tipado = type(args[key]['default'][0])
            gr.add_argument(*flags, default=args[key]['default'], help=args[key]['description'], dest=flags[0][1:],
                            metavar=args[key]['name'], type=tipado,
                            choices=args[key]['choices'], nargs='+')
        else:
            gr.add_argument(*flags, default=args[key]['default'], help=args[key]['description'], dest=flags[0][1:],
                            metavar=args[key]['name'], type=type(args[key]['default']),
                            choices=args[key]['choices'])

# We parse command line arguments:
opts = parser.parse_args(farg[1])
args = parser.parse_args(farg[1], namespace=ops)

# ##################################################################################
# 0) PREPARE FOR CALCULATIONS
# 0.1) Load data object:
d = Data(l=args.n, n=args.N, eta0=args.e, j0=args.j, delta=args.d, tfinal=args.T, system=args.s, fp=args.D, dt=args.dt)

# 0.2) Create connectivity matrix and extract eigenmodes
c = Connectivity(d.l, profile=args.c, fsmodes=args.jk, amplitude=10.0, data=d, degree=args.dg, saved=True)

# 0.3) Load initial conditions
if args.oic is False:
    d.load_ic(0.0, system=d.system, ext=args.ext)
else:
    # Override initial conditions generator:
    pass
if args.ic:
    logger.info("Forcing initial conditions generation...")
    d.new_ic = True

# 0.4) Load Firing rate class in case qif network is simulated
if d.system != 'nf' and not d.new_ic:
    fr = FiringRate(data=d, swindow=0.5, sampling=0.05)

# 0.5) Set perturbation configuration
# mode patch
if args.m == -1:
    args.m = range(0, 10)
p = list()
p.append(Perturbation(data=d, dt=args.pt, modes=args.m, amplitude=float(args.a), attack=args.A, cntmodes=c.eigenvectors,
                      t0=args.pt0, stype=args.sP, ptype=args.pT, duration=d.total_time))
if args.pV:
    p[0].amp = 0.0
# p.append(Perturbation(data=d, dt=args.pt, modes=[1], amplitude=float(args.a), attack=args.A, cntmodes=c.eigenvectors,
#                       t0=2.5, stype=args.sP, ptype=args.pT, duration=d.total_time))

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
if args.rast:
    raster = file('/home/jm/raster0.dat', 'w')
    raster.close()
    raster = file('/home/jm/raster0.dat', 'a')
else:
    raster = None
pbar = pb.ProgressBar(widgets=widgets, maxval=10 * (d.nsteps + 1)).start()
time1 = timer()
tstep = 0
temps = 0
nois = 0.0
noise_v = 0.0
kp = k = 0
freq = 5
# Time loop
while temps < d.tfinal:
    # Time step variables
    kp = tstep % d.nsteps
    k = (tstep + d.nsteps - 1) % d.nsteps
    k2p = tstep % 2
    k2 = (tstep + 2 - 1) % 2
    # ######################## - PERTURBATION  - ##
    # Perturbation at certain time
    for pert in p:
        if pert.t0step == tstep:
            pert.pbool = True
        if pert.pbool and not d.new_ic:
            if temps >= pert.t0:
                pert.timeevo(temps, freq=freq)
                pt0step = tstep * 1
        d.it[kp, :] += pert.input
    # Noisy perturbation
    if args.ns and not d.new_ic:
        if tstep % 1 == 0:
            # nois = np.sqrt(2.0 * d.dt / d.tau * args.nD) * np.random.randn(d.l / 10)
            # nois = np.dot(p.auxMat, nois)
            nois = np.sqrt(2.0 * d.dt / d.tau * args.nD) * np.random.rand(d.l)
        else:
            nois = 0.0

    d.it[kp, :] += d.tau / d.dt * nois

    # ######################## -  INTEGRATION  - ##
    # ######################## -      qif      - ##
    if d.system == 'qif' or d.system == 'both':
        tsyp = tstep % d.T_syn
        tskp = tstep % d.spiketime
        tsk = (tstep + d.spiketime - 1) % d.spiketime
        # We compute the Mean-field vector s_j
        s = (1.0 / d.N) * np.dot(c.cnt, np.dot(d.auxMat, np.dot(d.spikes, d.a_tau[:, tsyp])))

        # Another perturbation (directly changing mean potentials)
        for pert in p:
            if tstep == pert.t0step and args.pV and not d.new_ic:
                d.matrix[:, 0] += np.dot(pert.auxMatD, args.pD * pert.smod)

        if d.fp == 'noise':
            noise_v = np.sqrt(2.0 * d.dt / d.tau * d.delta) * noise(d.N)

        # Excitatory
        d.matrix = qifint(d.matrix, d.matrix[:, 0], d.matrix[:, 1], d.eta + d.tau / d.dt * noise_v,
                          s + d.it[kp, :],
                          temps, d.N, d.dN, d.dt, d.tau, d.vpeak, d.refr_tau, d.tau_peak)

        # Store spikes to create a raster plot:
        if args.rast:
            if tstep % 75 == 0:
                spikes = (d.matrix[:, 1] > temps)
                neurons = np.argwhere(spikes == True)
                times = d.matrix[spikes, 1]
                if len(times) > 0:
                    np.savetxt(raster, np.c_[times, neurons])

        # Prepare spike matrices for Mean-Field computation and firing rate measure
        # Excitatory
        d.spikes_mod[:, tsk] = 1 * d.matrix[:, 2]  # We store the spikes
        d.spikes[:, tsyp] = 1 * d.spikes_mod[:, tskp]
        # If we are just obtaining the initial conditions (a steady state) we don't need to
        # compute the firing rate.
        if not d.new_ic:
            # Voltage measure:
            # vma = (d.matrix[:, 1] <= temps)  # Neurons which are not in the refractory period
            # fr.vavg0[vma] += d.matrix[vma, 0]
            # fr.vavg += 1

            # ######################## -- FIRING RATE MEASURE -- ##
            fr.frspikes[:, tstep % fr.wsteps] = 1 * d.spikes[:, tsyp]
            fr.firingrate(tstep)
            # Distribution of Firing Rates
            if tstep > 0:
                fr.tspikes2 += d.matrix[:, 2]
                fr.ravg2 += 1  # Counter for the "instantaneous" distribution
                fr.ravg += 1  # Counter for the "total time average" distribution

                # Check if both populations (ex. and inh.) are doing the same thing
                # if not np.all(d.matrixE == d.matrixI):
                #     logger.debug("'Symmetry' breaking... not exactly.")

    # ######################## -  INTEGRATION  - ##
    # ######################## --   FR EQS.   -- ##
    if d.system == 'nf' or d.system == 'both':
        # Another perturbation (directly changing mean potentials)
        for pert in p:
            if tstep == pert.t0step and args.pV and not d.new_ic:
                d.v[k] += args.pD * pert.smod

        # We compute the Mean-field vector S ( 1.0/(2.0*pi)*dx = 1.0/l )
        d.sphi[k2p] = 1.0 / d.l * np.dot(c.cnt, d.r[k])
        # if tstep%100 == 0:
        #     logger.debug("Mean synaptic input: %s" % d.sphi[k2p])
        # -- Integration -- #
        d.r[kp] = d.r[k] + d.dt * (d.delta / pi + 2.0 * d.r[k] * d.v[k])
        d.v[kp] = d.v[k] + d.dt * (d.v[k] ** 2 + d.eta0 + d.sphi[k2p] - pi2 * d.r[k] ** 2 + d.it[kp])

    # Compute the frequency by hand (for a given node, typically at the center)

    # Time evolution
    pbar.update(10 * tstep + 1)
    temps += d.dt
    tstep += 1

# Finish pbar
pbar.finish()
# Stop the timer
print 'Total time: {}.'.format(timer() - time1)
if args.rast:
    raster.close()
###################################################################################
# 2) Post-Simulation, saving, plotting, analayzing.
logger.debug("Stationary firing rate : %f" % d.r[kp, 0])
logger.debug("Stationary mean membrane potential : %f" % d.v[kp, 0])

# Compute distribution of firing rates of neurons
tstep -= 1
temps -= d.dt
# th.thdist = th.theor_distrb(d.sphi[kp])

# Frequency analysis
envelopes = []
envelope = {}
if args.Frq:
    plot = ColorPlot(data=d)
    F.analyze(d.r.mean(axis=1) - d.r0, 0.0, d.tfinal, d.faketau, method='all', plotting=False)
    for phi in xrange(d.l):
        F.analyze(d.r[:, phi] - d.r0, 0.0, d.tfinal, d.faketau, method='all', plotting=False)
        F.analyze(plot.filter(d.r - d.r0)[:, phi], 0.0, d.tfinal, d.faketau, method='all', plotting=False)
    for pert in p:
        if pert.ptype == 'oscillatory' or pert.ptype == 'chirp':
            totalenvelope = LinearStability.total_envelope(d.r, d.tpoints, tau=d.faketau)
            totalenvelope['freqs'] = Perturbation.chirp_freqs(totalenvelope['t'], pert.chirp_t0, pert.fmin, pert.chirp_t1, pert.fmax)

            for pop in xrange(d.l):
                if 'qif' in d.systems:
                    envelope['qif'] = LinearStability.envelope2extreme(fr.r, fr.tempsfr, tau=d.faketau)
                if 'nf' in d.systems:
                    envelope['nf'] = LinearStability.envelope2extreme(d.r, d.tpoints, tau=d.faketau, pop=pop)
                    envelope['nf']['freqs'] = Perturbation.chirp_freqs(envelope['nf']['t'], pert.chirp_t0, pert.fmin,
                                                                       pert.chirp_t1, pert.fmax)
                envelopes.append(envelope)
                envelope = {}

# Save initial conditions
if d.new_ic:
    d.save_ic(temps)
    exit(0)

# Save data to dictionary
if not args.nos:
    # Register data to a dictionary
    if 'qif' in d.systems:
        # Distribution of firing rates over all time
        fr.frqif = fr.tspikes / (fr.ravg * d.dt) / d.faketau

        if 'nf' in d.systems:
            d.register_ts(fr, th)
        else:
            d.register_ts(fr)
    else:
        d.register_ts(th=th)

    sr.create_dict()
    sr.results['perturbation']['It'] = d.it
    for pert in p:
        if pert.ptype == 'oscillatory':
            sr.results['perturbation']['freqs'] = pert.freq
            if args.Frq and pert.ptype in ['']:
                for sys in d.systems:
                    sr.results[sys]['fr']['envelope'] = envelope[sys]
        if pert.ptype == 'chirp':
            sr.results['perturbation']['chirp'] = pert.chirp
            for sys in d.systems:
                sr.results[sys]['fr']['envelope'] = envelope[sys]
    sr.save()

# Save just some data and plot
if args.pl:
    plot = ColorPlot(data=d, tfinal=70)
    plot.cplot(d.r)
    plot.cplot(plot.filter(d.r))

# Preliminary plotting with gnuplot
if args.gpl:
    gpllog = logger.getChild('gnuplot')
    if d.nsteps > 10E6:
        points = d.nsteps / 10E6
        if points <= 1:
            points = 10
    else:
        points = 1
    gpllog.info("Plotting every %d points", points)
    gp = Gnuplot.Gnuplot(persist=1)
    p1 = []
    for k in xrange(d.l):
        p1.append(Gnuplot.PlotItems.Data(np.c_[d.tpoints[::points] * d.faketau, d.r[::points, k] / d.faketau],
                                         with_='lines'))
        # decay_time = decay(d.tpoints * d.faketau, d.r[:, k] / d.faketau)
        # logger.debug("Decay time for population %d is tau_%d = %f" % (k, k, decay_time))
    # p1_ex = Gnuplot.PlotItems.Data(np.c_[d.tpoints[::points] * d.faketau, d.r[::points, d.l / 2] / d.faketau],
    #                                with_='lines')
    # p2_ex = Gnuplot.PlotItems.Data(np.c_[d.tpoints[::points] * d.faketau, d.r[::points, d.l / 3] / d.faketau],
    #                                with_='lines')
    if args.s != 'nf':
        p2 = Gnuplot.PlotItems.Data(
            np.c_[np.array(fr.tempsfr) * d.faketau, np.array(fr.r)[::points, d.l / 2] / d.faketau],
            with_='lines')
    else:
        p2 = Gnuplot.PlotItems.Data(np.c_[d.tpoints[::points] * d.faketau, d.it[::points, d.l / 2] + d.r0 / d.faketau],
                                    with_='lines')
    # p1.append(p2)

    gp.plot(*p1)
    # gp2 = Gnuplot.Gnuplot(persist=1)
    # gp2.plot(p1v_ex, p1v_in)
    for pert in p:
        if args.Frq and pert.ptype in ('chirp', 'oscillatory'):
            gp3 = Gnuplot.Gnuplot(persist=1)
            penvelopes = []
            for pop in xrange(d.l):
                penvelopes.append(Gnuplot.PlotItems.Data(np.c_[envelopes[pop]['nf']['freqs'] / d.faketau, envelopes[pop]['nf']['envelope'] / d.faketau], with_='lines'))
            gp3.plot(*penvelopes)

            gp4 = Gnuplot.Gnuplot(persist=1)
            ptotalenvelope = Gnuplot.PlotItems.Data(
                np.c_[totalenvelope['freqs'] / d.faketau, totalenvelope['envelope'] / d.faketau],
                with_='lines')
            gp4.plot(ptotalenvelope)

    raw_input("Enter to exit ...")

    # np.savetxt("p%d.dat" % args.m, np.c_[d.tpoints[::points] * d.faketau, d.r[::points, d.l / 2] / d.faketau])

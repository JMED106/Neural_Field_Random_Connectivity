import numpy as np
import psutil
from scipy import stats, special
from scipy.fftpack import dct
from scipy.optimize import fsolve

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__author__ = 'Jose M. Esnaola Acebes'

""" This file contains classes and functions to be used in the Neural Field simulation.

    Data: (to store parameters, variables, and some functions)
    *****

    Connectivity:  (to build the connectivity matrix of the network)
    *************
    1. Functions:

        1.1. Gaussian pdf with mean 0.
        1.2. Non-periodic Mex-Hat function.
        1.3. Circular gaussian function: mex-hat type. (von Mises functions).

    2. Methods:

        2.1. Mode extraction by Fourier Series: eingevalues.
        2.2. Reconstruction of a function by means of a Fourier Series using 2.1.
        2.3. Computation of frequencies for a given connectivity (requires parameters).
        2.4. Linear response of the homogeneous state (requires parameters and initial
             conditions).
"""


class Data:
    """ Object designed to store data,
        and perform modifications of this data in case it is necessary.
    """

    def __init__(self, l=100, n=1E5, eta0=0, j0=0.0, delta=1.0, t0=0.0, tfinal=50.0,
                 dt=1E-3, delay=0.0, tau=1.0, faketau=20.0E-3, fp='lorentz', system='nf'):

        # 0.1) Network properties:
        self.l = l
        self.dx = 2.0 * np.pi / np.float(l)
        # Zeroth mode, determines firing rate of the homogeneous state
        self.j0 = j0  # default value

        # 0.3) Give the model parameters
        self.eta0 = eta0  # Constant external current mean value
        self.delta = delta  # Constant external current distribution width

        # 0.2) Define the temporal resolution and other time-related variables
        self.t0 = t0  # Initial time
        self.tfinal = tfinal  # Final time
        self.total_time = tfinal - t0  # Time of simulation
        self.dt = dt  # Time step

        self.taud = 100
        self.u = 0.05

        self.D = delay  # Synaptic time Delay (not implemented)
        self.intD = int(delay / dt)  # Synaptic time Delay in time steps
        self.tpoints = np.arange(t0, tfinal, dt)  # Points for the plots and others
        self.nsteps = len(self.tpoints)  # Total time steps
        self.tau = tau
        self.faketau = faketau  # time scale in ms

        # 0.7) FIRING RATE EQUATIONS
        self.r = np.zeros((self.nsteps, l))
        self.v = np.ones((self.nsteps, l)) * (-0.01)
        self.sphi = np.ones((2, l))
        self.r[len(self.r) - 1, :] = 0.1
        # Load INITIAL CONDITIONS
        self.sphi[len(self.sphi) - 1] = 0.0

        self.system = system
        self.systems = []
        if system == 'qif' or system == 'both':
            self.systems.append('qif')
        if system == 'nf' or system == 'both':
            self.systems.append('nf')

        # 0.8) QIF model parameters
        if system != 'nf':
            print "Loading QIF parameters:"
            print "***********************"
            self.fp = fp
            # sub-populations
            self.N = n
            # Excitatory and inhibitory populations
            self.neni = 0.5

            # sub-populations
            self.dN = int(np.float(n) / np.float(l))
            if self.dN * l != n:
                print('Warning: n, l not dividable')

            self.vpeak = 100.0  # Value of peak voltage (max voltage)
            # self.vreset = -self.vpeak  # Value of resetting voltage (min voltage)
            self.vreset = -100.0
            # --------------
            self.refr_tau = tau / self.vpeak - tau / self.vreset  # Refractory time in which the neuron is not excitable
            self.tau_peak = tau / self.vpeak  # Refractory time till the spike is generated
            # --------------
            self.T_syn = 10  # Number of steps for computing synaptic activation
            self.tau_syn = self.T_syn * dt  # time scale (??)
            # Weighting matrix (synaptic delay, in some sense).
            # We need T_syn vectors in order to improve the performance.
            if self.T_syn == 10:
                # Heaviside
                h_tau = 1.0 / self.tau_syn
                a_tau0 = np.transpose(h_tau * np.ones(self.T_syn))
            else:
                # Exponential (disabled by the Heaviside)
                # self.tau_syn /= 4
                a_tau0 = np.transpose((1.0 / self.tau_syn) * np.exp(-dt * np.arange(self.T_syn) / self.tau_syn))

            self.a_tau = np.zeros((self.T_syn, self.T_syn))  # Multiple weighting vectors (index shifted)
            for i in xrange(self.T_syn):
                self.a_tau[i] = np.roll(a_tau0, i, 0)

            # Distributions of the external current       -- FOR l populations --
            self.eta = None
            if fp == 'lorentz' or fp == 'gauss':
                print "+ Setting distribution of external currents: "
                self.eta = np.zeros(self.N)
                if fp == 'lorentz':
                    print '   - Lorentzian distribution of external currents'
                    # Uniform distribution
                    k = (2.0 * np.arange(1, self.dN + 1) - self.dN - 1.0) / (self.dN + 1.0)
                    # Cauchy ppf (stats.cauchy.ppf can be used here)
                    eta_pop_e = eta0 + delta * np.tan((np.pi / 2.0) * k)
                else:
                    print '   - Gaussian distribution of external currents'
                    k = (np.arange(1, self.dN + 1)) / (self.dN + 1.0)
                    eta_pop_e = eta0 + delta * stats.norm.ppf(k)

                del k
                for i in xrange(l):
                    self.eta[i * self.dN:(i + 1) * self.dN] = 1.0 * eta_pop_e
                del eta_pop_e
            elif fp == 'noise':
                print "+ Setting homogeneous population of neurons (identical), under GWN."
                self.eta = np.ones(self.N) * self.eta0
            else:
                print "This distribution is not implemented, yet."
                exit(-1)

            # QIF neurons matrices (declaration)
            self.matrix = np.ones(shape=(self.N, 3)) * 0
            self.spikes = np.ones(shape=(self.N, self.T_syn)) * 0  # Spike matrix (Ne x T_syn)

            # Single neuron recording (not implemented)
            self.singlev = np.ones(self.nsteps) * 0.0
            self.freqpoints = 25
            self.singleta = np.ones(self.freqpoints)
            self.singlfreqs = np.ones(self.freqpoints)
            self.singlaux = 0
            self.singl0 = 0.0

            # 0.8.1) QIF vectors and matrices (initial conditions are loaded after
            #                                  connectivity  matrix is created)
            self.spiketime = int(self.tau_peak / dt)
            self.s1time = self.T_syn + self.spiketime
            self.spikes_mod = np.ones(shape=(self.N, self.spiketime)) * 0  # Spike matrix (Ne x (T_syn + tpeak/dt))
            # Auxiliary matrix
            self.auxMat = np.zeros((self.l, self.N))
            for i in xrange(self.l):
                self.auxMat[i, i * self.dN:(i + 1) * self.dN] = 1.0

        # 0.9) Perturbation parameters
        self.PERT = False
        self.Input = 0.0
        self.It = np.zeros(self.nsteps)
        self.It[0] = self.Input * 1.0
        # input duration
        self.deltat = dt * 50
        self.inputtime = 10.0

        # Simulations parameters
        self.new_ic = False
        self.END = False

        # Post simulation:
        self.rstored = {x: None for x in self.systems}
        self.vstored = {x: None for x in self.systems}
        self.t = {x: None for x in self.systems}
        self.k = {x: None for x in self.systems}
        self.dr = {x: None for x in self.systems}

    def load_ic(self, j0, system='nf'):
        """ Loads initial conditions based on the parameters. It will try to load system that
            most closely resembles. The available systems are stored in a file.
        """
        # File path variables
        self.filepath = './init_conds/qif/'
        # TODO compute the fixed point taking into account the parameter space: HS or Bump?
        self.r0 = Connectivity.rtheory(j0, self.eta0, self.delta)

        if system == 'nf' or system == 'both':
            self.fileprm = '%.2lf-%.2lf-%.2lf-%d' % (j0, self.eta0, self.delta, self.l)
            if len(self.v[:, 0]) == 2:
                self.v[-1, :] = np.ones(self.l) * (-self.delta / (2.0 * self.r0 * np.pi))
            else:
                self.r[(self.nsteps - 1) % self.nsteps, :] = np.ones(self.l) * self.r0
                self.v[-1, :] = np.ones(self.l) * (-self.delta / (2.0 * self.r0 * np.pi))

        if system == 'qif' or system == 'both':
            print "Loading initial conditions ... "
            if np.abs(j0) < 1E-2:
                j0zero = 0.0
            else:
                j0zero = j0
            self.fileprm = '%s_%.2lf-%.2lf-%.2lf-%d' % (self.fp, j0zero, self.eta0, self.delta, self.l)
            # We first try to load files that correspond to chosen parameters
            try:
                self.spikes = np.load("%sic_qif_spikes_%s-%d.npy" % (self.filepath, self.fileprm, self.N))
                self.matrix = np.load("%sic_qif_matrix_%s-%d.npy" % (self.filepath, self.fileprm, self.N))
                print "Successfully loaded all data matrices."
            except IOError:
                print "Files do not exist or cannot be read. Trying the most similar combination."
                self.new_ic = True
            except ValueError:
                print "Not appropriate format of initial conditions. Check the files for logical errors..."
                exit(-1)

            # If the loading fails or new_ic is overridden we look for the closest combination in the data base
            database = None
            if self.new_ic is True:
                print "WARNING: New initial conditions will be created, wait until the simulation has finished."
                try:
                    database = np.load("%sinitial_conditions_%s.npy" % (self.filepath, self.fp))
                    if np.size(np.shape(database)) < 2:
                        database.resize((1, np.size(database)))
                    load = True
                except IOError:
                    print "Iinitial conditions database not found (%sinitial_conditions_%s)" % (self.filepath, self.fp)
                    print "Loading random conditions."
                    load = False

                # If the chosen combination is not in the database we create new initial conditions
                # for that combination: raising a warning to the user.
                # If the database has been successfully loaded we find the closest combination
                # Note that the number of populations must coincide
                if load is True and np.any(database[:, 0] == self.l) \
                        and np.any(database[:, -1] == self.N):
                    # mask combinations where population number match
                    ma = ((database[:, 0] == self.l) & (database[:, -1] == self.N))
                    # Find the closest combination by comparing with the theoretically obtained firing rate
                    idx = self.find_nearest(database[ma][:, -1], self.r0)
                    (j02, eta, delta, n) = database[ma][idx, 1:]
                    self.fileprm2 = '%s_%.2lf-%.2lf-%.2lf-%d' % (self.fp, j02, eta, delta, self.l)
                    try:
                        self.spikes = np.load("%sic_qif_spikes_%s-%d.npy" % (self.filepath, self.fileprm2, n))
                        self.matrix = np.load("%sic_qif_matrix_%s-%d.npy" % (self.filepath, self.fileprm2, n))
                        print "Successfully loaded all data matrices."
                    except IOError:
                        print "Files do not exist or cannot be read. This behavior wasn't expected ..."
                        exit(-1)
                    except ValueError:
                        print "Not appropriate format of initial conditions. Check the files for logical errors..."
                        exit(-1)
                else:  # Create new initial conditions from scratch (loading random conditions)
                    print "Generating new initial conditions. " \
                          "Run the program using the same conditions after the process finishes."
                    # We set excitatory and inhibitory neurons at the same initial conditions:
                    self.matrix[:, 0] = -0.1 * np.random.randn(self.N)

    def save_ic(self, temps):
        """ Function to save initial conditions """
        print "Saving configuration for initial conditions ..."
        np.save("%sic_qif_spikes_%s-%d" % (self.filepath, self.fileprm, self.N), self.spikes)
        self.matrix[:, 1] = self.matrix[:, 1] - (temps - self.dt)
        np.save("%sic_qif_matrix_%s-%d.npy" % (self.filepath, self.fileprm, self.N), self.matrix)

        # Introduce this combination into the database
        try:
            db = np.load("%sinitial_conditions_%s.npy" % (self.filepath, self.fp))
        except IOError:
            print "Initial conditions database not found (%sinitial_conditions_%s.npy)" % (self.filepath, self.fp)
            print "Creating database ..."
            db = False
        if db is False:
            np.save("%sinitial_conditions_%s" % (self.filepath, self.fp),
                    np.array([self.l, self.j0, self.eta0, self.delta, self.N]))
        else:
            db.resize(np.array(np.shape(db)) + [1, 0], refcheck=False)
            db[-1] = np.array([self.l, self.j0, self.eta0, self.delta, self.N])
            np.save("%sinitial_conditions_%s" % (self.filepath, self.fp), db)

    def register_ts(self, fr=None, th=None):
        """ Function that stores time series of the firing rate, mean membrane potential, etc.
            into a dictionary.
            :type fr: FiringRate()
            :param th: TheoreticalComputations() in tools
        """
        if self.system == 'qif' or self.system == 'both':
            self.rstored['qif'] = np.array(fr.r)
            self.vstored['qif'] = np.array(fr.v)
            self.t['qif'] = fr.tempsfr
            self.k['qif'] = None
            self.dr['qif'] = dict(all=fr.frqif0, inst=fr.rqif)

        if self.system == 'nf' or self.system == 'both':
            self.rstored['nf'] = self.r
            self.vstored['nf'] = self.v
            self.t['nf'] = self.tpoints
            self.k['nf'] = None
            self.dr['nf'] = th.thdist

    @staticmethod
    def find_nearest(array, value):
        """ Extract the argument of the closest value from array """
        idx = (np.abs(array - value)).argmin()
        return idx


class Connectivity:
    """ Mex-Hat type connectivity function creator and methods
        to extract properties from it: modes, frequencies, linear response.
    """

    def __init__(self, length=500, profile='mex-hat', amplitude=1.0, me=50, mi=5, j0=0.0,
                 refmode=None, refamp=None, fsmodes=None, data=None, degree=None, saved=True):
        """ In order to extract properties some parameters are needed: they can be
            called separately.
        """

        print "Creating connectivity matrix (depending on the size of the matrix (%d x %d) " \
              "this can take a lot of RAM)" % (length, length)
        # Number of points (sample) of the function. It should be the number of populations in the ring.
        self.l = length
        # Connectivity function and spatial coordinates
        self.cnt = np.zeros((length, length))
        # [i_n, j_n] = 2.0 * np.pi * np.float64(np.meshgrid(xrange(length), xrange(length))) / length - np.pi
        # ij = np.abs(i_n - j_n)
        [i_n, j_n] = np.meshgrid(xrange(length), xrange(length))
        ij = (i_n - j_n) * (2.0 * np.pi / length)
        del i_n, j_n  # Make sure you delete these matrices here !!!
        self.profile = profile
        # Type of connectivity (profile=['mex-hat', 'General Fourier Series: fs'])
        if profile == 'mex-hat':
            if (refmode is not None) and (refamp is not None):  # A reference mode has been selected
                # Generate connectivity function here using reference amplitude
                (self.je, self.me, self.ji, self.mi) = self.searchmode(refmode, refamp, me, mi)
            else:
                # Generate connectivity function with parameters amplitude, me, mi, j0
                self.je = amplitude + j0
                self.ji = amplitude
                self.me = me
                self.mi = mi
            self.cnt = self.vonmises(self.je, me, self.ji, mi, coords=ij)
            # Compute eigenmodes
            self.eigenmodes = self.vonmises_modes(self.je, me, self.ji, mi)
            self.eigenvectors = None
        elif profile == 'fs':
            # Generate fourier series with modes fsmodes
            if fsmodes is None:
                fsmodes = 10.0 * np.array([0, 1, 0.75, -0.25])  # Default values
            self.cnt = self.jcntvty(fsmodes, coords=ij)
            # TODO: separate excitatory and inhibitory connectivity
            self.eigenmodes = fsmodes
            self.eigenvectors = None
        elif profile == 'uniform':
            aij = None
            if saved:
                try:
                    aij = np.load("cnt.npy")
                    np.reshape(aij, (length, length))
                except (IOError, ValueError) as e:
                    logger.error(e)
                    aij = self.uniform_in_degree(length, degree)
            self.cnt = data.j0 * aij
            logger.info("Connectivity matrix:\n%s" % str(self.cnt))
            # For Hermitian matrices self.modes = np.linalg.eigh(self.cnt)
            (self.eigenmodes, self.eigenvectors) = np.linalg.eigh(self.cnt)
        elif profile == 'pecora1':
            fsmodes = np.array(fsmodes) * data.j0
            jc = fsmodes[1]
            jr = fsmodes[0]
            self.cnt, (self.eigenmodes, self.eigenvectors) = self.pecora1998_ex1(length, jc, jr, eta=data.eta0,
                                                                                 delta=data.delta)
        del ij

        # Compute frequencies for the ring model (if data is provided)
        if data is not None and profile in ('mex-hat', 'fs'):
            self.freqs = self.frequencies(self.eigenmodes, data)
            print np.array(self.freqs) / data.faketau
        elif profile in 'pecora1':
            self.freqs = self.frequencies(fsmodes, data, type='pecora', n=length, alpha=data.j0)
            print self.freqs
            np.savetxt("freqs.txt", self.freqs)

    def searchmode(self, mode, amp, me, mi):
        """ Function that creates a Mex-Hat connectivity with a specific amplitude (amp) in a given mode (mode)
        :param mode: target mode
        :param amp: desired amplitude
        :param me: concentration parameter for the excitatory connectivity
        :param mi: concentration parameter for the inhibitory connectivity
        :return: connectivity parameters
        """
        tol = 1E-3
        max_it = 10000
        it = 0
        diff1 = 10
        step = amp * 1.0 - 1.0
        if step <= 0:
            step = 0.1
        # Test parameters:
        (ji, je, me, mi) = (1.0, 1.0, me, mi)

        while it < max_it:
            cnt = self.vonmises(je, me, ji, mi, self.l)
            jk = self.jmodesdct(cnt, mode + 10)
            diff = np.abs(jk[mode] - amp)
            if diff <= tol:
                break
            else:
                # print diff
                if diff1 - diff < 0:  # Bad direction
                    step *= -0.5

                je += step
                ji = je
                diff1 = diff * 1.0
            it += 1
        return je, me, ji, mi

    @staticmethod
    def frequencies(modes, data=None, eta=None, tau=None, delta=None, r0=None, type='ring-all', alpha=0.0, n=100):
        """ Function that computes frequencies of decaying oscillations at the homogeneous state
        :param modes: array of modes, ordered from 0 to maximum wavenumber. If only zeroth mode is passed,
                      then it should be passed as an array. E.g. [1.0]. (J_0 = 1.0).
        :param data: data object containing all relevant info
        :param eta: if data is not passed, then we need the external current,
        :param tau: also time constant,
        :param delta: and also heterogeneity parameter.
        :param r0: We can pass the value of the stationary firing rate, or we can just let the function
                   compute the theoretical value.
        :return: an array of frequencies with same size as modes array.
        """
        # If a data object (containing all info is given)
        if data is not None:
            eta = data.eta0
            tau = data.tau
            delta = data.delta
            j0 = data.j0
        # If not:
        elif (eta is None) or (tau is None) or (delta is None):
            print 'Not enough data to compute frequencies'
            return None
        if r0 is None:  # We have to compute the firing rate at the stationary state
            if type == 'pecora':
                r0 = Connectivity.rtheory(0, eta, delta)[0]
                logger.info("r0: %f" % r0)
            else:
                r0 = Connectivity.rtheory(modes[0], eta, delta)
        r0u = r0 / tau
        f = []
        if type == 'ring-all':
            for k, m in enumerate(modes):
                if m / (2 * np.pi ** 2 * tau * r0u) <= 1:
                    f.append(r0u * np.sqrt(1.0 - m / (2 * np.pi ** 2 * tau * r0u)))
                else:
                    f.append(r0u * np.sqrt(m / (2 * np.pi ** 2 * tau * r0u) - 1.0))
                    print "Fixed point is above the Saddle Node bifurcation for k = %d: there are not " \
                          "decaying oscillations for the homogeneous state." % k
                    print "These values plus the one corresponding to the decay are now the actual decays of overdamped " \
                          "oscillations."
        elif type == 'pecora':
            for k in xrange(n):
                f.append(r0 * np.sqrt(1.0 + 2 * alpha * (np.sin(np.pi * k / n)) ** 2 / (r0 * np.pi ** 2)))
        return f

    @staticmethod
    def gauss0_pdf(x, std):
        return stats.norm.pdf(x, 0, std)

    @staticmethod
    def mexhat0(a1, std1, a2, std2, length=500):
        x = np.linspace(-np.pi, np.pi, length)
        return x, a1 * Connectivity.gauss0_pdf(x, std1) + a2 * Connectivity.gauss0_pdf(x, std2)

    @staticmethod
    def vonmises(je, me, ji, mi, length=None, coords=None):
        if coords is None:
            if length is None:
                length = 500
            theta = (2.0 * np.pi / length) * np.arange(length)
        else:
            theta = 1.0 * coords
        return je / special.i0(me) * np.exp(me * np.cos(theta)) - ji / special.i0(mi) * np.exp(mi * np.cos(theta))

    @staticmethod
    def jcntvty(jk, coords=None):
        """ Fourier series generator.
        :param jk: array of eigenvalues. Odd ordered modes of Fourier series (only cos part)
        :param coords: matrix of coordinates
        :return: connectivity matrix J(|phi_i - phi_j|)
        """
        jphi = 0
        for i in xrange(len(jk)):
            if i == 0:
                jphi = jk[0]
            else:
                # Factor 2.0 is to be coherent with the computation of the mean-field S, where
                # we devide all connectivity profile by (2\pi) (which is the spatial normalization factor)
                jphi += 2.0 * jk[i] * np.cos(i * coords)
        return jphi

    @staticmethod
    def jmodes0(a1, std1, a2, std2, n=20):
        return 1.0 / (2.0 * np.pi) * (
            a1 * np.exp(-0.5 * (np.arange(n)) ** 2 * std1 ** 2) + a2 * np.exp(-0.5 * (np.arange(n)) ** 2 * std2 ** 2))

    @staticmethod
    def jmodesdct(jcnt, nmodes=20):
        """ Extract fourier first 20 odd modes from jcnt function.
        :param jcnt: periodic odd function.
        :param nmodes: number of modes to return
        :return: array of nmodes amplitudes corresponding to the FOurie modes
        """
        l = np.size(jcnt)
        jk = dct(jcnt, type=2, norm='ortho')
        for i in xrange(len(jk)):
            if i == 0:
                jk[i] *= np.sqrt(1.0 / (4.0 * l))
            else:
                jk[i] *= np.sqrt(1.0 / (2.0 * l))
        return jk

    @staticmethod
    def vonmises_modes(je, me, ji, mi, n=20):
        """ Computes Fourier modes of a given connectivity profile, built using
            Von Mises circular gaussian functions (see Marti, Rinzel, 2013)
        """
        modes = np.arange(n)
        return je * special.iv(modes, me) / special.i0(me) - ji * special.iv(modes, mi) / special.i0(mi)

    @staticmethod
    def rtheory(j0, eta0, delta):
        r0 = 1.0
        func = lambda tau: (np.pi ** 2 * tau ** 4 - j0 * tau ** 3 - eta0 * tau ** 2 - delta ** 2 / (4 * np.pi ** 2))
        sol = fsolve(func, r0)
        return sol

    def linresponse(self):
        # TODO Linear Response (may be is better to do this in the perturbation class)
        pass

    @staticmethod
    def uniform_in_degree(n, degree, symmetric=True, balanced=True, min=-5, max=5):
        """ Function to generate a random connectivity matrix (adjacency matrix, Aij)
        :param n: number of nodes
        :param degree: in-degree (d^n) of the nodes
        :param symmetric: symmetric connectivity (bidirectional), by default is True.
        :param balanced: balanced network, overall connectivity is 0 for every node. By default is True.
        :param min: minimum connectivity weight (can be negative)
        :param max: maximum connectivity weight.
        """
        aij = np.zeros((n, n))
        # Number of non-zero values
        d_n = int(n * degree) - 1

        # Vector containing the values of connectivity (ordered by amplitude)
        if balanced:
            a0 = np.linspace(-max, max, d_n)  # Non-zero values
        else:
            a0 = (max - min) * np.random.rand(d_n) + min
            # noinspection PyUnresolvedReferences
            logger.info('The overall input is: %f' % np.add.reduce(a0))

        a0 = np.concatenate((a0, np.zeros(n - d_n)))  # We complete using zeros
        np.random.shuffle(a0)  # Shuffle the vector
        # We look for a zero value and take the first argument
        zerosargs = np.ma.where(a0 == 0.0)
        argzero = zerosargs[0][0]
        a0 = np.roll(a0, -argzero)
        print  a0

        w = range(n)

        for i in xrange(n):
            if symmetric:
                aij[i] = np.roll(a0, (-1) ** (n) * (-i))
                if i % 2 != 0:
                    aij[i] = np.flipud(aij[i])
            else:
                aij[i] = np.roll(a0, w.pop(np.random.randint(len(w))))
        np.save("cnt", aij)
        return aij

    def pecora1998_ex1(self, n, jc, jr=0, eta=0.0, delta=1.0, tau=20E-3):
        """ Connectivity matrix of a ring (periodic boundaries) with only first neighbours connections (jc) and
            recurrent connectivity (jr).
            " ... --> 0 <-- Jc --> 0 <-- Jc --> 0 <-- Jc --> 0 <-- ...
                    /Jr \        /Jr \        /Jr \        /Jr \
                    `->-'        `->-'        `->-'        `->-'
            See Pecora, PRE, 58,1. 1998
        """
        aij = np.zeros((n, n))
        aij[0, -1] = jc
        aij[0, 0] = jr
        aij[0, 1] = jc

        for i in xrange(1, n):
            aij[i] = np.roll(aij[0], i)
        # Compute eigenmodes and eigenvalues
        r0 = self.rtheory(0.0, eta, delta)[0]
        logger.info("Firing rate at the fix point (r*): %f" % (r0 / tau))
        # r02 = 1.0 / np.sqrt(np.pi**2*2.0) * np.sqrt(eta + np.sqrt(eta**2 + 1.0))
        v0 = -1.0 / (2 * np.pi * r0)
        J = np.array([[2 * v0, 2 * r0], [-2.0 * np.pi ** 2 * r0, 2 * v0]])
        E = np.array([[0, 0], [jc, 0]])
        eigenvalues = []
        eigenvectors = []
        # gammak = np.linalg.eigvals(aij) / jc
        for k in xrange(n):
            gammak = -4.0 * (np.sin(np.pi * k / n)) ** 2
            A = J + E * gammak
            eigen = np.linalg.eig(A)
            # logger.info(
            #     "For %d mode:\n\t Real part of Eingenvalue 0: %f\tEigenvector 0: %s\n\t "
            #     "Real part of Eingenvalue 1: %f\tEigenvector 1: %s" % (
            #         k, np.imag(eigen[0][0]) / (2.0 * np.pi * tau), str(eigen[1][0]),
            #         np.imag(eigen[0][1]) / (2.0 * np.pi * tau), str(eigen[1][1])))
            raw_input("Press ENTER to continue...")
            for lmbd, vect in zip(eigen[0], eigen[1]):
                if not np.isreal(vect[0]):
                    eigenvalues.append(lmbd)
                    # logger.info("Decay and Frequency of the %d mode: %f, %f" % (
                    #     k, (np.real(lmbd) * tau), (np.imag(lmbd) / (2.0 * np.pi) / tau)))
            j = np.exp(2.0 * np.pi * 1.0j * np.arange(0, n) * k / n)
            # noinspection PyUnresolvedReferences
            eigenvectors.append(np.real((1.0 / n) * np.add.reduce(np.diag(np.ones(n)) * j, axis=0)))

        return aij, (eigenvalues, eigenvectors)


class FiringRate:
    """ Class related to the measure of the firing rate of a neural network.
    """

    def __init__(self, data=None, swindow=1.0, sampling=0.01, points=None):
        # type: (Data(), float, float, int) -> object

        if data is None:
            self.d = Data()
        else:
            self.d = data

        self.swindow = swindow  # Time Window in which we measure the firing rate
        self.wsteps = np.ceil(self.swindow / self.d.dt)  # Time window in time-steps
        self.wones = np.ones(self.wsteps)

        # Frequency of measuremnts
        if points is not None:
            pp = points
            if pp > self.d.nsteps:
                pp = self.d.nsteps
            self.sampling = self.d.nsteps / pp
            self.samplingtime = self.sampling * self.d.dt
        else:
            self.samplingtime = sampling
            self.sampling = int(self.samplingtime / self.d.dt)

        # Firing rate of single neurons (distibution of firing rates)
        self.sampqift = 1.0 * self.swindow
        self.sampqif = int(self.sampqift / self.d.dt)

        self.tpoints_r = np.arange(0, self.d.tfinal, self.samplingtime)

        freemem = psutil.virtual_memory().available
        needmem = 8 * (self.wsteps + self.d.l) * data.N
        print "Approximately %d MB of memory will be allocated for FR measurement." % (needmem / (1024 ** 2))
        if (freemem - needmem) / (1024 ** 2) <= 0:
            print "MEMORY ERROR: not enough amount of memory available."
            exit(-1)
        elif (freemem - needmem) / (1024 ** 2) < 100:
            print "CRITICAL WARNING: very few amount of memory will be left."
            try:
                raw_input("Continue? (any key to continue, CTRL+D to terminate).")
            except EOFError:
                print "Terminating process."
                exit(-1)

        self.frspikes = 0 * np.zeros(shape=(data.N, self.wsteps))  # Secondary spikes matrix (for measuring)
        self.r = []  # Firing rate of the newtork(ring)
        self.rqif = []
        self.v = []  # Firing rate of the newtork(ring)
        self.vavg0 = 0.0 * np.ones(data.N)
        self.frqif0 = []  # Firing rate of individual qif neurons
        self.frqif = None

        # Total spikes of the network:
        self.tspikes = 0 * np.ones(data.N)
        self.tspikes2 = 0 * np.ones(data.N)

        # Theoretical distribution of firing rates
        self.thdist = dict()

        # Auxiliary counters
        self.ravg = 0
        self.ravg2 = 0
        self.vavg = 0

        # Times of firing rate measures
        self.t0step = None
        self.tfstep = None
        self.temps = None

        self.tfrstep = -1
        self.tfr = []
        self.tempsfr = []
        self.tempsfr2 = []

    def firingrate(self, tstep):
        """ Computes the firing rate for a given matrix of spikes. Firing rate is computed
            every certain time (sampling). Therefore at some time steps the firing rate is not computed,
        :param tstep: time step of the simulation
        :return: firing rate vector (matrix)
        """
        if (tstep + 1) % self.sampling == 0 and (tstep * self.d.dt >= self.swindow):
            self.tfrstep += 1
            self.temps = tstep * self.d.dt
            re = (1.0 / self.swindow) * (1.0 / self.d.dN) * np.dot(self.d.auxMat,
                                                                   np.dot(self.frspikes, self.wones))

            self.r.append(re)
            self.tempsfr.append(self.temps - self.swindow / 2.0)
            self.tempsfr2.append(self.temps)

            # Single neurons firing rate in a given time (averaging over a time window)
            self.rqif.append(self.tspikes2)
            self.rqif[-1] /= (self.ravg2 * self.d.dt * self.d.faketau)
            # We store total spikes for the "total time average" distribution of FR
            self.tspikes += self.tspikes2

            # Reset vectors and counter
            self.tspikes2 = 0.0 * np.ones(self.d.N)
            self.ravg2 = 0

            # Average of the voltages over a time window and over the populations
            self.v.append((1.0 / self.d.dN) * np.dot(self.d.auxMat, self.vavg0 / self.vavg))
            self.vavg = 0
            self.vavg0 = 0.0 * np.ones(self.d.N)

    def singlefiringrate(self, tstep):
        """ Computes the firing rate of individual neurons.
        :return: Nothing, results are stored at frqif0 and frqif_i
        """
        if (tstep + 1) % self.sampqif == 0 and (tstep * self.d.dt >= self.swindow):
            # Firing rate measure in a time window
            re = (1.0 / self.d.dt) * self.frspikes.mean(axis=1)
            self.frqif0.append(re)

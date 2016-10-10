import datetime
import os

import numba
import numpy as np

from nflib import Data, Connectivity

__author__ = 'Jose M. Esnaola Acebes'

""" Library containing different tools to use alongside the main simulation:

    + Perturbation class: creates a perturbation profile and handles perturbation timing.
    + Saving Class: contains a method to create a dictionary with the data and a method to save it.
    + Theoretical computation class: some computations based on theory.
    + A class that transforms a python dictionary into a python object (for easier usage).

"""


# Function that performs the integration (prepared for numba)
@numba.autojit
def qifint(v_exit_s1, v, exit0, eta_0, s_0, tiempo, number, dn, dt, tau, vpeak, refr_tau, tau_peak):
    """ This function checks (for each neuron) whether the neuron is in the
    refractory period, and computes the integration in case is NOT. If it is,
    then it adds a time step until the refractory period finishes.

    The spike is computed when the neuron in the refractory period, i.e.
    a neuron that has already crossed the threshold, reaches the midpoint
    in the refractory period, t_peak.
    :rtype : object
    """

    d = 1 * v_exit_s1
    # These steps are necessary in order to use Numba
    t = tiempo * 1.0
    for n in xrange(number):
        d[n, 2] = 0
        if t >= exit0[n]:
            d[n, 0] = v[n] + (dt / tau) * (v[n] * v[n] + eta_0[n] + tau * s_0[int(n / dn)])  # Euler integration
            if d[n, 0] >= vpeak:
                d[n, 1] = t + refr_tau - (tau_peak - 1.0 / d[n, 0])
                d[n, 2] = 1
                d[n, 0] = -d[n, 0]
    return d


@numba.autojit
def qifint_noise(v_exit_s1, v, exit0, eta_0, s_0, nois, tiempo, number, dn, dt, tau, vpeak, refr_tau, tau_peak):
    d = 1 * v_exit_s1
    # These steps are necessary in order to use Numba (don't ask why ...)
    t = tiempo * 1.0
    for n in xrange(number):
        d[n, 2] = 0
        if t >= exit0[n]:
            d[n, 0] = v[n] + (dt / tau) * (v[n] * v[n] + eta_0 + tau * s_0[int(n / dn)]) + nois[n]  # Euler integration
            if d[n, 0] >= vpeak:
                d[n, 1] = t + refr_tau - (tau_peak - 1.0 / d[n, 0])
                d[n, 2] = 1
                d[n, 0] = -d[n, 0]
    return d


def noise(length=100, disttype='g'):
    if disttype == 'g':
        return np.random.randn(length)


def find_nearest(array, value, ret='id'):
    """ find the nearest value or/and id of an "array" with respect to "value"
    """
    idx = (np.abs(array - value)).argmin()
    if ret == 'id':
        return idx
    elif ret == 'value':
        return array[idx]
    elif ret == 'both':
        return array[idx], idx
    else:
        print "Error in find_nearest."
        return -1


class Perturbation:
    """ Tool to handle perturbations: time, duration, shape (attack, decay, sustain, release (ADSR), etc. """

    def __init__(self, data=None, t0=2.5, dt=0.5, ptype='pulse', modes=None,
                 amplitude=1.0, attack='exponential', release='instantaneous'):
        if data is None:
            self.d = Data()
        else:
            self.d = data

        if modes is None:  # Default mode perturbation is first mode
            modes = [1]

        # Input at t0
        self.input = np.ones(self.d.l) * 0.0
        # Input time series
        self.it = np.zeros((self.d.nsteps, self.d.l))
        # Input ON/OFF
        self.pbool = False

        # Time parameters
        self.t0 = t0
        self.dt = dt
        self.tf = t0 + dt
        # Rise variables (attack) and parameters
        self.attack = attack
        self.taur = 0.2
        self.trmod = 0.01
        self.trmod0 = 1.0 * self.trmod
        # Decay (release) and parameters
        self.release = release
        self.taud = 0.2
        self.tdmod = 1.0
        self.mintd = 0.0

        # Amplitude parameters
        self.ptype = ptype
        self.amp = amplitude
        # Spatial modulation (wavelengths)
        self.phi = np.linspace(-np.pi, np.pi, self.d.l)
        self.smod = self.sptprofile(modes, self.amp)

    def sptprofile(self, modes, amp=1E-2):
        """ Gives the spatial profile of the perturbation: different wavelength and combinations
            of them can be produced.
        """
        sprofile = 0.0
        if np.isscalar(modes):
            print "Warning! 'modes' should be an iterable."
            modes = [modes]
        for m in modes:
            sprofile += amp * np.cos(m * self.phi)
        return sprofile

    def timeevo(self, temps):
        """ Time evolution of the perturbation """
        # Single pulse
        if self.ptype == 'pulse':
            # Release time, after tf
            if temps >= self.tf:
                if self.release == 'exponential' and self.tdmod > self.mintd:
                    self.tdmod -= (self.d.dt / self.taud) * self.tdmod
                    self.input = self.tdmod * self.smod
                elif self.release == 'instantaneous':
                    self.input = 0.0
                    self.pbool = False
            else:  # During the pulse (from t0 until tf)
                if self.attack == 'exponential' and self.trmod < 1.0:
                    self.trmod += (self.d.dt / self.taur) * self.trmod
                    self.tdmod = self.trmod
                    self.input = (self.trmod - self.trmod0) * self.smod
                elif self.attack == 'instantaneous':
                    if temps == self.t0:
                        self.input = self.amp
        elif self.ptype == 'oscillatory':
            pass


class SaveResults:
    """ Save Firing rate data to be plotted or to be loaded with numpy.
    """

    def __init__(self, data=None, cnt=None, pert=None, path='results', system='nf', parameters=None):
        if data is None:
            self.d = Data()
        else:
            self.d = data
        if cnt is None:
            self.cnt = Connectivity()
        else:
            self.cnt = cnt
        if pert is None:
            self.p = Perturbation()
        else:
            self.p = pert

        # Path of results (check path or create)
        if os.path.isdir("./%s" % path):
            self.path = "./results"
        else:  # Create the path
            os.path.os.mkdir("./%s" % path)
        # Define file paths depending on the system (nf, qif, both)
        self.fn = SaveResults.FileName(self.d, system)
        self.results = dict(parameters=dict(), connectivity=dict)
        # Parameters are store copying the configuration dictionary and other useful parameters (from the beginning)
        self.results['parameters'] = {'l': self.d.l, 'eta0': self.d.eta0, 'delta': self.d.delta, 'j0': self.d.j0,
                                      'tau': self.d.faketau, 'args': parameters}
        self.results['connectivity'] = {'type': cnt.profile, 'cnt': cnt.cnt, 'modes': cnt.modes, 'freqs': cnt.freqs}
        self.results['perturbation'] = {'t0': pert.t0}
        if cnt.profile == 'mex-hat':
            self.results['connectivity']['je'] = cnt.je
            self.results['connectivity']['ji'] = cnt.ji
            self.results['connectivity']['me'] = cnt.me
            self.results['connectivity']['mi'] = cnt.mi

        if system == 'qif' or system == 'both':
            self.results['qif'] = dict(fr=dict(), v=dict())
            self.results['parameters']['qif'] = {'N': self.d.N}
        if system == 'nf' or system == 'both':
            self.results['nf'] = dict(fr=dict(), v=dict())

    def create_dict(self, **kwargs):
        tol = self.d.total_time * (1.0 / 100.0)

        for system in self.d.systems:
            self.results[system]['t'] = self.d.t[system]
            self.results[system]['fr'] = dict(ring=self.d.rstored[system])
            self.results[system]['vstored'] = dict(ring=self.d.vstored[system])
            self.results[system]['fr']['distribution'] = self.d.dr[system]
            t = []
            if 'phi0' in kwargs:
                self.results[system]['fr']['ts'] = {'p%s' % str(phi0): self.d.rstored[system][:, phi0] for phi0 in
                                                    list(dict(kwargs)['phi0'])}
                self.results[system]['vstored']['ts'] = {'p%s' % str(phi0): self.d.vstored[system][:, phi0] for phi0 in
                                                   list(dict(kwargs)['phi0'])}
            if 't0' in kwargs:
                for t0 in list(dict(kwargs)['t0']):
                    t0p, t0pid = find_nearest(self.d.t[system], t0, ret='both')
                    if np.abs(t0p - t0) > tol:
                        # Emergency save
                        print "ERROR: time %lf not in data array." % t0
                        print "Closest time was %lf. Difference is %lf and tolerance %lf." % (
                            t0p, np.abs(t0p - t0), tol)
                        self.save()
                        exit(-1)
                    else:
                        t.append(t0pid)

                self.results[system]['fr']['profiles'] = {'t%s' % str(ti): self.d.rstored[system][ti] for ti in t}
                self.results[system]['vstored']['profiles'] = {'t%s' % str(ti): self.d.vstored[system][ti] for ti in t}
                del t

    def save(self):
        """ Saves all relevant data into a numpy object with date as file-name."""
        now = datetime.datetime.now().timetuple()[0:6]
        sday = "-".join(map(str, now[0:3]))
        shour = "_".join(map(str, now[3:]))
        np.save("%s/data_%s-%s" % (self.path, sday, shour), self.results)

    def time_series(self, ydata, filename, export=False, xdata=None):
        if export is False:
            np.save("%s/%s_y" % (self.path, filename), ydata)
            if xdata is not None:
                np.save("%s/%s_x" % (self.path, filename), xdata)
        else:  # We save it as a csv file
            np.savetxt("%s/%s.dat" % (self.path, filename), np.c_[xdata, ydata])

    def profiles(self):
        pass

    class FileName:
        """ This class just creates strings to be easily used and understood
            (May be is too much...)
        """

        def __init__(self, data, system):
            self.d = data
            if system == 'qif' or system == 'both':
                self.qif = self.Variables(data, 'qif')
            if system == 'nf' or system == 'both':
                self.nf = self.Variables(data, 'nf')

        @staticmethod
        def tpoints(d, system):
            return "%s_time-colorplot_%.2lf-%.2lf-%.2lf-%d" % (system, d.j0, d.eta0, d.delta, d.l)

        class Variables:
            def __init__(self, data, t):
                self.fr = SaveResults.FileName.FiringRate(data, t)
                self.v = SaveResults.FileName.MeanPotential(data, t)
                self.t = SaveResults.FileName.tpoints(data, t)

        class FiringRate:
            def __init__(self, data, system):
                self.d = data
                self.t = system
                self.colorplot()

            def colorplot(self):
                # Color plot (j0, eta0, delta, l)
                self.cp = "%s_fr-colorplot_%.2lf-%.2lf-%.2lf-%d" % (
                    self.t, self.d.j0, self.d.eta0, self.d.delta, self.d.l)

            def singlets(self, pop):
                # Single populations
                self.sp = "%s_fr-singlets-%d_%.2lf-%.2lf-%.2lf-%d" % (
                    self.t, pop, self.d.j0, self.d.eta0, self.d.delta, self.d.l)

            def profile(self, t0):
                # Profile at a given t0
                self.pr = "%s_fr-profile-%.2lf_%.2lf-%.2lf-%.2lf-%d" % (
                    self.t, t0, self.d.j0, self.d.eta0, self.d.delta, self.d.l)

        class MeanPotential:
            def __init__(self, data, system):
                self.d = data
                self.t = system
                self.colorplot()

            def colorplot(self):
                # Color plot (j0, eta0, delta, l)
                self.cp = "vstored-colorplot_%.2lf-%.2lf-%.2lf-%d" % (self.d.j0, self.d.eta0, self.d.delta, self.d.l)

            def singlets(self, pop):
                # Single populations
                self.sp = "vstored-singlets-%d_%.2lf-%.2lf-%.2lf-%d" % (pop, self.d.j0, self.d.eta0, self.d.delta, self.d.l)

            def profile(self, t0):
                # Profile at a given t0
                self.pr = "vstored-profile-%.2lf_%.2lf-%.2lf-%.2lf-%d" % (t0, self.d.j0, self.d.eta0, self.d.delta, self.d.l)


class TheoreticalComputations:
    def __init__(self, data=None, cnt=None, pert=None):
        if data is None:
            self.d = Data()
        else:
            self.d = data
        if cnt is None:
            self.cnt = Connectivity()
        else:
            self.cnt = cnt
        if pert is None:
            self.p = Perturbation()
        else:
            self.p = pert

        # Theoretical distribution of firing rates
        self.thdist = dict()

    def theor_distrb(self, s, points=10E3, rmax=3.0):
        """ Computes theoretical distribution of firing rates
        :param s: Mean field (J*rstored) rescaled (tau = 1)
        :param points: number of points for the plot
        :param rmax: maximum firing rate of the distribution
        :return: rho(rstored) vs rstored (output -> rstored, rho(rstored))
        """
        rpoints = int(points)
        rmax = rmax / self.d.faketau
        r = np.dot(np.linspace(0, rmax, rpoints).reshape(rpoints, 1), np.ones((1, self.d.l)))
        s = np.dot(np.ones((rpoints, 1)), s.reshape(1, self.d.l))
        geta = self.d.delta / ((self.d.eta0 - (np.pi ** 2 * self.d.faketau ** 2 * r ** 2 - s)) ** 2 + self.d.delta ** 2)
        rhor = 2.0 * np.pi * self.d.faketau ** 2 * geta.mean(axis=1) * r.T[0]
        # plt.plot(x.T[0], hr, 'rstored')
        return dict(x=r.T[0], y=rhor)


class DictToObj(object):
    """Class that transforms a dictionary d into an object. Note that dictionary keys must be
       strings and cannot be just numbers (even if they are strings:
               if a key is of the format '4', -> put some letter at the beginning,
               something of the style 'f4'.
       Class obtained from stackoverflow: user Nadia Alramli.
 `  """

    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [DictToObj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, DictToObj(b) if isinstance(b, dict) else b)

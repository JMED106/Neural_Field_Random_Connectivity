import getopt
import sys

import matplotlib.pyplot as plt
import numpy as np

from tools import DictToObj

__author__ = 'Jose M. Esnaola Acebes'

""" Load this file to easily handle data saved in main. It loads the numpy object saved
    in ./results and converts it to a Dict. The latter is converted to an object to
    be able to use the dot notation, instead of the brackets typical of dictionaries.
    To use it in python: run load_data.py -f <name_of_file>
"""

pi = np.pi
cplot = plt.pcolormesh


def __init__(argv):
    try:
        opts, args = getopt.getopt(argv, "hf:", ["file="])
    except getopt.GetoptError:
        print 'load_data.py [-f <file>]'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'load_data.py [-f <file>]'
            sys.exit()
        elif opt in ("-f", "--file"):
            filein = arg
            return filein
        else:
            print 'load_data.py [-f <file>]'
            sys.exit()


fin = __init__(sys.argv[1:])
d = np.load(fin)
data = DictToObj(d.item())
phi = np.linspace(-pi, pi, data.parameters.l)
phip = np.linspace(-pi, pi, data.parameters.l + 1)

# We store the time series in a different file (for easier management)
# Comment lines to disable :b
ts_r = np.array(data.qif.fr.ring) / data.parameters.tau
ts_t = np.array(data.qif.t) * data.parameters.tau
np.save("ts_r.npy", ts_r)
np.save("ts_t.npy", ts_t)

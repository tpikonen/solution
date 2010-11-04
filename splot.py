import sys
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy
import numpy as np
import scipy.interpolate as ip
from optparse import OptionParser
from atsasformats import read_dat
from yamlformats import read_ydat
from sxsplots import plot_iq

description="Plot DAT files."

usage="%prog <file1.dat> <file2.dat> ..."


def main():
    oprs = OptionParser(usage=usage, description=description)
    oprs.add_option("-n", "--no-errors",
        action="store_false", dest="err", default=True,
        help="Do not plot errorbands")
    oprs.add_option("-r", "--raw-errors",
        action="store_false", dest="smerr", default=True,
        help="Do not smooth errorbands")

    (opts, args) = oprs.parse_args()
    if(len(args) < 1):
        oprs.error("Input file argument required")

    dats = []
    for fname in args:
        if fname.endswith(('.yaml', '.ydat', '.yfil')):
            dat = read_ydat(fname).T
        else:
            dat = read_dat(fname).T
        dats.append((dat, fname))

    fig = plt.figure()
    plt.rcParams['legend.fontsize'] = 12.0
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
    for dat in dats:
        plot_iq(ax, dat[0], dat[1], opts.err, opts.smerr)
    fig.set_size_inches(15,10)

    plt.show()

if __name__ == "__main__":
    main()

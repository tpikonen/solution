import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from xformats.atsasformats import read_gnom
from optparse import OptionParser
from sxsplots import plot_iq, plot_pr

description="Plot GNOM output files."

usage="%prog <gnomfile1.out> <gnomfile2.out> ..."


def main():
    oprs = OptionParser(usage=usage, description=description)
    oprs.add_option("-n", "--no-errors",
        action="store_false", dest="err", default=True,
        help="Do not plot errorbands")
    oprs.add_option("-r", "--raw-errors",
        action="store_false", dest="smerr", default=True,
        help="Do not smooth errorbands")
    oprs.add_option("-e", "--no-experimental",
        action="store_false", dest="expdata", default=True,
        help="Do not plot the experimental data")

    (opts, args) = oprs.parse_args()
    if(len(args) < 1):
        oprs.error("Input file argument required")

    gnoms = []
    for fname in args:
        gnoms.append(read_gnom(fname))

    fig = plt.figure()
    plt.rcParams['legend.fontsize'] = 8.0
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_position([0.08, 0.1, 0.4, 0.8])
    for i in range(0, len(gnoms)):
        g = gnoms[i]
        lstr = g['filename'].rpartition('/')[2] + ", Rg=%g+-%g, I(0)=%g+-%g" % (g['Rg_real'], g['Rg_err_real'], g['I0_real'], g['I0_err_real'])
        plot_iq(ax1, g['Ireg'], label=lstr)
        if opts.expdata:
            plot_iq(ax1, g['Iexp'], err=opts.err, smerr=opts.smerr)
    ax1.set_xlabel("q / (1/nm)")
    ax1.set_ylabel("I(q)")
    ax1.legend()
    ax2 = fig.add_subplot(1,2,2)
    ax2.set_position([0.57, 0.1, 0.4, 0.8])
    for i in range(0, len(gnoms)):
        g = gnoms[i]
        plot_pr(ax2, g)
    ax2.legend()
    ax2.set_xlabel("r / nm")
    ax2.set_ylabel("P(r)")
    fig.set_size_inches(15,5)
    plt.show()


if __name__ == "__main__":
    main()

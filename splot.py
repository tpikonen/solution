import sys
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy
import numpy as np
import scipy.interpolate as ip
from optparse import OptionParser
from xformats.atsasformats import read_dat
from xformats.yamlformats import read_loopyaml
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

    Iunit = 'unknown'
    qunit = 'unknown'
    tups = []
    for fname in args:
        if fname.endswith(('.yaml', '.ydat', '.yfil')):
            yd = read_loopyaml(fname)
            if yd.has_key('Ierr'):
                dat = np.zeros((len(yd['Ierr']), 3))
                dat[:,2] = np.array(yd['Ierr'])
            else:
                dat = np.zeros((2,len(yd['q'])))
            dat[:,0] = np.array(yd['q'])
            dat[:,1] = np.array(yd['I'])
            Iunit = yd.get('I~unit', Iunit)
            qunit = yd.get('q~unit', qunit)
        else:
            dat = read_dat(fname).T
        tups.append((dat, fname))

    fig = plt.figure()
    plt.rcParams['legend.fontsize'] = 12.0
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
    for tup in tups:
        plot_iq(ax, tup[0], opts.err, opts.smerr, label=tup[1])
    fig.set_size_inches(15,10)

    plt.xlabel("q / %s" % qunit)
    plt.ylabel("I / %s" % Iunit)
    plt.show()

if __name__ == "__main__":
    main()

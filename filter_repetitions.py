import sys, os.path
import numpy as np
import matplotlib.pyplot as plt
import optparse
from xformats.yamlformats import read_yaml, write_yaml, read_ydat, write_ydat
from xformats.matformats import read_mat
from sxsplots import plot_iq
from utils import stack_datafiles, chivectors, mean_stack, md5_file

description="""\
Filter repetitions by comparing to the first one.
"""

usage="%prog [ scans.yaml | -m stack.mat [ stack2.mat ... ] ]"


def incmap_to_strings(incmap):
    """Return a list of strings representing the bool 2D array `incmap`.
    """
    ish = incmap.shape
    slist = []
    for rep in range(ish[0]):
        slist.append(''.join([ 'X' if incmap[rep,q] else '.'
            for q in range(ish[1]) ]))
    return slist


def strings_to_incmap(slist):
    """Return a bool 2D array constructed from list of strings.
    """
    inds = np.zeros((len(slist), len(slist[0])), dtype=np.bool)
    for i in range(len(slist)):
        inds[i,:] = np.array([ s == 'X' for s in slist[i] ])
    return inds


def write_filtered(filtered, first, aver, incmap, fname, inputfile=None, pos=-1):
    """Write an 'ydat' YAML file `fname` with filtered data and index array.

    `filtered` contains the filtered data, `incmap` the point by point inclusion
    array (bool matrix) of points used in the averaging and `first` the
    data set used in comparison for the filtering.
    """
    # FIXME: take a list of filenames which are filtered as an argument
    #        and write them to the file
    with open(fname, "w") as fp:
        indent = '  '
        fp.write('incmap: !!seq [\n' + indent)
        slist = incmap_to_strings(incmap.T)
        perrow = 1 + (80 / len(slist[0]))
        i = 0
        while i < perrow * ((len(slist)-1)/perrow): # until last row
            fp.write(slist[i])
            if (i+1) % perrow or not i:
                fp.write(', ')
            else:
                fp.write(',\n' + indent)
            i += 1
        while i < len(slist): # last row
            fp.write(slist[i])
            if i < len(slist)-1:
                fp.write(', ')
            else:
                fp.write(']\n')
            i += 1
        ad = {
            'method' : "filter_repetitions",
            'q~unit' : '1/nm',
            'I~unit' : 'arb.',
            'Ierr~unit' : 'arb.',
            'I_first~unit' : 'arb.',
            'Ierr_first~unit' : 'arb.',
            'I_all~unit' : 'arb.',
            'Ierr_all~unit' : 'arb.',
            }
        if inputfile:
            ad['inputfile'] = [ inputfile, md5_file(inputfile) ]
        if pos >= 0:
            ad['inputposition'] = int(pos)
        outarr = np.zeros((7, filtered.shape[1]))
        outarr[0:3,:] = filtered
        outarr[3:5,:] = first[1:3,:]
        outarr[5:7,:] = aver[1:3,:]
        cols = ['q', 'I', 'Ierr', 'I_first', 'Ierr_first', 'I_all', 'Ierr_all']
        write_ydat(outarr, fp, cols=cols, addict=ad, attributes=['~unit'])


def read_filtered(fname):
    """Return dat-array, inclusion map and first data read from file `fname`.
    """
    dat, yd = read_ydat(fname, addict=1)
    first = None
    aver = None
    if dat.shape[0] >= 5:
        first = np.zeros((3, dat.shape[1]))
        first[0,:] = dat[0,:]
        first[1:3,:] = dat[3:5,:]
    if dat.shape[0] >= 7:
        aver = np.zeros((3, dat.shape[1]))
        aver[0,:] = dat[0,:]
        aver[1:3,:] = dat[5:7,:]
    incmap = strings_to_incmap(yd['incmap']).T
    return (dat[0:3,:], first, aver, incmap)


def plot_filtered(filt, first, aver, incmap, figno=666, err=1, smerr=1):
    plt.figure(figno)
    plt.clf()
    plt.axes([0.1, 0.3, 0.8, 0.65])
    plt.hold(1)
    ax = plt.gca()
    plot_iq(ax, first.T, smerr=smerr, label="First rep")
    plot_iq(ax, aver.T, smerr=smerr, label="All reps")
    plot_iq(ax, filt.T, smerr=smerr, label="Filtered")
    plt.legend()

    plt.axis('tight')
    plt.axes([0.1, 0.05, 0.8, 0.15])
    plt.imshow(incmap, interpolation='nearest', aspect='auto')
    plt.plot(np.sum(incmap, axis=0)-0.5, 'y')
    plt.axis('tight')
    av = plt.axis()
    plt.axis([av[0], av[1], av[3], av[2]])
    plt.hold(0)
    plt.show()


def chifilter_points(reps, chi2cutoff=1.1, winhw=25, plot=0):
    """Return an average of repetitions statistically similar to the first.

    Array of repetitions `reps` has the shape (nreps, q/I/Ierr, len(q))
    and contains nreps curves with the q-scale and errors.
    The q-scales must be identical in all repetitions.

    Repetitions are compared to the first one point by point. The chi**2
    between first measurement and the repetition is calculated on an interval
    centered on the compared point with half-width `winhw`. Points which
    have chi**2 > `chi2cutoff` in are discarded from the averaging.
    """
    nreps = reps.shape[0]
    qlen = reps.shape[2]
    incmap = np.zeros((nreps, qlen), dtype=np.bool)
    incmap[0,:] = True
    def chi2wfilt(x, y, pos, winhw=winhw):
        ind = slice(max(0, pos-winhw), min(qlen, pos+winhw+1))
        chi2 = chivectors(x[:,ind], y[:,ind])
        return chi2 < chi2cutoff
    first = reps[0,...]
    for rep in range(1,nreps):
        for qind in range(qlen):
            incmap[rep,qind] = chi2wfilt(first, reps[rep,...], qind)

    filt = np.zeros((3, qlen))
    filt[0,:] = first[0,:]
    def sumsq(x): return np.sum(np.square(x))
    for qind in range(qlen):
        filt[1,qind] = np.mean(reps[incmap[:,qind], 1, qind])
        N = np.sum(incmap[:,qind])
        prop = np.sqrt(sumsq(reps[incmap[:,qind], 2, qind])) / N
#        sdev = np.std(reps[incmap[:,qind], 2, qind]) / np.sqrt(N)
#        filt[2,qind] = max(prop, sdev)
        filt[2,qind] = prop

    if plot:
        aver = mean_stack(reps)
        plot_filtered(filt, first, aver, incmap, figno=plot)

    return (filt, incmap)



def filter_matfile(fname, outstem):
    stack = read_mat(fname)
    for pos in range(stack.shape[0]):
        print("File: %s, pos %d" % (fname, pos))
        sys.stdout.flush()
        first = stack[pos,0,...]
        aver = mean_stack(stack[pos,...])
        filt, inds = chifilter_points(stack[pos,...])
        outname = "%s.p%02d.fil.ydat" % (outstem, pos)
        write_filtered(filt, first, aver, inds, outname, \
            os.path.basename(fname), pos)
        print(outname)


def main():
    oprs = optparse.OptionParser(usage=usage, description=description)
    oprs.add_option("-m", "--matfiles",
        action="store_true", dest="matfiles", default=False,
        help="Input arguments are matfiles instead of a YAML file.")
    (opts, args) = oprs.parse_args()
    if len(args) < 1:
        oprs.error("One or more input files needed.")

    if opts.matfiles:
        filenames = args
        for fname in filenames:
            outstem = "%s" % fname[:(fname.find('.mat'))]
            filter_matfile(fname, outstem)
    else:
        if len(args) > 1:
            oprs.error("Only one YAML scan list accepted.")
        scans = read_yaml(args[0])
        scannos = scans.keys()
        scannos.sort()
        for scanno in scannos:
            try:
                bufscan = scans[scanno][0]
            except TypeError:
                print("Scan #%03d is a buffer" % scanno)
                continue
            print("Scan #%03d" % scanno)
            fname = "s%d.mat" % scanno
            outstem = "%s" % fname[:(fname.find('.mat'))]
            filter_matfile(fname, outstem)

if __name__ == "__main__":
    main()

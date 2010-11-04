import numpy as np
import matplotlib.pyplot as plt
from yamlformats import read_yaml, write_yaml, read_ydat, write_ydat
from scipy.io.matlab.mio import loadmat
from sxsplots import plot_iq


def clean_indices(x, y):
    """Return indices which contain only good floats in x and y.
    """
    nn = np.logical_not(np.logical_or(
        np.logical_or(np.isnan(x[1,:]), np.isnan(y[1,:])),
        np.logical_or((x[2,:] <= 0.0), (y[2,:] <= 0.0))))
    return nn


def chivectors(x, y):
    nn = clean_indices(x, y)
    N = np.sum(nn)
    chi2 = (1.0/N)*np.sum((x[1,nn]-y[1,nn])**2 / (x[2,nn]**2+y[2,nn]**2))
    return chi2


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


def write_filtered(avg, incmap, fname, first=None):
    """Write an 'ydat' YAML file `fname` with filtered data and index array.

    `avg` contains the filtered data, `incmap` the point by point inclusion
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
        addict = {}
        if first is None:
            write_ydat(avgind[0], fp, addict=addict)
        else:
            dat = np.zeros((5,avg.shape[1]))
            dat[0:3,:] = avg
            dat[3:5,:] = first[1:3,:]
            cols = ['q', 'I', 'Ierr', 'I_first', 'Ierr_first']
            write_ydat(dat, fp, cols=cols, addict=addict)


def read_filtered(fname):
    """Return dat-array, inclusion map and first data read from file `fname`.
    """
    dat = read_ydat(fname)
    first = None
    if dat.shape[0] == 5:
        first = np.zeros((3, dat.shape[1]))
        first[0,:] = dat[0,:]
        first[1:3,:] = dat[3:5,:]
    yd = read_yaml(fname)
    incmap = strings_to_incmap(yd['incmap']).T
    return (dat[0:3,:], incmap, first)


def plot_filtered(avg, incmap, first=None, figno=666, err=1, smerr=1):
    plt.figure(figno)
    plt.clf()
    plt.axes([0.1, 0.3, 0.8, 0.65])
    plt.hold(1)
    ax = plt.gca()
    if first is not None:
        plot_iq(ax, first.T, err=err, smerr=smerr)
    plot_iq(ax, avg.T, err=err, smerr=smerr)
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
        for qind in range(qlen-20):
            incmap[rep,qind] = chi2wfilt(first, reps[rep,...], qind)

    avg = np.zeros((3, qlen))
    avg[0,:] = first[0,:]
    def sumsq(x): return np.sum(np.square(x))
    for qind in range(qlen):
        avg[1,qind] = np.mean(reps[incmap[:,qind], 1, qind])
        N = np.sum(incmap[:,qind])
        prop = np.sqrt(sumsq(reps[incmap[:,qind], 2, qind])) / N
#        sdev = np.std(reps[incmap[:,qind], 2, qind]) / np.sqrt(N)
#        avg[2,qind] = max(prop, sdev)
        avg[2,qind] = prop

    if plot:
        plot_filtered(avg, incmap, first, figno=plot)

    return (avg, incmap)


def run_filter_on_stacks(filelist):
    for fname in filelist:
        varname = fname[:-4]
        stack = loadmat(fname)[varname]
        for pos in range(stack.shape[0]):
            print("File: %s, pos %d" % (fname, pos))
            first = stack[pos,0,...]
            avg, inds = chifilter_points(stack[pos,...])
            outname = "%s_p%d.yfil" % (varname, pos)
            write_filtered(avg, inds, outname, first=first)
            print(outname)

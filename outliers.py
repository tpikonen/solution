import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2
from scipy.spatial.distance import squareform
from xformats.yamlformats import read_ydat, write_ydat
from xformats.matformats import read_mat
from sxsplots import plot_iq
from biosxs_reduce import chivectors, chi2cdm, mean_stack, md5_file
from clustering import plot_distmat, plot_distmat_marginal, plot_clusterhist


def read_outliers(fname):
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
    cdm = np.array(yd['chi2matrix'])
    incinds = np.array(yd['incinds'])
    threshold = yd['chi2cutoff']

    return (dat[0:3,:], first, aver, incinds, cdm, threshold)


def plot_outliers(filtered, first, aver, inclist, cdm, threshold):
    plt.clf()

    plt.subplot(221)
    sm = 1
    ax = plt.gca()
    plot_iq(ax, first.T, smerr=sm, label="First rep")
    plot_iq(ax, aver.T, smerr=sm, label="All reps")
    plot_iq(ax, filtered.T, smerr=sm, label="Filtered, %d reps" % len(inclist))
    plt.legend()

    plt.subplot(222)
    plot_distmat(cdm)

    plt.subplot(223)
    N = filtered.shape[-1]
    plot_clusterhist(cdm, inclist, N, threshold)

    plt.subplot(224)
    plot_distmat_marginal(cdm, threshold, sublist=inclist)

    plt.show()


def filter_outliers(reps, threshold=1.0, plot=1):
    """Filter by removing repetitions having mutual chisq above `threshold`.

    Returns a tuple containing the included indices and the condensed
    distance matrix.

    Repetitions are removed iteratively by checking which repetition
    contributes the largest number of over the threshold chi-squared values
    (outliers) in the chisq-distance matrix, and removing that point.
    If two repetitions cause an equal number of outliers, the repetition
    which has the highest chisq distance to a non-outlier distance matrix
    point is removed.
    """
    N = reps.shape[0]
    cdm = chi2cdm(reps)
    dmat = squareform(cdm)
    db = dmat > threshold
    dbelow = dmat.copy()
    dbelow[db] = 0.0
    incinds = range(N)
    chistats = np.sum(db[incinds][:,incinds], axis=0)
    while np.sum(chistats) > 0:
        incarr = np.array(incinds)
        inds = np.flipud(np.argsort(chistats))
        #print(incarr[inds])
        #print(chistats[inds])
        topinds = inds[chistats[inds] == chistats[inds[0]]]
        #print(incarr[topinds])
        maxs = np.max(dbelow[incarr[topinds]], axis=1)
        #print(maxs)
        maxind = topinds[np.argmax(maxs)]
        #print(incarr[maxind])
        dbelow[incarr[maxind],:] = 0.0
        dbelow[:,incarr[maxind]] = 0.0
        incinds.remove(incinds[maxind])
        #print(incinds)
        #print("")
        chistats = np.sum(db[incinds][:,incinds], axis=0)

    if plot:
        first = reps[0,...]
        aver = mean_stack(reps)
        filtered = mean_stack(reps[incinds,...])
        plot_outliers(filtered, first, aver, incinds, cdm, threshold)

    return incinds, cdm


def filter_matfile(fname, outstem, p_reject=0.001, plot=1):
    stack = read_mat(fname)
    md5 = md5_file(fname)
    print("Rejection probability: %0.3g" % p_reject)
    N = np.sum(np.logical_not(np.isnan(stack[0,0,1,:])))
    print("Number of valid channels: %d" % N)
    threshold = chi2.ppf(1.0 - p_reject, N) / N
    print("Chisq rejection threshold: %0.3g" % threshold)

    for pos in range(stack.shape[0]):
        reps = stack[pos,...]
        incinds, cdm = filter_outliers(reps, threshold=threshold, plot=plot)
        ms = mean_stack(reps[incinds,...])
        disinds = range(reps.shape[0])
        for i in incinds:
            disinds.remove(i)
        print("Pos %d, discarded: %s" % (pos, str(disinds)))
        ad = { 'chi2cutoff' : float(threshold),
            'rejection_prob' : float(p_reject),
            'incinds' : map(int, list(incinds)),
            'disinds' : map(int, list(disinds)),
            'chi2matrix' : map(float, list(cdm)),
            'method' : "filter_outliers",
            'inputfile' : [ fname, md5 ],
            'inputindex' : int(pos),
            'q~unit' : '1/nm',
            'I~unit' : 'arb.',
            'Ierr~unit' : 'arb.',
            'I_first~unit' : 'arb.',
            'Ierr_first~unit' : 'arb.',
            'I_all~unit' : 'arb.',
            'Ierr_all~unit' : 'arb.',
            }
        outarr = np.zeros((7, ms.shape[1]))
        outarr[0:3,:] = ms
        outarr[3:5,:] = reps[0,1:3,:]
        outarr[5:7,:] = mean_stack(reps)[1:3,:]

        outname = "%s%02d.yfil" % (outstem, pos)
        print(outname)
        write_ydat(outarr, outname, addict=ad,
            cols=['q','I','Ierr','I_first','Ierr_first','I_all','Ierr_all'],
            attributes=['~unit'])


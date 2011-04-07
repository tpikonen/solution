import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hc
import scipy.stats.distributions
import scipy.special
from scipy.spatial.distance import squareform
from optparse import OptionParser
from sxsplots import plot_iq
from biosxs_reduce import mean_stack, stack_datafiles, md5_file, chivectors, chi2cdm
from scipy.special import gammaln as gamln
from scipy.io import loadmat
from xformats.matformats import write_mat
from xformats.yamlformats import write_ydat, read_ydat

description="""\
Filter repeats by linkage-based clustering.
"""

usage="%prog file1.yfil file2.yfil ... "

default_chi2 = 2.2


def filter_with_linkage(links, threshold=1.0):
    """Return the indices of repetitions belonging to the selected clusters.
    """
    cc = hc.fcluster(links, threshold, criterion='distance')
    clusters = [ [] for x in range(cc.max()) ]
    for i in range(len(cc)):
        clusters[cc[i]-1].append(i)
    def len_cmp(x, y): return int(np.sign(len(x) - len(y)))
    clusters.sort(cmp=len_cmp, reverse=True)
    if len(clusters[0]) < 0.9 * (len(links)+1):
        logging.warning("Too many outliers")
    return clusters


def plot_distmat(cdm):
    """Plot the condensed distance matrix `cdm`.
    """
    dmat = squareform(cdm)
    md = np.mean(cdm)
    for i in range(dmat.shape[0]):
        dmat[i, i] = md
    plt.matshow(dmat, fignum=False)
    plt.colorbar()


def plot_distmat_marginal(cdm, threshold=None, sublist=None):
    """Plot the condensed distance matrix `cdm` by summing over one index.

    If `sublist` is given, also plot the sub-distance matrix for the
    elements given in it.
    """
    dmat = squareform(cdm)
    for i in range(dmat.shape[0]):
        dmat[i,i] = 0.0
    bardat = np.max(dmat, axis=0)
    plt.bar(np.arange(dmat.shape[0]) - 0.4, bardat, width=0.6)
    if sublist is not None:
        subarr = np.array(sublist)
        subarr.sort()
        subdmat = dmat[subarr][:, subarr]
        subdat = np.max(subdmat, axis=0)
        plt.bar(subarr - 0.2, subdat, width=0.6, color='red')
        bardat = np.concatenate((bardat, subdat))
    barmin = np.min(bardat)
    barmax = np.max(bardat)
    if threshold:
        plt.axhline(threshold, linestyle='--', color='magenta', label="Threshold = %0.3g" % threshold)
        plt.legend()
    else:
        threshold = barmin
    delta = (barmax - barmin) / 5.0
    xs = 0.5
    ax = [-xs, dmat.shape[0]-xs, min(threshold, barmin)-delta,
        max(threshold, barmax)+delta]
    ax = plt.axis(ax)


def plot_clusterhist(cdm, cluster, N, threshold):
    """Plot distance histograms from condensed distance matrix `cdm`.

    Plot distance histograms of full distance matrix, elements given in
    `cluster` and the theoretical chi-squared distribution for `N` elements.
    """
    plt.hold(1)
    nbins = 4*int(np.sqrt(len(cdm)))
    normed = False
    _, bins, _ = plt.hist(cdm, bins=nbins, normed=normed, histtype='bar', label='Full', color="blue")
    binw = bins[1] - bins[0]
    dmat = squareform(cdm)
    dd = squareform(dmat[cluster][:,cluster])
    plt.hist(dd, bins=bins, normed=normed, histtype='bar', label="Cluster", color="green", rwidth=0.6)
    xx = np.linspace(min(0.8, np.min(bins)), max(1.2, np.max(bins)), 128)
    plt.plot(xx, binw*len(cdm)*chi2norm_pdf(xx, N), label="Chisq_%d full" % N)
    plt.plot(xx, binw*len(dd)*chi2norm_pdf(xx, N), label="Chisq_%d clus" % N)
    plt.axvline(threshold, linestyle='--', color='magenta')
    plt.legend()
    plt.axis('tight')


def plot_dendrogram(links, threshold=1.0):
    """Plot the dendrogram defined by `links`.

    The `links` come from scipy.cluster.hierarchy.linkage().
    """
    hc.dendrogram(links, color_threshold=threshold)
    at = plt.axis()
    cchis = links[:,2]
    delta = 0.05*(np.max(cchis) - np.min(cchis))
    clusterchi2 = links[np.argmax(links[links[:,2] < threshold, 3]), 2]
    plt.axhline(threshold, linestyle='--', label="Threshold = %0.3g\nCluster = %0.3g"
        % (threshold, clusterchi2))
    plt.legend()
    plt.axis((at[0], at[1], np.min(cchis) - delta, max(threshold, np.max(cchis)) + delta))


def plot_clustering(filtered, first, aver, inclist, cdm, links, threshold):
    plt.clf()

    plt.subplot(221)
    sm = 1
    ax = plt.gca()
    plot_iq(ax, first.T, smerr=sm, label="First rep", color='blue')
    plot_iq(ax, aver.T, smerr=sm, label="All reps", color='red')
    plot_iq(ax, filtered.T, smerr=sm, label="Largest cluster, %d reps" % len(inclist), color='lawngreen')
    plt.legend()

    plt.subplot(222)
    plot_distmat(cdm)
#    plot_distmat_marginal(cdm)

    plt.subplot(223)
    N = filtered.shape[-1]
    plot_clusterhist(cdm, inclist, N, threshold)

    plt.subplot(224)
    plot_dendrogram(links, threshold)

    plt.show()


def cluster_reps(reps, threshold=1.0, plot=1):
    """Do clustering based `reps`.

    Returns a tuple with
    - The indices of the largest cluster found
    - The condensed distance matrix
    - Cluster linkage

    Keyword arguments:
        `threshold` : chisq threshold to use in discrimination.
        `plot` : Plot results, if True.
    """
    cdm = chi2cdm(reps)
    links = hc.linkage(cdm, method='complete')
    clist = filter_with_linkage(links, threshold)
    print("Clusters: %s" % str(clist))
    if plot:
        first = reps[0,...]
        aver = mean_stack(reps)
        filtered = mean_stack(reps[clist[0],...])
        plot_clustering(filtered, first, aver, clist[0], cdm, links, threshold)

    return (clist[0], cdm, links)


def chi2norm_pdf(x, k):
    """Return pdf of normalized chi^2_k distribution at x.

    The normalized chi^2_k is sum of squares of k independent standard normal
    variates divided by k.
    """
    # chi2_pdf() is the chi2.pdf implementation from scipy 0.8,
    # the version in scipy 0.7 is broken.
    def chi2_pdf(x, df):
        return np.exp((df/2.-1)*np.log(x+1e-300) - x/2. - gamln(df/2.) - (np.log(2)*df)/2.)
    #return k*scipy.stats.distributions.chi2.pdf(k*x, k)
    return k*chi2_pdf(k*x, k)


def read_clustered(fname):
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
    links = np.array(yd['linkage'])
    threshold = yd['chi2cutoff']

    return (dat[0:3,:], first, aver, incinds, cdm, links, threshold)


# FIXME: Determine the cutoff chi2 automatically from the CDM
def average_positions(filenames, chi2cutoff=1.15, write=True):
    """Filter and average over positions in a capillary.

    """
    filenames.sort()
    stack = stack_datafiles(filenames)

    incinds, cdm, links = cluster_reps(stack, threshold=chi2cutoff)
    ms = mean_stack(stack[incinds,...])

    disinds = range(len(filenames))
    for i in incinds:
        disinds.remove(i)
    included  = [ [filenames[i], md5_file(filenames[i])]
        for i in incinds ]
    discarded = [ [filenames[i], md5_file(filenames[i])]
        for i in disinds ]
    ad = { 'chi2cutoff': float(chi2cutoff),
        'included': included,
        'discarded': discarded,
        'chi2matrix' : map(float, list(cdm)),
        'incinds' : map(int, list(incinds)),
        'linkage' : [ map(float, ll) for ll in list(links) ] }

    outarr = np.zeros((7, ms.shape[1]))
    outarr[0:3,:] = ms
    outarr[3:5,:] = stack[0,1:3,:]
    outarr[5:7,:] = mean_stack(stack)[1:3,:]

    if write:
        fname = filenames[0]
        fname = "%s_clu.ydat" % fname[:(fname.find('_p'))]
        print(fname)
        write_ydat(outarr, fname, addict=ad, cols=['q', 'I', 'Ierr', 'I_first', 'Ierr_first', 'I_all', 'Ierr_all'])
    return ms


def main():
    oprs = OptionParser(usage=usage, description=description)
    oprs.add_option("-c", "--chisq-threshold",
        action="store", type="float", dest="threshold", default=default_chi2,
        help="chi-squared threshold for cluster splitting. Default is %3g" % default_chi2)
    (opts, args) = oprs.parse_args()
    if len(args) < 1:
        oprs.error("One or more input files needed.")
    average_positions(args, chi2cutoff=opts.threshold, write=1)

if __name__ == "__main__":
    main()

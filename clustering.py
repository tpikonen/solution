import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hc
import scipy.stats.distributions
import scipy.special
import temp_distance as dist # a fixed version of scipy.spatial.distance
from sxsplots import plot_iq
from biosxs_reduce import mean_stack
from scipy.special import gammaln as gamln


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


def plot_chidist(x, y):
    nn = clean_indices(x, y)
    N = np.sum(nn)
    chi2 = (x[1,nn]-y[1,nn])**2 / (x[2,nn]**2+y[2,nn]**2)
    plt.plot(chi2)
    return chi2


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


def get_subdists(dmat, elist):
    """Return distances of elems in `elist` from a dist. matrix `dmat`.
    """
    dists = []
    Nel = len(elist)
    for i in range(Nel):
        for j in range(i+1, Nel):
            dists.append(dmat[i, j])
    return dists


def plot_distmat(cdm):
    """Plot the condensed distance matrix `cdm`.
    """
    dmat = hc.distance.squareform(cdm)
    md = np.mean(cdm)
    for i in range(dmat.shape[0]):
        dmat[i, i] = md
    plt.matshow(dmat, fignum=False)
    plt.colorbar()


def plot_clusterhist(cdm, N, threshold):
    """Plot distance histograms from condensed distance matrix `cdm`.

    Plot distance histograms of full distance matrix, the largest cluster
    and the theoretical chi-squared distribution for `N` elements.
    """
    plt.hold(1)
    nbins = 4*int(np.sqrt(len(cdm)))
    (_, bins, _) = plt.hist(cdm, bins=nbins, normed=True, histtype='step', label='Full')
    links = hc.linkage(cdm, method='complete')
    clusters = filter_with_linkage(links, threshold)
    print("Clusters: %s" % str(clusters))
    dmat = hc.distance.squareform(cdm)
    for i in range(1): #len(clusters)):
        dd = get_subdists(dmat, clusters[i])
        plt.hist(dd, bins=bins, normed=True, histtype='step', label="Cluster #%d" % i)
    plt.plot(bins, chi2norm_pdf(bins, N), label="Chisq_%d PDF"%N)
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
    plt.axhline(threshold, linestyle='--', label="Threshold")
    #plt.legend()
    plt.axis((at[0], at[1], np.min(cchis) - delta, max(threshold, np.max(cchis)) + delta))


def plot_average(dat_fil, dat_all, cdm, threshold=1.0):
    """Plot outlier-filtered average and unfiltered average, plus cluster stats.
    """
    plt.clf()
    plt.subplot(221)
    ax = plt.gca()
    plot_iq(ax, dat_all.T, smerr=1)
    plot_iq(ax, dat_fil.T, smerr=1)
    plt.subplot(222)
    plot_distmat(cdm)
    plt.subplot(223)
    N = dat_fil.shape[1]
    plot_clusterhist(cdm, N, threshold)
    plt.subplot(224)
    links = hc.linkage(cdm, method='complete')
    plot_dendrogram(links, threshold)


def plot_repstats(reps, threshold=1.0):
    """Plot cluster statistics and filtered means of `reps`.
    """
    plt.clf()

    plt.subplot(221)
    sm = 1
    ax = plt.gca()
    plot_iq(ax, reps[0,...].T, smerr=sm, label="First rep")
    plot_iq(ax, mean_stack(reps).T, smerr=sm, label="All reps")
    cdm = distmat_reps(reps)
    links = hc.linkage(cdm, method='complete')
    clist = filter_with_linkage(links, threshold)
    plot_iq(ax, mean_stack(reps[clist[0],...]).T, smerr=sm, label="Largest cluster")
    plt.legend()

    plt.subplot(222)
    plot_distmat(cdm)

    plt.subplot(223)
    N = reps.shape[-1]
    plot_clusterhist(cdm, N, threshold)

    plt.subplot(224)
    links = hc.linkage(cdm, method='complete')
    plot_dendrogram(links, threshold)


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


def distmat_reps(reps):
    """Return a condensed chi**2 distance matrix from `reps`.
    """
    return dist.pdist(reps, metric=chivectors)


def test(threshold=1.1):
    rmean = 0.16
    rerr = 0.0033
    nn = scipy.stats.distributions.norm.rvs(rmean, rerr, size=(30, 1000))
    asma = np.zeros((30, 3, 1000))
    asma[:,0,:] = np.arange(1000)
    asma[:,1,:] = nn
    asma[:,2,:] = rerr*np.ones_like(nn)
    asma[0,1,:] = asma[0,1,:] + 0.4 * rerr
    asma[29,1,:] = asma[29,1,:] - 0.4 * rerr
    dmc = distmat_reps(asma)
#    links = hc.linkage(dmc)
#    cc = hc.fcluster(links, threshold)
#    print(cc)
    plot_average(asma[0,...], asma[1,...], dmc, threshold)
#    return links

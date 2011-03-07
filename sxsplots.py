import sys
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import scipy.interpolate as ip


def preplot(*args, **kwargs):
    dat = args[0]
    rest = args[1:]
    dsh = dat.shape
    if dsh[1] > dsh[0] and dsh[0] < 8:
        dat = dat.T
    fig = plt.gcf()
    ax = fig.gca()
    plot_iq(ax, dat, *rest, **kwargs)
    return ax


def splot(*args, **kwargs):
    """Semilogy plot of a dat-array."""
    ax = preplot(*args, **kwargs)
    ax.set_yscale('log')
    plt.show()


def llplot(*args, **kwargs):
    """Loglog-plot of a dat-array."""
    ax = preplot(*args, **kwargs)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()


def pplot(*args, **kwargs):
    """Porod plot (q vs. (q**4.0)*I(q)) of an (3, N) shaped array.
    """
    dat = args[0]
    rest = args[1:]
    dsh = dat.shape
    if dsh[0] > dsh[1] and dsh[1] < 8:
        dat = dat.T
    constant = kwargs.pop("constant", 0.0)
    q = dat[0,:]
    I = dat[1,:]
    plt.plot(q, (q**4.0)*(I-constant), *rest, **kwargs)


def logshow(arr):
    # Works with ID02 data
#    plt.imshow(np.log(np.abs(arr)+arr.max()*1e-4), interpolation='nearest')
    # Works with cSAXS data
    plt.imshow(np.log10(np.abs(arr)+1), interpolation='nearest')


def spsmooth(x, y, yerr):
    # Use the default smoothing in splrep with inverse stderr weights
    # FIXME: find a more intelligent way to filter out zeros in yerr
    nzerr = yerr.copy()
    zrepl = np.abs(np.median(yerr))
    if zrepl == 0.0:
        zrepl = 1.0 # give up
    nzerr[nzerr <= 0.0] = zrepl
    sp = ip.UnivariateSpline(x, y, w=1/nzerr)
    return sp(x)


def plot_pr(ax, g):
    prf = g['prf']
    nconst = np.trapz(prf[:,1], prf[:,0])
    lstr = g['filename'].rpartition('/')[2] + "/%1.3g, Rg=%g" % (nconst, g['Rg_rec'])
    savehold = ax.ishold()
    ax.hold(1)
    ax.plot(prf[:,0], prf[:,1]/nconst, label=lstr)
    ax.axhline(0.0, linewidth=0.4, color='black')
    ax.hold(savehold)


def plot_iq(ax, dat, err=1, smerr=0, **kwargs):
    """Plot array `dat` to axis `ax`.

    Keyword arguments:
        err:    If True, plot errors from the third column from `dat`.
        smerr:  If True, smooth the error with a spline interpolation.

    The standard matplotlib plot keyword args are also accepted.
    """
    inds = np.isnan(dat[:,1]) == False
    x = dat[inds,0]
    y = dat[inds,1]
    markind = mlab.find(y <= 0.0)
    okmin = np.abs(y[y != 0.0]).min()
    y[y == 0.0] = okmin
    savehold = ax.ishold()
    ax.hold(1)
    bandlabel = ''
    if(err and dat.shape[1] > 2):
        yerr = np.abs(dat[inds,2])
        minerr = y-yerr
        maxerr = y+yerr
        if smerr:
            minerr = spsmooth(x, minerr, yerr)
            maxerr = spsmooth(x, maxerr, yerr)
            outofband_ratio = (1.0*np.sum(y > maxerr)+np.sum(y < minerr))/len(y)
            bandlabel = "oob %d%%" % int(100*outofband_ratio)
        minerr[minerr < okmin] = okmin
        maxerr[maxerr < okmin] = okmin
        ax.fill_between(x, minerr, maxerr, alpha=0.1, linewidth=0, label=bandlabel)
#    if kwargs.has_key('label'):
#        label = kwargs.pop('label') + ' | ' + bandlabel
#    else:
#        label = bandlabel
    label = kwargs.pop('label', '')
    line = ax.semilogy(x, np.abs(y), '-', label=label, **kwargs)[0]
    ax.semilogy(x[markind], np.abs(y[markind]), 'o', color=line.get_color())
    ax.set_xlabel("q / (1/nm)")
    ax.set_ylabel("I")
    if label:
        ax.legend()
    ax.hold(savehold)

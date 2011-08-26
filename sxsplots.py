import sys
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import scipy.interpolate as ip
from solution.utils import clean_indices


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


def plot_pr(ax, g, **kwargs):
    prf = g['prf']
    nconst = np.trapz(prf[:,1], prf[:,0])
    lstr = g['filename'].rpartition('/')[2] + "/%1.3g, Rg_real=%g+-%g" % (nconst, g['Rg_real'], g['Rg_err_real'])
    savehold = ax.ishold()
    ax.hold(1)
    ax.plot(prf[:,0], prf[:,1]/nconst, label=lstr, **kwargs)
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


def plot_guinier(ax, dat, grange=None, Rg=None, I0=None, conc=np.nan, annotate=1, *kwargs):
    """Guinier plot of `dat` to axis `ax`."""
    def guinier(I0, Rg, q):
        return I0*np.exp(-(1.0/3)*(Rg*q)**2)
    inds = clean_indices(dat, np.ones_like(dat))
    q = dat[0,:]
    I = dat[1,:]
    Ierr = dat[2,:]
    if grange:
        glen = grange[1] - grange[0]
        pmin = max(0, grange[0] - glen/2)
        pmax = min(len(inds), grange[1] + glen/3)
        inds[0:pmin] = False
        inds[pmax:-1] = False
        inds[-1] = False
    restxt = "c = %3.3f g/l, I(0) = %3.3f, Rg = %3.3f nm" % (conc, I0, Rg)
    ax.errorbar(q[inds]**2, np.log10(I[inds]),\
        (1.0/np.log(10.0))*np.abs(Ierr[inds]/I[inds]), fmt='.', label=restxt,
        capsize=0)
    qline = q[inds]
    l0 = ax.plot(qline**2, np.log10(guinier(I0, Rg, qline)))[0]
    lcolor = l0.properties()['color']
    gstart = (q[grange[0]]**2, np.log10(guinier(I0, Rg, q[grange[0]])))
    gend = (q[grange[1]]**2, np.log10(guinier(I0, Rg, q[grange[1]])))
    plt.plot(gstart[0], gstart[1],
        '|', markersize=100, c=lcolor)
    plt.plot(gend[0], gend[1],
        '|', markersize=100, c=lcolor)

    if annotate:
        plt.annotate("q*Rg = %0.3g" % (Rg*q[grange[0]]), gstart,
            textcoords='offset points', xytext=(0, 60),
            arrowprops=dict(arrowstyle="->",
                connectionstyle="arc,angleA=180,armA=60,angleB=100,armB=20,rad=7"))
        plt.annotate("q*Rg = %0.3g" % (Rg*q[grange[1]]), gend,
            textcoords='offset points', xytext=(0, 60),
            arrowprops=dict(arrowstyle="->",
                connectionstyle="arc,angleA=180,armA=60,angleB=100,armB=25,rad=7"))

    plt.legend()
    plt.xlabel("q**2 / (1/nm**2)")
    plt.ylabel("log10(I) / arb.")
    plt.show()

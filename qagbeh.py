import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import modelfit as mf
from centerfits import fwhm

def get_firstpeak(Iagbeh):
    """Return an estimate for the first peak position in AgBeh scattering.
    """
    dlog = np.diff(np.log(Iagbeh))
    highs = 1.0*(dlog > 0.5*np.nanmax(dlog))
    peaks = mlab.find(np.diff(highs) < -0.5)
    return peaks[0]


def qagbeh(Iagbeh, first_index=None, wavel=0.1, Dlatt=5.8380):
    """Return q-scale optimized from silver behenate scattering.

    Keyword arguments:
        `first_index` : Approximate index of the first AgBeh-reflection
            in the `Iagbeh`. If None, then it is estimated with a heuristic.
        `wavel` : Wavelength of the radiation in nanometers (!).
        `Dlatt` : Lattice spacing in nm, default 5.8380 nm for AgBeh.
    """
    #FIXME: Use argmin(diff(diff(rfft(Iagbe)*rfft(Iagbe).conj())))
    #to determine the first_index ab initio?

    if first_index is None:
        first_index = get_firstpeak(Iagbeh)

    if len(Iagbeh) < first_index:
        raise ValueError("first_index must be inside the intensity array.")
    W = int(first_index // 4) # FIXME: This value is only suitable for AgBeh
    peaks = np.arange(first_index, len(Iagbeh)-W, first_index)

    npeaks = len(peaks)
    opos = []
    optpars = []
    for i in range(len(peaks)):
        cen = peaks[i]
        chan = int(cen)
        inds = np.arange(chan-W, chan+W)
        heightini = np.max(Iagbeh[inds])
        cenini = inds[np.argmax(Iagbeh[inds])]
        sigini = fwhm(inds, Iagbeh[inds]) / (2*np.sqrt(np.log(2)))

#        ginit = [0.0, 0.0, heightini, cenini, sigini]
#        mf.plotmodel(mf.gaussline, ginit, inds, Iagbeh[inds], Iagbeh[inds])
#        (gopt, chisq) = mf.modelfit(mf.gaussline, ginit, inds, Iagbeh[inds],
#            Iagbeh[inds])
#        opos.append(gopt[3])

        linit = [10.0*heightini, cenini, sigini, 0.0]
#        mf.plotmodel(mf.lorentzconstant, linit, inds, Iagbeh[inds],
#            Iagbeh[inds])
        (lopt, chisq) = mf.modelfit(mf.lorentzconstant, linit, inds,
            Iagbeh[inds], Iagbeh[inds], noplot=1)
        optpars.append((lopt, inds))
        opos.append(lopt[1])

    opos = np.array(opos)
    N = np.arange(1, len(peaks)+1)
    qn = (2*np.pi/Dlatt) * N

    def channo(params, n):
        """Return detector channel numbers corresponding to maximum `n`.

        `params` are [L, wavelength]
            `L`          : Sample to detector distance in pixel units.
            `wavelength` : Wavelength of the radiation in nm.
        """
        return params[0]*np.tan(2*np.arcsin(n*params[1]/(2*Dlatt)))

    # initial linear optimization for L
    chan_n = channo([1, wavel], N)
    lsqsol = np.linalg.lstsq(chan_n.reshape(len(N), 1), opos)
    L = lsqsol[0][0] # L is in units of pixel size

#    # Nonlinear optimization with variable L and wavelength
#    popt = mf.modelfit(channo, [L, wavel], N, opos, np.ones_like(opos))
#    plt.waitforbuttonpress()
#    print(popt)
#    L = popt[0][0]
#    wavel = popt[0][1]

    print("S-to-D = %g pixels (%g mm for Pilatus)" % (L, L*0.172))
    q = (4*np.pi/wavel)*np.sin(0.5*np.arctan(np.arange(len(Iagbeh))/L))
#   fitted_peakpos = (4*np.pi/wavel)*np.sin(0.5*np.arctan(opos/L))
    plt.subplot(121)
    plt.plot(N, channo([L, wavel], N)/opos -1, "+", np.arange(len(N)+2), np.zeros(len(N)+2))
    plt.title("Deviation of theoretical peak positions from fitted positions")
    plt.xlabel("Peak number")

    plt.subplot(122)
    plt.semilogy(q, Iagbeh)
    plt.xlabel("q / (1/nm)")
    plt.hold(1)
    for (fopt, inds) in optpars:
        plt.semilogy(q[inds], mf.lorentzconstant(fopt, inds))
        plt.axvline(q[np.round(fopt[1])], color='red')
    for ppos in qn:
        plt.axvline(ppos, color='blue')
    return q
    plt.hold(0)

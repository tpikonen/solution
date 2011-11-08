import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import radbin as r
import solution.modelfit as mf
from scipy.signal import medfilt
from centerfits import fwhm
from solution.xformats.matformats import read_matclean, write_mat


def get_firstpeak(Iagbeh, plot=0):
    """Return an estimate for the first peak position in AgBeh scattering.
    """
    #FIXME: Test with noisy data and check for smoothing algorithm other
    # than medfilt (Maybe with smoothing routine from Chris?).
    filt = medfilt(np.log10(1+Iagbeh), kernel_size=5)
    dlog = medfilt(np.diff(filt), kernel_size=5)
    highs = 1.0*(dlog > 0.5*np.nanmax(dlog))
    peaks = mlab.find(np.diff(highs) < -0.5)
    print("Initial peak positions: %s" % peaks)
    if plot:
        plt.clf()
        plt.hold(1)
        plt.plot(np.log10(1+Iagbeh), label="log10(I+1)")
        plt.plot(filt, label="log10(Ifilt)")
        plt.plot(dlog, label="diff(log10(Ifilt))")
        plt.plot(np.diff(highs), label="highs")
        for pp in peaks:
            plt.axvline(pp, color="red")
        plt.legend()
        plt.show()
    return peaks[0]


def qagbeh(I, first_index=None, numpeaks=None, wavel=0.1, Dlatt=5.8380, pixel=0.172, debug=0):
    """Return q-scale from (AgBeh) standard.

    Arguments:
        `I` : Scattering intensity from the standard.

    Keyword arguments:
        `first_index` : Approximate index of the first reflection
            in the `I`. If None, then it is estimated with a heuristic.
        `numpeaks` : Number of peaks to fit. If None, then a value is
            determined from `first_index` and length of `I`.

    See qagbeh() for rest of the keyword arguments.
    """
    width_per_firstindex = 0.1 # Peak half width in units of first peak position
    if first_index is None:
        first_index = get_firstpeak(I, plot=debug)
    if len(I) < first_index:
        raise ValueError("first_index must be inside the intensity array.")
    endcutoff = 2*int(first_index * width_per_firstindex)
    if numpeaks is None:
        endrange = len(I)-endcutoff
    else:
        endrange = min(len(I)-endcutoff, first_index*numpeaks+1)
    peaks = np.arange(first_index, endrange, first_index)
    return qagbeh_refine(I, peaks, wavel, Dlatt, pixel)


def qagbeh_refine(I, peaks, wavel=0.1, Dlatt=5.8380, pixel=0.172):
    """Return q-scale from (AgBeh) standard with given initial peak positions.

    Arguments:
        `I` : Scattering intensity from the standard.
        `peaks` : A list of initial peak indices to be fitted.

    Keyword arguments:
        `wavel` : Wavelength of the radiation in nanometers (!).
        `Dlatt` : Lattice spacing in nm, default 5.8380 nm for AgBeh.
        `pixel` : Pixel size in mm.
    """
    Iagbeh = I
    width_per_firstindex = 0.1 # Peak half width in units of first peak position

    # Use wider range around the peak for finding initial fitting parameters
    Wini = max(9, 2*int(peaks[0] * width_per_firstindex))
    Wfit = max(9, int(peaks[0] * width_per_firstindex))
    opos = []
    optpars = []
    for i in range(len(peaks)):
        chan = int(peaks[i])
        inds = np.arange(max(0, chan-Wini), chan+Wini)
        inds = inds[np.isfinite(Iagbeh[inds])]
        heightini = np.max(Iagbeh[inds])
        cenini = inds[np.argmax(Iagbeh[inds])]
        sigini = fwhm(inds, Iagbeh[inds]) / (2*np.sqrt(np.log(2)))
        linit = [10.0*heightini, cenini, sigini, 0.0]
        fitinds = np.arange(cenini-Wfit, cenini+Wfit)
        fitinds = fitinds[np.isfinite(Iagbeh[fitinds])]
        (lopt, chisq) = mf.modelfit(mf.lorentzconstant, linit, fitinds,
            Iagbeh[fitinds], Iagbeh[fitinds], noplot=1)
        optpars.append((lopt, fitinds))
        opos.append(lopt[1])

    opos = np.array(opos)
    print("Refined peak positions:\n %s" % opos)

    return qagbeh_fit(Iagbeh, opos, wavel=wavel, Dlatt=Dlatt, pixel=pixel)


def qagbeh_fit(I, peaks, wavel=0.1, Dlatt=5.8380, pixel=0.172):
    """Fit given peak positions to a planar detector model.
    """
    Iagbeh = I
    opos = peaks
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
    s_to_d = L*pixel # in mm

#    # Nonlinear optimization with variable L and wavelength
#    popt = mf.modelfit(channo, [L, wavel], N, opos, np.ones_like(opos))
#    plt.waitforbuttonpress()
#    print(popt)
#    L = popt[0][0]
#    wavel = popt[0][1]

    print("S-to-D = %g pixels with %.4g nm radiation (%g mm for %.4g mm pixel)" % (L, wavel, s_to_d, pixel))
    q = (4*np.pi/wavel)*np.sin(0.5*np.arctan(np.arange(len(Iagbeh))/L))
#   fitted_peakpos = (4*np.pi/wavel)*np.sin(0.5*np.arctan(opos/L))
    plt.clf()
    plt.subplot(121)
    plt.plot(N, channo([L, wavel], N)/opos -1, "+", np.arange(len(N)+2), np.zeros(len(N)+2))
    plt.title("Deviation of theoretical peak positions from fitted positions")
    plt.xlabel("Peak number")

    plt.subplot(122)
    plt.semilogy(q, Iagbeh, color='black')
    plt.xlabel("q / (1/nm)")
    plt.hold(1)
    for pp in peaks:
        plt.axvline(q[np.round(pp)], color='red', label="Fitted pos.")
    for ppos in qn:
        plt.axvline(ppos, color='blue', label="Ideal pos.")

    plt.hold(0)
    plt.title("Red: fitted, Blue: ideal positions")
    plt.show()

    return q, s_to_d, peaks

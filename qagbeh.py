import sys, os.path, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import detformats, csaxsformats, yamlformats, loopyaml
import radbin as r
import modelfit as mf
from scipy.signal import medfilt
from centerfits import fwhm
from optparse import OptionParser
from matformats import read_matclean, write_mat

description="""\
Determine q-scale from a diffraction standard giving equally spaced peaks,
silver behenate by default.

Writes  to the 'q' key in the input bins dictionary, and also to
'qscale.yaml' file.
"""

usage="%prog -b <radindfile.mat> [-o <outputfile.yaml>] file.cbf"


def get_firstpeak(Iagbeh, plot=0):
    """Return an estimate for the first peak position in AgBeh scattering.
    """
    #FIXME: Test with noisy data and check for smoothing algorithm other
    # than medfilt (Maybe with smoothing routine from Chris?).
    filt = medfilt(np.log(Iagbeh), kernel_size=5)
    dlog = medfilt(np.diff(filt), kernel_size=5)
    highs = 1.0*(dlog > 0.5*np.nanmax(dlog))
    peaks = mlab.find(np.diff(highs) < -0.5)
    print("Initial peak positions: %s" % peaks)
    if plot:
        plt.clf()
        plt.hold(1)
        plt.plot(dlog)
        for pp in peaks:
            plt.axvline(pp, color="red")
        plt.show()
    return peaks[0]


# FIXME: Get rid of wavelength argument, or find it automatically
def qagbeh(Iagbeh, first_index=None, wavel=0.1, Dlatt=5.8380, peaks=None, debug=0):
    """Return q-scale optimized from silver behenate scattering.

    Keyword arguments:
        `first_index` : Approximate index of the first AgBeh-reflection
            in the `Iagbeh`. If None, then it is estimated with a heuristic.
        `wavel` : Wavelength of the radiation in nanometers (!).
        `Dlatt` : Lattice spacing in nm, default 5.8380 nm for AgBeh.
        `peaks` : A list of all approximate peak indices (should be supplied
            only if automatic peak detection fails)
    """

    width_per_firstindex = 0.1 # Peak half width in units of first peak position

    if peaks is None:
        if first_index is None:
            first_index = get_firstpeak(Iagbeh, plot=debug)
        if len(Iagbeh) < first_index:
            raise ValueError("first_index must be inside the intensity array.")
        endcutoff = 2*int(first_index * width_per_firstindex)
        peaks = np.arange(first_index, len(Iagbeh)-endcutoff, first_index)

    # Use wider range around the peak for finding initial fitting parameters
    Wini = 2*int(peaks[0] * width_per_firstindex)
    Wfit = int(peaks[0] * width_per_firstindex)
    opos = []
    optpars = []
    for i in range(len(peaks)):
        chan = int(peaks[i])
        inds = np.arange(chan-Wini, chan+Wini)
        heightini = np.max(Iagbeh[inds])
        cenini = inds[np.argmax(Iagbeh[inds])]
        sigini = fwhm(inds, Iagbeh[inds]) / (2*np.sqrt(np.log(2)))
        linit = [10.0*heightini, cenini, sigini, 0.0]
        fitinds = np.arange(cenini-Wfit, cenini+Wfit)
        (lopt, chisq) = mf.modelfit(mf.lorentzconstant, linit, fitinds,
            Iagbeh[fitinds], Iagbeh[fitinds], noplot=1)
        optpars.append((lopt, fitinds))
        opos.append(lopt[1])

    opos = np.array(opos)
    print("Refined peak positions:\n %s" % opos)
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
    plt.clf()
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
        plt.axvline(q[np.round(fopt[1])], color='red', label="Fitted pos.")
    for ppos in qn:
        plt.axvline(ppos, color='blue', label="Theor. pos.")

    plt.hold(0)
    plt.show()

    return q


def main():
    framefile_ext = 'cbf'
    oprs = OptionParser(usage=usage, description=description)
    oprs.add_option("-b", "--bins",
        action="store", type="string", dest="binfile", default=None,
        help="Read bins from a 'radind.mat' file (output of make_radind).")
    oprs.add_option("-o", "--output",
        action="store", type="string", dest="outfile", default="qscale.yaml",
        help="Output file containing the q-scale in looped YAML format. Default is 'qscale.yaml'")
    oprs.add_option("-n", "--readonly-bins",
        action="store_false", dest="writebin", default=True,
        help="Do not write q-scale to the binfile given with option -b.")
    oprs.add_option("-d", "--debug",
        action="store_true", dest="debug", default=False,
        help="Show extra debug plots.")
    (opts, args) = oprs.parse_args()

    if len(args) == 1:
        fname = args[0]
#    elif(len(args) == 1 and os.path.isdir(args[0])):
    else:
        oprs.error("One input file required.")

    if opts.binfile:
        radind = read_matclean(opts.binfile)['radind']
    else:
        oprs.error("Binfile name is missing.")

    if not "indices" in radind:
        oprs.error("Binfile is missing 'indices' key.")

    frame = detformats.read_cbf(fname)
    Iagbeh = r.binstats(radind['indices'], frame.im)[0] # Get the mean

    q = qagbeh(Iagbeh, debug=opts.debug)
    lq = loopyaml.Loopdict({'q': map(float, list(q))}, ['q'])
    yamlformats.write_yaml(lq, opts.outfile)
    if opts.writebin:
        radind['q'] = q
        write_mat('radind', fname=opts.binfile, value=radind, overwrite=1)

if __name__ == "__main__":
    main()

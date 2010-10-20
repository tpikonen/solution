import sys, re
import numpy as np
import numpy.ma as ma
import matplotlib.mlab as mlab
import scipy.optimize as optim
import scipy.interpolate as ip
import matplotlib.pyplot as plt
import radbin as c
from sxsplots import logshow
from detformats import read_cbf, read_mask
from optparse import OptionParser

usage="%prog agbeh.cbf"
description="Determine the direct beam center from a diffraction pattern."

# FIXME: This file is a mess.

# Helper functions

def get_im_max(image):
    c_x = np.sum(image, axis=0).argmax()
    c_y = np.sum(image, axis=1).argmax()
    return [c_x, c_y]


def chivectors(x, y):
    nn = np.logical_not(np.logical_or(
        np.logical_or(np.isnan(x), np.isnan(y)),
        np.logical_or((x <= 0.0), (y <= 0.0))))
    N = np.sum(nn)
    chisq = (1.0/N) * ma.sum((x[nn] - y[nn])**2 / (x[nn]**2 + y[nn]**2))
    return chisq


def fwhm(x, y):
    """
    FWHM from an array (x,y) with interpolation.
    If index goes out of bounds during interpolation, revert to
    non-interpolated FWHM.
    """
    h = (np.max(y) - np.min(y))/2.0
    aboves = mlab.find(y > h)
    max_ab = aboves[-1]
    min_ab = aboves[0]
    try:
        ux1 = x[max_ab]
        uy1 = y[max_ab]
        ud  = (y[max_ab+1] - uy1) / (x[max_ab+1] - ux1)
        ux = (h - uy1) / ud + ux1
        lx1 = x[min_ab-1]
        ly1 = y[min_ab-1]
        ld  = (y[min_ab] - ly1) / (x[min_ab] - lx1)
        lx = (h - ly1) / ld + lx1
        w = ux - lx
    except IndexError:
        w = x[max_ab] - x[min_ab]
    return w


def secondmoment(x,y):
    m = np.sum(x*y)/np.sum(y)
    w = abs(np.sum(y*(x-m)**2)/np.sum(y))


def plotcens(arrlist):
    xs = [ a[0] for a in arrlist ]
    ys = [ a[1] for a in arrlist ]
    plt.plot(xs, ys, '+')


def mark_cross(center, **kwargs):
    """Mark a cross. Correct for matplotlib imshow funny coordinate system.
    """
    N = 20
    plt.hold(1)
    plt.axhline(y=center[1]-0.5, **kwargs)
    plt.axvline(x=center[0]-0.5, **kwargs)


def plotstart(image, startcen, mask=None,  plotit=True):
    sp = None
    if plotit:
        plt.ion()
        plt.figure(1)
#        sp = plt.subplot(111)
        plt.hold(0)
        if mask is None:
            logshow(image)
        else:
            logshow(image*mask)
        plt.hold(1)
        ax = plt.axis()
        mark_cross(startcen, color='yellow', linestyle='--')
        plt.axis(ax)
        plt.draw()
#        plt.show()
    return sp


def plotend(startcen, optcen, plotit=True):
    if(plotit):
        ax = plt.axis()
        mark_cross(optcen, color='white')
        plt.axis(ax)
        dd = max(np.linalg.norm(startcen - optcen), 10)
        llo = optcen - 4*dd
        lhi = optcen + 4*dd
        sp = plt.subplot(111)
        sp.set_xlim(llo[0], lhi[0])
        sp.set_ylim(llo[1], lhi[1])
        plt.show()
#        plt.waitforbuttonpress()
        plt.hold(0)


def get_min_gradient(image, cen, mask=None, start_exc=25):
    """Return the radial coordinate with the largest negative gradient.

    Needed for center refinement with sector or ring methods.

    Arguments:
        `image`: Image to use.
        `cen`: Tuple with the center coordinates.
        `mask`: Mask giving the pixels to be used in determination.
        `start_exc`: Number of points to exclude from the start (center) of the
            image radial gradient array
    """
    [I, n1] = c.radbin(image, cen, mask=mask)
    r = np.arange(len(I))
    yerr = np.sqrt(I)
    nzerr = yerr.copy()
    zrepl = np.abs(np.median(yerr))
    if zrepl == 0.0:
        zrepl = 1.0 # give up
    nzerr[nzerr <= 0.0] = zrepl
    sp = ip.UnivariateSpline(r, I, w=1/nzerr)
#    smoo = [ sp.derivatives(x)[0] for x in r ]
    grad = [ sp.derivatives(x)[1] for x in r ]
    gradmin = np.argmin(grad[start_exc:]) + start_exc
#    plt.plot(r, smoo, r, I, r, grad)
#    print("Min gradient: %d" % gradmin)
    return gradmin

# Fitting functions


#TODO Idea: calculate gradients on an array of points in the image,
# define a line going through the point to the gradient direction,
# calculate pair-wise intersections of the lines,
# cluster the resulting points and return the mean from the 'best' cluster.
# The advantage of this method would be to get an accurate initial center
# for centers outside the image.
# See: http://en.wikipedia.org/wiki/Structure_tensor
# See: linkinghub.elsevier.com/retrieve/pii/S0898122103800214
def centerfit_gradient(image, mask=None, plotit=False):
    pass


def symcen1d(arr, minlen=40, cen_exc=1):
    """Return the index of the symmetry center in a 1D array.

    The center of symmetry is determined by extracting two symmetric
    subarrays around a point and calculating chi-squared between them.
    This is done for (almost) all points, and the point with the smallest
    value of chi-squared is deemed the center of symmetry.

    Keyword arguments:
        `minlen` : Minimum length of subarrays from which chisq is calculated.
        `cen_exc`: Number of points around the tested center excluded from
            the symmetry check.

    The returned index will be in the range
    [minlen+cen_exc, len(arr)-(minlen+cen_exc)].
    """
    alen = len(arr)
    cexc = cen_exc
    mlen = minlen
    out = -1.0 * np.ones_like(arr)
    for i in range(cexc+mlen,alen-(cexc+mlen)):
        chimaxlen = min(i, alen-i)
        slf = slice(i+cexc,i+chimaxlen)
        slr = slice(i-cexc,i-chimaxlen, -1)
        out[i] = chivectors(arr[slf], arr[slr])
    out[np.isnan(out)] = -1
    M = out.max()
    out[out < 0] = M
    return out.argmin()
#    return out


def centerfit_1dsymmetry(image, minlen=100, cen_exc=15, mask=None, plotit=False):
    """Return indices of an approximate symmetry center in a 2D image.

    The symmetry center of the 2D array is calculated from 1D mean values
    calculated in horizontal and vertical directions.

    The 1D centers of symmetry are determined by extracting two symmetric
    subarrays around a point and calculating chi-squared between them.
    This is done for all points where `minlen`+`cen_exc` symmetric points
    are available. The point with the smallest value of chi-squared is
    returned as the center of symmetry.

    Keyword arguments:
        `minlen` : Minimum length of subarrays from which chisq is calculated.
        `cen_exc`: Number of points around the tested center excluded from
            the symmetry check.

    The returned index will be in the range
    [minlen+cen_exc, len(arr)-(minlen+cen_exc)].
    """
    if mask is None:
        im = ma.array(image)
    else:
        im = ma.array(image, mask=np.logical_not(mask))
    ymean = ma.mean(im, axis=0).astype(np.float64)
    xmean = ma.mean(im, axis=1).astype(np.float64)
    kwas = { 'minlen' : minlen, 'cen_exc' : cen_exc }
    return symcen1d(ymean, **kwas), symcen1d(xmean, **kwas)


def centerfit_peakwidth(image, startcen=None, peakrange=None, baseline=None, mask=None, plotit=False):
    """Return center from an image with a ring.

    Minimizes the width of the peak obtained from radial averaging
    in the given range as a function of the center position."""

    if peakrange is None:
        raise ValueError("peakrange must be supplied")

    if startcen is None:
        startcen = get_im_max(image)

    [r1, n1] = c.radbin(image, startcen, peakrange, mask=mask)
    if baseline == None:
        baseline = r1.min()
    x = np.arange(peakrange.size - 1)
    if(plotit):
        plt.hold(0)
        plt.plot(x, r1)
        plt.show()
#        plt.waitforbuttonpress()


    def ofun_ring(cen):
        """Return width of a Debye ring with the given center"""
        [r, n] = c.radbin(image, cen, peakrange, mask=mask)
        w = fwhm(x, r)
        # 2nd moment is not very stable
#        w = secondmoment(x, r)
#        print cen, w
#        sys.stdout.flush()
        return w

    def ofun_cylsymm(cen):
        """Calculate STD from counts in a (narrow) ring in the image"""
        R = 100
        [r, n] = c.radbin(image, cen, radrange=np.array([R,R+2]), phirange=np.linspace(0, 2*np.pi, np.floor(2*np.pi*R)), mask=mask)
        r = r[r != 0]
        w = np.std(r)
#        plt.hold(0)
#        plt.plot(r)
#        plt.show()
#        plt.waitforbuttonpress()
#        print cen, w
#        sys.stdout.flush()
        return w


    def ofun_sectors(cen):
        """Calculate two chi2s from two opposing sectors and multiply"""
        rstart = 120
        rstop = 220
        hwidth = 10 * (np.pi/180) # width in rads
        sect_cens = [0, 0.5*np.pi, np.pi, 1.5*np.pi]
        seclims = [ np.array([a - hwidth, a + hwidth])+np.pi/4.0 for a in sect_cens ]
        rints = [ c.radbin(image, cen, radrange=np.linspace(rstart,rstop), phirange=x, mask=mask)[0] for x in seclims ]
        chi2_1 = chivectors(rints[0], rints[2])
        chi2_2 = chivectors(rints[1], rints[3])

#        print(cen, chi2_1, chi2_2)
#        sys.stdout.flush()
#        plt.hold(0)
#        plt.plot(rints[0], 'b')
#        plt.hold(1)
#        plt.plot(rints[2], 'b')
#        plt.plot(rints[1], 'r')
#        plt.plot(rints[3], 'r')
#        plt.waitforbuttonpress()

        return chi2_1*chi2_2


    optcen = optim.fmin(ofun_sectors, startcen, disp=True)
    [r2, n2] = c.radbin(image, optcen, peakrange, mask=mask)
    if(plotit):
        plt.plot(x, r1, x, r2)
        plt.show()
    return optcen


def centerfit_ringvariance(image, startcen=None, ringradius=None, ringwidth=2,
                           mask=None, plotit=False):
    """Return symmetry center of an image by minimizing variance on a ring.

    Keyword arguments:
        `startcen`: Initial guess for the center. If not given, uses the
            position with the largest value.
        `ringradius`: Radial coordinate of the ring where optimization
            is performed. If not given, uses the value with the largest
            negative radial gradient, determined by using `startcen`
            as a center.
        `ringwidth`: Width of the ring used in optimization.
        `mask`: Mask giving the pixels to be used in determination.
        `plotit`: If true, plot images showing initial and final coordinates.
    """
    if startcen is None:
        startcen = get_im_max(image)
    if ringradius is None:
        ringradius = get_min_gradient(image, startcen, mask)

    plotstart(image, startcen, mask, plotit)

    def ofun_ringvariance(cen):
        """Calculate STD from counts in a (narrow) ring in the image"""
        [r, n] = c.radbin(image, cen, \
                    radrange=np.array([ringradius, ringradius+ringwidth]), \
                    phirange=np.linspace(0, 2*np.pi, \
                    np.floor(2*np.pi*ringradius)), mask=mask)
        r = r[r != 0]
        w = np.std(r)
        return w

    optcen = optim.fmin(ofun_ringvariance, startcen, disp=0)
    plotend(startcen, optcen, plotit)
    return optcen

def centerfit_sectors(image, startcen=None, rlimits=None, secwidth=10.0, mask=None, plotit=False):
    """Return beam center from an image with (approximate) cylindrical symmetry.

    The beam center must be inside the image coordinates, not outside the frame.

    Keyword arguments:
        `startcen`: Initial guess for the center. If not given, uses the
            position with the largest value.
        `rlimits`: Tuple giving the radial limits of the sector which is fitted.
            If not given, uses the value with the largest negative radial
            gradient, determined by using `startcen` as a center.
        `secwidth`: Width of the sector to be optimized in degrees.
        `mask`: Mask giving the pixels to be used in determination.
        `plotit`: If true, plot images showing initial and final coordinates.
    """
    if startcen is None:
        startcen = get_im_max(image)
    if rlimits is None:
#        rlimits = (120, 220)
        gmin = get_min_gradient(image, startcen, mask)
        rlimits = (gmin, gmin+100)

    rstart = rlimits[0]
    rstop = rlimits[1]
    hwidth = secwidth * (np.pi/180) # width in rads

    plotstart(image, startcen, mask, plotit)

    def ofun_sectors(cen):
        """Calculate two chi2s from two opposing sectors and multiply"""
        sect_cens = [0, 0.5*np.pi, np.pi, 1.5*np.pi]
        seclims = [ np.array([a - hwidth, a + hwidth])+np.pi/4.0 for a in sect_cens ]
        rints = [ c.radbin(image, cen, radrange=np.linspace(rstart,rstop), phirange=x, mask=mask)[0] for x in seclims ]
        chi2_1 = chivectors(rints[0], rints[2])
        chi2_2 = chivectors(rints[1], rints[3])
        return chi2_1*chi2_2

    optcen = optim.fmin(ofun_sectors, startcen, disp=0)
    plotend(startcen, optcen, plotit)
    return optcen


def parse_center(center_str):
    mob = re.match(' *([0-9.]+)[,]([0-9.]+) *', center_str)
    if mob is None or len(mob.groups()) != 2:
        return None
    else:
        return (float(mob.group(1)), float(mob.group(2)))

# Command line interface

def main():
    oprs = OptionParser(usage=usage, description=description)
    oprs.add_option("-n", "--noplot",
        action="store_true", dest="noplot", default=False,
        help="Do not plot the frame and center")
    oprs.add_option("-m", "--maskfile",
        action="store", type="string", dest="maskfile", default=None)
    oprs.add_option("-c", "--center",
        action="store", type="string", dest="inicen_str", default=None)
    (opts, args) = oprs.parse_args()

    inicen = None
    if opts.inicen_str is not None:
        inicen = parse_center(opts.inicen_str)
        if inicen is None:
            print >> sys.stderr, oprs.format_help()
            print >> sys.stderr, "Could not parse center"
            sys.exit(1)
        print("Using " + str(inicen) + " as initial center.")
    if(len(args) < 1):
        print >> sys.stderr, oprs.format_help()
        print >> sys.stderr, "At least one input file is needed"
        sys.exit(1)

    mask = None
    if opts.maskfile != None:
        mask = read_mask(opts.maskfile)

    init_func = centerfit_1dsymmetry
    refine_func = centerfit_sectors

    cens = np.zeros((len(args), 2))
    agbe = read_cbf(args[0])
    print(args[0])
    if inicen is None:
        startcen = init_func(agbe.im, mask=mask)
    else:
        startcen = inicen
    print(startcen)
    for i in range(0, len(args)):
        agbe = read_cbf(args[i])
        center = refine_func(agbe.im, startcen=startcen, plotit=(not opts.noplot), mask=mask)
        print(center.__repr__())
        cens[i,:] = center

    mcen = np.mean(cens, axis=0)
    dcen = np.std(cens, axis=0)
    # FIXME: Do proper clustering on centers
    tstr = refine_func.__name__ + "\n Center: " + str(mcen) + '+-' + str(dcen)
    print(tstr)
    plt.plot(cens[:,0], cens[:,1], '+b', mcen[0], mcen[1], '+r')
    plt.title(tstr)
    plt.show()


if __name__ == "__main__":
    main()

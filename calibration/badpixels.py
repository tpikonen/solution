import warnings, scipy.signal
import numpy as np
import numpy.ma as ma
from solution.xformats.detformats import read_cbf, read_eiger, write_pnglog


def pilatus_gapmask(modules, chipgaps=False, chipedges=2):
    """Return a mask for a Pilatus detector with a given module configuration.

    Pilatus 2M at cSAXS gives a frame of 1679 x1475 pixels. The
    Pilatus detectors are composed of modules with 487 x 195 pixels in each.
    The insensitive areas between modules are 7 pixels wide horizontally
    and 17 pixels wide vertically.

    Each module is composed of 8 x 2 chips with 60 x 97 pixels in each chip.
    There appears to be a 1 pixel physical gap between chips, resulting
    in total module size of 8*60 + 7 = 487 times 2*97 + 1 = 195 pixels.
    The counts in this missing pixel are (apparently) calculated by the
    detector firmware from neighbouring pixel values.

    The gap pixel and the pixels at the chip edges (1 or 2 pixels)
    produce count distributions which have ok mean values, but bad higher
    moments, i.e. variance which is either lower (gap pixels) or higher
    (edge pixels) than in the pixels in the center of the chip.

    Input parameters:
        `modules` : Sequence of length 2 giving the module configuration,
            e.g. (5, 2) for Pilatus 1M.
        `chipgaps`: If True, then also the chip gaps and edges are masked.
        `chipedges`: The number of pixels to mask in the edges of
            the pixels (0 to 2 are reasonable values).
    """
    # shape of a single module (without gap strips)
    mshape = (195, 487)
    # shape of a single chip
    cshape = (97, 60)
    # width of column module gap
    mgcw = 7
    # width of row module gap
    mgrw = 17
    # width of the chip gap
    cgw = 1
    # width of unreliable region at the chip edge
    cew = chipedges

    # shape of the whole detector
    pshape = (modules[0]*mshape[0] + (modules[0]-1)*mgrw,
              modules[1]*mshape[1] + (modules[1]-1)*mgcw)

    # First columns and rows of modules
    modstartrows = range(0, pshape[0], mshape[0]+mgrw)
    modstartcols = range(0, pshape[1], mshape[1]+mgcw)

    catl = lambda a,b: a + b
    modgaprows = reduce(catl, [ range(s-mgrw, s) for s in modstartrows[1:]])
    modgapcols = reduce(catl, [ range(s-mgcw, s) for s in modstartcols[1:]])

    mask = np.ones(pshape, dtype=np.bool)
    mask[modgaprows,:] = False
    mask[:,modgapcols] = False

    # First columns and rows of chips in a module
    chipstartrows = range(0, mshape[0], cshape[0]+cgw)
    chipstartcols = range(0, mshape[1], cshape[1]+cgw)

    chipgaprows = [ m + g for m in modstartrows
                          for g in [ s-1 for s in chipstartrows[1:]] ]
    chipgapcols = [ m + g for m in modstartcols
                          for g in [ s-1 for s in chipstartcols[1:]] ]

    chipedgerows = reduce(catl, [range(s,s+cew)+range(s+cshape[0]-cew,s+cshape[0])
                                for s in [ m + c
                                            for m in modstartrows
                                            for c in chipstartrows ]])
    chipedgecols = reduce(catl, [range(s,s+cew)+range(s+cshape[1]-cew,s+cshape[1])
                                for s in [ m + c
                                            for m in modstartcols
                                            for c in chipstartcols ]])
    if chipgaps:
        mask[chipgaprows,:] = False
        mask[:,chipgapcols] = False
        mask[chipedgerows,:] = False
        mask[:,chipedgecols] = False

    return mask


def pilatus2m_gapmask(chipgaps=False, chipedges=2):
    """Return a mask with the Pilatus 2M gaps.

    Pilatus 2M is composed of 3 x 8 modules and outputs a frame of
    1679 x1475 pixels.

    See function `pilatus_gapmask` for further details.

    Input parameters:
        `chipgaps` : If True, then also the chip gaps and edges are masked.
        `chipedges`: The number of pixels to mask in the edges of
            the pixels (0 to 2 are reasonable values).
    """
    return pilatus_gapmask((8,3), chipgaps, chipedges)


def pilatus1m_gapmask(chipgaps=False, chipedges=2):
    """Return a mask with the Pilatus 1M gaps.

    Pilatus 1M is composed of 2 x 5 modules and outputs a frame of
    1043 x 981 pixels.

    See function `pilatus_gapmask` for further details.

    Input parameters:
        `chipgaps` : If True, then also the chip gaps and edges are masked.
        `chipedges`: The number of pixels to mask in the edges of
            the pixels (0 to 2 are reasonable values).
    """
    return pilatus_gapmask((5,2), chipgaps, chipedges)


def match_shape_to_pilatus(inshape):
    """Return a pilatus module configuration matching a shape.

    If a match is not found, returns None.
    """
    # shape of a single module (without gap strips)
    mshape = (195, 487)
    # width of column module gap
    mgcw = 7
    # width of row module gap
    mgrw = 17

    def checkit(p, m, g):
        a = p + g
        b = m + g
        if a % b == 0:
            return a // b
        else:
            return None

    y = checkit(inshape[0], mshape[0], mgrw)
    x = checkit(inshape[1], mshape[1], mgcw)
    if y is not None and x is not None:
        return (y,x)
    else:
        return None

bool_add = ma.mask_or


def bool_sub(A, B):
    """Array logical operation 'A - B', equal to ( A and (not B) ).
    Truth table:

    A B 'A-B'
    T T F
    T F T
    F T F
    F F F
    """
    return np.logical_and(A, np.logical_not(B))


def find_bad_pixels(files, hot_threshold=np.inf, var_factor=5.0, chipgaps=False, read_fun=read_cbf):
    """Return a mask where pixels determined as dead or hot are masked."""

    # Reject pixel if the sum of this pixel in all frames is less than this
    dead_threshold = 1
    # Reject if ratio of pixel value to median filtered value exceeds this
    # in all frames
    medfilt_var_const = 3.0
    # Reject if ratio of pixel value to median filtered value exceeds this
    # even in a single frame
    medfilt_var_gross = 6.0
    # Curvature threshold above which deviates from median are not rejected
    laplace_threshold = 500.0


    # Laplace operator with diagonals, see
    # http://en.wikipedia.org/wiki/Discrete_Laplace_operator
    laplace = np.ones((3,3), dtype=np.float64)
    laplace[1,1] = -8.0

    f0 = read_fun(files[0])
    s = f0.im.shape
    modules = match_shape_to_pilatus(s)
    if modules is None:
        warnings.warn("Frame shape does not match any Pilatus. Not using a gap mask.")
        a_gaps = np.zeros(s, dtype=np.bool)
    else:
        a_gaps = np.logical_not(pilatus_gapmask(modules, chipgaps=chipgaps))
    # Boolean arrays of various invalid pixels (logical_not of a mask!)
    # Pixels above hot_threshold
    a_hots = ma.make_mask_none((s[0], s[1]))
    # Pixels always deviating from median filtered
    a_medf = np.ones((s[0], s[1]), dtype=np.bool)
    # Pixels grossly deviating from median filtered, even in a single frame
    a_gross = np.zeros((s[0], s[1]), dtype=np.bool)

    Asum = np.zeros((s[0], s[1]), dtype=np.float64)
    Asumsq = np.zeros((s[0], s[1]), dtype=np.float64)
    for i in range(0,len(files)):
        print(files[i])
        tim = (read_fun(files[i])).im.astype(np.float64)
        Asum += tim
        Asumsq += tim**2.0
        a_hots = bool_add(a_hots, (tim > hot_threshold))
        mfilt = scipy.signal.medfilt2d(tim, kernel_size=5)
        # Pixels where curvature is less than threshold
        lapm = (np.abs(scipy.signal.convolve2d(mfilt, laplace, mode='same',
            boundary='symm')) < laplace_threshold)
        absdev = np.abs(tim - mfilt)
        # Pixels deviating somewhat from median, not in curved regions
        mrej = (medfilt_var_const < (absdev/(mfilt+1))) * lapm
        a_medf = a_medf * mrej
        # Pixels deviating grossly from median filtered
        a_gross = bool_add(a_gross, (medfilt_var_gross < (absdev/(mfilt+1))) * lapm)

    sA = Asum
    mA = Asum / float(len(files))
    Esq = Asumsq / float(len(files))
    vA = Esq - mA**2

    a_consts = bool_sub((vA == 0),  a_gaps)
#    a_randoms = ((var_factor*vA) > mA)
    a_randoms = False # Something strange with variance, disabling for now
    a_deads = bool_sub((sA < dead_threshold),  a_gaps)
    # total bads outside gaps
    a_bads = reduce(bool_add, [a_consts, a_randoms, a_hots, a_deads, a_medf, a_gross])

    print("Frame has %dx%d = %d pixels" % (s[0], s[1], s[0]*s[1]))
    print("%d invalid pixels in module gaps" % (np.sum(a_gaps)))
    nconsts = np.sum(a_consts)
    print("%d constant pixels outside gaps, %d are >= %d" % \
            (nconsts, (nconsts - np.sum(a_deads)), dead_threshold))
#    print("Found %d pixels with variance larger than %f * mean" % (np.sum(m_randoms), var_factor))
    print("%d hot (count > %e) pixels" % (np.sum(a_hots), hot_threshold))
    print("%d pixels consistently deviating from median" % (np.sum(a_medf)))
    print("%d pixels grossly deviating from median" % (np.sum(a_gross)))
    print("Total of %d bad pixels outside of gaps" % (np.sum(a_bads)))
    mask = np.ones((s[0], s[1]), dtype=np.bool)
    mask = bool_sub(mask, bool_add(a_bads, a_gaps))
    print("Mask has a total of %d bad pixels" % (np.sum(mask == False)))

    return mask

import numpy as np
from sasformats import read_cbf, write_pnglog
from optparse import OptionParser

description="Write a bad pixel mask based on analysis of a series of input frames"

usage="%prog -o <output.png> [cbf-dir | file1.cbf file2.cbf ...]"


def pilatus2m_gaps(chipgaps=False, chipedges=2):
    """Return a mask with the cSAXS pilatus 2M pixel gaps.

    Pilatus 2M at cSAXS gives a frame of 1679 x1475 pixels. The detector is
    composed of 3 x 8 modules with 487 x 195 pixels in each module.
    The insensitive areas between modules are 7 pixels wide horizontally
    and 17 pixels wide vertically.

    Each module is composed of 8 x 2 chips with 60 x 97 pixels in each chip.
    There appears to be a 1 pixel physical gap between chips, resulting
    in total module size of 8*60 + 7 = 487 times 2*97 + 1 = 195 pixels.
    The counts in this missing pixel are apparently fabricated by the
    detector firmware from neighbouring pixel values.

    The gap pixel and the pixels at the chip edges (1 or 2 pixels, not sure)
    produce count distributions which have ok mean values, but bad higher
    moments, i.e. variance which is either lower (gap pixels) or higher
    (edge pixels) than in the pixels in the center of the chip.

    Input parameters:
        `chipgaps` : If True, then also the chip gaps and edges are masked.
        `chipedges`: The number of pixels to mask in the edges of
            the pixels (0 to 2 are reasonable values).
    """
    # shape of pilatus array
    pshape = (1679, 1475)
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


logical_plus = np.logical_or

def logical_minus(A, B):
    """Array logical operation 'A - B', equal to ( A and (not B) ).
    Truth table:

    A B 'A-B'
    T T F
    T F T
    F T F
    F F F
    """
    return np.logical_and(A, np.logical_not(B))

def find_bad_pixels(files, hot_threshold=np.inf, var_factor=5.0):
    """Return a mask where pixels determined as dead or hot are masked"""

    # FIXME: Do median filtering on each frame and check for anomalies?

    dead_threshold = 1

    f0 = read_cbf(files[0])
    s = f0.im.shape
    A = np.zeros((len(files), s[0], s[1]))
    m_hots = np.zeros((s[0], s[1]), dtype=np.bool)
    for i in range(0,len(files)):
        tim = (read_cbf(files[i])).im
        m_hots = logical_plus(m_hots, (tim > hot_threshold))
        A[i,:,:] = tim

    sA = np.sum(A, axis=0, dtype=np.float64)
    mA = np.mean(A, axis=0, dtype=np.float64)
    vA = np.var(A, axis=0, dtype=np.float64)

    m_gaps = (get_pilatus2m_gaps() == 0)
    m_consts = logical_minus((vA == 0),  m_gaps)
#    m_randoms = ((var_factor*vA) > mA)
    m_randoms = False # Something strange with variance, disabling for now
    m_deads = logical_minus((sA < dead_threshold),  m_gaps)
    # total bads outside gaps
    m_bads = logical_plus(logical_plus(m_consts, m_randoms), \
                            logical_plus(m_hots, m_deads) )

    print("Module gaps have %d invalid pixels" % (np.sum(m_gaps)))
    nconsts = np.sum(m_consts)
    print("Found %d constant pixels outside gaps, %d are >= %d" % \
            (nconsts, (nconsts - np.sum(m_deads)), dead_threshold))
#    print("Found %d pixels with variance larger than %f * mean" % (np.sum(m_randoms), var_factor))
    print("Found %d hot (count > %e) pixels" % (np.sum(m_hots), hot_threshold))
    print("Total of %d bad pixels outside of gaps" % (np.sum(m_bads)))
    mask = np.ones((s[0], s[1]), dtype=np.int8)
    mask = mask - logical_plus(m_bads, m_gaps)
    print("Mask has a total of %d bad pixels" % (np.sum(mask == 0)))

    return mask


def main():
    import sys, os.path, glob
    framefile_ext = 'cbf'
    oprs = OptionParser(usage=usage, description=description)
    oprs.add_option("-o", "--output",
        action="store", type="string", dest="outfile", default="")
    oprs.add_option("-t", "--hot-threshold",
        action="store", type="float", dest="hot_threshold", default=np.inf)
    (opts, args) = oprs.parse_args()

    files = []
    if(len(args) == 1 and os.path.isdir(args[0])):
        files = glob.glob(os.path.abspath(args[0]) + '/*.' + framefile_ext)
    elif(len(args) > 1):
        files = args

    if(len(files) < 2):
        print >> sys.stderr, oprs.format_help()
        print >> sys.stderr, "At least 2 input files are needed"
        sys.exit(1)

    if len(opts.outfile) > 0:
        outfile = opts.outfile
    else:
        print >> sys.stderr, oprs.format_help()
        print >> sys.stderr, "Output file name is missing"
        sys.exit(1)

    mask = find_bad_pixels(files, hot_threshold=opts.hot_threshold)
    write_pnglog(mask, outfile)


if __name__ == "__main__":
    main()


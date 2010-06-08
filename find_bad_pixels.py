import numpy as np
from sasformats import read_cbf, write_pnglog
from optparse import OptionParser

description="Write a bad pixel mask based on analysis of a series of input frames"

usage="%prog -o <output.png> [cbf-dir | file1.cbf file2.cbf ...]"


def get_pilatus2m_gaps():
    """Return a mask with the cSAXS pilatus 2M module gaps"""
    # shape of pilatus array
    pshape = (1679, 1475)
    # shape of a single module
    mshape = (195, 487)
    # width of dead column strip
    dcw = 7
    # width of dead row strip
    drw = 17
    # start indices of dead strips
    colstarts = [487, 981]
    rowstarts = range(195, 1679, 212)

    dcols = reduce(lambda a,b: a + b, [ range(s, s+dcw) for s in colstarts])
    drows = reduce(lambda a,b: a + b, [ range(s, s+drw) for s in rowstarts])

    mask = np.ones(pshape, dtype=np.int8)
    mask[drows,:] = 0
    mask[:,dcols] = 0

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


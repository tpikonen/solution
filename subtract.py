import sys, glob, os.path, logging
import numpy as np
from optparse import OptionParser
from scipy.io.matlab import savemat, loadmat
from xformats.yamlformats import read_ydat, read_yaml, write_ydat
from xformats.matformats import read_matclean
from biosxs_reduce import md5_file, clean_indices

description="""\
Subtract backgrounds and normalize to concentration.
"""

usage="%prog scans.yaml"

default_outdir = "./stacks"
default_indir = "./stacks"


def get_bg(indir, scanno, posno):
    """Read background SAXS curve from an ydat-file.
    """
    fname = indir + "/bufs%03d.p%02d.out.ydat" % (scanno, posno)
    return read_ydat(fname)


def errsubtract(a, b, bscale=1.0):
    """Return `a` - `b` with errors, where `a` and `b` are (3, N) arrays.

    Optionally, scale I and Ierr of `b` with `bscale` before subtracting.
    Input values should have the same q-scale (this is not checked).
    """
    nn = clean_indices(a, b)
    retval = np.zeros((3, a.shape[1]))
    retval[0,:] = a[0,:]
    retval[1,nn] = a[1,nn] - bscale*b[1,nn]
    retval[2,nn] = np.sqrt(np.square(a[2,nn]) + np.square(bscale*b[2,nn]))
    retval[1:3,np.logical_not(nn)] = np.nan
    return retval


def subtract_background_from_stacks(scanfile, indir, outdir, scannumber=-1):
    """Subtract background from SAXS data in MAT-file stacks.
    """
    scans = read_yaml(scanfile)
    if scannumber > 0:
        scannos = [ scannumber ]
    else:
        scannos = scans.keys()
        scannos.sort()
    for scanno in scannos:
        print("Scan #%03d" % scanno)
        try:
            bufscan = scans[scanno][0]
        except TypeError:
            print("Scan #%03d is a buffer" % scanno)
            continue
        try:
            conc = scans[scanno][1]
        except TypeError:
            print("No concentration for scan #02d." % scanno)
            conc = 1.0
        print("Using concentration %g g/l." % conc)
        stackname = "s%03d" % scanno
        stack = loadmat(indir+'/'+stackname+'.mat')[stackname]
        subs = np.zeros_like(stack)
        (npos, nrep, _, _) = stack.shape
        for pos in range(npos):
            print(pos)
            buf = get_bg(indir, bufscan, pos)
            for rep in range(nrep):
                subs[pos,rep,...] = errsubtract(stack[pos,rep,...], buf)
                subs[pos,rep,1:3,:] = subs[pos,rep,1:3,:] / conc
        outname = "subs%03d" % scanno
        savemat(outdir+'/'+outname + ".mat", {outname: subs}, do_compression=1)


def excess_ratio(scanfile, qrange=[4.0, 5.0], cnorm=True):
    """Return ratio of (sam/buf)-1 in subtractions in the `qrange` given.

    Subtractions are made from data given in `scanfile` as in
    'subtract_background_from_ydats()', but only the ratios of
    sample / buffer intensity are returned in a list of arrays.

    If `cnorm` is True (default) then the ratio is normalized to the
    concentration read from the scanfile.

    Results from this function can be used to calibrate high-q normalized
    subtraction.
    """
    scans = read_yaml(scanfile)
    scannos = scans.keys()
    scannos.sort()
    indir = '.'
    mlist = []
    indlist = []
    clist = []
    for scanno in scannos:
        try:
            bufscan = scans[scanno][0]
        except TypeError:
            logging.warning("Scan #%03d is a buffer" % scanno)
            continue
        try:
            conc = scans[scanno][1]
        except TypeError:
            print("No concentration for scan #02d." % scanno)
            raise TypeError()
        logging.warning("Scan #%03d" % scanno)
        filelist = glob.glob(indir+"/s%03d.*.fil.ydat" % scanno)
        marr= np.zeros((len(filelist)))
        for posno in xrange(len(filelist)):
            bufname = indir + "/bufs%03d.p%02d.out.ydat" % (bufscan, posno)
            buf, dbuf = read_ydat(bufname, addict=1)
            fname = indir + "/s%03d.p%02d.fil.ydat" % (scanno, posno)
            sam, dsam = read_ydat(fname, addict=1)
            # Assumes the standard q, I, Ierr ordering in index 0 columns
            q = sam[0,:]
            qind = np.logical_and(q > qrange[0], q < qrange[1])
            ratio = np.mean(sam[1,qind]) / np.mean(buf[1,qind]) - 1.0
            if cnorm:
                ratio = ratio / conc
            marr[posno] = ratio
        mlist.append(marr)
        indlist.append(scanno)
        clist.append(conc)
    return mlist, indlist, clist


def subtract_background_from_ydats(scanfile, indir, outdir, scannumber=-1):
    """Subtract backround from SAXS data in .ydat files.
    """
    scans = read_yaml(scanfile)
    if scannumber > 0:
        scannos = [ scannumber ]
    else:
        scannos = scans.keys()
        scannos.sort()
    for scanno in scannos:
        print("Scan #%03d" % scanno)
        try:
            bufscan = scans[scanno][0]
        except TypeError:
            print("Scan #%03d is a buffer" % scanno)
            continue
        try:
            conc = scans[scanno][1]
        except TypeError:
            print("No concentration for scan #02d." % scanno)
            conc = 1.0
        print("Using concentration %g g/l." % conc)
        filelist = glob.glob(indir+"/s%03d.*.fil.ydat" % scanno)
        for posno in xrange(len(filelist)):
            bufname = indir + "/bufs%03d.p%02d.out.ydat" % (bufscan, posno)
            buf, dbuf = read_ydat(bufname, addict=1)
            fname = indir + "/s%03d.p%02d.fil.ydat" % (scanno, posno)
            sam, dsam = read_ydat(fname, addict=1)
            outname = os.path.basename(fname)
            outname = outdir+'/'+outname[:outname.find('.fil.ydat')]+'.sub.ydat'
            # Assumes the standard q, I, Ierr ordering in index 0 columns
            sub = errsubtract(sam, buf)
            sub[1:3,:] = sub[1:3,:] / conc
            ad = {
                'samfile': [os.path.basename(fname), md5_file(fname)],
                'buffile': [os.path.basename(bufname), md5_file(bufname)],
                'position' : dsam.get('inputposition', "unknown"),
                'q~unit' : dsam.get('q~unit', "unknown"),
                'I~unit' : dsam.get('I~unit', "unknown"),
                'Ierr~unit' : dsam.get('Ierr~unit', "unknown"),
                }
            write_ydat(sub, outname, addict=ad, attributes=['~unit'])
            print(os.path.basename(outname))


def main():
    oprs = OptionParser(usage=usage, description=description)
    oprs.add_option("-o", "--outdir",
        action="store", type="string", dest="outdir", default=default_outdir,
        help="Output directory. Default is '%s'" % default_outdir)
    oprs.add_option("-i", "--indir",
        action="store", type="string", dest="indir", default=default_indir,
        help="Output directory. Default is '%s'" % default_outdir)
    oprs.add_option("-m", "--matfiles",
        action="store_true", dest="matfiles", default=False,
        help="Subtract from and save to MAT-file stacks.")
    oprs.add_option("-s", "--scannumber",
        action="store", type="int", dest="scannumber", default=-1,
        help="Only process the scan number given here.")
    (opts, args) = oprs.parse_args()
    if len(args) < 1:
        oprs.error("Scanfile argument required")
    scanfile = args[0]
    if opts.matfiles:
        subtract_background_from_stacks(scanfile, opts.indir, opts.outdir, opts.scannumber)
    else:
        subtract_background_from_ydats(scanfile, opts.indir, opts.outdir, opts.scannumber)

if __name__ == "__main__":
    main()

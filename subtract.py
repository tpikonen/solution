import sys, glob, os.path
import numpy as np
from optparse import OptionParser
from scipy.io.matlab import savemat, loadmat
from xformats.yamlformats import read_ydat, read_yaml, write_ydat
from xformats.matformats import read_matclean
from biosxs_reduce import read_experiment_conf, errsubtract, md5_file

description="""\
Subtract backgrounds and normalize to concentration.
"""

usage="%prog scans.yaml"

default_conf = "experiment_conf.yaml"
default_modulus = 10
default_outdir = "./stacks"
default_indir = "./stacks"


def get_bg(indir, scanno, posno):
    """Read background SAXS curve from an ydat-file.
    """
    fname = indir + "/bufs%02d.p%02d.out.ydat" % (scanno, posno)
    return read_ydat(fname)


def subtract_background_from_stacks(scanfile, conffile, indir, outdir):
    """Subtract background from SAXS data in MAT-file stacks.
    """
    scans = read_yaml(scanfile)
    conf = read_experiment_conf(conffile)
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
        stackname = "s%02d" % scanno
#        bstackname = "bufs%02d" % bufscan
        stack = loadmat(indir+'/'+stackname+'.mat')[stackname]
#        bufstack = loadmat(indir+'/'+bstackname+'.mat')[bstackname]
        subs = np.zeros_like(stack)
        (npos, nrep, _, _) = stack.shape
        for pos in range(npos):
            print(pos)
            buf = get_bg(indir, bufscan, pos)
#            buf = bufstack[pos]
            for rep in range(nrep):
                subs[pos,rep,...] = errsubtract(stack[pos,rep,...], buf)
                subs[pos,rep,1:3,:] = subs[pos,rep,1:3,:] / conc
        outname = "subs%02d" % scanno
        savemat(outdir+'/'+outname + ".mat", {outname: subs}, do_compression=1)


def subtract_background_from_ydats(scanfile, conffile, indir, outdir):
    """Subtract backround from SAXS data in .ydat files.
    """
    scans = read_yaml(scanfile)
    conf = read_experiment_conf(conffile)
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
        filelist = glob.glob(indir+"/s%02d.*.fil.ydat" % scanno)
        for posno in xrange(len(filelist)):
            bufname = indir + "/bufs%02d.p%02d.out.ydat" % (bufscan, posno)
            buf, dbuf = read_ydat(bufname, addict=1)
            fname = indir + "/s%02d.p%02d.fil.ydat" % (scanno, posno)
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
    oprs.add_option("-c", "--conffile",
        action="store", type="string", dest="conf", default=default_conf,
        help="Configuration file to use. Default is '%s'" % default_conf)
    oprs.add_option("-o", "--outdir",
        action="store", type="string", dest="outdir", default=default_outdir,
        help="Output directory. Default is '%s'" % default_outdir)
    oprs.add_option("-i", "--indir",
        action="store", type="string", dest="indir", default=default_indir,
        help="Output directory. Default is '%s'" % default_outdir)
    oprs.add_option("-m", "--matfiles",
        action="store_true", dest="matfiles", default=False,
        help="Subtract from and save to MAT-file stacks.")
    (opts, args) = oprs.parse_args()
    if len(args) < 1:
        oprs.error("Scanfile argument required")
    scanfile = args[0]
    if opts.matfiles:
        subtract_background_from_stacks(scanfile, opts.conf, opts.indir, opts.outdir)
    else:
        subtract_background_from_ydats(scanfile, opts.conf, opts.indir, opts.outdir)

if __name__ == "__main__":
    main()


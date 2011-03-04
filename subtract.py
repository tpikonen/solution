import sys
import numpy as np
from optparse import OptionParser
from scipy.io.matlab import savemat, loadmat
from xformats.yamlformats import read_ydat, read_yaml
from xformats.matformats import read_matclean
from biosxs_reduce import read_experiment_conf, errsubtract

description="""\
Subtract backgrounds from stacks.
"""

usage="%prog scans.yaml"

default_conf = "experiment_conf.yaml"
default_modulus = 10
default_outdir = "./stacks"
default_indir = "./stacks"


def get_buf(indir, scanno, posno):
    """Read buffer SAXS curve from an ydat-file.
    """
    fname = indir + "/bufs%02d_p%02d.yfil" % (scanno, posno)
    return read_ydat(fname)


def subtract_background(scanfile, conffile, indir, outdir):
    """Subtract buffers from SAXS curves in stacks.
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
            buf = get_buf(indir, bufscan, pos)
#            buf = bufstack[pos]
            for rep in range(nrep):
                subs[pos,rep,...] = errsubtract(stack[pos,rep,...], buf)
                subs[pos,rep,1:3,:] = subs[pos,rep,1:3,:] / conc
        outname = "subs%02d" % scanno
        savemat(outdir+'/'+outname + ".mat", {outname: subs}, do_compression=1)


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
    (opts, args) = oprs.parse_args()
    if len(args) < 1:
        oprs.error("Scanfile argument required")
    scanfile = args[0]
    subtract_background(scanfile, opts.conf, opts.indir, opts.outdir)

if __name__ == "__main__":
    main()


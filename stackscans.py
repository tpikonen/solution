import optparse
from utils import *

description="""\
Write MAT-file stacks of scans given in input file, grouped by position.
"""

usage="%prog scans.yaml"


def main():
    default_conf = "experiment_conf.yaml"
    default_modulus = 10
    default_outdir = "./stacks"
    oprs = optparse.OptionParser(usage=usage, description=description)
    oprs.add_option("-c", "--conffile",
        action="store", type="string", dest="conf", default=default_conf,
        help="Configuration file to use. Default is '%s'" % default_conf)
    oprs.add_option("-o", "--outdir",
        action="store", type="string", dest="outdir", default=default_outdir,
        help="Output directory. Default is '%s'" % default_outdir)
    oprs.add_option("-m", "--modulus",
        action="store", type="int", dest="modulus", default=default_modulus,
        help="Modulus, i.e. number of positions in the scan. Default %d" \
            % default_modulus)
    oprs.add_option("-s", "--scannumber",
        action="store", type="int", dest="scannumber", default=-1,
        help="Only process the scan number given here.")
    (opts, args) = oprs.parse_args()
    scanfile = None
    if opts.scannumber < 0:
        if len(args) < 1:
            oprs.error("Scanfile argument required")
        else:
            scanfile = args[0]
    stack_files(scanfile, opts.conf, opts.outdir, opts.modulus, scannumber=opts.scannumber)


if __name__ == "__main__":
    main()

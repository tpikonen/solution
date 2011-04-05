import optparse
from biosxs_reduce import *

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
    (opts, args) = oprs.parse_args()
    if len(args) < 1:
        oprs.error("Scanfile argument required")
    scanfile = args[0]
    stack_files(scanfile, opts.conf, opts.outdir, opts.modulus)


if __name__ == "__main__":
    main()

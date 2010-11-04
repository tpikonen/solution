from optparse import OptionParser
from filter_repetitions import *

description="Plot a result from filtering and averaging repetitions."

usage="%prog <file1.yfil>"


def main():
    oprs = OptionParser(usage=usage, description=description)
    oprs.add_option("-n", "--no-errors",
        action="store_false", dest="err", default=True,
        help="Do not plot errorbands")
    oprs.add_option("-r", "--raw-errors",
        action="store_false", dest="smerr", default=True,
        help="Do not smooth errorbands")

    (opts, args) = oprs.parse_args()
    if(len(args) < 1):
        oprs.error("Input file argument required.")
    if(len(args) > 1):
        oprs.error("Only one input file accepted.")

    (dat, inds, first) = read_filtered(args[0])
    plot_filtered(dat, inds, first, 1, opts.err, opts.smerr)


if __name__ == "__main__":
    main()


import numpy as np
from optparse import OptionParser
from biosxs_reduce import mean_stack
from outliers import *

description="Average repeated buffer measurements by filtering outliers."

usage="%prog stack.mat [ stack2.mat ... ]"

#default_P = 0.005 # 1/200 = abt. one outlier in 20 reps
default_P = 0.001

def main():
    oprs = OptionParser(usage=usage, description=description)
    oprs.add_option("-P", "--rejection-probability",
        action="store", type="float", dest="p_reject", default=default_P,
        help="Statistical rejection probability for repetitions drawn from the same distribution. Default is %3g" % default_P)
    oprs.add_option("-n", "--noplot",
        action="store_false", dest="plot", default=True,
        help="Do not plot results for each repetition.")
    (opts, args) = oprs.parse_args()

    if(len(args) < 1):
        oprs.error("One or more input files are required.")

    for fname in args:
        outstem = "buf%s_p" % fname[:(fname.find('.mat'))]
        filter_matfile(fname, outstem, p_reject=opts.p_reject, plot=opts.plot)

if __name__ == "__main__":
    main()


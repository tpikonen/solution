import os.path
import numpy as np
from scipy.io.matlab import *


def read_matclean(fname, **kwargs):
    """Read a struct from a mat-file to a cleaned dict.

    Skips '__version__', '__globals__' and '__header__'.
    See scipy.io.loadmat for keyword arguments.
    """
    d = loadmat(fname, struct_as_record=1, squeeze_me=1, chars_as_strings=1, **kwargs)
    d.pop('__version__')
    d.pop('__header__')
    d.pop('__globals__')
    outd = {}
    for varname in d.keys():
        rec = d[varname]
        dout = {}
        for kk in rec.dtype.fields.keys():
            dout[kk] = rec[kk][()]
        outd.update({varname : dout})
    return outd


def read_mat(fname, **kwargs):
    """Return a single variable read from a mat-file.

    Use read_matclean() if the mat-file contains more than one variable.

    Does not do squeezing.

    See scipy.io.loadmat for keyword arguments.
    """
    d = loadmat(fname, **kwargs)
    d.pop('__version__')
    d.pop('__header__')
    d.pop('__globals__')
    kk = d.keys()
    if len(kk) > 1:
        raise ValueError("More than one variable in MAT-file.")
    return d[kk[0]]


def write_mat(varname, fname=None, value=None, overwrite=False):
    """Write a Numpy array to a MAT-file.

    `varname` gives the variable name inside the MAT-file dictionary.

    If `fname` is not given, the output file name is formed by appending
    '.mat' to `varname`.

    `value` is the value to be written.

    This function does not overwrite existing files, unless `overwrite`
    keyword argument is True.
    """
    if value is None:
        raise ValueError("No value given")
    if fname is None:
        fname = varname + '.mat'
    if os.path.isfile(fname) and not overwrite:
        raise IOError("File '%s' exists. Give 'overwrite=True' argument to overwrite." % fname)
    savemat(fname, {varname: value}, do_compression=1, oned_as='row')

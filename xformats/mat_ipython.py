import os.path
import numpy as np
from scipy.io.matlab import *
from solution.xformats.matformats import *


def read_matdict(fname, **kwargs):
    """Read a struct from a mat-file to a dict in ipython interactive namespace.

    Skips '__version__', '__globals__' and '__header__'.
    See scipy.io.loadmat for keyword arguments.
    """
    d = read_matclean(fname)
    ip = get_ipython()
    ip.user_ns.update(d)
    for varname in d.keys():
        print(varname)


def read_matfile(fname, **kwargs):
    """Read variables from a mat-file to ipython interactive namespace.

    Does not do squeezing.

    Skips '__version__', '__globals__' and '__header__'.
    See scipy.io.loadmat for keyword arguments.
    """
    d = loadmat(fname, **kwargs)
    d.pop('__version__')
    d.pop('__header__')
    d.pop('__globals__')
    ip = get_ipython()
    ip.user_ns.update(d)
    for varname in d.keys():
        print(varname)


def read_matvars(fname, **kwargs):
    """Read variables from a mat-file to ipython interactive namespace.

    Skips '__version__', '__globals__' and '__header__'.
    See scipy.io.loadmat for keyword arguments.
    """
    d = loadmat(fname, struct_as_record=1, squeeze_me=1, chars_as_strings=1, **kwargs)
    d.pop('__version__')
    d.pop('__header__')
    d.pop('__globals__')
    ip = get_ipython()
    ip.user_ns.update(d)


def write_matvars(variables, fname, overwrite=False):
    """Write ipython interactive variables to a MAT-file.

    `variables` gives a list of variable names in the ipython interactive
    namespace.

    `fname` gives the output file name.

    This function does not overwrite existing files, unless `overwrite`
    keyword argument is True.
    """
    ip = get_ipython()
    outdic = {}
    for vv in variables:
        try:
            outdic[vv] = ip.user_ns[vv]
        except KeyError:
            raise ValueError("Variable '%s' not found in IPython namespace." % vv)
    if os.path.isfile(fname) and not overwrite:
        raise IOError("File '%s' exists. Give 'overwrite=True' argument to overwrite." % fname)
    savemat(fname, outdic, do_compression=1, oned_as='row')

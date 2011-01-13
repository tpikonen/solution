import IPython.ipapi, os.path
import numpy as np
from scipy.io.matlab import *

def read_matdict(fname, **kwargs):
    """Read a struct from a mat-file to a dict in ipython interactive namespace.

    Skips '__version__', '__globals__' and '__header__'.
    See scipy.io.loadmat for keyword arguments.
    """
    d = loadmat(fname, struct_as_record=1, squeeze_me=1, chars_as_strings=1, **kwargs)
    d.pop('__version__')
    d.pop('__header__')
    d.pop('__globals__')
    api = IPython.ipapi.get()
    for varname in d.keys():
        rec = d[varname]
        dout = {}
        for kk in rec.dtype.fields.keys():
            dout[kk] = rec[kk][()]
        api.to_user_ns({varname : dout})
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
    api = IPython.ipapi.get()
    api.to_user_ns(d)
    for vname in d.keys():
        print(vname)


def write_mat(varname, fname=None, value=None, overwrite=False):
    """Write a Numpy array to a MAT-file.

    `varname` gives the variable name inside the MAT-file dictionary.

    If `fname` is not given, the output file name is formed by appending
    '.mat' to `varname`.

    If `value` is not given, the value of variable `varname` in the IPython
    namespace is used, if found.

    This function does not overwrite existing files, unless `overwrite`
    keyword argument is True.
    """
    if value is None:
        api = IPython.ipapi.get()
        try:
            value = api.user_ns[varname]
        except KeyError:
            raise ValueError("Variable '%s' not found in IPython namespace." % varname)
#    if isinstance(value, np.ndarray):
#        raise ValueError("value must be a NumPy array.")
    if fname is None:
        fname = varname + '.mat'
    if os.path.isfile(fname) and not overwrite:
        raise IOError("File '%s' exists. Give 'overwrite=True' argument to overwrite." % fname)
    savemat(fname, {varname: value}, do_compression=1, oned_as='row')

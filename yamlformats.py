from __future__ import with_statement
import numpy as np
import yaml, loopyaml, logging


def read_yaml(fname):
    """Read a YAML structure from a given file and return it."""
    with open(fname, 'r') as f:
        d = yaml.load(f)
    return d


def write_yaml(o, fname, **kwargs):
    """Write a given object to a YAML file fname."""
    with open(fname, 'w') as f:
        loopyaml.dump(o, f, **kwargs)


def read_ydat(fname):
    """Return a (N, ncols) numpy array read from a 'YAML-dat' (or ydat) file.
    """
    yd = read_yaml(fname)
    ncols = len(yd['=cols'])
    P = len(yd['=loop'])
    assert((P % ncols ) == 0)
    yarr = np.array(yd['=loop']).reshape((P / ncols, ncols)).T
    # FIXME: Assumes that the columns are in a 'canonical' order [q, I, Ierr]
    return yarr.squeeze()


def write_ydat(arr, fname, cols=['q', 'I', 'Ierr'], addict=None):
    """Write a rank-2 array `arr` to YAML-dat file `fname`.

    The first index of the array is assumed to be columns and the second rows,
    but if there are more than 3 columns and 3 or less rows, the array is
    transposed before writing.

    Keyword argument `cols` must contain a list of names for each column.
    Keyword argument `addict` can contain an additional dictionary which is
        added to the output yaml dictionary.
    """
    if arr.shape[0] != len(cols) and arr.shape[0] > 3 and arr.shape[1] <= 3:
        arr = arr.T
    if arr.shape[0] > len(cols):
        raise ValueError("Not enough column names given")
    if addict is None:
        outdic = {}
    else:
        outdic = addict
    for i in range(arr.shape[0]):
        outdic[cols[i]] = map(float, list(arr[i, :]))
    ld = loopyaml.Loopdict(outdic, loopvars=cols[:arr.shape[0]])
    write_yaml(ld, fname)

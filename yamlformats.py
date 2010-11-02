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


def write_ydat(arr, fname, cols=['q', 'I', 'Ierr'], addict={}):
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
    outdic = addict
    for i in range(arr.shape[0]):
        outdic[cols[i]] = map(float, list(arr[i, :]))
    ld = loopyaml.Loopdict(outdic, loopvars=cols[:arr.shape[0]])
    write_yaml(ld, fname)


def write_ystack(stack, fname, addict={}):
    """Write a rank-3 array `stack` to YAML-dat file `fname`.

    The shape of `stack` should be (reps, q/I/Ierr, val).

    Keyword argument `addict` can contain an additional dictionary which is
        added to the output yaml dictionary.
    """
    if (stack[:,0,:] - stack[0,0,:]).any():
        raise ValueError("q-scales must be identical.")
    outdic = addict
    outdic['q'] = map(float, list(stack[0,0,:]))
    cols = ['q']
    for i in range(stack.shape[0]):
        Ikey = 'I_%d' % i
        Ierrkey = 'Ierr_%d' % i
        outdic[Ikey] = map(float, list(stack[i, 1, :]))
        cols.append(Ikey)
        outdic[Ierrkey] = map(float, list(stack[i, 2, :]))
        cols.append(Ierrkey)
    print(cols)
    ld = loopyaml.Loopdict(outdic, loopvars=cols)
    write_yaml(ld, fname)

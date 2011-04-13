import numpy as np
from xformats.atsasformats import read_dat
from xformats.yamlformats import read_loopyaml


def read_sasdata(fname):
    """Return a Numpy data array read either from .dat or .ydat file.

    Returns a (dat, qunit, Iunit) tuple.
    """
    Iunit = 'unknown'
    qunit = 'unknown'
    if fname.endswith(('.yaml', '.ydat', '.yfil')):
        yd = read_loopyaml(fname)
        if yd.has_key('Ierr'):
            dat = np.zeros((len(yd['Ierr']), 3))
            dat[:,2] = np.array(yd['Ierr'])
        else:
            dat = np.zeros((2,len(yd['q'])))
        dat[:,0] = np.array(yd['q'])
        dat[:,1] = np.array(yd['I'])
        Iunit = yd.get('I~unit', Iunit)
        qunit = yd.get('q~unit', qunit)
    else:
        dat = read_dat(fname).T
    return dat, qunit, Iunit

import numpy as np
from atsasformats import read_dat
from yamlformats import read_loopyaml


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


def maybe_uncompress(filename):
    """Uncompress a file if it's compressed with typical compression programs.

    Return a (fname, is_temporary) tuple, where `fname` is the name
    of the file which is guaranteed to be uncompressed, and `is_temporary`
    is a boolean indicating whether the file is temporarary, and should be
    deleted after reading.
    """
    import tempfile

    def write_to_temp(fin):
        tf = tempfile.NamedTemporaryFile(suffix=".read_cbf", delete=False)
        tf.file.write(fin.read())
        fin.close()
        tf.file.close()
        return tf.name

    is_temporary = False
    # FIXME: Use magic to detect file type.
    if filename.endswith(".bz2"):
        import bz2
        fin = bz2.BZ2File(filename, mode='r')
        fname = write_to_temp(fin)
        is_temporary = True
    elif filename.endswith(".gz"):
        import gzip
        fin = gzip.GzipFile(filename, mode='r')
        fname = write_to_temp(fin)
        is_temporary = True
    else:
        fname = filename

    return (fname, is_temporary)

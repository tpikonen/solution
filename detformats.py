from __future__ import with_statement
import numpy as np

class det_struct:
    __doc__ = """Pseudo-structure for detector frame data.

    Known fields:
    a.im     = ccd counts
    a.xo     = beam center in x-dir
    a.yo     = beam center in y-dir
    a.int0   = beam intensity before sample
    a.int1   = beam intensity after sample
    a.trans  = transmission of the sample (int1 / int0)
    a.wavel = x-ray wavelength (in m)
    a.sdist  = sample-detector distance (in m)
    a.time   = exposure time
    a.q      = q-scale (in 1/nm)
    """
    pass


def read_tif_pilatus(fname):
    """Read a .tif file from a Pilatus detector."""
    headerlen = 580
    datastart = 4096
    import matplotlib.image, re
    # PIL is teh suxor
    junk = matplotlib.image.imread(fname)
    pshape = junk.shape[0:2]
    bytelen = 4*np.prod(pshape)
    del junk
    fin = open(fname)
    fin.seek(headerlen)
    hdr = fin.read(datastart - headerlen)
    m = re.search('# Exposure_time ([0-9.]*) s', hdr)
    d = det_struct()
    d.time = float(m.group(1))
    fin.seek(4096)
    blob = fin.read(bytelen)
    if len(blob) != bytelen:
        raise ValueError("Could not read all %d bytes." % bytelen)
    im = np.fromstring(blob, dtype=np.uint32)
    im = im.reshape(pshape)
    d.im = im
    return d


def read_spec(fname):
    """Read a spec scan file to a dictionary."""
    import specparser as sp
    fin = open(fname)
    p = sp.specparser(fin)
    d = p.parse()
    fin.close()
    return d


def read_mar345hdr(fname):
    """Read fields and values from a MAR file to a dictionary.
    Multiple values (like in the REMARK field are not supported,
    only the last read value remains."""

    fid = open(fname)
    done = False
    # discard the first line
    line = fid.readline()
    d = {}
    while not done:
        line = fid.readline()
        m = re.search('^([A-Z0-9]*) *(.*) *$', line)
        if m.group(0) == 'END OF HEADER':
            done = True
        elif m.group(1) != '':
            d[m.group(1)] = m.group(2)
    return d


def read_mar345im(fname):
    import subprocess, re, os
    TMPDIR = "/tmp"
    outfile = TMPDIR + "/" + re.sub('\.mar2300$', '.raw32_2300', fname)
    cmdlst = ["marcvt",  "-raw32", "-o", TMPDIR, fname]
    try:
        nullfd = open("/dev/null")
        subprocess.check_call(cmdlst, stdout=nullfd)
        fd = open(outfile)
        fd.seek(0, os.SEEK_END)
        sz = fd.tell()
        fd.close()
        dim = np.sqrt(sz/4.0)
        if dim != np.floor(dim):
            raise ValueError
        mm = np.memmap(outfile, dtype='uint32', mode='r', shape=(dim,dim))
        a = np.array(mm)
        del mm
    finally:
        subprocess.check_call(["rm", outfile], stdout=nullfd)
        nullfd.close()
    return a


def read_specscan(fname, scanno):
    """Return a given scan from specfile as a Numpy array."""
    import PyMca.specfile
    sf = PyMca.specfile.Specfile(fname)
    sd = sf.select(str(scanno))
    return sd.data()


def read_id2(fname):
    """
    a=read_id2(filename)

    In:
        filename = name of the id2 ccd file (EDF)
    Out:
        a.im     = ccd counts
        a.xo     = beam center in x-dir
        a.yo     = beam center in y-dir
        a.int0   = beam intensity before sample
        a.int1   = beam intensity after sample
        a.trans  = transmission of the sample (int1 / int0)
        a.wavel = x-ray wavelength (in m)
        a.sdist  = sample-detector distance (in m)
        a.time   = exposure time
        a.q      = q-scale (in 1/nm)
    """
    import PyMca.specfile
    import PyMca.EdfFile
    edf=PyMca.EdfFile.EdfFile(fname)
    hdr = edf.GetHeader(0)
    getfl = lambda key: float(hdr[key].split()[0])
    d = det_struct()
    d.header = hdr
    d.im = edf.GetData(0)
    d.jorma = 4.0
    d.xo = getfl("Center_1")
    d.yo = getfl("Center_2")
    d.int0 = getfl("Intensity0")
    d.int1 = getfl("Intensity1")
    d.trans = (d.int1 / d.int0)
    d.wavel = getfl("WaveLength")
    d.time = getfl("ExposureTime")
    d.psize1 = getfl("PSize_1")
    d.psize2 = getfl("PSize_2")
    d.sdist = getfl("SampleDistance")
    d.title = hdr["Title"]
    psize = (d.psize1 + d.psize2) / 2
    d.q = 4*np.pi*np.sin(0.5*np.arctan(psize*np.arange(0.0,1024.0,1.0)/d.sdist) ) / (d.wavel/1e-9);
    return d


def read_mask(fname):
    """Read a mask (Numpy bool array) from an image file"""
    import matplotlib.image
    mfloat = matplotlib.image.imread(fname)
    if len(mfloat.shape) == 2:
        mask = (mfloat[:,:] != 0.0)
    else:
        mask = (mfloat[:,:,0] != 0.0)
    return mask


def write_pnglog(a, fname):
    """Take a logarithm from a 2D numpy array, scale the range to
    fit into a byte and write it to a PNG file.
    """
    import PIL.Image
    bscale = np.log(abs(a)+1)
    bint = ((255.0/bscale.max())*bscale).astype('uint8')
    pim = PIL.Image.fromarray(bint)
    fp = open(fname, 'w');
    pim.save(fp, 'png')
    fp.close()


def read_cbf(fname, have_pycbf=1):
    """Read CBFs (Crystallographic Binary File) output from the
    Pilatus detector at cSAXS.
    """
    # TI 2010-03-03
    import re

    MAXHEADER = 4096
    CBF_SIGNATURE = "###CBF: VERSION"
    BINARY_SIGNATURE = '\x0c\x1a\x04\xd5'
    fid = open(fname)
    hdr = fid.read(MAXHEADER)
    if(hdr[0:15] != "###CBF: VERSION"):
        print("Warning: CBF header not present, trying to read anyway")

    d = det_struct()

    m = re.search('# Exposure_time ([0-9.]*) s', hdr)
    d.time = float(m.group(1))

    m = re.search('conversions="([^"]*)"', hdr)
    compression = m.group(1)
    if compression != "x-CBF_BYTE_OFFSET":
        raise ValueError("Compression types other than x-CBF_BYTE_OFFSET are not supported")

    m = re.search('X-Binary-Size: ([0-9]*)', hdr)
    bsize = int(m.group(1))

    m = re.search('X-Binary-Element-Type: "([^"]*)"', hdr)
    eltype = m.group(1)
    if eltype != "signed 32-bit integer":
        raise ValueError("Values types other than signed 32-bit integer are not supported")

    m = re.search('X-Binary-Size-Fastest-Dimension: ([0-9]*)', hdr)
    ydim = int(m.group(1))

    m = re.search('X-Binary-Size-Second-Dimension: ([0-9]*)', hdr)
    xdim = int(m.group(1))

    m = re.search('X-Binary-Element-Byte-Order: ([A-Z\_]*)', hdr)
    byteorder = m.group(1)
    if byteorder != "LITTLE_ENDIAN":
        raise ValueError("Byte orders other than little endian are not supported")

    m = re.search(BINARY_SIGNATURE, hdr)
    datastart = int(m.end(0))

    if have_pycbf:
        d.im = offset_decompress_pycbf(fname, [xdim, ydim])
    else:
        fid.seek(datastart)
        dstr = fid.read(bsize)
        fid.close()
        d.im = offset_decompress_python(dstr, [xdim, ydim])

    return d


def offset_decompress_pycbf(fname, dims):
    """Decompression of BYTE_OFFSET packing, CBFlib implementation"""
    import pycbf
    h = pycbf.cbf_handle_struct()
    h.read_file(fname,pycbf.MSG_DIGEST)
    h.select_datablock(0)
    h.select_category(0)
    h.select_column(2)
    if h.get_typeofvalue().find("bnry") < 0:
        raise ValueError("CBF Binary block not in the usual place")
    s = h.get_integerarray_as_string()
    d = np.fromstring(s, dtype='i4')
    return d.reshape(dims)


def offset_decompress_python(dstr, dims):
    """Decompression of BYTE_OFFSET packing, slow Python implementation"""
    import struct as st
    im = np.zeros(dims[0]*dims[1], dtype='i4')
    i = 0
    o = 0
    prev = 0
    while i < len(dstr):
        delta = st.unpack('<b', dstr[i])[0]
        i = i+1
        if delta == -128:
            delta = st.unpack('<h', dstr[i:i+2])[0]
            i = i+2
            if delta == -32768:
                delta = st.unpack('<i', dstr[i:i+4])[0]
                i = i+4
                if delta == -2147483648:
                    raise ValueError("Only up to 32-bit values supported")
        im[o] = prev + delta
        prev = im[o]
        o = o+1

    return im.reshape(dims[0], dims[1])

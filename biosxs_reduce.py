from __future__ import with_statement
import glob, re, os.path, sys, logging, hashlib
import loopyaml
import numpy as np
import radbin as r
from xformats.detformats import read_cbf, read_eiger, read_spec
from xformats.yamlformats import read_yaml, write_yaml, read_ydat, write_ydat
from xformats.atsasformats import read_dat
from xformats.matformats import read_matclean
from scipy.io.matlab import savemat, loadmat


def read_experiment_conf(fname):
    """Return a dictionary with values read from conffile.
    """
    yd = read_yaml(fname)
    c = {}
    c['Basedir'] = yd['basedir']
    c['Pilatusdir'] = c['Basedir'] + yd['pilatusdir']
    c['Eigerdir'] = c['Basedir'] + yd.get('eigerdir', '')
    c['Expno'] = int(yd['expno'])
    c['Detno'] = int(yd['detno'])
    c['Cbfext'] = yd['cbfext']
    c['Specfile'] = c['Basedir'] + yd['specfile']
    c['Indfile'] = c['Basedir'] + yd['indfile']
    return c


def clean_indices(x, y):
    """Return indices which contain only good floats in x and y.
    """
    nn = np.logical_not(np.logical_or(
        np.logical_or(np.isnan(x[1,:]), np.isnan(y[1,:])),
        np.logical_or((x[2,:] <= 0.0), (y[2,:] <= 0.0))))
    return nn


def chimodel(model, data):
    """Return the chi-squared between model (errors ignored) and data."""
    x = model
    y = data
    nn = clean_indices(x, y)
    N = np.sum(nn)
    chi2 = (1.0/N)*np.sum((x[1,nn]-y[1,nn])**2 / y[2,nn]**2)
    return chi2


def chivectors(x, y):
    """Return the chi-squared between two experimental data with errors."""
    nn = clean_indices(x, y)
    N = np.sum(nn)
    chi2 = (1.0/N)*np.sum((x[1,nn]-y[1,nn])**2 / (x[2,nn]**2+y[2,nn]**2))
    return chi2


def chi2cdm(X):
    """Return the pairwise chi-squared distance between observations in `X`.

    The return value is a condensed distance matrix. Use
    scipy.spatial.distance.squareform() to convert to a square distance
    matrix.
    """
    m = X.shape[0]
    dm = np.zeros((m * (m - 1) / 2,), dtype=np.double)
    k = 0
    for i in xrange(0, m - 1):
        for j in xrange(i+1, m):
            dm[k] = chivectors(X[i], X[j])
            k = k + 1

    return dm


def get_framefilename(conf, scanno, pointno, burstno):
    """Return the filename of a frame at a given scan, point, and burst number.
    """
    c = conf
    fname = "%s/S%05d/e%05d_%d_%05d_%05d_%05d.%s" % (c['Pilatusdir'], scanno, c['Expno'], c['Detno'], scanno, pointno, burstno, c['Cbfext'])
    return fname


def eiger_filename(conf, scanno, pointno, burstno):
    """Return the filename of a frame at a given scan, point, and burst number.
    """
    c = conf
    fname = "%s/S%05d/e%05d_%d_%05d_%05d_%05d.%s" % (c['Eigerdir'], scanno, c['Expno'], c['Detno'], scanno, pointno, burstno, "h5")
    return fname


def get_binned(indices, framefile):
    fr = read_cbf(framefile)
    stats = r.binstats(indices, fr.im, calculate=(1,0,0,1))
    return (stats[0], stats[3]) # Mean and Poisson count std of mean


def eiger_binned(indices, framefile):
    fr = read_eiger(framefile)
    stats = r.binstats(indices, fr.im, calculate=(1,0,1,1))
    return (stats[0], stats[2])


def get_diode(specscans, scanno, pointno):
    """Return diode counts from a given scan and point.

    Argument `specscans` is the list returned from Specparser.parse()['scans'].
    Scan number `scanno` is indexed starting from 1, as in SPEC.
    """
    scan = specscans[scanno]
    assert(scanno == scan['number'])
    diodeval = scan['counters']['diode'][pointno]
    return diodeval


def get_dsum(specscans, scanno, pointno):
    """Return dsum counts from a given scan and point.

    Argument `specscans` is the list returned from Specparser.parse()['scans'].
    Scan number `scanno` is indexed starting from 1, as in SPEC.
    """
    scan = specscans[scanno]
    assert(scanno == scan['number'])
    dval = scan['counters']['dSum'][pointno]
    return dval


def get_scanlen(specscans, scanno):
    """Return the number of points in `scanno`.

    Scan number `scanno` is indexed starting from 1, as in SPEC.
    """
    scan = specscans[scanno]
    assert(scanno == scan['number'])
    return scan['npoints']


def md5_file(fname):
    md5 = hashlib.md5()
    with open(fname) as f:
        md5.update(f.read())
        h = md5.hexdigest()
    return h


def errsubtract(a, b):
    """Return a - b with errors, where a and b are (3, N) arrays.

    Input values should have the same q-scale (this is not checked).
    """
    nn = clean_indices(a, b)
    retval = np.zeros((3, a.shape[1]))
    retval[0,:] = a[0,:]
    retval[1,nn] = a[1,nn] - b[1,nn]
    retval[2,nn] = np.sqrt(np.square(a[2,nn]) + np.square(b[2,nn]))
    retval[1:3,np.logical_not(nn)] = np.nan
    return retval


def stack_datafiles(fnames):
    """Return an array containing curves from the given list of files.

    Files can be in (.dat/.yaml/.ydat) format.
    """
    if fnames[0].endswith(".dat") or fnames[0].endswith(".fit"):
        read_func = read_dat
    else:
        read_func = read_ydat

    arr0 = read_func(fnames[0])
    outarr = np.zeros((len(fnames), arr0.shape[0], arr0.shape[1]))
    outarr[0,:,:] = arr0
    for i in range(1, len(fnames)):
        outarr[i,:,:] = read_func(fnames[i])
    return outarr


def mean_stack(stack):
    """Return the mean (and error) of a (M, 3, n) stack along 1st dimension.
    """
    ish = stack.shape
    assert(ish[1] >= 3)
    retval = np.zeros((3, ish[2]))
    retval[0,:] = stack[0,0,:]
    retval[1,:] = np.mean(stack[:,1,:], axis=0)
    retval[2,:] = np.sqrt(np.sum(np.square(stack[:,2,:]), axis=0))/ish[0]
    return retval


def sum_stack(stack):
    """Return the sum (and error) of (M, 3, n) stack along 1st dimension.
    """
    ish = stack.shape
    retval = np.zeros((ish[1], ish[2]))
    retval[0,:] = stack[0,0,:] # q
    retval[1,:] = np.sum(stack[:,1,:], axis=0)
    retval[2,:] = np.sqrt(np.sum(np.square(stack[:,2,:]), axis=0))
    return retval


def write_stack_ydat(fname, stack, fnames, dvals, conf):
    """Write a single position from a stack to an .ydat file.
    """
    sh = stack.shape
    outarr = np.zeros((2*sh[0]+1, sh[-1]))
    outarr[0,:] = stack[0,0,:] # q
    for pos in xrange(sh[0]):
        outarr[2*pos+1,:] = stack[pos,1,:] # I
        outarr[2*pos+2,:] = stack[pos,2,:] # Ierr
    ad = {  'frames': list(fnames),
            'transmissions': dvals,
            'indfile': [ os.path.basename(conf['Indfile']),
                        md5_file(conf['Indfile']) ],
            'q~unit': '1/nm',
        }
    cols = ['q']
    Icols = [ "I%02d" % n for n in range(len(fnames))]
    errcols = [ "Ierr%02d" % n for n in range(len(fnames))]
    cols.extend([ col for lsub in zip(Icols, errcols) for col in lsub ])
    ad.update([ ("I%02d~unit" % n, "arb.") for n in range(len(fnames)) ])
    ad.update([ ("Ierr%02d~unit" % n, "arb.") for n in range(len(fnames)) ])
    write_ydat(outarr, fname, cols=cols, addict=ad, attributes=['~unit'])


def stack_eiger(conf, scanno, specscans, radind, modulus=10):
    """Return an array with 1D curves in different positions in a scan.

    Argument `modulus` gives the number of unique positions in a scan.

    Return value is an array with coordinates [posno, repno, q/I/err, data]
    and shape (number_of_positions, number_of_repetitions, 3, len(q)).
    """
    q = radind['q']
    scanlen = get_scanlen(specscans, scanno)
    if (scanlen % modulus) != 0:
        raise ValueError\
            ("Number of points in a scan is not divisible by modulus.")

    numreps = scanlen / modulus
    stack = np.zeros((modulus, numreps, 3, len(q)))
    fnames = [ [] for x in range(modulus) ]

    for posno in range(modulus):
        print("scan #%d, pos %d" % (scanno, posno))
        sys.stdout.flush()
        repno = 0
        for pointno in range(posno, scanlen, modulus):
            # Normalize dSum to average of 1.
            dval = get_dsum(specscans, scanno, pointno) / 22450965
            frname = eiger_filename(conf, scanno, pointno, 0)
            (I, err) = eiger_binned(radind['indices'], frname)
            stack[posno, repno, 0, :] = q
            stack[posno, repno, 1, :] = I/dval
            stack[posno, repno, 2, :] = err/dval
            md5 = md5_file(frname)
            fnames[posno].append((frname, md5))
            repno = repno+1

    return stack, fnames


def stack_scan(conf, scanno, specscans, radind, modulus=10):
    """Return normalized 1D curves grouped by positions and repetitions.

    Argument `modulus` gives the number of unique positions in a scan.

    The intensity (and it's error) values are normalized by the 'diode'
    counter in `specscans`.

    Return value is an array with coordinates [posno, repno, q/I/err, data]
    and shape (number_of_positions, number_of_repetitions, 3, len(q)).
    """
    q = radind['q']
    scanlen = get_scanlen(specscans, scanno)
    if (scanlen % modulus) != 0:
        raise ValueError\
            ("Number of points in a scan is not divisible by modulus.")

    numreps = scanlen / modulus
    stack = np.zeros((modulus, numreps, 3, len(q)))
    fnames = [ [] for x in range(modulus) ]
    dvals = [ [] for x in range(modulus) ]

    for posno in range(modulus):
        print("scan #%d, pos %d" % (scanno, posno))
        sys.stdout.flush()
        repno = 0
        for pointno in range(posno, scanlen, modulus):
            # Normalize transmission to a reasonable value
            dval = get_diode(specscans, scanno, pointno) / 10000.0
            frname = get_framefilename(conf, scanno, pointno, 0)
            (I, err) = get_binned(radind['indices'], frname)
            stack[posno, repno, 0, :] = q
            stack[posno, repno, 1, :] = I/dval
            stack[posno, repno, 2, :] = err/dval
            md5 = md5_file(frname)
            fnames[posno].append([os.path.basename(frname), md5])
            dvals[posno].append(dval)
            repno = repno+1

    return stack, fnames, dvals


def stack_files(scanfile, conffile, outdir, modulus=10, eiger=0, matfile=1):
    """Create stacks from scans read from `scanfile` and write the to files.
    """
    if not os.path.isdir(outdir):
        # FIXME: Create the directory
        raise IOError("Output directory does not exist.")
    scans = read_yaml(scanfile)
    conf = read_experiment_conf(conffile)
    specscans = read_spec(conf['Specfile'])
    radind = read_matclean(conf['Indfile'])['radind']
    q = radind['q']
    scannos = scans.keys()
    scannos.sort()
    for scanno in scannos:
        outname = "s%03d" % scanno
        if eiger:
            stack, fnames, dvals = \
                stack_eiger(conf, scanno, specscans, radind, modulus)
        else:
            stack, fnames, dvals = \
                stack_scan(conf, scanno, specscans, radind, modulus)
        stack = stack.squeeze()
        if matfile:
            savemat(outdir+'/'+outname + ".mat", {outname: stack}, do_compression=1)
        else:
            for pos in range(stack.shape[0]):
                outfname = outname+'.p%02d.all.ydat' % pos
                write_stack_ydat(outdir+'/'+outfname, stack[pos], fnames[pos], dvals[pos], conf)
                print("Wrote output to '%s'." % outfname)

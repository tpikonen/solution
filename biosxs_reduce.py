from __future__ import with_statement
import glob, re, os.path, sys, logging, hashlib
import loopyaml
import numpy as np
import radbin as r
from detformats import read_cbf, read_spec
from yamlformats import read_yaml, write_yaml, read_ydat, write_ydat
from csaxsformats import read_pickle
from atsasformats import read_dat
from scipy.io.matlab import savemat, loadmat

#Basedir = '/afs/psi.ch/project/cxs/users/ikonen/ND-Rhod-2010-04-24/Data10/'
#Pilatusdir = Basedir + 'pilatus/S00000-00999/'
#Expno = 12594
#Detno = 1
#Specfile = Basedir + 'spec/dat-files/specES1_started_2010_05_22_1133.dat'

Basedir = None
Pilatusdir = None
Expno = None
Detno = None
Cbfext = None
Specfile = None
Indfile = None

def read_experiment_conf(fname):
    """Set global variables from conffile.
    """
    global Basedir
    global Pilatusdir
    global Expno
    global Detno
    global Cbfext
    global Specfile
    global Indfile

    yd = read_yaml(fname)
    Basedir = yd['basedir']
    Pilatusdir = Basedir + yd['pilatusdir']
    Expno = int(yd['expno'])
    Detno = int(yd['detno'])
    Cbfext = yd['cbfext']
    Specfile = Basedir + yd['specfile']
    Indfile = Basedir + yd['indfile']


def read_experiment_dict(fname):
    """Return a dictionary with values read from conffile.
    """
    yd = read_yaml(fname)
    c = {}
    c['Basedir'] = yd['basedir']
    c['Pilatusdir'] = c['Basedir'] + yd['pilatusdir']
    c['Expno'] = int(yd['expno'])
    c['Detno'] = int(yd['detno'])
    c['Cbfext'] = yd['cbfext']
    c['Specfile'] = c['Basedir'] + yd['specfile']
    c['Indfile'] = c['Basedir'] + yd['indfile']
    return c

#def read_frame(scanno, pointno, burstno):
#    """Return a frame from a given scan and point.
#    """
#    fname = get_framefilename(scanno, pointno, burstno)
#    frame = read_cbf(fname)
#    return frame


def get_framefilename(scanno, pointno, burstno):
    """Return the filename of a frame at a given scan, point, and burst number.
    """
    fname = "%s/S%05d/e%05d_%d_%05d_%05d_%05d.%s" % (Pilatusdir, scanno, Expno, Detno, scanno, pointno, burstno, Cbfext)
    return fname


def get_binned(indices, framefile):
    fr = read_cbf(framefile)
    stats = r.binstats(indices, fr.im, calculate=(1,0,0,1))
    return (stats[0], stats[3]) # Mean and Poisson count std of mean


def get_diode(spec, scanno, pointno):
    """Return diode counts from a given scan and point.

    Argument `spec` is the dict returned from Specparser.parse().
    Scan number `scanno` is indexed starting from 1, as in SPEC.
    """
    scan = spec['scans'][scanno-1]
    assert(scanno == scan['number'])
    diodeval = scan['counters']['diode'][pointno]
    return diodeval


def get_scanlen(spec, scanno):
    """Return the number of points in `scanno`.

    Scan number `scanno` is indexed starting from 1, as in SPEC.
    """
    scan = spec['scans'][scanno-1]
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
    retval = np.zeros_like(a)
    retval[0,:] = a[0,:]
    retval[1,nn] = a[1,nn] - b[1,nn]
    retval[2,nn] = np.sqrt(np.square(a[2,nn]) + np.square(b[2,nn]))
    return retval


def stack_ydat(stem, poslist, ending=".yaml"):
    """Return a stack of ydat-files in poslist starting with `stem`.
    """
    ll = []
    for p in poslist:
        fname = "%s%d%s" % (stem, p, ending)
        ll.append(read_ydat(fname))
    stack = np.dstack(ll).transpose((2,0,1))
    return stack


def mean_stack(stack):
    """Return the mean (and error) of a (M, 3, n) stack along 1st dimension.
    """
    ish = stack.shape
    retval = np.zeros((ish[1], ish[2]))
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


def clean_indices(x, y):
    """Return indices which contain only good floats in x and y.
    """
    nn = np.logical_not(np.logical_or(
        np.logical_or(np.isnan(x[1,:]), np.isnan(y[1,:])),
        np.logical_or((x[2,:] <= 0.0), (y[2,:] <= 0.0))))
    return nn


def chivectors(x, y):
    nn = clean_indices(x, y)
    N = np.sum(nn)
    chi2 = (1.0/N)*np.sum((x[1,nn]-y[1,nn])**2 / (x[2,nn]**2+y[2,nn]**2))
    return chi2


def chifilter(rowstack, cutoff=1.2):
    """Filter observations by comparing to the first one in stack.

    Returns a tuple (indices, chis). chis are the chi**2 values
    between the first row in `rowstack` and the rest of the rows (chis[0]
    is always 0.0). indices is a boolean index array. If row r has
    chi^2 less than `cutoff`, then indices[r] == True.
    First row is always included, i.e. indices[0] == True.

    Argument `rowstack` should be an array of shape (M, 3, n), where
    M is the number of observations (repeats, positions etc.) and
    n is the number of points in a single observation.
    """

    ish = rowstack.shape
    inds = np.zeros((ish[0]), dtype=np.bool)
    inds[0] = True
    chis = np.zeros((ish[0]))
    first = rowstack[0, :, :]
    for repno in range(1, ish[0]):
        chi2 = chivectors(rowstack[repno, :, :], first)
        chis[repno] = chi2
        if chi2 < cutoff:
            inds[repno] = True
    return inds, chis


def filter_stack(stack, fnames, chi2cutoff=1.5):
    """Return the mean of stack over repetitions, discarding outliers.

    Return value is an array with coordinates [posno, q/I/err, data].
    """
    ish = stack.shape
    outarr = np.zeros((ish[0], ish[2], ish[3]))
    diclist = []
    for posno in range(ish[0]):
        outdic = {"included": [], "discarded": [], "chi2cutoff": chi2cutoff}
        rinds, chis = chifilter(stack[posno, ...], cutoff=chi2cutoff)
        for repno in range(len(rinds)):
            if not rinds[repno]:
                logging.warning("posno: %d, rep: %d, chi2: %g > %g !"
                    % (posno, repno, chis[repno], chi2cutoff))
        outarr[posno, :, :] = mean_stack(stack[posno, rinds, :, :])
        incinds = list(np.arange(ish[1])[rinds])
        disinds = list(np.arange(ish[1])[np.logical_not(rinds)])
        outdic['included'] = [ [os.path.basename(fnames[posno][ind][0]),
            fnames[posno][ind][1], float(chis[ind])] for ind in incinds ]
        outdic['discarded'] = [ [os.path.basename(fnames[posno][ind][0]),
            fnames[posno][ind][1], float(chis[ind])] for ind in disinds ]
        outdic['q'] = map(float, list(outarr[posno,0, :]))
        outdic['I'] = map(float, list(outarr[posno,1, :]))
        outdic['Ierr'] = map(float, list(outarr[posno,2, :]))
        ld = loopyaml.Loopdict(outdic, loopvars=['q', 'I', 'Ierr'])
        diclist.append(ld)

    return outarr, diclist


def stack_scan(scanno, spec, q, radind, modulus=10):
    """Return an array with 1D curves in different positions in a scan.

    Argument `modulus` gives the number of unique positions in a scan.

    Return value is an array with coordinates [posno, repno, q/I/err, data]
    and shape (number_of_positions, number_of_repetitions, 3, len(q)).
    """
    scanlen = get_scanlen(spec, scanno)
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
            # Normalize transmission to a reasonable value
            dval = get_diode(spec, scanno, pointno) / 10000.0
            frname = get_framefilename(scanno, pointno, 0)
            (I, err) = get_binned(radind['indices'], frname)
            stack[posno, repno, 0, :] = q
            stack[posno, repno, 1, :] = I/dval
            stack[posno, repno, 2, :] = err/dval
            md5 = md5_file(frname)
            fnames[posno].append((frname, md5))
            repno = repno+1

    return stack, fnames


def stack_files(scanfile, conffile, outdir):
    """Create stacks from scans read from `scanfile` and write the to files.
    """
    scans = read_yaml(scanfile)
    read_experiment_conf(conffile)
    spec = read_spec(Specfile)
    radind = read_pickle(Indfile)
    q = radind['q']
    scannos = scans.keys()
    scannos.sort()
    for scanno in scannos:
        outname = "stack_%03d" % scanno
        stack, fnames = stack_scan(scanno, spec, q, radind, modulus=10)
        savemat(outdir+'/'+outname + ".mat", {outname: stack}, do_compression=1)
        fstack, fdict = filter_stack(stack, fnames, chi2cutoff=1.2)
        for i in range(len(fdict)):
            yname = "dat_%04d_%01d.yaml" % (scanno, i)
            write_yaml(fdict[i], outdir+'/'+yname)

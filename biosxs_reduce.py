from __future__ import with_statement
import glob, re, os.path, sys, logging, hashlib
import loopyaml
import numpy as np
import radbin as r
from detformats import read_cbf, read_spec
from csaxsformats import write_yaml, read_ydat
from scipy.io.matlab import savemat

Datadir = '/afs/psi.ch/project/cxs/users/ikonen/ND-Rhod-2010-04-24/Data10/'
Pilatusdir = Datadir + 'pilatus/S00000-00999/'
Expno = 12594
Detno = 1
Specfile = Datadir + 'spec/dat-files/specES1_started_2010_05_22_1133.dat'


#def read_frame(scanno, pointno, burstno):
#    """Return a frame from a given scan and point.
#    """
#    fname = get_framefilename(scanno, pointno, burstno)
#    frame = read_cbf(fname)
#    return frame


def get_framefilename(scanno, pointno, burstno):
    """Return the filename of a frame at a given scan, point, and burst number.
    """
    fname = "%s/S%05d/e%05d_%d_%05d_%05d_%05d.cbf" % (Pilatusdir, scanno, Expno, Detno, scanno, pointno, burstno)
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
    return len(scan['points'])


def md5_file(fname):
    md5 = hashlib.md5()
    with open(fname) as f:
        md5.update(f.read())
        h = md5.hexdigest()
    return h


def stack_ydat(stem, poslist, ending=".yaml"):
    """Return a stack of ydat-files in poslist starting with `stem`.
    """
    ll = []
    for p in poslist:
        fname = "%s%d%s" % (stem, p, ending)
        ll.append(read_ydat(fname).T)
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


def chivectors(x, y):
    nn = np.logical_not(np.logical_or(
        np.logical_or(np.isnan(x[1,:]), np.isnan(y[1,:])),
        np.logical_or((x[2,:] <= 0.0), (y[2,:] <= 0.0))))
    N = np.sum(nn)
    chi2 = (1.0/N)*np.sum((x[1,nn]-y[1,nn])**2 / (x[2,nn]**2+y[2,nn]**2))
    return chi2


def chifilter(stack, posno, cutoff=1.5):
    """Return an index vector of valid repetition numbers.

    Repetitions in a positions are compared to the first rep,
    and if significant chi^2 difference is found, the frames after
    this difference starts are rejected, i.e. not included in the
    return list.
    """

    ish = stack.shape
    first = stack[posno, 0, :, :]
    inds = np.zeros((ish[1]), dtype=np.bool)
    inds[0] = True
    chis = np.zeros((ish[1]))
    for repno in range(1, ish[1]):
        chi2 = chivectors(stack[posno, repno, :, :], first)
        chis[repno] = chi2
        if chi2 > cutoff:
            logging.warning("pos: %d, rep: %d, chi2: %g > %g !"
                % (posno, repno, chi2, cutoff))
        else:
            inds[repno] = True
    return inds, chis


def filter_stack(stack, fnames, chi2cutoff=1.5):
    """Return the mean of stack over repetitions, discarding outliers.

    Return value is an array with coordinates [posno, q/I/err, data].
    """

    def sumerr(x):
        return np.sqrt(np.sum(np.square(x), axis=0))

    def meanerr(x):
        return np.sqrt(np.sum(np.square(x), axis=0))/x.shape[0]

    ish = stack.shape
    outarr = np.zeros((ish[0], ish[2], ish[3]))
    diclist = []
    for posno in range(ish[0]):
        outdic = {"included": [], "discarded": [], "chi2cutoff": chi2cutoff}
        rinds, chis = chifilter(stack, posno, cutoff=chi2cutoff)
        outarr[posno, 0, :] = stack[posno, 0, 0, :] # q
        if len(rinds) > 1:
            outarr[posno, 1, :] = np.mean(stack[posno, rinds, 1, :], axis=0)
            outarr[posno, 2, :] = meanerr(stack[posno, rinds, 2, :])
        else:
            outarr[posno, 1, :] = stack[posno, rinds, 1, :]
            outarr[posno, 2, :] = stack[posno, rinds, 2, :]
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


def stack_files(idict, spec, q, radind, outdir):
    """Create stacks from scans in `idict` and write the to files.
    """
    scanstrs = idict.keys()
    scanstrs.sort()
    scannos = map(int, scanstrs) # just a check before looping over everything
    for ss in scanstrs:
        scanno = int(ss)
        outname = "stack_" + ss
        stack, fnames = stack_scan(scanno, spec, q, radind)
        savemat(outdir+'/'+outname + ".mat", {outname: stack}, do_compression=1)
        fstack, fdict = filter_stack(stack, fnames, chi2cutoff=1.2)
        for i in range(len(fdict)):
            yname = "dat_%04d_%01d.yaml" % (scanno, i)
            write_yaml(fdict[i], outdir+'/'+yname)

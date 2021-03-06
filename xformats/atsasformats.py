import re, StringIO
import numpy as np

def read_dat(fname):
    """Return dat-file contents as a Numpy array. Ignores the header and footer.
    """
    fid = open(fname)
    fstr = fid.read()
    fid.close()
    ss = ''.join(re.findall('^ *([-]*[0-9]+\.[0-9]+.*\n)', fstr, re.MULTILINE))
    out = StringIO.StringIO(ss)
    d = np.loadtxt(out)
    out.close()
    return d.T


def write_dat(arr, ff, comment="", skipz=False):
    """Write a dat-file from a given [n,3] sized array `arr`.

    Argument `ff` can be either a string, in which case it's interpreted
    as a filename, or an object (such as an open file) with a write method.

    comment: Write this string as a first line in the file.
    skipz: Skip data points i where arr[i,1] <= 0.

    If the second dimension of `arr` is much (10x) larger than first,
    the array is transposed before writing.
    """
    def write_dat_fp(arr, fp):
        fp.write(comment + "\n")
        # write a 3-line header to keep primus happy
        fp.write("#\n")
        fp.write("#\n")
        for i in range(arr.shape[0]):
            if not skipz or arr[i,1] > 0:
                for j in range(arr.shape[1]):
                    fp.write("  %12E" % arr[i,j])
                fp.write("\n")

    ish = arr.shape
    if (ish[1] > 3 and ish[0] <= 3) or ish[1]/ish[0] > 10:
        arr = arr.transpose()
    if hasattr(ff, 'write'):
        write_dat_fp(arr, ff)
    else:
        with open(ff, "w") as fp:
            write_dat_fp(arr, fp)


def read_gnom(fname):
    """Return contents of a GNOM output file (*.out) in dictionary.

    Keys:
        "prf"  : Numpy array with the p(r) function and errors.
        "Ireg" : Numpy array with regularized q vs. I.
        "Iexp" : Numpy array with q, Iexp, Error and Ireg.
        "alpha_grid" : Numpy array with the values of the alpha grid search.
        "alpha_search" : Array with values of the alpha golden section search.
        "alpha" : Chosen value of alpha.
        "solution_is" : Verbal description of the solution.
        "Rg_rec" : Reciprocal space Rg.
        "I0_rec" : Reciprocal space I(0).
        "Rg_real" : Real space Rg.
        "Rg_err_real" : Real space Rg error.
        "I0_real" : Real space I(0).
        "I0_err_real" : Real space I(0) error.
        "title" : Title string for the run.
        "filename" : Name of the input file.
    """
    fid = open(fname)
    fstr = fid.read(128000)
    fid.close()
    d = {}
    prstr = re.split('R *P\(R\) *ERROR[\r\n]*| *Reciprocal space:', fstr)[1]
    out = StringIO.StringIO(prstr)
    d['prf'] = np.loadtxt(out)
    regstr = re.split('S *J EXP *ERROR *J REG *I REG *[\r\n]*|[\r\n]* *Distance distribution', fstr)[1]
    qreg = np.array([float(s) for s in re.findall('^ *([0-9\.\-+Ee]+).*$', regstr, re.MULTILINE)])
    Ireg = np.array([float(s) for s in re.findall('([0-9\.\-+Ee]+)[\s\r\n]*$', regstr, re.MULTILINE)])
    d['Ireg'] = np.array([qreg, Ireg]).T
    expstr = '\n'.join(re.findall('^ *(?: +[0-9\.\-+Ee]+){5}[\s\r\n]*$', regstr, re.MULTILINE))
    out = StringIO.StringIO(expstr)
    expdat = np.loadtxt(out)
    d['Iexp'] = expdat[:,:4]
    alist = re.split('Alpha    Discrp  Oscill  Stabil  Sysdev  Positv  Valcen    Total *[\r\n]*|[\r\n]* *\*\*\*  Golden section search to maximize estimate \*\*\*|[\r\n] *####            Final results            ####', fstr)
    def split_junk(ss):
        return re.sub(r'(.{12})(.{8})(.{8})(.{8})(.{8})(.{8})(.{8})(.{9})', \
            r'\1 \2 \3 \4 \5 \6 \7 \8', ss)
    out = StringIO.StringIO(split_junk(alist[1]))
    d['alpha_grid'] = np.loadtxt(out)
    out = StringIO.StringIO(split_junk(alist[3]))
    d['alpha_search'] = np.loadtxt(out)
    d['alpha'] = float(re.search(".*Current ALPHA +: +([0-9\.\-+Ee]+) +Rg.*", fstr).group(1))
    d['solution_is'] = re.search(".*which is +([A-Z ]+) +solution.*", fstr).group(1)
    recmatch = re.search("Reciprocal space: Rg = *([0-9\.\-+Ee]+) *, I\(0\) = *([0-9\.\-+Ee]+)", fstr)
    d['Rg_rec'] = float(recmatch.group(1))
    d['I0_rec'] = float(recmatch.group(2))
    realmatch = re.search("Real space: Rg = *([0-9\.\-+Ee]+) +\+- +([0-9\.\-+Ee]+).*I\(0\) = *([0-9\.\-+Ee]+) +\+- +([0-9\.\-+Ee]+).*", fstr)
    d['Rg_real'] = float(realmatch.group(1))
    d['Rg_err_real'] = float(realmatch.group(2))
    d['I0_real'] = float(realmatch.group(3))
    d['I0_err_real'] = float(realmatch.group(4))
    d['title'] = re.search(".*Run title: *([^\r\n]*)", fstr).group(1)
    d['filename'] = re.search(".*Input file\(s\) : *([^\r\n]+)", fstr).group(1)
    return d

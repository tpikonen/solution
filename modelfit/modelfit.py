import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# --- Some models

def powerlawconst(params, x):
    """
    Return values of a powerlaw + constant (scale*x**power + const).

    :Parameters:
      - `params`: [scale, power, const]
      - `x`: values where function is computed

    :Returns:
      - `y`: values
    """
    return params[0]*x**params[1] + params[2]


def powerlawff(params, x, formfac=None):
    """
    Return values of a powerlaw + constant + formfactor
    (scale*x**power + const + ffscale*formfac).

    :Parameters:
      - `params`: [scale, power, const, ffscale]
      - `x`: values where function is computed
      - `formfac`: vector of size(x) defining the formfactor model
    :Returns:
      - `y`: values
    """
    return params[0]*x**params[1] + params[2] + params[3] * formfac


def gaussmodel(params, x):
    """
    Return values of a Gauss function.

    :Parameters:
      - `params`: [height, center, sigma]
      - `x`: values where function is computed

    :Returns:
      - `y`: values of gaussian
    """
    return params[0]*np.exp(-(params[1] - x)**2 / (params[2]**2))


def gaussline(params, x):
    """
    Return values of Gaussian + line
    :Parameters:
      - `params`: [slope, constant, height, center, sigma]
      - `x`: values where function is computed

    :Returns:
      - `y`: values of the function
    """
    yline = params[0]*x + params[1]
    ygauss = params[2]*np.exp(-(params[3] - x)**2 / (params[4]**2))
    return yline + ygauss


def lorentzconstant(params, x):
    """
    Return values of Lorentzian + constant
    :Parameters:
      - `params`: [height, center, width, constant]
      - `x`: values where function is computed

    :Returns:
      - `y`: values of the function
    """
    return params[0] / (1 + ((x-params[1])/params[2])**2) / (np.pi*params[2]) + params[3]

# The chi-squared optimization target function

def chimodel(modelf, params, x, y, vary, **kwargs):
    """
    Return chi**2 of model function vs. data.

    :Parameters:
      - `modelf`: model function of form f(params, x)
      - `params`: list of parameters to `modelf`
      - `x`: x axis of the data
      - `y`: values of data
      - `vary`: variance of y

    :Returns:
      - `chi2`: chi**2 of model vs. data
    """
    return np.sum( (y - modelf(params, x, **kwargs))**2 / np.abs(vary) )


def plotmodel(modelf, params, x, y, vary, residuals=0, **kwargs):
    """Linear plot of model vs. data at given parameter values.

    modelf : Model function
    params : parameters to modelf to be optimized
    x, y, vary : The data
    residuals : If True, plot residuals as well
    """
    plt.clf()
    fig = plt.gcf()
    if residuals:
        rax = fig.add_axes([0.1, 0.1, 0.85, 0.15])
        rax.plot(x, (y - modelf(params, x, **kwargs)) / vary, label="Residuals / sigma")
        ax = fig.add_axes([0.1, 0.3, 0.85, 0.60])
        ax.semilogy(x, modelf(params, x, **kwargs), label="Model")
        ax.hold(1)
        ax.semilogy(x, y, label="Data")
        ax.legend()
    else:
        plt.semilogy(x, modelf(params, x, **kwargs), label="Model")
        plt.hold(1)
        plt.semilogy(x, y, label="Data")
        plt.legend()


def semilogymodel(modelf, params, x, y, vary, residuals=0, **kwargs):
    """Semilogy plot of model vs. data at given parameter values.

    modelf : Model function
    params : parameters to modelf to be optimized
    x, y, vary : The data
    residuals : If True, plot residuals as well
    """
    plt.clf()
    fig = plt.gcf()
    if residuals:
        ax = fig.add_axes([0.1, 0.3, 0.85, 0.60])
        ax.semilogy(x, modelf(params, x, **kwargs), label="Model")
        ax.hold(1)
        ax.semilogy(x, y, label="Data")
        ax.legend()
        rax = fig.add_axes([0.1, 0.1, 0.85, 0.15])
        rax.plot(x, (y - modelf(params, x, **kwargs)) / vary, label="Residuals / sigma")
    else:
        plt.semilogy(x, modelf(params, x, **kwargs), label="Model")
        plt.hold(1)
        plt.semilogy(x, y, label="Data")
        plt.legend()


def modelfit(modelf, params, x, y, vary, residuals=0, noplot=0, **kwargs):
    """Fit modelf(params, x, **kwargs) to data given by (x,y, vary)

    modelf : Model function
    params : parameters to modelf to be optimized
    x, y, vary : The data
    residuals : If True, plot residuals as well
    noplot : If True, do not plot the fit
    """
    vary = np.abs(vary)
    mach = np.MachAr()
    zind = vary < mach.tiny
    if zind.any():
        print("VARIANCE CONTAINS ZEROS! Replacing with ones.")
        vary[zind] = 1.0
    if not noplot:
        plotmodel(modelf, params, x, y, vary, residuals=residuals, **kwargs)
    f = lambda pars: chimodel(modelf, tuple(pars), x, y, vary, **kwargs)
    popt, chisq, iter, fcalls, warnflag = scipy.optimize.fmin(f, list(params), full_output=1)
    if not noplot:
        plotmodel(modelf, popt, x, y, vary, residuals=residuals, **kwargs)
    return popt, chisq

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def funaver(fun, p, x):
    """Average 1-argument function `fun` over distribution p(x).
    """
    dx = np.diff(x)
    val = dx[0]*fun(x[0])*p[0]
    for i in range(len(dx)):
        val = val + dx[i]*fun(x[i+1])*p[i+1]
    return val / np.trapz(p, x)

def sphere(R, q):
    """Return the scattering form factor of a sphere with radius `R`."""
    f = 9*( (np.sin(q*R) - (q*R)*np.cos(q*R)) / (q*R)**3 )**2;
    return f


def sphere_dist_normal(Rcen, Rsigma, q, Ngrid=100):
    """Return form factor from spheres with a normally distributed radius.

    `Ngrid` samples are drawn around `Rcen` from `Rcen` - 3*`Rsigma` to
    `Rcen` + 3*`Rsigma`
    """
    x = np.linspace(max(0.0, Rcen-3*Rsigma), max(0.0, Rcen+3*Rsigma), Ngrid)
    p = norm.pdf(x, Rcen, Rsigma)
    return funaver(lambda R: sphere(R, q), p, x)


def spheredist_model(params, q):
    """Sphere with normal distribution of radii."""
    return params[0]*sphere_dist_normal(params[1], params[2], q) + \
        params[3]*q**params[4] + params[5]


def hardsphere_dist_normal(volfrac, Rcen, Rsigma, q, Ngrid=100):
    """Scattering from interacting hard spheres with a normally distributed radius.

    `Ngrid` samples are drawn around `Rcen` from `Rcen` - 3*`Rsigma` to
    `Rcen` + 3*`Rsigma`
    """
    x = np.linspace(max(0.0, Rcen-3*Rsigma), max(0.0, Rcen+3*Rsigma), Ngrid)
    p = norm.pdf(x, Rcen, Rsigma)
#    plt.plot(x, p)
#    print(np.trapz(p, x))
    return funaver(lambda R: hardsphere(volfrac, R, q), p, x)


def hard_sphere_sf(volfrac, R, q):
    """Return the hard sphere solution structure factor.

    Structure factor as a function of `q` for solutions with volume
    fraction `volfrac` and sphere radius `R`.

    See J.S. Pedersen, Adv. Colloid Interface Sci. 70 (1997) 171-210 and
    Kinning et. al., Macromolecules 17 (1984) 1712-1718.
    """
    a = (1.0 + 2*volfrac)**2 / (1.0 - volfrac)**4
    b = -6*volfrac*(1.0 + volfrac/2)**2 / (1.0 - volfrac)**2
    g = volfrac*a / 2.0

    A = 2.0*R*q

    G = a * (np.sin(A) - A*np.cos(A)) / A**2 + \
        b * (2*A*np.sin(A) + (2.0 - A**2)*np.cos(A) - 2.0) / A**3 + \
        g * (-A**4 * np.cos(A) + 4*((3*A**2 - 6.0)*np.cos(A) + \
            (A**3 - 6*A)*np.sin(A) + 6.0)) / A**5

    s = 1.0 / (1.0 + 24*volfrac*G/A)

    return s


def hard_sphere_sf_model(params, q):
    return hard_sphere_sf(params[0], params[1], q)


def hardsphere(volfrac, R, q):
    """Scattering from interacting hard spheres (structure factor * form factor).
    """
    return sphere(R, q) * hard_sphere_sf(volfrac, R, q)


def hardsphere_dist_model(params, q):
    """Return the full hard sphere model (structure and form factors).

    Variable list `params` should have the following variables:
       params[0]: Volume fraction.
        params[1]: Radius of hard-sphere interaction.
        params[2]: Center of particle form factor distribution.
        params[3]: Sigma of particle form factor distribution.
        params[4]: Overall scaling factor for structurefactor*formfactor
        params[5]: Constant.
        params[6]: Powerlaw scaling factor.
        params[7]: Powerlaw exponent.
    """
    return params[4] * sphere_dist_normal(params[2], params[3], q) * \
            hard_sphere_sf(params[0], params[1], q) + params[5] + \
            params[6] * q**params[7]


def hsd_model(params, q):
    """Full hard sphere model (structure and form factors).

    Variable list `params` should have the following variables:
        params[0]: Volume fraction.
        params[1]: Center of hard-sphere radius distribution
        params[2]: Sigma of hard-sphere radius distribution.
        params[3]: Overall scaling factor for structurefactor*formfactor
        params[4]: Constant.
        params[5]: Powerlaw scaling factor.
        params[6]: Powerlaw exponent.
    """
    return params[3] * \
        hardsphere_dist_normal(params[0], params[1], params[2], q) + \
        params[4] + params[5] * q**params[6]


def hsd_plot(ax, params, q):
    ax.plot(q, hsd_model(params, q), label="Full model")
    ax.plot(q, hardsphere_dist_normal(params[0], params[1], params[2], q),
        label="Hard spheres")
    ax.plot(q, hard_sphere_sf(params[0], params[1], q), label="Structure factor")
    ax.plot(q, params[5] * q**params[6], label="Powerlaw")
    ax.plot(q, params[4]*np.ones_like(q), label="Constant")
    plt.show()


def sf_model(params, q):
    """Hard-sphere structure factor model with fixed sphere form factor."""
    # Costants from form factor fit to higher q (> 0.05 1/nm) data.
    sphere_I = 7.94844219e+00 * sphere_dist_normal(8.0704e+01, 1.854e+00, q)
    I = params[5]*hard_sphere_sf(params[0], params[1], q)*sphere_I \
        * params[2]*q**params[3] + params[4]
    return I

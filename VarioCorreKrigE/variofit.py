"""
This file contains the functions required for estimating the semivariance as well as options to fit desired models to
the data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from tqdm.auto import tqdm
from matplotlib import gridspec
from scipy import special
from scipy.optimize import minimize
from numba import njit

# Geographical Distance Function
def haversine_oq(lon1, lat1, lon2, lat2, radians=False, earth_rad=6371.227):
    """
    Allows to calculate geographical distance
    using the haversine formula.

    :param lon1: longitude of the first set of locations
    :type lon1: numpy.ndarray
    :param lat1: latitude of the frist set of locations
    :type lat1: numpy.ndarray
    :param lon2: longitude of the second set of locations
    :type lon2: numpy.float64
    :param lat2: latitude of the second set of locations
    :type lat2: numpy.float64
    :keyword radians: states if locations are given in terms of radians
    :type radians: bool
    :keyword earth_rad: radius of the earth in km
    :type earth_rad: float
    :returns: geographical distance in km
    :rtype: numpy.ndarray
    """
    if not radians:
        cfact = np.pi / 180.
        lon1 = cfact * lon1
        lat1 = cfact * lat1
        lon2 = cfact * lon2
        lat2 = cfact * lat2

    # Number of locations in each set of points
    if not np.shape(lon1):
        nlocs1 = 1
        lon1 = np.array([lon1])
        lat1 = np.array([lat1])
    else:
        nlocs1 = np.max(np.shape(lon1))
    if not np.shape(lon2):
        nlocs2 = 1
        lon2 = np.array([lon2])
        lat2 = np.array([lat2])
    else:
        nlocs2 = np.max(np.shape(lon2))
    # Pre-allocate array
    distance = np.zeros((nlocs1, nlocs2))
    i = 0
    while i < nlocs2:
        # Perform distance calculation
        dlat = lat1 - lat2[i]
        dlon = lon1 - lon2[i]
        aval = (np.sin(dlat / 2.) ** 2.) + (np.cos(lat1) * np.cos(lat2[i]) * (np.sin(dlon / 2.) ** 2.))
        distance[:, i] = (2. * earth_rad * np.arctan2(np.sqrt(aval), np.sqrt(1 - aval))).T
        i += 1
    return distance

# Semivariogram Estimators
@njit
def matheron(x):
    """Matheron (classical) semivariogram from increments x = |z_i - z_j|.

    References
    Matheron, G. (1962): Traité de Géostatistique Appliqué, Tonne 1. Memoires de Bureau de Recherches Géologiques et Miniéres, Paris.

    Matheron, G. (1965): Les variables regionalisées et leur estimation. Editions Masson et Cie, 212 S., Paris.

    """
    if x.size == 0:
        return np.nan

    return 0.5 * np.sum(x**2) / x.size

@njit
def cressie_hawkins(x):
    """Cressie–Hawkins robust estimator.

    References
    Cressie, N., and D. Hawkins (1980): Robust estimation of the variogram. Math. Geol., 12, 115-125.

    """
    n = x.size

    if x.size == 0:
        return np.nan

    A = 0.457 + 0.494/n + 0.045/(n**2)
    return 0.5 * (np.mean(np.sqrt(x))**4) / A

@njit
def dowd(x):
    """Dowd median-based estimator.

    References
    Dowd, P. A., (1984): The variogram and kriging: Robust and resistant estimators, in Geostatistics for Natural Resources Characterization. Edited by G. Verly et al., pp. 91 - 106, D. Reidel, Dordrecht.

    """
    return 1.099 * (np.nanmedian(x)**2)

# Semivariogram Models
def spherical(h, r, c0, b=0.0):
    """
    Semivariogram: Spherical (compact support)

    Definition
    ----------
    Set a = r and x = h / a. Then
        γ(h) = b + c0 * [ 1.5 x - 0.5 x^3 ]     for 0 <= x <= 1
                b + c0                          for x  >  1

    Parameters
    ----------
    h : array-like or float
        Nonnegative lag distance(s).
    r : float
        Effective range; equals the compact-support radius a.
    c0 : float
        Partial sill (γ plateau height minus nugget).
    b : float, default 0.0
        Nugget.

    Returns
    -------
    gamma : ndarray or float
        Semivariogram values with the same shape as `h`.

    Notes
    -----
    Reaches the sill exactly at h = r (compact support).
    """

    a = r
    h = np.asarray(h, float)
    x = h / a
    part = b + c0 * (1.5*x - 0.5*x**3)
    out = np.where(h <= a, part, b + c0)
    return out

def exponential(h, r, c0, b=0.0):
    """
    Semivariogram: Exponential

    Definition
    ----------
    Use a = r / 3 (≈95% of the sill at h = r). Then
        γ(h) = b + c0 * ( 1 - exp(-h / a) )

    Parameters
    ----------
    h : array-like or float
        Nonnegative lag distance(s).
    r : float
        Effective range; mapping a = r / 3.
    c0 : float
        Partial sill.
    b : float, default 0.0
        Nugget.

    Returns
    -------
    gamma : ndarray or float
        Semivariogram values with the same shape as `h`.

    Notes
    -----
    Approaches the sill asymptotically (never exactly reaches it).
    """

    a = r / 3.0
    h = np.asarray(h, float)
    return b + c0 * (1.0 - np.exp(-h / a))

def gaussian(h, r, c0, b=0.0):
    """
    Semivariogram: Gaussian

    Definition
    ----------
    Use a = r / 2 (≈95% of the sill at h = r). Then
        γ(h) = b + c0 * ( 1 - exp( - (h / a)^2 ) )

    Parameters
    ----------
    h : array-like or float
        Nonnegative lag distance(s).
    r : float
        Effective range; mapping a = r / 2.
    c0 : float
        Partial sill.
    b : float, default 0.0
        Nugget.

    Returns
    -------
    gamma : ndarray or float
        Semivariogram values with the same shape as `h`.

    Notes
    -----
    Very smooth near the origin; faster decay than exponential.
    """

    a = r / 2.0
    h = np.asarray(h, float)
    return b + c0 * (1.0 - np.exp(-(h / a)**2))

def cubic(h, r, c0, b=0.0):
    """
    Semivariogram: Cubic (compact support)

    Definition
    ----------
    Set a = r and x = h / a. Then
        γ(h) = b + c0 * [ 7 x^2 - (35/4) x^3 + (7/2) x^5 - (3/4) x^7 ]  for 0 <= x < 1
                b + c0                                                   for x >= 1

    Parameters
    ----------
    h : array-like or float
        Nonnegative lag distance(s).
    r : float
        Effective range; equals the compact-support radius a.
    c0 : float
        Partial sill.
    b : float, default 0.0
        Nugget.

    Returns
    -------
    gamma : ndarray or float
        Semivariogram values with the same shape as `h`.

    Notes
    -----
    Compact support like spherical, with a different interior polynomial.
    """

    a = r
    h = np.asarray(h, float)
    x = h / a
    poly = 7*x**2 - (35.0/4.0)*x**3 + (7.0/2.0)*x**5 - (3.0/4.0)*x**7
    out = np.where(h < a, b + c0 * poly, b + c0)
    return out

def powered_exponential(h, r, c0, beta, b=0.0):
    """
    Semivariogram: Powered exponential (a.k.a. Stable)

    Definition
    ----------
    Use a = r / (3)^(1/beta) (≈95% of the sill at h = r). Then
        γ(h) = b + c0 * ( 1 - exp( - (h / a)^beta ) ),   0 < beta <= 2

    Parameters
    ----------
    h : array-like or float
        Nonnegative lag distance(s).
    r : float
        Effective range; mapping a = r / (3)^(1/beta).
    c0 : float
        Partial sill.
    beta : float
        Shape exponent, 0 < beta <= 2.  beta=1 → exponential, beta=2 → Gaussian.
    b : float, default 0.0
        Nugget.

    Returns
    -------
    gamma : ndarray or float
        Semivariogram values with the same shape as `h`.

    Notes
    -----
    Interpolates smoothly between exponential and Gaussian behaviors.
    """

    a = r / (3.0 ** (1.0 / beta))
    h = np.asarray(h, float)
    return b + c0 * (1.0 - np.exp(- (h / a)**beta))

def matern(h, r, c0, s, b=0.0):
    """
    Semivariogram: Matérn

    Definition
    ----------
    Set a = r / 2 and u = 2 * (h * sqrt(s)) / a. Then
        γ(h) = b + c0 * [ 1 - (2 / Γ(s)) * ((h * sqrt(s)) / a)^s * K_s( 2 * (h * sqrt(s)) / a ) ]
    where K_s is the modified Bessel function of the second kind.

    Parameters
    ----------
    h : array-like or float
        Nonnegative lag distance(s).
    r : float
        Effective range; mapping a = r / 2 (≈95% of the sill at h = r).
    c0 : float
        Partial sill.
    s : float
        Smoothness parameter (ν = s > 0). Smaller s → rougher field; large s → Gaussian-like.
    b : float, default 0.0
        Nugget.

    Returns
    -------
    gamma : ndarray or float
        Semivariogram values with the same shape as `h`.

    Notes
    -----
    Implemented with safe handling at h=0 (returns b). Requires scipy.special.kv.
    """

    a = r / 2.0
    h = np.asarray(h, float)
    u = 2.0 * (h * np.sqrt(s)) / a
    # Avoid NaNs at h=0: set gamma(0)=b
    with np.errstate(divide='ignore', invalid='ignore'):
        term = (2.0 / special.gamma(s)) * ((h * np.sqrt(s)) / a)**s * special.kv(s, u)
    out = b + c0 * (1.0 - term)
    out = np.where(h == 0.0, b, out)
    return out

def damped_cosine_angle(theta_deg, c, c0, b=0.0):
    """
    Semivariogram: Damped-cosine in angle (degrees)

    Definition
    ----------
    For angular separation θ (in degrees), define the correlation-like term
        R(θ) = cos(θ * π/180) * exp(-θ / c)
    and set
        γ(θ) = b + c0 * [ 1 - R(θ) ].

    Parameters
    ----------
    theta_deg : array-like or float
        Angular separation(s) in degrees (typical range 0–180).
    c : float
        Damping angle in degrees; larger c → slower angular decorrelation.
    c0 : float
        Partial sill.
    b : float, default 0.0
        Nugget.

    Returns
    -------
    gamma : ndarray or float
        Semivariogram values with the same shape as `theta_deg`.

    Notes
    -----
    Use only where lags are angular (e.g., directional models on a sphere).
    Not a standard Euclidean variogram model; intended for angular dependence.
    """

    theta_deg = np.asarray(theta_deg, float)
    th = np.radians(theta_deg)
    return b + c0 * (1.0 - np.cos(th) * np.exp(-theta_deg / c))

def angular_dissimilarity(theta_deg, c, c0, b=0.0):
    """
    Angular semivariogram (Padonou–Roustant, 2016) in DEGREES.

    R0(θ) = (1 + θ/c) * (1 - θ/180)^(180/c),  with θ in [0, 180].
    γ(θ)  = b + c0 * [1 - R0(θ)]

    Parameters
    ----------
    theta_deg : array_like
        Angular separation(s) in degrees.
    c : float
        Positive damping/scale parameter (>0).
    c0 : float
        Partial sill (>=0).
    b : float, default 0.0
        Nugget (>=0).

    Returns
    -------
    gamma : ndarray
        Semivariogram values with the same shape as `theta_deg`.
    """
    theta = np.asarray(theta_deg, dtype=float)
    # clamp to valid domain [0, 180]
    th = np.clip(theta, 0.0, 180.0)

    c = float(c)
    if c <= 0.0:
        raise ValueError("Parameter c must be > 0 for angular_dissimilarity.")

    # base in [0,1]
    base = np.clip(1.0 - th / 180.0, 0.0, 1.0)
    exponent = 180.0 / c
    pref = 1.0 + (th / c)

    # stable power: base**exponent where base>0, else 0
    pow_term = np.where(base > 0.0, np.power(base, exponent), 0.0)
    R0 = pref * pow_term  # correlation kernel in [0,1]

    return b + c0 * (1.0 - R0)

VARIOGRAM_MODELS = {
    "spherical": spherical,
    "exponential": exponential,
    "gaussian": gaussian,
    "cubic": cubic,
    "powered_exponential": powered_exponential,
    "matern": matern,
    "damped_cosine_angle": damped_cosine_angle,
    "angular_dissimilarity": angular_dissimilarity,
}

# Define Fitting Weights
def compute_distance_weights(h_lag, n_j, weight_type='inverse-linear weighting', weight_params = None):

    """
    Build per-bin weights for fitting.

    Parameters
    ----------
    h_lag : (k,) array_like of float
        Bin centers (same order as the target vector).
    n_j : (k,) array_like of float
        Pair counts per bin.
    weight_type : {'inverse-linear weighting','exponential weighting','powered weighting', 'linear weighting', None, 'ols'}
        If None/'ols', returns ones (plain OLS).
        'inverse-linear weighting': w(h)=n_j * 1/(1+h/b)
        'exponential weighting'   : w(h)=n_j * exp(-h/b)
        'powered weighting'     : w(h)=n_j * (1+h/b)^(-alpha)
        'linear weighting'      : w(h)=n_j * ones(h)
    weight_params : list[float] | dict | None
        If list, expected [b, alpha]; if dict, keys {'b','alpha'}.
        For 'inverse-linear weighting' and 'exponential weighting', only 'b' is used.

    Returns
    -------
    weights : (k,) ndarray of float
        Weight per bin.

    Raises
    ------
    ValueError
        If `weight_type` is unknown or required params missing.
    """

    h_lag = np.asarray(h_lag, float)
    n_j = np.asarray(n_j, float)

    if weight_type == 'inverse-linear weighting':
        w = n_j * (1.0 / (1.0 + h_lag / weight_params[0]))
    elif weight_type == 'exponential weighting':
        w = n_j * np.exp(-h_lag / weight_params[0])
    elif weight_type == 'powered weighting':
        w = n_j * (1.0 + h_lag / weight_params[0]) ** (-weight_params[1])
    elif weight_type == 'linear weighting':
        w = n_j * np.ones_like(h_lag, dtype=float)
    elif weight_type is None or weight_type == 'ols':
        w =  np.ones_like(h_lag, dtype=float)
    else:
        raise ValueError("Invalid weight_type: choose None/'ols', 'inverse-linear weighting', 'exponential weighting', 'powered weighting' or 'linear weighting'")

    return w

# Objective Function(s) for Fitting
def objective_func(params, h, gamma, weights, semivario_fn):
    """
    Weighted SSE objective: minimize Σ w_i [y_i - model_fn(h_i; θ)]^2.

    Parameters
    ----------
    params : sequence of float
        θ in the order expected by `model_fn`.
    h, y, weights : (k,) arrays
        Bin centers, target values (gamma or rho), and weights.
    model_fn : callable
        Signature `model_fn(h, *params)` → (k,) array.

    Returns
    -------
    float
        Weighted sum of squared residuals.
    """

    gamma_pred = semivario_fn(h, *params)
    return np.sum(weights * (gamma - gamma_pred)**2)

def make_init_and_bounds(model, h, gamma, xmax_factor=2.0, fix_nugget=True, fix_sill=False):
    """
    Initial guesses & bounds for semivariogram models.

    Parameters
    ----------
    model : {'spherical','exponential','gaussian','cubic',
             'powered_exponential','matern','damped_cosine_angle'}
    h : array_like (k,)
        Bin centers.
    gamma : array_like (k,)
        Experimental semivariogram at bin centers.
    xmax_factor : float, default 2.0
        Upper bound for range-like parameter (r or c_deg): xmax_factor * max(h).
    fix_nugget : bool, default True
        If True, nugget b is fixed at 0.0 (bounds (0,0) and init 0).
    fix_sill : bool, default False
        If True, partial sill c0 is fixed at 1.0 (bounds (1,1) and init 1).

    Returns
    -------
    x0 : tuple
        Initial parameter vector.
    bounds : tuple of (low, high) tuples
        Bounds aligned with `x0`.

    Notes
    -----
    - Range-like params (r or c_deg) lower-bounded >0 and upper-capped at
      `xmax_factor * max(h)` to prevent near-flat fits on flexible kernels.
    - If `fix_nugget`, nugget b is fixed at 0 via bounds (0,0).
    - If `fix_sill`, partial sill c0 is fixed at 1 via bounds (1,1).
    """

    h = np.asarray(h, float).ravel()
    g = np.asarray(gamma, float).ravel()

    # robust lag scales
    mask_pos = np.isfinite(h) & (h > 0)
    h_min = float(np.nanmin(h[mask_pos])) if np.any(mask_pos) else 1.0
    h_max = float(np.nanmax(h[mask_pos])) if np.any(mask_pos) else 1.0

    # simple inits
    r0 = 0.5 * h_max                       # range start
    b0 = float(np.nanmin(g))               # nugget start
    c0 = max(float(np.nanmax(g) - b0), 1e-9)  # partial sill

    # range bounds: positive, upper-capped by xmax_factor*max(h)
    r_lo = max(1e-3, 0.5 * h_min)
    r_hi = xmax_factor * h_max if np.isfinite(h_max) and h_max > 0 else None
    r_bounds = (r_lo, r_hi)

    if model in ("spherical", "exponential", "gaussian", "cubic"):
        x0 = (r0, c0, 0.0 if fix_nugget else b0)
        bounds = (
            r_bounds,                                           # r
            (1.0, 1.0) if fix_sill else (0.0, None),           # c0
            (0.0, 0.0) if fix_nugget else (0.0, None),         # b
        )

    elif model == "powered_exponential":
        x0 = (r0, c0, 1.0, 0.0 if fix_nugget else b0)           # (r, c0, beta, b)
        bounds = (
            r_bounds,                                           # r
            (1.0, 1.0) if fix_sill else (0.0, None),           # c0
            (1e-2, 2.0),                                       # beta
            (0.0, 0.0) if fix_nugget else (0.0, None),         # b
        )

    elif model == "matern":
        x0 = (r0, c0, 0.5, 0.0 if fix_nugget else b0)           # (r, c0, s, b)
        bounds = (
            r_bounds,                                           # r
            (1.0, 1.0) if fix_sill else (0.0, None),           # c0
            (1e-3, 5.0),                                       # s (=nu)
            (0.0, 0.0) if fix_nugget else (0.0, None),         # b
        )

    elif model in ("damped_cosine_angle", "angular_dissimilarity"):
        # h must be in DEGREES here (0..180)
        deg_cap = 180.0

        mask_pos = np.isfinite(h) & (h > 0)
        h_min = float(np.nanmin(h[mask_pos])) if np.any(mask_pos) else 1.0
        h_max = float(np.nanmax(h[mask_pos])) if np.any(mask_pos) else 1.0

        # initial guess for angular scale (degrees)
        c_init = max(1e-3, 0.5 * h_max)
        c_lo = max(1e-3, 0.5 * h_min)

        # per-model cap
        if model == "damped_cosine_angle":
            model_factor = xmax_factor  # e.g., 2.0
        else:  # "angular_dissimilarity"
            model_factor = min(xmax_factor, 1.0)

        # cap by data AND domain
        c_hi_raw = model_factor * h_max if (np.isfinite(h_max) and h_max > 0.0) else deg_cap
        c_hi = min(deg_cap, c_hi_raw)

        # guard against degenerate ranges (tiny/identical angles)
        if not np.isfinite(c_hi) or c_hi <= c_lo:
            c_hi = c_lo * 1.01

        # make sure the initial guess is feasible
        c_init = float(np.clip(c_init, c_lo, c_hi))

        # (c_deg, c0, b) for variogram family
        x0 = (c_init, 1.0 if fix_sill else c0, 0.0 if fix_nugget else b0)
        bounds = (
            (c_lo, c_hi),  # c_deg (scale in degrees)
            (1.0, 1.0) if fix_sill else (0.0, None),  # c0 (partial sill)
            (0.0, 0.0) if fix_nugget else (0.0, None),  # b  (nugget)
        )

    else:
        raise ValueError("Unknown model")

    return x0, bounds

# R2 and Packing Semivariogram model parameters
def r2_score_weighted(y, yhat, w=None):
    """
    Weighted coefficient of determination, R^2.

    Computes
        R^2_w = 1 - SSE_w / SST_w
    where
        SSE_w = Σ_i w_i (y_i - ŷ_i)^2
        SST_w = Σ_i w_i (y_i - ȳ_w)^2
        ȳ_w   = (Σ_i w_i y_i) / (Σ_i w_i)

    If `w` is None, all weights are treated as 1 (ordinary R^2).

    Parameters
    ----------
    y : array-like, shape (n,) or (n, 1)
        Observed values at bin centers (or targets in general).
    yhat : array-like, shape (n,) or (n, 1)
        Model predictions at the same locations.
    w : array-like, shape (n,), optional
        Nonnegative weights (e.g., pair counts, or distance-decay × counts).
        If None, uses equal weights.

    Returns
    -------
    r2 : float
        Weighted R^2 in (-inf, 1]. Returns `np.nan` if the weighted variance
        `SST_w` is zero (e.g., all `y` identical under the weights).

    Notes
    -----
    • When `w ≡ 1`, R^2_w reduces to ordinary R^2.
    • For variogram/correlogram fitting, a common choice is `w = N(h)` (pair
      counts) or `w = N(h) × decay(h)` to emphasize short lags.
    • R^2 measures goodness of fit relative to the (weighted) mean, not absolute
      error magnitude.
    """

    y = np.asarray(y, float).ravel()
    yhat = np.asarray(yhat, float).ravel()
    if w is None:
        ybar = np.mean(y)
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - ybar)**2)
    else:
        w = np.asarray(w, float).ravel()
        wsum = np.sum(w)
        if wsum == 0:
            return np.nan
        ybar = np.sum(w * y) / wsum
        ss_res = np.sum(w * (y - yhat)**2)
        ss_tot = np.sum(w * (y - ybar)**2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def pack_params(model_type, theta):

    """
    Semivariogram-model parameter packing.

    Parameter layouts expected by VARIOGRAM_MODELS:

      - spherical / exponential / gaussian / cubic : (r, c0, b)
      - powered_exponential                        : (r, c0, beta, b)
      - matern                                     : (r, c0, s, b)      # s = smoothness (ν)
      - damped_cosine_angle                        : (c, c0, b)         # c = damping angle (degrees)

    where
      r    : effective range (model-specific mapping to 'a')
      c0   : partial sill
      b    : nugget
      beta : shape exponent for powered exponential (0 < beta ≤ 2)
      s    : Matérn smoothness ν
      c    : angular damping (degrees) for the damped-cosine model
    """

    if model_type in ("spherical", "exponential", "gaussian", "cubic"):
        names = ("r", "c0", "b")
    elif model_type == "powered_exponential":
        names = ("r", "c0", "beta", "b")
    elif model_type == "matern":
        names = ("r", "c0", "s", "b")
    elif model_type in ("damped_cosine_angle","angular_dissimilarity"):
        names = ("c", "c0", "b")          # c = damping (degrees)
    else:
        raise ValueError("Unknown model_type")
    return {k: float(v) for k, v in zip(names, theta)}

def theta_from_params(params, model_type):
    """
    Unpack in the same order expected by VARIOGRAM_MODELS signatures.
    """
    if model_type in ("spherical","exponential","gaussian","cubic"):
        order = ("r","c0","b")
    elif model_type == "powered_exponential":
        order = ("r","c0","beta","b")
    elif model_type == "matern":
        order = ("r","c0","s","b")
    elif model_type in ("damped_cosine_angle", "angular_dissimilarity"):
        order = ("c","c0","b")     # damping c (deg), c0, b
    else:
        raise ValueError("Unknown model_type")
    return [float(params[k]) for k in order]

# Main Function
def variofit(values, coordinates, distance_type, max_distance, bin_size, estimator_type, model_type, weight_fn,
             weight_params, xmax_factor = 2.0, fix_nugget = True, fix_sill =  False, plot = False):

    """
    Compute an experimental semivariogram and fit a user-selected variogram model.

    Parameters
    ----------
    values : array_like, shape (n,)
        Sample values z_i at each coordinate.
    coordinates : array_like, shape (n, d)
        Sample locations.
        If distance_type == 'geographic', coordinates[:,0]=lat (deg), coordinates[:,1]=lon (deg).
        If distance_type == 'cartesian' : (x, y)
        If distance_type == 'euclidean', coordinates are in linear units (e.g., km or m or dimensionless).
    distance_type : {'geographic', 'euclidean'}
        Distance metric for lag computation.
        'geographic' uses haversine over a sphere (Earth radius 6371.227 km) → lags in km.
        'euclidean' uses standard euclidean norm → lags in the same units as `coordinates`.
        'cartesian' uses the x, y values to compute the euclidean distance between pairs
    max_distance : float
        Maximum lag to include (same units as the chosen distance_type).
    bin_size : float
        Width of each lag bin (same units as `max_distance`). Bin centers are
        computed as `bin_size/2 + i*bin_size`.
    estimator_type : {'Matheron', 'CressieHawkins', 'Dowd'}
        Semivariogram estimator per bin, applied to increments |z_i - z_j|.
    model_type : {'exponential', 'cubic', 'powered_exponential', 'matern',
                  'gaussian', 'spherical', 'damped_cosine_angle'}
        Variogram model to fit. See Notes for parameterizations.
    weight_fn : {None, 'ols', 'inverse-linear weighting', 'exponential weighting', 'powered weighting', 'linear weighting'}
        Bin weight scheme for fitting. None/'ols' gives equal weights per bin.
        Other schemes multiply a distance-decay weight by the bin pair-count.
    weight_params : list or dict or None
        Parameters for the chosen weight_fn. If list, expected [b, alpha].
        If dict, expected keys {'b', 'alpha'}. `alpha` is ignored for
        'inverse-linear weighting' and 'exponential weighting'. If None, defaults to b = 0.25*max(h), alpha=1.0.
    fix_nugget : bool, default True
        If True, fixes the nugget (b) to 0.0 during fitting.
    fix_sill : bool, default False
        If True, fixes the partial sill (c0) to 1.0 during fitting (useful for normalized data).
    plot : bool, default False
        If True, shows a two-panel plot: bin counts and semivariogram with fitted curve.

    Returns
    -------
    h_lag : ndarray, shape (k, 1)
        Lag bin centers actually used (non-empty bins).
    n_obs : ndarray, shape (k, 1)
        Pair counts per bin.
    gamma : ndarray, shape (k, 1)
        Experimental semivariogram values per bin.
    params : dict
        Fitted model parameters (keys depend on `model_type`).
    r2_wls : float
        Weighted R^2 computed at bin centers using the same weights passed to the objective.
    r2_ols : float
        Ordinary R^2 computed at bin centers using weights as ones passed to the objective.

    Notes
    -----
    - Estimators:
      * Matheron: 0.5 * mean( (z_i - z_j)^2 ).
      * Cressie–Hawkins: robust, uses small-sample bias correction.
      * Dowd: median-based, robust to outliers.

    - Model parameterizations (your exact implementations):
      * exponential:     gamma(h)= b + c0 * (1 - exp(-h/a)) with internal a=r/3.
      * gaussian:        gamma(h)= b + c0 * (1 - exp(-(h/a)^2)) with a=r/2.
      * spherical:       compact support; r is the effective range.
      * cubic:           compact support; r is the effective range.
      * powered_exponential: gamma(h)= b + c0 * (1 - exp(-(h/a)^beta)), a = r / 3^(1/beta).
      * matern:          gamma(h)= b + c0 * [1 - (2/Gamma(s)) ((h√s)/a)^s K_s(2(h√s)/a)], with a=r/2.
      * damped_cosine_angle: gamma(θ)= b + c0 * [1 - cos(θ) * exp(-θ/c)],
      * angular_dissimilarity:
                             where θ is in degrees and c is a damping angle in degrees.

    - Angular vs distance lags:
      * 'damped_cosine_angle' expects angular lags in **degrees**. If you pass
        distance-based `h_lag`, the fit will be meaningless—use this model only
        when your bins represent angles.

    - Weights:
      * 'inverse-linear weighting': n_j * 1/(1 + h/b)
      * 'exponential weighting':   n_j * exp(-h/b)
      * 'powered weighting':     n_j * (1 + h/b)^(-alpha)
      * 'linear weighting':   n_j * ones(size.h,)
      * 'ols':              ones(size.h,)
      Each of these is multiplied by the bin pair-count.

    - Optimization:
      The fit minimizes sum_i w_i * (gamma_i - model(h_i; θ))^2 with bounds
      and initial guesses chosen by `make_init_and_bounds`.

    """

    # ensure arrays
    values = np.asarray(values, float)
    coords  = np.asarray(coordinates, float)
    n = len(values)

    # Define arrays for storage
    nmax = round(max_distance/bin_size)
    h_lag = np.zeros((nmax,1))
    gamma = np.zeros((nmax,1))
    n_obs = np.zeros((nmax,1))

    # define semivariance estimator
    if estimator_type == "Matheron":
        semivarioest_fn = matheron
    elif estimator_type == "CressieHawkins":
        semivarioest_fn = cressie_hawkins
    elif estimator_type == "Dowd":
        semivarioest_fn = dowd
    else:
        raise ValueError("Invalid estimator: choose from 'Matheron', 'CressieHawkins', or 'Dowd'")

    # define semivariance model
    if model_type == "exponential":
        semivariomodel_fn = exponential
    elif model_type == "cubic":
        semivariomodel_fn = cubic
    elif model_type == "powered_exponential":
        semivariomodel_fn = powered_exponential
    elif model_type == "matern":
        semivariomodel_fn = matern
    elif model_type == "gaussian":
        semivariomodel_fn = gaussian
    elif model_type == "spherical":
        semivariomodel_fn = spherical
    elif model_type == "damped_cosine_angle":
        semivariomodel_fn = damped_cosine_angle
    elif model_type == "angular_dissimilarity":
        semivariomodel_fn = angular_dissimilarity
    else:
        raise ValueError("Invalid Model: Choose from 'exponential', 'cubic', 'powered_exponential', 'matern', "
                         "'spherical', 'gaussian', 'angular_dissimilarity' or 'damped_cosine_angle'")

    # compute pairwise distances
    if distance_type == 'geographic':
        lat = coords[:, 0]
        lon = coords[:, 1]
        distance = np.asarray(haversine_oq(lon, lat, lon, lat, radians=False, earth_rad=6371.227), dtype=float)
        distance_ratio = np.rint(distance / bin_size)

    elif distance_type == 'cartesian':
        # coords: (n,2) = (x,y) in same units as bin_size
        if coords.shape[1] != 2:
            raise ValueError("cartesian requires coordinates shape (n,2): (x, y)")
        dx = coords[:, None, 0] - coords[None, :, 0]
        dy = coords[:, None, 1] - coords[None, :, 1]
        distance = np.hypot(dx, dy)
        distance_ratio = np.rint(distance / bin_size)

    elif distance_type == 'euclidean':
        diff = coords[:, None, :] - coords[None, :, :]
        distance = np.linalg.norm(diff, axis=-1)
        distance_ratio = np.rint(distance / bin_size)

    elif distance_type == 'angular':
        theta = np.asarray(coords, float).ravel()
        # Cosine of angular differences
        cos_diff = np.cos(theta[:, None] - theta[None, :])
        ang_rad = np.arccos(np.clip(cos_diff, -1.0, 1.0))
        distance = np.degrees(ang_rad)
        distance_ratio = np.rint(distance / bin_size)

    else:
        raise ValueError("Invalid distance_type: choose 'geographic', 'cartesian', 'angular', or 'euclidean'")

    # compute semivariance
    # for i in tqdm(range(1,nmax+1), desc="Computing semivariance: ", leave=False):
    for i in range(1,nmax+1):
        [site1, site2] = np.where(distance_ratio == i)
        if len(site1) > 0:
            h_lag[i-1,0]=bin_size/2+(i-1)*bin_size
            n_obs[i-1,0]= len(site1)
            x = np.abs(values[site1] - values[site2])
            gamma[i-1, 0] = semivarioest_fn(x)
        else:
            h_lag[i-1,0]=np.nan
            n_obs[i-1,0]=np.nan
            gamma[i-1,0]=np.nan

    # drop empty bins/nan estimates in semivariance
    keep = ~np.isnan(gamma).any(axis=1)
    h_lag = h_lag[keep]
    n_obs = n_obs[keep]
    gamma = gamma[keep]

    # fit semivariogram model
    h = h_lag.ravel()
    g = gamma.ravel()
    m = n_obs.ravel()

    if weight_fn is None or str(weight_fn).lower() == "ols":
        weights = np.ones_like(h, dtype=float)
    else:
        if isinstance(weight_params, dict):
            b = weight_params.get("b", 0.25 * float(h.max()) if h.size else 1.0)
            alpha = weight_params.get("alpha", 1.0)
            weight_params = [b, alpha]
        # expect [b, alpha] for all schemes (alpha ignored for 'inverse linear weighting'/'exponential weighting')
        weights = compute_distance_weights(h, m, weight_type=weight_fn, weight_params=weight_params)

    # initial guess & bounds (use model_type)
    x0, bounds = make_init_and_bounds(model_type, h, g, xmax_factor, fix_nugget, fix_sill)

    #  optimize
    res = minimize(
        fun=lambda th: objective_func(th, h, g, weights, semivariomodel_fn),
        x0=x0,
        bounds=bounds
        # method="L-BFGS-B"
    )
    theta_hat = res.x

    # get rsquared fit
    g_fit_bins = semivariomodel_fn(h, *theta_hat)
    r2_wls = r2_score_weighted(g, g_fit_bins, w=weights)
    r2_ols = r2_score_weighted(g, g_fit_bins, w=None)

    # store parameters
    params = pack_params(model_type, theta_hat)

    # smooth curve for plotting
    xlag_fit = np.linspace(0.0, float(max_distance), 1000)
    gamma_pred = semivariomodel_fn(xlag_fit, *theta_hat)

    # Create Plot for Semi-variance
    if plot:
        fig = plt.figure(figsize=(12,7),dpi = 200)
        # set height ratios for subplots
        gs_plot = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

        # the first subplot
        ax0 = plt.subplot(gs_plot[0])
        ax0.bar(h_lag[:,0],n_obs[:,0], edgecolor='black', align='center', width=bin_size/2)
        ax0.grid(which = 'minor')
        # the second subplot
        # shared axis X
        ax1 = plt.subplot(gs_plot[1], sharex = ax0)
        ax1.plot(h_lag[:,0],gamma[:,0],'o',markeredgecolor='black')
        ax1.plot()
        ax1.plot(xlag_fit, gamma_pred, '-k')

        plt.setp(ax0.get_xticklabels(), visible=False)

        # remove last tick label for the second subplot
        yticks = ax1.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
        ax0.set_ylabel('Number of Lags, N')
        ax0.set_ylim(0, max(n_obs[:,0]))

        # set legend
        ax1.legend(['Experimental', r'Model, $R^2$ (WLS|OLS) = %.2f|%.2f'%(r2_wls, r2_ols)], loc='upper left')

        # set minor ticks
        ax1.set_xticks(h_lag[:,0])
        ax0.xaxis.grid(True, which='major',linestyle='--')
        ax1.xaxis.grid(True, which='major',linestyle='--')

        # remove vertical gap between subplots
        plt.xlim(0, max_distance)
        plt.ylim(0,max(gamma[:,0])+0.3)
        plt.ylabel(r'Semivariance, $\gamma$ (%s)'%estimator_type)
        plt.xlabel('lag distance')
        plt.subplots_adjust(hspace=.0)
        plt.show(fig)
    else:
        pass

    return h_lag, n_obs, gamma, params, r2_wls, r2_ols

# Main function: multi fitting
def variofitmulti(
    df,
    values_col,
    index_col,
    coord_cols,
    distance_type,
    max_distance,
    bin_size,
    estimator_type,
    model_type,
    weight_fn=None,
    weight_params=None,
    xmax_factor=2.0,
    fix_nugget=True,
    fix_sill=False,
    plot_single =False,
    plot_summary =False,
):
    """
    Fit an experimental semivariogram per group in `values_index`.
    Fit an experimental semivariogram per group in `index_col`, and collect:
      - summary with n_samples, n_bins, r2_wls, r2_ols and fitted params
      - wide DataFrames for n_obs and gamma (first column is h_lag)

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing values, ids, and coordinate columns.
    values_col : str
        Column name for the target values (e.g., 'PGA').
    index_col : str
        Column name whose values define groups (e.g., 'evid').
    coord_cols : tuple[str, str] or list[str]
        If distance_type == 'geographic': (lat_col, lon_col) in degrees.
        If distance_type == 'cartesian' : (x, y)
        If distance_type == 'euclidean' : list of coordinate columns (1+).
    distance_type : {'geographic','euclidean'}
        Passed through to `variofit`.
    max_distance, bin_size, estimator_type, model_type, weight_fn, weight_params, plot
        Passed through to `variofit` unchanged.

    Returns
    -------
    summary : DataFrame
        One row per group with counts, mean/std, weighted R², and params.
    df_n_obs : DataFrame
        Wide matrix with column 'h_lag' followed by one column per group.
    df_gamma : DataFrame
        Same shape as df_n_obs, storing experimental semivariance.
    results : dict
        {group_id: (h_lag, n_obs, gamma, params, r2_wls, r_ols)}
    """
    results = {}
    summary_rows = []

    # full list of bin centers (global index for wide frames)
    nmax = int(np.ceil(float(max_distance) / float(bin_size)))
    full_h = (bin_size / 2.0) + np.arange(nmax, dtype=float) * float(bin_size)

    # guard against fp drift when matching later
    full_h = np.round(full_h, 12)

    # initialize wide frames with the global bin axis
    df_n_obs  = pd.DataFrame({"h_lag": full_h})
    df_gamma  = pd.DataFrame({"h_lag": full_h})
    param_keys = None

    gb = df.groupby(index_col, sort=False)

    for gid, gdf in tqdm(gb, total=gb.ngroups, desc="Fitting groups"):

        # values
        vals = gdf[values_col].to_numpy(dtype=float)
        mu   = float(np.mean(vals))
        sig  = float(np.std(vals, ddof = 1))

        # coordinates
        if distance_type == 'geographic':
            lat = gdf[coord_cols[0]].to_numpy(dtype=float)
            lon = gdf[coord_cols[1]].to_numpy(dtype=float)
            coords = np.column_stack([lat, lon])
        elif distance_type == 'euclidean':
            coords = gdf[list(coord_cols)].to_numpy(dtype=float)
        else:
            raise ValueError("distance_type must be 'geographic' or 'euclidean'")

        # call variofit main function
        res = variofit(
            values=vals,
            coordinates=coords,
            distance_type=distance_type,
            max_distance=max_distance,
            bin_size=bin_size,
            estimator_type=estimator_type,
            model_type=model_type,
            weight_fn=weight_fn,
            weight_params=weight_params,
            xmax_factor=xmax_factor,
            fix_nugget=fix_nugget,
            fix_sill=fix_sill,
            plot=plot_single,
        )

        h_lag, n_obs, gamma, params, r2_wls, r2_ols = res

        # store per-group result
        results[gid] = res

        # lock param column order from the first group
        if param_keys is None:
            param_keys = list(params.keys())

        # align this group's vectors to the global bin axis
        # round to avoid tiny fp mismatches
        h = np.round(h_lag.ravel(), 12)
        s_n = pd.Series(n_obs.ravel(), index=h)
        s_g = pd.Series(gamma.ravel(), index=h)

        df_n_obs[gid] = s_n.reindex(full_h).to_numpy()
        df_gamma[gid] = s_g.reindex(full_h).to_numpy()

        # summary row
        summary_rows.append({
            "values_index": gid,
            "n_samples": int(len(vals)),
            "mean": mu,
            "std": sig,
            "n_bins": int(h_lag.shape[0] if hasattr(h_lag, "shape") else len(h_lag)),
            "r2_wls": float(r2_wls),
            "r2_ols": float(r2_ols),
            **{k: float(params.get(k, np.nan)) for k in param_keys},
        })

    summary = pd.DataFrame(summary_rows, columns=(["values_index","n_samples", "mean", "std","n_bins","r2_wls","r2_ols"] + param_keys))

    if plot_summary:

        # long (tidy) data for plotting
        g_long = df_gamma.melt(id_vars='h_lag', var_name='group', value_name='gamma')
        n_long = df_n_obs.melt(id_vars='h_lag', var_name='group', value_name='n_obs')
        M = g_long.merge(n_long, on=['h_lag','group'])

        # drop NaNs so x, y, c lengths align
        M = M[M['gamma'].notna()]

        fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
        sc = ax.scatter(
            M['h_lag'].to_numpy(),
            M['gamma'].to_numpy(),
            c=M['n_obs'].to_numpy(),      # <-- single '=' and 1D
            s=12,
            cmap=plt.get_cmap("coolwarm"),
            norm=mpl.colors.LogNorm(),
            alpha=0.8,
            edgecolor='k',
            linewidths=0.1,
            zorder = 1
        )

        # semivariogram model fit plots
        xlag_fit = np.linspace(0.0, float(max_distance), 1000)
        fn = VARIOGRAM_MODELS[model_type]

        thetas = np.array(
            [theta_from_params(params_g, model_type)
             for _, (h_lag_g, n_obs_g, gamma_g, params_g, r2_wls_g, r2_ols_g) in results.items()],
            dtype=float
        )

        # mean and percentile parameter vectors
        theta_median = np.nanmedian(thetas, axis=0)
        theta_mean = np.nanmean(thetas, axis=0)

        # evaluate model with those θ on a smooth grid
        y_median = fn(xlag_fit, *theta_median)
        y_mean = fn(xlag_fit, *theta_mean)

        # get bounds from fitted curves
        Ys = []
        for gid, res_g in results.items():
            # res_g is (h_lag, n_obs, gamma, params, r2_wls, r2_ols) in your latest version
            try:
                _, _, _, params_g, *_ = res_g  # robust to trailing items
            except Exception:
                continue
            theta_g = theta_from_params(params_g, model_type)
            Ys.append(fn(xlag_fit, *theta_g))

        if Ys:
            Y = np.asarray(Ys, dtype=float)  # shape: (n_groups, len(xlag_fit))
            # pointwise percentiles across *curves*
            y_lo = np.nanpercentile(Y, 5.0, axis=0)
            y_hi = np.nanpercentile(Y, 95.0, axis=0)

            # ensure proper ordering & (optionally) enforce non-decreasing semivariograms
            lo = np.minimum(y_lo, y_hi)
            hi = np.maximum(y_lo, y_hi)
            y_lo = np.maximum.accumulate(lo)
            y_hi = np.maximum.accumulate(hi)
        else:
            y_lo = y_median.copy()
            y_hi = y_median.copy()

        # plot mean + band
        ax.plot(xlag_fit, y_mean, color='k', lw=1.5, label='mean fit', zorder=1000)
        ax.plot(xlag_fit, y_median, color='k', ls = '--', lw=1.5, label='median fit', zorder=1000)
        ax.fill_between(xlag_fit, y_lo, y_hi, color='forestgreen', alpha=0.15, label='5–95% fit', zorder=0)

        for gid, (h_lag_g, n_obs_g, gamma_g, params_g, r2_wls_g, r2_ols) in results.items():
            theta_g = theta_from_params(params_g, model_type)
            y_fit_g = fn(xlag_fit, *theta_g)
            ax.plot(xlag_fit, y_fit_g, '-k' ,lw=0.2, alpha=0.8)


        cb = plt.colorbar(sc, ax=ax, pad =0.02, fraction=0.04, aspect=40)
        cb.set_label('Observations per bin, n')
        ax.legend(loc='best', frameon=False)
        ax.set_xlabel("lag distance")
        ax.set_ylabel(r'Semivariance, $\gamma$ (%s)' % estimator_type)
        ax.set_xlim(0,max_distance)
        ax.set_ylim(0, float(M['gamma'].max()))
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.show()


    return summary, df_n_obs, df_gamma, results
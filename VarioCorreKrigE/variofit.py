"""
Utilities for estimating and fitting auto- and cross-semivariograms.

The module supports distance-based and angular lags, optional
correlation-form output, grouped fitting, and matrix-style summaries.
"""

import warnings
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
from sklearn.neighbors import BallTree

# Suppress noisy pandas performance warnings triggered by wide-frame operations.
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# plotting utils
def _set_ylim_from_points_and_fit(ax, gamma_pts, gamma_fit, allow_negative=False, pad_frac=0.05, min_pad=0.1):
    """
    Set a sensible y-range from experimental points and fitted values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to update.
    gamma_pts, gamma_fit : array_like
        Experimental ordinates and fitted ordinates already being plotted.
    allow_negative : bool, default False
        If True, keep both positive and negative values in view.
        If False, force the lower limit to 0.
    pad_frac : float, default 0.05
        Fractional padding based on the plotted data span.
    min_pad : float, default 0.1
        Minimum absolute padding added to the limits.
    """
    y = np.concatenate([np.asarray(gamma_pts).ravel(), np.asarray(gamma_fit).ravel()])
    y = y[np.isfinite(y)]
    if y.size == 0:
        ax.set_ylim(-1, 1)
        return

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    span = max(1e-6, y_max - y_min)
    pad = max(min_pad, pad_frac * span)

    if allow_negative:
        ax.set_ylim(y_min - pad, y_max + pad)
    else:
        ax.set_ylim(0, y_max + pad)

# Geographical Distance Function
def haversine_oq(lon1, lat1, lon2, lat2, radians=False, earth_rad=6371.227):
    """
    Compute great-circle distance using the haversine formula.

    Parameters
    ----------
    lon1, lat1 : array_like or float
        Longitudes and latitudes of the first set of locations.
    lon2, lat2 : array_like or float
        Longitudes and latitudes of the second set of locations.
    radians : bool, default False
        If False, inputs are assumed to be in degrees and are converted to radians.
    earth_rad : float, default 6371.227
        Earth radius in kilometres.

    Returns
    -------
    distance : ndarray
        Pairwise distance matrix in kilometres with shape (nlocs1, nlocs2).

    Notes
    -----
    This function preserves the original OpenQuake-style behaviour used elsewhere
    in the codebase.
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
def cross_matheron(dz1, dz2):
    """
    Classical (Matheron-style) experimental cross-semivariogram estimator.

        γ12(h) = 0.5 * mean( dz1 * dz2 )

    Parameters
    ----------
    dz1, dz2 : 1D arrays of float
        Paired increments for the two variables in a bin.

    Returns
    -------
    gamma12 : float
        Cross-semivariance estimate for this bin.
    """
    n = dz1.shape[0]
    if n == 0:
        return 0.0  # or np.nan, but we never call this with empty bins

    s = 0.0
    for k in range(n):
        s += dz1[k] * dz2[k]

    return 0.5 * (s / n)

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
def compute_distance_weights(h_lag, n_j, weight_type='inverse-linear weighting', weight_params=None):
    """
    Compute lag-bin weights for model fitting.

    The returned weights are applied at the bin level in the weighted least-squares
    objective. Most schemes combine a distance-decay term with the number of pairs
    falling in the bin.

    Parameters
    ----------
    h_lag : array_like, shape (k,)
        Lag-bin centres.
    n_j : array_like, shape (k,)
        Number of pairs in each lag bin.
    weight_type : {'inverse-linear weighting', 'inverse-linear squared weighting',
                   'exponential weighting', 'powered weighting',
                   'linear weighting', None, 'ols'}, default 'inverse-linear weighting'
        Weighting rule used to construct the bin weights.
    weight_params : list, tuple, dict, or None, default None
        Parameters controlling the chosen weighting rule.

        If a sequence is provided, the expected order is [b, alpha].
        If a dictionary is provided, the expected keys are {'b', 'alpha'}.

        The parameter `alpha` is only used for 'powered weighting'.

    Returns
    -------
    weights : ndarray, shape (k,)
        Weight assigned to each lag bin.

    Raises
    ------
    ValueError
        If the weighting scheme is unknown or required parameters are missing.
    """
    h_lag = np.asarray(h_lag, float)
    n_j = np.asarray(n_j, float)

    b = None
    alpha = None

    if isinstance(weight_params, dict):
        b = weight_params.get("b", None)
        alpha = weight_params.get("alpha", None)
    elif weight_params is not None:
        wp = list(weight_params)
        if len(wp) >= 1:
            b = wp[0]
        if len(wp) >= 2:
            alpha = wp[1]

    if weight_type == 'inverse-linear weighting':
        if b is None or b <= 0:
            raise ValueError("inverse-linear weighting requires weight_params['b'] > 0")
        w = n_j / (1.0 + h_lag / b)

    elif weight_type == 'exponential weighting':
        if b is None or b <= 0:
            raise ValueError("exponential weighting requires weight_params['b'] > 0")
        w = n_j * np.exp(-h_lag / b)

    elif weight_type == 'powered weighting':
        if b is None or b <= 0 or alpha is None:
            raise ValueError("powered weighting requires weight_params with b > 0 and alpha")
        w = n_j * (1.0 + h_lag / b) ** (-alpha)

    elif weight_type == 'linear weighting':
        w = n_j * np.ones_like(h_lag, dtype=float)

    elif weight_type == 'inverse-linear squared weighting':
        w = np.where(h_lag > 0.0, n_j / h_lag**2, 0.0)

    elif weight_type is None or weight_type == 'ols':
        w = np.ones_like(h_lag, dtype=float)

    else:
        raise ValueError(
            "Invalid weight_type: choose None/'ols', 'inverse-linear weighting', "
            "'inverse-linear squared weighting', 'exponential weighting', "
            "'powered weighting' or 'linear weighting'"
        )

    return w

# Objective Function(s) for Fitting
def _expand_theta_fixed_total_sill(model_type, theta_free):
    """
    Expand the reduced parameter vector used under the constraint c0 + b = 1.

    When the total sill is fixed to 1 and the nugget is free, the optimizer works
    with a reduced parameter vector. This helper reconstructs the full parameter
    vector expected by the variogram model.

    Parameters
    ----------
    model_type : str
        Name of the variogram model.
    theta_free : array_like
        Reduced parameter vector used in the constrained optimisation.

    Returns
    -------
    theta_full : list[float]
        Full parameter vector in the order expected by the selected model.
    """
    th = list(np.asarray(theta_free, float).ravel())

    if model_type in ("spherical", "exponential", "gaussian", "cubic"):
        # full: (r, c0, b)
        r, b = th
        c0 = 1.0 - b
        return [r, c0, b]

    elif model_type == "powered_exponential":
        # full: (r, c0, beta, b)
        r, beta, b = th
        c0 = 1.0 - b
        return [r, c0, beta, b]

    elif model_type == "matern":
        # full: (r, c0, s, b)
        r, s, b = th
        c0 = 1.0 - b
        return [r, c0, s, b]

    elif model_type in ("damped_cosine_angle", "angular_dissimilarity"):
        # full: (c, c0, b)
        c, b = th
        c0 = 1.0 - b
        return [c, c0, b]

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _compress_init_bounds_fixed_total_sill(model_type, x0_full, bounds_full):
    """
    Convert full initial values and bounds to the reduced form used when c0 + b = 1.

    This is the inverse companion to `_expand_theta_fixed_total_sill()`. It removes
    the redundant sill parameter so the optimiser only sees the free parameters.

    Parameters
    ----------
    model_type : str
        Name of the variogram model.
    x0_full : sequence of float
        Initial parameter vector in the full model parameterisation.
    bounds_full : sequence of tuple
        Bounds in the full model parameterisation.

    Returns
    -------
    x0 : list[float]
        Reduced initial parameter vector.
    bounds : list[tuple]
        Reduced bounds aligned with `x0`.
    """
    x0_full = list(np.asarray(x0_full, float).ravel())
    bounds_full = list(bounds_full)

    b_bounds = (0.0, 1.0)

    if model_type in ("spherical", "exponential", "gaussian", "cubic"):
        # full: (r, c0, b) -> free: (r, b)
        x0 = [x0_full[0], np.clip(x0_full[2], 0.0, 1.0)]
        bounds = [bounds_full[0], b_bounds]

    elif model_type == "powered_exponential":
        # full: (r, c0, beta, b) -> free: (r, beta, b)
        x0 = [x0_full[0], x0_full[2], np.clip(x0_full[3], 0.0, 1.0)]
        bounds = [bounds_full[0], bounds_full[2], b_bounds]

    elif model_type == "matern":
        # full: (r, c0, s, b) -> free: (r, s, b)
        x0 = [x0_full[0], x0_full[2], np.clip(x0_full[3], 0.0, 1.0)]
        bounds = [bounds_full[0], bounds_full[2], b_bounds]

    elif model_type in ("damped_cosine_angle", "angular_dissimilarity"):
        # full: (c, c0, b) -> free: (c, b)
        x0 = [x0_full[0], np.clip(x0_full[2], 0.0, 1.0)]
        bounds = [bounds_full[0], b_bounds]

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return x0, bounds


def _objective_func_fixed_total_sill(theta_free, h, g, weights, model_type, semivariomodel_fn):
    """
    Objective under the constraint c0 + b = 1.
    """
    theta_full = _expand_theta_fixed_total_sill(model_type, theta_free)
    return objective_func(theta_full, h, g, weights, semivariomodel_fn)

def objective_func(params, h, gamma, weights, semivario_fn):

    """
    Weighted least-squares objective used for variogram fitting.

    The objective is

        sum_i w_i [gamma_i - model(h_i; theta)]^2.

    Parameters
    ----------
    params : sequence of float
        Model parameters in the order expected by `semivario_fn`.
    h : array_like, shape (k,)
        Lag-bin centres.
    gamma : array_like, shape (k,)
        Experimental ordinates at the bin centres.
    weights : array_like, shape (k,)
        Bin weights.
    semivario_fn : callable
        Model function with signature `semivario_fn(h, *params)`.

    Returns
    -------
    float
        Weighted sum of squared residuals.
    """

    gamma_pred = semivario_fn(h, *params)
    return np.sum(weights * (gamma - gamma_pred)**2)

def make_init_and_bounds(model, h, gamma, xmax_factor=2.0, fix_nugget=True, fix_sill=False):
    """
    Build initial values and parameter bounds for a chosen variogram model.

    This is an internal helper used by the fitting routines. At this helper level,
    `fix_sill=True` means the partial sill parameter `c0` is fixed at 1. In the
    higher-level fitting functions, additional logic may instead enforce the total
    sill constraint `c0 + b = 1` when the nugget is free.

    Parameters
    ----------
    model : {'spherical', 'exponential', 'gaussian', 'cubic',
             'powered_exponential', 'matern', 'damped_cosine_angle',
             'angular_dissimilarity'}
        Variogram model name.
    h : array_like, shape (k,)
        Lag-bin centres.
    gamma : array_like, shape (k,)
        Experimental ordinates at the lag-bin centres.
    xmax_factor : float, default 2.0
        Multiplier used to cap the upper bound of the range-like parameter.
    fix_nugget : bool, default True
        If True, fix the nugget parameter `b` at 0.
    fix_sill : bool, default False
        If True, fix the partial sill parameter `c0` at 1 within this helper.

    Returns
    -------
    x0 : tuple
        Initial parameter vector.
    bounds : tuple of tuple
        Bounds aligned with `x0`.

    Notes
    -----
    Range-like parameters are lower-bounded away from zero and upper-bounded by
    `xmax_factor * max(h)` to reduce unstable near-flat fits.
    """

    h = np.asarray(h, float).ravel()
    g = np.asarray(gamma, float).ravel()

    # robust lag scales
    mask_pos = np.isfinite(h) & (h > 0)
    h_min = float(np.nanmin(h[mask_pos])) if np.any(mask_pos) else 1.0
    h_max = float(np.nanmax(h[mask_pos])) if np.any(mask_pos) else 1.0

    # simple inits
    r0 = 0.25 * h_max                       # range start
    b0 = float(np.nanmin(g))               # nugget start
    c0 = max(float(np.nanmax(g) - b0), 1e-9)  # partial sill

    # range bounds: positive, upper-capped by xmax_factor*max(h)
    r_lo = max(1e-3, 0.25 * h_min)
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
    Pack a fitted parameter vector into a named dictionary.

    The returned dictionary uses parameter names consistent with the selected model,
    which makes later plotting, reporting, and transformation steps easier to read.

    Parameters
    ----------
    model_type : str
        Name of the fitted variogram model.
    theta : sequence of float
        Parameter vector in the order expected by that model.

    Returns
    -------
    params : dict
        Dictionary mapping parameter names to fitted values.
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

def _validate_transform(transform):
    if transform not in (None, "correlation"):
        raise ValueError("transform must be None or 'correlation'")


def _get_total_sill_from_params(params, eps=1e-12):
    """
    Extract the total sill c0 + b from a parameter dictionary.

    Parameters
    ----------
    params : dict
        Parameter dictionary expected to contain at least `c0` and optionally `b`.
    eps : float, default 1e-12
        Small tolerance used to reject zero or negative sill values.

    Returns
    -------
    sill : float
        Total sill c0 + b.

    Raises
    ------
    ValueError
        If the total sill is missing, non-finite, or not strictly positive.
    """
    params = {} if params is None else dict(params)
    c0 = float(params.get("c0", np.nan))
    b = float(params.get("b", 0.0))
    sill = c0 + b

    if not np.isfinite(sill) or sill <= eps:
        raise ValueError("Total sill c0 + b must be finite and > 0 for correlation transform.")
    return sill


def _gamma_to_correlation(gamma, sill):
    """
    Convert semivariogram ordinates to correlation ordinates.

    The transformation is

        rho(h) = 1 - gamma(h) / sill.

    Parameters
    ----------
    gamma : array_like
        Semivariogram ordinates.
    sill : float
        Positive sill used for normalization.

    Returns
    -------
    rho : ndarray
        Correlation ordinates on the same shape as `gamma`.
    """
    gamma = np.asarray(gamma, float)
    return 1.0 - gamma / float(sill)


def _pack_corr_params_from_vario(model_type, params, eps=1e-12):
    """
    Normalize fitted semivariogram parameters into correlation form.

    The range and shape parameters are left unchanged. The sill-related parameters
    are rescaled so that

        c0 + b = 1.

    Parameters
    ----------
    model_type : str
        Name of the fitted model. Included for interface consistency.
    params : dict
        Fitted semivariogram parameters.
    eps : float, default 1e-12
        Tolerance passed to the sill check.

    Returns
    -------
    params_corr : dict
        Parameter dictionary in normalized correlation form.
    """
    p = dict(params)
    sill = _get_total_sill_from_params(p, eps=eps)

    p["c0"] = float(p["c0"]) / sill
    p["b"]  = float(p["b"])  / sill
    return p


def _renormalize_correlation_params(params, eps=1e-12):
    """
    Used only for aggregated mean/median params in summary plots so that
    the correlation-form params remain on the c0 + b = 1 constraint.
    """
    p = dict(params)
    if ("c0" in p) and ("b" in p):
        s = float(p["c0"]) + float(p["b"])
        if np.isfinite(s) and s > eps:
            p["c0"] = float(p["c0"]) / s
            p["b"]  = float(p["b"])  / s
    return p


def _evaluate_model_from_params(model_type, h, params, transform=None):
    """
    Evaluate a fitted model from its parameter dictionary.

    Parameters
    ----------
    model_type : str
        Name of the fitted model.
    h : array_like
        Lag values at which the model should be evaluated.
    params : dict
        Parameter dictionary for the selected model.
    transform : {None, 'correlation'}, default None
        If None, evaluate the model in semivariogram space.
        If 'correlation', evaluate the normalized correlation-form model.

    Returns
    -------
    y : ndarray
        Model ordinates evaluated at `h`.

    Notes
    -----
    For `transform='correlation'`, `params` are assumed to already be normalized so
    that c0 + b = 1.
    """
    h = np.asarray(h, float)
    theta = theta_from_params(params, model_type)
    gamma_like = VARIOGRAM_MODELS[model_type](h, *theta)

    if transform is None:
        return gamma_like

    elif transform == "correlation":
        # Since c0+b=1 in the normalized correlation-form params,
        # the positive-lag correlation is just 1 - gamma(h).
        rho = 1.0 - gamma_like

        # exact zero-lag value is 1, not the 0+ limit
        rho0 = float(params.get("c0", 0.0)) + float(params.get("b", 0.0))
        rho = np.where(h == 0.0, rho0, rho)
        return rho

    else:
        raise ValueError("transform must be None or 'correlation'")


def _plot_correlation_model_piecewise(ax, x, y, params, color="k", lw=2.0, ls="-",
                                      label=None, zorder=4, alpha_plot=1.0,
                                      show_zero_point=True, jump_ls=":"):
    """
    Plot a correlation model with explicit nugget discontinuity:

        rho(0)  = 1
        rho(0+) = 1 - b

    Here params are assumed to be in normalized correlation form:
        c0 + b = 1
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    b0 = float(params.get("b", 0.0))
    has_nugget = np.isfinite(b0) and (b0 > 1e-12)

    pos = x > 0

    if has_nugget:
        ax.plot(
            x[pos], y[pos],
            color=color, lw=lw, ls=ls, label=label,
            zorder=zorder, alpha=alpha_plot
        )

        ax.plot(
            [0.0, 0.0], [1.0, 1.0 - b0],
            color=color, lw=max(1.0, lw * 0.8), ls=jump_ls,
            zorder=zorder, alpha=alpha_plot
        )

        if show_zero_point:
            ax.plot(
                0.0, 1.0, "o",
                ms=4, mfc="white", mec=color,
                zorder=zorder + 0.1, alpha=alpha_plot
            )
    else:
        ax.plot(
            x, y,
            color=color, lw=lw, ls=ls, label=label,
            zorder=zorder, alpha=alpha_plot
        )

# Pair construction and binning helpers
def _make_lag_axis(nmax, bin_size, lag_repr="center"):
    """
    Construct the representative lag value for each equal-width bin.

    Parameters
    ----------
    nmax : int
        Number of bins.
    bin_size : float
        Bin width.
    lag_repr : {"center", "edge", "upper"}, default "center"
        Representative lag attached to each bin:
        - "center": midpoint of [k*bin_size, (k+1)*bin_size)
        - "edge" / "upper": upper edge of [k*bin_size, (k+1)*bin_size)

    Returns
    -------
    h_full : ndarray, shape (nmax,)
        Representative lag values for bins 0..nmax-1.
    """
    lag_repr = str(lag_repr).lower().strip()

    if lag_repr == "center":
        return (bin_size / 2.0) + np.arange(nmax, dtype=float) * float(bin_size)
    elif lag_repr in ("edge", "upper"):
        return np.arange(1, nmax + 1, dtype=float) * float(bin_size)
    else:
        raise ValueError("lag_repr must be one of {'center', 'edge', 'upper'}.")

def _build_pair_arrays(values, coords, distance_type, max_distance, bin_size,
                       balltree_leaf_size=40, max_neighbors=None):
    """
    Construct pairwise distance and increment arrays for experimental variogram fitting.

    For geographic, cartesian, and euclidean distances, this helper first attempts
    to build the pair list using a BallTree radius search. If that path does not
    yield any valid pairs, it falls back to a dense pairwise-distance calculation.

    Parameters
    ----------
    values : array_like, shape (n,)
        Observation values.
    coords : array_like, shape (n, d)
        Coordinates associated with the observations.
    distance_type : {'geographic', 'geographical', 'cartesian', 'euclidean', 'angular'}
        Distance metric used to define lags.
    max_distance : float
        Maximum lag distance retained.
    bin_size : float
        Lag-bin width. Included for API symmetry with the binning step; pair
        retention here is controlled by `max_distance`.
    balltree_leaf_size : int, default 40
        Leaf size used by BallTree where applicable.
    max_neighbors : int or None, default None
        Optional cap on the number of neighbours retained per point when BallTree
        is used.

    Returns
    -------
    d : ndarray
        Pairwise distances for the retained i < j pairs.
    dz : ndarray
        Absolute increments |z_i - z_j| for the same retained pairs.
    """
    values = np.asarray(values, float)
    coords = np.asarray(coords, float)
    n = len(values)

    dt = str(distance_type).lower()
    EARTH_RAD = 6371.227  # km

    d = None
    dz = None

    use_bt = dt in ("geographic", "geographical", "cartesian", "euclidean")

    if use_bt:
        if dt in ("geographic", "geographical"):
            lat_deg = coords[:, 0]
            lon_deg = coords[:, 1]
            lat_rad = np.deg2rad(lat_deg)
            lon_rad = np.deg2rad(lon_deg)
            pts = np.column_stack([lat_rad, lon_rad])

            tree = BallTree(pts, metric="haversine", leaf_size=balltree_leaf_size)
            radius = (float(max_distance) + 0.5 * float(bin_size)) / EARTH_RAD

            ind_list, dist_list = tree.query_radius(
                pts, r=radius, return_distance=True, sort_results=True
            )

            d_list = []
            dz_list = []
            for i in range(n):
                inds = ind_list[i]
                dists_rad = dist_list[i]

                mask = inds > i
                if not np.any(mask):
                    continue

                js = inds[mask]
                d_ij = dists_rad[mask] * EARTH_RAD

                if max_neighbors is not None and max_neighbors > 0:
                    js = js[:max_neighbors]
                    d_ij = d_ij[:max_neighbors]

                d_list.append(d_ij)
                dz_list.append(np.abs(values[i] - values[js]))

            if d_list:
                d = np.concatenate(d_list)
                dz = np.concatenate(dz_list)

        elif dt in ("cartesian", "euclidean"):
            pts = coords
            tree = BallTree(pts, metric="euclidean", leaf_size=balltree_leaf_size)
            radius = float(max_distance) + 0.5 * float(bin_size)

            ind_list, dist_list = tree.query_radius(
                pts, r=radius, return_distance=True, sort_results=True
            )

            d_list = []
            dz_list = []
            for i in range(n):
                inds = ind_list[i]
                dists = dist_list[i]

                mask = inds > i
                if not np.any(mask):
                    continue

                js = inds[mask]
                d_ij = dists[mask]

                if max_neighbors is not None and max_neighbors > 0:
                    js = js[:max_neighbors]
                    d_ij = d_ij[:max_neighbors]

                d_list.append(d_ij)
                dz_list.append(np.abs(values[i] - values[js]))

            if d_list:
                d = np.concatenate(d_list)
                dz = np.concatenate(dz_list)

    # fallback to dense matrix
    if (d is None) or (dz is None) or (d.size == 0):
        if dt in ("geographic", "geographical"):
            lat = coords[:, 0]
            lon = coords[:, 1]
            dist_full = np.asarray(
                haversine_oq(lon, lat, lon, lat, radians=False, earth_rad=EARTH_RAD),
                dtype=float,
            )
        elif dt == "cartesian":
            if coords.shape[1] != 2:
                raise ValueError("cartesian requires coordinates shape (n,2): (x, y)")
            dx = coords[:, None, 0] - coords[None, :, 0]
            dy = coords[:, None, 1] - coords[None, :, 1]
            dist_full = np.hypot(dx, dy)
        elif dt == "euclidean":
            diff = coords[:, None, :] - coords[None, :, :]
            dist_full = np.linalg.norm(diff, axis=-1)
        elif dt == "angular":
            theta_deg = np.asarray(coords, float)
            if theta_deg.ndim == 2:
                if theta_deg.shape[1] != 1:
                    raise ValueError("angular distance requires a single angular coordinate per row.")
            theta = np.radians(theta_deg.ravel())
            cos_diff = np.cos(theta[:, None] - theta[None, :])
            ang_rad = np.arccos(np.clip(cos_diff, -1.0, 1.0))
            dist_full = np.degrees(ang_rad)
        else:
            raise ValueError(
                "Invalid distance_type: choose 'geographic', 'geographical', "
                "'cartesian', 'angular', or 'euclidean'"
            )

        iu, ju = np.triu_indices(n, k=1)
        d = dist_full[iu, ju]
        dz = np.abs(values[iu] - values[ju])

    return d, dz

def _binned_semivariogram_from_pairs(
    d,
    dz,
    bin_size,
    max_distance,
    estimator_type,
    semivarioest_fn,
    lag_repr="center",
):
    """
    Bin pairwise distances and increments into experimental semivariogram ordinates.

    Pair membership is defined by interval binning:
        bin k contains distances in [k*bin_size, (k+1)*bin_size),
    with the final bin capped at `max_distance`.

    Parameters
    ----------
    d : array_like
        Pairwise distances.
    dz : array_like
        Pairwise increments |z_i - z_j|.
    bin_size : float
        Bin width.
    max_distance : float
        Maximum retained lag distance.
    estimator_type : {'Matheron', 'CressieHawkins', 'Dowd'}
        Experimental semivariogram estimator.
    semivarioest_fn : callable
        Bin-level estimator used when `estimator_type != "Matheron"`.
    lag_repr : {"center", "edge", "upper"}, default "center"
        Representative lag attached to each bin:
        - "center": midpoint of [k*bin_size, (k+1)*bin_size)
        - "edge"/"upper": upper edge of [k*bin_size, (k+1)*bin_size)

    Returns
    -------
    h_full : ndarray, shape (nmax, 1)
        Representative lag values for all bins.
    counts_full : ndarray, shape (nmax, 1)
        Pair counts for all bins.
    gamma_full : ndarray, shape (nmax, 1)
        Experimental semivariance for all bins; NaN for empty bins.
    h_valid : ndarray, shape (k, 1)
        Representative lag values for non-empty bins.
    counts_valid : ndarray, shape (k, 1)
        Pair counts for non-empty bins.
    gamma_valid : ndarray, shape (k, 1)
        Experimental semivariance for non-empty bins.
    """
    nmax = int(np.ceil(float(max_distance) / float(bin_size)))
    if nmax <= 0:
        raise ValueError("max_distance / bin_size must be > 0.")

    d = np.asarray(d, dtype=float).ravel()
    dz = np.asarray(dz, dtype=float).ravel()

    if d.shape != dz.shape:
        raise ValueError("d and dz must have the same shape.")

    mask = np.isfinite(d) & np.isfinite(dz) & (d >= 0.0) & (d <= float(max_distance))
    if not np.any(mask):
        raise ValueError(
            "No pair distances fell into [0, max_distance]; check max_distance/bin_size."
        )

    d_use = d[mask]
    dz_use = dz[mask]

    bin_idx = np.floor(d_use / float(bin_size)).astype(int)
    bin_idx = np.minimum(bin_idx, nmax - 1)

    h_full = _make_lag_axis(nmax, bin_size, lag_repr=lag_repr)
    counts = np.bincount(bin_idx, minlength=nmax).astype(float)

    gamma_full = np.full(nmax, np.nan, dtype=float)

    if estimator_type == "Matheron":
        sum_w = np.bincount(bin_idx, weights=0.5 * dz_use**2, minlength=nmax)
        with np.errstate(invalid="ignore", divide="ignore"):
            gamma_full = sum_w / counts
        gamma_full[counts == 0] = np.nan
    else:
        for k in range(nmax):
            mask_k = (bin_idx == k)
            if not np.any(mask_k):
                continue
            gamma_full[k] = semivarioest_fn(dz_use[mask_k])

    keep = np.isfinite(gamma_full)

    return (
        h_full.reshape(-1, 1),
        counts.reshape(-1, 1),
        gamma_full.reshape(-1, 1),
        h_full[keep].reshape(-1, 1),
        counts[keep].reshape(-1, 1),
        gamma_full[keep].reshape(-1, 1),
    )

def _bootstrap_summary(arr, qlo=2.5, qhi=97.5):
    """
    Summarise bootstrap samples column-wise.

    Parameters
    ----------
    arr : array_like
        Bootstrap samples with shape (n_boot, n_points) or (n_points,).
    qlo, qhi : float, default 2.5, 97.5
        Lower and upper percentiles used for the interval summary.

    Returns
    -------
    mean : ndarray
        Column-wise mean across valid bootstrap samples.
    q_low : ndarray
        Lower percentile bound.
    q_high : ndarray
        Upper percentile bound.
    n_valid : int
        Number of bootstrap rows containing at least one finite value.
    """
    arr = np.asarray(arr, float)
    if arr.ndim == 1:
        arr = arr[None, :]

    valid_rows = np.any(np.isfinite(arr), axis=1)
    if not np.any(valid_rows):
        out = np.full(arr.shape[1], np.nan, dtype=float)
        return out, out.copy(), out.copy(), 0

    arrv = arr[valid_rows]
    return (
        np.nanmean(arrv, axis=0),
        np.nanpercentile(arrv, qlo, axis=0),
        np.nanpercentile(arrv, qhi, axis=0),
        int(arrv.shape[0]),
    )


def variofit(values, coordinates, distance_type, max_distance, bin_size, estimator_type, model_type, weight_fn,
             weight_params, xmax_factor=2.0, fix_nugget=True, fix_sill=False, plot=False, plot_path=None,
             balltree_leaf_size=40, max_neighbors=None,
             bootstrap=None, bootstrap_method="pair", bootstrap_ci=(2.5, 97.5), random_state=None,
             transform=None, lag_repr="center"):
    """
    Compute an experimental semivariogram and fit a user-selected variogram model,
    optionally with bootstrap resampling for uncertainty quantification.

    Parameters
    ----------
    values : array_like, shape (n,)
        Sample values z_i at each coordinate.
    coordinates : array_like, shape (n, d)
        Sample locations.
        If distance_type == 'geographic', coordinates[:, 0] = latitude (deg),
        coordinates[:, 1] = longitude (deg).
        If distance_type == 'cartesian', coordinates are (x, y).
        If distance_type == 'euclidean', coordinates are in linear units
        (e.g., km, m, or dimensionless).
    distance_type : {'geographic', 'cartesian', 'euclidean', 'angular'}
        Distance metric for lag computation.
        'geographic' uses haversine distance over a sphere
        (Earth radius 6371.227 km), returning lags in km.
        'cartesian' uses Euclidean distance on (x, y).
        'euclidean' uses the Euclidean norm in the coordinate space provided.
        'angular' expects site angles in degrees as input and computes angular
        separation lags in degrees.
    max_distance : float
        Maximum lag to include (same units as the chosen distance_type).
    bin_size : float
        Width of each lag bin (same units as `max_distance`).

    lag_repr : {'center', 'edge', 'upper'}, default 'center'
        Representative lag attached to each equal-width bin:
        - 'center': midpoint of [k*bin_size, (k+1)*bin_size)
        - 'edge'/'upper': upper edge of [k*bin_size, (k+1)*bin_size)

        This affects the x-values used for plotting, weighting, and fitting, but
        does not change which pairs fall into each bin.
    estimator_type : {'Matheron', 'CressieHawkins', 'Dowd'}
        Semivariogram estimator applied within each lag bin using increments
        |z_i - z_j|.
    model_type : {'exponential', 'cubic', 'powered_exponential', 'matern',
                  'gaussian', 'spherical', 'damped_cosine_angle',
                  'angular_dissimilarity'}
        Variogram model to fit. See Notes for parameterizations.
    weight_fn : {None, 'ols', 'inverse-linear weighting',
                 'exponential weighting', 'powered weighting',
                 'linear weighting', inverse-linear squared weighting}
        Bin weight scheme for fitting. None or 'ols' gives equal weights per bin.
        Other schemes multiply a distance-decay weight by the bin pair count.
    weight_params : list or dict or None
        Parameters for the chosen `weight_fn`.
        If list, expected format is [b, alpha].
        If dict, expected keys are {'b', 'alpha'}.
        `alpha` is ignored for 'inverse-linear weighting' and
        'exponential weighting'.
        If None, defaults to b = 0.25 * max(h), alpha = 1.0.
    xmax_factor : float, default 2.0
        Scaling factor used when constructing upper bounds and initial guesses
        for the model range parameter in `make_init_and_bounds`.
    fix_nugget : bool, default True
        If True, fixes the nugget (b) to 0.0 during fitting.
    fix_sill : bool, default False
        If True, fixes the total sill (c0 + b) to 1.0 during fitting.
        If the nugget is also free, the partial sill is constrained as c0 = 1 - b.
    plot : bool, default False
        If True, shows a two-panel plot of lag counts and the experimental
        semivariogram with the fitted model. When bootstrapping is enabled,
        the plot additionally shows bootstrap sample curves, the bootstrap mean,
        and the percentile confidence band.
    plot_path : str or None, default None
        Optional file path for saving the plot. If None, the plot is shown
        interactively.
    balltree_leaf_size : int, default 40
        Leaf size for BallTree. Only used for
        'geographic', 'cartesian', and 'euclidean' distance types.
    max_neighbors : int or None, default None
        If not None, caps the number of neighbors per point when using BallTree.
    bootstrap : int or None, default None
        Number of bootstrap replicates. If None, no bootstrapping is performed
        and only the fitted experimental semivariogram is returned.
    bootstrap_method : {'pair', 'point'}, default 'pair'
        Bootstrap resampling scheme used when `bootstrap` is not None.

        - 'pair' resamples the already-constructed pair list (distance and
          increment pairs) with replacement.
        - 'point' resamples observation points with replacement and rebuilds
          the pair list from the resampled dataset.

        The 'pair' method is typically more stable, whereas 'point' is usually
        more conservative because it perturbs the underlying site configuration.
    bootstrap_ci : tuple(float, float), default (2.5, 97.5)
        Lower and upper percentile bounds used for the bootstrap confidence
        interval, e.g. (5, 95) for a 5th–95th percentile interval.
    random_state : int or None, default None
        Seed for reproducible bootstrap resampling.
    transform : {None, 'correlation'}, default None
        Output transform applied after fitting.

        - None:
            Return the usual semivariogram outputs.
        - 'correlation':
            Keep fitting in semivariogram space, but return:
              * experimental ordinates transformed to correlation space
              * fitted parameters in normalized correlation-form
              * fitted/bootstrapped curves evaluated in correlation space

        For the auto-case, the transformed parameterization uses:
            c0_corr = c0 / (c0 + b)
            b_corr  = b  / (c0 + b)
        with all range/shape parameters unchanged.

    Returns
    -------
    h_lag : ndarray, shape (k, 1)
        Representative lag values of the non-empty bins, according to `lag_repr`.
    n_obs : ndarray, shape (k, 1)
        Pair counts per bin.
    gamma : ndarray, shape (k, 1)
        Experimental semivariogram values per bin.
    params : dict
        Dictionary of fitted model parameters (keys depend on `model_type`).

        If bootstrapping is enabled, bootstrap results are also stored under
        `params["bootstrap"]`, including the bootstrap sample curves, bootstrap
        mean, percentile confidence intervals, and sampled fitted parameters.
        If bootstrapping is disabled, `params["bootstrap"]` is None.
    r2_wls : float
        Weighted R² computed at the representative lag values using the same
        weights passed to the fitting objective.
    r2_ols : float
        Ordinary R² computed at the representative lag values using equal weights.

    Notes
    -----
    - Estimators:
      * Matheron:
            gamma(h) = 0.5 * mean((z_i - z_j)^2)
      * Cressie–Hawkins:
            robust estimator with small-sample bias correction.
      * Dowd:
            median-based robust estimator.

    - Model parameterizations (following the exact implementations used here):
      * exponential:
            gamma(h) = b + c0 * (1 - exp(-h / a)), with internal a = r / 3
      * gaussian:
            gamma(h) = b + c0 * (1 - exp(-(h / a)^2)), with a = r / 2
      * spherical:
            compact-support model, where r is the effective range
      * cubic:
            compact-support model, where r is the effective range
      * powered_exponential:
            gamma(h) = b + c0 * (1 - exp(-(h / a)^beta)),
            with a = r / 3^(1 / beta)
      * matern:
            gamma(h) = b + c0 * [1 - (2 / Gamma(s)) * ((h * sqrt(s)) / a)^s
            * K_s(2 * (h * sqrt(s)) / a)], with a = r / 2
      * damped_cosine_angle:
            gamma(theta) = b + c0 * [1 - cos(theta) * exp(-theta / c)]
      * angular_dissimilarity:
            angular semivariogram based on angular lag in degrees, with c
            representing a damping angle in degrees

    - Angular vs distance lags:
      * 'damped_cosine_angle' and 'angular_dissimilarity' expect angular lags
        in degrees. If distance-based lags are passed, the fitted model will
        not be meaningful.

    - Weights:
      * 'inverse-linear weighting':
            n_j * 1 / (1 + h / b)
      * 'exponential weighting':
            n_j * exp(-h / b)
      * 'powered weighting':
            n_j * (1 + h / b)^(-alpha)
      * 'linear weighting':
            n_j
      * 'ols':
            1
        These are applied at the bin level, with distance-based weights
        multiplied by the corresponding bin pair counts.

    - Optimization:
      The fit minimizes

          sum_i w_i * [gamma_i - model(h_i; theta)]^2

      with bounds and initial guesses chosen by `make_init_and_bounds`.

    - Bootstrap interpretation:
      * 'pair' bootstrap quantifies uncertainty conditional on the observed
        pair structure and is generally smoother and more stable.
      * 'point' bootstrap quantifies uncertainty at the observation/site level
        and is usually more variable because the pair structure is rebuilt each
        replicate.
    """

    _validate_transform(transform)

    values = np.asarray(values, float)
    coords = np.asarray(coordinates, float)
    n = len(values)

    if n < 2:
        raise ValueError("Need at least 2 points to compute a variogram.")

    nmax = int(np.ceil(float(max_distance) / float(bin_size)))
    if nmax <= 0:
        raise ValueError("max_distance / bin_size must be > 0.")

    # estimator
    if estimator_type == "Matheron":
        semivarioest_fn = matheron
    elif estimator_type == "CressieHawkins":
        semivarioest_fn = cressie_hawkins
    elif estimator_type == "Dowd":
        semivarioest_fn = dowd
    else:
        raise ValueError("Invalid estimator: choose from 'Matheron', 'CressieHawkins', or 'Dowd'")

    # model
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
        raise ValueError(
            "Invalid Model: Choose from 'exponential', 'cubic', 'powered_exponential', "
            "'matern', 'spherical', 'gaussian', 'angular_dissimilarity' or 'damped_cosine_angle'"
        )

    # -------------------------------------------------
    # main pair list and experimental semivariogram
    # -------------------------------------------------
    d, dz = _build_pair_arrays(
        values, coords, distance_type, max_distance, bin_size,
        balltree_leaf_size=balltree_leaf_size,
        max_neighbors=max_neighbors,
    )

    h_full, n_full, gamma_full, h_lag, n_obs, gamma = _binned_semivariogram_from_pairs(
        d, dz, bin_size, max_distance, estimator_type, semivarioest_fn, lag_repr=lag_repr
    )

    if h_lag.size == 0:
        raise ValueError(
            "No valid lag bins with a defined experimental semivariance were found. "
            "Check the data, bin_size, max_distance, and the available pair structure."
        )

    # -------------------------------------------------
    # main fit
    # -------------------------------------------------
    h = h_lag.ravel()
    g = gamma.ravel()
    m = n_obs.ravel()

    if weight_fn is None or str(weight_fn).lower() == "ols":
        weights = np.ones_like(h, dtype=float)
    else:
        if weight_params is None:
            weight_params_fit = [0.25 * float(h.max()) if h.size else 1.0, 1.0]
        elif isinstance(weight_params, dict):
            b = weight_params.get("b", 0.25 * float(h.max()) if h.size else 1.0)
            alpha = weight_params.get("alpha", 1.0)
            weight_params_fit = [b, alpha]
        else:
            weight_params_fit = weight_params

        weights = compute_distance_weights(
            h, m, weight_type=weight_fn, weight_params=weight_params_fit
        )

    if fix_sill and not fix_nugget:
        # Enforce total sill = 1, so c0 = 1 - b
        x0_full, bounds_full = make_init_and_bounds(
            model_type, h, g, xmax_factor, fix_nugget=False, fix_sill=False
        )
        x0, bounds = _compress_init_bounds_fixed_total_sill(model_type, x0_full, bounds_full)

        res = minimize(
            fun=lambda th: _objective_func_fixed_total_sill(
                th, h, g, weights, model_type, semivariomodel_fn
            ),
            x0=x0,
            bounds=bounds,
        )
        if not res.success:
            raise RuntimeError(f"Variogram-model optimization failed: {res.message}")
        theta_hat = np.asarray(_expand_theta_fixed_total_sill(model_type, res.x), float)

    else:
        x0, bounds = make_init_and_bounds(
            model_type, h, g, xmax_factor, fix_nugget, fix_sill
        )

        res = minimize(
            fun=lambda th: objective_func(th, h, g, weights, semivariomodel_fn),
            x0=x0,
            bounds=bounds,
        )
        if not res.success:
            raise RuntimeError(f"Variogram-model optimization failed: {res.message}")
        theta_hat = np.asarray(res.x, float)

    # raw semivariogram-space fit diagnostics
    g_fit_bins_raw = semivariomodel_fn(h, *theta_hat)
    r2_wls = r2_score_weighted(g, g_fit_bins_raw, w=weights)
    r2_ols = r2_score_weighted(g, g_fit_bins_raw, w=None)

    params_raw = pack_params(model_type, theta_hat)
    xlag_fit = np.linspace(0.0, float(np.max(h_lag[:, 0]) + bin_size / 2.0), 1000)
    gamma_pred_raw = semivariomodel_fn(xlag_fit, *theta_hat)

    if transform is None:
        gamma = np.asarray(gamma, float)
        params = params_raw
        g_fit_bins = g_fit_bins_raw
        gamma_pred = gamma_pred_raw

    elif transform == "correlation":
        sill = _get_total_sill_from_params(params_raw)
        params = _pack_corr_params_from_vario(model_type, params_raw)

        # experimental ordinates: transform from fitted semivariogram sill
        gamma = _gamma_to_correlation(np.asarray(gamma[:, 0], float), sill).reshape(-1, 1)

        # fitted curve / fitted bin ordinates: evaluate in correlation-form model space
        g_fit_bins = _evaluate_model_from_params(model_type, h, params, transform="correlation")
        gamma_pred = _evaluate_model_from_params(model_type, xlag_fit, params, transform="correlation")

    param_keys = list(params.keys())

    # -------------------------------------------------
    # bootstrap
    # -------------------------------------------------
    boot = None
    if bootstrap is not None:
        n_boot = int(bootstrap)
        if n_boot < 1:
            raise ValueError("bootstrap must be None or a positive integer.")

        method = str(bootstrap_method).lower()
        if method not in {"pair", "point"}:
            raise ValueError("bootstrap_method must be 'pair' or 'point'.")

        qlo, qhi = map(float, bootstrap_ci)
        if not (0.0 <= qlo < qhi <= 100.0):
            raise ValueError("bootstrap_ci must satisfy 0 <= low < high <= 100.")

        rng = np.random.default_rng(random_state)

        gamma_boot = np.full((n_boot, h_full.shape[0]), np.nan, dtype=float)
        model_boot = np.full((n_boot, xlag_fit.size), np.nan, dtype=float)

        param_boot = {k: np.full(n_boot, np.nan, dtype=float) for k in param_keys}
        r2_wls_boot = np.full(n_boot, np.nan, dtype=float)
        r2_ols_boot = np.full(n_boot, np.nan, dtype=float)

        for b_ix in range(n_boot):
            try:
                # --------------------------
                # resample
                # --------------------------
                if method == "pair":
                    draw = rng.integers(0, d.size, size=d.size)
                    d_b = d[draw]
                    dz_b = dz[draw]

                elif method == "point":
                    point_draw = rng.integers(0, n, size=n)
                    values_b = values[point_draw]
                    coords_b = coords[point_draw]

                    d_b, dz_b = _build_pair_arrays(
                        values_b, coords_b, distance_type, max_distance, bin_size,
                        balltree_leaf_size=balltree_leaf_size,
                        max_neighbors=max_neighbors,
                    )

                # --------------------------
                # experimental variogram
                # --------------------------
                h_full_b, n_full_b, gamma_full_b, h_lag_b, n_obs_b, gamma_b = _binned_semivariogram_from_pairs(
                    d_b, dz_b, bin_size, max_distance, estimator_type, semivarioest_fn, lag_repr=lag_repr
                )

                gamma_boot[b_ix, :] = gamma_full_b.ravel()

                # --------------------------
                # fit model
                # --------------------------
                h_b = h_lag_b.ravel()
                g_b = gamma_b.ravel()
                m_b = n_obs_b.ravel()

                if weight_fn is None or str(weight_fn).lower() == "ols":
                    weights_b = np.ones_like(h_b, dtype=float)
                else:
                    if weight_params is None:
                        weight_params_b = [0.25 * float(h_b.max()) if h_b.size else 1.0, 1.0]
                    elif isinstance(weight_params, dict):
                        wb = weight_params.get("b", 0.25 * float(h_b.max()) if h_b.size else 1.0)
                        wa = weight_params.get("alpha", 1.0)
                        weight_params_b = [wb, wa]
                    else:
                        weight_params_b = weight_params

                    weights_b = compute_distance_weights(
                        h_b, m_b, weight_type=weight_fn, weight_params=weight_params_b
                    )

                if fix_sill and not fix_nugget:
                    x0_full_b, bounds_full_b = make_init_and_bounds(
                        model_type, h_b, g_b, xmax_factor, fix_nugget=False, fix_sill=False
                    )
                    x0_b, bounds_b = _compress_init_bounds_fixed_total_sill(
                        model_type, x0_full_b, bounds_full_b
                    )

                    res_b = minimize(
                        fun=lambda th: _objective_func_fixed_total_sill(
                            th, h_b, g_b, weights_b, model_type, semivariomodel_fn
                        ),
                        x0=x0_b,
                        bounds=bounds_b,
                    )
                    if not res_b.success:
                        continue
                    theta_b = np.asarray(_expand_theta_fixed_total_sill(model_type, res_b.x), float)

                else:
                    x0_b, bounds_b = make_init_and_bounds(
                        model_type, h_b, g_b, xmax_factor, fix_nugget, fix_sill
                    )

                    res_b = minimize(
                        fun=lambda th: objective_func(th, h_b, g_b, weights_b, semivariomodel_fn),
                        x0=x0_b,
                        bounds=bounds_b,
                    )
                    if not res_b.success:
                        continue
                    theta_b = np.asarray(res_b.x, float)

                # raw fitted values for fit diagnostics
                g_fit_b_raw = semivariomodel_fn(h_b, *theta_b)
                r2_wls_boot[b_ix] = r2_score_weighted(g_b, g_fit_b_raw, w=weights_b)
                r2_ols_boot[b_ix] = r2_score_weighted(g_b, g_fit_b_raw, w=None)

                p_b_raw = pack_params(model_type, theta_b)

                if transform is None:
                    gamma_boot[b_ix, :] = gamma_full_b.ravel()
                    model_boot[b_ix, :] = semivariomodel_fn(xlag_fit, *theta_b)
                    p_b_store = p_b_raw

                elif transform == "correlation":
                    sill_b = _get_total_sill_from_params(p_b_raw)
                    p_b_store = _pack_corr_params_from_vario(model_type, p_b_raw)

                    gamma_boot[b_ix, :] = _gamma_to_correlation(gamma_full_b.ravel(), sill_b)
                    model_boot[b_ix, :] = _evaluate_model_from_params(
                        model_type, xlag_fit, p_b_store, transform="correlation"
                    )

                for k in param_keys:
                    param_boot[k][b_ix] = p_b_store[k]

            except Exception:
                # keep NaNs for failed replicate
                continue

        gamma_mean, gamma_q05, gamma_q95, n_gamma_success = _bootstrap_summary(gamma_boot, qlo, qhi)
        model_mean, model_q05, model_q95, n_model_success = _bootstrap_summary(model_boot, qlo, qhi)

        boot = {
            "n_bootstrap": n_boot,
            "method": method,
            "ci": (qlo, qhi),
            "random_state": random_state,
            "successful_gamma_bootstrap": n_gamma_success,
            "successful_model_bootstrap": n_model_success,
            "h_lag_full": h_full[:, 0].copy(),
            "n_obs_full": n_full[:, 0].copy(),
            "gamma_samples": gamma_boot,
            "gamma_mean": gamma_mean,
            "gamma_q05": gamma_q05,
            "gamma_q95": gamma_q95,
            "xlag_fit": xlag_fit.copy(),
            "model_samples": model_boot,
            "model_mean": model_mean,
            "model_q05": model_q05,
            "model_q95": model_q95,
            "param_samples": param_boot,
            "r2_wls_samples": r2_wls_boot,
            "r2_ols_samples": r2_ols_boot,
        }

    if boot is not None:
        params["bootstrap"] = boot

    # -------------------------------------------------
    # plot
    # -------------------------------------------------
    if plot:
        fig = plt.figure(figsize=(12, 7), dpi=200)
        gs_plot = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

        ax0 = plt.subplot(gs_plot[0])
        ax0.bar(
            h_lag[:, 0],
            n_obs[:, 0],
            edgecolor="black",
            align="center",
            width=bin_size / 2.0,
        )
        ax0.grid(which="minor")

        ax1 = plt.subplot(gs_plot[1], sharex=ax0)

        # experimental points
        ax1.plot(
                h_lag[:, 0], gamma[:, 0],
                "o", markeredgecolor="black", color="tab:blue",
                label="Experimental", zorder=5
        )

        # bootstrap fitted curves + mean + CI
        if boot is not None:
            first = True
            for yb in boot["model_samples"]:
                if np.any(np.isfinite(yb)):
                    ax1.plot(
                        xlag_fit, yb,
                        color="0.7", lw=0.8, alpha=0.2,
                        label="Bootstrap samples" if first else None,
                        zorder=1
                    )
                    first = False

            if np.any(np.isfinite(boot["model_q05"])) and np.any(np.isfinite(boot["model_q95"])):
                ax1.fill_between(
                    xlag_fit,
                    boot["model_q05"],
                    boot["model_q95"],
                    color="tab:orange",
                    alpha=0.20,
                    label=f"Bootstrap {boot['ci'][0]:g}-{boot['ci'][1]:g}% CI",
                    zorder=2
                )

            if np.any(np.isfinite(boot["model_mean"])):
                ax1.plot(
                    xlag_fit, boot["model_mean"],
                    color="tab:orange", lw=2.0,
                    label="Bootstrap mean",
                    zorder=3
                )

        # main fitted curve
        if transform is None:
            _plot_variogram_model_piecewise(
                ax1,
                xlag_fit,
                gamma_pred,
                params=params,
                color="k",
                lw=2.0,
                ls="-",
                label=r"Model, $R^2$ (WLS|OLS) = %.2f|%.2f" % (r2_wls, r2_ols),
                zorder=4,
                show_zero_point=True
            )
        else:
            _plot_correlation_model_piecewise(
                ax1,
                xlag_fit,
                gamma_pred,
                params=params,
                color="k",
                lw=2.0,
                ls="-",
                label=r"Model, $R^2$ (WLS|OLS) = %.2f|%.2f" % (r2_wls, r2_ols),
                zorder=4,
                show_zero_point=True
            )

        plt.setp(ax0.get_xticklabels(), visible=False)

        if transform is None:
            yticks = ax1.yaxis.get_major_ticks()
            if yticks:
                yticks[-1].label1.set_visible(False)

        ax0.set_ylabel("Number of Lags, N")
        ax0.set_ylim(0, max(n_obs[:, 0]))

        ax1.legend(loc="upper left")

        ax1.set_xticks(h_lag[:, 0])
        num_ticks = len(h_lag[:, 0])

        if num_ticks <= 30:
            step = 1
        elif num_ticks <= 50:
            step = 2
        elif num_ticks <= 70:
            step = 3
        elif num_ticks <= 90:
            step = 4
        else:
            step = max(1, num_ticks // 20)

        for i, label in enumerate(ax1.get_xticklabels()):
            if i % step != 0:
                label.set_visible(False)

        ax0.xaxis.grid(True, which="major", linestyle="--")
        ax1.xaxis.grid(True, which="major", linestyle="--")

        y_candidates = [
            gamma[:, 0],
            gamma_pred,
        ]
        if boot is not None:
            y_candidates.extend([
                boot["model_mean"],
                boot["model_q05"],
                boot["model_q95"],
            ])

        y_all = np.concatenate([np.asarray(yc, float).ravel() for yc in y_candidates])
        y_all = y_all[np.isfinite(y_all)]

        ax1.set_xlim(0, float(np.max(h_lag[:, 0]) + bin_size / 2.0))

        if transform is None:
            ymax = np.nanmax(y_all) if y_all.size else 1.0
            ax1.set_ylim(0, ymax + 0.075)
            ax1.set_ylabel(r"Semivariance, $\gamma$ (%s)" % estimator_type)

        else:
            ax0.set_ylabel("Number of Lags, N", labelpad=22)
            ax1.set_yticks([-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])
            ax1.set_ylim(-1, 1)
            ax1.set_ylabel(r"Correlation, $\rho$")
            ax1.legend(loc="lower left")

        ax1.set_xlabel("lag distance")

        plt.subplots_adjust(hspace=0.0)

        if plot_path is not None:
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    return h_lag, n_obs, gamma, params, r2_wls, r2_ols

# Grouped variogram fitting
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
    plot_single=False,
    plot_summary=False,
    summary_figpath=None,
    show_progress=True,
    balltree_leaf_size=40,
    max_neighbors=None,
    transform=None,
    lag_repr="center",
):
    """
    Fit an experimental variogram separately for each group in a table.

    For each unique value of `index_col`, the function subsets the data, calls
    `variofit`, and stores the experimental ordinates, fitted parameters, and fit
    statistics. Because all groups share the same lag-bin definition, the outputs
    can also be assembled into wide lag-by-group tables for later comparison or
    summary plotting.

    All groups share a **common equal-width bin grid** defined by `max_distance`
    and `bin_size`. The representative lag axis attached to that grid is controlled
    by `lag_repr`, so the results can be combined into wide matrices with one
    column per group.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing the values, grouping IDs, and coordinates.
    values_col : str
        Column name of the variable to be variogrammed (e.g. 'PGA', residuals).
    index_col : str
        Column whose values define groups (e.g. 'evid' for event-wise variograms).
    coord_cols : tuple[str, str] or list[str] or str
        Coordinate columns, depending on `distance_type`:

        - 'geographic' or 'geographical':
            (lat_col, lon_col) in degrees.
        - 'cartesian':
            (x_col, y_col) in the same linear units as `bin_size`.
        - 'euclidean':
            list/tuple of columns forming an (n, d) coordinate array.
        - 'angular':
            single column name or (colname,) with angles in degrees.

    distance_type : {'geographic', 'geographical', 'cartesian', 'angular', 'euclidean'}
        Passed directly to `variofit` to select the distance metric.
    max_distance : float
        Maximum lag used to define the bin grid. Together with `bin_size` this
        sets the number of bins used by `variofit`.
    bin_size : float
        Width of each lag bin (same units as `max_distance`).
    estimator_type : {'Matheron', 'CressieHawkins', 'Dowd'}
        Semivariogram estimator passed through to `variofit`.
    model_type : {'exponential', 'cubic', 'powered_exponential', 'matern',
                  'gaussian', 'spherical', 'damped_cosine_angle',
                  'angular_dissimilarity'}
        Variogram model to fit in each group (see `variofit` for details).
    weight_fn : {None, 'ols', 'inverse-linear weighting', 'exponential weighting',
                 'powered weighting', 'linear weighting'}, optional
        Bin weighting scheme for the WLS fit in `variofit`.
    weight_params : list, dict, or None, optional
        Parameters for `weight_fn` (e.g. {'b': 50.0, 'alpha': 1.0}). See `variofit`.
    xmax_factor : float, default 2.0
        Scaling factor used when constructing upper bounds and initial guesses
        for the model range parameter in `make_init_and_bounds`.
    fix_nugget : bool, default True
        If True, fixes the nugget (b) to 0.0 during fitting.
    fix_sill : bool, default False
        If True, fixes the total sill (c0 + b) to 1.0 during fitting.
        If the nugget is also free, the partial sill is constrained as c0 = 1 - b.
    plot_single : bool, default False
        If True, `variofit` produces a two-panel plot for each group.
    plot_summary : bool, default False
        If True, produce a single summary figure showing:
          - all experimental semivariograms as coloured points (by n_obs), and
          - mean / median fitted curves plus 5–95% envelope over groups.
    summary_figpath : str or None, default None
        If not None and `plot_summary` is True, save the summary figure to this path.
    show_progress : bool, default True
        If True, wrap the group loop in a tqdm progress bar.
    balltree_leaf_size : int, default 40
        Passed to `variofit`. Leaf size for the BallTree used to generate pairs
        for 'geographic', 'geographical', 'cartesian', and 'euclidean' distances.
    max_neighbors : int or None, default None
        Passed to `variofit`. If not None, limits the number of neighbours per
        point when using the BallTree. This can reduce computation for very
        large groups at the cost of an approximate variogram.
    transform : {None, 'correlation'}, default None
        Output transform applied after fitting.

        - None:
            Return the usual semivariogram outputs.
        - 'correlation':
            Keep fitting in semivariogram space, but return:
              * experimental ordinates transformed to correlation space
              * fitted parameters in normalized correlation-form
              * fitted/bootstrapped curves evaluated in correlation space

        For the auto-case, the transformed parameterization uses:
            c0_corr = c0 / (c0 + b)
            b_corr  = b  / (c0 + b)
        with all range/shape parameters unchanged.
    lag_repr : {'center', 'edge', 'upper'}, default 'center'
        Representative lag attached to each equal-width bin. This controls the
        x-axis used in the returned wide lag tables and in the per-group fitting,
        but does not change bin membership.
    Returns
    -------
    summary : pandas.DataFrame
        One row per group with:
          - 'values_index' : group ID (from `index_col`)
          - 'n_samples'    : number of observations in the group
          - 'mean', 'std'  : sample statistics of `values_col`
          - 'n_bins'       : number of non-empty variogram bins
          - 'r2_wls', 'r2_ols' : weighted and unweighted R² of the fitted model
          - plus one column per model parameter (e.g. 'r', 'c0', 'b').
    df_n_obs : pandas.DataFrame
        Wide matrix of pair counts per bin. First column 'h_lag' is the global
        representative lag axis; one additional column per group with n_obs at each lag.
    df_gamma : pandas.DataFrame
        Wide matrix of experimental semivariance on the same lag axis as df_n_obs.
    results : dict
        Mapping {group_id: (h_lag, n_obs, gamma, params, r2_wls, r2_ols)} with
        the raw `variofit` output for each group.
    """

    _validate_transform(transform)

    font_size = 12
    results = {}
    summary_rows = []

    # full list of representative lag values (global index for wide frames)
    nmax = int(np.ceil(float(max_distance) / float(bin_size)))
    full_h = _make_lag_axis(nmax, bin_size, lag_repr=lag_repr)
    full_h = np.round(full_h, 12)

    # initialize wide frames with the global bin axis
    df_n_obs = pd.DataFrame({"h_lag": full_h})
    df_gamma = pd.DataFrame({"h_lag": full_h})

    param_keys = None

    df = df.copy()
    df[index_col] = df[index_col].astype(str).str.strip()
    gb = df.groupby(index_col, sort=False, observed=False)

    dt = str(distance_type).lower().strip()

    if show_progress:
        group_iter = tqdm(gb, total=gb.ngroups, desc="Fitting groups")
    else:
        group_iter = gb

    for gid, gdf in group_iter:

        # values
        vals = gdf[values_col].to_numpy(dtype=float)
        mu = float(np.mean(vals))
        sig = float(np.std(vals, ddof=1))

        # coordinates (match variofit capability)
        if dt in ("geographic", "geographical"):
            lat = gdf[coord_cols[0]].to_numpy(dtype=float)
            lon = gdf[coord_cols[1]].to_numpy(dtype=float)
            coords = np.column_stack([lat, lon])
            distance_type_pass = "geographic" if dt == "geographic" else "geographical"

        elif dt == "cartesian":
            if len(coord_cols) != 2:
                raise ValueError("cartesian requires coord_cols=(x_col, y_col)")
            x = gdf[coord_cols[0]].to_numpy(dtype=float)
            y = gdf[coord_cols[1]].to_numpy(dtype=float)
            coords = np.column_stack([x, y])
            distance_type_pass = "cartesian"

        elif dt == "euclidean":
            coords = gdf[list(coord_cols)].to_numpy(dtype=float)
            distance_type_pass = "euclidean"

        elif dt == "angular":
            # allow "theta" or ("theta",)
            if isinstance(coord_cols, str):
                theta = gdf[coord_cols].to_numpy(dtype=float)
            elif isinstance(coord_cols, (list, tuple)) and len(coord_cols) == 1:
                theta = gdf[coord_cols[0]].to_numpy(dtype=float)
            else:
                raise ValueError("angular requires a single column name in coord_cols")
            coords = theta
            distance_type_pass = "angular"

        else:
            raise ValueError(
                "Invalid distance_type: choose 'geographic', 'geographical', "
                "'cartesian', 'angular', or 'euclidean'"
            )

        # call variofit main function
        res = variofit(
            values=vals,
            coordinates=coords,
            distance_type=distance_type_pass,
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
            balltree_leaf_size=balltree_leaf_size,
            max_neighbors=max_neighbors,
            transform=transform,
            lag_repr=lag_repr,
        )

        h_lag, n_obs, gamma, params, r2_wls, r2_ols = res

        results[gid] = res

        if param_keys is None:
            param_keys = list(params.keys())
            # param_keys = [k for k in params.keys() if k != "bootstrap"]

        # align this group's vectors to the global bin axis
        h = np.round(np.asarray(h_lag).ravel(), 12)
        s_n = pd.Series(np.asarray(n_obs).ravel(), index=h)
        s_g = pd.Series(np.asarray(gamma).ravel(), index=h)

        df_n_obs[gid] = s_n.reindex(full_h).to_numpy()
        df_gamma[gid] = s_g.reindex(full_h).to_numpy()

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

    summary = pd.DataFrame(
        summary_rows,
        columns=(["values_index","n_samples","mean","std","n_bins","r2_wls","r2_ols"] + (param_keys or []))
    )

    # ---- summary plot block unchanged (your existing code) ----
    if plot_summary:
        # long (tidy) data for plotting
        g_long = df_gamma.melt(id_vars='h_lag', var_name='group', value_name='gamma')
        n_long = df_n_obs.melt(id_vars='h_lag', var_name='group', value_name='n_obs')
        M = g_long.merge(n_long, on=['h_lag','group'])
        M = M[M['gamma'].notna()]

        fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
        sc = ax.scatter(
            M['h_lag'].to_numpy(),
            M['gamma'].to_numpy(),
            c=M['n_obs'].to_numpy(),
            s=12,
            cmap=plt.get_cmap("coolwarm"),
            norm=mpl.colors.LogNorm(),
            alpha=0.8,
            edgecolor='k',
            linewidths=0.1,
            zorder=1
        )

        x_plot_max = float(np.max(full_h) + bin_size / 2.0)
        xlag_fit = np.linspace(0.0, x_plot_max, 1000)

        # aggregated params
        params_mean = {}
        params_median = {}

        for k in param_keys:
            vals_k = []
            for _, (_, _, _, params_g, _, _) in results.items():
                vals_k.append(float(params_g.get(k, np.nan)))
            vals_k = np.asarray(vals_k, float)
            params_mean[k] = float(np.nanmean(vals_k))
            params_median[k] = float(np.nanmedian(vals_k))

        if transform == "correlation":
            params_mean = _renormalize_correlation_params(params_mean)
            params_median = _renormalize_correlation_params(params_median)

        y_mean = _evaluate_model_from_params(model_type, xlag_fit, params_mean, transform=transform)
        y_median = _evaluate_model_from_params(model_type, xlag_fit, params_median, transform=transform)

        Ys = []
        for gid, res_g in results.items():
            try:
                _, _, _, params_g, *_ = res_g
            except Exception:
                continue
            Ys.append(_evaluate_model_from_params(model_type, xlag_fit, params_g, transform=transform))

        if Ys:
            Y = np.asarray(Ys, dtype=float)
            y_lo = np.nanpercentile(Y, 5.0, axis=0)
            y_hi = np.nanpercentile(Y, 95.0, axis=0)

            lo = np.minimum(y_lo, y_hi)
            hi = np.maximum(y_lo, y_hi)

            if transform is None:
                # semivariograms should be nondecreasing
                y_lo = np.maximum.accumulate(lo)
                y_hi = np.maximum.accumulate(hi)
            else:
                # correlations should be nonincreasing
                y_lo = np.minimum.accumulate(lo)
                y_hi = np.minimum.accumulate(hi)

        else:
            y_lo = y_median.copy()
            y_hi = y_median.copy()

        if transform is None:
            _plot_variogram_model_piecewise(
                ax, xlag_fit, y_mean, params_mean,
                color='k', lw=1.5, ls='-',
                label='mean fit', zorder=1000, show_zero_point=True
            )
            _plot_variogram_model_piecewise(
                ax, xlag_fit, y_median, params_median,
                color='k', lw=1.5, ls='--',
                label='median fit', zorder=1000, show_zero_point=True
            )
        else:
            _plot_correlation_model_piecewise(
                ax, xlag_fit, y_mean, params_mean,
                color='k', lw=1.5, ls='-',
                label='mean fit', zorder=1000, show_zero_point=True
            )
            _plot_correlation_model_piecewise(
                ax, xlag_fit, y_median, params_median,
                color='k', lw=1.5, ls='--',
                label='median fit', zorder=1000, show_zero_point=True
            )

        ax.fill_between(xlag_fit, y_lo, y_hi, color='forestgreen', alpha=0.15, label='5–95% fit', zorder=0)

        for gid, (h_lag_g, n_obs_g, gamma_g, params_g, r2_wls_g, r2_ols_g) in results.items():
            y_fit_g = _evaluate_model_from_params(model_type, xlag_fit, params_g, transform=transform)
            ax.plot(xlag_fit, y_fit_g, '-k', lw=0.2, alpha=0.6)

        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.04, aspect=40)
        cb.set_label('Observations per bin, n', fontsize = font_size)
        cb.ax.tick_params(labelsize=font_size)
        ax.legend(loc='best', frameon=False, fontsize = font_size)

        ax.set_xlabel("lag distance", fontsize=font_size)

        if transform is None:
            ax.set_ylabel(r'Semivariance, $\gamma$ (%s)' % estimator_type, fontsize=font_size)
            ax.set_ylim(0, float(np.nanmax(M['gamma'].to_numpy())))

        else:
            ax.set_ylabel(r'Correlation, $\rho$', fontsize=font_size)
            ax.set_ylim(-1, 1)

        ax.tick_params(axis='both', labelsize=font_size)
        ax.set_xlim(0, x_plot_max)
        ax.grid(True, linestyle='--', alpha=0.3)

        if summary_figpath is not None:
            fig.savefig(summary_figpath, dpi=200, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    return summary, df_n_obs, df_gamma, results

# Cross-variogram fitting and helper functions
def _compute_distance_and_ratio(coords, distance_type, bin_size):
    """
    Compute the full pairwise distance matrix and its floor-based bin-index matrix.

    This helper is only used in the parts of the cross-variogram workflow that
    still rely on the dense distance-ratio formulation rather than the pair-list
    workflow used elsewhere.

    Parameters
    ----------
    coords : array_like, shape (n, d)
        Coordinates for all observations.
    distance_type : {'geographic', 'geographical', 'cartesian', 'euclidean', 'angular'}
        Distance metric used to construct the full distance matrix.
    bin_size : float
        Lag-bin width used to convert distances to floor-based interval-bin indices.

    Returns
    -------
    distance : ndarray, shape (n, n)
        Full pairwise distance matrix.
    distance_ratio : ndarray, shape (n, n)
        Floor(distance / bin_size) matrix used for interval-based lag assignment.
    """
    coords = np.asarray(coords, float)

    distance_type = str(distance_type).lower().strip()
    if distance_type == "geographical":
        distance_type = "geographic"

    if distance_type == 'geographic':
        # coords[:,0]=lat, coords[:,1]=lon
        lat = coords[:, 0]
        lon = coords[:, 1]
        distance = np.asarray(
            haversine_oq(lon, lat, lon, lat, radians=False, earth_rad=6371.227),
            dtype=float
        )
        distance_ratio = np.floor(distance / float(bin_size)).astype(int)

    elif distance_type == 'cartesian':
        # coords: (n,2) = (x,y)
        if coords.shape[1] != 2:
            raise ValueError("cartesian requires coordinates shape (n,2): (x, y)")
        dx = coords[:, None, 0] - coords[None, :, 0]
        dy = coords[:, None, 1] - coords[None, :, 1]
        distance = np.hypot(dx, dy)
        distance_ratio = np.floor(distance / float(bin_size)).astype(int)

    elif distance_type == 'euclidean':
        diff = coords[:, None, :] - coords[None, :, :]
        distance = np.linalg.norm(diff, axis=-1)
        distance_ratio = np.floor(distance / float(bin_size)).astype(int)

    elif distance_type == 'angular':
        # coords expected as angles in degrees
        theta_deg = np.asarray(coords, float)
        if theta_deg.ndim == 2:
            if theta_deg.shape[1] != 1:
                raise ValueError("angular distance requires a single angular coordinate per row.")
        theta = np.radians(theta_deg.ravel())
        cos_diff = np.cos(theta[:, None] - theta[None, :])
        ang_rad = np.arccos(np.clip(cos_diff, -1.0, 1.0))
        distance = np.degrees(ang_rad)
        distance_ratio = np.floor(distance / float(bin_size)).astype(int)

    else:
        raise ValueError(
            "Invalid distance_type: choose 'geographic', 'cartesian', 'angular', or 'euclidean'"
        )

    return distance, distance_ratio

def _adjust_bounds_for_cross(model_type, x0, bounds, g, allow_negative_sill=True):
    """
    Relax the partial-sill bounds for cross-variogram fitting.

    Cross-variograms can legitimately have negative fitted partial sills, so this
    helper widens the bounds on `c0` around zero while leaving all other bounds
    unchanged. If the partial sill is already fixed, the bounds are returned
    unchanged.

    Parameters
    ----------
    model_type : str
        Name of the fitted model.
    x0 : sequence of float
        Initial parameter vector.
    bounds : sequence of tuple
        Bounds aligned with `x0`.
    g : array_like
        Experimental cross-semivariogram ordinates.
    allow_negative_sill : bool, default True
        If False, return the input values unchanged.

    Returns
    -------
    x0 : tuple
        Possibly adjusted initial parameter vector.
    bounds : tuple
        Possibly adjusted bounds.
    """
    if not allow_negative_sill:
        return x0, bounds

    # In your parameterizations, c0 is always the 2nd parameter
    c0_idx = 1

    g = np.asarray(g, float)
    gmax = float(np.nanmax(np.abs(g))) if g.size else 1.0

    # numeric cap based on observed magnitude
    cap = max(1e-6, 2.0 * gmax)

    bounds = list(bounds)

    # If c0 is fixed, do not relax
    try:
        lo0, hi0 = bounds[c0_idx]
    except Exception:
        return x0, tuple(bounds)

    if lo0 is not None and hi0 is not None and float(lo0) == float(hi0):
        return x0, tuple(bounds)

    lo, hi = -cap, cap
    bounds[c0_idx] = (lo, hi)

    x0 = list(x0)
    x0[c0_idx] = float(np.clip(x0[c0_idx], lo, hi))

    return tuple(x0), tuple(bounds)

def _get_nugget_from_params(params):
    """
    Safely extract the nugget parameter from a parameter dictionary.

    Parameters
    ----------
    params : dict or None
        Parameter dictionary that may contain `b`.

    Returns
    -------
    b0 : float
        Nugget value if present and finite; otherwise 0.
    """
    params = {} if params is None else dict(params)
    b0 = float(params.get("b", 0.0))
    return b0 if np.isfinite(b0) else 0.0


def _plot_variogram_model_piecewise(ax, x, y, params, color="k", lw=2.0, ls="-",
                                    label=None, zorder=4, alpha_plot=1.0,
                                    show_zero_point=True, jump_ls=":"):
    """
    Plot a variogram model with explicit nugget discontinuity:

        gamma(0)  = 0
        gamma(0+) = b

    The positive-lag branch is plotted for x > 0 only, and a vertical jump
    is drawn at x = 0 when b > 0.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    b0 = _get_nugget_from_params(params)
    has_nugget = np.isfinite(b0) and (b0 > 1e-12)

    pos = x > 0

    if has_nugget:
        ax.plot(
            x[pos], y[pos],
            color=color, lw=lw, ls=ls, label=label,
            zorder=zorder, alpha=alpha_plot
        )

        ax.plot(
            [0.0, 0.0], [0.0, b0],
            color=color, lw=max(1.0, lw * 0.8), ls=jump_ls,
            zorder=zorder, alpha=alpha_plot
        )

        if show_zero_point:
            ax.plot(
                0.0, 0.0, "o",
                ms=4, mfc="white", mec=color,
                zorder=zorder + 0.1, alpha=alpha_plot
            )
    else:
        ax.plot(
            x, y,
            color=color, lw=lw, ls=ls, label=label,
            zorder=zorder, alpha=alpha_plot
        )

def _binned_cross_semivariogram_from_pairs(
    d,
    dz1,
    dz2,
    bin_size,
    max_distance,
    lag_repr="center",
):
    """
    Bin pairwise distances and paired increments into experimental
    cross-semivariogram ordinates.

    Pair membership is defined by interval binning:
        bin k contains distances in [k*bin_size, (k+1)*bin_size),
    with the final bin capped at `max_distance`.

    Parameters
    ----------
    d : array_like
        Pairwise distances.
    dz1, dz2 : array_like
        Paired increments (z1_i - z1_j) and (z2_i - z2_j).
    bin_size : float
        Bin width.
    max_distance : float
        Maximum retained lag distance.
    lag_repr : {"center", "edge", "upper"}, default "center"
        Representative lag attached to each bin:
        - "center": midpoint of [k*bin_size, (k+1)*bin_size)
        - "edge"/"upper": upper edge of [k*bin_size, (k+1)*bin_size)

    Returns
    -------
    h_full : ndarray, shape (nmax, 1)
        Representative lag values for all bins.
    counts_full : ndarray, shape (nmax, 1)
        Pair counts for all bins.
    gamma_full : ndarray, shape (nmax, 1)
        Experimental cross-semivariance for all bins; NaN for empty bins.
    h_valid : ndarray, shape (k, 1)
        Representative lag values for non-empty bins.
    counts_valid : ndarray, shape (k, 1)
        Pair counts for non-empty bins.
    gamma_valid : ndarray, shape (k, 1)
        Experimental cross-semivariance for non-empty bins.
    """
    nmax = int(np.ceil(float(max_distance) / float(bin_size)))
    if nmax <= 0:
        raise ValueError("max_distance / bin_size must be > 0.")

    d = np.asarray(d, dtype=float).ravel()
    dz1 = np.asarray(dz1, dtype=float).ravel()
    dz2 = np.asarray(dz2, dtype=float).ravel()

    if not (d.shape == dz1.shape == dz2.shape):
        raise ValueError("d, dz1, and dz2 must have the same shape.")

    mask = (
        np.isfinite(d)
        & np.isfinite(dz1)
        & np.isfinite(dz2)
        & (d >= 0.0)
        & (d <= float(max_distance))
    )
    if not np.any(mask):
        raise ValueError(
            "No pair distances fell into [0, max_distance]; check max_distance/bin_size."
        )

    d_use = d[mask]
    dz1_use = dz1[mask]
    dz2_use = dz2[mask]

    bin_idx = np.floor(d_use / float(bin_size)).astype(int)
    bin_idx = np.minimum(bin_idx, nmax - 1)

    h_full = _make_lag_axis(nmax, bin_size, lag_repr=lag_repr)
    counts = np.bincount(bin_idx, minlength=nmax).astype(float)

    cp = 0.5 * dz1_use * dz2_use
    sumcp = np.bincount(bin_idx, weights=cp, minlength=nmax)

    gamma_full = np.full(nmax, np.nan, dtype=float)
    nonempty = counts > 0
    gamma_full[nonempty] = sumcp[nonempty] / counts[nonempty]

    keep = np.isfinite(gamma_full)

    return (
        h_full.reshape(-1, 1),
        counts.reshape(-1, 1),
        gamma_full.reshape(-1, 1),
        h_full[keep].reshape(-1, 1),
        counts[keep].reshape(-1, 1),
        gamma_full[keep].reshape(-1, 1),
    )

def crossvariofit(
    values1,
    values2,
    coordinates,
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
    allow_negative_sill=True,
    plot=False,
    balltree_leaf_size=40,
    max_neighbors=None,
    transform=None,
    lag_repr="center",
):
    """
    Compute an experimental cross-semivariogram and fit the chosen model.

    This follows the same workflow as `variofit`, but replaces the auto-semivariogram
    estimator with the classical cross-semivariogram estimator

        γ12(h) = 0.5 * mean[(z1_i - z1_j) (z2_i - z2_j)].

    When `transform="correlation"`, the fitted cross-semivariogram is normalised in
    the same way as the diagonal case, i.e.

        ρ12(h) = 1 - γ12(h) / C12(0).

    Unlike `variofit`, this function does not currently implement bootstrap
    resampling.

    Parameters
    ----------
    values1, values2 : array_like, shape (n,)
        Two variables measured at the same coordinates.
    coordinates : array_like, shape (n, d)
        Same rule-set as `variofit`.
    distance_type, max_distance, bin_size, model_type, weight_fn, weight_params,
    xmax_factor, fix_nugget, fix_sill
        Same meaning as in `variofit`.
    estimator_type : str
        For cross-variograms, only 'Matheron' is supported.
    allow_negative_sill : bool, default True
        If True, relaxes the c0 bounds to allow negative cross-sill.
    plot : bool, default False
        If True, makes a two-panel plot (counts + cross-semivariogram fit).
    balltree_leaf_size : int, default 40
        Leaf size for BallTree (used for 'geographic', 'cartesian', 'euclidean').
    max_neighbors : int or None, default None
        If not None, limits the number of neighbours per point when using BallTree.
    transform : {None, 'correlation'}, default None
        Optional output transform applied after fitting.

        - None:
            Return the fitted cross-semivariogram.
        - 'correlation':
            Return the normalized cross-correlation form
                rho12(h) = 1 - gamma12(h) / C12(0),
            using the fitted zero-lag cross-covariance C12(0) = c0 + b.
    lag_repr : {'center', 'edge', 'upper'}, default 'center'
        Representative lag attached to each equal-width bin:
        - 'center': midpoint of [k*bin_size, (k+1)*bin_size)
        - 'edge'/'upper': upper edge of [k*bin_size, (k+1)*bin_size)

        This affects the x-values used for plotting, weighting, and fitting, but
        does not change which pairs fall into each bin.
    Returns
    -------
    h_lag, n_obs, gamma12, params, r2_wls, r2_ols
        Same tuple structure as `variofit`, with `h_lag` giving the representative
        lag values of the non-empty bins according to `lag_repr`.
    """

    _validate_transform(transform)

    distance_type = str(distance_type).lower().strip()
    if distance_type == "geographical":
        distance_type = "geographic"

    values1 = np.asarray(values1, float)
    values2 = np.asarray(values2, float)
    coords  = np.asarray(coordinates, float)

    if values1.shape != values2.shape:
        raise ValueError("values1 and values2 must have the same shape")

    # estimator: cross version only supports Matheron
    if estimator_type != "Matheron":
        raise ValueError("For cross-variograms, only 'Matheron' is currently supported.")

    # model function from registry
    if model_type not in VARIOGRAM_MODELS:
        raise ValueError("Invalid Model: must be one of %s" % list(VARIOGRAM_MODELS.keys()))
    semivariomodel_fn = VARIOGRAM_MODELS[model_type]

    if fix_sill and allow_negative_sill:
        raise ValueError(
            "fix_sill=True is incompatible with allow_negative_sill=True for cross-variograms."
        )

    n = len(values1)
    if n < 2:
        raise ValueError("Need at least 2 points to compute a cross-variogram.")

    # number of bins
    nmax = int(np.ceil(float(max_distance) / float(bin_size)))
    if nmax <= 0:
        raise ValueError("max_distance / bin_size must be > 0.")

    dt = str(distance_type).lower().strip()
    if dt == "geographical":
        dt = "geographic"

    # ------------------------------------------------------------------
    # Case 1: ANGULAR -> keep existing dense logic via _compute_distance_and_ratio
    # ------------------------------------------------------------------
    if dt == "angular":
        distance_full, _ = _compute_distance_and_ratio(coords, "angular", bin_size)

        iu, ju = np.triu_indices(n, k=1)
        d = distance_full[iu, ju]
        dz1 = values1[iu] - values1[ju]
        dz2 = values2[iu] - values2[ju]

        _, _, _, h_lag, n_obs, gamma = _binned_cross_semivariogram_from_pairs(
            d, dz1, dz2, bin_size, max_distance, lag_repr=lag_repr
        )

    # ------------------------------------------------------------------
    # Case 2: geographic / cartesian / euclidean -> BallTree radius search
    # ------------------------------------------------------------------
    else:
        # Build neighbour list with BallTree
        EARTH_RAD = 6371.227  # km
        d_list   = []
        dz1_list = []
        dz2_list = []

        if dt == "geographic":
            # coords[:,0] = lat (deg), coords[:,1] = lon (deg)
            lat_deg = coords[:, 0]
            lon_deg = coords[:, 1]
            lat_rad = np.deg2rad(lat_deg)
            lon_rad = np.deg2rad(lon_deg)
            pts = np.column_stack([lat_rad, lon_rad])

            tree = BallTree(pts, metric="haversine", leaf_size=balltree_leaf_size)
            radius = (float(max_distance) + 0.5 * float(bin_size)) / EARTH_RAD  # radians

            ind_list, dist_list = tree.query_radius(
                pts, r=radius, return_distance=True, sort_results=True
            )

            for i in range(n):
                inds = ind_list[i]
                dists_rad = dist_list[i]
                mask = inds > i  # j > i, no self-pairs
                if not np.any(mask):
                    continue

                js = inds[mask]
                d_ij = dists_rad[mask] * EARTH_RAD  # back to km

                if max_neighbors is not None and max_neighbors > 0:
                    js = js[:max_neighbors]
                    d_ij = d_ij[:max_neighbors]

                d_list.append(d_ij)
                dz1_list.append(values1[i] - values1[js])
                dz2_list.append(values2[i] - values2[js])

        elif dt in ("cartesian", "euclidean"):
            pts = coords
            tree = BallTree(pts, metric="euclidean", leaf_size=balltree_leaf_size)
            radius = float(max_distance) + 0.5 * float(bin_size)

            ind_list, dist_list = tree.query_radius(
                pts, r=radius, return_distance=True, sort_results=True
            )

            for i in range(n):
                inds = ind_list[i]
                dists = dist_list[i]
                mask = inds > i
                if not np.any(mask):
                    continue

                js = inds[mask]
                d_ij = dists[mask]

                if max_neighbors is not None and max_neighbors > 0:
                    js = js[:max_neighbors]
                    d_ij = d_ij[:max_neighbors]

                d_list.append(d_ij)
                dz1_list.append(values1[i] - values1[js])
                dz2_list.append(values2[i] - values2[js])

        else:
            raise ValueError(
                "Invalid distance_type for crossvariofit: choose 'geographic', "
                "'cartesian', 'euclidean', or 'angular'"
            )

        if not d_list:
            raise ValueError("No neighbour pairs found within the specified max_distance.")

        d   = np.concatenate(d_list)
        dz1 = np.concatenate(dz1_list)
        dz2 = np.concatenate(dz2_list)

        # Binning
        _, _, _, h_lag, n_obs, gamma = _binned_cross_semivariogram_from_pairs(
            d, dz1, dz2, bin_size, max_distance, lag_repr=lag_repr
        )

    h = h_lag.ravel()
    g = gamma.ravel()
    m = n_obs.ravel()

    # Guard against the case where no valid lag bins remain after filtering.
    if h.size == 0:
        raise ValueError(
            "No valid lag bins with a defined experimental cross-semivariance were found. "
            "Check the data, bin_size, max_distance, and the available pair structure."
        )

    # Build lag-bin weights
    if weight_fn is None or str(weight_fn).lower() == "ols":
        weights = np.ones_like(h, dtype=float)
    else:
        if weight_params is None:
            # Default consistent with your doc intent: b=0.25*max(h), alpha=1.0
            weight_params = [0.25 * float(h.max()), 1.0]
        elif isinstance(weight_params, dict):
            b = weight_params.get("b", 0.25 * float(h.max()) if h.size else 1.0)
            alpha = weight_params.get("alpha", 1.0)
            weight_params = [b, alpha]

        weights = compute_distance_weights(
            h, m, weight_type=weight_fn, weight_params=weight_params
        )

    # Build initial values and bounds; relax c0 bounds for cross terms if needed.
    if fix_sill and not fix_nugget:
        # Enforce total sill = 1, so c0 = 1 - b
        x0_full, bounds_full = make_init_and_bounds(
            model_type, h, g, xmax_factor=xmax_factor, fix_nugget=False, fix_sill=False
        )
        x0, bounds = _compress_init_bounds_fixed_total_sill(model_type, x0_full, bounds_full)

        res = minimize(
            fun=lambda th: _objective_func_fixed_total_sill(
                th, h, g, weights, model_type, semivariomodel_fn
            ),
            x0=x0,
            bounds=bounds,
        )
        if not res.success:
            raise RuntimeError(f"Cross-variogram-model optimization failed: {res.message}")
        theta_hat = np.asarray(_expand_theta_fixed_total_sill(model_type, res.x), float)

    else:

        x0, bounds = make_init_and_bounds(
            model_type, h, g, xmax_factor=xmax_factor, fix_nugget=fix_nugget, fix_sill=fix_sill
        )
        x0, bounds = _adjust_bounds_for_cross(
            model_type, x0, bounds, g, allow_negative_sill=allow_negative_sill
        )

        res = minimize(
            fun=lambda th: objective_func(th, h, g, weights, semivariomodel_fn),
            x0=x0,
            bounds=bounds
        )
        if not res.success:
            raise RuntimeError(f"Cross-variogram-model optimization failed: {res.message}")
        theta_hat = np.asarray(res.x, float)

    # raw semivariogram-space fit diagnostics
    g_fit_bins_raw = semivariomodel_fn(h, *theta_hat)
    r2_wls = r2_score_weighted(g, g_fit_bins_raw, w=weights)
    r2_ols = r2_score_weighted(g, g_fit_bins_raw, w=None)

    params_raw = pack_params(model_type, theta_hat)
    xlag_fit = np.linspace(0.0, float(max_distance + bin_size / 2.0), 1000)
    gamma_pred_raw = semivariomodel_fn(xlag_fit, *theta_hat)

    if transform is None:
        gamma = np.asarray(gamma, float)
        params = params_raw
        g_fit_bins = g_fit_bins_raw
        gamma_pred = gamma_pred_raw

    elif transform == "correlation":
        sill = float(params_raw.get("c0", np.nan)) + float(params_raw.get("b", 0.0))
        if not np.isfinite(sill) or sill <= 0.0:
            raise ValueError(
                "transform='correlation' requires a positive fitted cross-sill "
                "(c0 + b > 0). The fitted cross-semivariogram returned c0 + b <= 0, "
                "so the normalized cross-correlation transform is not defined."
            )

        params = _pack_corr_params_from_vario(model_type, params_raw)

        gamma = _gamma_to_correlation(
            np.asarray(gamma[:, 0], float),
            sill
        ).reshape(-1, 1)

        g_fit_bins = _evaluate_model_from_params(
            model_type, h, params, transform="correlation"
        )
        gamma_pred = _evaluate_model_from_params(
            model_type, xlag_fit, params, transform="correlation"
        )

    if plot:
        fig = plt.figure(figsize=(12, 7), dpi=200)
        gs_plot = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

        ax0 = plt.subplot(gs_plot[0])
        ax0.bar(h_lag[:, 0], n_obs[:, 0], edgecolor='black', align='center', width=bin_size / 2)
        ax0.grid(which='minor')

        ax1 = plt.subplot(gs_plot[1], sharex=ax0)
        ax1.plot(
            h_lag[:, 0], gamma[:, 0],
            'o', markeredgecolor='black',
            label='Cross Experimental', zorder=5
        )

        if transform is None:
            _plot_variogram_model_piecewise(
                ax1,
                xlag_fit,
                gamma_pred,
                params=params,
                color='k',
                lw=2.0,
                ls='-',
                label=r'Model, $R^2$ (WLS|OLS) = %.2f|%.2f' % (r2_wls, r2_ols),
                zorder=4,
                show_zero_point=True
            )
        else:
            _plot_correlation_model_piecewise(
                ax1,
                xlag_fit,
                gamma_pred,
                params=params,
                color='k',
                lw=2.0,
                ls='-',
                label=r'Model, $R^2$ (WLS|OLS) = %.2f|%.2f' % (r2_wls, r2_ols),
                zorder=4,
                show_zero_point=True
            )

        plt.setp(ax0.get_xticklabels(), visible=False)

        if transform is None:
            yticks = ax1.yaxis.get_major_ticks()
            if len(yticks) > 0:
                yticks[-1].label1.set_visible(False)

        ax0.set_ylabel('Number of Lags, N')
        ax0.set_ylim(0, np.nanmax(n_obs[:, 0]) if n_obs.size else 1)

        ax1.set_xlim(0, float(max_distance + bin_size / 2.0))

        if transform is None:
            _set_ylim_from_points_and_fit(ax1, gamma[:, 0], gamma_pred, allow_negative=allow_negative_sill)
            ax1.set_ylabel(r'Cross-semivariance, $\gamma_{12}$')
            ax1.legend(loc='upper left')

        else:
            ax0.set_ylabel("Number of Lags, N", labelpad=22)
            ax1.set_yticks([-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])
            ax1.set_ylim(-1.0, 1.0)
            ax1.set_ylabel(r'Cross-correlation, $\rho_{12}$')
            ax1.legend(loc='lower left')

        ax1.set_xlabel('lag distance')
        ax1.set_xticks(h_lag[:, 0])

        num_ticks = len(h_lag[:, 0])

        if num_ticks <= 30:
            step = 1
        elif num_ticks <= 50:
            step = 2
        elif num_ticks <= 70:
            step = 3
        elif num_ticks <= 90:
            step = 4
        else:
            step = max(1, num_ticks // 20)

        for i, label in enumerate(ax1.get_xticklabels()):
            if i % step != 0:
                label.set_visible(False)

        ax0.xaxis.grid(True, which='major', linestyle='--')
        ax1.xaxis.grid(True, which='major', linestyle='--')

        plt.subplots_adjust(hspace=.0)
        plt.show(fig)

    return h_lag, n_obs, gamma, params, r2_wls, r2_ols

def multicrossvariofit(
    df,
    values_cols,
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
    allow_negative_sill=True,
    plot_single=False,
    plot_matrix=False,
    balltree_leaf_size=40,
    max_neighbors=None,
    transform=None,
    lag_repr="center",
):
    """
    Fit the full auto/cross variogram matrix for a set of variables.

    Diagonal terms are fitted with `variofit`; off-diagonal terms are fitted with
    `crossvariofit`. The function returns both a long-form summary table and a set
    of square parameter/R² matrices indexed by variable name.

    For geographic, cartesian, and euclidean coordinates, both the auto- and
    cross-variogram paths use the same BallTree-based neighbour search so that only
    pairs within `max_distance` are retained.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing the variables and coordinates.
    values_cols : sequence of str
        Column names of the variables to include in the auto/cross variogram matrix.
    coord_cols : sequence of str or str
        Coordinate columns, interpreted according to `distance_type`.

        - 'geographic':
            (lat_col, lon_col) in degrees.
        - 'cartesian':
            (x_col, y_col) in linear units.
        - 'euclidean':
            list/tuple of columns forming an (n, d) coordinate array.
        - 'angular':
            a single column of angles in degrees.
    distance_type : {'geographic', 'geographical', 'cartesian', 'euclidean', 'angular'}
        Distance metric used to define lags.
    max_distance : float
        Maximum lag distance retained.
    bin_size : float
        Lag-bin width.
    estimator_type : str
        Experimental estimator. Must be 'Matheron' because cross-variograms are
        currently only implemented for the classical Matheron estimator.
    model_type : {'exponential', 'cubic', 'powered_exponential', 'matern',
                  'gaussian', 'spherical', 'damped_cosine_angle',
                  'angular_dissimilarity'}
        Variogram model fitted to every diagonal and off-diagonal term.
    weight_fn : str or None, default None
        Bin-weighting rule passed through to `variofit` and `crossvariofit`.
    weight_params : list, dict, or None, default None
        Parameters controlling `weight_fn`.
    xmax_factor : float, default 2.0
        Multiplier used to cap the upper bound of the range-like parameter.
    fix_nugget : bool, default True
        If True, fix the nugget parameter at 0.
    fix_sill : bool, default False
        If True, apply the fixed-sill logic used by the underlying fitting functions.
    allow_negative_sill : bool, default True
        Passed only to `crossvariofit`. If True, cross-partial-sill values are
        allowed to be negative in semivariogram space.
    plot_single : bool, default False
        If True, show the individual fit plot for each diagonal and off-diagonal fit.
    plot_matrix : bool, default False
        If True, show a lower-triangular matrix plot of the fitted auto/cross
        variograms or correlations.
    balltree_leaf_size : int, default 40
        BallTree leaf size passed to the underlying fitting functions.
    max_neighbors : int or None, default None
        Optional cap on the number of neighbours retained per point when BallTree
        is used.
    transform : {None, 'correlation'}, default None
        Output transform applied after fitting.

        - None:
            Return variograms in semivariogram space.
        - 'correlation':
            Return normalized auto- and cross-correlation forms using the same
            normalization rule as the corresponding fitted sill.
    lag_repr : {'center', 'edge', 'upper'}, default 'center'
        Representative lag attached to each equal-width bin for both the auto-
        and cross-variogram fits. This affects plotting, weighting, and fitting
        x-values, but does not change bin membership.

    Returns
    -------
    summary : pandas.DataFrame
        Long-form summary with one row per fitted term. Includes variable names,
        whether the term is auto or cross, the number of samples and lag bins,
        fit statistics, and the fitted model parameters.
    results : dict
        Dictionary keyed by `(var_i, var_j)` containing the raw outputs of
        `variofit` or `crossvariofit`.
    param_mats : dict[str, pandas.DataFrame]
        Dictionary of square parameter matrices, one for each fitted model
        parameter, indexed and columned by `values_cols`.
    r2_mats : dict[str, pandas.DataFrame]
        Dictionary containing the weighted and ordinary R² matrices with keys
        `r2_wls` and `r2_ols`.

    Notes
    -----
    Only the lower triangle is computed explicitly for cross terms; the opposite
    orientation is filled by symmetry in the returned dictionaries and matrices.
    """

    _validate_transform(transform)

    distance_type = str(distance_type).lower().strip()
    if distance_type == "geographical":
        distance_type = "geographic"

    # Ensure the estimator is always 'Matheron' for cross-variograms
    if estimator_type != "Matheron":
        raise ValueError("For cross-variograms, only 'Matheron' is supported.")

    df = df.copy()
    values_cols = list(values_cols)

    # --- basic column validation
    missing_vals = [c for c in values_cols if c not in df.columns]
    if missing_vals:
        raise ValueError(f"Missing value columns in df: {missing_vals}")

    if distance_type not in ("geographic", "cartesian", "euclidean", "angular"):
        raise ValueError("distance_type must be 'geographic', 'cartesian', 'angular', or 'euclidean'")

    # --- Build coordinates array once
    if distance_type == 'geographic':
        lat = df[coord_cols[0]].to_numpy(dtype=float)
        lon = df[coord_cols[1]].to_numpy(dtype=float)
        coords = np.column_stack([lat, lon])

    elif distance_type == 'cartesian':
        x = df[coord_cols[0]].to_numpy(dtype=float)
        y = df[coord_cols[1]].to_numpy(dtype=float)
        coords = np.column_stack([x, y])

    elif distance_type == 'euclidean':
        coords = df[list(coord_cols)].to_numpy(dtype=float)

    elif distance_type == 'angular':
        if isinstance(coord_cols, (list, tuple)) and len(coord_cols) == 1:
            coords = df[coord_cols[0]].to_numpy(dtype=float)
        elif isinstance(coord_cols, str):
            coords = df[coord_cols].to_numpy(dtype=float)
        else:
            raise ValueError("For angular distance_type, provide a single column in coord_cols.")

    # --- Storage for results and summary
    results = {}
    summary_rows = []
    param_keys = None

    p = len(values_cols)
    n_samples = int(len(df))

    # --- Diagonal: auto-variograms
    for i, vi in enumerate(values_cols):
        vals_i = df[vi].to_numpy(dtype=float)

        res = variofit(
            values=vals_i,
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
            balltree_leaf_size=balltree_leaf_size,
            max_neighbors=max_neighbors,
            transform=transform,
            lag_repr=lag_repr,
        )

        h_lag, n_obs, gamma, params, r2_wls, r2_ols = res
        results[(vi, vi)] = res

        if param_keys is None:
            param_keys = list(params.keys())

        summary_rows.append({
            "var_i": vi,
            "var_j": vi,
            "type": "auto",
            "n_samples": n_samples,
            "n_bins": int(h_lag.shape[0]),
            "r2_wls": float(r2_wls),
            "r2_ols": float(r2_ols),
            **{k: float(params.get(k, np.nan)) for k in param_keys},
        })

    # --- Off-diagonal: cross-variograms (compute for i<j)
    for i in range(p):
        vi = values_cols[i]
        vals_i = df[vi].to_numpy(dtype=float)

        for j in range(i + 1, p):
            vj = values_cols[j]
            vals_j = df[vj].to_numpy(dtype=float)

            # Get cross-variogram for each pair (vi, vj)
            res = crossvariofit(
                values1=vals_i,
                values2=vals_j,
                coordinates=coords,
                distance_type=distance_type,
                max_distance=max_distance,
                bin_size=bin_size,
                estimator_type="Matheron",  # strictly enforce 'Matheron' here
                model_type=model_type,
                weight_fn=weight_fn,
                weight_params=weight_params,
                xmax_factor=xmax_factor,
                fix_nugget=fix_nugget,
                fix_sill=fix_sill,
                allow_negative_sill=allow_negative_sill,
                plot=plot_single,
                balltree_leaf_size=balltree_leaf_size,
                max_neighbors=max_neighbors,
                transform=transform,
                lag_repr=lag_repr,
            )

            h_lag, n_obs, gamma, params, r2_wls, r2_ols = res

            # Store results for both orientations (vi, vj) and (vj, vi)
            results[(vi, vj)] = res
            results[(vj, vi)] = res

            if param_keys is None:
                param_keys = list(params.keys())

            summary_rows.append({
                "var_i": vi,
                "var_j": vj,
                "type": "cross",
                "n_samples": n_samples,
                "n_bins": int(h_lag.shape[0]),
                "r2_wls": float(r2_wls),
                "r2_ols": float(r2_ols),
                **{k: float(params.get(k, np.nan)) for k in param_keys},
            })

    summary = pd.DataFrame(
        summary_rows,
        columns=["var_i", "var_j", "type", "n_samples", "n_bins", "r2_wls", "r2_ols"] + (param_keys or [])
    )

    # --- Build parameter matrices
    idx = values_cols
    param_mats = {}
    for k in (param_keys or []):
        mat = pd.DataFrame(np.nan, index=idx, columns=idx, dtype=float)
        for a in idx:
            for b in idx:
                if (a, b) in results:
                    params_ab = results[(a, b)][3]
                    mat.loc[a, b] = float(params_ab.get(k, np.nan))
        param_mats[k] = mat

    # --- R2 matrices
    r2_wls_mat = pd.DataFrame(np.nan, index=idx, columns=idx, dtype=float)
    r2_ols_mat = pd.DataFrame(np.nan, index=idx, columns=idx, dtype=float)
    for a in idx:
        for b in idx:
            if (a, b) in results:
                _, _, _, _, r2w, r2o = results[(a, b)]
                r2_wls_mat.loc[a, b] = float(r2w)
                r2_ols_mat.loc[a, b] = float(r2o)

    r2_mats = {"r2_wls": r2_wls_mat, "r2_ols": r2_ols_mat}

    # --- Matrix plot: lower triangle only
    if plot_matrix:
        nmax = int(np.ceil(float(max_distance) / float(bin_size)))
        x_plot_max = float(np.max(_make_lag_axis(nmax, bin_size, lag_repr=lag_repr)) + bin_size / 2.0)
        xfit = np.linspace(0.0, x_plot_max, 600)

        if transform is None:
            y_all = []
            for (vi, vj), res in results.items():
                i_idx = values_cols.index(vi)
                j_idx = values_cols.index(vj)
                if j_idx > i_idx:
                    continue
                _, _, gamma_ij, _, _, _ = res
                y_all.extend(gamma_ij.ravel())

            y_all = np.asarray(y_all, float)
            y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
            y_min = min(0.0, y_min)
            pad = 0.05 * (y_max - y_min)

        else:
            y_min, y_max = -1.0, 1.0
            pad = 0.0

        fig, axes = plt.subplots(
            p, p, figsize=(2.4 * p, 2.4 * p), dpi=200, sharex=False, sharey=True
        )

        if p == 1:
            axes = np.array([[axes]])

        for i, vi in enumerate(values_cols):
            for j, vj in enumerate(values_cols):
                ax = axes[i, j]

                if j > i:
                    ax.set_axis_off()
                    continue

                res = results.get((vi, vj), None)
                if res is None:
                    ax.set_axis_off()
                    continue

                h_lag, n_obs, gamma_ij, params_ij, r2w, r2o = res
                h = h_lag.ravel()
                g = gamma_ij.ravel()

                if i == j:
                    gfit = _evaluate_model_from_params(
                        model_type, xfit, params_ij, transform=transform
                    )
                else:
                    gfit = _evaluate_model_from_params(
                        model_type, xfit, params_ij, transform=transform
                    )

                ax.plot(h, g, 'o', ms=2.5, markeredgecolor='black')

                if transform is None:
                    _plot_variogram_model_piecewise(
                        ax,
                        xfit,
                        gfit,
                        params=params_ij,
                        color='k',
                        lw=0.8,
                        ls='-',
                        zorder=2,
                        show_zero_point=True
                    )
                else:
                    if i == j:
                        _plot_correlation_model_piecewise(
                            ax,
                            xfit,
                            gfit,
                            params=params_ij,
                            color='k',
                            lw=0.8,
                            ls='-',
                            zorder=2,
                            show_zero_point=True
                        )
                    else:
                        _plot_correlation_model_piecewise(
                            ax,
                            xfit,
                            gfit,
                            params=params_ij,
                            color='k',
                            lw=0.8,
                            ls='-',
                            zorder=2,
                            show_zero_point=True
                        )

                ax.set_title(f"{vi} × {vj}" if i != j else vi, fontsize=8)

                ax.set_xlim(0, x_plot_max)
                ax.set_ylim(y_min - pad, y_max + pad)
                ax.grid(True, linestyle='--', alpha=0.2)

                if transform == "correlation":
                    ax.set_yticks([-1.00, -0.50, 0.00, 0.50, 1.00])

                if i == p - 1:
                    ax.set_xlabel("lag", fontsize=8)
                else:
                    ax.set_xlabel("")

                if j == 0:
                    if transform is None:
                        ax.set_ylabel("γ", fontsize=8)
                    else:
                        ax.set_ylabel("ρ", fontsize=8)
                else:
                    ax.set_ylabel("")

        plt.tight_layout()
        plt.show()

    return summary, results, param_mats, r2_mats
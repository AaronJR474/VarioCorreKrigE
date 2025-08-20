"""
This file contains the functions required for estimating the correlation coefficients for correleograms as well as
options to fit desired models to the data.
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
from scipy import stats

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

# Correlation Estimators
def pearsonr_uncen(value1, value2):
    """
    Pearson *uncentered* correlation (cosine similarity).

    Computes
        rho_u = sum_i (x_i y_i) / sqrt( sum_i x_i^2 * sum_i y_i^2 )

    i.e., the cosine of the angle between the two vectors, without subtracting
    means. This is sometimes called *cosine similarity* and differs from the
    traditional (centered) Pearson correlation.

    Parameters
    ----------
    value1 : array-like of shape (n,)
        First vector (e.g., Z_i values from pairs in a lag bin).
    value2 : array-like of shape (n,)
        Second vector (e.g., Z_j values from the same pairs).
        Must correspond elementwise to `value1`.
    Returns
    -------
    rho : float
        Uncentered correlation in [-1, 1]. If fewer than two finite pairs are
        available, or if either vector has zero L2 norm after filtering finite
        pairs, returns `np.nan`.

    Notes
    -----
    • More sensitive to scale and offset than centered Pearson.
    • If you want invariance to additive offsets, use `pearsonr_cen`.
    • NaN/Inf handling should be performed by the caller or within this
      function by masking to finite pairs prior to computation.
    """

    if value1.size < 3:
        return np.nan

    p_uncen = ((np.sum(value1 * value2) * (1 / value1.size)) /
              (np.sqrt((1 / value1.size) * np.sum(value1 ** 2)) *
                np.sqrt((1 / value1.size) * np.sum(value2 ** 2))
    ))

    return p_uncen

def pearsonr_cen(value1, value2):

    """
    Pearson correlation coefficient (centered).

    Computes the standard Pearson correlation (mean-centered) between the two
    vectors. Typical implementation uses `scipy.stats.pearsonr` under the hood.

    Parameters
    ----------
    value1 : array-like of shape (n,)
        First vector (e.g., Z_i values from pairs in a lag bin).
    value2 : array-like of shape (n,)
        Second vector (e.g., Z_j values from the same pairs).

    Returns
    -------
    rho : float
        Centered Pearson correlation in [-1, 1]. If either input is constant
        (zero variance) after filtering finite pairs, SciPy emits a warning and
        returns `np.nan`.
    pvalue : float, optional
        If you propagate the SciPy return value, a two-sided p-value is also
        available. If you only return the coefficient, document that here.

    Notes
    -----
    • SciPy’s `pearsonr` does **not** ignore NaNs by default. You should drop
      non-finite pairs (NaN/Inf) before calling it:
        mask = np.isfinite(value1) & np.isfinite(value2)
        pearsonr(value1[mask], value2[mask])
    • Requires at least three finite observations.
    """

    if value1.size < 3:
        return np.nan

    res = stats.pearsonr(value1, value2).statistic

    return res

def spearmanr_bin(value1, value2):
    """
    Spearman's rank correlation coefficient (ρ).

    Computes Pearson correlation on the ranks of the data (monotonic association),
    typically via `scipy.stats.spearmanr`, which handles ties by average ranks.

    Parameters
    ----------
    value1 : array-like of shape (n,)
        First vector (e.g., Z_i values from pairs in a lag bin).
    value2 : array-like of shape (n,)
        Second vector (e.g., Z_j values from the same pairs).

    Returns
    -------
    rho : float
        Spearman rank correlation in [-1, 1]. If fewer than two finite pairs
        remain after filtering, returns `np.nan`.
    pvalue : float, optional
        If you propagate the SciPy return value, a two-sided p-value is also
        available. If you only return ρ, document that here.

    Notes
    -----
    • More robust to outliers and nonlinear monotone relationships than Pearson.
    • SciPy’s `spearmanr` does not ignore NaNs by default; drop non-finite
      pairs before calling:
        mask = np.isfinite(value1) & np.isfinite(value2)
        spearmanr(value1[mask], value2[mask])
    • Requires at least three finite observations.

    """
    if value1.size < 3:
        return np.nan

    res = stats.spearmanr(value1, value2).statistic

    return res

# Correlation Models
def _apply_alpha(h, R0, alpha):
    """Enforce rho(0)=1, scale R0(h) by alpha for h>0."""
    h = np.asarray(h, float)
    out = np.asarray(R0, float)
    # for h==0 set to 1 exactly; for h>0 multiply by alpha
    return np.where(h == 0.0, 1.0, alpha * out)

def spherical(h, r, alpha=1.0):
    """
    Correlation kernel: Spherical (compact support)

    Definition
    ----------
    Let a = r and x = h / a. The unscaled kernel R0(h) is
        R0(h) = 1 - [1.5 x - 0.5 x^3]    for 0 <= x <= 1
                0                         for x  >  1
    The returned correlation enforces rho(0) = 1 and applies an "alpha" scale
    away from zero-lag (nugget-like discontinuity):
        rho(h) = 1                   if h == 0
                 alpha * R0(h)       if h  > 0

    Parameters
    ----------
    h : array-like or float
        Nonnegative lag distance(s).
    r : float
        Effective range; equals the compact-support radius a.
    alpha : float, default 1.0
        0 <= alpha <= 1. Scales correlation for h > 0 while keeping rho(0) = 1.

    Returns
    -------
    rho : ndarray or float
        Correlation values with the same shape as `h`.

    Notes
    -----
    R0(0) = 1; R0 is nonnegative and vanishes for h >= r.
    """

    a = float(r)
    h = np.asarray(h, float)
    x = h / a
    R0 = np.where(h <= a, 1.0 - (1.5*x - 0.5*x**3), 0.0)
    return _apply_alpha(h, R0, alpha)

def exponential(h, r, alpha=1.0):
    """
    Correlation kernel: Exponential

    Definition
    ----------
    Effective range r corresponds to a = r / 3 (95% decorrelation).
    Unscaled kernel:
        R0(h) = exp(-h / a)
    Returned correlation:
        rho(h) = 1                   if h == 0
                 alpha * R0(h)       if h  > 0

    Parameters
    ----------
    h : array-like or float
        Nonnegative lag distance(s).
    r : float
        Effective range; mapped to a = r / 3.
    alpha : float, default 1.0
        0 <= alpha <= 1. Scales correlation for h > 0; rho(0) = 1 always.

    Returns
    -------
    rho : ndarray or float
        Correlation values with the same shape as `h`.

    Notes
    -----
    Monotone decreasing, long-tailed compared to Gaussian.
    """

    a = float(r) / 3.0
    h = np.asarray(h, float)
    R0 = np.exp(-h / a)
    return _apply_alpha(h, R0, alpha)

def gaussian(h, r, alpha=1.0):
    """
    Correlation kernel: Gaussian

    Definition
    ----------
    Effective range r corresponds to a = r / 2 (≈95% decorrelation).
    Unscaled kernel:
        R0(h) = exp( - (h / a)^2 )
    Returned correlation:
        rho(h) = 1                   if h == 0
                 alpha * R0(h)       if h  > 0

    Parameters
    ----------
    h : array-like or float
        Nonnegative lag distance(s).
    r : float
        Effective range; mapped to a = r / 2.
    alpha : float, default 1.0
        0 <= alpha <= 1. Scales correlation for h > 0; rho(0) = 1 always.

    Returns
    -------
    rho : ndarray or float
        Correlation values with the same shape as `h`.

    Notes
    -----
    Very smooth; decays faster than exponential.
    """

    a = float(r) / 2.0
    h = np.asarray(h, float)
    R0 = np.exp(- (h / a)**2)
    return _apply_alpha(h, R0, alpha)

def cubic(h, r, alpha=1.0):
    """
    Correlation kernel: Cubic (compact support)

    Definition
    ----------
    Let a = r and x = h / a. The unscaled kernel R0(h) is
        R0(h) = 1 - [ 7 x^2 - (35/4) x^3 + (7/2) x^5 - (3/4) x^7 ]  for 0 <= x <= 1
                0                                                   for x  >  1
    Returned correlation:
        rho(h) = 1                   if h == 0
                 alpha * R0(h)       if h  > 0

    Parameters
    ----------
    h : array-like or float
        Nonnegative lag distance(s).
    r : float
        Effective range; equals the compact-support radius a.
    alpha : float, default 1.0
        0 <= alpha <= 1. Scales correlation for h > 0; rho(0) = 1 always.

    Returns
    -------
    rho : ndarray or float
        Correlation values with the same shape as `h`.

    Notes
    -----
    Compact support like spherical but with a different interior shape.
    """

    a = float(r)
    h = np.asarray(h, float)
    x = h / a
    poly = 7.0*x**2 - (35.0/4.0)*x**3 + (7.0/2.0)*x**5 - (3.0/4.0)*x**7
    R0 = np.where(h <= a, 1.0 - poly, 0.0)
    return _apply_alpha(h, R0, alpha)

def powered_exponential(h, r, beta, alpha=1.0):
    """
    Correlation kernel: Powered exponential (a.k.a. Stable)

    Definition
    ----------
    Effective range r corresponds to a = r / (3)^(1/beta) (≈95% decorrelation).
    Unscaled kernel:
        R0(h) = exp( - (h / a)^beta ),   with 0 < beta <= 2
    Returned correlation:
        rho(h) = 1                   if h == 0
                 alpha * R0(h)       if h  > 0

    Parameters
    ----------
    h : array-like or float
        Nonnegative lag distance(s).
    r : float
        Effective range; mapped to a = r / (3)^(1/beta).
    beta : float
        Shape exponent, 0 < beta <= 2. beta=1 gives exponential; beta=2 gives Gaussian.
    alpha : float, default 1.0
        0 <= alpha <= 1. Scales correlation for h > 0; rho(0) = 1 always.

    Returns
    -------
    rho : ndarray or float
        Correlation values with the same shape as `h`.

    Notes
    -----
    Interpolates smoothly between exponential (beta=1) and Gaussian (beta=2).
    """

    a = float(r) / (3.0 ** (1.0 / float(beta)))
    h = np.asarray(h, float)
    R0 = np.exp(- (h / a)**beta)
    return _apply_alpha(h, R0, alpha)

def matern(h, r, nu, alpha=1.0):
    """
    Correlation kernel: Matérn

    Definition
    ----------
    Uses the same parameterization as your earlier code: set a = r / 2 and
    u = 2 * (h * sqrt(nu)) / a. The unscaled kernel is
        R0(h) = (2 / Gamma(nu)) * ((h * sqrt(nu)) / a)^nu * K_nu( 2 * (h * sqrt(nu)) / a )
    where K_nu is the modified Bessel function of the second kind.
    Returned correlation:
        rho(h) = 1                   if h == 0
                 alpha * R0(h)       if h  > 0

    Parameters
    ----------
    h : array-like or float
        Nonnegative lag distance(s).
    r : float
        Effective range; mapped to a = r / 2 (≈95% decorrelation under this scaling).
    nu : float
        Smoothness parameter (ν > 0). Smaller ν → rougher field; large ν → approaches Gaussian.
    alpha : float, default 1.0
        0 <= alpha <= 1. Scales correlation for h > 0; rho(0) = 1 always.

    Returns
    -------
    rho : ndarray or float
        Correlation values with the same shape as `h`.

    Notes
    -----
    Implementation uses safe handling at h=0 (rho(0)=1) and relies on scipy.special.kv.
    This parameterization is equivalent to the standard Matérn up to a rescaling of 'a'.
    """

    a  = float(r) / 2.0
    nu = float(nu)
    h  = np.asarray(h, float)

    # argument for modified Bessel K_nu
    u = 2.0 * (h * np.sqrt(nu)) / a
    with np.errstate(divide='ignore', invalid='ignore'):
        term = (2.0 / special.gamma(nu)) * ((h * np.sqrt(nu)) / a)**nu * special.kv(nu, u)

    # set exactly to 1 at h=0
    R0 = term
    if np.ndim(h) == 0:
        R0 = 1.0 if h == 0.0 else term
    else:
        R0 = np.where(h == 0.0, 1.0, term)

    return _apply_alpha(h, R0, alpha)

def damped_cosine_angle(theta_deg, c, alpha=1.0):
    """
    Correlation kernel: Damped cosine in angle (degrees)

    Definition
    ----------
    For angular separation θ (in degrees), the unscaled kernel is
        R0(θ) = cos(θ * π/180) * exp(-θ / c)
    Returned correlation:
        rho(θ) = 1                     if θ == 0
                 alpha * R0(θ)         if θ  > 0

    Parameters
    ----------
    theta_deg : array-like or float
        Angular separation(s) in degrees (0 <= θ <= 180 typically).
    c : float
        Damping angle in degrees; larger c → slower decay with angle.
    alpha : float, default 1.0
        0 <= alpha <= 1. Scales correlation for θ > 0; rho(0) = 1 always.

    Returns
    -------
    rho : ndarray or float
        Correlation values with the same shape as `theta_deg`.

    Notes
    -----
    Use only when lags represent angular distances, not linear distances.
    """

    theta_deg = np.asarray(theta_deg, float)
    th = np.radians(theta_deg)
    R0 = np.cos(th) * np.exp(-theta_deg / float(c))
    return _apply_alpha(theta_deg, R0, alpha)  # theta=0 -> 1; else alpha*R0

def angular_dissimilarity(theta_deg, c, alpha=1.0):
    """
    Padonou–Roustant (2016) angular correlation kernel (degrees).

    Unscaled kernel:
        R0(θ) = (1 + θ/c) * (1 - θ/180)^(180/c),  θ in [0, 180], c > 0.
    Returned correlation:
        ρ(θ) = 1                      if θ == 0
               α · R0(θ)              if θ  > 0

    Parameters
    ----------
    theta_deg : array-like or float
        Angular separation(s) in degrees. Values will be clamped to [0, 180].
    c : float
        Damping/length parameter in degrees (c > 0).
    alpha : float, default 1.0
        Scale for θ>0 (0 < alpha ≤ 1). Keeps ρ(0) = 1 exactly.

    Returns
    -------
    rho : ndarray or float
        Correlation values with the same shape as `theta_deg`.

    Notes
    -----
    - For small θ, R0(θ) ≈ (1 + θ/c) * exp(-θ/c) ≤ 1, so it is well-behaved around 0.
    - At θ = 180°, the base term is 0, hence R0=0.
    - If your angles can exceed 180° and you still want this kernel, fold to [0,180].
    """
    theta = np.asarray(theta_deg, float)
    # clamp to valid domain
    th = np.clip(theta, 0.0, 180.0)

    # base term in [0,1]; avoid negative base^fraction
    base = np.clip(1.0 - th / 180.0, 0.0, 1.0)

    # exponent and prefactor
    c = float(c)
    if c <= 0.0:
        raise ValueError("Parameter c must be > 0 for angular_dissimilarity.")

    exponent = 180.0 / c
    pref = 1.0 + (th / c)

    # safe power: base**exponent where base>0, else 0
    pow_term = np.where(base > 0.0, np.power(base, exponent), 0.0)
    R0 = pref * pow_term

    # enforce rho(0)=1 and apply alpha for θ>0 (use the *original* theta for the 0-test)
    return _apply_alpha(theta, R0, alpha)


CORRELATION_MODELS = {
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
def objective_func(params, h, rho, weights, correlation_fn):
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

    rho_pred = correlation_fn(h, *params)
    return np.sum(weights * (rho - rho_pred)**2)

def _small_lag_alpha0(h, rho, k=3):
    """
    Robust initial alpha from the k smallest positive lags.
    Falls back to max(rho) if needed; clipped to (0,1].
    """
    h = np.asarray(h, float).ravel()
    r = np.asarray(rho, float).ravel()
    mask = np.isfinite(h) & np.isfinite(r) & (h > 0)
    if not np.any(mask):
        a0 = np.nanmax(r) if np.any(np.isfinite(r)) else 1.0
    else:
        idx = np.argsort(h[mask])[:min(k, np.count_nonzero(mask))]
        a0 = np.nanmedian(r[mask][idx])
        if not np.isfinite(a0):
            a0 = np.nanmax(r) if np.any(np.isfinite(r)) else 1.0
    return float(np.clip(a0, 1e-3, 1.0))

def make_init_and_bounds(model, h, rho, xmax_factor=2.0, fix_alpha=True):

    """
    Initial guesses & bounds for *correlation* models.

    Parameter layouts
    -----------------
    spherical/exponential/gaussian/cubic : (r, alpha)
    powered_exponential                  : (r, beta, alpha)
    matern                               : (r, nu, alpha)
    damped_cosine_angle                  : (c_deg, alpha)

    Notes
    -----
    - alpha enforces ρ(0)=1 then scales ρ(h>0) by alpha∈(0,1] (nugget scaling).
    - If `fix_alpha`, alpha is fixed at 1 using bounds (1,1).
    - Range-like params capped at `xmax_factor * max(h)`.
    """

    h = np.asarray(h, float).ravel()
    r = np.asarray(rho, float).ravel()

    # positive lags only for scale calculations
    mask_pos = np.isfinite(h) & (h > 0)
    h_min = float(np.nanmin(h[mask_pos])) if np.any(mask_pos) else 1.0
    h_max = float(np.nanmax(h[mask_pos])) if np.any(mask_pos) else 1.0

    # initial guesses
    r0 = 0.5 * h_max if np.isfinite(h_max) and h_max > 0 else 1.0
    alpha0 = 1.0 if fix_alpha else np.clip(
        np.nanmedian(r[np.argsort(h[mask_pos])[:min(3, np.count_nonzero(mask_pos))]])
        if np.any(mask_pos) else np.nanmax(r) if np.any(np.isfinite(r)) else 1.0,
        1e-3, 1.0
    )

    # range bounds: lower > 0; upper capped at xmax_factor * h_max
    r_lo = max(1e-6, 0.5 * h_min)
    r_hi = xmax_factor * h_max if np.isfinite(h_max) and h_max > 0 else None
    r_bounds = (r_lo, r_hi)

    # alpha bounds
    alpha_bounds = (1.0, 1.0) if fix_alpha else (1e-6, 1.0)

    if model in ("spherical", "exponential", "gaussian", "cubic"):
        x0 = (r0, alpha0)
        bounds = (r_bounds, alpha_bounds)

    elif model == "powered_exponential":
        beta0 = 1.0
        x0 = (r0, beta0, alpha0)
        bounds = (
            r_bounds,          # r (capped at 2*max(h))
            (1e-2, 2.0),       # beta
            alpha_bounds       # alpha
        )

    elif model == "matern":
        nu0 = 0.5
        x0 = (r0, nu0, alpha0)
        bounds = (
            r_bounds,          # r (capped at 2*max(h))
            (1e-3, 5.0),       # nu
            alpha_bounds       # alpha
        )

    elif model in ("damped_cosine_angle", "angular_dissimilarity"):
        # theta is in degrees; valid separations are 0..180
        deg_cap = 180.0

        # positive, finite lags only for scale calculation
        mask_pos = np.isfinite(h) & (h > 0)
        h_min = float(np.nanmin(h[mask_pos])) if np.any(mask_pos) else 1.0
        h_max = float(np.nanmax(h[mask_pos])) if np.any(mask_pos) else 1.0

        # initial guess ~ half the max observed angle, but never < 1e-3
        c0 = max(1e-3, 0.5 * h_max if np.isfinite(h_max) and h_max > 0 else 1.0)
        # lower bound ~ half the smallest nonzero angle (keeps c > 0 and avoids overfitting tiny scales)
        c_lo = max(1e-3, 0.5 * h_min if np.isfinite(h_min) and h_min > 0 else 1e-3)

        # upper bound: different per kernel
        if model == "angular_dissimilarity":
            # clamp at 180°, as the kernel is defined on [0, 180]
            c_hi_raw = xmax_factor * h_max if np.isfinite(h_max) and h_max > 0 else deg_cap
            c_hi = min(deg_cap, c_hi_raw)
        else:  # damped_cosine_angle
            # no need to clamp at 180°; allow slow damping if needed
            c_hi = xmax_factor * h_max if np.isfinite(h_max) and h_max > 0 else None

        x0 = (c0, alpha0)
        bounds = ((c_lo, c_hi), alpha_bounds)

    else:
        raise ValueError("Unknown model")

    return x0, bounds

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
    Correlation-model parameter packing.

    Parameter layouts expected by CORRELATION_MODELS:

      - spherical / exponential / gaussian / cubic : (r, alpha)
      - powered_exponential                        : (r, beta, alpha)
      - matern                                     : (r, s, alpha)     # s = smoothness (ν)
      - damped_cosine_angle/angular dissimilarity  : (c, alpha)        # c = damping angle (degrees)

    where
      r     : range-like parameter (model-specific mapping to scale 'a')
      beta  : shape exponent for powered exponential (0 < beta ≤ 2)
      s     : Matérn smoothness ν (typically 1e-3 ≤ s ≤ 5)
      c     : angular damping (degrees) for the damped-cosine model
      alpha : correlation scale for h > 0 (0 ≤ alpha ≤ 1), with ρ(0) = 1

    Notes
    -----
    The correlation kernels are implemented to satisfy ρ(0)=1.
    The parameter alpha scales ρ(h) away from zero-lag, allowing a
    reduction in correlation at infinitesimal lags analogous to a nugget
    effect in variograms (i.e., α ≈ c0 / (b + c0) under unit variance).
    """
    if model_type in ("spherical", "exponential", "gaussian", "cubic"):
        names = ("r", "alpha")
    elif model_type == "powered_exponential":
        names = ("r", "beta", "alpha")
    elif model_type == "matern":
        names = ("r", "s", "alpha")  # 's' plays the role of nu
    elif model_type in ("damped_cosine_angle","angular_dissimilarity"):
        names = ("c", "alpha")       # c in degrees
    else:
        raise ValueError("Unknown model_type")
    return {k: float(v) for k, v in zip(names, theta)}

def theta_from_params(params, model_type):
    """
    Unpack in the same order expected by your CORRELATION_MODELS call signatures.
    """
    if model_type in ("spherical", "exponential", "gaussian", "cubic"):
        order = ("r", "alpha")
    elif model_type == "powered_exponential":
        order = ("r", "beta", "alpha")
    elif model_type == "matern":
        order = ("r", "s", "alpha")
    elif model_type in ("damped_cosine_angle","angular_dissimilarity"):
        order = ("c", "alpha")
    else:
        raise ValueError("Unknown model_type")
    return [float(params[k]) for k in order]

# Main function: correfit
def correfit(values, coordinates, distance_type, max_distance, bin_size, correlation_type, model_type, weight_fn,
             weight_params, max_lagfit_factor = 2, fix_alpha = True, plot = False):

    """
    Compute an empirical correlogram and fit a parametric correlation model.

    Parameters
    ----------
    values : (n,) array_like of float
        Sample values Z_i.
    coordinates : (n,d) array_like of float
        Sample locations. If distance_type=='geographic', columns are (lat_deg, lon_deg).
    distance_type : {'geographic','euclidean','cartesian','angular'}
        Distance metric for lag binning.
    max_distance : float
        Max separation (same units as chosen distance metric).
    bin_size : float
        Bin width (same units as max_distance).
    correlation_type : {'pearsonr','uncentered pearsonr','spearman'}
        Per-bin correlation estimator.
    model_type : see CORRELATION_MODELS
        Correlation kernel name.
    weight_fn : {'ols','inverse-linear weighting','exponential weighting','powered_weighting', None}, optional
        Bin-weight scheme (None/'ols' → equal weights).
    weight_params : dict or list, optional
        Params for `weight_fn`. If dict, keys {'b','alpha'}; if list, [b, alpha].
    max_lagfit_factor : float, default 2.0
        Upper cap for range-like parameters (r or c).
    fix_alpha : bool, default True
        If True, fixes alpha=1 (ρ(0)=1 and no nugget scaling).
    plot : bool, default False
        If True, render 2-panel plot (pair counts; empirical vs model ρ(h)).

    Returns
    -------
    h_lag : (k,1) ndarray of float
    n_obs : (k,1) ndarray of float
    rho   : (k,1) ndarray of float
    params : dict[str,float]
        Fitted parameters in the order documented by `pack_params`.
    r2_wls : float
        Weighted R² using the same weights used in fitting.
    r2_ols : float
        Ordinary R² using ones as weights used in fitting.
    """

    # ensure arrays
    values = np.asarray(values, float)
    coords  = np.asarray(coordinates, float)
    n = len(values)

    # Define arrays for storage
    nmax = round(max_distance/bin_size)
    h_lag = np.zeros((nmax,1))
    rho = np.zeros((nmax,1))
    n_obs = np.zeros((nmax,1))

    # pick correlation estimator
    if correlation_type == "uncentered pearsonr":
        correlation_fn = pearsonr_uncen
    elif correlation_type == "pearsonr":
        correlation_fn = pearsonr_cen
    elif correlation_type == "spearman":
        correlation_fn = spearmanr_bin
    else:
        raise ValueError("Invalid estimator: choose from 'uncentered pearsonr', 'pearsonr', or 'spearman'")

    # pick correlation model
    if model_type == "exponential":
        correlationmodel_fn = exponential
    elif model_type == "cubic":
        correlationmodel_fn = cubic
    elif model_type == "powered_exponential":
        correlationmodel_fn = powered_exponential
    elif model_type == "matern":
        correlationmodel_fn = matern
    elif model_type == "gaussian":
        correlationmodel_fn = gaussian
    elif model_type == "spherical":
        correlationmodel_fn = spherical
    elif model_type == "damped_cosine_angle":
        correlationmodel_fn = damped_cosine_angle
    elif model_type == "angular_dissimilarity":
        correlationmodel_fn = angular_dissimilarity
    else:
        raise ValueError("Invalid Model: Choose from 'exponential', 'cubic', 'powered_exponential', 'matern', "
                         "'spherical', 'gaussian', 'damped_cosine_angle' or 'angular_dissimilarity'")

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

    # compute Correlation Coefficient
    for i in range(1,nmax+1):
        [site1, site2] = np.where(distance_ratio == i)
        if len(site1) > 0:
            h_lag[i-1,0]=bin_size/2+(i-1)*bin_size
            n_obs[i-1,0]= len(site1)
            rho[i-1, 0] = correlation_fn(values[site1], values[site2])
        else:
            h_lag[i-1,0]=np.nan
            n_obs[i-1,0]=np.nan
            rho[i-1,0]=np.nan

    # drop empty bins/nan estimates in semivariance
    keep = ~np.isnan(rho).any(axis=1)
    h_lag = h_lag[keep]
    n_obs = n_obs[keep]
    rho = rho[keep]

    # fit model
    h = h_lag.ravel()
    g = rho.ravel()
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
    x0, bounds = make_init_and_bounds(model_type, h, g, xmax_factor=max_lagfit_factor, fix_alpha = fix_alpha)

    #  optimize
    res = minimize(
        fun=lambda th: objective_func(th, h, g, weights, correlationmodel_fn),
        x0=x0,
        bounds=bounds
        # method="L-BFGS-B"
    )
    theta_hat = res.x

    # get rsquared fit
    g_fit_bins = correlationmodel_fn(h, *theta_hat)
    r2_wls = r2_score_weighted(g, g_fit_bins, w=weights)
    r2_ols = r2_score_weighted(g, g_fit_bins, w=None)

    # store parameters
    params = pack_params(model_type, theta_hat)

    # smooth curve for plotting
    xlag_fit = np.linspace(0.0, float(max(h_lag[:,0])+bin_size/2), 1000)
    rho_pred = correlationmodel_fn(xlag_fit, *theta_hat)

    # Create Plot for Semi-variance
    if plot:
        fig = plt.figure(figsize=(12,7),dpi = 200)
        # set height ratios for subplots
        gs_plot = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

        # the first subplot
        ax0 = plt.subplot(gs_plot[0])
        ax0.bar(h_lag[:,0],n_obs[:,0], edgecolor='black', align='center', width=bin_size/2)
        ax0.grid(which = 'minor')
        yt = ax0.get_yticks()
        if yt.size > 1:
            ax0.set_yticks(yt[1:])
        # the second subplot
        # shared axis X
        ax1 = plt.subplot(gs_plot[1], sharex = ax0)
        ax1.plot(h_lag[:,0],rho[:,0],'o',markeredgecolor='black')
        ax1.plot()
        ax1.plot(xlag_fit, rho_pred, '-k')
        ax1.axhline(0.0, color='k', lw=1.0, ls='--', alpha=0.7)
        plt.setp(ax0.get_xticklabels(), visible=False)

        # remove last tick label for the second subplot
        yticks = ax1.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(True)
        ax0.set_ylabel('Number of Lags, N', labelpad=22)
        ax0.set_ylim(0, max(n_obs[:,0]))

        # set legend
        ax1.legend(['Experimental', r'Model, $R^2$ (WLS|OLS) = %.2f|%.2f'%(r2_wls, r2_ols)], loc='lower left')

        # set minor ticks
        ax1.set_xticks(h_lag[:,0])
        ax0.xaxis.grid(True, which='major',linestyle='--')
        ax1.xaxis.grid(True, which='major',linestyle='--')
        ax1.set_yticks([-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])

        # remove vertical gap between subplots
        plt.xlim(0, max(h_lag[:,0])+bin_size/2)
        plt.ylim(-1,1)
        plt.ylabel(r'Correlation Coefficient, $\rho$ (%s)'%correlation_type)
        plt.xlabel('lag distance')
        plt.subplots_adjust(hspace=.0)
        plt.show(fig)
    else:
        pass

    return h_lag, n_obs, rho, params, r2_wls, r2_ols

# Main Function: correfitmulti
def correfitmulti(
    df,
    values_col,
    index_col,
    coord_cols,
    distance_type,
    max_distance,
    bin_size,
    correlation_type,
    model_type,
    weight_fn=None,
    weight_params=None,
    max_lagfit_factor=2.0,
    fix_alpha=True,
    plot_single =False,
    plot_summary =False,
):
    """
    Fit an empirical correlogram per group in `index_col` and summarize results.

    For each group:
      1) Build pairwise distances and bin by lag.
      2) Estimate empirical correlation per bin (Pearson centered/uncentered or Spearman).
      3) Fit a parametric correlation model (e.g., exponential, Gaussian, Matérn) by
         weighted least squares, with optional distance/pair-count weighting.
      4) Store fitted parameters, weighted R^2, and binned statistics.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table with values, grouping IDs, and coordinates.
    values_col : str
        Column containing the variable to correlate (e.g., 'PGA').
    index_col : str
        Column identifying groups (e.g., event ID); fitting is independent per group.
    coord_cols : (str, str) or list[str]
        If distance_type == 'geographic': (lat_col, lon_col) in degrees.
        If distance_type == 'cartesian' : (x, y)
        If distance_type == 'euclidean' or 'angular': one or more coordinate columns.
    distance_type : {'geographic','euclidean','cartesian','angular'}
        Distance metric for binning pairs by lag.
    max_distance : float
        Maximum separation considered; defines number of lag bins.
    bin_size : float
        Bin width for lag classes.
    correlation_type : {'pearsonr','uncentered pearsonr','spearman'}
        Per-bin correlation estimator.
    model_type : {'spherical','exponential','gaussian','cubic','powered_exponential','matern','damped_cosine_angle'}
        Correlation kernel to fit.
    weight_fn : {'ols','inverse-linear weighting','exponential weighting','powered weighting', None}, optional
        Bin weights. None/'ols' gives equal weights; others apply distance decay and
        multiply by pair counts.
    weight_params : dict or list, optional
        Parameters for the weighting (e.g., {'b': scale, 'alpha': power}).
    max_lagfit_factor : float, default 2.0
        Upper cap for range-like parameters (e.g., r, angular c) as multiple of max(h)
        to prevent near-flat solutions on flexible kernels.
    fix_alpha : bool, default True
        If True, fix nugget scale alpha=1 (no nugget). If False, estimate alpha ∈ (0,1].
    plot_single : bool, default False
        Plot per-group correlogram fit during the loop.
    plot_summary : bool, default False
        After all groups, plot all empirical points (colored by n_obs) and overlay
        the mean / median model and 5–95% band across fitted parameters.

    Returns
    -------
    summary : pandas.DataFrame
        One row per group with counts (n_samples, n_bins), mean/std of values,
        weighted R^2, and fitted parameters in consistent columns.
    df_n_obs : pandas.DataFrame
        Wide matrix of bin pair counts (columns = groups, first column = 'h_lag').
    df_rho : pandas.DataFrame
        Wide matrix of empirical correlations (columns = groups, first column = 'h_lag').
    results : dict
        Mapping {group_id: (h_lag, n_obs, rho, params_dict, r2_wls, r2_ols)} for downstream use.
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
    df_rho  = pd.DataFrame({"h_lag": full_h})
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
        res = correfit(
            values=vals,
            coordinates=coords,
            distance_type=distance_type,
            max_distance=max_distance,
            bin_size=bin_size,
            correlation_type=correlation_type,
            model_type=model_type,
            weight_fn=weight_fn,
            weight_params=weight_params,
            max_lagfit_factor=max_lagfit_factor,
            fix_alpha=fix_alpha,
            plot=plot_single,
        )

        h_lag, n_obs, rho, params, r2_wls, r2_ols = res

        # store per-group result
        results[gid] = res

        # lock param column order from the first group
        if param_keys is None:
            param_keys = list(params.keys())

        # align this group's vectors to the global bin axis
        # round to avoid tiny fp mismatches
        h = np.round(h_lag.ravel(), 12)
        s_n = pd.Series(n_obs.ravel(), index=h)
        s_g = pd.Series(rho.ravel(), index=h)

        df_n_obs[gid] = s_n.reindex(full_h).to_numpy()
        df_rho[gid] = s_g.reindex(full_h).to_numpy()

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
        g_long = df_rho.melt(id_vars='h_lag', var_name='group', value_name='rho')
        n_long = df_n_obs.melt(id_vars='h_lag', var_name='group', value_name='n_obs')
        M = g_long.merge(n_long, on=['h_lag','group'])

        # drop NaNs so x, y, c lengths align
        M = M[M['rho'].notna()]

        fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
        sc = ax.scatter(
            M['h_lag'].to_numpy(),
            M['rho'].to_numpy(),
            c=M['n_obs'].to_numpy(),      # <-- single '=' and 1D
            s=12,
            cmap=plt.get_cmap("coolwarm"),
            norm=mpl.colors.LogNorm(vmin=1),
            alpha=0.8,
            edgecolor='k',
            linewidths=0.1,
            zorder = 1
        )

        # semivariogram model fit plots
        xlag_fit = np.linspace(0.0, float(max_distance), 1000)
        fn = CORRELATION_MODELS[model_type]

        # unpack values for plotting
        thetas = np.array(
            [theta_from_params(params_g, model_type)
             for _, (h_lag_g, n_obs_g, rho_g, params_g, r2_wls_g, r2_ols_g) in results.items()],
            dtype=float
        )

        # mean and percentile parameter vectors
        theta_median = np.nanmedian(thetas, axis=0)
        theta_mean = np.nanmean(thetas, axis=0)
        theta_5 = np.nanpercentile(thetas, 5, axis=0)
        theta_95 = np.nanpercentile(thetas, 95, axis=0)

        # evaluate model with those θ on a smooth grid
        y_median = fn(xlag_fit, *theta_median)
        y_mean = fn(xlag_fit, *theta_mean)
        y_5 = fn(xlag_fit, *theta_5)
        y_95 = fn(xlag_fit, *theta_95)

        # plot mean + band
        ax.plot(xlag_fit, y_mean, color='k', lw=1.5, label='mean fit', zorder=1000)
        ax.plot(xlag_fit, y_median, color='k', ls = '--', lw=1.5, label='median fit', zorder=1000)
        ax.fill_between(xlag_fit, y_5, y_95, color='forestgreen', alpha=0.15, label='5–95% CI fit', zorder=0)

        for gid, (h_lag_g, n_obs_g, rho_g, params_g, r2_wls_g, r2_ols_g) in results.items():
            theta_g = theta_from_params(params_g, model_type)
            y_fit_g = fn(xlag_fit, *theta_g)
            ax.plot(xlag_fit, y_fit_g, '-k', lw=0.2, alpha=0.8)

        cb = plt.colorbar(sc, ax=ax, pad =0.02, fraction=0.04, aspect=40)
        cb.set_label('Observations per bin, n')
        ax.legend(loc='lower left', frameon=False)
        ax.set_xlabel("lag distance")
        ax.set_ylabel(r'Correlation Coefficient, $\rho$ (%s)' % correlation_type)
        ax.set_xlim(0,max_distance)
        ax.set_ylim(-1, 1)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.show()

    return summary, df_n_obs, df_rho, results
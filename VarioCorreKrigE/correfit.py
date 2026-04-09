"""
Tools for estimating empirical correlograms and fitting parametric
correlation models, with optional bootstrap uncertainty quantification.
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
    Compute great-circle distance between two sets of longitude/latitude points.

    Parameters
    ----------
    lon1, lat1 : array-like or float
        Longitudes and latitudes for the first set of points.
    lon2, lat2 : array-like or float
        Longitudes and latitudes for the second set of points.
    radians : bool, default False
        If True, inputs are assumed to already be in radians.
    earth_rad : float, default 6371.227
        Earth radius in kilometres.

    Returns
    -------
    distance : ndarray
        Matrix of pairwise distances in kilometres with shape (n1, n2).

    Notes
    -----
    This function is used internally when `distance_type='geographic'`.
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
        Uncentered correlation in [-1, 1]. If fewer than three finite pairs are
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
        Centered Pearson correlation in [-1, 1]. Returns `np.nan` if fewer than
        three values are provided or if the correlation is undefined.


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
        Spearman rank correlation in [-1, 1]. If fewer than three finite pairs
        remain after filtering, returns `np.nan`.

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

def matern(h, r, s, alpha=1.0):
    """
    Correlation kernel: Matérn

    Definition
    ----------
    Uses the same parameterization as your earlier code: set a = r / 2 and
    u = 2 * (h * sqrt(s)) / a
    R0(h) = (2 / Gamma(s)) * ((h * sqrt(s)) / a)^s * K_s( 2 * (h * sqrt(s)) / a )
    where K_s is the modified Bessel function of the second kind.
    Returned correlation:
        rho(h) = 1                   if h == 0
                 alpha * R0(h)       if h  > 0

    Parameters
    ----------
    h : array-like or float
        Nonnegative lag distance(s).
    r : float
        Effective range; mapped to a = r / 2 (≈95% decorrelation under this scaling).
    s : float
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

    a = float(r) / 2.0
    s = float(s)
    h = np.asarray(h, float)

    u = 2.0 * (h * np.sqrt(s)) / a
    with np.errstate(divide='ignore', invalid='ignore'):
        term = (2.0 / special.gamma(s)) * ((h * np.sqrt(s)) / a)**s * special.kv(s, u)

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
def compute_distance_weights(h_lag, n_j, weight_type='inverse-linear weighting', weight_params=None):
    """
    Build per-bin weights for fitting.

    Parameters
    ----------
    h_lag : (k,) array_like of float
        Bin centres.
    n_j : (k,) array_like of float
        Pair counts per bin.
    weight_type : {'inverse-linear weighting', 'inverse-linear squared weighting',
                   'exponential weighting', 'powered weighting',
                   'linear weighting', None, 'ols'}
        Weighting scheme.
    weight_params : list[float] | tuple[float, ...] | None
        Parameters used by the weighting scheme.
        Expected order is [b, alpha]. For 'inverse-linear weighting' and
        'exponential weighting', only b is used. When calling from higher-level
        wrappers, dictionaries may be converted to this list form before being
        passed here.

    Returns
    -------
    weights : (k,) ndarray of float
        Weight per bin.

    Raises
    ------
    ValueError
        If `weight_type` is unknown or required parameters are missing.
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
def objective_func(params, h, rho, weights, correlation_fn):
    """
    Weighted SSE objective: minimize Σ w_i [rho_i - correlation_fn(h_i; θ)]^2.

    Parameters
    ----------
    params : sequence of float
        Parameter vector in the order expected by `correlation_fn`.
    h : (k,) array_like
        Bin centres.
    rho : (k,) array_like
        Empirical correlation values at the bin centres.
    weights : (k,) array_like
        Per-bin fitting weights.
    correlation_fn : callable
        Signature `correlation_fn(h, *params)` -> (k,) array.

    Returns
    -------
    float
        Weighted sum of squared residuals.
    """

    rho_pred = correlation_fn(h, *params)
    return np.sum(weights * (rho - rho_pred)**2)

def make_init_and_bounds(model, h, rho, xmax_factor=2.0, fix_alpha=True):
    """
    Build initial values and parameter bounds for a correlation-model fit.

    Parameters
    ----------
    model : str
        Name of the correlation kernel.
    h : array-like of float
        Lag-bin centres used in the fit.
    rho : array-like of float
        Empirical correlation values at the same lags.
    xmax_factor : float, default 2.0
        Upper multiplier used to cap range-like parameters relative to the
        largest observed lag.
    fix_alpha : bool, default True
        If True, fix `alpha=1` and fit a model with no nugget jump. If False,
        estimate `alpha` in `(0, 1)`.

    Returns
    -------
    x0 : tuple
        Initial parameter values in the order expected by the chosen kernel.
    bounds : tuple
        Bounds for the optimizer in the same order as `x0`.

    Notes
    -----
    `alpha` controls the drop from `rho(0)=1` to the positive-lag branch.
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
        c_init = max(1e-3, 0.5 * h_max if np.isfinite(h_max) and h_max > 0 else 1.0)
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

        x0 = (c_init, alpha0)
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

def _alpha_from_c0_b(c0, b):
    c0 = float(c0)
    b = float(b)
    sigma2 = c0 + b
    if sigma2 <= 0.0:
        return 0.0
    return c0 / sigma2


def _c0_b_from_alpha(alpha, sigma2=1.0):
    alpha = float(alpha)
    sigma2 = float(sigma2)
    c0 = alpha * sigma2
    b = (1.0 - alpha) * sigma2
    return c0, b, sigma2


def pack_params(model_type, theta, sigma2=1.0):
    """
    Convert fitted kernel parameters into a public parameter dictionary.

    Parameters
    ----------
    model_type : str
        Name of the fitted kernel.
    theta : sequence of float
        Parameter vector in the order expected by the kernel function.
    sigma2 : float, default 1.0
        Total variance used when mapping `alpha` to the variance-style summary
        terms `c0` and `b`.

    Returns
    -------
    params : dict
        Dictionary containing the kernel shape parameters together with:
        - `alpha`: positive-lag scaling
        - `c0`: structured variance component
        - `b`: nugget variance component
        - `sigma2`: total variance

    Notes
    -----
    Under the current parameterization, `alpha = c0 / (c0 + b)`.
    """

    theta = [float(v) for v in theta]

    if model_type in ("spherical", "exponential", "gaussian", "cubic"):
        names = ("r", "alpha")
    elif model_type == "powered_exponential":
        names = ("r", "beta", "alpha")
    elif model_type == "matern":
        names = ("r", "s", "alpha")
    elif model_type in ("damped_cosine_angle", "angular_dissimilarity"):
        names = ("c", "alpha")
    else:
        raise ValueError("Unknown model_type")

    params = {k: float(v) for k, v in zip(names, theta)}

    alpha = float(params.get("alpha", 1.0))
    c0, b, sigma2 = _c0_b_from_alpha(alpha, sigma2=sigma2)

    params["c0"] = c0
    params["b"] = b
    params["sigma2"] = sigma2

    return params

def theta_from_params(params, model_type):
    """
    Unpack parameters in the callable order expected by CORRELATION_MODELS.

    Accepts either:
      - explicit alpha in params, or
      - c0 and b, from which alpha is derived as c0 / (c0 + b).
    """
    params = dict(params)

    if "alpha" in params:
        alpha = float(params["alpha"])
    elif ("c0" in params) or ("b" in params):
        c0 = float(params.get("c0", 0.0))
        b = float(params.get("b", 0.0))
        alpha = _alpha_from_c0_b(c0, b)
    else:
        alpha = 1.0

    if model_type in ("spherical", "exponential", "gaussian", "cubic"):
        return [float(params["r"]), alpha]
    elif model_type == "powered_exponential":
        return [float(params["r"]), float(params["beta"]), alpha]
    elif model_type == "matern":
        return [float(params["r"]), float(params["s"]), alpha]
    elif model_type in ("damped_cosine_angle", "angular_dissimilarity"):
        return [float(params["c"]), alpha]
    else:
        raise ValueError("Unknown model_type")

# Main helpers and fitting functions
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

def _safe_bin_correlation(correlation_fn, x, y, min_pairs=3):
    """
    Safe bin-level correlation with finite filtering and constant checks.
    Returns np.nan if the correlation is undefined.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]

    if x.size < min_pairs:
        return np.nan

    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan

    try:
        val = correlation_fn(x, y)
    except Exception:
        return np.nan

    return float(val) if np.isfinite(val) else np.nan

def _prepare_crosscorrelation_geometry(coordinates, distance_type, bin_size, max_distance, lag_repr="center"):
    """
    Precompute pair geometry and lag-bin assignments for a fixed coordinate set.

    Pair membership is defined by interval binning:
        bin k contains distances in [k*bin_size, (k+1)*bin_size),
    with the final bin capped at `max_distance`.

    This depends only on the coordinates and distance settings, so it can be
    reused across many cross-correlation fits on the same sites.

    Parameters
    ----------
    coordinates : array_like
        Site coordinates.
    distance_type : {'geographic', 'cartesian', 'angular', 'euclidean'}
        Distance metric used to form lag bins.
    bin_size : float
        Bin width.
    max_distance : float
        Maximum retained lag distance.
    lag_repr : {"center", "edge", "upper"}, default "center"
        Representative lag attached to each equal-width bin.

    Returns
    -------
    pair_geometry : dict
        Dictionary containing directional pair indices, interval-bin indices,
        and the representative lag axis.
    """
    coords = np.asarray(coordinates, float)
    n = len(coords)

    if n < 2:
        raise ValueError("Need at least 2 points to compute a cross-correlogram.")

    dt = str(distance_type).lower()

    if dt == "geographic":
        lat = coords[:, 0]
        lon = coords[:, 1]
        distance = np.asarray(
            haversine_oq(lon, lat, lon, lat, radians=False, earth_rad=6371.227),
            dtype=float
        )

    elif dt == "cartesian":
        if coords.shape[1] != 2:
            raise ValueError("cartesian requires coordinates shape (n,2): (x, y)")
        dx = coords[:, None, 0] - coords[None, :, 0]
        dy = coords[:, None, 1] - coords[None, :, 1]
        distance = np.hypot(dx, dy)

    elif dt == "euclidean":
        diff = coords[:, None, :] - coords[None, :, :]
        distance = np.linalg.norm(diff, axis=-1)

    elif dt == "angular":
        theta_deg = np.asarray(coords, float)
        if theta_deg.ndim == 2:
            if theta_deg.shape[1] != 1:
                raise ValueError("angular distance requires a single angular coordinate per row.")
        theta = np.radians(theta_deg.ravel())
        cos_diff = np.cos(theta[:, None] - theta[None, :])
        ang_rad = np.arccos(np.clip(cos_diff, -1.0, 1.0))
        distance = np.degrees(ang_rad)

    else:
        raise ValueError(
            "Invalid distance_type: choose 'geographic', 'cartesian', 'angular', or 'euclidean'"
        )

    nmax = int(np.ceil(float(max_distance) / float(bin_size)))
    if nmax <= 0:
        raise ValueError("max_distance / bin_size must be > 0.")

    i_idx, j_idx = np.triu_indices(n, k=1)
    d = distance[i_idx, j_idx]

    keep = np.isfinite(d) & (d >= 0.0) & (d <= float(max_distance))
    i_idx = i_idx[keep]
    j_idx = j_idx[keep]
    d = d[keep]

    bin_idx = np.floor(d / float(bin_size)).astype(int)
    bin_idx = np.minimum(bin_idx, nmax - 1)

    h_full = _make_lag_axis(nmax, bin_size, lag_repr=lag_repr)

    return {
        "i_idx": i_idx,
        "j_idx": j_idx,
        "bin_idx": bin_idx,
        "h_full": h_full,
    }

def _binned_correlogram_from_precomputed_pairs(values1, values2, pair_geometry, correlation_fn):
    """
    Build a binned correlogram using precomputed pair geometry.
    """
    values1 = np.asarray(values1, float)
    values2 = np.asarray(values2, float)

    i_idx = pair_geometry["i_idx"]
    j_idx = pair_geometry["j_idx"]
    bin_idx = pair_geometry["bin_idx"]
    h_full = pair_geometry["h_full"]

    x = values1[i_idx]
    y = values2[j_idx]

    finite_pair = np.isfinite(x) & np.isfinite(y)
    counts = np.bincount(bin_idx[finite_pair], minlength=h_full.size).astype(float)

    rho_full = np.full(h_full.size, np.nan, dtype=float)

    for k in range(h_full.size):
        if counts[k] == 0:
            continue
        mask_k = (bin_idx == k)
        rho_full[k] = _safe_bin_correlation(
            correlation_fn, x[mask_k], y[mask_k], min_pairs=3
        )

    keep = np.isfinite(rho_full)

    return (
        h_full.reshape(-1, 1),          # all bins
        counts.reshape(-1, 1),          # all counts
        rho_full.reshape(-1, 1),        # all rho
        h_full[keep].reshape(-1, 1),    # valid h
        counts[keep].reshape(-1, 1),    # valid counts
        rho_full[keep].reshape(-1, 1),  # valid rho
    )

def _build_correlation_pair_arrays(values, coordinates, distance_type):
    """
    Build directional correlation pair arrays:
        d = pair distances
        x = values at site i
        y = values at site j

    Uses all off-diagonal directional pairs (i,j), i != j, to preserve the
    behaviour of the original correfit implementation.
    """
    values = np.asarray(values, float)
    coords = np.asarray(coordinates, float)
    n = len(values)

    if n < 2:
        raise ValueError("Need at least 2 points to compute a correlogram.")

    dt = str(distance_type).lower()

    if dt == "geographic":
        lat = coords[:, 0]
        lon = coords[:, 1]
        distance = np.asarray(
            haversine_oq(lon, lat, lon, lat, radians=False, earth_rad=6371.227),
            dtype=float
        )

    elif dt == "cartesian":
        if coords.shape[1] != 2:
            raise ValueError("cartesian requires coordinates shape (n,2): (x, y)")
        dx = coords[:, None, 0] - coords[None, :, 0]
        dy = coords[:, None, 1] - coords[None, :, 1]
        distance = np.hypot(dx, dy)

    elif dt == "euclidean":
        diff = coords[:, None, :] - coords[None, :, :]
        distance = np.linalg.norm(diff, axis=-1)

    elif dt == "angular":
        theta_deg = np.asarray(coords, float)
        if theta_deg.ndim == 2:
            if theta_deg.shape[1] != 1:
                raise ValueError("angular distance requires a single angular coordinate per row.")
        theta = np.radians(theta_deg.ravel())
        cos_diff = np.cos(theta[:, None] - theta[None, :])
        ang_rad = np.arccos(np.clip(cos_diff, -1.0, 1.0))
        distance = np.degrees(ang_rad)

    else:
        raise ValueError(
            "Invalid distance_type: choose 'geographic', 'cartesian', "
            "'angular', or 'euclidean'"
        )

    i_idx, j_idx = np.triu_indices(n, k=1)

    d = distance[i_idx, j_idx]
    x = values[i_idx]
    y = values[j_idx]

    return d, x, y

def _binned_correlogram_from_pairs(d, x, y, bin_size, max_distance, correlation_fn, lag_repr="center"):
    """
    Build full-bin and valid-bin correlogram arrays from pair lists.

    Pair membership is defined by interval binning:
        bin k contains distances in [k*bin_size, (k+1)*bin_size),
    with the final bin capped at `max_distance`.
    """
    nmax = int(np.ceil(float(max_distance) / float(bin_size)))
    if nmax <= 0:
        raise ValueError("max_distance / bin_size must be > 0.")

    d = np.asarray(d, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if not (d.shape == x.shape == y.shape):
        raise ValueError("d, x, and y must have the same shape.")

    mask = np.isfinite(d) & (d >= 0.0) & (d <= float(max_distance))
    if not np.any(mask):
        raise ValueError("No pair distances fell into [0, max_distance]; check max_distance/bin_size.")

    d_use = d[mask]
    x_use = x[mask]
    y_use = y[mask]

    bin_idx = np.floor(d_use / float(bin_size)).astype(int)
    bin_idx = np.minimum(bin_idx, nmax - 1)

    h_full = _make_lag_axis(nmax, bin_size, lag_repr=lag_repr)

    finite_pair = np.isfinite(x_use) & np.isfinite(y_use)
    counts = np.bincount(bin_idx[finite_pair], minlength=nmax).astype(float)

    rho_full = np.full(nmax, np.nan, dtype=float)

    for k in range(nmax):
        if counts[k] == 0:
            continue
        mask_k = (bin_idx == k)
        rho_full[k] = _safe_bin_correlation(
            correlation_fn, x_use[mask_k], y_use[mask_k], min_pairs=3
        )

    keep = np.isfinite(rho_full)

    return (
        h_full.reshape(-1, 1),
        counts.reshape(-1, 1),
        rho_full.reshape(-1, 1),
        h_full[keep].reshape(-1, 1),
        counts[keep].reshape(-1, 1),
        rho_full[keep].reshape(-1, 1),
    )


def _bootstrap_summary(arr, qlo=2.5, qhi=97.5):
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

def _get_alpha_from_params(params):
    """
    Return alpha for the current correlation-model parameterization.

    Priority:
      1) use params['alpha'] if present
      2) else derive alpha = c0 / (c0 + b) from params['c0'], params['b']
      3) else return 1.0
    """
    params = {} if params is None else dict(params)

    if "alpha" in params and params["alpha"] is not None:
        alpha0 = float(params["alpha"])
        if np.isfinite(alpha0):
            return alpha0

    if ("c0" in params) or ("b" in params):
        c0 = float(params.get("c0", 0.0))
        b = float(params.get("b", 0.0))
        sigma2 = c0 + b
        if sigma2 > 0.0:
            return c0 / sigma2

    return 1.0


def _plot_correlation_model_piecewise(ax, x, y, params, color="k", lw=2.0, ls="-",
                                      label=None, zorder=4, alpha_plot=1.0,
                                      show_zero_point=True, jump_ls=":"):
    """
    Plot a correlation model with explicit nugget discontinuity.

    For correlation models with nugget:
        rho(0)  = 1
        rho(0+) = alpha = c0 / (c0 + b)

    The positive-lag branch is plotted for x > 0 only, and a vertical jump
    is drawn at x = 0 when alpha < 1.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    alpha0 = _get_alpha_from_params(params)
    has_nugget = np.isfinite(alpha0) and (not np.isclose(alpha0, 1.0, atol=1e-8, rtol=0.0))

    pos = x > 0

    if has_nugget:
        ax.plot(
            x[pos], y[pos],
            color=color, lw=lw, ls=ls, label=label,
            zorder=zorder, alpha=alpha_plot
        )

        ax.plot(
            [0.0, 0.0], [alpha0, 1.0],
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

def correfit(values, coordinates, distance_type, max_distance, bin_size,
             correlation_type, model_type, weight_fn=None, weight_params=None,
             max_lagfit_factor=2, fix_alpha=True, plot=False,
             bootstrap=None, bootstrap_method="pair",
             bootstrap_ci=(2.5, 97.5), random_state=None, lag_repr="center"):
    """
    Estimate an empirical correlogram and fit a parametric correlation model.

    Parameters
    ----------
    values : array-like of float
        Values observed at the sample locations.
    coordinates : array-like
        Coordinates of the sample locations.
    distance_type : {'geographic', 'euclidean', 'cartesian', 'angular'}
        Distance metric used to form lag bins.
    max_distance : float
        Maximum lag distance included in the empirical correlogram.
    bin_size : float
        Width of each lag bin.
    lag_repr : {'center', 'edge', 'upper'}, default 'center'
        Representative lag attached to each equal-width bin:
        - 'center': midpoint of [k*bin_size, (k+1)*bin_size)
        - 'edge'/'upper': upper edge of [k*bin_size, (k+1)*bin_size)

        This affects the x-values used for plotting, weighting, and fitting, but
        does not change which pairs fall into each bin.
    correlation_type : {'pearsonr', 'uncentered pearsonr', 'spearman'}
        Correlation estimator used within each lag bin.
    model_type : str
        Name of the correlation kernel to fit.
    weight_fn : str or None
        Weighting scheme used in the weighted least-squares fit.
    weight_params : dict, list, or None
        Parameters used by the chosen weighting scheme.
    max_lagfit_factor : float, default 2
        Upper cap for range-like model parameters during fitting.
    fix_alpha : bool, default True
        If True, fix `alpha=1` and fit a model without a nugget jump. If False,
        estimate `alpha`.
    plot : bool, default False
        If True, plot the empirical correlogram and the fitted model.
    bootstrap : int or None, default None
        Number of bootstrap replicates. If None, no bootstrap is run.
    bootstrap_method : {'pair', 'point'}, default 'pair'
        Resampling scheme used for the bootstrap.
    bootstrap_ci : tuple(float, float), default (2.5, 97.5)
        Percentile interval used for the bootstrap confidence band.
    random_state : int or None, default None
        Seed for the random number generator used in the bootstrap.

    Returns
    -------
    h_lag : ndarray
        Representative lag values of the retained bins, according to `lag_repr`.
    n_obs : ndarray
        Number of directional pairs in each retained bin.
    rho : ndarray
        Empirical correlation in each retained bin.
    params : dict
        Fitted model parameters. Bootstrap results are added here when requested.
    r2_wls : float
        Weighted R² from the fitted model.
    r2_ols : float
        Ordinary R² from the fitted model.

    Notes
    -----
    The fitted model always satisfies `rho(0)=1`. When `alpha < 1`, the model
    includes a nugget-style jump between zero lag and the positive-lag branch.

    For `distance_type='angular'`, the supplied angular coordinates are assumed to be in degrees.
    """

    values = np.asarray(values, float)
    coords = np.asarray(coordinates, float)
    n = len(values)

    if n < 2:
        raise ValueError("Need at least 2 points to compute a correlogram.")

    nmax = int(np.ceil(float(max_distance) / float(bin_size)))
    if nmax <= 0:
        raise ValueError("max_distance / bin_size must be > 0.")

    # correlation estimator
    if correlation_type == "uncentered pearsonr":
        correlation_fn = pearsonr_uncen
    elif correlation_type == "pearsonr":
        correlation_fn = pearsonr_cen
    elif correlation_type == "spearman":
        correlation_fn = spearmanr_bin
    else:
        raise ValueError(
            "Invalid estimator: choose from 'uncentered pearsonr', 'pearsonr', or 'spearman'"
        )

    # correlation model
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
        raise ValueError(
            "Invalid Model: Choose from 'exponential', 'cubic', 'powered_exponential', "
            "'matern', 'spherical', 'gaussian', 'damped_cosine_angle' or "
            "'angular_dissimilarity'"
        )

    # -------------------------------------------------
    # main pair list and empirical correlogram
    # -------------------------------------------------
    d, x, y = _build_correlation_pair_arrays(values, coords, distance_type)

    h_full, n_full, rho_full, h_lag, n_obs, rho = _binned_correlogram_from_pairs(
        d, x, y, bin_size, max_distance, correlation_fn, lag_repr=lag_repr
    )

    if h_lag.size == 0:
        raise ValueError(
            "No valid lag bins with a defined empirical correlation were found. "
            "Check the data, bin_size, max_distance, and minimum usable pairs per bin."
        )
    # -------------------------------------------------
    # main fit
    # -------------------------------------------------
    h = h_lag.ravel()
    g = rho.ravel()
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

    x0, bounds = make_init_and_bounds(
        model_type, h, g, xmax_factor=max_lagfit_factor, fix_alpha=fix_alpha
    )

    res = minimize(
        fun=lambda th: objective_func(th, h, g, weights, correlationmodel_fn),
        x0=x0,
        bounds=bounds,
    )
    if not res.success:
        raise RuntimeError(f"Correlation-model optimization failed: {res.message}")
    theta_hat = np.asarray(res.x, float)

    # Enforce fixed parameters exactly after optimization
    if fix_alpha:
        theta_hat[-1] = 1.0

    g_fit_bins = correlationmodel_fn(h, *theta_hat)
    r2_wls = r2_score_weighted(g, g_fit_bins, w=weights)
    r2_ols = r2_score_weighted(g, g_fit_bins, w=None)

    params = pack_params(model_type, theta_hat)
    params.pop("bootstrap", None)
    param_keys = list(params.keys())

    xlag_fit = np.linspace(0.0, float(np.max(h_lag[:, 0]) + bin_size / 2.0), 1000)
    rho_pred = correlationmodel_fn(xlag_fit, *theta_hat)

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

        rho_boot = np.full((n_boot, h_full.shape[0]), np.nan, dtype=float)
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
                    x_b = x[draw]
                    y_b = y[draw]

                elif method == "point":
                    point_draw = rng.integers(0, n, size=n)
                    values_b = values[point_draw]
                    coords_b = coords[point_draw]

                    d_b, x_b, y_b = _build_correlation_pair_arrays(
                        values_b, coords_b, distance_type
                    )

                # --------------------------
                # empirical correlogram
                # --------------------------
                h_full_b, n_full_b, rho_full_b, h_lag_b, n_obs_b, rho_b = _binned_correlogram_from_pairs(
                    d_b, x_b, y_b, bin_size, max_distance, correlation_fn, lag_repr=lag_repr
                )

                rho_boot[b_ix, :] = rho_full_b.ravel()

                # --------------------------
                # fit model
                # --------------------------
                h_b = h_lag_b.ravel()
                g_b = rho_b.ravel()
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

                x0_b, bounds_b = make_init_and_bounds(
                    model_type, h_b, g_b, xmax_factor=max_lagfit_factor, fix_alpha=fix_alpha
                )

                res_b = minimize(
                    fun=lambda th: objective_func(th, h_b, g_b, weights_b, correlationmodel_fn),
                    x0=x0_b,
                    bounds=bounds_b,
                )
                if not res_b.success:
                    continue
                theta_b = np.asarray(res_b.x, float)
                if fix_alpha:
                    theta_b[-1] = 1.0

                model_boot[b_ix, :] = correlationmodel_fn(xlag_fit, *theta_b)

                g_fit_b = correlationmodel_fn(h_b, *theta_b)
                r2_wls_boot[b_ix] = r2_score_weighted(g_b, g_fit_b, w=weights_b)
                r2_ols_boot[b_ix] = r2_score_weighted(g_b, g_fit_b, w=None)

                p_b = pack_params(model_type, theta_b)
                for k in param_keys:
                    param_boot[k][b_ix] = p_b[k]

            except Exception:
                continue

        rho_mean, rho_q05, rho_q95, n_rho_success = _bootstrap_summary(rho_boot, qlo, qhi)
        model_mean, model_q05, model_q95, n_model_success = _bootstrap_summary(model_boot, qlo, qhi)

        boot = {
            "n_bootstrap": n_boot,
            "method": method,
            "ci": (qlo, qhi),
            "random_state": random_state,
            "successful_rho_bootstrap": n_rho_success,
            "successful_model_bootstrap": n_model_success,
            "h_lag_full": h_full[:, 0].copy(),
            "n_obs_full": n_full[:, 0].copy(),
            "rho_samples": rho_boot,
            "rho_mean": rho_mean,
            "rho_q05": rho_q05,
            "rho_q95": rho_q95,
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
            h_lag[:, 0], n_obs[:, 0],
            edgecolor="black", align="center", width=bin_size / 2.0
        )
        ax0.grid(which="minor")
        yt = ax0.get_yticks()
        if yt.size > 1:
            ax0.set_yticks(yt[1:])

        ax1 = plt.subplot(gs_plot[1], sharex=ax0)

        # experimental points
        ax1.plot(
            h_lag[:, 0], rho[:, 0],
            "o", markeredgecolor="black", color="tab:blue",
            label="Experimental", zorder=5
        )

        # bootstrap fitted curves + mean + CI
        if boot is not None:
            first = True
            alpha_samples = None

            if "param_samples" in boot:
                if "alpha" in boot["param_samples"]:
                    alpha_samples = np.asarray(boot["param_samples"]["alpha"], float)
                elif ("c0" in boot["param_samples"]) and ("b" in boot["param_samples"]):
                    c0_s = np.asarray(boot["param_samples"]["c0"], float)
                    b_s = np.asarray(boot["param_samples"]["b"], float)
                    s2_s = c0_s + b_s
                    alpha_samples = np.where(s2_s > 0.0, c0_s / s2_s, np.nan)

            for i, yb in enumerate(boot["model_samples"]):
                if not np.any(np.isfinite(yb)):
                    continue

                alpha_i = 1.0
                if alpha_samples is not None and i < len(alpha_samples) and np.isfinite(alpha_samples[i]):
                    alpha_i = float(alpha_samples[i])

                _plot_correlation_model_piecewise(
                    ax1,
                    xlag_fit,
                    yb,
                    params={"alpha": alpha_i},
                    color="0.7",
                    lw=0.8,
                    ls="-",
                    label="Bootstrap samples" if first else None,
                    zorder=1,
                    alpha_plot=0.20,
                    show_zero_point=False,
                )

                first = False

            alpha_band = 1.0
            if alpha_samples is not None and np.any(np.isfinite(alpha_samples)):
                alpha_band = float(np.nanmean(alpha_samples))

            has_nugget_band = np.isfinite(alpha_band) and (alpha_band < (1.0 - 1e-12))

            if np.any(np.isfinite(boot["model_q05"])) and np.any(np.isfinite(boot["model_q95"])):
                if has_nugget_band:
                    pos = xlag_fit > 0
                    ax1.fill_between(
                        xlag_fit[pos],
                        boot["model_q05"][pos],
                        boot["model_q95"][pos],
                        color="tab:orange",
                        alpha=0.20,
                        label=f"Bootstrap {boot['ci'][0]:g}-{boot['ci'][1]:g}% CI",
                        zorder=2
                    )
                else:
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
                boot_mean_params = {"alpha": float(alpha_band)} if np.isfinite(alpha_band) else {"alpha": 1.0}

                _plot_correlation_model_piecewise(
                    ax1,
                    xlag_fit,
                    boot["model_mean"],
                    params=boot_mean_params,
                    color="tab:orange",
                    lw=2.0,
                    ls="-",
                    label="Bootstrap mean",
                    zorder=3
                )


        # main fitted curve
        _plot_correlation_model_piecewise(
            ax1,
            xlag_fit,
            rho_pred,
            params=params,
            color="k",
            lw=2.0,
            ls="-",
            label=r"Model, $R^2$ (WLS|OLS) = %.2f|%.2f" % (r2_wls, r2_ols),
            zorder=4
        )

        ax1.axhline(0.0, color="k", lw=1.0, ls="--", alpha=0.7)
        plt.setp(ax0.get_xticklabels(), visible=False)

        yticks = ax1.yaxis.get_major_ticks()
        if yticks:
            yticks[-1].label1.set_visible(True)

        ax0.set_ylabel("Number of Lags, N", labelpad=22)
        ax0.set_ylim(0, max(n_obs[:, 0]))

        ax1.legend(loc="lower left")

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
        ax1.set_yticks([-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])

        ax1.set_xlim(0, float(np.max(h_lag[:, 0]) + bin_size / 2.0))
        ax1.set_ylim(-1, 1)
        ax1.set_ylabel(r"Correlation Coefficient, $\rho$ (%s)" % correlation_type)
        ax1.set_xlabel("lag distance")
        plt.subplots_adjust(hspace=0.0)
        plt.show()

    return h_lag, n_obs, rho, params, r2_wls, r2_ols

# Main Function: correfitmulti
def correfitmulti(df, values_col, index_col, coord_cols, distance_type,
                  max_distance, bin_size, correlation_type, model_type,
                  weight_fn=None, weight_params=None, max_lagfit_factor=2.0,
                  fix_alpha=True, plot_single=False, plot_summary=False,
                  lag_repr="center"):
    """
    Fit a correlogram separately for each group in a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing values, group IDs, and coordinates.
    values_col : str
        Column containing the values to correlate.
    index_col : str
        Column used to define independent groups.
    coord_cols : list or tuple
        Coordinate columns used to compute lag distances.
    distance_type : {'geographic', 'euclidean', 'cartesian', 'angular'}
        Distance metric used to form lag bins.
    max_distance : float
        Maximum lag distance included in each fit.
    bin_size : float
        Width of each lag bin.
    correlation_type : {'pearsonr', 'uncentered pearsonr', 'spearman'}
        Correlation estimator used within each lag bin.
    model_type : str
        Name of the correlation kernel to fit.
    weight_fn : str or None, optional
        Weighting scheme used in the fit.
    weight_params : dict or list, optional
        Parameters used by the weighting scheme.
    max_lagfit_factor : float, default 2.0
        Upper cap for range-like model parameters.
    fix_alpha : bool, default True
        If True, fit without a nugget jump. If False, estimate `alpha`.
    plot_single : bool, default False
        If True, plot each group fit as it is computed.
    plot_summary : bool, default False
        If True, plot all group fits together with mean/median summary curves.
    lag_repr : {'center', 'edge', 'upper'}, default 'center'
        Representative lag attached to each equal-width bin. This controls the
        x-axis used in the returned wide lag tables and in the per-group fitting,
        but does not change bin membership.

    Returns
    -------
    summary : pandas.DataFrame
        One row per group with fitted parameters and fit statistics.
    df_n_obs : pandas.DataFrame
        Pair counts by representative lag value and group.
    df_rho : pandas.DataFrame
        Empirical correlations by representative lag value and group.
    results : dict
        Raw fit results for each group.

    Notes
    -----
    This is a grouped wrapper around `correfit`.

    For `distance_type='angular'`, the supplied angular coordinates are assumed to be in degrees.
    """

    results = {}
    summary_rows = []

    # full list of representative lag values (global index for wide frames)
    nmax = int(np.ceil(float(max_distance) / float(bin_size)))
    full_h = _make_lag_axis(nmax, bin_size, lag_repr=lag_repr)

    # guard against fp drift when matching later
    full_h = np.round(full_h, 12)

    # initialize wide frames with the global bin axis
    df_n_obs  = pd.DataFrame({"h_lag": full_h})
    df_rho  = pd.DataFrame({"h_lag": full_h})
    param_keys = None

    df = df.copy()
    df[index_col] = df[index_col].astype(str).str.strip()
    gb = df.groupby(index_col, sort=False, observed=False)

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

        elif distance_type == 'cartesian':
            x = gdf[coord_cols[0]].to_numpy(dtype=float)
            y = gdf[coord_cols[1]].to_numpy(dtype=float)
            coords = np.column_stack([x, y])

        elif distance_type == 'euclidean':
            coords = gdf[list(coord_cols)].to_numpy(dtype=float)

        elif distance_type == 'angular':
            if isinstance(coord_cols, str):
                coords = gdf[coord_cols].to_numpy(dtype=float)
            elif isinstance(coord_cols, (list, tuple)) and len(coord_cols) == 1:
                coords = gdf[coord_cols[0]].to_numpy(dtype=float)
            else:
                raise ValueError("For angular distance_type, provide a single column in coord_cols.")
        else:
            raise ValueError("distance_type must be 'geographic', 'cartesian', 'euclidean', or 'angular'")

        # call correfit main function
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
            lag_repr=lag_repr,
        )

        h_lag, n_obs, rho, params, r2_wls, r2_ols = res

        # store per-group result
        results[gid] = res

        # lock param column order from the first group
        if param_keys is None:
            param_keys = [k for k, v in params.items() if np.isscalar(v)]

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
            c=M['n_obs'].to_numpy(),
            s=12,
            cmap=plt.get_cmap("coolwarm"),
            norm=mpl.colors.LogNorm(vmin=1),
            alpha=0.8,
            edgecolor='k',
            linewidths=0.1,
            zorder = 1
        )

        # correlation model fit plots
        x_plot_max = float(np.max(full_h) + bin_size / 2.0)
        xlag_fit = np.linspace(0.0, x_plot_max, 1000)
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
        alpha_mean = float(np.nanmean(summary["alpha"]))
        alpha_median = float(np.nanmedian(summary["alpha"]))

        has_nugget_band = np.isfinite(alpha_mean) and (alpha_mean < (1.0 - 1e-12))

        if has_nugget_band:
            pos = xlag_fit > 0
            ax.fill_between(
                xlag_fit[pos], y_5[pos], y_95[pos],
                color='forestgreen', alpha=0.15, label='5–95% CI fit', zorder=0
            )
        else:
            ax.fill_between(
                xlag_fit, y_5, y_95,
                color='forestgreen', alpha=0.15, label='5–95% CI fit', zorder=0
            )

        mean_params = {"alpha": alpha_mean}
        median_params = {"alpha": alpha_median}

        _plot_correlation_model_piecewise(
            ax, xlag_fit, y_mean, mean_params,
            color='k', lw=1.5, ls='-', label='mean fit', zorder=1000
        )
        _plot_correlation_model_piecewise(
            ax, xlag_fit, y_median, median_params,
            color='k', lw=1.5, ls='--', label='median fit', zorder=1000
        )

        for gid, (h_lag_g, n_obs_g, rho_g, params_g, r2_wls_g, r2_ols_g) in results.items():
            theta_g = theta_from_params(params_g, model_type)
            y_fit_g = fn(xlag_fit, *theta_g)
            _plot_correlation_model_piecewise(
                ax, xlag_fit, y_fit_g, params_g,
                color='k', lw=0.2, ls='-', zorder=2, alpha_plot=0.8,
                show_zero_point=False
            )

        cb = plt.colorbar(sc, ax=ax, pad =0.02, fraction=0.04, aspect=40)
        cb.set_label('Observations per bin, n')
        ax.legend(loc='lower left', frameon=False)
        ax.set_xlabel("lag distance")
        ax.set_ylabel(r'Correlation Coefficient, $\rho$ (%s)' % correlation_type)
        ax.set_xlim(0,x_plot_max)
        ax.set_ylim(-1, 1)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.show()

    return summary, df_n_obs, df_rho, results

# Main function: single cross-correlation fit
def _build_crosscorrelation_pair_arrays(values1, values2, coordinates, distance_type):
    """
    Build directional cross-correlation pair arrays:
        d = pair distances
        x = values1 at site i
        y = values2 at site j

    Uses all off-diagonal directional pairs (i,j), i != j.
    """
    values1 = np.asarray(values1, float)
    values2 = np.asarray(values2, float)
    coords = np.asarray(coordinates, float)

    if values1.shape != values2.shape:
        raise ValueError("values1 and values2 must have the same shape")

    n = len(values1)
    if n < 2:
        raise ValueError("Need at least 2 points to compute a cross-correlogram.")

    dt = str(distance_type).lower()

    if dt == "geographic":
        lat = coords[:, 0]
        lon = coords[:, 1]
        distance = np.asarray(
            haversine_oq(lon, lat, lon, lat, radians=False, earth_rad=6371.227),
            dtype=float
        )

    elif dt == "cartesian":
        if coords.shape[1] != 2:
            raise ValueError("cartesian requires coordinates shape (n,2): (x, y)")
        dx = coords[:, None, 0] - coords[None, :, 0]
        dy = coords[:, None, 1] - coords[None, :, 1]
        distance = np.hypot(dx, dy)

    elif dt == "euclidean":
        diff = coords[:, None, :] - coords[None, :, :]
        distance = np.linalg.norm(diff, axis=-1)

    elif dt == "angular":
        theta_deg = np.asarray(coords, float)
        if theta_deg.ndim == 2:
            if theta_deg.shape[1] != 1:
                raise ValueError("angular distance requires a single angular coordinate per row.")
        theta = np.radians(theta_deg.ravel())
        cos_diff = np.cos(theta[:, None] - theta[None, :])
        ang_rad = np.arccos(np.clip(cos_diff, -1.0, 1.0))
        distance = np.degrees(ang_rad)

    else:
        raise ValueError(
            "Invalid distance_type: choose 'geographic', 'cartesian', 'angular', or 'euclidean'"
        )

    i_idx, j_idx = np.triu_indices(n, k=1)

    d = distance[i_idx, j_idx]
    x = values1[i_idx]
    y = values2[j_idx]

    return d, x, y

def crosscorrefit(values1, values2, coordinates, distance_type, max_distance,
                  bin_size, correlation_type, model_type, weight_fn=None,
                  weight_params=None, max_lagfit_factor=2.0,
                  fix_alpha=True, plot=False, pair_geometry=None, lag_repr="center"):
    """
    Estimate an empirical cross-correlogram and fit the same kernel form used in `correfit`.

    Parameters
    ----------
    values1, values2 : array-like of float
        Two variables observed at the same locations.
    coordinates : array-like
        Coordinates of the sample locations.
    distance_type : {'geographic', 'euclidean', 'cartesian', 'angular'}
        Distance metric used to form lag bins.
    max_distance : float
        Maximum lag distance included in the fit.
    bin_size : float
        Width of each lag bin.
    correlation_type : {'pearsonr', 'uncentered pearsonr', 'spearman'}
        Correlation estimator used within each lag bin.
    model_type : str
        Name of the correlation kernel to fit.
    weight_fn : str or None, optional
        Weighting scheme used in the fit.
    weight_params : dict or list, optional
        Parameters used by the weighting scheme.
    max_lagfit_factor : float, default 2.0
        Upper cap for range-like model parameters.
    fix_alpha : bool, default True
        If True, fit without a nugget jump. If False, estimate `alpha`.
    plot : bool, default False
        If True, plot the empirical cross-correlogram and fitted model.
    pair_geometry : dict or None, optional
        Precomputed pair geometry returned by
        `_prepare_crosscorrelation_geometry`. If provided, the function reuses
        the same pair indices and lag-bin assignments instead of recomputing
        them from `coordinates`. This is mainly useful inside
        `multicrosscorrefit()` when many variable pairs share the same sites
        and distance settings.
    lag_repr : {'center', 'edge', 'upper'}, default 'center'
        Representative lag attached to each equal-width bin:
        - 'center': midpoint of [k*bin_size, (k+1)*bin_size)
        - 'edge'/'upper': upper edge of [k*bin_size, (k+1)*bin_size)

        This affects the x-values used for plotting, weighting, and fitting, but
        does not change which pairs fall into each bin.
    Returns
    -------
    h_lag : ndarray
        Representative lag values of the retained bins, according to `lag_repr`.
    n_obs : ndarray
        Number of directional pairs in each retained bin.
    rho : ndarray
        Empirical cross-correlation in each retained bin.
    params : dict
        Fitted model parameters in the same format returned by `correfit`.
    r2_wls : float
        Weighted R² from the fitted model.
    r2_ols : float
        Ordinary R² from the fitted model.

    Notes
    -----
    This fits the spatial shape of the cross-correlogram only. It does not
    introduce a separate same-site cross-correlation amplitude.

    For `distance_type='angular'`, the supplied angular coordinates are assumed to be in degrees.
    """

    values1 = np.asarray(values1, float)
    values2 = np.asarray(values2, float)
    coords = np.asarray(coordinates, float)

    if values1.shape != values2.shape:
        raise ValueError("values1 and values2 must have the same shape")

    if model_type not in CORRELATION_MODELS:
        raise ValueError(f"Invalid model_type: {model_type}")

    # estimator
    if correlation_type == "uncentered pearsonr":
        correlation_fn = pearsonr_uncen
    elif correlation_type == "pearsonr":
        correlation_fn = pearsonr_cen
    elif correlation_type == "spearman":
        correlation_fn = spearmanr_bin
    else:
        raise ValueError("Invalid estimator: choose from 'uncentered pearsonr', 'pearsonr', or 'spearman'")

    model_fn = CORRELATION_MODELS[model_type]

    # empirical cross-correlogram
    if pair_geometry is None:
        d, x, y = _build_crosscorrelation_pair_arrays(values1, values2, coords, distance_type)
        h_full, n_full, rho_full, h_lag, n_obs, rho = _binned_correlogram_from_pairs(
            d, x, y, bin_size, max_distance, correlation_fn, lag_repr=lag_repr
        )
    else:
        expected_h = _make_lag_axis(
            int(np.ceil(float(max_distance) / float(bin_size))),
            bin_size,
            lag_repr=lag_repr,
        )
        h_pair = np.asarray(pair_geometry["h_full"], float).ravel()

        if h_pair.shape != expected_h.shape or not np.allclose(h_pair, expected_h):
            raise ValueError(
                "pair_geometry is inconsistent with the requested lag_repr/bin grid. "
                "Rebuild pair_geometry with the same max_distance, bin_size, and "
                "lag_repr passed to crosscorrefit()."
            )

        h_full, n_full, rho_full, h_lag, n_obs, rho = _binned_correlogram_from_precomputed_pairs(
            values1, values2, pair_geometry, correlation_fn
        )

    if h_lag.size == 0:
        raise ValueError(
            "No valid lag bins with a defined empirical correlation were found. "
            "Check the data, bin_size, max_distance, and minimum usable pairs per bin."
        )

    h = h_lag.ravel()
    g = rho.ravel()
    m = n_obs.ravel()

    if weight_fn is None or str(weight_fn).lower() == "ols":
        weights = np.ones_like(h, dtype=float)
    else:
        if weight_params is None:
            weight_params_fit = [0.25 * float(h.max()) if h.size else 1.0, 1.0]
        elif isinstance(weight_params, dict):
            bpar = weight_params.get("b", 0.25 * float(h.max()) if h.size else 1.0)
            apar = weight_params.get("alpha", 1.0)
            weight_params_fit = [bpar, apar]
        else:
            weight_params_fit = weight_params

        weights = compute_distance_weights(
            h, m, weight_type=weight_fn, weight_params=weight_params_fit
        )

    x0, bounds = make_init_and_bounds(
        model_type, h, g, xmax_factor=max_lagfit_factor, fix_alpha=fix_alpha
    )

    res = minimize(
        fun=lambda th: objective_func(th, h, g, weights, model_fn),
        x0=x0,
        bounds=bounds,
    )
    if not res.success:
        raise RuntimeError(f"Cross-correlation-model optimization failed: {res.message}")
    theta_hat = np.asarray(res.x, float)

    if fix_alpha:
        theta_hat[-1] = 1.0

    g_fit_bins = model_fn(h, *theta_hat)
    r2_wls = r2_score_weighted(g, g_fit_bins, w=weights)
    r2_ols = r2_score_weighted(g, g_fit_bins, w=None)

    params = pack_params(model_type, theta_hat)

    xlag_fit = np.linspace(0.0, float(np.max(h_lag[:, 0]) + bin_size / 2.0), 1000)
    rho_pred = model_fn(xlag_fit, *theta_hat)

    if plot:
        fig = plt.figure(figsize=(12, 7), dpi=200)
        gs_plot = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

        ax0 = plt.subplot(gs_plot[0])
        ax0.bar(
            h_lag[:, 0], n_obs[:, 0],
            edgecolor="black", align="center", width=bin_size / 2.0
        )
        ax0.grid(which="minor")
        yt = ax0.get_yticks()
        if yt.size > 1:
            ax0.set_yticks(yt[1:])

        ax1 = plt.subplot(gs_plot[1], sharex=ax0)

        ax1.plot(
            h_lag[:, 0], rho[:, 0],
            "o", markeredgecolor="black", color="tab:blue",
            label="Cross experimental", zorder=5
        )

        _plot_correlation_model_piecewise(
            ax1,
            xlag_fit,
            rho_pred,
            params=params,
            color="k",
            lw=2.0,
            ls="-",
            label=r"Model, $R^2$ (WLS|OLS) = %.2f|%.2f" % (r2_wls, r2_ols),
            zorder=4,
            show_zero_point=True
        )

        ax1.axhline(0.0, color="k", lw=1.0, ls="--", alpha=0.7)
        plt.setp(ax0.get_xticklabels(), visible=False)

        ax0.set_ylabel("Number of Lags, N", labelpad=22)
        ax0.set_ylim(0, max(n_obs[:, 0]) if n_obs.size else 1)

        ax1.legend(loc="lower left")
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
        ax1.set_yticks([-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])

        ax1.set_xlim(0, float(np.max(h_lag[:, 0]) + bin_size / 2.0))
        ax1.set_ylim(-1, 1)
        ax1.set_ylabel(r"Cross-correlation, $\rho_{12}$ (%s)" % correlation_type)
        ax1.set_xlabel("lag distance")
        plt.subplots_adjust(hspace=0.0)
        plt.show()

    return h_lag, n_obs, rho, params, r2_wls, r2_ols

# main function: multicross-fit
def multicrosscorrefit(df, values_cols, coord_cols, distance_type, max_distance,
                       bin_size, correlation_type, model_type, weight_fn=None,
                       weight_params=None, max_lagfit_factor=2.0,
                       fix_alpha=True, plot_single=False, plot_matrix=False,
                       lag_repr="center"):
    """
    Fit cross-correlograms for all variable pairs in a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the variables and coordinates.
    values_cols : list of str
        Columns to be paired and fitted.
    coord_cols : list, tuple, or str
        Coordinate columns used to compute lag distances.
    distance_type : {'geographic', 'euclidean', 'cartesian', 'angular'}
        Distance metric used to form lag bins.
    max_distance : float
        Maximum lag distance included in each fit.
    bin_size : float
        Width of each lag bin.
    correlation_type : {'pearsonr', 'uncentered pearsonr', 'spearman'}
        Correlation estimator used within each lag bin.
    model_type : str
        Name of the correlation kernel to fit.
    weight_fn : str or None, optional
        Weighting scheme used in the fit.
    weight_params : dict or list, optional
        Parameters used by the weighting scheme.
    max_lagfit_factor : float, default 2.0
        Upper cap for range-like model parameters.
    fix_alpha : bool, default True
        If True, fit without a nugget jump. If False, estimate `alpha`.
    plot_single : bool, default False
        If True, plot each pairwise fit during the loop.
    plot_matrix : bool, default False
        If True, plot the lower-triangular matrix of pairwise fits.
    lag_repr : {'center', 'edge', 'upper'}, default 'center'
        Representative lag attached to each equal-width bin for both the auto-
        and cross-correlation fits. This affects plotting, weighting, and fitting
        x-values, but does not change bin membership.

    Returns
    -------
    summary : pandas.DataFrame
        Pairwise fit summary table.
    results : dict
        Raw fit results for each variable pair.
    param_mats : dict[str, pandas.DataFrame]
        Parameter matrices for each fitted parameter.
    r2_mats : dict[str, pandas.DataFrame]
        Matrices of weighted and ordinary R² values.

    Notes
    -----
    This is the pairwise wrapper around `crosscorrefit`.

    For `distance_type='angular'`, the supplied angular coordinates are assumed to be in degrees.
    """

    df = df.copy()
    values_cols = list(values_cols)

    missing_vals = [c for c in values_cols if c not in df.columns]
    if missing_vals:
        raise ValueError(f"Missing value columns in df: {missing_vals}")

    if model_type not in CORRELATION_MODELS:
        raise ValueError(f"Invalid model_type: {model_type}")

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
    else:
        raise ValueError("distance_type must be 'geographic', 'cartesian', 'euclidean', or 'angular'")

    pair_geometry = _prepare_crosscorrelation_geometry(
        coords, distance_type, bin_size, max_distance, lag_repr=lag_repr
    )

    results = {}
    summary_rows = []
    param_keys = None

    p = len(values_cols)
    n_samples = int(len(df))

    for i, vi in enumerate(values_cols):
        vals_i = df[vi].to_numpy(dtype=float)

        for j in range(i, p):
            vj = values_cols[j]
            vals_j = df[vj].to_numpy(dtype=float)

            res = crosscorrefit(
                values1=vals_i,
                values2=vals_j,
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
                pair_geometry=pair_geometry,
                lag_repr=lag_repr,
            )

            h_lag, n_obs, rho, params, r2_wls, r2_ols = res

            results[(vi, vj)] = res
            results[(vj, vi)] = res

            if param_keys is None:
                param_keys = [k for k, v in params.items() if np.isscalar(v)]

            summary_rows.append({
                "var_i": vi,
                "var_j": vj,
                "type": "auto" if i == j else "cross",
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

    r2_wls_mat = pd.DataFrame(np.nan, index=idx, columns=idx, dtype=float)
    r2_ols_mat = pd.DataFrame(np.nan, index=idx, columns=idx, dtype=float)
    for a in idx:
        for b in idx:
            if (a, b) in results:
                _, _, _, _, r2w, r2o = results[(a, b)]
                r2_wls_mat.loc[a, b] = float(r2w)
                r2_ols_mat.loc[a, b] = float(r2o)

    r2_mats = {"r2_wls": r2_wls_mat, "r2_ols": r2_ols_mat}

    if plot_matrix:
        fn = CORRELATION_MODELS[model_type]
        nmax = int(np.ceil(float(max_distance) / float(bin_size)))
        x_plot_max = float(np.max(_make_lag_axis(nmax, bin_size, lag_repr=lag_repr)) + bin_size / 2.0)
        xfit = np.linspace(0.0, x_plot_max, 600)

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

                h_lag, n_obs, rho_ij, params_ij, r2w, r2o = res
                h = h_lag.ravel()
                g = rho_ij.ravel()

                theta = theta_from_params(params_ij, model_type)
                gfit = fn(xfit, *theta)

                ax.plot(h, g, 'o', ms=2.5, markeredgecolor='black')

                _plot_correlation_model_piecewise(
                    ax, xfit, gfit, params_ij,
                    color='k', lw=0.8, ls='-', zorder=2,
                    show_zero_point=True
                )

                ax.set_title(f"{vi} × {vj}" if i != j else vi, fontsize=8)
                ax.set_xlim(0, x_plot_max)
                ax.set_ylim(-1, 1)
                ax.grid(True, linestyle='--', alpha=0.2)

                if i == p - 1:
                    ax.set_xlabel("lag", fontsize=8)
                if j == 0:
                    ax.set_ylabel("ρ", fontsize=8)

        plt.tight_layout()
        plt.show()

    return summary, results, param_mats, r2_mats
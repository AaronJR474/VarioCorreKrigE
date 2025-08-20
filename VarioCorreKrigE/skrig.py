"""
This file contains the functions required for simple kriging utilizing user specified variograms params or derived from
variofit or correfit.
"""

# import modules
import numpy as np
from numpy.random import default_rng
from scipy.linalg import cho_factor, cho_solve
from pyproj import Geod
from typing import Callable, Dict, Optional, Sequence, Tuple, Union, Literal
from tqdm.auto import tqdm

# from package
from VarioCorreKrigE.variofit import VARIOGRAM_MODELS
from VarioCorreKrigE.correfit import CORRELATION_MODELS

# geographic pairwise distance and pairwise distance functions
def geodesic_pairwise(X_src, X_dst, *, ellps="WGS84", return_az=False, chunk_cols=None):
    """
    Great-circle pairwise distances between two sets of [lat, lon] points (deg).
    Returns distances in km with the correct shapes:
      - X_src: (n,2), X_dst: (m,2)  ->  D: (n,m)

    Parameters
    ----------
    X_src : ndarray, shape (n, 2)   columns [lat, lon] in degrees
    X_dst : ndarray, shape (m, 2)   columns [lat, lon] in degrees
    ellps : str                     pyproj ellipsoid name (e.g. 'WGS84')
    return_az : bool                if True, also return (fwd_az, back_az) arrays
    chunk_cols : int or None        process destination columns in chunks to reduce memory

    Returns
    -------
    D_km : ndarray, shape (n, m)    distances in kilometers
    az   : tuple or None            (fwd_az, back_az) if return_az=True, else None
    """
    X_src = np.asarray(X_src, float)
    X_dst = np.asarray(X_dst, float)
    n = X_src.shape[0]
    m = X_dst.shape[0]

    lat1 = X_src[:, 0]
    lon1 = X_src[:, 1]
    lat2 = X_dst[:, 0]
    lon2 = X_dst[:, 1]

    geod = Geod(ellps=ellps)

    # allocate outputs
    D_m = np.empty((n, m), dtype=float)
    if return_az:
        fwd = np.empty((n, m), dtype=float)
        back = np.empty((n, m), dtype=float)

    # optionally chunk by destination columns to control memory
    if chunk_cols is None or chunk_cols <= 0:
        chunk_cols = m  # single block

    for j0 in range(0, m, chunk_cols):
        j1 = min(m, j0 + chunk_cols)
        k = j1 - j0

        # broadcast to (n,k)
        lon1M = np.broadcast_to(lon1[:, None], (n, k))
        lat1M = np.broadcast_to(lat1[:, None], (n, k))
        lon2M = np.broadcast_to(lon2[None, j0:j1], (n, k))
        lat2M = np.broadcast_to(lat2[None, j0:j1], (n, k))

        fwd_az, back_az, dist_m = geod.inv(lon1M, lat1M, lon2M, lat2M)
        D_m[:, j0:j1] = dist_m

        if return_az:
            fwd[:, j0:j1] = fwd_az
            back[:, j0:j1] = back_az

    D_km = D_m / 1000.0
    return (D_km, (fwd, back)) if return_az else (D_km, None)

def pairwise_distances(
    coords: np.ndarray,
    targets: np.ndarray,
    *,
    distance_type: str = "euclidean",
    projection: str = 'WGS84'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build pairwise distance matrices D_nn (n×n) and D_nt (n×m).

    Parameters
    ----------
    coords : (n,d) float array
        Observation locations.
        - 'geographic': columns [lat, lon] in degrees (WGS84 etc.).
        - 'euclidean'/'cartesian': linear coordinates (units arbitrary).
        - 'angular': 1D angles (radians), shape (n,) or (n,1).
    targets : (m,d) float array
        Target locations, matching the format of `coords`.
    distance_type : {'geographic','euclidean','cartesian','angular'}
    projection : str, default 'WGS84'
        Ellipsoid name for `pyproj.Geod` when `distance_type='geographic'`.

    Returns
    -------
    D_nn : (n,n) float array
        Distances among observations.
        - geographic: kilometers
        - euclidean/cartesian: same linear units as input
        - angular: **degrees**
    D_nt : (n,m) float array
        Distances from observations to targets (same units as above).
    """

    X  = np.asarray(coords,  float)
    XT = np.asarray(targets, float)

    if distance_type == "geographic":

        # lat,lon columns in degrees
        D_nn, _ = geodesic_pairwise(X,  X,  ellps=projection, return_az=False)   # (n,n)
        D_nt, _ = geodesic_pairwise(X,  XT, ellps=projection, return_az=False)   # (n,m)

        # sanity
        if D_nt.ndim != 2 or D_nt.shape != (X.shape[0], XT.shape[0]):
            raise ValueError(f"Geographic D_nt has shape {D_nt.shape}, expected {(X.shape[0], XT.shape[0])}")
        return D_nn, D_nt

    elif distance_type == "euclidean":
        diff_nn = X[:, None, :] - X[None, :, :]
        diff_nt = X[:, None, :] - XT[None, :, :]
        return np.linalg.norm(diff_nn, axis=-1), np.linalg.norm(diff_nt, axis=-1)

    elif distance_type == "cartesian":
        # 2D planar for map-projected coords in UTM (x,y)
        x, y   = X[:, 0],  X[:, 1]
        xT, yT = XT[:, 0], XT[:, 1]
        D_nn = np.hypot(x[:, None]-x[None, :], y[:, None]-y[None, :])
        D_nt = np.hypot(x[:, None]-xT[None, :], y[:, None]-yT[None, :])
        return D_nn, D_nt

    elif distance_type == "angular":
        # Angles must be in radians. Accept 1D arrays or single-column 2D arrays.
        if coords.ndim == 2 and coords.shape[1] != 1:
            raise ValueError("For distance_type='angular', coords must be 1D angles (radians) or shape (n,1).")
        if targets.ndim == 2 and targets.shape[1] != 1:
            raise ValueError("For distance_type='angular', targets must be 1D angles (radians) or shape (m,1).")

        theta_obs = np.asarray(coords, float).ravel()   # (n,)
        theta_tar = np.asarray(targets, float).ravel()  # (m,)

        # Cosine of angular differences
        cos_nn = np.cos(theta_obs[:, None] - theta_obs[None, :])  # (n, n)
        cos_nt = np.cos(theta_obs[:, None] - theta_tar[None, :])  # (n, m)

        # Clamp for numerical safety, then arccos -> distances in [0, π]
        D_nn = np.degrees(np.arccos(np.clip(cos_nn, -1.0, 1.0)))
        D_nt = np.degrees(np.arccos(np.clip(cos_nt, -1.0, 1.0)))

        return D_nn, D_nt

    else:
        raise ValueError("distance_type must be 'geographic', 'euclidean', 'angular', or 'cartesian'")

# Build Covariance Matrices from Variogram or Correlation Models
def build_covariance_nn_nt(
    coords: np.ndarray,
    targets: np.ndarray,
    *,
    model_family: str,               # 'variogram' or 'correlation'
    model_type: str,                 # name in your model dict
    theta: Sequence[float],          # parameters in the callable order
    params: Optional[Dict[str, float]] = None,  # for variogram: needs {'c0','b'}
    sigma2: Optional[float] = None,  # total variance if using correlation (else inferred from params)
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    jitter: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build (C_nn, C_nt) from either a variogram or correlation kernel.

    For variogram γ(h) with partial sill c0 and nugget b:
        C(h) = (c0+b) - γ(h);  diag(C) = c0+b.
    For correlation ρ(h):
        C(h) = σ^2 ρ(h);       diag(C) = σ^2  (σ^2 provided or default 1).

    Returns
    -------
    C_nn : (n,n) ndarray
    C_nt : (n,m) ndarray
    sigma2_used : float
        The total variance placed on the diagonal (used later for kriging variance).
    """

    # compute pairwise distances
    D_nn, D_nt = pairwise_distances(coords, targets, distance_type=distance_type, projection = projection)

    # choose model based on inputs
    if model_family == "variogram":
        if model_type not in VARIOGRAM_MODELS:
            raise ValueError(f"Unknown variogram model_type: {model_type}")
        model_fn = VARIOGRAM_MODELS[model_type]
    elif model_family == "correlation":
        if model_type not in CORRELATION_MODELS:
            raise ValueError(f"Unknown correlation model_type: {model_type}")
        model_fn = CORRELATION_MODELS[model_type]
    else:
        raise ValueError("model_family must be 'variogram' or 'correlation'")

    if model_family == "variogram":
        if params is None:
            raise ValueError("For model_family='variogram', provide params with at least {'c0','b'}.")
        c0 = float(params.get("c0", 1.0))
        b  = float(params.get("b",  0.0))
        sigma2_used = c0 + b

        G_nn = model_fn(D_nn, *theta)  # γ_nn
        G_nt = model_fn(D_nt, *theta)  # γ_nt

        C_nn = sigma2_used - G_nn
        C_nt = sigma2_used - G_nt
        np.fill_diagonal(C_nn, sigma2_used)

    elif model_family == "correlation":
        sigma2_used = 1.0 if sigma2 is None else float(sigma2)
        R_nn = model_fn(D_nn, *theta)  # ρ_nn
        R_nt = model_fn(D_nt, *theta)  # ρ_nt
        C_nn = sigma2_used * R_nn
        C_nt = sigma2_used * R_nt
        np.fill_diagonal(C_nn, sigma2_used)

    else:
        raise ValueError("model_family must be 'variogram' or 'correlation'")

    # tiny diagonal jitter to stabilize Cholesky without changing the model form
    C_nn[np.diag_indices_from(C_nn)] += (jitter * sigma2_used)
    return C_nn, C_nt, sigma2_used

# Build Covariance Matrices from Custom Variogram or Correlation Models
def build_covariance_custom_correlation(
    blocks_nn: dict, blocks_nt: dict,
    custom_kernel, theta, sigma2: float = 1.0, jitter: float = 1e-10
):
    """
    From custom distance blocks and a custom correlation kernel, build C_nn, C_nt.

    custom_kernel(blocks_nn, blocks_nt, theta) -> (R_nn, R_nt)
      - blocks_nn: dict of {name: D_nn}, each D_nn is (n,n)
      - blocks_nt: dict of {name: D_nt}, each D_nt is (n,m)
      - theta: parameter vector for the kernel

    Returns
    -------
    C_nn, C_nt, sigma2
    """
    R_nn, R_nt = custom_kernel(blocks_nn, blocks_nt, theta)  # correlation
    C_nn = sigma2 * R_nn
    C_nt = sigma2 * R_nt
    # ensure exact variance on the diagonal + tiny jitter
    n = C_nn.shape[0]
    diag_idx = np.diag_indices(n)
    C_nn[diag_idx] = sigma2
    C_nn[diag_idx] += jitter * sigma2
    return C_nn, C_nt, sigma2

def build_covariance_custom_variogram(
    blocks_nn: dict, blocks_nt: dict,
    custom_kernel, theta, c0: float, b: float, jitter: float = 1e-10
):
    """
    From custom distance blocks and a custom variogram kernel, build C_nn, C_nt.

    custom_kernel(blocks_nn, blocks_nt, theta) -> (G_nn, G_nt)
      - G = γ(h) semivariogram values
    C(h) = (c0 + b) - γ(h)

    Returns
    -------
    C_nn, C_nt, sigma2
    """
    G_nn, G_nt = custom_kernel(blocks_nn, blocks_nt, theta)  # variogram
    sigma2 = float(c0) + float(b)
    C_nn = sigma2 - G_nn
    C_nt = sigma2 - G_nt
    n = C_nn.shape[0]
    diag_idx = np.diag_indices(n)
    C_nn[diag_idx] = sigma2
    C_nn[diag_idx] += jitter * sigma2
    return C_nn, C_nt, sigma2

def merge_blocks(blocks_nn_list, blocks_nt_list):
    """Merge multiple {name: matrix} dicts into single dicts."""
    blocks_nn = {}
    blocks_nt = {}
    for dnn in blocks_nn_list:
        blocks_nn.update(dnn)
    for dnt in blocks_nt_list:
        blocks_nt.update(dnt)
    return blocks_nn, blocks_nt

# Estimation of mean for Simple Kriging
def estimate_mean_gls_from_C(z: np.ndarray, C: np.ndarray) -> float:
    """
    GLS estimate of the global mean:
        m_hat = (1^T C^{-1} z) / (1^T C^{-1} 1)

    Parameters
    ----------
    values : (n,) array
        Observed Z values.
    C_nn : (n,n) ndarray
        Covariance among observations.

    Returns
    -------
    float
        GLS mean.
    """
    z = np.asarray(z, float).ravel()
    n = z.size
    ones = np.ones(n, float)
    cF = cho_factor(C, overwrite_a=False, check_finite=False)
    Ci1 = cho_solve(cF, ones, check_finite=False)
    Ciz = cho_solve(cF, z,   check_finite=False)
    denom = float(ones @ Ci1)
    if not np.isfinite(denom) or denom <= 0:
        return float(np.mean(z))
    return float((ones @ Ciz) / denom)

# Simple Kriging Function: Default
def simple_kriging(
    values: np.ndarray,
    coords: np.ndarray,
    targets: np.ndarray,
    *,
    model_family: str,                 # 'variogram' or 'correlation'
    model_type: str,                   # key in your registries
    theta: Sequence[float],            # parameters in callable order
    params: Optional[Dict[str, float]] = None,  # for variogram: {'c0','b'}
    sigma2: Optional[float] = None,    # for correlation: total variance (default 1.0)
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    mean: Union[str, float] = "gls",   # 'gls' | 'zero' | numeric
    jitter: float = 1e-10,
    return_weights: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Simple Kriging with either a VARIOGRAM or a CORRELATION model.

    Estimator (per target x0):
        w solves C_nn w = C_nt(:,x0)
        z_SK(x0) = m + w^T (z - m·1)
                 = sum_i w_i z_i + (1 - sum_i w_i) m
        σ_SK^2(x0) = σ^2 - w^T C_nt(:,x0)

    Parameters
    ----------
    values : (n,) array
        Observed Z at coords.
    coords : (n,d) array
        Observation locations.
        If distance_type=='geographic', coords[:,0]=lat, coords[:,1]=lon (degrees).
    targets : (m,d) array
        Target locations to estimate.
    model_family : {'variogram','correlation'}
        Choose which family your `model_fn` and `theta` represent.
    model_type : str
        Model name (not used by the solver; useful for logging).
    theta : sequence of float
        Parameters for model_fn in the order it expects.
    params : dict, optional
        If model_family=='variogram', must contain {'c0','b'} for sill = c0+b.
        Ignored for correlation.
    sigma2 : float, optional
        If model_family=='correlation', the total variance placed on the diagonal.
        If None, σ^2=1 is used (appropriate for standardized residuals).
    distance_type : {'euclidean','geographic'}
        Distance type.
    projection : str
        ellipsoid from pyproj
    mean : {'gls','zero'} or float
        - 'gls'  : estimate global mean via GLS under the chosen kernel
        - 'zero' : use 0
        - float  : use this numeric mean
    jitter : float
        Small diagonal stabilization factor (×σ^2).
    return_weights : bool
        If True, also return weights W with shape (m,n).

    Returns
    -------
    est : (m,) ndarray
        SK estimates at targets.
    var : (m,) ndarray
        SK variances at targets.
    W : (m,n) ndarray  (only if return_weights=True)
        Kriging weights for each target.
    """

    z  = np.asarray(values, float).ravel()
    X  = np.asarray(coords, float)
    XT = np.asarray(targets, float)
    n  = z.size

    # 1) Covariance blocks (internally selects model function by family+type)
    C_nn, C_nt, sigma2_used = build_covariance_nn_nt(
        X, XT,
        model_family=model_family,
        model_type=model_type,
        theta=theta,
        params=params,
        sigma2=sigma2,
        distance_type=distance_type,
        projection=projection,
        jitter=jitter
    )

    # 2) Mean m
    if isinstance(mean, (int, float)):
        mval = float(mean)
    elif mean == "zero":
        mval = 0.0
    elif mean == "gls":
        mval = estimate_mean_gls_from_C(z, C_nn)
    else:
        raise ValueError("mean must be 'gls', 'zero', or a numeric value")

    # 3) Solve C_nn W^T = C_nt^T  (multi-RHS with Cholesky)
    cF = cho_factor(C_nn, overwrite_a=False, check_finite=False)
    Wt = cho_solve(cF, C_nt, check_finite=False)  # (n, m)
    W  = Wt.T                                     # (m, n)
    sumw = W.sum(axis=1)                          # (m,)

    # 4) Estimates and variances
    est = W @ z + (1.0 - sumw) * mval
    # σ_K^2(x0) = σ^2 - w^T c, c = C_nt[:, i]
    var = sigma2_used - np.einsum("ij,ij->i", W, C_nt.T)

    return (est, var, W) if return_weights else (est, var)

# Simple Kriging Function: Custom correlation
def simple_kriging_custom_corr(values, blocks_nn, blocks_nt, custom_kernel, theta,
                               sigma2=1.0, mean='gls', jitter=1e-10, return_weights=False):
    """
    Simple Kriging using a CUSTOM CORRELATION kernel.
    Distances are provided as blocks; kernel returns (R_nn, R_nt).
    """
    z = np.asarray(values, float).ravel()
    n = z.size
    # Build covariance from blocks + kernel
    C_nn, C_nt, sigma2_used = build_covariance_custom_correlation(
        blocks_nn, blocks_nt, custom_kernel, theta, sigma2=sigma2, jitter=jitter
    )
    # Mean
    if isinstance(mean, (int,float)):
        mval = float(mean)
    elif mean == 'zero':
        mval = 0.0
    elif mean == 'gls':
        mval = estimate_mean_gls_from_C(z, C_nn)
    else:
        mval = 0.0  # keep it simple
    # Solve for weights
    cF = cho_factor(C_nn, overwrite_a=False, check_finite=False)
    Wt = cho_solve(cF, C_nt, check_finite=False)  # (n,m)
    W  = Wt.T                                     # (m,n)
    sumw = W.sum(axis=1)
    # Est & Var
    est = W @ z + (1.0 - sumw) * mval
    var = sigma2_used - np.einsum("ij,ij->i", W, C_nt.T)
    return (est, var, W) if return_weights else (est, var)

# Simple Kriging Function: Custom variogram
def simple_kriging_custom_vario(values, blocks_nn, blocks_nt, custom_kernel, theta, c0, b,
                                mean='gls', jitter=1e-10, return_weights=False):

    """
    Simple Kriging using a CUSTOM VARIOGRAM kernel.
    Distances are provided as blocks; kernel returns (R_nn, R_nt).
    """

    z = np.asarray(values, float).ravel()
    C_nn, C_nt, sigma2 = build_covariance_custom_variogram(blocks_nn, blocks_nt, custom_kernel, theta, c0, b, jitter)
    mval = estimate_mean_gls_from_C(z, C_nn) if mean == 'gls' else (0.0 if mean=='zero' else float(mean))
    cF  = cho_factor(C_nn, overwrite_a=False, check_finite=False)
    Wt  = cho_solve(cF, C_nt, check_finite=False); W = Wt.T
    est = W @ z + (1 - W.sum(axis=1)) * mval
    var = np.clip(sigma2 - np.einsum("ij,ij->i", W, C_nt.T), 0.0, None)
    return (est, var, W) if return_weights else (est, var)

# Simple Kriging: SGS
# normal score transform from 0 - 1
def normal_score_transform(
    y: np.ndarray, eps: float = 1e-6
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """
    Empirical normal-score (Gaussian anamorphosis) transform with a stable inverse.

    Parameters
    ----------
    y : (n,) array_like
        Raw values to transform.
    eps : float, default 1e-6
        Tail clipping for empirical CDF to avoid +/-inf when applying Phi^{-1}.

    Returns
    -------
    z : (n,) ndarray
        Normal-score values z = Phi^{-1}(F_hat(y)).
    inv : callable
        Inverse mapping inv(znew) -> yhat using a monotone piecewise-linear map.

    Notes
    -----
    - We build an empirical CDF from sorted y and map probs p in (eps, 1-eps) to z via
      z = Phi^{-1}(p). The inverse uses linear interpolation on (z_sorted <-> y_sorted).
    - If y contains duplicates, the transform remains monotone and invertible (ties
      yield flat steps, handled by interpolation).
    """
    y = np.asarray(y, float).ravel()
    if y.size == 0:
        raise ValueError("normal_score_transform: empty input")

    # sort and empirical CDF
    ys = np.sort(y)
    # positions i=1..n -> probs via (i-0.5)/n (Hazen), clipped
    n = ys.size
    p = (np.arange(1, n + 1) - 0.5) / n
    p = np.clip(p, eps, 1.0 - eps)

    # map to z via inverse normal CDF
    from scipy.stats import norm
    zs = norm.ppf(p)

    # forward: y -> z (interpolate on ys->zs)
    z = np.interp(y, ys, zs)

    # inverse: z -> y (interpolate on zs->ys)
    def inv(znew: np.ndarray) -> np.ndarray:
        znew = np.asarray(znew, float)
        return np.interp(znew, zs, ys)

    return z, inv

# Simple Kriging: Sequential Gaussian Simulation
def sgs_simple_kriging(
    values: np.ndarray,
    coords: np.ndarray,
    targets: np.ndarray,
    *,
    model_family: Literal["variogram", "correlation"],
    model_type: str,
    theta: Sequence[float],
    params: Optional[dict] = None,
    sigma2: Optional[float] = None,
    distance_type: Literal["euclidean", "cartesian", "geographic", "angular"] = "euclidean",
    mean: Union[Literal["gls", "zero"], float] = "zero",
    n_realizations: int = 1,
    transform: Literal["ns", "none"] = "ns",
    random_state: Optional[int] = None,
    jitter: float = 1e-10,
    progress: bool = True,
    max_neighbors: Optional[int] = None,   # e.g., 32 or 64; None = use all
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Sequential Gaussian Simulation (SGS) using simple kriging as the local solver.

    Parameters
    ----------
    values : (n,) array_like
        Conditioning data values at `coords`.
    coords : (n,d) array_like
        Conditioning locations.
    targets : (m,d) array_like
        Simulation grid/nodes to simulate sequentially.
    model_family : {'variogram','correlation'}
        Which kernel family you fitted.
    model_type : str
        Kernel/model name (key for your model dictionaries; used inside simple_kriging).
    theta : sequence of float
        Model parameters in the order expected by the chosen kernel.
    params : dict, optional
        If model_family='variogram', must include {'c0','b'} for sill=c0+b.
        Ignored for model_family='correlation'.
    sigma2 : float, optional
        If model_family='correlation', the total variance placed on the diagonal.
        If None, sigma2=1 is used (appropriate when values are standardized).
    distance_type : {'euclidean','cartesian','geographic','angular'}
        Distance metric to use (must match how you fitted the kernel).
    mean : {'gls','zero'} or float, default 'zero'
        Simple-kriging mean:
          - 'gls'  : estimate global mean under current covariance (recommended if not normalized)
          - 'zero' : use 0.0 (typical on normal-score scale)
          - float  : fixed numeric mean
    n_realizations : int, default 1
        Number of independent realizations to produce.
    transform : {'ns','none'}, default 'ns'
        'ns'   : apply normal-score transform to `values`, simulate on N(0,1) scale, then back-transform.
        'none' : simulate directly on the provided scale (ensure your kernel/mean fit that scale).
    random_state : int, optional
        Seed for random path and Gaussian draws.
    jitter : float, default 1e-10
        Optional diagonal stabilization factor multiplied by sigma^2 in `simple_kriging`.

    Returns
    -------
    sims : (n_realizations, m) ndarray
        Simulated values at the `targets` nodes, in the **original** scale if transform='ns'.
    sims_ns : (n_realizations, m) ndarray or None
        The simulated values on normal-score scale (returned only when transform='ns').

    Notes
    -----
    - Complexity is O(m) local SK solves with dynamically growing conditioning set.
      This is accurate but not the fastest; for big grids, consider local neighborhoods
      and/or rank-1 Cholesky updates.
    - `angular` distance type expects angles (radians) for inputs; our internal distance
      builder converts to degrees to match your angular kernels.
    """
    rng = np.random.default_rng(random_state)

    v = np.asarray(values, float).ravel()
    X = np.asarray(coords,  float)
    T = np.asarray(targets, float)
    n, d = X.shape
    m = T.shape[0]

    # normal-score transform (if requested)
    if transform == "ns":
        v_ns, inv_ns = normal_score_transform(v)
        to_sim_scale = inv_ns
        cond_vals0 = v_ns
    elif transform == "none":
        to_sim_scale = lambda z: z
        cond_vals0 = v
    else:
        raise ValueError("transform must be 'ns' or 'none'")

    # pre-allocate outputs
    sims = np.empty((n_realizations, m), float)
    sims_ns = np.empty((n_realizations, m), float) if transform == "ns" else None

    # pre-allocate growing conditioning arrays
    # We keep capacity for n + m points and use a moving length 'ncond'
    cX_buf = np.empty((n + m, d), float)
    cX_buf[:n] = X
    cz_buf = np.empty(n + m, float)
    cz_buf[:n] = cond_vals0

    # progress bars
    try:
        pbar_outer = tqdm(range(n_realizations), disable=not progress, desc="SGS realizations")
    except Exception:
        # tqdm not available
        pbar_outer = range(n_realizations)

    for rix in pbar_outer:
        # fresh random path each realization
        path = rng.permutation(m)

        # reset conditioning head index
        ncond = n
        # (copy initial conditioning values so we don't overwrite previous realization)
        cX_buf[:n] = X
        cz_buf[:n] = cond_vals0
        sim_ns = np.empty(m, float)

        # inner progress
        try:
            inner_iter = tqdm(path, disable=not progress, desc=f"nodes (realization {rix+1}/{n_realizations})", total=m)
        except Exception:
            inner_iter = path

        for j in inner_iter:
            # optional kNN neighbor selection (Euclidean/Cartesian only)
            if (max_neighbors is not None) and (distance_type in ("euclidean", "cartesian")):
                # compute distances from current target to current conditioning set
                diff = cX_buf[:ncond] - T[j]
                dists = np.sqrt(np.einsum("ij,ij->i", diff, diff))
                k = min(max_neighbors, ncond)
                # indices of k nearest
                nn_idx = np.argpartition(dists, k-1)[:k]
                cX_use = cX_buf[nn_idx]
                cz_use = cz_buf[nn_idx]
            else:
                # use all current conditioning points
                cX_use = cX_buf[:ncond]
                cz_use = cz_buf[:ncond]

            # local SK for a single node
            est, var = simple_kriging(
                values=cz_use,
                coords=cX_use,
                targets=T[j:j+1, :],
                model_family=model_family,
                model_type=model_type,
                theta=theta,
                params=params,
                sigma2=sigma2,
                distance_type=distance_type,
                mean=mean,
                jitter=jitter,
                return_weights=False
            )
            mu = float(est[0])
            s2 = float(max(var[0], 0.0))

            # Gaussian draw
            draw = rng.normal(mu, np.sqrt(s2), size=1)[0]

            # append to conditioning set
            cz_buf[ncond] = draw
            cX_buf[ncond] = T[j]
            ncond += 1

            # record in path position
            sim_ns[j] = draw

        # save realization
        if transform == "ns":
            sims_ns[rix] = sim_ns
            sims[rix] = to_sim_scale(sim_ns)
        else:
            sims[rix] = sim_ns

    return (sims, sims_ns) if transform == "ns" else (sims, None)

# Simple Kriging: Custom Sequential Gaussian Simulation
def sgs_simple_kriging_custom_corr(
    values: np.ndarray,               # (n,)
    blocks_all: Dict[str, np.ndarray],# each (N,N) with N = n_obs + m_targets
    n_obs: int,                       # number of original data points (first n rows/cols in blocks_all)
    *,
    custom_kernel: Callable[[Dict[str, np.ndarray], Sequence[float]], np.ndarray],
    theta: Sequence[float],
    sigma2: float,
    n_realizations: int = 1,
    mean: Union[str, float] = "zero",
    random_state: Optional[int] = None,
    jitter: float = 1e-10,
    max_neighbors: Optional[int] = None,
    neighbor_metric: Optional[str] = None,   # <-- which metric to use for neighbor ranking
    progress: bool = True,
) -> np.ndarray:
    """
    Sequential Gaussian Simulation (SGS) using a custom correlation kernel and
    prebuilt full (N,N) distance blocks (N = n_obs + m_targets).

    The targets are assumed to be the last m rows/cols in each blocks_all[k].

    Parameters
    ----------
    values : (n,) array_like
        Conditioning data values at the first `n_obs` positions.
    blocks_all : dict[str, ndarray]
        For each metric key (e.g. 'E','S','A'), a full symmetric (N,N) distance
        matrix with the top-left (n_obs,n_obs) block = data–data, bottom-right
        (m,m) block = target–target, and off-diagonals data–target/target–data.
    n_obs : int
        Number of original data points (first n rows/cols).
    custom_kernel : callable(blocks_subset: dict[str, ndarray], theta) -> ndarray
        Must return the correlation matrix from the passed blocks subset.
        Called per node on (n_cur x n_cur) and (n_cur x 1) subsets.
    theta : sequence of float
        Kernel parameters for custom_kernel.
    sigma2 : float
        Total variance to scale the correlation matrix to covariance.
    n_realizations : int, default 1
        Number of independent realizations.
    mean : {'gls','zero'} or float, default 'zero'
        Simple-kriging mean passed through to the inner SK solver
        used inside `simple_kriging_custom_corr`.
    random_state : int, optional
        RNG seed.
    jitter : float, default 1e-10
        Diagonal jitter passed through to SK solver.
    max_neighbors : int, optional
        If provided, restrict each local SK to the K nearest conditioning points
        (measured with `neighbor_metric` in `blocks_all`).
    neighbor_metric : str, optional
        Which key in `blocks_all` to use for neighbor ranking. If None, the first
        key in `blocks_all` is used.
    progress : bool, default True
        Show tqdm progress bars.

    Returns
    -------
    sims : (n_realizations, m) ndarray
        Simulated values at the m target nodes, in the same order as the
        last m indices of the provided `blocks_all` matrices.
    """
    rng = default_rng(random_state)
    z0 = np.asarray(values, float).ravel()
    n = int(n_obs)
    first_block = next(iter(blocks_all.values()))
    N = first_block.shape[0]
    m = N - n
    if m <= 0:
        raise ValueError("blocks_all must include targets (N = n_obs + m_targets, with m_targets > 0).")

    # choose metric for neighbor selection
    if neighbor_metric is None:
        neighbor_metric = next(iter(blocks_all.keys()))
    if neighbor_metric not in blocks_all:
        raise KeyError(f"neighbor_metric '{neighbor_metric}' not found in blocks_all keys {list(blocks_all.keys())}")

    sims = np.empty((n_realizations, m), float)
    target_indices_global = np.arange(n, n + m, dtype=int)

    outer_iter = tqdm(range(n_realizations), desc="SGS realizations") if progress else range(n_realizations)
    for rix in outer_iter:
        # fresh conditioning set for this realization
        obs_idx = list(range(n))   # global indices of current conditioning points
        cz = z0.copy()
        path = rng.permutation(m)  # simulate targets in random order
        sim = np.empty(m, float)

        inner_iter = tqdm(path, total=m, desc=f"nodes (realization {rix+1}/{n_realizations})") if progress else path
        for jj in inner_iter:
            g_t = target_indices_global[jj]  # global index of the current target

            # --- Neighbor selection (if requested) ---
            if (max_neighbors is not None) and (len(obs_idx) > max_neighbors):
                # distances from current conditioning points to current target,
                # in the neighbor_metric’s full (N,N) matrix
                B_sel = blocks_all[neighbor_metric]
                dvec = B_sel[np.array(obs_idx, dtype=int), g_t]   # shape (n_cur,)
                # pick K smallest finite distances
                finite = np.isfinite(dvec)
                if not np.any(finite):
                    # fall back: no finite distances -> use all
                    obs_sel = obs_idx
                else:
                    # order only finite entries
                    fin_pos = np.nonzero(finite)[0]
                    order = np.argsort(dvec[finite])
                    take = min(max_neighbors, fin_pos.size)
                    sel_local = fin_pos[order[:take]]             # positions in obs_idx
                    obs_sel = [obs_idx[i] for i in sel_local]
            else:
                obs_sel = obs_idx

            # slice (n_cur x n_cur) and (n_cur x 1) blocks for each metric to the chosen neighbors
            idx_arr = np.array(obs_sel, dtype=int)
            blocks_nn = {k: B[np.ix_(idx_arr, idx_arr)] for k, B in blocks_all.items()}
            blocks_nt = {k: B[np.ix_(idx_arr, [g_t])]    for k, B in blocks_all.items()}

            # local SK with custom correlation
            est, var = simple_kriging_custom_corr(
                values=cz if obs_sel is obs_idx else cz[np.array([obs_idx.index(i) for i in obs_sel], dtype=int)],
                blocks_nn=blocks_nn,
                blocks_nt=blocks_nt,
                custom_kernel=custom_kernel,
                theta=theta,
                sigma2=sigma2,
                mean=mean,
                jitter=jitter,
                return_weights=False
            )

            mu = float(est[0]); s2 = float(max(var[0], 0.0))
            draw = rng.normal(mu, np.sqrt(s2), size=1)[0]

            # grow conditioning set
            obs_idx.append(g_t)
            cz = np.append(cz, draw)
            sim[jj] = draw

            if progress:
                inner_iter.set_postfix_str(f"cond={len(obs_idx)}")

        sims[rix, :] = sim

    return sims

# build covariance matrices for custom simulation
def build_blocks_all(blocks_nn, blocks_nt, blocks_tt):
    out = {}
    for k in blocks_nn:
        OO = blocks_nn[k]; OT = blocks_nt[k]; TT = blocks_tt[k]
        if any(isinstance(x, dict) for x in (OO, OT, TT)):
            raise TypeError(
                f"blocks_*['{k}'] must be numeric ndarrays, not dicts. "
                "Pass arrays like D_nn, D_nt, D_tt for each metric key."
            )
        OO = np.asarray(OO, float); OT = np.asarray(OT, float); TT = np.asarray(TT, float)
        TO = OT.T
        out[k] = np.block([[OO, OT], [TO, TT]])
    return out
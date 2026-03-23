"""
Simple kriging and related simulation utilities for built-in and custom
variogram/correlation models.
"""

# import modules
import numpy as np
from numpy.random import default_rng
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
from typing import Callable, Dict, Optional, Sequence, Tuple, Union, Literal
from tqdm.auto import tqdm

from VarioCorreKrigE.krig_utils import (
    geodesic_pairwise,
    pairwise_distances,
    _prepare_balltree_coordinates,
    query_nearest_neighbors_balltree,
    _resolve_model_theta,
    _resolve_correlation_components,
    _resolve_variogram_components,
    build_covariance_nn_nt,
    build_covariance_custom_correlation,
    build_covariance_custom_variogram,
    merge_blocks,
    build_blocks_all,
)

# Estimation of mean for Simple Kriging
def estimate_mean_gls_from_C(z: np.ndarray, C: np.ndarray) -> float:
    """
    Estimate a constant global mean using generalized least squares.

    The estimator is

        m_hat = (1^T C^{-1} z) / (1^T C^{-1} 1)

    Parameters
    ----------
    z : array_like, shape (n,)
        Observed values.
    C : ndarray, shape (n, n)
        Covariance matrix among observations.

    Returns
    -------
    float
        GLS estimate of the constant mean.

    Notes
    -----
    If the GLS denominator is non-finite or non-positive, the arithmetic mean is returned.
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
    values,
    coords,
    targets,
    *,
    model_family: str,
    model_type: str,
    theta: Optional[Sequence[float]] = None,
    params: Optional[dict] = None,
    sigma2: Optional[float] = None,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    mean: Union[str, float] = "gls",
    jitter: float = 1e-10,
    chunk_cols: Optional[int] = None,
    max_neighbors: Optional[int] = None,
    balltree_leaf_size: int = 40,
    return_weights: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    """
    Simple Kriging with either a variogram or a correlation model.

    Estimator (per target x0)
    -------------------------
    w solves C_nn w = C_nt(:, x0)

    z_SK(x0) = m + w^T (z - m 1)
             = sum_i w_i z_i + (1 - sum_i w_i) m

    var_SK(x0) = sigma2 - w^T C_nt(:, x0)

    Parameters
    ----------
    values : (n,) array_like
        Observed values at `coords`.
    coords : (n, d) array_like
        Observation locations.
    targets : (m, d) array_like
        Target locations to estimate.
    model_family : {'variogram', 'correlation'}
        Model family used to build the covariance.
    model_type : str
        Built-in model name.
    theta : sequence of float, optional
        Callable-order parameter vector. Used only when `params` is not given.
    params : dict, optional
        Preferred parameter representation.
        - Variogram: should contain at least c0 and b, plus model shape parameters.
        - Correlation: may contain alpha directly, or c0 and b, plus model shape parameters.
    sigma2 : float, optional
        Total variance scale for correlation models. If provided, it overrides the
        total sill implied by `params`; any `alpha` or `c0`/`b` information in
        `params` is then used only to define the structured-versus-nugget split.
        If neither `sigma2` nor `params['c0']`/`params['b']` is provided, a unit
        variance scale is used.
    distance_type : {'euclidean', 'cartesian', 'geographic', 'angular'}
        Distance metric used to construct pairwise distances.
    projection : str, default 'WGS84'
        Ellipsoid name used for geographic distances.
    mean : {'gls', 'zero'} or floating[any], default 'gls'
        Mean used by simple kriging.
        - 'gls'  : estimate the global mean under the current covariance model
        - 'zero' : use 0.0
        - float  : use a fixed numeric mean
    jitter : float, default 1e-10
        Diagonal stabilization factor multiplied by sigma2.
    chunk_cols : int or None, default None
        Optional column chunk size used during pairwise distance construction to
        reduce peak memory usage. Returned covariance matrices are still dense.
    max_neighbors : int or None, default None
        If provided, use only the `max_neighbors` nearest conditioning points
        for each target. Neighbor selection is performed with BallTree for all
        supported distance types:
        - geographic   : haversine ranking on [lat, lon] in radians
        - euclidean    : Euclidean ranking
        - cartesian    : Euclidean ranking
        - angular      : Euclidean ranking on unit-circle embedding
        If None, all observations are used.
    balltree_leaf_size : int, default 40
        Leaf size passed to BallTree when `max_neighbors` is used.
    return_weights : bool, default False
        If True, also return kriging weights.

    Returns
    -------
    est : (m,) ndarray
        Simple-kriging estimates at targets.
    var : (m,) ndarray
        Simple-kriging variances at targets.
    W : (m, n) ndarray, optional
        Kriging weights for each target, returned only when `return_weights=True`.

    Notes
    -----
    When `max_neighbors` is used, the kriging system is solved separately for
    each target using its local neighborhood. In that case, if `mean='gls'`,
    the GLS mean is computed on the local neighborhood covariance matrix, not on
    the full global dataset.

    """

    z  = np.asarray(values, float).ravel()
    X  = np.asarray(coords, float)
    XT = np.asarray(targets, float)

    n = z.size
    m = XT.shape[0]

    # ------------------------------------------------------------------
    # Global solve: use all observations for all targets
    # ------------------------------------------------------------------
    use_local = (max_neighbors is not None) and (int(max_neighbors) < n)

    if not use_local:
        C_nn, C_nt, sigma2_used = build_covariance_nn_nt(
            X, XT,
            model_family=model_family,
            model_type=model_type,
            theta=theta,
            params=params,
            sigma2=sigma2,
            distance_type=distance_type,
            projection=projection,
            jitter=jitter,
            chunk_cols=chunk_cols,
        )

        if isinstance(mean, (int, float)):
            mval = float(mean)
        elif mean == "zero":
            mval = 0.0
        elif mean == "gls":
            mval = estimate_mean_gls_from_C(z, C_nn)
        else:
            raise ValueError("mean must be 'gls', 'zero', or a numeric value")

        cF = cho_factor(C_nn, overwrite_a=False, check_finite=False)
        Wt = cho_solve(cF, C_nt, check_finite=False)  # (n, m)
        W  = Wt.T                                     # (m, n)
        sumw = W.sum(axis=1)

        est = W @ z + (1.0 - sumw) * mval
        var = np.clip(sigma2_used - np.einsum("ij,ij->i", W, C_nt.T), 0.0, None)

        return (est, var, W) if return_weights else (est, var)

    # ------------------------------------------------------------------
    # Local solve: target-specific nearest-neighbor subsets via BallTree
    # ------------------------------------------------------------------
    nn_ind = query_nearest_neighbors_balltree(
        X,
        XT,
        distance_type=distance_type,
        max_neighbors=max_neighbors,
        balltree_leaf_size=balltree_leaf_size,
    )

    est = np.empty(m, dtype=float)
    var = np.empty(m, dtype=float)
    W_full = np.zeros((m, n), dtype=float) if return_weights else None

    for j in range(m):
        idx = np.asarray(nn_ind[j], dtype=int)

        X_loc = X[idx]
        z_loc = z[idx]

        C_nn, C_nt, sigma2_used = build_covariance_nn_nt(
            X_loc, XT[j:j+1, :],
            model_family=model_family,
            model_type=model_type,
            theta=theta,
            params=params,
            sigma2=sigma2,
            distance_type=distance_type,
            projection=projection,
            jitter=jitter,
            chunk_cols=chunk_cols,
        )

        if isinstance(mean, (int, float)):
            mval = float(mean)
        elif mean == "zero":
            mval = 0.0
        elif mean == "gls":
            mval = estimate_mean_gls_from_C(z_loc, C_nn)
        else:
            raise ValueError("mean must be 'gls', 'zero', or a numeric value")

        cF = cho_factor(C_nn, overwrite_a=False, check_finite=False)
        w = cho_solve(cF, C_nt, check_finite=False).ravel()

        est[j] = w @ z_loc + (1.0 - np.sum(w)) * mval
        var[j] = np.clip(sigma2_used - (w @ C_nt[:, 0]), 0.0, None)

        if return_weights:
            W_full[j, idx] = w

    return (est, var, W_full) if return_weights else (est, var)

# Simple Kriging Function: Custom correlation
def simple_kriging_custom_corr(values, blocks_nn, blocks_nt, custom_kernel, theta,
                               sigma2=1.0, mean='gls', jitter=1e-10, return_weights=False):
    """
    Simple kriging using a custom correlation kernel.

    The custom kernel must return correlation blocks `(R_nn, R_nt)` from the
    supplied distance or dissimilarity blocks. Covariance is then formed as

        C(h) = sigma2 * rho(h).

    Parameters
    ----------
    values : array_like, shape (n,)
        Observed values.
    blocks_nn : dict[str, ndarray]
        Distance or dissimilarity blocks among observations.
    blocks_nt : dict[str, ndarray]
        Distance or dissimilarity blocks from observations to targets.
    custom_kernel: Callable[
        [Dict[str, np.ndarray], Dict[str, np.ndarray], Sequence[float]],
        Tuple[np.ndarray, np.ndarray]
    ],
    theta : sequence of float
        Parameter vector passed directly to `custom_kernel`.
    sigma2 : float, default 1.0
        Total variance used to scale the correlation model.
    mean : {'gls', 'zero'} or float, default 'gls'
        Mean used by simple kriging.
    jitter : float, default 1e-10
        Small diagonal stabilization factor.
    return_weights : bool, default False
        If True, also return the kriging weights.

    Returns
    -------
    est : ndarray, shape (m,)
        Simple-kriging estimates at the targets.
    var : ndarray, shape (m,)
        Simple-kriging variances at the targets.
    W : ndarray, shape (m, n), optional
        Kriging weights, returned only when `return_weights=True`.
    """
    z = np.asarray(values, float).ravel()
    n = z.size

    # Build covariance from blocks + kernel
    C_nn, C_nt, sigma2_used = build_covariance_custom_correlation(
        blocks_nn, blocks_nt, custom_kernel, theta, sigma2=sigma2, jitter=jitter
    )
    C_nn = 0.5 * (C_nn + C_nn.T)

    # Mean
    if isinstance(mean, (int, float)):
        mval = float(mean)
    elif mean == "zero":
        mval = 0.0
    elif mean == "gls":
        mval = estimate_mean_gls_from_C(z, C_nn)
    else:
        raise ValueError("mean must be 'gls', 'zero', or a numeric value")

    # Solve for weights
    cF = cho_factor(C_nn, overwrite_a=False, check_finite=False)
    Wt = cho_solve(cF, C_nt, check_finite=False)  # (n,m)
    W  = Wt.T                                     # (m,n)
    sumw = W.sum(axis=1)
    # Est & Var
    est = W @ z + (1.0 - sumw) * mval
    var = np.clip(sigma2_used - np.einsum("ij,ij->i", W, C_nt.T), 0.0, None)
    return (est, var, W) if return_weights else (est, var)

# Simple Kriging Function: Custom variogram
def simple_kriging_custom_vario(
    values,
    blocks_nn,
    blocks_nt,
    custom_kernel,
    theta,
    mean='gls',
    jitter=1e-10,
    return_weights=False,
):
    """
    Simple kriging using a custom variogram kernel.

    The custom kernel must return semivariogram blocks `(G_nn, G_nt)` from the
    supplied distance or dissimilarity blocks. Covariance is then formed internally as

        C(h) = (c0 + b) - gamma(h).

    Parameters
    ----------
    values : array_like, shape (n,)
        Observed values.
    blocks_nn : dict[str, ndarray]
        Distance or dissimilarity blocks among observations.
    blocks_nt : dict[str, ndarray]
        Distance or dissimilarity blocks from observations to targets.
    custom_kernel : callable
        Function with signature
            custom_kernel(blocks_nn, blocks_nt, theta) -> (G_nn, G_nt)
        returning semivariogram blocks.
    theta : sequence of float
        Parameter vector passed directly to `custom_kernel`.
    c0 : float
        Partial sill.
    b : float
        Nugget.
    mean : {'gls', 'zero'} or float, default 'gls'
        Mean used by simple kriging.
    jitter : float, default 1e-10
        Small diagonal stabilization factor applied inside the covariance builder.
    return_weights : bool, default False
        If True, also return the kriging weights.

    Returns
    -------
    est : ndarray, shape (m,)
        Simple-kriging estimates at the targets.
    var : ndarray, shape (m,)
        Simple-kriging variances at the targets.
    W : ndarray, shape (m, n), optional
        Kriging weights, returned only when `return_weights=True`.
    """
    z = np.asarray(values, float).ravel()

    C_nn, C_nt, sigma2 = build_covariance_custom_variogram(
        blocks_nn, blocks_nt, custom_kernel, theta, jitter=jitter
    )
    C_nn = 0.5 * (C_nn + C_nn.T)

    if isinstance(mean, (int, float)):
        mval = float(mean)
    elif mean == "zero":
        mval = 0.0
    elif mean == "gls":
        mval = estimate_mean_gls_from_C(z, C_nn)
    else:
        raise ValueError("mean must be 'gls', 'zero', or a numeric value")

    cF = cho_factor(C_nn, overwrite_a=False, check_finite=False)
    Wt = cho_solve(cF, C_nt, check_finite=False)
    W = Wt.T

    est = W @ z + (1.0 - W.sum(axis=1)) * mval
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
    theta: Optional[Sequence[float]] = None,
    params: Optional[dict] = None,
    sigma2: Optional[float] = None,
    distance_type: Literal["euclidean", "cartesian", "geographic", "angular"] = "euclidean",
    mean: Union[Literal["gls", "zero"], float] = "zero",
    n_realizations: int = 1,
    transform: Literal["ns", "none"] = "ns",
    random_state: Optional[int] = None,
    jitter: float = 1e-10,
    chunk_cols: Optional[int] = None,
    progress: bool = True,
    max_neighbors: Optional[int] = None,
    balltree_leaf_size: int = 40,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Sequential Gaussian Simulation (SGS) using simple kriging as the local solver.

    Parameters
    ----------
    values : (n,) array_like
        Conditioning data values at `coords`.
    coords : (n, d) array_like
        Conditioning locations.
    targets : (m, d) array_like
        Simulation nodes.
    model_family : {'variogram', 'correlation'}
        Model family used by the local simple-kriging solve.
    model_type : str
        Built-in model name.
    theta : sequence of float, optional
        Callable-order parameter vector. Used only when `params` is not given.
    params : dict, optional
        Preferred parameter representation for the local kriging model.
    sigma2 : float, optional
        Total variance scale for correlation models. If provided, it overrides the
        total sill implied by `params`; any `alpha` or `c0`/`b` information in
        `params` is then used only to define the structured-versus-nugget split.
    distance_type : {'euclidean', 'cartesian', 'geographic', 'angular'}
        Distance metric used by the local kriging solve.
    mean : {'gls', 'zero'} or float, default 'zero'
        Mean passed to the local simple-kriging solve.
    n_realizations : int, default 1
        Number of independent realizations.
    transform : {'ns', 'none'}, default 'ns'
        'ns'   : normal-score transform before simulation, then back-transform.
        'none' : simulate directly on the provided scale.
    random_state : int, optional
        RNG seed for random visiting paths and Gaussian draws.
    jitter : float, default 1e-10
        Diagonal stabilization factor used in local kriging solves.
    chunk_cols : int or None, default None
        Optional chunk size passed to local distance construction.
    progress : bool, default True
        Show tqdm progress bars.
    max_neighbors : int or None, default None
        If provided, use only the nearest conditioning points in each local solve.
        Neighbor selection is delegated to `simple_kriging(...)`, which uses
        BallTree for all supported distance types.
    balltree_leaf_size : int, default 40
        Leaf size passed to BallTree when `max_neighbors` is used.

    Returns
    -------
    sims : (n_realizations, m) ndarray
        Simulated values at target nodes, on the original scale when transform='ns'.
    sims_ns : (n_realizations, m) ndarray or None
        Simulated values on the normal-score scale, returned only when transform='ns'.

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
            est, var = simple_kriging(
                values=cz_buf[:ncond],
                coords=cX_buf[:ncond],
                targets=T[j:j+1, :],
                model_family=model_family,
                model_type=model_type,
                theta=theta,
                params=params,
                sigma2=sigma2,
                distance_type=distance_type,
                mean=mean,
                jitter=jitter,
                chunk_cols=chunk_cols,
                max_neighbors=max_neighbors,
                balltree_leaf_size=balltree_leaf_size,
                return_weights=False,
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
    values: np.ndarray,
    blocks_all: Dict[str, np.ndarray],
    n_obs: int,
    *,
    custom_kernel: Callable[
        [Dict[str, np.ndarray], Dict[str, np.ndarray], Sequence[float]],
        Tuple[np.ndarray, np.ndarray]
    ],
    theta: Sequence[float],
    sigma2: float,
    n_realizations: int = 1,
    mean: Union[str, float] = "zero",
    random_state: Optional[int] = None,
    jitter: float = 1e-10,
    max_neighbors: Optional[int] = None,
    neighbor_metric: Optional[str] = None,
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
    custom_kernel : callable
        Function with signature
            custom_kernel(blocks_nn, blocks_nt, theta) -> (R_nn, R_nt)
        returning correlation blocks.
    theta : sequence of float
        Parameter vector passed directly to `custom_kernel`.
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
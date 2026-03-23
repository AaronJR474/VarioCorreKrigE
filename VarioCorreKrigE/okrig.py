"""
VarioCorreKrigE: Ordinary Kriging (OK)

This module mirrors the structure of `skrig.py` (Simple Kriging) but replaces the
Simple Kriging solver with Ordinary Kriging:

- Mean is treated as unknown but constant.
- Unbiasedness is enforced via the constraint sum(w) = 1, implemented through
  a Lagrange multiplier (μ).

Core OK system (covariance form):

    C_nn w + 1 μ = c_n0
    1^T  w       = 1

where:
- C_nn: covariance between observations (n x n)
- c_n0: covariance between observations and target (n x 1)
- w: kriging weights (n x 1)
- μ: Lagrange multiplier (scalar)

Efficient solve using Cholesky of C_nn only (avoids factoring the indefinite
augmented matrix):

    w = C_nn^{-1} c_n0 - μ C_nn^{-1} 1
    μ = (1^T C_nn^{-1} c_n0 - 1) / (1^T C_nn^{-1} 1)

Prediction:
    z_hat = w^T z

Kriging variance (covariance form; with this μ sign convention):
    var = σ^2 - w^T c_n0 - μ
where σ^2 is the sill/variance used by the covariance model.
"""

from __future__ import annotations
from typing import Callable, Dict, Optional, Sequence, Tuple, Literal
import warnings
import numpy as np
from numpy.random import default_rng
from scipy.linalg import cho_factor, cho_solve
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

# --------------------------------------------------------------------------------------
# Ordinary Kriging core
# --------------------------------------------------------------------------------------

def _ok_weights_from_cov(
    C_nn: np.ndarray,
    C_nt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute OK weights for multiple targets using Cholesky on C_nn.
    Returns:
      W: (m, n) weights for each of m targets
      mu: (m,) Lagrange multipliers
    """
    n = C_nn.shape[0]
    ones = np.ones(n, float)

    cf = cho_factor(C_nn, lower=True, check_finite=False)

    Ci1 = cho_solve(cf, ones, check_finite=False)               # (n,)
    denom = float(ones @ Ci1)                                   # scalar

    CiC = cho_solve(cf, C_nt, check_finite=False)               # (n, m)
    a = (ones @ CiC)                                            # (m,)
    mu = (a - 1.0) / denom                                      # (m,)

    # W^T = CiC - Ci1 * mu
    Wt = CiC - Ci1[:, None] * mu[None, :]
    W = Wt.T
    return W, mu


def ordinary_kriging(
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
    jitter: float = 1e-10,
    chunk_cols: Optional[int] = None,
    max_neighbors: Optional[int] = None,
    balltree_leaf_size: int = 40,
    return_weights: bool = False,
):
    """
    Ordinary kriging with either a built-in variogram model or a built-in correlation model.

    This routine treats the mean as unknown but constant and enforces the ordinary-
    kriging unbiasedness constraint through a Lagrange multiplier.

    Parameters
    ----------
    values : array_like, shape (n,)
        Observed values at `coords`.
    coords : array_like, shape (n, d)
        Observation locations.
    targets : array_like, shape (m, d)
        Target locations to estimate.
    model_family : {'variogram', 'correlation'}
        Model family used to build the covariance matrix.
    model_type : str
        Name of the built-in model.
    theta : sequence of float, optional
        Callable-order parameter vector. Used when `params` is not supplied.
    params : dict, optional
        Preferred parameter representation for the selected model.
    sigma2 : float, optional
        Total variance scale used for correlation models. If provided, it overrides
        the total sill implied by `params`; any `alpha` or `c0`/`b` information in
        `params` is then used only to define the structured-versus-nugget split.
    distance_type : {'euclidean', 'cartesian', 'geographic', 'angular'}, default 'euclidean'
        Distance metric used to construct pairwise distances.
    projection : str, default 'WGS84'
        Ellipsoid name used for geographic distances.
    jitter : float, default 1e-10
        Small diagonal stabilization factor.
    chunk_cols : int or None, default None
        Optional chunk size used during pairwise distance construction.
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
        If True, also return the kriging weights.

    Returns
    -------
    est : ndarray, shape (m,)
        Ordinary-kriging estimates at the targets.
    var : ndarray, shape (m,)
        Ordinary-kriging variances at the targets.
    W : ndarray, shape (m, n), optional
        Kriging weights, returned only when `return_weights=True`.

    Notes
    -----
    When `max_neighbors` is used, the kriging system is solved separately for
    each target using its local neighborhood.
    """

    z = np.asarray(values, float).reshape(-1)
    X = np.asarray(coords, float)
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

        W, mu = _ok_weights_from_cov(C_nn, C_nt)

        est = W @ z
        wTc = np.einsum("ij,ij->i", W, C_nt.T)
        var = np.maximum(sigma2_used - wTc - mu, 0.0)

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

        W_loc, mu_loc = _ok_weights_from_cov(C_nn, C_nt)
        w = W_loc[0]

        est[j] = w @ z_loc
        var[j] = np.maximum(sigma2_used - (w @ C_nt[:, 0]) - mu_loc[0], 0.0)

        if return_weights:
            W_full[j, idx] = w

    return (est, var, W_full) if return_weights else (est, var)


def ordinary_kriging_custom_corr(
    values,
    blocks_nn,
    blocks_nt,
    custom_kernel: Callable,
    theta,
    *,
    sigma2: float = 1.0,
    jitter: float = 1e-10,
    return_weights: bool = False,
):
    """
    Ordinary kriging using a custom correlation kernel.

    The custom kernel must return correlation blocks `(R_nn, R_nt)` from the
    supplied distance blocks, after which covariance is formed as

        C(h) = sigma2 * rho(h).

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
            custom_kernel(blocks_nn, blocks_nt, theta) -> (R_nn, R_nt)
        returning correlation blocks.
    theta : sequence of float
        Parameter vector passed directly to `custom_kernel`.
    sigma2 : float, default 1.0
        Total variance used to scale the correlation model.
    jitter : float, default 1e-10
        Small diagonal stabilization factor.
    return_weights : bool, default False
        If True, also return the kriging weights.

    Returns
    -------
    est : ndarray, shape (m,)
        Ordinary-kriging estimates.
    var : ndarray, shape (m,)
        Ordinary-kriging variances.
    W : ndarray, shape (m, n), optional
        Kriging weights, returned only when `return_weights=True`.
    """
    z = np.asarray(values, float).reshape(-1)

    C_nn, C_nt, sigma2_used = build_covariance_custom_correlation(
        blocks_nn, blocks_nt, custom_kernel, theta, sigma2=sigma2, jitter=jitter
    )
    C_nn = 0.5 * (C_nn + C_nn.T)

    W, mu = _ok_weights_from_cov(C_nn, C_nt)

    est = W @ z
    wTc = np.einsum("ij,ij->i", W, C_nt.T)
    var = sigma2_used - wTc - mu
    var = np.maximum(var, 0.0)

    if return_weights:
        return est, var, W
    return est, var

def ordinary_kriging_custom_vario(
    values,
    blocks_nn,
    blocks_nt,
    custom_kernel: Callable,
    theta,
    *,
    jitter: float = 1e-10,
    return_weights: bool = False,
):
    """
    Ordinary kriging using a custom variogram kernel.

    The custom kernel must return semivariogram blocks `(G_nn, G_nt)` from the
    supplied distance blocks. Covariance is then formed internally as

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
    jitter : float, default 1e-10
        Small diagonal stabilization factor applied inside the covariance builder.
    return_weights : bool, default False
        If True, also return the kriging weights.

    Returns
    -------
    est : ndarray, shape (m,)
        Ordinary-kriging estimates.
    var : ndarray, shape (m,)
        Ordinary-kriging variances.
    W : ndarray, shape (m, n), optional
        Kriging weights, returned only when `return_weights=True`.
    """
    z = np.asarray(values, float).reshape(-1)

    C_nn, C_nt, sigma2_used = build_covariance_custom_variogram(
        blocks_nn, blocks_nt, custom_kernel, theta, jitter=jitter
    )
    C_nn = 0.5 * (C_nn + C_nn.T)

    W, mu = _ok_weights_from_cov(C_nn, C_nt)

    est = W @ z
    wTc = np.einsum("ij,ij->i", W, C_nt.T)
    var = np.maximum(sigma2_used - wTc - mu, 0.0)

    if return_weights:
        return est, var, W
    return est, var

# --------------------------------------------------------------------------------------
# Optional: Normal-score transform + SGS (ported from skrig.py but using OK)
# --------------------------------------------------------------------------------------

def normal_score_transform(values, *, nquantiles=None):
    """
    Transform data to normal scores using an empirical quantile map.

    Parameters
    ----------
    values : array_like, shape (n,)
        Input data on the original scale.
    nquantiles : int or None, default None
        Number of empirical quantiles used to build the monotone transform.
        If None, `min(1000, n)` is used.

    Returns
    -------
    z_ns : ndarray, shape (n,)
        Input values mapped to normal-score space.
    inverse_map : callable
        Monotone interpolation function mapping normal scores back to the original scale.

    Notes
    -----
    This is a quantile-based transform intended for SGS workflows where simulation is
    performed on an approximately Gaussian scale and then back-transformed.
    """
    z = np.asarray(values, float).reshape(-1)
    n = z.size
    if n == 0:
        raise ValueError("normal_score_transform: empty input")

    if nquantiles is None:
        nquantiles = min(1000, n)

    qs = np.linspace(0.0, 1.0, nquantiles)
    zq = np.quantile(z, qs)

    from scipy.stats import norm
    nq = norm.ppf(np.clip(qs, 1e-6, 1 - 1e-6))

    forward = lambda x: np.interp(x, zq, nq)
    inverse = lambda y: np.interp(y, nq, zq)

    return forward(z), inverse

def sgs_ordinary_kriging(
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
    projection: str = "WGS84",
    transform: Literal["ns", "none"] = "ns",
    n_realizations: int = 1,
    random_state: Optional[int] = None,
    jitter: float = 1e-10,
    chunk_cols: Optional[int] = None,
    max_neighbors: Optional[int] = None,
    balltree_leaf_size: int = 40,
    progress: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:

    """
    Experimental sequential simulation using ordinary kriging as the local solver.

    This is not classical SGS: because ordinary kriging re-estimates the local mean
    at each node, the simulated field can drift and need not reproduce the intended
    marginal distribution or target covariance as reliably as SK-based SGS.
    Prefer `sgs_simple_kriging(...)` for production SGS.

    Parameters
    ----------
    values : (n,) array_like
        Conditioning data values at the observation coordinates.
    coords : (n, d) array_like
        Conditioning locations.
    targets : (m, d) array_like
        Simulation nodes.
    model_family : {'variogram', 'correlation'}
        Model family used by the local ordinary-kriging solve.
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
    projection : str, default 'WGS84'
        Ellipsoid name used for geographic distances.
    transform : {'ns', 'none'}, default 'ns'
        'ns'   : normal-score transform before simulation, then back-transform.
        'none' : simulate directly on the provided scale.
    n_realizations : int, default 1
        Number of independent realizations.
    random_state : int, optional
        RNG seed for random visiting paths and Gaussian draws.
    jitter : float, default 1e-10
        Diagonal stabilization factor used in local kriging solves.
    chunk_cols : int or None, default None
        Optional chunk size passed to local distance construction.
    max_neighbors : int or None, default None
        If provided, use only the nearest conditioning points in each local solve.
        Neighbor selection is delegated to `ordinary_kriging(...)`, which uses
        BallTree for all supported distance types.
    balltree_leaf_size : int, default 40
        Leaf size passed to BallTree when `max_neighbors` is used.
    progress : bool, default True
        Show tqdm progress bars.

    Returns
    -------
    sims : (n_realizations, m) ndarray
        Simulated values at target nodes, on the original scale when transform='ns'.
    sims_ns : (n_realizations, m) ndarray or None
        Simulated values on the normal-score scale, returned only when transform='ns'.
    """

    warnings.warn(
        "sgs_ordinary_kriging is experimental and is not classical SGS. "
        "Because ordinary kriging re-estimates the local mean sequentially, "
        "the simulated field can drift. Prefer sgs_simple_kriging for production SGS.",
        RuntimeWarning,
        stacklevel=2,
    )

    rng = np.random.default_rng(random_state)

    v = np.asarray(values, float).ravel()
    X = np.asarray(coords, float)
    T = np.asarray(targets, float)
    n, d = X.shape
    m = T.shape[0]

    # transform handling
    if transform == "ns":
        v_ns, inv_ns = normal_score_transform(v)
        to_sim_scale = inv_ns
        cond_vals0 = v_ns
    elif transform == "none":
        to_sim_scale = lambda z: z
        cond_vals0 = v
    else:
        raise ValueError("transform must be 'ns' or 'none'")

    sims = np.empty((n_realizations, m), float)
    sims_ns = np.empty((n_realizations, m), float) if transform == "ns" else None

    # conditioning buffers
    cX_buf = np.empty((n + m, d), float)
    cX_buf[:n] = X
    cz_buf = np.empty(n + m, float)
    cz_buf[:n] = cond_vals0

    try:
        pbar_outer = tqdm(range(n_realizations), disable=not progress, desc="SGS(OK) realizations")
    except Exception:
        pbar_outer = range(n_realizations)

    for rix in pbar_outer:
        # random path
        path = rng.permutation(m)

        # reset conditioning
        ncond = n
        cX_buf[:n] = X
        cz_buf[:n] = cond_vals0
        sim_ns = np.empty(m, float)

        try:
            inner_iter = tqdm(path, disable=not progress,
                              desc=f"nodes (realization {rix+1}/{n_realizations})",
                              total=m)
        except Exception:
            inner_iter = path

        for j in inner_iter:
            est, var = ordinary_kriging(
                values=cz_buf[:ncond],
                coords=cX_buf[:ncond],
                targets=T[j:j+1, :],
                model_family=model_family,
                model_type=model_type,
                theta=theta,
                params=params,
                sigma2=sigma2,
                distance_type=distance_type,
                projection=projection,
                jitter=jitter,
                chunk_cols=chunk_cols,
                max_neighbors=max_neighbors,
                balltree_leaf_size=balltree_leaf_size,
                return_weights=False,
            )

            mu = float(est[0])
            s2 = float(max(var[0], 0.0))

            draw = rng.normal(mu, np.sqrt(s2))

            # append to conditioning
            cz_buf[ncond] = draw
            cX_buf[ncond] = T[j]
            ncond += 1

            sim_ns[j] = draw

        if transform == "ns":
            sims_ns[rix] = sim_ns
            sims[rix] = to_sim_scale(sim_ns)
        else:
            sims[rix] = sim_ns

    return (sims, sims_ns) if transform == "ns" else (sims, None)

def sgs_ordinary_kriging_custom_corr(
    values: np.ndarray,               # (n,)
    blocks_all: Dict[str, np.ndarray],# each (N,N) with N = n_obs + m_targets
    n_obs: int,
    *,
    custom_kernel: Callable[
            [Dict[str, np.ndarray], Dict[str, np.ndarray], Sequence[float]],
            Tuple[np.ndarray, np.ndarray]
    ],
    theta: Sequence[float],
    sigma2: float = 1.0,
    n_realizations: int = 1,
    random_state: Optional[int] = None,
    jitter: float = 1e-10,
    max_neighbors: Optional[int] = None,
    neighbor_metric: Optional[str] = None,
    progress: bool = True,
) -> np.ndarray:
    """
    Sequential Gaussian Simulation (SGS) using **ordinary kriging** as the
    local estimator, with a custom correlation kernel and precomputed full
    distance blocks.

    This function mirrors the structure of `sgs_simple_kriging_custom_corr`,
    but uses **ordinary kriging** (OK) instead of simple kriging (SK).
    Ordinary kriging enforces the unbiasedness constraint and therefore does
    not require a fixed mean parameter.

    Parameters
    ----------
    values : (n,) array_like
        Conditioning data values at the first `n_obs` locations.

    blocks_all : dict[str, ndarray]
        A dictionary of full (N × N) distance or dissimilarity matrices,
        one per metric used by the custom kernel.
        Here `N = n_obs + m_targets`, with:
            - rows/cols 0 … n_obs−1   = observed data locations
            - rows/cols n_obs … N−1   = target nodes to be simulated

        Each matrix must be symmetric and aligned identically.

    n_obs : int
        Number of original conditioning data points.
        The remaining `m = N − n_obs` entries correspond to simulation targets.

    custom_kernel : callable
        Function with signature

            custom_kernel(blocks_nn, blocks_nt, theta) -> (R_nn, R_nt)

        where `blocks_nn` and `blocks_nt` are dictionaries of sliced distance or
        dissimilarity blocks for the current neighborhood. The function must return
        the correlation blocks used by the local ordinary-kriging solve.

    theta : sequence of float
        Parameter vector passed directly to `custom_kernel`.

    sigma2 : float
        Total variance used to scale the correlation matrix into a covariance
        matrix:  C = sigma2 * R.

    n_realizations : int, default 1
        Number of independent SGS realizations to generate.

    random_state : int, optional
        Seed for the random number generator controlling both the random
        visiting path and the Gaussian draws.

    jitter : float, default 1e-10
        Small diagonal stabilisation term added inside the OK system.

    max_neighbors : int, optional
        If provided, each local OK solve uses only the K nearest conditioning
        points, where “nearest” is determined by `neighbor_metric`.

    neighbor_metric : str, optional
        Key in `blocks_all` specifying which distance matrix to use for
        neighbor ranking.
        If None, the first key in `blocks_all` is used.

    progress : bool, default True
        Whether to display tqdm progress bars for realizations and nodes.

    Returns
    -------
    sims : (n_realizations, m) ndarray
        Simulated values at the `m` target nodes, in the same order as the
        last `m` indices of the provided `blocks_all` matrices.

    Notes
    -----
    - This is a **pure ordinary‑kriging SGS**: the local estimator enforces
      unbiasedness via the OK constraint and does not use a fixed mean.
    - All covariance structure comes from the user‑supplied `custom_kernel`
      and the precomputed block matrices.
    - The conditioning set grows sequentially as each simulated value is
      appended.
    """

    warnings.warn(
        "sgs_ordinary_kriging is experimental and is not classical SGS. "
        "Because ordinary kriging re-estimates the local mean sequentially, "
        "the simulated field can drift. Prefer sgs_simple_kriging for production SGS.",
        RuntimeWarning,
        stacklevel=2,
    )

    rng = default_rng(random_state)
    z0 = np.asarray(values, float).ravel()
    n = int(n_obs)

    first_block = next(iter(blocks_all.values()))
    N = first_block.shape[0]
    m = N - n
    if m <= 0:
        raise ValueError("blocks_all must include targets (N = n_obs + m_targets).")

    # choose metric for neighbor selection
    if neighbor_metric is None:
        neighbor_metric = next(iter(blocks_all.keys()))
    if neighbor_metric not in blocks_all:
        raise KeyError(f"neighbor_metric '{neighbor_metric}' not found in blocks_all keys {list(blocks_all.keys())}")

    sims = np.empty((n_realizations, m), float)
    target_indices_global = np.arange(n, n + m, dtype=int)

    outer_iter = tqdm(range(n_realizations), desc="SGS(OK) realizations") if progress else range(n_realizations)
    for rix in outer_iter:

        # conditioning set for this realization
        obs_idx = list(range(n))
        cz = z0.copy()
        path = rng.permutation(m)
        sim = np.empty(m, float)

        inner_iter = tqdm(path, total=m,
                          desc=f"nodes (realization {rix+1}/{n_realizations})") if progress else path

        for jj in inner_iter:
            g_t = target_indices_global[jj]

            # --- Neighbor selection ---
            if (max_neighbors is not None) and (len(obs_idx) > max_neighbors):
                B_sel = blocks_all[neighbor_metric]
                dvec = B_sel[np.array(obs_idx, dtype=int), g_t]

                finite = np.isfinite(dvec)
                if not np.any(finite):
                    obs_sel = obs_idx
                else:
                    fin_pos = np.nonzero(finite)[0]
                    order = np.argsort(dvec[finite])
                    take = min(max_neighbors, fin_pos.size)
                    sel_local = fin_pos[order[:take]]
                    obs_sel = [obs_idx[i] for i in sel_local]
            else:
                obs_sel = obs_idx

            idx_arr = np.array(obs_sel, dtype=int)

            # slice blocks for OK
            blocks_nn = {k: B[np.ix_(idx_arr, idx_arr)] for k, B in blocks_all.items()}
            blocks_nt = {k: B[np.ix_(idx_arr, [g_t])]    for k, B in blocks_all.items()}

            # local ORDINARY KRIGING with custom correlation
            est, var = ordinary_kriging_custom_corr(
                values=cz if obs_sel is obs_idx else cz[np.array([obs_idx.index(i) for i in obs_sel], dtype=int)],
                blocks_nn=blocks_nn,
                blocks_nt=blocks_nt,
                custom_kernel=custom_kernel,
                theta=theta,
                sigma2=sigma2,
                jitter=jitter,
                return_weights=False,
            )

            mu = float(est[0])
            s2 = float(max(var[0], 0.0))
            draw = rng.normal(mu, np.sqrt(s2))

            # grow conditioning set
            obs_idx.append(g_t)
            cz = np.append(cz, draw)
            sim[jj] = draw

            if progress:
                inner_iter.set_postfix_str(f"cond={len(obs_idx)}")

        sims[rix, :] = sim

    return sims

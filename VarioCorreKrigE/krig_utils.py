"""
Utilities for kriging
"""

from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from pyproj import Geod
from sklearn.neighbors import BallTree

from VarioCorreKrigE.variofit import VARIOGRAM_MODELS
from VarioCorreKrigE.correfit import CORRELATION_MODELS

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
    projection: str = "WGS84",
    chunk_cols: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build pairwise distance matrices D_nn (n×n) and D_nt (n×m).

    Parameters
    ----------
    coords : (n,d) float array
        Observation locations.
        - 'geographic': columns [lat, lon] in degrees (WGS84 etc.).
        - 'euclidean'/'cartesian': linear coordinates (units arbitrary).
        - 'angular': 1D angles in degrees, shape (n,) or (n,1).
          Internally these are converted to radians for circular separation
          calculations, and the returned angular distances are in degrees.
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

    X = np.asarray(coords, float)
    XT = np.asarray(targets, float)

    n = X.shape[0]
    m = XT.shape[0]

    def _cc(ncols):
        if chunk_cols is None or chunk_cols <= 0:
            return ncols
        return int(chunk_cols)

    if distance_type == "geographic":
        D_nn, _ = geodesic_pairwise(X, X, ellps=projection, return_az=False, chunk_cols=_cc(n))
        D_nt, _ = geodesic_pairwise(X, XT, ellps=projection, return_az=False, chunk_cols=_cc(m))
        return D_nn, D_nt

    elif distance_type == "euclidean":
        D_nn = np.empty((n, n), dtype=float)
        D_nt = np.empty((n, m), dtype=float)

        cc_n = _cc(n)
        cc_m = _cc(m)

        for j0 in range(0, n, cc_n):
            j1 = min(n, j0 + cc_n)
            diff = X[:, None, :] - X[None, j0:j1, :]
            D_nn[:, j0:j1] = np.linalg.norm(diff, axis=-1)

        for j0 in range(0, m, cc_m):
            j1 = min(m, j0 + cc_m)
            diff = X[:, None, :] - XT[None, j0:j1, :]
            D_nt[:, j0:j1] = np.linalg.norm(diff, axis=-1)

        return D_nn, D_nt

    elif distance_type == "cartesian":
        if X.shape[1] != 2 or XT.shape[1] != 2:
            raise ValueError("For distance_type='cartesian', coords and targets must have shape (n,2) and (m,2).")

        x, y = X[:, 0], X[:, 1]
        xT, yT = XT[:, 0], XT[:, 1]

        D_nn = np.empty((n, n), dtype=float)
        D_nt = np.empty((n, m), dtype=float)

        cc_n = _cc(n)
        cc_m = _cc(m)

        for j0 in range(0, n, cc_n):
            j1 = min(n, j0 + cc_n)
            D_nn[:, j0:j1] = np.hypot(x[:, None] - x[None, j0:j1],
                                      y[:, None] - y[None, j0:j1])

        for j0 in range(0, m, cc_m):
            j1 = min(m, j0 + cc_m)
            D_nt[:, j0:j1] = np.hypot(x[:, None] - xT[None, j0:j1],
                                      y[:, None] - yT[None, j0:j1])

        return D_nn, D_nt

    elif distance_type == "angular":
        if X.ndim == 2 and X.shape[1] != 1:
            raise ValueError("For distance_type='angular', coords must be 1D angles in degrees or shape (n,1).")
        if XT.ndim == 2 and XT.shape[1] != 1:
            raise ValueError("For distance_type='angular', targets must be 1D angles in degrees or shape (m,1).")

        theta_obs = np.deg2rad(np.asarray(X, float).ravel())
        theta_tar = np.deg2rad(np.asarray(XT, float).ravel())

        D_nn = np.empty((n, n), dtype=float)
        D_nt = np.empty((n, m), dtype=float)

        cc_n = _cc(n)
        cc_m = _cc(m)

        for j0 in range(0, n, cc_n):
            j1 = min(n, j0 + cc_n)
            cos_nn = np.cos(theta_obs[:, None] - theta_obs[None, j0:j1])
            D_nn[:, j0:j1] = np.degrees(np.arccos(np.clip(cos_nn, -1.0, 1.0)))

        for j0 in range(0, m, cc_m):
            j1 = min(m, j0 + cc_m)
            cos_nt = np.cos(theta_obs[:, None] - theta_tar[None, j0:j1])
            D_nt[:, j0:j1] = np.degrees(np.arccos(np.clip(cos_nt, -1.0, 1.0)))

        return D_nn, D_nt

    else:
        raise ValueError("distance_type must be 'geographic', 'euclidean', 'angular', or 'cartesian'")

# balltree/max neighbour search helpers
def _prepare_balltree_coordinates(
    X: np.ndarray,
    distance_type: str,
) -> Tuple[np.ndarray, str]:
    """
    Convert coordinates into a BallTree-ready feature space.

    Parameters
    ----------
    X : ndarray
        Input coordinates.
    distance_type : {'geographic', 'euclidean', 'cartesian', 'angular'}
        Distance type used elsewhere in the kriging workflow.

    Returns
    -------
    X_tree : ndarray
        Coordinates transformed into the feature space used by BallTree.
    metric : str
        BallTree metric name.

    Notes
    -----
    - geographic:
        Input is assumed to be [lat, lon] in degrees. BallTree(haversine)
        expects radians, so this helper converts to radians.
    - euclidean / cartesian:
        Coordinates are passed through unchanged and queried with Euclidean distance.
    - angular:
        Input is assumed to be angles in degrees. Internally they are converted
        to radians, embedded on the unit circle as [cos(theta), sin(theta)], and
        queried with Euclidean distance. This preserves nearest-neighbor ordering
        with respect to angular separation.
    """
    X = np.asarray(X, float)

    if distance_type == "geographic":
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(
                "For distance_type='geographic', coordinates must have shape (n, 2) "
                "with columns [lat, lon] in degrees."
            )
        return np.deg2rad(X), "haversine"

    elif distance_type in ("euclidean", "cartesian"):
        if X.ndim != 2:
            raise ValueError(
                f"For distance_type='{distance_type}', coordinates must be a 2D array."
            )
        return X, "euclidean"

    elif distance_type == "angular":
        if X.ndim == 2:
            if X.shape[1] != 1:
                raise ValueError(
                    "For distance_type='angular', coordinates must be 1D angles in degrees or shape (n, 1)."
                )
            theta_deg = X[:, 0]
            theta = np.deg2rad(theta_deg)
        else:
            theta_deg = X.ravel()
            theta = np.deg2rad(theta_deg)

        X_tree = np.column_stack([np.cos(theta), np.sin(theta)])
        return X_tree, "euclidean"

    else:
        raise ValueError(
            "distance_type must be 'geographic', 'euclidean', 'angular', or 'cartesian'"
        )

def query_nearest_neighbors_balltree(
    coords: np.ndarray,
    targets: np.ndarray,
    *,
    distance_type: str = "euclidean",
    max_neighbors: Optional[int] = None,
    balltree_leaf_size: int = 40,
) -> np.ndarray:
    """
    Query nearest observation indices for each target using BallTree.

    Parameters
    ----------
    coords : ndarray, shape (n, d)
        Observation coordinates.
    targets : ndarray, shape (m, d)
        Target coordinates.
    distance_type : {'geographic', 'euclidean', 'cartesian', 'angular'}
        Distance type used for neighbor selection.
    max_neighbors : int or None, default None
        Number of nearest observations to return per target.
        If None, all observations are returned in nearest-neighbor order.
    balltree_leaf_size : int, default 40
        Leaf size passed to BallTree.

    Returns
    -------
    ind : ndarray, shape (m, k)
        Observation indices for the k nearest neighbors of each target.

    Notes
    -----
    For `distance_type='geographic'`, neighbor ranking is based on BallTree's
    haversine metric on [lat, lon] in radians. This is used only for selecting
    neighbors; the actual covariance matrices are still built later using the
    package's standard distance machinery.
    """
    X_tree, metric = _prepare_balltree_coordinates(coords, distance_type)
    XT_tree, _ = _prepare_balltree_coordinates(targets, distance_type)

    n = X_tree.shape[0]
    if n == 0:
        raise ValueError("coords must contain at least one observation.")

    if max_neighbors is None:
        k = n
    else:
        k = int(max_neighbors)
        if k < 1:
            raise ValueError("max_neighbors must be None or a positive integer.")
        k = min(k, n)

    tree = BallTree(X_tree, leaf_size=balltree_leaf_size, metric=metric)
    _, ind = tree.query(XT_tree, k=k, return_distance=True, sort_results=True)
    return ind


# Build Covariance Matrices from Variogram or Correlation Models
def _resolve_model_theta(theta, params, model_type, model_family):
    """
    Resolve the callable parameter vector for built-in models.

    Priority:
      1) if params is provided, derive theta from params
      2) else if theta is provided, use theta
      3) else raise
    """
    if params is not None:
        from VarioCorreKrigE.utils import theta_from_params
        return theta_from_params(params, model_type, family=model_family)

    if theta is not None:
        return list(np.asarray(theta, float).ravel())

    raise ValueError(
        f"For model_family='{model_family}', provide either params or theta."
    )

def _resolve_correlation_components(params=None, sigma2=None):
    """
    Resolve c0, b, sigma2_used, and alpha for correlation models.

    Rules
    -----
    1) If sigma2 is provided, it sets the total variance scale.
    2) params['alpha'] or the ratio c0 / (c0 + b) controls only the
       structured-vs-nugget split.
    3) If neither alpha nor c0/b are provided, assume alpha = 1 and b = 0.
    """
    params = {} if params is None else dict(params)

    # First resolve the structural fraction alpha.
    if "alpha" in params:
        alpha = float(params["alpha"])

    elif ("c0" in params) or ("b" in params):
        c0_in = float(params.get("c0", 0.0))
        b_in = float(params.get("b", 0.0))
        s_in = c0_in + b_in

        if not np.isfinite(s_in) or s_in <= 0.0:
            raise ValueError(
                "For correlation models, params must satisfy c0 + b > 0 "
                "when alpha is inferred from c0 and b."
            )

        alpha = c0_in / s_in

    else:
        alpha = 1.0

    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1].")

    # Then resolve the total variance scale.
    if sigma2 is not None:
        sigma2_used = float(sigma2)
    elif ("c0" in params) or ("b" in params):
        sigma2_used = float(params.get("c0", 0.0)) + float(params.get("b", 0.0))
    else:
        sigma2_used = 1.0

    if not np.isfinite(sigma2_used) or sigma2_used < 0.0:
        raise ValueError("sigma2 must be finite and nonnegative.")

    c0 = alpha * sigma2_used
    b = (1.0 - alpha) * sigma2_used

    return c0, b, sigma2_used, alpha

def _resolve_variogram_components(theta=None, params=None, model_type=None):
    """
    Resolve theta_eff, c0, b, and sigma2_used for variogram models.

    Priority:
      1) if params is provided, derive theta from params and use params['c0'], params['b']
      2) else if theta is provided, infer c0 and b from theta by model_type
      3) else raise
    """
    if params is not None:
        theta_eff = _resolve_model_theta(theta, params, model_type, "variogram")
        c0 = float(params.get("c0", 0.0))
        b = float(params.get("b", 0.0))
        return theta_eff, c0, b, c0 + b

    if theta is None:
        raise ValueError(
            "For model_family='variogram', provide either params or theta."
        )

    theta_eff = list(np.asarray(theta, float).ravel())

    if model_type in ("spherical", "exponential", "gaussian", "cubic"):
        # (r, c0, b)
        c0 = float(theta_eff[1])
        b = float(theta_eff[2])

    elif model_type == "powered_exponential":
        # (r, c0, beta, b)
        c0 = float(theta_eff[1])
        b = float(theta_eff[3])

    elif model_type == "matern":
        # (r, c0, s, b)
        c0 = float(theta_eff[1])
        b = float(theta_eff[3])

    elif model_type in ("damped_cosine_angle", "angular_dissimilarity"):
        # (c, c0, b)
        c0 = float(theta_eff[1])
        b = float(theta_eff[2])

    else:
        raise ValueError(f"Unknown variogram model_type: {model_type}")

    return theta_eff, c0, b, c0 + b

def _resolve_custom_variogram_components(theta):
    """
    Resolve theta, c0, b, and sigma2 for custom variogram models.

    Convention
    ----------
    The last two entries of theta are assumed to be:
        theta[-2] = c0   (partial sill)
        theta[-1] = b    (nugget)

    Returns
    -------
    theta_eff : list[float]
        Full theta vector passed unchanged to the custom variogram kernel.
    c0 : float
        Partial sill.
    b : float
        Nugget.
    sigma2 : float
        Total sill = c0 + b.
    """
    theta_eff = list(np.asarray(theta, float).ravel())

    if len(theta_eff) < 3:
        raise ValueError(
            "Custom variogram theta must contain at least two parameters, "
            "with theta[-2]=c0 and theta[-1]=b."
        )

    c0 = float(theta_eff[-2])
    b = float(theta_eff[-1])
    sigma2 = c0 + b

    if not np.isfinite(c0) or c0 < 0.0:
        raise ValueError("Custom variogram c0 must be finite and nonnegative.")
    if not np.isfinite(b) or b < 0.0:
        raise ValueError("Custom variogram b must be finite and nonnegative.")
    if not np.isfinite(sigma2) or sigma2 < 0.0:
        raise ValueError("Custom variogram sigma2 = c0 + b must be finite and nonnegative.")

    return theta_eff, c0, b, sigma2

def build_covariance_nn_nt(
    coords: np.ndarray,
    targets: np.ndarray,
    *,
    model_family: str,
    model_type: str,
    theta: Optional[Sequence[float]] = None,
    params: Optional[Dict[str, float]] = None,
    sigma2: Optional[float] = None,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    jitter: float = 1e-10,
    chunk_cols: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build (C_nn, C_nt) from either a variogram or correlation kernel.

    Parameters
    ----------
    model_family : {'variogram', 'correlation'}
    model_type : str
        Name of the built-in model.
    theta : sequence of float, optional
        Callable-order parameter vector. Used only when `params` is not given.
    params : dict, optional
        Preferred parameter representation.
        - Variogram: should contain at least c0 and b, plus the model shape params
          (e.g., r, beta, s, c).
        - Correlation: may contain alpha directly, or c0 and b, plus the model
          shape params.
    sigma2 : float, optional
        Total variance scale for correlation models. If provided, it overrides the
        total sill implied by `params`. The parameter split from `params` is then
        interpreted only in relative terms.

    Returns
    -------
    C_nn : (n, n) ndarray
    C_nt : (n, m) ndarray
    sigma2_used : float
        Total variance placed on the diagonal.
    """

    D_nn, D_nt = pairwise_distances(
        coords, targets,
        distance_type=distance_type,
        projection=projection,
        chunk_cols=chunk_cols
    )

    if model_family == "variogram":
        if model_type not in VARIOGRAM_MODELS:
            raise ValueError(f"Unknown variogram model_type: {model_type}")
        model_fn = VARIOGRAM_MODELS[model_type]

        theta_eff, c0, b, sigma2_used = _resolve_variogram_components(
            theta=theta, params=params, model_type=model_type
        )

        G_nn = model_fn(D_nn, *theta_eff)
        G_nt = model_fn(D_nt, *theta_eff)

        C_nn = sigma2_used - G_nn
        C_nt = sigma2_used - G_nt
        np.fill_diagonal(C_nn, sigma2_used)

    elif model_family == "correlation":
        if model_type not in CORRELATION_MODELS:
            raise ValueError(f"Unknown correlation model_type: {model_type}")
        model_fn = CORRELATION_MODELS[model_type]

        theta_eff = _resolve_model_theta(theta, params, model_type, "correlation")

        if params is not None:
            c0, b, sigma2_used, alpha = _resolve_correlation_components(
                params=params, sigma2=sigma2
            )
        else:
            if model_type in ("spherical", "exponential", "gaussian", "cubic"):
                alpha = float(theta_eff[1])
            elif model_type == "powered_exponential":
                alpha = float(theta_eff[2])
            elif model_type == "matern":
                alpha = float(theta_eff[2])
            elif model_type in ("damped_cosine_angle", "angular_dissimilarity"):
                alpha = float(theta_eff[1])
            else:
                raise ValueError(f"Unknown correlation model_type: {model_type}")

            if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
                raise ValueError("alpha must be finite and lie in [0, 1].")

            sigma2_used = 1.0 if sigma2 is None else float(sigma2)
            if not np.isfinite(sigma2_used) or sigma2_used < 0.0:
                raise ValueError("sigma2 must be finite and nonnegative.")

            c0 = alpha * sigma2_used
            b = (1.0 - alpha) * sigma2_used

        # overwrite alpha in the callable theta vector so the resolved alpha is used
        if model_type in ("spherical", "exponential", "gaussian", "cubic"):
            theta_eff = [theta_eff[0], alpha]
        elif model_type == "powered_exponential":
            theta_eff = [theta_eff[0], theta_eff[1], alpha]
        elif model_type == "matern":
            theta_eff = [theta_eff[0], theta_eff[1], alpha]
        elif model_type in ("damped_cosine_angle", "angular_dissimilarity"):
            theta_eff = [theta_eff[0], alpha]
        else:
            raise ValueError(f"Unknown correlation model_type: {model_type}")

        R_nn = model_fn(D_nn, *theta_eff)
        R_nt = model_fn(D_nt, *theta_eff)

        C_nn = sigma2_used * R_nn
        C_nt = sigma2_used * R_nt
        np.fill_diagonal(C_nn, sigma2_used)

    else:
        raise ValueError("model_family must be 'variogram' or 'correlation'")

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
    blocks_nn: dict,
    blocks_nt: dict,
    custom_kernel,
    theta,
    jitter: float = 1e-10,
):
    """
    From custom distance blocks and a custom variogram kernel, build C_nn, C_nt.

    custom_kernel(blocks_nn, blocks_nt, theta) -> (G_nn, G_nt)
      - G = gamma(h) semivariogram values

    Convention
    ----------
    theta[-2] = c0   (partial sill)
    theta[-1] = b    (nugget)

    Covariance is formed as
        C(h) = (c0 + b) - gamma(h)

    Returns
    -------
    C_nn, C_nt, sigma2
    """
    theta_eff, c0, b, sigma2 = _resolve_custom_variogram_components(theta)

    G_nn, G_nt = custom_kernel(blocks_nn, blocks_nt, theta_eff)

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

# build covariance matrices for custom simulation
def build_blocks_all(blocks_nn, blocks_nt, blocks_tt):
    """
    Assemble full observation/target block matrices for custom SGS workflows.

    For each metric key, this stacks the observation-observation, observation-
    target, target-observation, and target-target blocks into a single full
    symmetric matrix

        [ OO  OT ]
        [ TO  TT ]

    where `TO = OT.T`.

    Parameters
    ----------
    blocks_nn : dict[str, ndarray]
        Observation-observation blocks, one array per metric.
    blocks_nt : dict[str, ndarray]
        Observation-target blocks, one array per metric.
    blocks_tt : dict[str, ndarray]
        Target-target blocks, one array per metric.

    Returns
    -------
    out : dict[str, ndarray]
        Full `(n_obs + n_targets, n_obs + n_targets)` block matrix for each
        metric key.

    Raises
    ------
    TypeError
        If any supplied block for a metric is itself a dictionary rather than
        a numeric array.
    """
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
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union
from itertools import combinations, combinations_with_replacement, product

import numpy as np
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statistics import NormalDist
from tqdm.auto import tqdm
from VarioCorreKrigE.krig_utils import (
    pairwise_distances,
    query_nearest_neighbors_balltree,
    _resolve_model_theta,
    _resolve_correlation_components,
    _resolve_variogram_components,
)
from VarioCorreKrigE.variofit import VARIOGRAM_MODELS
from VarioCorreKrigE.correfit import CORRELATION_MODELS

# --------------------------------------------------------------------------------------
# utilities
# --------------------------------------------------------------------------------------
def _compute_cokriging_distance_blocks(
    Xz: np.ndarray,
    Xy: np.ndarray,
    XT: np.ndarray,
    *,
    distance_type: str,
    projection: str,
    chunk_cols: Optional[int],
):
    """
    Compute the five distance blocks used by two-variable cokriging.

    Returns
    -------
    Dzz : (nz, nz) ndarray
        Distances among primary samples.
    Dyy : (ny, ny) ndarray
        Distances among secondary samples.
    Dzy : (nz, ny) ndarray
        Distances from primary to secondary samples.
    Dz0 : (nz, m) ndarray
        Distances from primary samples to targets.
    Dy0 : (ny, m) ndarray
        Distances from secondary samples to targets.
    """
    Dzz, _ = pairwise_distances(
        Xz, Xz,
        distance_type=distance_type,
        projection=projection,
        chunk_cols=chunk_cols,
    )
    Dyy, _ = pairwise_distances(
        Xy, Xy,
        distance_type=distance_type,
        projection=projection,
        chunk_cols=chunk_cols,
    )
    _, Dzy = pairwise_distances(
        Xz, Xy,
        distance_type=distance_type,
        projection=projection,
        chunk_cols=chunk_cols,
    )
    _, Dz0 = pairwise_distances(
        Xz, XT,
        distance_type=distance_type,
        projection=projection,
        chunk_cols=chunk_cols,
    )
    _, Dy0 = pairwise_distances(
        Xy, XT,
        distance_type=distance_type,
        projection=projection,
        chunk_cols=chunk_cols,
    )
    return Dzz, Dyy, Dzy, Dz0, Dy0

def _coerce_coordinates(X, distance_type: str, name: str) -> np.ndarray:
    """
    Coerce coordinates to the shapes expected by pairwise_distances(...) and
    BallTree helpers.

    Supported conventions
    ---------------------
    - geographic: (n, 2) geographic coordinate pairs in the ordering expected by
      pairwise_distances(...)
    - cartesian:  (n, 2) planar x/y coordinates
    - euclidean:  (n, d) general d-dimensional coordinates. If a 1D array is
      supplied, it is interpreted as (n, 1), i.e. n one-dimensional coordinates.
    - angular:    (n, 1) scalar angular coordinates

    Returns
    -------
    X : ndarray
        Coordinate array reshaped and validated for the requested distance_type.
    """
    X = np.asarray(X, float)

    if distance_type in ("geographic", "cartesian"):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(
                f"{name} must have shape (n, 2) for distance_type='{distance_type}'."
            )
        return X

    elif distance_type == "euclidean":
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"{name} must be a 2D array for distance_type='euclidean'.")
        return X

    elif distance_type == "angular":
        # store as column vector so downstream slicing stays uniform
        X = np.asarray(X, float).reshape(-1, 1)
        return X

    else:
        raise ValueError(
            "distance_type must be 'geographic', 'euclidean', 'angular', or 'cartesian'"
        )

def _build_rotation_matrix(rotation_matrix: dict) -> np.ndarray:
    """
    Build the planar anisotropy transform used by the cokriging routines.

    The returned matrix is applied by right multiplication:

        X_rot = X @ A

    for planar coordinate arrays X with shape (n, 2).

    This helper intentionally follows the anisotropy convention used in the
    reference cokriging example adopted by this module:

        A = diag(1 / a_max, 1 / a_min) @ R(azimuth)

    where R(azimuth) is the planar rotation matrix

        [[ cos(theta),  sin(theta)],
         [-sin(theta),  cos(theta)]]

    and theta is the azimuth in radians.

    Distances are computed after mapping coordinates into this transformed
    space, so larger values of a_max or a_min correspond to longer effective
    correlation ranges along those directions.

    Parameters
    ----------
    rotation_matrix : dict
        Dictionary with required keys:
            - 'azimuth' : float
                Rotation angle in degrees.
            - 'a_max' : float
                Positive scaling associated with the major axis.
            - 'a_min' : float
                Positive scaling associated with the minor axis.

    Returns
    -------
    A : (2, 2) ndarray
        Right-multiplication transform used by _apply_rotation_matrix(...).

    Notes
    -----
    The wording "rotate then scale" can be ambiguous depending on whether one
    thinks in terms of rotating points or rotating axes. This helper does not
    attempt to resolve that ambiguity abstractly; it implements the exact
    matrix convention used by the reference example and expected by the rest of
    this module.
    """
    if not isinstance(rotation_matrix, dict):
        raise ValueError(
            "rotation_matrix must be a dict with keys "
            "{'azimuth', 'a_max', 'a_min'}."
        )

    required = ("azimuth", "a_max", "a_min")
    missing = [k for k in required if k not in rotation_matrix]
    if missing:
        raise ValueError(
            f"rotation_matrix is missing required key(s): {missing}. "
            "Expected keys are {'azimuth', 'a_max', 'a_min'}."
        )

    azimuth = float(rotation_matrix["azimuth"])
    a_max = float(rotation_matrix["a_max"])
    a_min = float(rotation_matrix["a_min"])

    if not np.isfinite(azimuth):
        raise ValueError("rotation_matrix['azimuth'] must be finite.")
    if not np.isfinite(a_max) or a_max <= 0.0:
        raise ValueError("rotation_matrix['a_max'] must be finite and > 0.")
    if not np.isfinite(a_min) or a_min <= 0.0:
        raise ValueError("rotation_matrix['a_min'] must be finite and > 0.")

    theta = np.deg2rad(azimuth)

    R = np.array(
        [
            [np.cos(theta),  np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )
    S = np.array(
        [
            [1.0 / a_max, 0.0],
            [0.0, 1.0 / a_min],
        ],
        dtype=float,
    )

    # Match the established anisotropy convention:
    # X_rot = X @ A, with A = S @ R.
    A = np.dot(S, R)

    return A

def _make_rotated_direct_model_eval_spec(
    model_spec: dict,
    *,
    rotation_matrix: Optional[dict],
) -> dict:
    """
    Build the direct-model specification used during rotated covariance
    evaluation.

    Parameters
    ----------
    model_spec : dict
        Direct covariance-model specification.
    rotation_matrix : dict or None
        Optional planar anisotropy specification.

    Returns
    -------
    s_eval : dict
        Model specification to be used during covariance evaluation.

    Notes
    -----
    If a planar rotation/anisotropy transform is supplied, the direct model is
    interpreted in the same example-style anisotropic parameterization already
    used for rotated LMC structures, where the physical range is carried by
    (a_max, a_min). Therefore the scalar model range is forced to 1.0 during
    covariance evaluation to avoid double-counting the range.
    """
    if rotation_matrix is None:
        return model_spec

    if not isinstance(model_spec, dict):
        raise ValueError("model_spec must be a covariance-model specification dict.")

    s_eval = dict(model_spec)
    family = str(s_eval["model_family"]).lower().strip()

    if family not in ("variogram", "correlation"):
        raise ValueError(
            "Rotated direct covariance models must have model_family "
            "'variogram' or 'correlation'."
        )

    if s_eval.get("params", None) is not None:
        params = dict(s_eval["params"])
        range_key = _lmc_range_param_name(s_eval["model_type"])
        params[range_key] = 1.0
        s_eval["params"] = params
        return s_eval

    if s_eval.get("theta", None) is not None:
        theta = list(s_eval["theta"])
        if len(theta) < 1:
            raise ValueError(
                "Rotated direct model theta must contain at least the range parameter."
            )
        theta[0] = 1.0
        s_eval["theta"] = theta
        return s_eval

    raise ValueError(
        "Rotated direct models must define params or theta so the scalar range "
        "can be forced to 1.0 during covariance evaluation."
    )

def _apply_rotation_matrix(
    X: np.ndarray,
    *,
    rotation_matrix: Optional[dict],
    distance_type: str,
    name: str,
) -> np.ndarray:
    """
    Apply the rotated anisotropy transform to planar 2D coordinates.

    Notes
    -----
    - This is only supported for planar 2D coordinates.
    - Therefore it is only allowed for distance_type='euclidean' or 'cartesian'.
    - For geographic coordinates, first project the coordinates to planar x/y
      and then use this helper.
    """
    if rotation_matrix is None:
        return X

    if distance_type not in ("euclidean", "cartesian"):
        raise ValueError(
            "rotation_matrix is only supported for planar 2D coordinates with "
            "distance_type='euclidean' or 'cartesian'. For geographic data, "
            "first project lon/lat to planar x/y coordinates."
        )

    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(
            f"{name} must have shape (n, 2) when using rotation_matrix. "
            "This option is intended for planar x/y coordinates."
        )

    A = _build_rotation_matrix(rotation_matrix)
    return X @ A

def _compute_cokriging_distance_blocks_rotated(
    Xz: np.ndarray,
    Xy: np.ndarray,
    XT: np.ndarray,
    *,
    distance_type: str,
    projection: str,
    chunk_cols: Optional[int],
    rotation_matrix: Optional[dict] = None,
):
    """
    Compute the five cokriging distance blocks after applying an optional
    planar anisotropy transform.

    Parameters
    ----------
    Xz, Xy, XT : ndarray
        Primary coordinates, secondary coordinates, and target coordinates.
    rotation_matrix : dict or None, default None
        Optional planar anisotropy specification with keys
        {'azimuth', 'a_max', 'a_min'}.

    Returns
    -------
    Dzz, Dyy, Dzy, Dz0, Dy0 : ndarray
        Distance blocks among primary samples, among secondary samples,
        between primary and secondary samples, and from each sample set to
        the targets.

    Notes
    -----
    This helper is intended for covariance assembly only. If rotation_matrix
    is None, the coordinates are used unchanged.
    """

    Xz_use = _apply_rotation_matrix(
        Xz,
        rotation_matrix=rotation_matrix,
        distance_type=distance_type,
        name="primary_coords",
    )
    Xy_use = _apply_rotation_matrix(
        Xy,
        rotation_matrix=rotation_matrix,
        distance_type=distance_type,
        name="secondary_coords",
    )
    XT_use = _apply_rotation_matrix(
        XT,
        rotation_matrix=rotation_matrix,
        distance_type=distance_type,
        name="targets",
    )

    return _compute_cokriging_distance_blocks(
        Xz_use,
        Xy_use,
        XT_use,
        distance_type=distance_type,
        projection=projection,
        chunk_cols=chunk_cols,
    )

def _resolve_lmc_rotation_matrices(
    rotation_matrix: Optional[dict],
    n_structures: int,
):
    """
    Resolve the top-level LMC rotation specification into one entry per structure.

    Supported forms
    ---------------
    None
        No anisotropy transform for any structure.

    {0: {...}, 1: {...}, ...}
        Zero-based structure-indexed rotations.

    {1: {...}, 2: {...}, ...}
        One-based structure-indexed rotations.

    Notes
    -----
    A single global rotation dict like {'azimuth': ..., 'a_max': ..., 'a_min': ...}
    is intentionally NOT accepted for covariance_mode='lmc'. In this revised
    workflow, LMC anisotropy must be specified per structure.
    """
    if rotation_matrix is None:
        return [None] * int(n_structures)

    if not isinstance(rotation_matrix, dict):
        raise ValueError(
            "For covariance_mode='lmc', rotation_matrix must be None or a dict "
            "keyed by structure index, e.g. {0: {...}, 1: {...}}."
        )

    # Reject direct-mode/global style input
    if {"azimuth", "a_max", "a_min"}.issubset(rotation_matrix.keys()):
        raise ValueError(
            "For covariance_mode='lmc', rotation_matrix must be specified per "
            "structure, e.g. {0: {...}, 1: {...}}. A single global rotation "
            "dict is not supported."
        )

    # Allow int keys or stringified ints
    try:
        rot_map = {int(k): v for k, v in rotation_matrix.items()}
    except Exception as e:
        raise ValueError(
            "For covariance_mode='lmc', rotation_matrix keys must be structure "
            "indices (0..L-1 or 1..L)."
        ) from e

    expected0 = set(range(n_structures))
    expected1 = set(range(1, n_structures + 1))
    got = set(rot_map.keys())

    if got == expected0:
        offset = 0
    elif got == expected1:
        offset = 1
    else:
        raise ValueError(
            "For covariance_mode='lmc', rotation_matrix must contain one entry "
            f"for every structure. Expected keys {sorted(expected0)} or "
            f"{sorted(expected1)}, got {sorted(got)}."
        )

    rot_list = []
    for i in range(n_structures):
        rot_i = rot_map[i + offset]
        _ = _build_rotation_matrix(rot_i)  # validate
        rot_list.append(rot_i)

    return rot_list


def _make_rotated_lmc_structure_eval_spec(
    structure: dict,
    *,
    rotation_matrix: Optional[dict],
) -> dict:
    """
    Build the structure specification used during rotated LMC covariance assembly.

    Parameters
    ----------
    structure : dict
        Original LMC structure specification.
    rotation_matrix : dict or None
        Optional per-structure anisotropy specification.

    Returns
    -------
    s_eval : dict
        Structure specification to be used during covariance evaluation.

    Notes
    -----
    If a per-structure rotation is supplied, the structure is interpreted in the
    example-style anisotropic parameterization where the physical range is carried
    by (a_max, a_min). Therefore the scalar model range is forced to 1.0 during
    covariance evaluation to avoid double-counting the range.
    """
    if rotation_matrix is None:
        return structure

    if structure["model_family"] == "nugget":
        return structure

    s_eval = dict(structure)

    if s_eval.get("params", None) is not None:
        params = dict(s_eval["params"])
        range_key = _lmc_range_param_name(s_eval["model_type"])
        params[range_key] = 1.0
        s_eval["params"] = params
        return s_eval

    if s_eval.get("theta", None) is not None:
        theta = list(s_eval["theta"])
        if len(theta) < 1:
            raise ValueError(
                "Rotated LMC structure theta must contain at least the range parameter."
            )
        theta[0] = 1.0
        s_eval["theta"] = theta
        return s_eval

    raise ValueError(
        "Rotated LMC structures must define params or theta so the scalar range "
        "can be forced to 1.0 during covariance assembly."
    )

def _clean_values_and_coords(values, coords, *, distance_type: str, name: str):
    """
    Remove rows with non-finite values or non-finite coordinates.
    """
    z = np.asarray(values, float).ravel()
    X = _coerce_coordinates(coords, distance_type=distance_type, name=name)

    if z.size != X.shape[0]:
        raise ValueError(f"{name}: values and coords must have matching lengths.")

    mask = np.isfinite(z) & np.all(np.isfinite(X), axis=1)
    z = z[mask]
    X = X[mask]

    if z.size == 0:
        raise ValueError(f"{name}: no finite samples remain after filtering.")

    return z, X


def _resolve_simple_mean(mean, values, *, var_name: str) -> float:
    """
    Resolve the known mean used by simple cokriging.

    Supported:
      - numeric value
      - 'zero'
      - 'sample'
    """
    if isinstance(mean, (int, float)):
        return float(mean)

    if mean == "zero":
        return 0.0

    if mean == "sample":
        return float(np.mean(values))

    raise ValueError(
        f"{var_name}: mean must be a numeric value, 'zero', or 'sample'."
    )


def _compute_standardization_stats(values, *, name: str):
    """
    Compute sample mean and sample standard deviation after removing non-finite
    values.

    Returns
    -------
    mu : float
        Sample mean.
    sd : float
        Sample standard deviation with ddof=1.
    """
    v = np.asarray(values, float).ravel()
    v = v[np.isfinite(v)]

    if v.size == 0:
        raise ValueError(f"{name}: no finite values available for standardization.")

    mu = float(np.mean(v))
    sd = float(np.std(v, ddof=1))

    if not np.isfinite(sd) or sd <= 0.0:
        raise ValueError(
            f"{name}: standard deviation must be finite and > 0 for standardize=True."
        )

    return mu, sd


def _standardize_values(z: np.ndarray, *, name: str):
    """
    z_std = (z - mean) / std
    """
    mu, sd = _compute_standardization_stats(z, name=name)
    z_std = (np.asarray(z, float) - mu) / sd
    return z_std, mu, sd


def estimate_collocated_correlation(
    primary_values,
    secondary_values,
    *,
    method: str = "pearson",
    return_summary: bool = False,
    alpha: float = 0.05,
    ci_method: str = "auto",
    cluster_ids=None,
    n_boot: int = 5000,
    random_state: Optional[int] = None,
    return_boot_samples: bool = False,
):
    """
    Estimate the collocated correlation rho0 from paired primary/secondary values,
    optionally with an uncertainty interval.

    This helper assumes the inputs are already aligned as colocated pairs.
    It is mainly intended for later SCCK / ICCK Markov workflows, but it is
    also useful when a transformed cross-variogram model is used as a
    unit-zero-lag cross-shape and needs a separate cross_scale=rho0.

    Parameters
    ----------
    primary_values, secondary_values : array_like, shape (n,)
        Paired values at colocated locations.
    method : {'pearson', 'uncentered'}, default 'pearson'
        Correlation estimator.

        - 'pearson'    : centered Pearson correlation.
        - 'uncentered' : cosine-style uncentered correlation.
    return_summary : bool, default False
        If False, return only rho_hat as a float (backward-compatible behavior).
        If True, return a dictionary containing rho_hat, rho_lo, rho_hi, and
        metadata.
    alpha : float, default 0.05
        Two-sided interval level. For example alpha=0.05 gives a 95% interval.
        Used only when return_summary=True.
    ci_method : {'auto', 'fisher', 'bootstrap', 'cluster_bootstrap', 'none'}, default 'auto'
        Method used to quantify uncertainty when return_summary=True.

        - 'auto':
            * Pearson + no cluster_ids -> Fisher z interval
            * Pearson + cluster_ids    -> cluster bootstrap
            * Uncentered               -> bootstrap or cluster bootstrap
        - 'fisher':
            Allowed only for method='pearson' and no cluster_ids.
        - 'bootstrap':
            Row-wise paired bootstrap.
        - 'cluster_bootstrap':
            Bootstrap entire clusters defined by cluster_ids.
        - 'none':
            Return rho_hat with rho_lo/rho_hi=None.
    cluster_ids : array_like or None, default None
        Optional cluster labels for cluster-aware uncertainty estimation.
        Must have the same original length as primary_values and
        secondary_values. Non-finite paired observations are filtered
        internally, and the same mask is then applied to cluster_ids.
        Typical choices are event IDs, site IDs, or investigation-cluster IDs.
    n_boot : int, default 5000
        Number of bootstrap replicates when bootstrap-based uncertainty is used.
    random_state : int or None, default None
        Seed for reproducible bootstrap intervals.
    return_boot_samples : bool, default False
        If True and a bootstrap-based method is used, include the bootstrap
        sample array under key 'boot_samples' in the returned dictionary.
        Degenerate bootstrap replicates that make the estimator undefined are
        stored as NaN and ignored when forming the interval.

    Returns
    -------
    rho0 : float
        Returned when return_summary=False. This preserves the original API.
    out : dict
        Returned when return_summary=True, with keys:
            - 'rho_hat' : float
            - 'rho_lo' : float or None
            - 'rho_hi' : float or None
            - 'method' : str
            - 'ci_method' : str
            - 'n_pairs' : int
            - 'alpha' : float
            - 'n_clusters' : int or None
            - 'se_fisher_z' : float or None
            - 'boot_samples' : ndarray, optional

    Notes
    -----
    - Fisher z uncertainty is appropriate only for Pearson correlation and
      approximately independent pairs.
    - For clustered or dependent data, cluster bootstrap is generally more
      defensible.
    - For method='uncentered', Fisher z is not used; bootstrap-based intervals
      are preferred.
    """
    z = np.asarray(primary_values, float).ravel()
    y = np.asarray(secondary_values, float).ravel()

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)

    mask = np.isfinite(z) & np.isfinite(y)
    z = z[mask]
    y = y[mask]

    if cluster_ids is not None:
        if cluster_ids.shape[0] != mask.shape[0]:
            raise ValueError(
                "cluster_ids must have the same original length as the paired inputs."
            )
        cluster_ids = cluster_ids[mask]

    n = z.size
    if n < 2:
        raise ValueError("Need at least 2 finite colocated pairs to estimate rho0.")

    method = str(method).lower().strip()
    if method not in ("pearson", "uncentered"):
        raise ValueError("method must be 'pearson' or 'uncentered'.")

    def _estimate(x, yy):
        if method == "pearson":
            if np.all(x == x[0]) or np.all(yy == yy[0]):
                raise ValueError("Pearson correlation is undefined for constant input.")
            return float(np.corrcoef(x, yy)[0, 1])

        denom = np.sqrt(np.sum(x**2) * np.sum(yy**2))
        if denom <= 0.0:
            raise ValueError("Uncentered correlation is undefined when a norm is zero.")
        return float(np.sum(x * yy) / denom)

    rho_hat = _estimate(z, y)

    if not return_summary:
        return rho_hat

    if not np.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be finite and lie strictly between 0 and 1.")

    ci_method = str(ci_method).lower().strip()
    allowed_ci = ("auto", "fisher", "bootstrap", "cluster_bootstrap", "none")
    if ci_method not in allowed_ci:
        raise ValueError(
            "ci_method must be one of "
            "{'auto', 'fisher', 'bootstrap', 'cluster_bootstrap', 'none'}."
        )

    if not isinstance(n_boot, (int, np.integer)) or int(n_boot) <= 0:
        raise ValueError("n_boot must be a positive integer.")
    n_boot = int(n_boot)

    n_clusters = None
    if cluster_ids is not None:
        n_clusters = int(np.unique(cluster_ids).size)

    # ------------------------------------------------------------------
    # resolve uncertainty method
    # ------------------------------------------------------------------
    if ci_method == "auto":
        if method == "pearson":
            ci_method_used = "cluster_bootstrap" if cluster_ids is not None else "fisher"
        else:
            ci_method_used = "cluster_bootstrap" if cluster_ids is not None else "bootstrap"
    else:
        ci_method_used = ci_method

    if ci_method_used == "fisher":
        if method != "pearson":
            raise ValueError("ci_method='fisher' is only valid for method='pearson'.")
        if cluster_ids is not None:
            raise ValueError(
                "ci_method='fisher' assumes approximately independent pairs; "
                "do not combine it with cluster_ids."
            )
        if n < 4:
            raise ValueError(
                "Fisher-z interval requires at least 4 finite paired observations."
            )

    if ci_method_used == "cluster_bootstrap" and cluster_ids is None:
        raise ValueError(
            "ci_method='cluster_bootstrap' requires cluster_ids."
        )

    # ------------------------------------------------------------------
    # uncertainty calculation
    # ------------------------------------------------------------------
    rho_lo = None
    rho_hi = None
    se_fisher_z = None
    boot_samples = None

    if ci_method_used == "none":
        pass

    elif ci_method_used == "fisher":
        # 95% normal critical via inverse CDF of standard normal
        rho_clip = np.clip(rho_hat, -1.0 + 1e-15, 1.0 - 1e-15)
        fisher_z = np.arctanh(rho_clip)
        se_fisher_z = float(1.0 / np.sqrt(n - 3.0))
        zcrit = float(NormalDist().inv_cdf(1.0 - alpha / 2.0))

        z_lo = fisher_z - zcrit * se_fisher_z
        z_hi = fisher_z + zcrit * se_fisher_z

        rho_lo = float(np.tanh(z_lo))
        rho_hi = float(np.tanh(z_hi))

    elif ci_method_used in ("bootstrap", "cluster_bootstrap"):
        rng = np.random.default_rng(random_state)
        boot_samples = np.full(n_boot, np.nan, dtype=float)

        if ci_method_used == "bootstrap":
            idx_all = np.arange(n)
            for b in range(n_boot):
                idx = rng.choice(idx_all, size=n, replace=True)
                try:
                    boot_samples[b] = _estimate(z[idx], y[idx])
                except ValueError:
                    # Degenerate resample (for example constant input in a
                    # Pearson resample); leave as NaN and drop later.
                    pass

        else:
            unique_clusters = np.unique(cluster_ids)
            G = unique_clusters.size
            if G < 2:
                raise ValueError(
                    "cluster_bootstrap requires at least 2 unique clusters."
                )

            rows_by_cluster = {
                g: np.where(cluster_ids == g)[0]
                for g in unique_clusters
            }

            for b in range(n_boot):
                sampled_clusters = rng.choice(unique_clusters, size=G, replace=True)
                idx = np.concatenate([rows_by_cluster[g] for g in sampled_clusters])
                try:
                    boot_samples[b] = _estimate(z[idx], y[idx])
                except ValueError:
                    # Degenerate cluster resample; leave as NaN and drop later.
                    pass

        n_valid_boot = int(np.isfinite(boot_samples).sum())
        if n_valid_boot < 2:
            raise ValueError(
                "Too few valid bootstrap replicates were obtained to form an interval. "
                "This usually indicates degenerate paired data or extremely small "
                "effective sample size."
            )

        rho_lo = float(np.nanquantile(boot_samples, alpha / 2.0))
        rho_hi = float(np.nanquantile(boot_samples, 1.0 - alpha / 2.0))

    else:
        raise RuntimeError(f"Unhandled ci_method_used='{ci_method_used}'.")

    out = {
        "rho_hat": float(rho_hat),
        "rho_lo": rho_lo,
        "rho_hi": rho_hi,
        "method": method,
        "ci_method": ci_method_used,
        "n_pairs": int(n),
        "alpha": float(alpha),
        "n_clusters": n_clusters,
        "se_fisher_z": se_fisher_z,
    }

    if return_boot_samples and boot_samples is not None:
        out["boot_samples"] = boot_samples

    return out

def _resolve_rho0(
    rho0: Optional[float] = None,
    *,
    primary_values_for_rho0=None,
    secondary_values_for_rho0=None,
    method: str = "pearson",
) -> float:
    """
    Resolve the collocated correlation rho0 used by SCCK / ICCK Markov models.

    Parameters
    ----------
    rho0 : float or None, default None
        Explicit collocated correlation.
    primary_values_for_rho0, secondary_values_for_rho0 : array_like or None
        Paired colocated primary/secondary values used to estimate rho0 when
        rho0 is not supplied directly.
    method : {'pearson', 'uncentered'}, default 'pearson'
        Estimator passed through to estimate_collocated_correlation(...).

    Returns
    -------
    rho0 : float
        Finite collocated zero-lag correlation.
        For method='pearson', affine rescaling does not change the value.
        For method='uncentered', inputs should already be centered/standardized
        (for example on z-score or normal-score scale).
    """
    if rho0 is None:
        if primary_values_for_rho0 is None or secondary_values_for_rho0 is None:
            raise ValueError(
                "Provide either rho0 directly or paired "
                "primary_values_for_rho0 / secondary_values_for_rho0."
            )
        rho0 = estimate_collocated_correlation(
            primary_values_for_rho0,
            secondary_values_for_rho0,
            method=method,
        )

    rho0 = float(rho0)
    if not np.isfinite(rho0) or abs(rho0) > 1.0:
        raise ValueError("rho0 must be finite and lie in [-1, 1].")

    return rho0

def make_covmodel_spec(
    *,
    model_family: str,
    model_type: str,
    params: Optional[dict] = None,
    theta: Optional[Sequence[float]] = None,
    sigma2: Optional[float] = None,
    cross_scale: Optional[float] = None,
) -> dict:
    """
    Convenience packer for covariance-model specifications used by the cokriging
    routines in this module.

    Notes
    -----
    - For direct models built from transformed variofit(..., transform='correlation'),
      use:
          model_family='variogram'
          params=<normalized variofit params with c0+b=1>
      and leave cross_scale=None.

    - For a cross model built from transformed crossvariofit(..., transform='correlation'),
      the returned params describe a unit-zero-lag cross-shape. In that case pass:
          cross_scale=rho0
      so the actual cross-covariance becomes rho0 * shape(h).

    - For a raw cross-variogram fit (not normalized), leave cross_scale=None and
      let the fitted c0+b carry the cross-covariance amplitude directly.
    """
    return {
        "model_family": model_family,
        "model_type": model_type,
        "params": params,
        "theta": theta,
        "sigma2": sigma2,
        "cross_scale": cross_scale,
    }


# --------------------------------------------------------------------------------------
# Model evaluation on arbitrary distance blocks
# --------------------------------------------------------------------------------------

def _resolve_correlation_theta_and_sigma(
    model_type: str,
    *,
    theta: Optional[Sequence[float]] = None,
    params: Optional[dict] = None,
    sigma2: Optional[float] = None,
):
    """
    Resolve the callable-order correlation parameters and variance scale for an
    arbitrary distance block.

    Parameters
    ----------
    model_type : str
        Correlation model name.
    theta : sequence of float or None, default None
        Explicit parameter vector in callable order.
    params : dict or None, default None
        Named model parameters.
    sigma2 : float or None, default None
        Optional covariance scale override.

    Returns
    -------
    theta_eff : list
        Parameter vector in the callable order expected by CORRELATION_MODELS.
    sigma2_used : float
        Variance scale used when evaluating the covariance block.

    Notes
    -----
    This helper mirrors the correlation parameter handling already used in
    krig_utils.build_covariance_nn_nt(...).
    """
    theta_eff = _resolve_model_theta(theta, params, model_type, "correlation")

    if params is not None:
        _, _, sigma2_used, alpha = _resolve_correlation_components(
            params=params,
            sigma2=sigma2,
        )
    else:
        # match the current conventions already used in krig_utils.build_covariance_nn_nt
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

    # overwrite alpha in theta_eff exactly the same way as krig_utils
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

    return theta_eff, float(sigma2_used)


def _covariance_from_distances(
    D: np.ndarray,
    model_spec: dict,
    *,
    set_diagonal: bool = False,
    jitter: float = 1e-10,
) -> Tuple[np.ndarray, float]:
    """
    Evaluate a covariance block on a supplied distance matrix.

    Parameters
    ----------
    D : ndarray
        Distance matrix.
    model_spec : dict
        Output of make_covmodel_spec(...) or equivalent.
    set_diagonal : bool, default False
        If True and D is square, force the diagonal to the model variance and
        add jitter * variance.
    jitter : float, default 1e-10
        Diagonal stabilization factor. Used only when set_diagonal=True.

    Returns
    -------
    C : ndarray
        Covariance block.
    sigma2_used : float
        Base variance/covariance scale before any optional cross_scale.
    """
    D = np.asarray(D, float)

    family = model_spec["model_family"]
    model_type = model_spec["model_type"]
    params = model_spec.get("params", None)
    theta = model_spec.get("theta", None)
    sigma2 = model_spec.get("sigma2", None)
    cross_scale = model_spec.get("cross_scale", None)

    if family == "variogram":
        if model_type not in VARIOGRAM_MODELS:
            raise ValueError(f"Unknown variogram model_type: {model_type}")

        model_fn = VARIOGRAM_MODELS[model_type]
        theta_eff, _, _, sigma2_used = _resolve_variogram_components(
            theta=theta,
            params=params,
            model_type=model_type,
        )

        G = model_fn(D, *theta_eff)
        C = sigma2_used - G

    elif family == "correlation":
        if model_type not in CORRELATION_MODELS:
            raise ValueError(f"Unknown correlation model_type: {model_type}")

        model_fn = CORRELATION_MODELS[model_type]
        theta_eff, sigma2_used = _resolve_correlation_theta_and_sigma(
            model_type,
            theta=theta,
            params=params,
            sigma2=sigma2,
        )

        R = model_fn(D, *theta_eff)
        C = sigma2_used * R

    else:
        raise ValueError("model_family must be 'variogram' or 'correlation'.")

    # Optional external scale for cross-covariance shapes
    if cross_scale is not None:
        C = float(cross_scale) * C

    if set_diagonal:
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError("set_diagonal=True requires a square distance matrix.")

        diag_val = float(cross_scale) * float(sigma2_used) if cross_scale is not None else float(sigma2_used)
        ii = np.diag_indices(D.shape[0])
        C[ii] = diag_val
        C[ii] += jitter * max(abs(diag_val), 1.0)

    return C, float(sigma2_used)

def _unit_covariance_from_distances(
    D: np.ndarray,
    model_spec: dict,
) -> np.ndarray:
    """
    Evaluate a unit-variance covariance/correlation shape from a direct model.

    Notes
    -----
    This helper is used whenever a direct covariance model must be moved onto a
    unit-variance correlation scale, for example in the collocated Markov-model
    routines and other normalized-shape calculations.
    """
    C, sigma2_used = _covariance_from_distances(
        D,
        model_spec,
        set_diagonal=False,
        jitter=0.0,
    )

    if not np.isfinite(sigma2_used) or sigma2_used <= 0.0:
        raise ValueError(
            "Direct covariance model must imply a positive finite zero-lag variance "
            "for SCCK / ICCK."
        )

    return np.asarray(C, float) / float(sigma2_used)

def _unit_lmc_direct_covariance_from_distances(
    D: np.ndarray,
    structures: Sequence[dict],
    *,
    component: str,
) -> np.ndarray:
    """
    Evaluate a unit-variance direct covariance shape from an LMC.

    Parameters
    ----------
    component : {'primary', 'secondary'}
        Which direct covariance to evaluate.
    """
    D = np.asarray(D, float)

    if component not in ("primary", "secondary"):
        raise ValueError("component must be 'primary' or 'secondary'.")

    idx = 0 if component == "primary" else 1

    C = np.zeros_like(D, dtype=float)
    sigma2 = 0.0

    for s in structures:
        B = _validate_lmc_coregionalization_matrix(s["B"])
        R = _unit_structure_covariance_from_distances(D, s)
        C += B[idx, idx] * R
        sigma2 += B[idx, idx]

    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError(
            f"LMC {component} variance must be positive."
        )

    return C / float(sigma2)

def _evaluate_direct_unit_shape(
    D: np.ndarray,
    *,
    covariance_mode: str,
    primary_model: Optional[dict] = None,
    secondary_model: Optional[dict] = None,
    structures: Optional[Sequence[dict]] = None,
    component: str,
) -> np.ndarray:
    """
    Evaluate a unit direct covariance shape for the requested variable.

    Parameters
    ----------
    component : {'primary', 'secondary'}
        Which direct covariance shape to evaluate.
    """
    if covariance_mode == "direct":
        if component == "primary":
            if primary_model is None:
                raise ValueError("primary_model is required.")
            return _unit_covariance_from_distances(D, primary_model)
        elif component == "secondary":
            if secondary_model is None:
                raise ValueError("secondary_model is required.")
            return _unit_covariance_from_distances(D, secondary_model)
        else:
            raise ValueError("component must be 'primary' or 'secondary'.")

    elif covariance_mode == "lmc":
        if structures is None:
            raise ValueError("structures are required for covariance_mode='lmc'.")
        return _unit_lmc_direct_covariance_from_distances(
            D,
            structures,
            component=component,
        )

    else:
        raise ValueError("covariance_mode must be 'direct' or 'lmc'.")

def _evaluate_direct_unit_shape_from_coords(
    Xa: np.ndarray,
    Xb: np.ndarray,
    *,
    covariance_mode: str,
    component: str,
    primary_model: Optional[dict] = None,
    secondary_model: Optional[dict] = None,
    structures: Optional[Sequence[dict]] = None,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    chunk_cols: Optional[int] = None,
    same_coordinates: bool = False,
) -> np.ndarray:
    """
    Evaluate a unit direct covariance/correlation shape between two coordinate sets.

    Parameters
    ----------
    Xa, Xb : ndarray
        Coordinate arrays.
    covariance_mode : {'direct', 'lmc'}
        Direct covariance mode or LMC mode.
    component : {'primary', 'secondary'}
        Which direct covariance shape to evaluate.
    primary_model, secondary_model : dict or None
        Direct covariance specifications used when covariance_mode='direct'.
    structures : sequence of dict or None
        LMC structures used when covariance_mode='lmc'.
    distance_type : {'geographic', 'euclidean', 'cartesian', 'angular'}
        Distance convention passed to pairwise_distances(...).
    projection : str, default 'WGS84'
        Projection / ellipsoid argument passed through where relevant.
    rotation_matrix : dict or None
        Optional anisotropy specification. In direct mode this is a single
        planar rotation dict. In LMC mode it may be a per-structure mapping.
    chunk_cols : int or None, default None
        Optional chunk size passed to pairwise_distances(...).
    same_coordinates : bool, default False
        If True, treat Xa and Xb as the same sample set and request the square
        distance block. Otherwise request the cross-distance block.

    Returns
    -------
    R : ndarray
        Unit-variance direct covariance/correlation block.

    Notes
    -----
    This helper normalizes direct covariance models to unit variance so the
    downstream covariance algebra is carried out on the correlation scale.

    When rotation_matrix is supplied in direct mode, the rotated evaluation uses
    the example-style anisotropic parameterization in which (a_max, a_min)
    carry the physical directional ranges. Therefore the scalar model range is
    forced to 1.0 during covariance evaluation to avoid double-counting the
    range.
    """
    Xa = _coerce_coordinates(Xa, distance_type, "Xa")
    Xb = _coerce_coordinates(Xb, distance_type, "Xb")

    covariance_mode = str(covariance_mode).lower().strip()
    if component not in ("primary", "secondary"):
        raise ValueError("component must be 'primary' or 'secondary'.")

    def _get_distance_block(Xa_use, Xb_use):
        if same_coordinates:
            D, _ = pairwise_distances(
                Xa_use, Xb_use,
                distance_type=distance_type,
                projection=projection,
                chunk_cols=chunk_cols,
            )
        else:
            _, D = pairwise_distances(
                Xa_use, Xb_use,
                distance_type=distance_type,
                projection=projection,
                chunk_cols=chunk_cols,
            )
        return D

    if covariance_mode == "direct":
        Xa_use = _apply_rotation_matrix(
            Xa,
            rotation_matrix=rotation_matrix,
            distance_type=distance_type,
            name="Xa",
        )
        Xb_use = _apply_rotation_matrix(
            Xb,
            rotation_matrix=rotation_matrix,
            distance_type=distance_type,
            name="Xb",
        )

        D = _get_distance_block(Xa_use, Xb_use)

        if component == "primary":
            if primary_model is None:
                raise ValueError("primary_model is required.")
            primary_model_eval = _make_rotated_direct_model_eval_spec(
                primary_model,
                rotation_matrix=rotation_matrix,
            )
            return _unit_covariance_from_distances(D, primary_model_eval)

        if secondary_model is None:
            raise ValueError("secondary_model is required.")
        secondary_model_eval = _make_rotated_direct_model_eval_spec(
            secondary_model,
            rotation_matrix=rotation_matrix,
        )
        return _unit_covariance_from_distances(D, secondary_model_eval)

    if covariance_mode != "lmc":
        raise ValueError("covariance_mode must be 'direct' or 'lmc'.")

    if structures is None:
        raise ValueError("structures are required for covariance_mode='lmc'.")

    if rotation_matrix is None:
        D = _get_distance_block(Xa, Xb)
        return _unit_lmc_direct_covariance_from_distances(
            D,
            structures,
            component=component,
        )

    idx = 0 if component == "primary" else 1
    rot_list = _resolve_lmc_rotation_matrices(rotation_matrix, len(structures))

    C = np.zeros((Xa.shape[0], Xb.shape[0]), dtype=float)
    sigma2 = 0.0
    distance_cache = {}

    def _rotation_cache_key(rot):
        if rot is None:
            return None
        return (
            float(rot["azimuth"]),
            float(rot["a_max"]),
            float(rot["a_min"]),
        )

    for s, rot_use in zip(structures, rot_list):
        B = _validate_lmc_coregionalization_matrix(s["B"])
        sigma2 += B[idx, idx]

        s_eval = _make_rotated_lmc_structure_eval_spec(
            s,
            rotation_matrix=rot_use,
        )

        rot_key = _rotation_cache_key(rot_use)
        if rot_key not in distance_cache:
            Xa_use = _apply_rotation_matrix(
                Xa,
                rotation_matrix=rot_use,
                distance_type=distance_type,
                name="Xa",
            )
            Xb_use = _apply_rotation_matrix(
                Xb,
                rotation_matrix=rot_use,
                distance_type=distance_type,
                name="Xb",
            )
            distance_cache[rot_key] = _get_distance_block(Xa_use, Xb_use)

        D = distance_cache[rot_key]
        R = _unit_structure_covariance_from_distances(D, s_eval)
        C += B[idx, idx] * R

    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError(
            f"LMC {component} variance must be positive."
        )

    return C / float(sigma2)
# --------------------------------------------------------------------------------------
# SCK block assembly and solve
# --------------------------------------------------------------------------------------

def _assemble_sck_conditioning_matrix(
    primary_coords: np.ndarray,
    secondary_coords: np.ndarray,
    *,
    primary_model: dict,
    secondary_model: dict,
    cross_model: dict,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    jitter: float = 1e-10,
    chunk_cols: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Assemble only the conditioning matrix

        K = [[Czz, Czy],
             [Cyz, Cyy]]

    for two-variable cokriging.

    Notes
    -----
    This function is intentionally separated from target-side RHS assembly so
    the Cholesky factorization of K can be reused across target batches.
    """
    Xz = _coerce_coordinates(primary_coords, distance_type, "primary_coords")
    Xy = _coerce_coordinates(secondary_coords, distance_type, "secondary_coords")

    Xz_use = _apply_rotation_matrix(
        Xz,
        rotation_matrix=rotation_matrix,
        distance_type=distance_type,
        name="primary_coords",
    )
    Xy_use = _apply_rotation_matrix(
        Xy,
        rotation_matrix=rotation_matrix,
        distance_type=distance_type,
        name="secondary_coords",
    )

    primary_model_eval = _make_rotated_direct_model_eval_spec(
        primary_model,
        rotation_matrix=rotation_matrix,
    )
    secondary_model_eval = _make_rotated_direct_model_eval_spec(
        secondary_model,
        rotation_matrix=rotation_matrix,
    )
    cross_model_eval = _make_rotated_direct_model_eval_spec(
        cross_model,
        rotation_matrix=rotation_matrix,
    )

    nz = Xz.shape[0]
    ny = Xy.shape[0]

    K = np.empty((nz + ny, nz + ny), dtype=float)

    # Czz
    Dzz, _ = pairwise_distances(
        Xz_use, Xz_use,
        distance_type=distance_type,
        projection=projection,
        chunk_cols=chunk_cols,
    )
    K[:nz, :nz], sigma2_primary = _covariance_from_distances(
        Dzz, primary_model_eval, set_diagonal=True, jitter=jitter
    )
    del Dzz

    # Cyy
    Dyy, _ = pairwise_distances(
        Xy_use, Xy_use,
        distance_type=distance_type,
        projection=projection,
        chunk_cols=chunk_cols,
    )
    K[nz:, nz:], _ = _covariance_from_distances(
        Dyy, secondary_model_eval, set_diagonal=True, jitter=jitter
    )
    del Dyy

    # Czy / Cyz
    _, Dzy = pairwise_distances(
        Xz_use, Xy_use,
        distance_type=distance_type,
        projection=projection,
        chunk_cols=chunk_cols,
    )
    Czy, _ = _covariance_from_distances(
        Dzy, cross_model_eval, set_diagonal=False, jitter=jitter
    )
    K[:nz, nz:] = Czy
    K[nz:, :nz] = Czy.T
    del Dzy, Czy

    return K, float(sigma2_primary)


def _assemble_sck_rhs(
    primary_coords: np.ndarray,
    secondary_coords: np.ndarray,
    targets: np.ndarray,
    *,
    primary_model: dict,
    cross_model: dict,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    chunk_cols: Optional[int] = None,
) -> np.ndarray:
    """
    Assemble only the target-side RHS block

        k0 = [cz0,
              cy0]

    for one or more targets.
    """
    Xz = _coerce_coordinates(primary_coords, distance_type, "primary_coords")
    Xy = _coerce_coordinates(secondary_coords, distance_type, "secondary_coords")
    XT = _coerce_coordinates(targets, distance_type, "targets")

    Xz_use = _apply_rotation_matrix(
        Xz,
        rotation_matrix=rotation_matrix,
        distance_type=distance_type,
        name="primary_coords",
    )
    Xy_use = _apply_rotation_matrix(
        Xy,
        rotation_matrix=rotation_matrix,
        distance_type=distance_type,
        name="secondary_coords",
    )
    XT_use = _apply_rotation_matrix(
        XT,
        rotation_matrix=rotation_matrix,
        distance_type=distance_type,
        name="targets",
    )

    primary_model_eval = _make_rotated_direct_model_eval_spec(
        primary_model,
        rotation_matrix=rotation_matrix,
    )
    cross_model_eval = _make_rotated_direct_model_eval_spec(
        cross_model,
        rotation_matrix=rotation_matrix,
    )

    nz = Xz.shape[0]
    ny = Xy.shape[0]
    m = XT.shape[0]

    k0 = np.empty((nz + ny, m), dtype=float)

    # cz0
    _, Dz0 = pairwise_distances(
        Xz_use, XT_use,
        distance_type=distance_type,
        projection=projection,
        chunk_cols=chunk_cols,
    )
    k0[:nz, :], _ = _covariance_from_distances(
        Dz0, primary_model_eval, set_diagonal=False)
    del Dz0

    # cy0
    _, Dy0 = pairwise_distances(
        Xy_use, XT_use,
        distance_type=distance_type,
        projection=projection,
        chunk_cols=chunk_cols,
    )
    k0[nz:, :], _ = _covariance_from_distances(
        Dy0, cross_model_eval, set_diagonal=False)
    del Dy0

    return k0


def _assemble_sck_system(
    primary_coords: np.ndarray,
    secondary_coords: np.ndarray,
    targets: np.ndarray,
    *,
    primary_model: dict,
    secondary_model: dict,
    cross_model: dict,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    jitter: float = 1e-10,
    chunk_cols: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Backward-compatible wrapper: assemble K and k0 together.
    """
    K, sigma2_primary = _assemble_sck_conditioning_matrix(
        primary_coords,
        secondary_coords,
        primary_model=primary_model,
        secondary_model=secondary_model,
        cross_model=cross_model,
        distance_type=distance_type,
        projection=projection,
        rotation_matrix=rotation_matrix,
        jitter=jitter,
        chunk_cols=chunk_cols,
    )
    k0 = _assemble_sck_rhs(
        primary_coords,
        secondary_coords,
        targets,
        primary_model=primary_model,
        cross_model=cross_model,
        distance_type=distance_type,
        projection=projection,
        rotation_matrix=rotation_matrix,
        chunk_cols=chunk_cols,
    )
    return K, k0, sigma2_primary


def _factor_cokriging_matrix(
    K: np.ndarray,
    *,
    check_positive_definite: bool = True,
):
    """
    Factor a cokriging conditioning matrix once so it can be reused across
    multiple right-hand sides / target batches.
    """
    K = np.asarray(K, float)

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square 2D array.")

    try:
        return cho_factor(K, overwrite_a=False, check_finite=False)
    except Exception as e:
        msg = (
            "Cokriging covariance matrix is not numerically positive definite. "
            "This usually means the supplied covariance model(s) are not mutually "
            "compatible or the system is too ill-conditioned."
        )
        if check_positive_definite:
            raise np.linalg.LinAlgError(msg) from e
        raise


def _solve_sck_from_factor(
    cf,
    k0: np.ndarray,
) -> np.ndarray:
    """
    Solve a simple cokriging system from a precomputed Cholesky factorization.
    """
    k0 = np.asarray(k0, float)

    if k0.ndim == 1:
        k0 = k0[:, None]
    elif k0.ndim != 2:
        raise ValueError("k0 must be a 1D or 2D array.")

    return cho_solve(cf, k0, check_finite=False)


def _solve_ordinary_cokriging_from_factor(
    cf,
    k0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the single-constraint ordinary cokriging system from a precomputed
    Cholesky factorization of K.
    """
    k0 = np.asarray(k0, float)

    if k0.ndim == 1:
        k0 = k0[:, None]
    elif k0.ndim != 2:
        raise ValueError("k0 must be a 1D or 2D array.")

    n = k0.shape[0]
    ones = np.ones(n, dtype=float)

    Ki1 = cho_solve(cf, ones, check_finite=False)       # (n,)
    denom = float(ones @ Ki1)

    if (not np.isfinite(denom)) or (denom <= 0.0):
        raise np.linalg.LinAlgError(
            "Ordinary cokriging constraint system is singular or ill-conditioned "
            "because 1^T K^{-1} 1 is not finite and positive."
        )

    Kik0 = cho_solve(cf, k0, check_finite=False)        # (n, m)
    mu = ((ones @ Kik0) - 1.0) / denom                  # (m,)

    Wt = Kik0 - Ki1[:, None] * mu[None, :]
    return Wt, np.asarray(mu, float).ravel()


def _solve_sck_system(
    K: np.ndarray,
    k0: np.ndarray,
    *,
    check_positive_definite: bool = True,
) -> np.ndarray:
    """
    Backward-compatible wrapper: factor then solve.
    """
    K = np.asarray(K, float)
    k0 = np.asarray(k0, float)

    if k0.ndim == 1:
        k0 = k0[:, None]
    elif k0.ndim != 2:
        raise ValueError("k0 must be a 1D or 2D array.")

    if k0.shape[0] != K.shape[0]:
        raise ValueError("K and k0 have incompatible leading dimensions.")

    cf = _factor_cokriging_matrix(
        K,
        check_positive_definite=check_positive_definite,
    )
    return _solve_sck_from_factor(cf, k0)


def _solve_ordinary_cokriging_system(
    K: np.ndarray,
    k0: np.ndarray,
    *,
    check_positive_definite: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible wrapper: factor then solve.
    """
    K = np.asarray(K, float)
    k0 = np.asarray(k0, float)

    if k0.ndim == 1:
        k0 = k0[:, None]
    elif k0.ndim != 2:
        raise ValueError("k0 must be a 1D or 2D array.")

    if k0.shape[0] != K.shape[0]:
        raise ValueError("K and k0 have incompatible leading dimensions.")

    cf = _factor_cokriging_matrix(
        K,
        check_positive_definite=check_positive_definite,
    )
    return _solve_ordinary_cokriging_from_factor(cf, k0)


def _run_ordinary_cokriging_core(
    *,
    z: np.ndarray,
    y: np.ndarray,
    Xz: np.ndarray,
    Xy: np.ndarray,
    XT: np.ndarray,
    mu_z: Optional[float],
    sd_z: float,
    standardize: bool,
    assemble_system_fn,
    assemble_conditioning_fn=None,
    assemble_rhs_fn=None,
    target_batch_size: Optional[int] = None,
    distance_type: str,
    max_neighbors_primary: Optional[int],
    max_neighbors_secondary: Optional[int],
    balltree_leaf_size: int,
    check_positive_definite: bool,
    return_weights: bool,
    show_progress: bool = True,
):
    """
    Run the global or local single-constraint ordinary cokriging solve.

    Notes
    -----
    The single unbiasedness constraint is applied across all cokriging weights:

        sum(w_primary) + sum(w_secondary) = 1

    In global mode, if assemble_conditioning_fn and assemble_rhs_fn are both
    supplied, the conditioning matrix is factorized once and the targets may be
    processed in batches through repeated right-hand-side solves. Otherwise the
    full K / k0 system is assembled and solved in one step.

    In local mode, neighbor selection is performed on the original cleaned
    coordinates. Any anisotropy transform is applied later during covariance
    evaluation only.
    """
    nz = z.size
    ny = y.size
    m = XT.shape[0]

    use_local_z = (max_neighbors_primary is not None) and (int(max_neighbors_primary) < nz)
    use_local_y = (max_neighbors_secondary is not None) and (int(max_neighbors_secondary) < ny)
    use_local = use_local_z or use_local_y

    if not use_local:
        if assemble_conditioning_fn is None or assemble_rhs_fn is None:
            K, k0, sigma2_primary = assemble_system_fn(Xz, Xy, XT)
            Wt, mu = _solve_ordinary_cokriging_system(
                K,
                k0,
                check_positive_definite=check_positive_definite,
            )

            W = Wt.T
            wz = W[:, :nz]
            wy = W[:, nz:]

            est_work = (wz @ z) + (wy @ y)
            var_work = np.maximum(
                sigma2_primary - np.einsum("ij,ij->i", W, k0.T) - mu,
                0.0,
            )

            if standardize:
                est = mu_z + sd_z * est_work
                var = (sd_z ** 2) * var_work
            else:
                est = est_work
                var = var_work

            if return_weights:
                return est, var, wz, wy
            return est, var

        K, sigma2_primary = assemble_conditioning_fn(Xz, Xy)
        cf = _factor_cokriging_matrix(
            K,
            check_positive_definite=check_positive_definite,
        )
        del K

        if target_batch_size is None:
            batch_size = m
        else:
            batch_size = int(target_batch_size)
            if batch_size <= 0:
                raise ValueError("target_batch_size must be None or a positive integer.")

        est = np.empty(m, dtype=float)
        var = np.empty(m, dtype=float)

        if return_weights:
            wz = np.empty((m, nz), dtype=float)
            wy = np.empty((m, ny), dtype=float)

        batch_iter = range(0, m, batch_size)
        if show_progress and batch_size < m:
            batch_iter = tqdm(batch_iter, desc="Processing estimate batches")

        for j0 in batch_iter:
            j1 = min(j0 + batch_size, m)

            k0 = assemble_rhs_fn(Xz, Xy, XT[j0:j1])
            Wt, mu = _solve_ordinary_cokriging_from_factor(cf, k0)
            W = Wt.T

            wz_b = W[:, :nz]
            wy_b = W[:, nz:]

            est_work_b = (wz_b @ z) + (wy_b @ y)
            var_work_b = np.maximum(
                sigma2_primary - np.einsum("ij,ij->i", W, k0.T) - mu,
                0.0,
            )

            if standardize:
                est[j0:j1] = mu_z + sd_z * est_work_b
                var[j0:j1] = (sd_z ** 2) * var_work_b
            else:
                est[j0:j1] = est_work_b
                var[j0:j1] = var_work_b

            if return_weights:
                wz[j0:j1, :] = wz_b
                wy[j0:j1, :] = wy_b

        if return_weights:
            return est, var, wz, wy
        return est, var

    if use_local_z:
        nn_z = query_nearest_neighbors_balltree(
            Xz, XT,
            distance_type=distance_type,
            max_neighbors=max_neighbors_primary,
            balltree_leaf_size=balltree_leaf_size,
        )
    else:
        nn_z = [np.arange(nz)] * m

    if use_local_y:
        nn_y = query_nearest_neighbors_balltree(
            Xy, XT,
            distance_type=distance_type,
            max_neighbors=max_neighbors_secondary,
            balltree_leaf_size=balltree_leaf_size,
        )
    else:
        nn_y = [np.arange(ny)] * m

    est = np.empty(m, dtype=float)
    var = np.empty(m, dtype=float)

    if return_weights:
        if use_local_z:
            wz_full = [None] * m
        else:
            wz_full = np.zeros((m, nz), dtype=float)

        if use_local_y:
            wy_full = [None] * m
        else:
            wy_full = np.zeros((m, ny), dtype=float)
    else:
        wz_full = None
        wy_full = None

    j_iter = tqdm(range(m), desc="Processing estimate") if show_progress else range(m)

    for j in j_iter:
        idx_z = np.asarray(nn_z[j], dtype=int)
        idx_y = np.asarray(nn_y[j], dtype=int)

        Xz_loc = Xz[idx_z]
        Xy_loc = Xy[idx_y]
        z_loc = z[idx_z]
        y_loc = y[idx_y]
        XT_j = XT[j:j+1]

        K, k0, sigma2_primary = assemble_system_fn(Xz_loc, Xy_loc, XT_j)
        Wt, mu = _solve_ordinary_cokriging_system(
            K,
            k0,
            check_positive_definite=check_positive_definite,
        )

        w = Wt[:, 0]
        mu_j = float(mu[0])

        nloc_z = idx_z.size
        wz = w[:nloc_z]
        wy = w[nloc_z:]

        est_work_j = (wz @ z_loc) + (wy @ y_loc)
        var_work_j = max(sigma2_primary - float(w @ k0[:, 0]) - mu_j, 0.0)

        if standardize:
            est[j] = mu_z + sd_z * est_work_j
            var[j] = (sd_z ** 2) * var_work_j
        else:
            est[j] = est_work_j
            var[j] = var_work_j

        if return_weights:
            if use_local_z:
                wz_full[j] = (idx_z.copy(), wz.copy())
            else:
                wz_full[j, :] = wz

            if use_local_y:
                wy_full[j] = (idx_y.copy(), wy.copy())
            else:
                wy_full[j, :] = wy

    if return_weights:
        return est, var, wz_full, wy_full
    return est, var
# --------------------------------------------------------------------------------------
# LMC helpers
# --------------------------------------------------------------------------------------

def _validate_lmc_coregionalization_matrix(B, *, tol: float = 1e-12) -> np.ndarray:
    """
    Validate and symmetrize a 2x2 coregionalization matrix.

    Each LMC structure must carry a positive semi-definite 2x2 matrix

        B = [[b_zz, b_zy],
             [b_yz, b_yy]]

    so that the multivariate covariance remains admissible.
    """
    B = np.asarray(B, float)

    if B.shape != (2, 2):
        raise ValueError("Each LMC coregionalization matrix B must have shape (2, 2).")

    if not np.all(np.isfinite(B)):
        raise ValueError("Each LMC coregionalization matrix B must be finite.")

    # enforce symmetry numerically
    B = 0.5 * (B + B.T)

    evals = np.linalg.eigvalsh(B)
    if np.any(evals < -tol):
        raise ValueError(
            "Each LMC coregionalization matrix B must be positive semi-definite. "
            f"Smallest eigenvalue = {evals.min():.6e}"
        )

    return B

def make_lmc_structure_spec(
    *,
    B,
    model_family: str,
    model_type: str,
    params: Optional[dict] = None,
    theta: Optional[Sequence[float]] = None,
    sigma2: Optional[float] = None,
    name: Optional[str] = None,
    visible_in_diagnostic: bool = True,
) -> dict:
    """
    Pack one LMC structure specification.

    Parameters
    ----------
    B : array_like, shape (2, 2)
        Positive semi-definite 2x2 coregionalization matrix for this structure.
    model_family : {'variogram', 'correlation', 'nugget'}
        Scalar kernel family used for this structure.
    model_type : str
        Built-in model name for 'variogram' or 'correlation'.
        Use model_type='nugget' when model_family='nugget'.
    params, theta, sigma2
        Parameters passed through to the scalar kernel evaluator.
    name : str or None, default None
        Optional structure name used in summaries and plots.
    visible_in_diagnostic : bool, default True
        If False, the structure is omitted from diagnostic component plots unless
        hidden structures are explicitly requested.

    Notes
    -----
    In the LMC, the scalar kernel defines only the shared isotropic shape.
    The amplitude and cross-coupling are carried by B.

    Per-structure anisotropy for cokriging is supplied separately through the
    top-level simple_cokriging(..., rotation_matrix=...) or
    ordinary_cokriging(..., rotation_matrix=...) argument in LMC mode. It is
    not stored directly inside the structure specification.
    """
    B = _validate_lmc_coregionalization_matrix(B)

    family = str(model_family).lower().strip()
    if family not in ("variogram", "correlation", "nugget"):
        raise ValueError("model_family must be 'variogram', 'correlation', or 'nugget'.")

    if family == "nugget":
        if str(model_type).lower().strip() != "nugget":
            raise ValueError("For model_family='nugget', use model_type='nugget'.")

    return {
        "B": B,
        "model_family": family,
        "model_type": model_type,
        "params": params,
        "theta": theta,
        "sigma2": sigma2,
        "name": model_type if name is None else str(name),
        "visible_in_diagnostic": bool(visible_in_diagnostic),
    }

def summarize_lmc_structures(structures, *, tol: float = 1e-12) -> np.ndarray:
    """
    Sum the 2x2 coregionalization matrices over all structures.

    Useful for checking the implied zero-lag covariance matrix:
        C(0) = sum_l B_l
    """
    if len(structures) == 0:
        raise ValueError("At least one LMC structure is required.")

    Bsum = np.zeros((2, 2), dtype=float)
    for s in structures:
        Bsum += _validate_lmc_coregionalization_matrix(s["B"], tol=tol)
    return Bsum


def _unit_structure_covariance_from_distances(
    D: np.ndarray,
    structure: dict,
    *,
    zero_tol: float = 1e-12,
) -> np.ndarray:
    """
    Evaluate the unit-shape covariance/correlation curve for one LMC structure.

    Parameters
    ----------
    D : ndarray
        Distance matrix or lag vector.
    structure : dict
        LMC structure specification.
    zero_tol : float, default 1e-12
        Tolerance used to identify zero lag for nugget structures.

    Returns
    -------
    R : ndarray
        Unit-shape covariance/correlation with zero-lag value equal to 1.
    """
    D = np.asarray(D, float)

    family = structure["model_family"]
    model_type = structure["model_type"]
    params = structure.get("params", None)
    theta = structure.get("theta", None)
    sigma2 = structure.get("sigma2", None)

    # pure nugget structure
    if family == "nugget":
        return np.isclose(D, 0.0, atol=zero_tol, rtol=0.0).astype(float)

    # use the existing scalar covariance builder, then normalize to unit variance
    C, sigma2_used = _covariance_from_distances(
        D,
        {
            "model_family": family,
            "model_type": model_type,
            "params": params,
            "theta": theta,
            "sigma2": sigma2,
            "cross_scale": None,
        },
        set_diagonal=False,
        jitter=0.0,
    )

    if not np.isfinite(sigma2_used) or sigma2_used <= 0.0:
        raise ValueError(
            "Each scalar LMC structure must imply a positive finite zero-lag variance "
            "before coregionalization scaling."
        )

    return C / float(sigma2_used)

def _assemble_lmc_sck_conditioning_matrix(
    primary_coords: np.ndarray,
    secondary_coords: np.ndarray,
    *,
    structures: Sequence[dict],
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    jitter: float = 1e-10,
    chunk_cols: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Assemble only the LMC conditioning matrix K.

    Notes
    -----
    This version intentionally avoids caching full distance-block tuples, because
    that cache is a major source of peak-memory blow-up for large problems.
    """
    Xz = _coerce_coordinates(primary_coords, distance_type, "primary_coords")
    Xy = _coerce_coordinates(secondary_coords, distance_type, "secondary_coords")

    if len(structures) == 0:
        raise ValueError("At least one LMC structure must be supplied.")

    rot_list = _resolve_lmc_rotation_matrices(rotation_matrix, len(structures))

    nz = Xz.shape[0]
    ny = Xy.shape[0]

    K = np.zeros((nz + ny, nz + ny), dtype=float)
    Kzz = K[:nz, :nz]
    Kyy = K[nz:, nz:]
    Kzy = K[:nz, nz:]

    Bsum = np.zeros((2, 2), dtype=float)

    for s, rot_use in zip(structures, rot_list):
        B = _validate_lmc_coregionalization_matrix(s["B"])
        Bsum += B

        s_eval = _make_rotated_lmc_structure_eval_spec(
            s,
            rotation_matrix=rot_use,
        )

        Xz_use = _apply_rotation_matrix(
            Xz,
            rotation_matrix=rot_use,
            distance_type=distance_type,
            name="primary_coords",
        )
        Xy_use = _apply_rotation_matrix(
            Xy,
            rotation_matrix=rot_use,
            distance_type=distance_type,
            name="secondary_coords",
        )

        Dzz, _ = pairwise_distances(
            Xz_use, Xz_use,
            distance_type=distance_type,
            projection=projection,
            chunk_cols=chunk_cols,
        )
        Rzz = _unit_structure_covariance_from_distances(Dzz, s_eval)
        Kzz += B[0, 0] * Rzz
        del Dzz, Rzz

        Dyy, _ = pairwise_distances(
            Xy_use, Xy_use,
            distance_type=distance_type,
            projection=projection,
            chunk_cols=chunk_cols,
        )
        Ryy = _unit_structure_covariance_from_distances(Dyy, s_eval)
        Kyy += B[1, 1] * Ryy
        del Dyy, Ryy

        _, Dzy = pairwise_distances(
            Xz_use, Xy_use,
            distance_type=distance_type,
            projection=projection,
            chunk_cols=chunk_cols,
        )
        Rzy = _unit_structure_covariance_from_distances(Dzy, s_eval)
        Kzy += B[0, 1] * Rzy
        del Dzy, Rzy

    sigma2_primary = float(Bsum[0, 0])
    sigma2_secondary = float(Bsum[1, 1])

    if not np.isfinite(sigma2_primary) or sigma2_primary <= 0.0:
        raise ValueError("The summed LMC primary variance sum_l B_l[0,0] must be positive.")
    if not np.isfinite(sigma2_secondary) or sigma2_secondary <= 0.0:
        raise ValueError("The summed LMC secondary variance sum_l B_l[1,1] must be positive.")

    Kzz[np.diag_indices(nz)] = sigma2_primary
    Kyy[np.diag_indices(ny)] = sigma2_secondary

    Kzz[np.diag_indices(nz)] += jitter * max(sigma2_primary, 1.0)
    Kyy[np.diag_indices(ny)] += jitter * max(sigma2_secondary, 1.0)

    K[nz:, :nz] = K[:nz, nz:].T
    return K, sigma2_primary


def _assemble_lmc_sck_rhs(
    primary_coords: np.ndarray,
    secondary_coords: np.ndarray,
    targets: np.ndarray,
    *,
    structures: Sequence[dict],
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    chunk_cols: Optional[int] = None,
) -> np.ndarray:
    """
    Assemble only the LMC RHS block k0 for one or more targets.
    """
    Xz = _coerce_coordinates(primary_coords, distance_type, "primary_coords")
    Xy = _coerce_coordinates(secondary_coords, distance_type, "secondary_coords")
    XT = _coerce_coordinates(targets, distance_type, "targets")

    if len(structures) == 0:
        raise ValueError("At least one LMC structure must be supplied.")

    rot_list = _resolve_lmc_rotation_matrices(rotation_matrix, len(structures))

    nz = Xz.shape[0]
    ny = Xy.shape[0]
    m = XT.shape[0]

    k0 = np.zeros((nz + ny, m), dtype=float)
    kz0 = k0[:nz, :]
    ky0 = k0[nz:, :]

    for s, rot_use in zip(structures, rot_list):
        B = _validate_lmc_coregionalization_matrix(s["B"])

        s_eval = _make_rotated_lmc_structure_eval_spec(
            s,
            rotation_matrix=rot_use,
        )

        Xz_use = _apply_rotation_matrix(
            Xz,
            rotation_matrix=rot_use,
            distance_type=distance_type,
            name="primary_coords",
        )
        Xy_use = _apply_rotation_matrix(
            Xy,
            rotation_matrix=rot_use,
            distance_type=distance_type,
            name="secondary_coords",
        )
        XT_use = _apply_rotation_matrix(
            XT,
            rotation_matrix=rot_use,
            distance_type=distance_type,
            name="targets",
        )

        _, Dz0 = pairwise_distances(
            Xz_use, XT_use,
            distance_type=distance_type,
            projection=projection,
            chunk_cols=chunk_cols,
        )
        Rz0 = _unit_structure_covariance_from_distances(Dz0, s_eval)
        kz0 += B[0, 0] * Rz0
        del Dz0, Rz0

        _, Dy0 = pairwise_distances(
            Xy_use, XT_use,
            distance_type=distance_type,
            projection=projection,
            chunk_cols=chunk_cols,
        )
        Ry0 = _unit_structure_covariance_from_distances(Dy0, s_eval)
        ky0 += B[1, 0] * Ry0
        del Dy0, Ry0

    return k0


def _assemble_lmc_sck_system(
    primary_coords: np.ndarray,
    secondary_coords: np.ndarray,
    targets: np.ndarray,
    *,
    structures: Sequence[dict],
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    jitter: float = 1e-10,
    chunk_cols: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Backward-compatible wrapper: assemble K and k0 together.
    """
    K, sigma2_primary = _assemble_lmc_sck_conditioning_matrix(
        primary_coords,
        secondary_coords,
        structures=structures,
        distance_type=distance_type,
        projection=projection,
        rotation_matrix=rotation_matrix,
        jitter=jitter,
        chunk_cols=chunk_cols,
    )
    k0 = _assemble_lmc_sck_rhs(
        primary_coords,
        secondary_coords,
        targets,
        structures=structures,
        distance_type=distance_type,
        projection=projection,
        rotation_matrix=rotation_matrix,
        chunk_cols=chunk_cols,
    )
    return K, k0, sigma2_primary

def make_lmc_kernel_spec(
    *,
    model_family: str,
    model_type: str,
    params: Optional[dict] = None,
    theta: Optional[Sequence[float]] = None,
    sigma2: Optional[float] = None,
    name: Optional[str] = None,
) -> dict:
    """
    Pack one free scalar LMC kernel specification.

    Notes
    -----
    This defines only the shared kernel shape used by the LMC.
    The amplitudes and cross-couplings are fitted later through the
    2x2 coregionalization matrices B_l.
    """
    family = str(model_family).lower().strip()
    if family not in ("variogram", "correlation"):
        raise ValueError("model_family must be 'variogram' or 'correlation'.")

    return {
        "model_family": family,
        "model_type": model_type,
        "params": params,
        "theta": theta,
        "sigma2": sigma2,
        "name": model_type if name is None else str(name),
    }

def align_lmc_experimental_bins(
    h_primary,
    rho_primary,
    n_primary,
    h_secondary,
    rho_secondary,
    n_secondary,
    h_cross,
    rho_cross,
    n_cross,
    *,
    round_decimals: int = 12,
):
    """
    Intersect the valid lag bins from primary, secondary, and cross curves.

    Returns
    -------
    h : (k,) ndarray
        Common lag centers.
    rz : (k,) ndarray
        Primary experimental correlation on common bins.
    ry : (k,) ndarray
        Secondary experimental correlation on common bins.
    rzy : (k,) ndarray
        Cross experimental correlation on common bins.
    wz, wy, wzy : (k,) ndarray
        Default per-bin weights from pair counts.
    """
    hp = np.round(np.asarray(h_primary, float).ravel(), round_decimals)
    hs = np.round(np.asarray(h_secondary, float).ravel(), round_decimals)
    hc = np.round(np.asarray(h_cross, float).ravel(), round_decimals)

    rp = np.asarray(rho_primary, float).ravel()
    rs = np.asarray(rho_secondary, float).ravel()
    rc = np.asarray(rho_cross, float).ravel()

    np_ = np.asarray(n_primary, float).ravel()
    ns_ = np.asarray(n_secondary, float).ravel()
    nc_ = np.asarray(n_cross, float).ravel()

    common = np.intersect1d(np.intersect1d(hp, hs), hc)
    if common.size == 0:
        raise ValueError(
            "No common lag bins were found across primary / secondary / cross curves. "
            "Use the same max_distance and bin_size for all three fits."
        )

    def _take(h, y, w, common_h):
        out_y = np.empty(common_h.size, dtype=float)
        out_w = np.empty(common_h.size, dtype=float)
        for i, hh in enumerate(common_h):
            idx = np.where(h == hh)[0]
            if idx.size != 1:
                raise ValueError(f"Lag alignment failed at h={hh}.")
            out_y[i] = y[idx[0]]
            out_w[i] = w[idx[0]]
        return out_y, out_w

    rz, wz = _take(hp, rp, np_, common)
    ry, wy = _take(hs, rs, ns_, common)
    rzy, wzy = _take(hc, rc, nc_, common)

    return common, rz, ry, rzy, wz, wy, wzy

def _evaluate_free_kernel_shapes_on_lags(
    h: np.ndarray,
    free_kernels: Sequence[dict],
) -> np.ndarray:
    """
    Evaluate the unit covariance/correlation shapes for a list of free LMC kernels
    on a common lag grid.

    Returns
    -------
    R_list : (L, n_lags) ndarray
        Unit-shape curves for the L free kernels.
    """
    h = np.asarray(h, float).ravel()

    R_list = []
    for ks in free_kernels:
        s_tmp = {
            "B": np.eye(2),
            "model_family": ks["model_family"],
            "model_type": ks["model_type"],
            "params": ks.get("params", None),
            "theta": ks.get("theta", None),
            "sigma2": ks.get("sigma2", None),
            "name": ks.get("name", ks["model_type"]),
            "visible_in_diagnostic": False,
        }
        R_list.append(_unit_structure_covariance_from_distances(h, s_tmp))

    return np.asarray(R_list, dtype=float)

def evaluate_lmc_on_lags(
    h,
    structures,
):
    """
    Evaluate total and per-structure LMC curves for Z, Y, and ZY on a 1D lag grid.

    Parameters
    ----------
    h : array_like
        Lag distances.
    structures : sequence of dict
        Full LMC structures. Each structure must already contain its fitted
        2x2 coregionalization matrix B.

    Returns
    -------
    out : dict
        Dictionary with keys:
            'h'
                Lag distances.
            'rho_z'
                Total primary direct curve.
            'rho_y'
                Total secondary direct curve.
            'rho_zy'
                Total cross curve.
            'components'
                Per-structure component contributions.
            'B_total'
                Sum of the structure-wise coregionalization matrices.

    Notes
    -----
    This helper evaluates the isotropic 1D lag-based LMC representation only.
    Any per-structure anisotropy later used during cokriging covariance assembly
    is not represented here.
    """
    h = np.asarray(h, float).ravel()

    rho_z = np.zeros_like(h, dtype=float)
    rho_y = np.zeros_like(h, dtype=float)
    rho_zy = np.zeros_like(h, dtype=float)
    components = []

    B_total = np.zeros((2, 2), dtype=float)

    for s in structures:
        B = _validate_lmc_coregionalization_matrix(s["B"])
        B_total += B

        R = _unit_structure_covariance_from_distances(h, s)

        comp_z = B[0, 0] * R
        comp_y = B[1, 1] * R
        comp_zy = B[0, 1] * R

        rho_z += comp_z
        rho_y += comp_y
        rho_zy += comp_zy

        components.append({
            "name": s.get("name", s["model_type"]),
            "visible_in_diagnostic": bool(s.get("visible_in_diagnostic", True)),
            "B": B,
            "R": R,
            "rho_z": comp_z,
            "rho_y": comp_y,
            "rho_zy": comp_zy,
            "rho_z0": float(B[0, 0]),
            "rho_y0": float(B[1, 1]),
            "rho_zy0": float(B[0, 1]),
        })

    return {
        "h": h,
        "rho_z": rho_z,
        "rho_y": rho_y,
        "rho_zy": rho_zy,
        "components": components,
        "B_total": B_total,
    }

def make_lmc_nugget_B(
    b_primary: float,
    b_secondary: float,
    b_cross: float = 0.0,
) -> np.ndarray:
    """
    Build a PSD nugget coregionalization matrix.

    Parameters
    ----------
    b_primary, b_secondary : float
        Direct nugget fractions on the standardized correlation scale. Each must
        be finite and nonnegative.
    b_cross : float, default 0.0
        Cross nugget. Must be finite and satisfy
        |b_cross| <= sqrt(b_primary * b_secondary).

    Returns
    -------
    B0 : (2, 2) ndarray
        Positive semi-definite nugget coregionalization matrix.
    """
    b_primary = float(b_primary)
    b_secondary = float(b_secondary)
    b_cross = float(b_cross)

    if not np.isfinite(b_primary) or b_primary < 0.0:
        raise ValueError("b_primary must be finite and nonnegative.")
    if not np.isfinite(b_secondary) or b_secondary < 0.0:
        raise ValueError("b_secondary must be finite and nonnegative.")
    if not np.isfinite(b_cross):
        raise ValueError("b_cross must be finite.")

    lim = np.sqrt(b_primary * b_secondary)
    if abs(b_cross) > lim + 1e-12:
        raise ValueError(
            f"Cross nugget violates PSD: |b_cross|={abs(b_cross)} > "
            f"sqrt(b_primary*b_secondary)={lim}."
        )

    B0 = np.array([
        [b_primary, b_cross],
        [b_cross, b_secondary],
    ], dtype=float)

    return _validate_lmc_coregionalization_matrix(B0)

def make_lmc_stabilizer_structure(
    eps_primary: float = 1e-6,
    eps_secondary: float = 1e-6,
    eps_cross: float = 0.0,
    *,
    name: str = "stabilizer",
):
    """
    Build a tiny nugget-like structure used only as a numerical stabilizer.

    This is not intended to represent a modelling nugget unless the user
    explicitly chooses to interpret it that way.
    """
    B = np.array([
        [float(eps_primary), float(eps_cross)],
        [float(eps_cross),   float(eps_secondary)],
    ], dtype=float)

    return make_lmc_structure_spec(
        B=B,
        model_family="nugget",
        model_type="nugget",
        name=name,
        visible_in_diagnostic=False,
    )

def check_lmc_feasibility(
    h,
    rho_primary,
    rho_secondary,
    *,
    free_kernels,
    fixed_structures=None,
):
    """
    Check whether the primary and secondary direct curves are reachable under the
    current shared-kernel span.

    Notes
    -----
    - This checks only the direct curves rho_z(h) and rho_y(h).
    - It does not check cross-curve feasibility, because the cross term depends on
      the fitted coregionalization couplings after optimization.
    """
    h = np.asarray(h, float).ravel()
    rz_exp = np.asarray(rho_primary, float).ravel()
    ry_exp = np.asarray(rho_secondary, float).ravel()

    fixed_structures = [] if fixed_structures is None else list(fixed_structures)

    if len(fixed_structures) > 0:
        fixed_eval = evaluate_lmc_on_lags(h, fixed_structures)
        B_fixed = fixed_eval["B_total"]
        rz_fixed = fixed_eval["rho_z"]
        ry_fixed = fixed_eval["rho_y"]
    else:
        B_fixed = np.zeros((2, 2), dtype=float)
        rz_fixed = np.zeros_like(h)
        ry_fixed = np.zeros_like(h)

    rem_z = 1.0 - B_fixed[0, 0]
    rem_y = 1.0 - B_fixed[1, 1]
    if rem_z < 0.0 or rem_y < 0.0:
        raise ValueError("Fixed structures exceed the standardized direct variances.")

    R = _evaluate_free_kernel_shapes_on_lags(h, free_kernels)

    rz_min = rz_fixed + rem_z * np.min(R, axis=0)
    rz_max = rz_fixed + rem_z * np.max(R, axis=0)

    ry_min = ry_fixed + rem_y * np.min(R, axis=0)
    ry_max = ry_fixed + rem_y * np.max(R, axis=0)

    z_ok = (rz_exp >= rz_min - 1e-12) & (rz_exp <= rz_max + 1e-12)
    y_ok = (ry_exp >= ry_min - 1e-12) & (ry_exp <= ry_max + 1e-12)

    return {
        "h": h,
        "rz_min": rz_min,
        "rz_max": rz_max,
        "ry_min": ry_min,
        "ry_max": ry_max,
        "z_ok": z_ok,
        "y_ok": y_ok,
    }

def _evaluate_covmodel_curve_on_lags(h, model_spec):
    """
    Evaluate a direct fitted covariance/correlation model on 1D lag values.
    """
    h = np.asarray(h, float).ravel()
    C, _ = _covariance_from_distances(
        h,
        model_spec,
        set_diagonal=False,
        jitter=0.0,
    )
    return np.asarray(C, float).ravel()


def plot_lmc_structures(
    h,
    rho_primary_exp,
    rho_secondary_exp,
    rho_cross_exp,
    *,
    structures,
    direct_models: Optional[dict] = None,
    plot_diagnostic: bool = False,
    show_hidden: bool = False,
    h_plot=None,
    smooth_n: int = 500,
    ylim=(-1.0, 1.0),
    figsize=(12, 4),
):
    """
    Plot the fitted LMC curves against the experimental correlograms.

    Default view
    ------------
    Plots:
        - experimental points
        - total fitted LMC
        - direct fitted curves, if direct_models is supplied

    Diagnostic view
    ---------------
    If plot_diagnostic=True, also plot each LMC component. Structures with
    visible_in_diagnostic=False are hidden unless show_hidden=True.

    Notes
    -----
    This helper plots the isotropic 1D lag-based LMC fit only. It does not
    visualize any anisotropic per-structure transforms that may later be used
    during covariance assembly in simple_cokriging(..., covariance_mode='lmc')
    or ordinary_cokriging(..., covariance_mode='lmc').
    """
    h = np.asarray(h, float).ravel()
    rz_exp = np.asarray(rho_primary_exp, float).ravel()
    ry_exp = np.asarray(rho_secondary_exp, float).ravel()
    rzy_exp = np.asarray(rho_cross_exp, float).ravel()

    if h_plot is None:
        h_plot = np.linspace(0.0, float(np.max(h)), smooth_n)
    else:
        h_plot = np.asarray(h_plot, float).ravel()

    out_plot = evaluate_lmc_on_lags(h_plot, structures)
    comps = out_plot["components"]

    direct_models = {} if direct_models is None else dict(direct_models)
    direct_curves = {
        "rho_z": None,
        "rho_y": None,
        "rho_zy": None,
    }

    if direct_models.get("primary", None) is not None:
        direct_curves["rho_z"] = _evaluate_covmodel_curve_on_lags(h_plot, direct_models["primary"])
    if direct_models.get("secondary", None) is not None:
        direct_curves["rho_y"] = _evaluate_covmodel_curve_on_lags(h_plot, direct_models["secondary"])
    if direct_models.get("cross", None) is not None:
        direct_curves["rho_zy"] = _evaluate_covmodel_curve_on_lags(h_plot, direct_models["cross"])

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=180, sharex=True, sharey=True)

    panels = [
        ("rho_z",  rz_exp,  r"Primary: $\rho_z(h)$",   "tab:red"),
        ("rho_y",  ry_exp,  r"Secondary: $\rho_y(h)$", "tab:green"),
        ("rho_zy", rzy_exp, r"Cross: $\rho_{zy}(h)$",  "tab:blue"),
    ]

    zero_lag_key = {
        "rho_z": "rho_z0",
        "rho_y": "rho_y0",
        "rho_zy": "rho_zy0",
    }

    for ax, (key, y_exp, ttl, cexp) in zip(axes, panels):
        if direct_curves[key] is not None:
            ax.plot(
                h_plot,
                direct_curves[key],
                color=cexp,
                lw=1.8,
                alpha=0.9,
                label="Direct fit",
            )

        ax.plot(h_plot, out_plot[key], color="k", lw=2.2, label="LMC total")

        ax.plot(
            h,
            y_exp,
            "o",
            color=cexp,
            markeredgecolor="black",
            label="Experimental",
        )

        if plot_diagnostic:
            for i, c in enumerate(comps):
                if (not c.get("visible_in_diagnostic", True)) and (not show_hidden):
                    continue

                amp0 = float(c[zero_lag_key[key]])
                label = f"{c['name']} [{i+1}], ρ(0)={amp0:.3g}"

                line, = ax.plot(
                    h_plot,
                    c[key],
                    lw=1.2,
                    alpha=0.95,
                    label=label,
                )

                ax.plot(
                    [0.0], [amp0],
                    marker="s",
                    ms=4.0,
                    color=line.get_color(),
                    zorder=4,
                )

        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_title(ttl)
        ax.set_xlabel("lag")
        ax.set_ylim(*ylim)
        ax.set_xlim(float(np.min(h_plot)), float(np.max(h_plot)))
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    axes[0].set_ylabel(r"$\rho(h)$")
    plt.tight_layout()
    plt.show()

    return out_plot

def fit_lmc_coregionalization(
    h,
    rho_primary,
    rho_secondary,
    rho_cross,
    *,
    free_kernels,
    fixed_structures=None,
    w_primary=None,
    w_secondary=None,
    w_cross=None,
    rho0_target=None,
    rho0_bounds=None,
    rho0_penalty_weight: float = 0.0,
    cross_misfit_weight: float = 1.0,
    require_feasible: bool = False,
    normalize_weights: bool = True,
    plot: bool = False,
    plot_diagnostic: bool = False,
    show_hidden_diagnostic: bool = False,
    direct_models: Optional[dict] = None,
    plot_ylim=(-1.0, 1.0),
    smooth_n: int = 500,
    options=None,
):
    """
    Fit the coregionalization matrices B_l for a fixed set of shared LMC kernels.

    Parameters
    ----------
    h : array_like
        Lag centers.
    rho_primary, rho_secondary, rho_cross : array_like
        Experimental primary, secondary, and cross correlograms on the same lag grid.
    free_kernels : sequence of dict
        Shared scalar kernels whose shapes are fixed during the fit.
    fixed_structures : sequence of dict or None, default None
        Optional fixed LMC structures added before fitting the free B matrices.
    w_primary, w_secondary, w_cross : array_like or None, default None
        Optional per-lag weights for the three curves.
    rho0_target : float or None, default None
        Optional target collocated cross-correlation used in a quadratic penalty.
        If supplied, rho0_bounds must also be supplied.
    rho0_bounds : sequence of 2 floats or None, default None
        Optional lower/upper bounds imposed directly on the fitted zero-lag cross
        correlation rho0_fit = B_total[0,1].

        - If rho0_bounds is None and rho0_target is None, the original unconstrained
          behaviour is preserved.
        - If rho0_bounds is supplied and rho0_target is None, a true constrained fit
          is used without a rho0 penalty term.
        - If rho0_target is supplied, rho0_bounds must also be supplied. In that case
          the fit is constrained to rho0_bounds and may additionally include the
          quadratic penalty toward rho0_target.
    rho0_penalty_weight : float, default 0.0
        Nonnegative weight of the rho0 penalty term.
    cross_misfit_weight : float, default 1.0
        Nonnegative relative weight assigned to the cross-correlogram misfit.
    require_feasible : bool, default False
        If True, reject kernel sets whose shared-kernel span cannot reproduce the
        direct primary/secondary curves.
    normalize_weights : bool, default True
        If True, normalize each weight vector to sum to one.
    plot : bool, default False
        If True, plot the fitted result.
    plot_diagnostic : bool, default False
        If True and plot=True, also show the individual LMC components.
    show_hidden_diagnostic : bool, default False
        If True, include hidden diagnostic structures such as stabilizers.
    direct_models : dict or None, default None
        Optional dict with keys {'primary', 'secondary', 'cross'} used for overlaying
        the direct fitted models in plots.
    plot_ylim : tuple, default (-1.0, 1.0)
        Y-axis limits for plots.
    smooth_n : int, default 500
        Number of points in the smooth plotting grid.
    options : dict or None, default None
        Options passed to scipy.optimize.minimize.

    Returns
    -------
    out : dict
        Dictionary containing fitted structures, fitted curves, misfit summaries,
        feasibility information, rho0-constraint summaries, and the optimizer result.

    Notes
    -----
    This function fits only the 2x2 coregionalization matrices B_l. The shared
    scalar kernel ranges and shape parameters are taken as fixed through
    free_kernels.

    The fitting is performed on isotropic 1D lag curves only. Any optional
    per-structure anisotropy used later for LMC cokriging is introduced during
    covariance assembly inside simple_cokriging(..., covariance_mode='lmc',
    rotation_matrix=...) or ordinary_cokriging(..., covariance_mode='lmc',
    rotation_matrix=...); it is not estimated here.

    When rho0_bounds is active, the constraint is imposed directly on the fitted
    total zero-lag cross-correlation rho0_fit = B_total[0,1] using a smooth
    constrained optimization.
    """
    h = np.asarray(h, float).ravel()
    rz_hat = np.asarray(rho_primary, float).ravel()
    ry_hat = np.asarray(rho_secondary, float).ravel()
    rzy_hat = np.asarray(rho_cross, float).ravel()

    if not (h.size == rz_hat.size == ry_hat.size == rzy_hat.size):
        raise ValueError("h, rho_primary, rho_secondary, and rho_cross must have the same length.")

    if rho0_target is not None and rho0_bounds is None:
        raise ValueError(
            "If rho0_target is supplied, rho0_bounds must also be supplied."
        )

    cross_misfit_weight = float(cross_misfit_weight)
    if (not np.isfinite(cross_misfit_weight)) or (cross_misfit_weight < 0.0):
        raise ValueError("cross_misfit_weight must be finite and nonnegative.")

    rho0_penalty_weight = float(rho0_penalty_weight)
    if (not np.isfinite(rho0_penalty_weight)) or (rho0_penalty_weight < 0.0):
        raise ValueError("rho0_penalty_weight must be finite and nonnegative.")

    L = len(free_kernels)
    if L < 1:
        raise ValueError("free_kernels must contain at least one kernel.")

    fixed_structures = [] if fixed_structures is None else list(fixed_structures)

    def _prep_w(w, n):
        if w is None:
            out = np.ones(n, dtype=float)
        else:
            out = np.asarray(w, float).ravel()
            if out.size != n:
                raise ValueError("Weight arrays must match len(h).")

        if not np.all(np.isfinite(out)):
            raise ValueError("Weight arrays must be finite.")
        if np.any(out < 0.0):
            raise ValueError("Weight arrays must be nonnegative.")

        if normalize_weights:
            s = np.sum(out)
            if s <= 0.0:
                raise ValueError("Weights must have positive sum.")
            out = out / s

        return out

    wz = _prep_w(w_primary, h.size)
    wy = _prep_w(w_secondary, h.size)
    wzy = _prep_w(w_cross, h.size)

    if len(fixed_structures) > 0:
        fixed_eval = evaluate_lmc_on_lags(h, fixed_structures)
        B_fixed = fixed_eval["B_total"]
        rz_fixed = fixed_eval["rho_z"]
        ry_fixed = fixed_eval["rho_y"]
        rzy_fixed = fixed_eval["rho_zy"]
    else:
        B_fixed = np.zeros((2, 2), dtype=float)
        rz_fixed = np.zeros_like(h)
        ry_fixed = np.zeros_like(h)
        rzy_fixed = np.zeros_like(h)

    rem_z = 1.0 - B_fixed[0, 0]
    rem_y = 1.0 - B_fixed[1, 1]

    if rem_z < 0.0 or rem_y < 0.0:
        raise ValueError("Fixed structures exceed the standardized direct variances.")

    rho0_lo = None
    rho0_hi = None
    if rho0_bounds is not None:
        try:
            rho0_lo, rho0_hi = [float(v) for v in rho0_bounds]
        except Exception as e:
            raise ValueError(
                "rho0_bounds must be a length-2 sequence (rho0_lo, rho0_hi)."
            ) from e

        if (not np.isfinite(rho0_lo)) or (not np.isfinite(rho0_hi)):
            raise ValueError("rho0_bounds entries must be finite.")
        if rho0_lo > rho0_hi:
            raise ValueError("rho0_bounds must satisfy rho0_lo <= rho0_hi.")
        if rho0_lo < -1.0 or rho0_hi > 1.0:
            raise ValueError("rho0_bounds must lie within [-1, 1].")

        if rho0_target is not None:
            rho0_target = float(rho0_target)
            if (not np.isfinite(rho0_target)) or abs(rho0_target) > 1.0:
                raise ValueError("rho0_target must be finite and lie in [-1, 1].")
            if rho0_target < rho0_lo or rho0_target > rho0_hi:
                raise ValueError(
                    "rho0_target must lie within rho0_bounds."
                )

    R_list = _evaluate_free_kernel_shapes_on_lags(h, free_kernels)

    feasibility = check_lmc_feasibility(
        h,
        rz_hat,
        ry_hat,
        free_kernels=free_kernels,
        fixed_structures=fixed_structures,
    )

    n_bad_z = int((~feasibility["z_ok"]).sum())
    n_bad_y = int((~feasibility["y_ok"]).sum())
    n_bad_total = n_bad_z + n_bad_y
    is_feasible = (n_bad_total == 0)

    if require_feasible and not is_feasible:
        raise ValueError(
            f"Kernel set is infeasible for the direct curves: "
            f"{n_bad_z} bad primary bins, {n_bad_y} bad secondary bins."
        )

    # Global achievable interval for the total zero-lag cross term contributed
    # by the free structures under the current standardized direct-variance budget.
    rho0_free_span = float(np.sqrt(max(rem_z, 0.0) * max(rem_y, 0.0)))
    rho0_reachable_lo = float(B_fixed[0, 1] - rho0_free_span)
    rho0_reachable_hi = float(B_fixed[0, 1] + rho0_free_span)

    if rho0_bounds is not None:
        lo_eff = max(rho0_lo, rho0_reachable_lo)
        hi_eff = min(rho0_hi, rho0_reachable_hi)
        if hi_eff < lo_eff - 1e-12:
            raise ValueError(
                "rho0_bounds has empty intersection with the reachable rho0 interval "
                f"[{rho0_reachable_lo:.6g}, {rho0_reachable_hi:.6g}] implied by the "
                "fixed structures and remaining direct-variance budget."
            )

    def _softmax(x):
        x = np.asarray(x, float)
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / np.sum(ex)

    def _unpack(p):
        az_raw = p[:L]
        ay_raw = p[L:2 * L]
        eta_raw = p[2 * L:3 * L]

        az = rem_z * _softmax(az_raw) if rem_z > 0 else np.zeros(L, dtype=float)
        ay = rem_y * _softmax(ay_raw) if rem_y > 0 else np.zeros(L, dtype=float)

        eta = np.tanh(eta_raw)

        B_list = []
        for l in range(L):
            off = eta[l] * np.sqrt(max(az[l], 0.0) * max(ay[l], 0.0))
            B = np.array([
                [az[l], off],
                [off, ay[l]],
            ], dtype=float)
            B_list.append(B)

        return az, ay, eta, B_list

    def _evaluate_from_B_list(B_list):
        rz_fit = rz_fixed.copy()
        ry_fit = ry_fixed.copy()
        rzy_fit = rzy_fixed.copy()

        for l in range(L):
            Rl = R_list[l]
            rz_fit += B_list[l][0, 0] * Rl
            ry_fit += B_list[l][1, 1] * Rl
            rzy_fit += B_list[l][0, 1] * Rl

        rho0_fit = B_fixed[0, 1] + sum(B[0, 1] for B in B_list)
        return rz_fit, ry_fit, rzy_fit, rho0_fit

    def _rho0_from_p(p):
        _, _, _, B_list = _unpack(p)
        return float(B_fixed[0, 1] + sum(B[0, 1] for B in B_list))

    def _objective(p):
        _, _, _, B_list = _unpack(p)
        rz_fit, ry_fit, rzy_fit, rho0_fit = _evaluate_from_B_list(B_list)

        misfit_z = np.sum(wz * (rz_hat - rz_fit) ** 2)
        misfit_y = np.sum(wy * (ry_hat - ry_fit) ** 2)
        misfit_zy = np.sum(wzy * (rzy_hat - rzy_fit) ** 2)

        obj = misfit_z + misfit_y + float(cross_misfit_weight) * misfit_zy

        if rho0_target is not None and rho0_penalty_weight > 0.0:
            obj += float(rho0_penalty_weight) * (rho0_fit - float(rho0_target)) ** 2

        return float(obj)

    p0 = np.zeros(3 * L, dtype=float)

    # When rho0 is constrained, seed eta so the initial guess is as close as
    # possible to a feasible/meaningful rho0 under uniform direct allocations.
    if rho0_bounds is not None and L > 0 and rho0_free_span > 0.0:
        if rho0_target is not None:
            rho0_start = float(rho0_target)
        else:
            rho0_start = 0.5 * (rho0_lo + rho0_hi)

        rho0_start = min(max(rho0_start, rho0_lo), rho0_hi)
        rho0_start = min(max(rho0_start, rho0_reachable_lo), rho0_reachable_hi)

        az0 = rem_z / float(L) if rem_z > 0.0 else 0.0
        ay0 = rem_y / float(L) if rem_y > 0.0 else 0.0
        eta_denom = float(L) * np.sqrt(max(az0, 0.0) * max(ay0, 0.0))

        if eta_denom > 0.0:
            eta0 = (rho0_start - float(B_fixed[0, 1])) / eta_denom
            eta0 = float(np.clip(eta0, -1.0 + 1e-8, 1.0 - 1e-8))
            p0[2 * L:3 * L] = np.arctanh(eta0)

    opt_method = "L-BFGS-B"
    constraints = ()

    if rho0_bounds is not None:
        opt_method = "SLSQP"
        constraints = (
            {"type": "ineq", "fun": lambda p, lo=float(rho0_lo): _rho0_from_p(p) - lo},
            {"type": "ineq", "fun": lambda p, hi=float(rho0_hi): hi - _rho0_from_p(p)},
        )

    res = minimize(
        _objective,
        x0=p0,
        method=opt_method,
        constraints=constraints,
        options={"maxiter": 2000} if options is None else options,
    )
    if not res.success:
        raise RuntimeError(f"LMC optimization failed: {res.message}")

    az, ay, eta, B_list = _unpack(res.x)
    rz_fit, ry_fit, rzy_fit, rho0_fit = _evaluate_from_B_list(B_list)

    misfit_z = float(np.sum(wz * (rz_hat - rz_fit) ** 2))
    misfit_y = float(np.sum(wy * (ry_hat - ry_fit) ** 2))
    misfit_zy = float(np.sum(wzy * (rzy_hat - rzy_fit) ** 2))

    rho0_penalty = 0.0
    if rho0_target is not None and rho0_penalty_weight > 0.0:
        rho0_penalty = float(rho0_penalty_weight) * (rho0_fit - float(rho0_target)) ** 2

    rho0_constraint_violation = 0.0
    if rho0_bounds is not None:
        rho0_constraint_violation = max(
            float(rho0_lo - rho0_fit),
            float(rho0_fit - rho0_hi),
            0.0,
        )

    fitted_structures = list(fixed_structures)
    for l, ks in enumerate(free_kernels):
        s = make_lmc_structure_spec(
            B=B_list[l],
            model_family=ks["model_family"],
            model_type=ks["model_type"],
            params=ks.get("params", None),
            theta=ks.get("theta", None),
            sigma2=ks.get("sigma2", None),
        )
        s["name"] = ks.get("name", ks["model_type"])
        fitted_structures.append(s)

    fit_eval = evaluate_lmc_on_lags(h, fitted_structures)

    out = {
        "structures": fitted_structures,
        "free_B_matrices": B_list,
        "weights_primary": az,
        "weights_secondary": ay,
        "etas": eta,
        "B_total": fit_eval["B_total"],
        "rho0_fit": float(rho0_fit),
        "rho_primary_fit": fit_eval["rho_z"],
        "rho_secondary_fit": fit_eval["rho_y"],
        "rho_cross_fit": fit_eval["rho_zy"],
        "misfit_primary": misfit_z,
        "misfit_secondary": misfit_y,
        "misfit_cross": misfit_zy,
        "rho0_penalty": rho0_penalty,
        "cross_misfit_weight": float(cross_misfit_weight),
        "rho0_penalty_weight": float(rho0_penalty_weight),
        "rho0_target": None if rho0_target is None else float(rho0_target),
        "rho0_bounds": None if rho0_bounds is None else (float(rho0_lo), float(rho0_hi)),
        "rho0_constraint_active": bool(rho0_bounds is not None),
        "rho0_reachable_interval": (float(rho0_reachable_lo), float(rho0_reachable_hi)),
        "rho0_constraint_violation": float(rho0_constraint_violation),
        "objective": float(res.fun),
        "optimizer": res,
        "optimizer_method": opt_method,
        "feasibility": feasibility,
        "is_feasible": bool(is_feasible),
        "n_bad_z": int(n_bad_z),
        "n_bad_y": int(n_bad_y),
        "n_bad_total": int(n_bad_total),
    }

    if plot:
        plot_lmc_structures(
            h,
            rz_hat,
            ry_hat,
            rzy_hat,
            structures=fitted_structures,
            direct_models=direct_models,
            plot_diagnostic=plot_diagnostic,
            show_hidden=show_hidden_diagnostic,
            smooth_n=smooth_n,
            ylim=plot_ylim,
        )

    return out

def _lmc_range_param_name(model_type: str) -> str:
    """
    Return the range-like parameter name used by the built-in model.
    """
    model_type = str(model_type).lower().strip()

    if model_type in ("spherical", "exponential", "gaussian", "cubic",
                      "powered_exponential", "matern"):
        return "r"
    elif model_type in ("damped_cosine_angle", "angular_dissimilarity"):
        return "c"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _lmc_shape_param_names(model_type: str):
    """
    Return the non-range shape parameter names for the built-in model.
    """
    model_type = str(model_type).lower().strip()

    if model_type == "powered_exponential":
        return ("beta",)
    elif model_type == "matern":
        return ("s",)
    else:
        return tuple()


def _unique_sorted_floats(values, *, round_decimals: int = 12, positive_only: bool = True):
    """
    Small helper to deduplicate float candidates robustly.
    """
    vals = np.asarray(values, float).ravel()
    vals = vals[np.isfinite(vals)]

    if positive_only:
        vals = vals[vals > 0.0]

    if vals.size == 0:
        return np.array([], dtype=float)

    vals = np.round(vals, round_decimals)
    vals = np.unique(vals)
    vals.sort()
    return vals.astype(float)


def _kernel_summary_dict(kernel_spec: dict) -> dict:
    """
    Compact metadata summary for one kernel spec.
    """
    model_type = kernel_spec["model_type"]
    range_key = _lmc_range_param_name(model_type)
    shape_keys = _lmc_shape_param_names(model_type)

    params = {} if kernel_spec.get("params", None) is None else dict(kernel_spec["params"])

    out = {
        "name": kernel_spec.get("name", model_type),
        "model_family": kernel_spec["model_family"],
        "model_type": model_type,
        "range_key": range_key,
        "range_value": float(params.get(range_key, np.nan)),
    }

    for sk in shape_keys:
        out[sk] = float(params.get(sk, np.nan))

    return out

def _build_direct_models_from_params(
    *,
    model_family: str,
    model_type: str,
    params_primary: Optional[dict] = None,
    params_secondary: Optional[dict] = None,
    params_cross: Optional[dict] = None,
    cross_scale: Optional[float] = None,
):
    """
    Build direct fitted model specs for plotting.

    Notes
    -----
    - Uses the same family/type as the direct fitted models that generated
      params_primary / params_secondary / params_cross.
    - If params_cross is a normalized cross-shape, pass cross_scale=rho0_target.
      If params_cross already carries amplitude, leave cross_scale=None.
    """
    out = {"primary": None, "secondary": None, "cross": None}

    if params_primary is not None:
        out["primary"] = make_covmodel_spec(
            model_family=model_family,
            model_type=model_type,
            params=params_primary,
        )

    if params_secondary is not None:
        out["secondary"] = make_covmodel_spec(
            model_family=model_family,
            model_type=model_type,
            params=params_secondary,
        )

    if params_cross is not None:
        out["cross"] = make_covmodel_spec(
            model_family=model_family,
            model_type=model_type,
            params=params_cross,
            cross_scale=cross_scale,
        )

    return out

def build_candidate_shared_kernel_library(
    *,
    model_family: str,
    model_type: str,
    params_primary: dict,
    params_secondary: dict,
    params_cross: Optional[dict] = None,
    n_ranges: int = 7,
    range_spacing: str = "linear",
    range_padding_frac: float = 0.15,
    include_direct_ranges: bool = True,
    extra_ranges=None,
    include_shape_average: bool = True,
    extra_shape_values: Optional[dict] = None,
    prefix: str = "K",
):
    """
    Build a library of shared scalar LMC kernels guided by the direct fitted
    primary / secondary / cross models.

    Notes
    -----
    - params_cross is optional, but if supplied it also informs the candidate
      ranges/shapes.
    - For model_family='variogram', this module uses the convention
      c0=structured sill and b=nugget. Therefore the generated free kernels are
      normalized structured unit-sill shapes with c0=1.0 and b=0.0.
    - This helper only builds candidate shared isotropic kernels for the 1D
      lag-based LMC fitting stage; it does not fit the B matrices.
    - Any anisotropy used later for cokriging is handled separately during
      covariance assembly.
    """
    model_family = str(model_family).lower().strip()
    model_type = str(model_type).lower().strip()

    if model_family not in ("variogram", "correlation"):
        raise ValueError("model_family must be 'variogram' or 'correlation'.")

    range_key = _lmc_range_param_name(model_type)
    shape_keys = _lmc_shape_param_names(model_type)

    param_sets = [dict(params_primary), dict(params_secondary)]
    if params_cross is not None:
        param_sets.append(dict(params_cross))

    # ------------------------------------------------------------
    # Range candidates
    # ------------------------------------------------------------
    range_vals = []
    for p in param_sets:
        if range_key in p and np.isfinite(p[range_key]) and float(p[range_key]) > 0.0:
            range_vals.append(float(p[range_key]))

    if len(range_vals) < 2:
        raise ValueError(
            f"Need at least two valid '{range_key}' values across direct fitted models."
        )

    rlo = min(range_vals)
    rhi = max(range_vals)

    if np.isclose(rlo, rhi):
        range_min = 0.75 * rlo
        range_max = 1.25 * rhi
    else:
        range_min = max(1e-12, rlo * (1.0 - range_padding_frac))
        range_max = rhi * (1.0 + range_padding_frac)

    if range_spacing == "linear":
        range_grid = np.linspace(range_min, range_max, int(n_ranges))
    elif range_spacing == "log":
        range_grid = np.geomspace(range_min, range_max, int(n_ranges))
    else:
        raise ValueError("range_spacing must be 'linear' or 'log'.")

    range_candidates = list(range_grid)
    if include_direct_ranges:
        range_candidates.extend(range_vals)
    if extra_ranges is not None:
        range_candidates.extend(np.asarray(extra_ranges, float).ravel().tolist())

    range_candidates = _unique_sorted_floats(range_candidates, positive_only=True)
    if range_candidates.size == 0:
        raise ValueError("No valid candidate ranges were constructed.")

    # ------------------------------------------------------------
    # Shape candidates
    # ------------------------------------------------------------
    extra_shape_values = {} if extra_shape_values is None else dict(extra_shape_values)
    shape_candidate_lists = []

    for sk in shape_keys:
        vals = []
        for p in param_sets:
            if sk in p and np.isfinite(p[sk]) and float(p[sk]) > 0.0:
                vals.append(float(p[sk]))

        if len(vals) == 0:
            raise ValueError(
                f"No valid candidate values were found for shape parameter '{sk}'."
            )

        if include_shape_average:
            vals.append(float(np.mean(vals)))

        if sk in extra_shape_values and extra_shape_values[sk] is not None:
            vals.extend(np.asarray(extra_shape_values[sk], float).ravel().tolist())

        vals = _unique_sorted_floats(vals, positive_only=True)
        if vals.size == 0:
            raise ValueError(
                f"No valid candidate values were constructed for shape parameter '{sk}'."
            )

        shape_candidate_lists.append(vals.tolist())

    if len(shape_candidate_lists) == 0:
        shape_combos = [tuple()]
    else:
        shape_combos = list(product(*shape_candidate_lists))

    # ------------------------------------------------------------
    # Build kernel library
    # ------------------------------------------------------------
    kernel_library = []
    counter = 1

    for rval in range_candidates:
        for shape_vals in shape_combos:
            if model_family == "variogram":
                params = {range_key: float(rval), "c0": 1.0, "b": 0.0}
            else:
                params = {range_key: float(rval), "alpha": 1.0}

            for sk, sv in zip(shape_keys, shape_vals):
                params[sk] = float(sv)

            label_parts = [f"{prefix}{counter}", f"{range_key}={float(rval):.4g}"]
            for sk, sv in zip(shape_keys, shape_vals):
                label_parts.append(f"{sk}={float(sv):.4g}")

            kernel_library.append(
                make_lmc_kernel_spec(
                    model_family=model_family,
                    model_type=model_type,
                    params=params,
                    name=", ".join(label_parts),
                )
            )
            counter += 1

    return kernel_library

def build_candidate_kernel_sets(
    kernel_library,
    *,
    n_structures: int = 2,
    allow_repeated: bool = False,
    sort_within_set: bool = True,
):
    """
    Turn a scalar kernel library into candidate shared-kernel sets.

    Examples
    --------
    n_structures=1 -> [[K1], [K2], ...]
    n_structures=2 -> [[K1,K2], [K1,K3], ...]
    """
    kernel_library = list(kernel_library)

    if len(kernel_library) == 0:
        raise ValueError("kernel_library must contain at least one kernel.")

    if n_structures < 1:
        raise ValueError("n_structures must be >= 1.")

    if n_structures == 1:
        return [[k] for k in kernel_library]

    if allow_repeated:
        comb_iter = combinations_with_replacement(range(len(kernel_library)), n_structures)
    else:
        comb_iter = combinations(range(len(kernel_library)), n_structures)

    out = []
    for idxs in comb_iter:
        ks = [kernel_library[i] for i in idxs]

        if sort_within_set:
            ks = sorted(
                ks,
                key=lambda s: _kernel_summary_dict(s)["range_value"]
            )

        out.append(ks)

    if len(out) == 0:
        raise ValueError("No candidate kernel sets were generated.")

    return out

def search_lmc_model_space(
    h,
    rho_primary,
    rho_secondary,
    rho_cross,
    *,
    fixed_structures=None,
    candidate_kernel_sets=None,
    model_family: Optional[str] = None,
    model_type: Optional[str] = None,
    params_primary: Optional[dict] = None,
    params_secondary: Optional[dict] = None,
    params_cross: Optional[dict] = None,
    n_ranges: int = 7,
    range_spacing: str = "linear",
    range_padding_frac: float = 0.15,
    include_direct_ranges: bool = True,
    extra_ranges=None,
    include_shape_average: bool = True,
    extra_shape_values: Optional[dict] = None,
    n_structures: int = 2,
    allow_repeated_kernels: bool = False,
    w_primary=None,
    w_secondary=None,
    w_cross=None,
    rho0_target=None,
    rho0_bounds=None,
    cross_misfit_weight_grid=(0.25, 0.5, 1.0, 2.0, 5.0),
    rho0_penalty_weight_grid=(0.0, 0.01, 0.1, 1.0, 10.0),
    require_feasible: bool = False,
    prefer_feasible: bool = True,
    normalize_weights: bool = True,
    plot_best: bool = False,
    plot_best_diagnostic: bool = False,
    show_hidden_diagnostic: bool = False,
    plot_ylim=(-1.0, 1.0),
    smooth_n: int = 500,
    options=None,
    verbose: bool = True,
    show_progress: bool = True,
):
    """
    Search over shared LMC kernel sets, cross-misfit weights, and optional
    rho0-penalty weights.

    The outer search varies:
        1. the candidate shared-kernel set,
        2. cross_misfit_weight,
        3. rho0_penalty_weight

    unless rho0_bounds is supplied while rho0_target is None, in which case the
    rho0 penalty is inactive and only a single rho0_penalty_weight=0.0 branch is
    evaluated.

    Parameters
    ----------
    candidate_kernel_sets : sequence or None, default None
        Candidate shared-kernel sets. If None, they are constructed automatically
        from params_primary / params_secondary / params_cross.
    rho0_target : float or None, default None
        Optional target collocated cross-correlation. If supplied, rho0_bounds
        must also be supplied.
    rho0_bounds : sequence of 2 floats or None, default None
        Optional lower/upper bounds imposed on rho0_fit during the inner fit.
        If supplied while rho0_target is None, the search performs constrained
        fits without any rho0-target penalty.
    require_feasible : bool, default False
        If True, reject infeasible kernel sets during the inner fit.
    prefer_feasible : bool, default True
        If True, rank feasible fits ahead of infeasible fits when selecting the best.
    plot_best : bool, default False
        If True, plot the best fit after the search.
    plot_best_diagnostic : bool, default False
        If True and plot_best=True, also show the individual LMC components.

    Returns
    -------
    out : dict
        Dictionary with keys:
            'best'                 : best successful search record
            'records'              : sorted successful search records
            'candidate_kernel_sets': evaluated candidate kernel sets
            'direct_models'        : auto-built direct models used for plotting, if any

    Notes
    -----
    This search is performed on the isotropic 1D lag-based LMC fitting problem.
    It does not search over anisotropy transforms. Per-structure anisotropy, if
    used, is introduced later during covariance assembly in
    simple_cokriging(..., covariance_mode='lmc', rotation_matrix=...) or
    ordinary_cokriging(..., covariance_mode='lmc', rotation_matrix=...).
    """
    h = np.asarray(h, float).ravel()
    rz = np.asarray(rho_primary, float).ravel()
    ry = np.asarray(rho_secondary, float).ravel()
    rzy = np.asarray(rho_cross, float).ravel()

    if not (h.size == rz.size == ry.size == rzy.size):
        raise ValueError("h, rho_primary, rho_secondary, and rho_cross must have the same length.")

    if rho0_target is not None and rho0_bounds is None:
        raise ValueError(
            "If rho0_target is supplied, rho0_bounds must also be supplied."
        )

    fixed_structures = [] if fixed_structures is None else list(fixed_structures)

    # ------------------------------------------------------------
    # Auto-build direct models for default plotting
    # ------------------------------------------------------------

    direct_models = None
    if (
            model_family is not None
            and model_type is not None
            and params_primary is not None
            and params_secondary is not None
    ):
        cross_scale_plot = None

        if params_cross is not None and rho0_target is not None:
            family_l = str(model_family).lower().strip()

            if family_l == "correlation":
                cross_scale_plot = rho0_target

            elif family_l == "variogram":
                c0_cross = float(params_cross.get("c0", np.nan))
                b_cross = float(params_cross.get("b", np.nan))

                if (
                        np.isfinite(c0_cross)
                        and np.isfinite(b_cross)
                        and np.isclose(c0_cross + b_cross, 1.0)
                ):
                    cross_scale_plot = rho0_target

        direct_models = _build_direct_models_from_params(
            model_family=model_family,
            model_type=model_type,
            params_primary=params_primary,
            params_secondary=params_secondary,
            params_cross=params_cross,
            cross_scale=cross_scale_plot,
        )

    # ------------------------------------------------------------
    # Build candidate kernel sets if not supplied
    # ------------------------------------------------------------
    if candidate_kernel_sets is None:
        if model_family is None or model_type is None:
            raise ValueError(
                "If candidate_kernel_sets is not supplied, then model_family and model_type must be provided."
            )
        if params_primary is None or params_secondary is None:
            raise ValueError(
                "If candidate_kernel_sets is not supplied, then params_primary and params_secondary must be provided."
            )

        kernel_library = build_candidate_shared_kernel_library(
            model_family=model_family,
            model_type=model_type,
            params_primary=params_primary,
            params_secondary=params_secondary,
            params_cross=params_cross,
            n_ranges=n_ranges,
            range_spacing=range_spacing,
            range_padding_frac=range_padding_frac,
            include_direct_ranges=include_direct_ranges,
            extra_ranges=extra_ranges,
            include_shape_average=include_shape_average,
            extra_shape_values=extra_shape_values,
            prefix="K",
        )

        candidate_kernel_sets = build_candidate_kernel_sets(
            kernel_library,
            n_structures=n_structures,
            allow_repeated=allow_repeated_kernels,
            sort_within_set=True,
        )
    else:
        candidate_kernel_sets = [list(ks) for ks in candidate_kernel_sets]

    cross_grid = [float(v) for v in cross_misfit_weight_grid]

    if rho0_bounds is not None and rho0_target is None:
        rho0_grid = [0.0]
    else:
        rho0_grid = [float(v) for v in rho0_penalty_weight_grid]

    records = []
    best = None
    n_total = len(candidate_kernel_sets) * len(cross_grid) * len(rho0_grid)
    counter = 0

    kernel_sets_iter = candidate_kernel_sets
    if show_progress:
        kernel_sets_iter = tqdm(
            candidate_kernel_sets,
            total=len(candidate_kernel_sets),
            desc="Kernel sets"
        )

    def _rank_key(rec):
        feas_rank = 0 if rec.get("is_feasible", False) else 1
        if prefer_feasible:
            return (feas_rank, rec["objective"])
        return (rec["objective"], feas_rank)

    for iks, kernel_set in enumerate(kernel_sets_iter):
        kernel_meta = [_kernel_summary_dict(k) for k in kernel_set]

        for cross_w in cross_grid:
            for rho0_w in rho0_grid:
                counter += 1

                try:
                    fit = fit_lmc_coregionalization(
                        h,
                        rz,
                        ry,
                        rzy,
                        free_kernels=kernel_set,
                        fixed_structures=fixed_structures,
                        w_primary=w_primary,
                        w_secondary=w_secondary,
                        w_cross=w_cross,
                        rho0_target=rho0_target,
                        rho0_bounds=rho0_bounds,
                        rho0_penalty_weight=rho0_w,
                        cross_misfit_weight=cross_w,
                        require_feasible=require_feasible,
                        normalize_weights=normalize_weights,
                        plot=False,
                        plot_ylim=plot_ylim,
                        smooth_n=smooth_n,
                        options=options,
                    )

                    rec = {
                        "status": "ok",
                        "candidate_index": iks,
                        "cross_misfit_weight": float(cross_w),
                        "rho0_penalty_weight": float(rho0_w),
                        "rho0_target": None if rho0_target is None else float(rho0_target),
                        "rho0_bounds": None if rho0_bounds is None else tuple(float(v) for v in rho0_bounds),
                        "rho0_constraint_active": bool(rho0_bounds is not None),
                        "n_structures_free": int(len(kernel_set)),
                        "objective": float(fit["objective"]),
                        "rho0_fit": float(fit["rho0_fit"]),
                        "misfit_primary": float(fit["misfit_primary"]),
                        "misfit_secondary": float(fit["misfit_secondary"]),
                        "misfit_cross": float(fit["misfit_cross"]),
                        "rho0_penalty": float(fit["rho0_penalty"]),
                        "n_bad_z": int(fit["n_bad_z"]),
                        "n_bad_y": int(fit["n_bad_y"]),
                        "n_bad_total": int(fit["n_bad_total"]),
                        "is_feasible": bool(fit["is_feasible"]),
                        "kernel_meta": kernel_meta,
                        "fit": fit,
                    }
                    records.append(rec)

                    if (best is None) or (_rank_key(rec) < _rank_key(best)):
                        best = rec

                except Exception as e:
                    rec = {
                        "status": "failed",
                        "candidate_index": iks,
                        "cross_misfit_weight": float(cross_w),
                        "rho0_penalty_weight": float(rho0_w),
                        "rho0_target": None if rho0_target is None else float(rho0_target),
                        "rho0_bounds": None if rho0_bounds is None else tuple(float(v) for v in rho0_bounds),
                        "rho0_constraint_active": bool(rho0_bounds is not None),
                        "n_structures_free": int(len(kernel_set)),
                        "objective": np.inf,
                        "is_feasible": False,
                        "error": str(e),
                        "kernel_meta": kernel_meta,
                    }
                    records.append(rec)

                if verbose:
                    print(f"[{counter:>4d}/{n_total}] done")

    if best is None:
        raise RuntimeError("All LMC search candidates failed.")

    records_sorted = sorted(
        [r for r in records if r["status"] == "ok"],
        key=_rank_key,
    )

    if plot_best:
        plot_lmc_structures(
            h,
            rz,
            ry,
            rzy,
            structures=best["fit"]["structures"],
            direct_models=direct_models,
            plot_diagnostic=plot_best_diagnostic,
            show_hidden=show_hidden_diagnostic,
            smooth_n=smooth_n,
            ylim=plot_ylim,
        )

    return {
        "best": best,
        "records": records_sorted,
        "candidate_kernel_sets": candidate_kernel_sets,
        "direct_models": direct_models,
    }

def _validate_simple_cokriging_runtime_options(
    *,
    covariance_mode: str,
    distance_type: str,
    rotation_matrix,
    chunk_cols: Optional[int],
    max_neighbors_primary: Optional[int],
    max_neighbors_secondary: Optional[int],
    target_batch_size: Optional[int],
    balltree_leaf_size: int,
    jitter: float,
    n_lmc_structures: Optional[int] = None,
):
    """
    Validate shared runtime options for the simple/ordinary cokriging front ends.

    Parameters
    ----------
    chunk_cols : int or None
        Optional distance-chunk size. Must be None or a positive integer.
    max_neighbors_primary, max_neighbors_secondary : int or None
        Optional neighborhood sizes. Each must be None or a positive integer.
    target_batch_size : int or None
        Optional global batching size. Must be None or a positive integer.
        This option may later be ignored by local solves, but it is validated
        here for API consistency.
    balltree_leaf_size : int
        BallTree leaf size. Must be a positive integer.
    jitter : float
        Diagonal stabilization value added during covariance assembly. Must be
        finite and nonnegative.
    n_lmc_structures : int or None, default None
        Number of LMC structures. Required only when validating
        covariance_mode='lmc' with a non-None rotation_matrix.

    Notes
    -----
    This helper checks generic runtime arguments and rotation/anisotropy
    specifications. Compatibility of the covariance models themselves is
    checked later when assembling and solving the cokriging system.
    """
    if chunk_cols is not None:
        if not isinstance(chunk_cols, (int, np.integer)) or int(chunk_cols) <= 0:
            raise ValueError("chunk_cols must be None or a positive integer.")

    for name, value in (
        ("max_neighbors_primary", max_neighbors_primary),
        ("max_neighbors_secondary", max_neighbors_secondary),
    ):
        if value is not None:
            if not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"{name} must be None or a positive integer.")

    if target_batch_size is not None:
        if not isinstance(target_batch_size, (int, np.integer)) or int(target_batch_size) <= 0:
            raise ValueError("target_batch_size must be None or a positive integer.")

    if not isinstance(balltree_leaf_size, (int, np.integer)) or int(balltree_leaf_size) <= 0:
        raise ValueError("balltree_leaf_size must be a positive integer.")

    if not np.isfinite(jitter) or float(jitter) < 0.0:
        raise ValueError("jitter must be finite and nonnegative.")

    if rotation_matrix is not None and distance_type not in ("euclidean", "cartesian"):
        raise ValueError(
            "rotation_matrix is only supported for planar coordinates with "
            "distance_type='euclidean' or 'cartesian'."
        )

    if covariance_mode == "direct":
        if rotation_matrix is not None:
            if not isinstance(rotation_matrix, dict):
                raise ValueError(
                    "For covariance_mode='direct', rotation_matrix must be None "
                    "or a dict with keys {'azimuth', 'a_max', 'a_min'}."
                )
            _ = _build_rotation_matrix(rotation_matrix)

    elif covariance_mode == "lmc":
        if rotation_matrix is not None:
            if n_lmc_structures is None:
                raise ValueError(
                    "n_lmc_structures must be provided when validating "
                    "rotation_matrix for covariance_mode='lmc'."
                )
            _ = _resolve_lmc_rotation_matrices(rotation_matrix, n_lmc_structures)

    else:
        raise ValueError("covariance_mode must be 'direct' or 'lmc'.")

# simple cokriging: main
def _prepare_simple_cokriging_inputs(
    primary_values,
    primary_coords,
    secondary_values,
    secondary_coords,
    targets,
    *,
    distance_type: str,
    standardize: bool,
    mean_primary,
    mean_secondary,
):
    """
    Prepare cleaned values, coordinates, and working-scale moments for two-
    variable simple / ordinary cokriging.

    Parameters
    ----------
    primary_values, secondary_values : array_like
        Observed primary and secondary values.
    primary_coords, secondary_coords : array_like
        Coordinates associated with the primary and secondary samples.
    targets : array_like
        Prediction locations.
    distance_type : str
        Distance convention. The alias 'geographical' is normalized internally
        to 'geographic'.
    standardize : bool
        If True, both variables are standardized independently using their own
        sample means and sample standard deviations.
    mean_primary, mean_secondary : {'sample', 'zero'} or float
        Means used only when standardize=False.

    Returns
    -------
    out : dict
        Dictionary containing cleaned coordinates, working-scale values, target
        coordinates, and the primary back-transformation quantities.

    Notes
    -----
    - Coordinates are cleaned and coerced here, but any optional anisotropy
      transform is applied later during covariance assembly.
    - When standardize=True, the returned primary and secondary working values
      are both zero-mean and unit-variance on the sample scale.
    - Only the primary mean and standard deviation are carried forward for
      back-transformation, because the public prediction target is always the
      primary variable.
    - The returned mz and my are used only by the simple-cokriging paths when
      standardize=False; the ordinary-cokriging paths ignore them.
    """
    dt = str(distance_type).lower().strip()
    if dt == "geographical":
        dt = "geographic"

    z_raw, Xz = _clean_values_and_coords(
        primary_values, primary_coords, distance_type=dt, name="primary"
    )
    y_raw, Xy = _clean_values_and_coords(
        secondary_values, secondary_coords, distance_type=dt, name="secondary"
    )
    XT = _coerce_coordinates(targets, dt, "targets")

    if standardize:
        z, mu_z, sd_z = _standardize_values(z_raw, name="primary")
        y, _, _ = _standardize_values(y_raw, name="secondary")
        mz = 0.0
        my = 0.0
    else:
        z = z_raw.copy()
        y = y_raw.copy()
        mu_z = None
        sd_z = 1.0
        mz = _resolve_simple_mean(mean_primary, z, var_name="primary")
        my = _resolve_simple_mean(mean_secondary, y, var_name="secondary")

    return {
        "distance_type": dt,
        "z": z,
        "y": y,
        "Xz": Xz,
        "Xy": Xy,
        "XT": XT,
        "mz": mz,
        "my": my,
        "mu_z": mu_z,
        "sd_z": sd_z,
        "standardize": bool(standardize),
    }

def _prepare_collocated_secondary_at_targets(
    collocated_secondary_values,
    targets,
    *,
    secondary_training_values,
    standardize: bool,
):
    """
    Prepare the collocated secondary datum Y(u) used by SCCK / OCCK / ICCK /
    OICCK.

    Parameters
    ----------
    collocated_secondary_values : array_like, shape (m,)
        Secondary values collocated with the target locations.
    targets : ndarray, shape (m, d)
        Target coordinates. Only the number of targets is used here.
    secondary_training_values : array_like
        Reference secondary values defining the mean/scale used to place the
        collocated target-side secondary data on the collocated-cokriging
        working scale.
        When standardize=False, these values may still be used to define the
        reference secondary mean if mean_secondary='sample' in the calling code.
    standardize : bool
        If True, the returned target-side secondary values are standardized
        using the mean and sample standard deviation of
        secondary_training_values.

    Returns
    -------
    y0 : (m,) ndarray
        Collocated secondary values on the working scale used by the solver.
    mu_y : float or None
        Mean of the secondary training values when standardize=True, otherwise
        None.
    sd_y : float
        Sample standard deviation of the secondary training values when
        standardize=True, otherwise 1.0.
    """
    y0 = np.asarray(collocated_secondary_values, float).ravel()
    m = np.asarray(targets).shape[0]

    if y0.size != m:
        raise ValueError(
            "collocated_secondary_values must have the same length as targets."
        )

    if not np.all(np.isfinite(y0)):
        raise ValueError("collocated_secondary_values must be finite.")

    if standardize:
        mu_y, sd_y = _compute_standardization_stats(
            secondary_training_values,
            name="secondary",
        )
        y0 = (y0 - mu_y) / sd_y
    else:
        mu_y = None
        sd_y = 1.0

    return y0, mu_y, sd_y

def _evaluate_mm2_primary_unit_shape_from_coords(
    Xa: np.ndarray,
    Xb: np.ndarray,
    *,
    rho0: float,
    secondary_model: Optional[dict] = None,
    residual_model: Optional[dict] = None,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    chunk_cols: Optional[int] = None,
    same_coordinates: bool = False,
) -> np.ndarray:
    """
    Evaluate the MM2 primary unit-correlation shape

        rho_z_MM2(h) = rho0**2 * rho_y(h) + (1 - rho0**2) * rho_r(h)

    where rho_y(h) is the secondary correlogram and rho_r(h) is the residual
    correlogram fitted so that rho_z_MM2(h) approximates the experimental
    primary correlogram.

    Notes
    -----
    This helper currently supports only direct-mode covariance specifications.
    """
    if secondary_model is None:
        raise ValueError("MM2 requires secondary_model.")
    if residual_model is None:
        raise ValueError(
            "MM2 requires residual_model, i.e. the model for rho_r(h)."
        )

    # rho_y(h)
    Ry = _evaluate_direct_unit_shape_from_coords(
        Xa,
        Xb,
        covariance_mode="direct",
        component="secondary",
        primary_model=None,
        secondary_model=secondary_model,
        structures=None,
        distance_type=distance_type,
        projection=projection,
        rotation_matrix=rotation_matrix,
        chunk_cols=chunk_cols,
        same_coordinates=same_coordinates,
    )

    # rho_r(h) passed through the 'primary' slot of the low-level helper
    Rr = _evaluate_direct_unit_shape_from_coords(
        Xa,
        Xb,
        covariance_mode="direct",
        component="primary",
        primary_model=residual_model,
        secondary_model=None,
        structures=None,
        distance_type=distance_type,
        projection=projection,
        rotation_matrix=rotation_matrix,
        chunk_cols=chunk_cols,
        same_coordinates=same_coordinates,
    )

    w = float(rho0) ** 2
    return w * Ry + (1.0 - w) * Rr

def _assemble_scck_system(
    primary_coords: np.ndarray,
    targets: np.ndarray,
    *,
    markov_model: str,
    rho0: float,
    primary_model: Optional[dict] = None,
    secondary_model: Optional[dict] = None,
    residual_model: Optional[dict] = None,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    chunk_cols: Optional[int] = None,
    jitter: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble the SCCK system for a single target.

    Notes
    -----
    This helper is kept only as a thin wrapper for backward compatibility.
    SCCK is assembled target-by-target because the collocated row/column depends
    on the target location.
    """
    XT = _coerce_coordinates(targets, distance_type, "targets")
    if XT.shape[0] != 1:
        raise NotImplementedError(
            "_assemble_scck_system(...) supports only one target. "
            "Use _assemble_scck_system_one_target(...) in a loop."
        )

    return _assemble_scck_system_one_target(
        primary_coords,
        XT[0],
        markov_model=markov_model,
        rho0=rho0,
        primary_model=primary_model,
        secondary_model=secondary_model,
        residual_model=residual_model,
        distance_type=distance_type,
        projection=projection,
        rotation_matrix=rotation_matrix,
        chunk_cols=chunk_cols,
        jitter=jitter,
    )

def _assemble_scck_system_one_target(
    primary_coords: np.ndarray,
    target: np.ndarray,
    *,
    markov_model: str,
    rho0: float,
    primary_model: Optional[dict] = None,
    secondary_model: Optional[dict] = None,
    residual_model: Optional[dict] = None,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    chunk_cols: Optional[int] = None,
    jitter: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble the single-target SCCK linear system on the unit-correlation scale.

    Parameters
    ----------
    primary_coords : ndarray
        Coordinates of the primary samples used for this local or global solve.
    target : ndarray
        One target location.
    markov_model : {'mm1', 'mm2'}
        Markov closure used for SCCK.
    rho0 : float
        Collocated primary-secondary correlation.
    primary_model : dict or None
        Direct primary covariance/correlation model rho_z(h), required for MM1.
    secondary_model : dict or None
        Direct secondary covariance/correlation model rho_y(h), required for MM2.
    residual_model : dict or None
        Direct residual covariance/correlation model rho_r(h), required for MM2.
    distance_type : {'geographic', 'euclidean', 'cartesian', 'angular'}
        Distance convention used when evaluating the model blocks.
    projection : str, default 'WGS84'
        Projection / ellipsoid argument passed through where relevant.
    rotation_matrix : dict or None, default None
        Optional planar anisotropy specification with keys
        {'azimuth', 'a_max', 'a_min'}.
    chunk_cols : int or None, default None
        Optional chunk size passed to pairwise_distances(...).
    jitter : float, default 1e-10
        Nonnegative diagonal stabilization added to the SCCK system.

    Returns
    -------
    K : (nz+1, nz+1) ndarray
        SCCK coefficient matrix for one target.
    r : (nz+1,) ndarray
        Right-hand side for one target.

    Notes
    -----
    This implementation is direct-only.

    The returned K / r pair is shared by both SCCK and OCCK:
    - SCCK solves K w = r.
    - OCCK solves K w + 1 mu = r with a single ordinary constraint.

    - MM1 uses rho_z(h) for the primary direct shape and rho_zy(h) = rho0 * rho_z(h).
    - MM2 uses rho_y(h) and rho_r(h) to reconstruct
      rho_z(h) = rho0**2 * rho_y(h) + (1 - rho0**2) * rho_r(h),
      and rho_zy(h) = rho0 * rho_y(h).
    """
    Xz = _coerce_coordinates(primary_coords, distance_type, "primary_coords")
    XT = _coerce_coordinates(np.asarray(target).reshape(1, -1), distance_type, "target")

    markov_model = str(markov_model).lower().strip()

    if markov_model == "mm1":
        if primary_model is None:
            raise ValueError("MM1 requires primary_model.")

        Rzz = _evaluate_direct_unit_shape_from_coords(
            Xz,
            Xz,
            covariance_mode="direct",
            component="primary",
            primary_model=primary_model,
            secondary_model=None,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=True,
        )

        rz0 = _evaluate_direct_unit_shape_from_coords(
            Xz,
            XT,
            covariance_mode="direct",
            component="primary",
            primary_model=primary_model,
            secondary_model=None,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )[:, 0]

        cy0 = rho0 * rz0

    elif markov_model == "mm2":
        if secondary_model is None:
            raise ValueError("MM2 requires secondary_model.")
        if residual_model is None:
            raise ValueError("MM2 requires residual_model.")

        Rzz = _evaluate_mm2_primary_unit_shape_from_coords(
            Xz,
            Xz,
            rho0=rho0,
            secondary_model=secondary_model,
            residual_model=residual_model,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=True,
        )

        rz0 = _evaluate_mm2_primary_unit_shape_from_coords(
            Xz,
            XT,
            rho0=rho0,
            secondary_model=secondary_model,
            residual_model=residual_model,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )[:, 0]

        cy0 = rho0 * _evaluate_direct_unit_shape_from_coords(
            Xz,
            XT,
            covariance_mode="direct",
            component="secondary",
            primary_model=None,
            secondary_model=secondary_model,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )[:, 0]

    else:
        raise ValueError("markov_model must be 'mm1' or 'mm2'.")

    nz = Xz.shape[0]

    K = np.zeros((nz + 1, nz + 1), dtype=float)
    K[:nz, :nz] = Rzz
    np.fill_diagonal(K[:nz, :nz], 1.0)
    K[np.diag_indices(nz)] += jitter

    K[:nz, nz] = cy0
    K[nz, :nz] = cy0
    K[nz, nz] = 1.0 + jitter

    r = np.concatenate([rz0, [rho0]])
    return K, r

def _assemble_icck_system_one_target(
    primary_coords: np.ndarray,
    secondary_coords: np.ndarray,
    target: np.ndarray,
    *,
    markov_model: str,
    rho0: float,
    primary_model: Optional[dict] = None,
    secondary_model: Optional[dict] = None,
    residual_model: Optional[dict] = None,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    chunk_cols: Optional[int] = None,
    jitter: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble the single-target ICCK linear system on the unit-correlation scale.

    Parameters
    ----------
    primary_coords, secondary_coords : ndarray
        Coordinates of the primary and secondary samples used for this local or
        global solve.
    target : ndarray
        One target location.
    markov_model : {'mm1', 'mm2'}
        Markov closure used for the primary-secondary cross blocks.
    rho0 : float
        Collocated primary-secondary correlation.
    primary_model : dict or None
        Direct primary covariance/correlation model rho_z(h), required for MM1.
        Under MM1, the same model is also used for rho_y(h), consistent with the
        intrinsic Markov I closure.
    secondary_model : dict or None
        Direct secondary covariance/correlation model rho_y(h), required for MM2.
    residual_model : dict or None
        Direct residual covariance/correlation model rho_r(h), required for MM2.
    distance_type : {'geographic', 'euclidean', 'cartesian', 'angular'}
        Distance convention used when evaluating the model blocks.
    projection : str, default 'WGS84'
        Projection / ellipsoid argument passed through where relevant.
    rotation_matrix : dict or None, default None
        Optional planar anisotropy specification with keys
        {'azimuth', 'a_max', 'a_min'}.
    chunk_cols : int or None, default None
        Optional chunk size passed to pairwise_distances(...).
    jitter : float, default 1e-10
        Nonnegative diagonal stabilization added to the ICCK system.

    Returns
    -------
    K : (nz+ny+1, nz+ny+1) ndarray
        ICCK coefficient matrix for one target.
    r : (nz+ny+1,) ndarray
        Right-hand side for one target.

    Notes
    -----
    This implementation is direct-only.

    The returned K / r pair is shared by both ICCK and OICCK:
    - ICCK solves K w = r.
    - OICCK solves K w + 1 mu = r with a single ordinary constraint.

    The assembled system combines neighboring primary samples, neighboring
    secondary samples, and one collocated secondary datum at the target. The
    system is built on the standardized unit-correlation scale.

    - MM1 uses rho_z(h) for both direct blocks and rho_zy(h) = rho0 * rho_z(h).
    - MM2 uses rho_y(h) and rho_r(h) to reconstruct
      rho_z(h) = rho0**2 * rho_y(h) + (1 - rho0**2) * rho_r(h),
      and rho_zy(h) = rho0 * rho_y(h).
    """
    Xz = _coerce_coordinates(primary_coords, distance_type, "primary_coords")
    Xy = _coerce_coordinates(secondary_coords, distance_type, "secondary_coords")
    XT = _coerce_coordinates(np.asarray(target).reshape(1, -1), distance_type, "target")

    markov_model = str(markov_model).lower().strip()

    if markov_model == "mm1":
        if primary_model is None:
            raise ValueError("MM1 requires primary_model.")

        Rzz = _evaluate_direct_unit_shape_from_coords(
            Xz,
            Xz,
            covariance_mode="direct",
            component="primary",
            primary_model=primary_model,
            secondary_model=None,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=True,
        )
        rz0 = _evaluate_direct_unit_shape_from_coords(
            Xz,
            XT,
            covariance_mode="direct",
            component="primary",
            primary_model=primary_model,
            secondary_model=None,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )[:, 0]

        # MM1 assumes rho_y(h) = rho_z(h)
        Ryy = _evaluate_direct_unit_shape_from_coords(
            Xy,
            Xy,
            covariance_mode="direct",
            component="primary",
            primary_model=primary_model,
            secondary_model=None,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=True,
        )
        Rzy = rho0 * _evaluate_direct_unit_shape_from_coords(
            Xz,
            Xy,
            covariance_mode="direct",
            component="primary",
            primary_model=primary_model,
            secondary_model=None,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )
        ry0 = _evaluate_direct_unit_shape_from_coords(
            Xy,
            XT,
            covariance_mode="direct",
            component="primary",
            primary_model=primary_model,
            secondary_model=None,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )[:, 0]
        rzy0 = rho0 * ry0

    elif markov_model == "mm2":
        if secondary_model is None:
            raise ValueError("MM2 requires secondary_model.")
        if residual_model is None:
            raise ValueError("MM2 requires residual_model.")

        # 1) primary-primary uses rho_z_MMII
        Rzz = _evaluate_mm2_primary_unit_shape_from_coords(
            Xz,
            Xz,
            rho0=rho0,
            secondary_model=secondary_model,
            residual_model=residual_model,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=True,
        )

        # 5) primary-to-target uses rho_z_MMII
        rz0 = _evaluate_mm2_primary_unit_shape_from_coords(
            Xz,
            XT,
            rho0=rho0,
            secondary_model=secondary_model,
            residual_model=residual_model,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )[:, 0]

        # 4) secondary-secondary uses rho_y
        Ryy = _evaluate_direct_unit_shape_from_coords(
            Xy,
            Xy,
            covariance_mode="direct",
            component="secondary",
            primary_model=None,
            secondary_model=secondary_model,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=True,
        )

        # 2,3) primary-secondary uses rho0 * rho_y
        Rzy = rho0 * _evaluate_direct_unit_shape_from_coords(
            Xz,
            Xy,
            covariance_mode="direct",
            component="secondary",
            primary_model=None,
            secondary_model=secondary_model,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )

        # 9) collocated-secondary link uses rho_y
        ry0 = _evaluate_direct_unit_shape_from_coords(
            Xy,
            XT,
            covariance_mode="direct",
            component="secondary",
            primary_model=None,
            secondary_model=secondary_model,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )[:, 0]

        # 8) collocated-primary link uses rho0 * rho_y
        rzy0 = rho0 * _evaluate_direct_unit_shape_from_coords(
            Xz,
            XT,
            covariance_mode="direct",
            component="secondary",
            primary_model=None,
            secondary_model=secondary_model,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )[:, 0]

    else:
        raise ValueError("markov_model must be 'mm1' or 'mm2'.")

    nz = Xz.shape[0]
    ny = Xy.shape[0]

    K = np.zeros((nz + ny + 1, nz + ny + 1), dtype=float)
    K[:nz, :nz] = Rzz
    K[:nz, nz:nz+ny] = Rzy
    K[nz:nz+ny, :nz] = Rzy.T
    K[nz:nz+ny, nz:nz+ny] = Ryy

    np.fill_diagonal(K[:nz, :nz], 1.0)
    np.fill_diagonal(K[nz:nz+ny, nz:nz+ny], 1.0)

    ii_z = np.arange(nz)
    ii_y = np.arange(ny)
    K[ii_z, ii_z] += jitter
    K[nz + ii_y, nz + ii_y] += jitter

    K[:nz, nz+ny] = rzy0
    K[nz+ny, :nz] = rzy0
    K[nz:nz+ny, nz+ny] = ry0
    K[nz+ny, nz:nz+ny] = ry0
    K[nz+ny, nz+ny] = 1.0 + jitter

    r = np.concatenate([rz0, rho0 * ry0, [rho0]])
    return K, r

def _assemble_scck_base_conditioning_matrix(
    primary_coords: np.ndarray,
    *,
    markov_model: str,
    rho0: float,
    primary_model: Optional[dict] = None,
    secondary_model: Optional[dict] = None,
    residual_model: Optional[dict] = None,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    chunk_cols: Optional[int] = None,
    jitter: float = 1e-10,
) -> np.ndarray:
    """
    Assemble the target-invariant SCCK neighbor block A.

    This is the primary-primary block only. The collocated-secondary row/column
    is target-dependent and is handled separately.
    """
    Xz = _coerce_coordinates(primary_coords, distance_type, "primary_coords")
    markov_model = str(markov_model).lower().strip()

    if markov_model == "mm1":
        if primary_model is None:
            raise ValueError("MM1 requires primary_model.")

        A = _evaluate_direct_unit_shape_from_coords(
            Xz,
            Xz,
            covariance_mode="direct",
            component="primary",
            primary_model=primary_model,
            secondary_model=None,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=True,
        )

    elif markov_model == "mm2":
        if secondary_model is None:
            raise ValueError("MM2 requires secondary_model.")
        if residual_model is None:
            raise ValueError("MM2 requires residual_model.")

        A = _evaluate_mm2_primary_unit_shape_from_coords(
            Xz,
            Xz,
            rho0=rho0,
            secondary_model=secondary_model,
            residual_model=residual_model,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=True,
        )

    else:
        raise ValueError("markov_model must be 'mm1' or 'mm2'.")

    A = np.asarray(A, float)
    ii = np.diag_indices(A.shape[0])
    A[ii] = 1.0
    A[ii] += jitter
    return A


def _assemble_scck_target_blocks(
    primary_coords: np.ndarray,
    targets: np.ndarray,
    *,
    markov_model: str,
    rho0: float,
    primary_model: Optional[dict] = None,
    secondary_model: Optional[dict] = None,
    residual_model: Optional[dict] = None,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    chunk_cols: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble the target-dependent SCCK blocks for one or more targets.

    Returns
    -------
    b0 : (nz, m) ndarray
        Top RHS block rz0.
    c0 : (nz, m) ndarray
        Target-dependent collocated coupling cy0.
    """
    Xz = _coerce_coordinates(primary_coords, distance_type, "primary_coords")
    XT = _coerce_coordinates(targets, distance_type, "targets")
    markov_model = str(markov_model).lower().strip()

    if markov_model == "mm1":
        if primary_model is None:
            raise ValueError("MM1 requires primary_model.")

        rz0 = _evaluate_direct_unit_shape_from_coords(
            Xz,
            XT,
            covariance_mode="direct",
            component="primary",
            primary_model=primary_model,
            secondary_model=None,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )
        cy0 = float(rho0) * rz0

    elif markov_model == "mm2":
        if secondary_model is None:
            raise ValueError("MM2 requires secondary_model.")
        if residual_model is None:
            raise ValueError("MM2 requires residual_model.")

        rz0 = _evaluate_mm2_primary_unit_shape_from_coords(
            Xz,
            XT,
            rho0=rho0,
            secondary_model=secondary_model,
            residual_model=residual_model,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )

        ry0_for_cross = _evaluate_direct_unit_shape_from_coords(
            Xz,
            XT,
            covariance_mode="direct",
            component="secondary",
            primary_model=None,
            secondary_model=secondary_model,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )
        cy0 = float(rho0) * ry0_for_cross

    else:
        raise ValueError("markov_model must be 'mm1' or 'mm2'.")

    return np.asarray(rz0, float), np.asarray(cy0, float)


def _assemble_icck_base_conditioning_matrix(
    primary_coords: np.ndarray,
    secondary_coords: np.ndarray,
    *,
    markov_model: str,
    rho0: float,
    primary_model: Optional[dict] = None,
    secondary_model: Optional[dict] = None,
    residual_model: Optional[dict] = None,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    chunk_cols: Optional[int] = None,
    jitter: float = 1e-10,
) -> np.ndarray:
    """
    Assemble the target-invariant ICCK neighbor block A.

    This includes the primary-primary, primary-secondary, and secondary-secondary
    blocks only. The collocated-secondary row/column is target-dependent and is
    handled separately.
    """
    Xz = _coerce_coordinates(primary_coords, distance_type, "primary_coords")
    Xy = _coerce_coordinates(secondary_coords, distance_type, "secondary_coords")
    markov_model = str(markov_model).lower().strip()

    if markov_model == "mm1":
        if primary_model is None:
            raise ValueError("MM1 requires primary_model.")

        Rzz = _evaluate_direct_unit_shape_from_coords(
            Xz,
            Xz,
            covariance_mode="direct",
            component="primary",
            primary_model=primary_model,
            secondary_model=None,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=True,
        )
        Ryy = _evaluate_direct_unit_shape_from_coords(
            Xy,
            Xy,
            covariance_mode="direct",
            component="primary",
            primary_model=primary_model,
            secondary_model=None,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=True,
        )
        Rzy = float(rho0) * _evaluate_direct_unit_shape_from_coords(
            Xz,
            Xy,
            covariance_mode="direct",
            component="primary",
            primary_model=primary_model,
            secondary_model=None,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )

    elif markov_model == "mm2":
        if secondary_model is None:
            raise ValueError("MM2 requires secondary_model.")
        if residual_model is None:
            raise ValueError("MM2 requires residual_model.")

        Rzz = _evaluate_mm2_primary_unit_shape_from_coords(
            Xz,
            Xz,
            rho0=rho0,
            secondary_model=secondary_model,
            residual_model=residual_model,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=True,
        )
        Ryy = _evaluate_direct_unit_shape_from_coords(
            Xy,
            Xy,
            covariance_mode="direct",
            component="secondary",
            primary_model=None,
            secondary_model=secondary_model,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=True,
        )
        Rzy = float(rho0) * _evaluate_direct_unit_shape_from_coords(
            Xz,
            Xy,
            covariance_mode="direct",
            component="secondary",
            primary_model=None,
            secondary_model=secondary_model,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )

    else:
        raise ValueError("markov_model must be 'mm1' or 'mm2'.")

    nz = Xz.shape[0]
    ny = Xy.shape[0]

    A = np.zeros((nz + ny, nz + ny), dtype=float)
    A[:nz, :nz] = np.asarray(Rzz, float)
    A[:nz, nz:] = np.asarray(Rzy, float)
    A[nz:, :nz] = np.asarray(Rzy, float).T
    A[nz:, nz:] = np.asarray(Ryy, float)

    ii_z = np.arange(nz)
    ii_y = np.arange(ny)
    A[ii_z, ii_z] = 1.0 + jitter
    A[nz + ii_y, nz + ii_y] = 1.0 + jitter

    return A


def _assemble_icck_target_blocks(
    primary_coords: np.ndarray,
    secondary_coords: np.ndarray,
    targets: np.ndarray,
    *,
    markov_model: str,
    rho0: float,
    primary_model: Optional[dict] = None,
    secondary_model: Optional[dict] = None,
    residual_model: Optional[dict] = None,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    chunk_cols: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble the target-dependent ICCK blocks for one or more targets.

    Returns
    -------
    b0 : (nz+ny, m) ndarray
        Upper RHS block [rz0; rho0 * ry0].
    c0 : (nz+ny, m) ndarray
        Target-dependent collocated coupling [rzy0; ry0].
    """
    Xz = _coerce_coordinates(primary_coords, distance_type, "primary_coords")
    Xy = _coerce_coordinates(secondary_coords, distance_type, "secondary_coords")
    XT = _coerce_coordinates(targets, distance_type, "targets")
    markov_model = str(markov_model).lower().strip()

    if markov_model == "mm1":
        if primary_model is None:
            raise ValueError("MM1 requires primary_model.")

        rz0 = _evaluate_direct_unit_shape_from_coords(
            Xz,
            XT,
            covariance_mode="direct",
            component="primary",
            primary_model=primary_model,
            secondary_model=None,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )
        ry0 = _evaluate_direct_unit_shape_from_coords(
            Xy,
            XT,
            covariance_mode="direct",
            component="primary",
            primary_model=primary_model,
            secondary_model=None,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )
        rzy0 = float(rho0) * ry0

    elif markov_model == "mm2":
        if secondary_model is None:
            raise ValueError("MM2 requires secondary_model.")
        if residual_model is None:
            raise ValueError("MM2 requires residual_model.")

        rz0 = _evaluate_mm2_primary_unit_shape_from_coords(
            Xz,
            XT,
            rho0=rho0,
            secondary_model=secondary_model,
            residual_model=residual_model,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )
        ry0 = _evaluate_direct_unit_shape_from_coords(
            Xy,
            XT,
            covariance_mode="direct",
            component="secondary",
            primary_model=None,
            secondary_model=secondary_model,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )
        rzy0 = float(rho0) * _evaluate_direct_unit_shape_from_coords(
            Xz,
            XT,
            covariance_mode="direct",
            component="secondary",
            primary_model=None,
            secondary_model=secondary_model,
            structures=None,
            distance_type=distance_type,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            same_coordinates=False,
        )

    else:
        raise ValueError("markov_model must be 'mm1' or 'mm2'.")

    b0 = np.vstack([np.asarray(rz0, float), float(rho0) * np.asarray(ry0, float)])
    c0 = np.vstack([np.asarray(rzy0, float), np.asarray(ry0, float)])
    return b0, c0


def _solve_batched_collocated_simple_from_factor(
    cf,
    b0: np.ndarray,
    c0: np.ndarray,
    *,
    rhs_tail: float,
    tail_diag: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve a batch of simple collocated systems using a pre-factored base block A.

    For each target:
        [[A, c],
         [cᵀ, d]] [w_top] = [b]
                    [w_c ]   [rhs_tail]
    """
    b0 = np.asarray(b0, float)
    c0 = np.asarray(c0, float)

    if b0.ndim == 1:
        b0 = b0[:, None]
    if c0.ndim == 1:
        c0 = c0[:, None]

    if b0.shape != c0.shape:
        raise ValueError("b0 and c0 must have the same shape.")

    Ab = cho_solve(cf, b0, check_finite=False)
    Ac = cho_solve(cf, c0, check_finite=False)

    schur = float(tail_diag) - np.einsum("ij,ij->j", c0, Ac)
    if np.any(~np.isfinite(schur)) or np.any(schur <= 0.0):
        raise np.linalg.LinAlgError(
            "Collocated cokriging Schur complement is not finite and positive."
        )

    w_tail = (float(rhs_tail) - np.einsum("ij,ij->j", c0, Ab)) / schur
    w_top = Ab - Ac * w_tail[None, :]

    return w_top, np.asarray(w_tail, float).ravel()


def _solve_batched_collocated_ordinary_from_factor(
    cf,
    b0: np.ndarray,
    c0: np.ndarray,
    *,
    rhs_tail: float,
    tail_diag: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve a batch of ordinary collocated systems using a pre-factored base block A.

    The unbiasedness constraint is:
        1ᵀ w_top + w_tail = 1
    """
    b0 = np.asarray(b0, float)
    c0 = np.asarray(c0, float)

    if b0.ndim == 1:
        b0 = b0[:, None]
    if c0.ndim == 1:
        c0 = c0[:, None]

    if b0.shape != c0.shape:
        raise ValueError("b0 and c0 must have the same shape.")

    n = b0.shape[0]
    ones = np.ones(n, dtype=float)

    A1 = cho_solve(cf, ones, check_finite=False)
    beta = float(ones @ A1)
    if (not np.isfinite(beta)) or (beta <= 0.0):
        raise np.linalg.LinAlgError(
            "Ordinary collocated cokriging constraint system is singular or ill-conditioned."
        )

    Ab = cho_solve(cf, b0, check_finite=False)
    Ac = cho_solve(cf, c0, check_finite=False)

    alpha = ones @ Ac
    delta = np.einsum("ij,ij->j", c0, Ac)
    eps = np.einsum("ij,i->j", c0, A1)
    tau = ones @ Ab
    ups = np.einsum("ij,ij->j", c0, Ab)

    M11 = 1.0 - alpha
    M12 = -beta
    M21 = float(tail_diag) - delta
    M22 = 1.0 - eps

    rhs1 = 1.0 - tau
    rhs2 = float(rhs_tail) - ups

    det = M11 * M22 - M12 * M21
    if np.any(~np.isfinite(det)) or np.any(np.abs(det) <= 1e-14):
        raise np.linalg.LinAlgError(
            "Ordinary collocated cokriging constraint system is singular or ill-conditioned."
        )

    w_tail = (rhs1 * M22 - M12 * rhs2) / det
    mu = (M11 * rhs2 - rhs1 * M21) / det
    w_top = Ab - Ac * w_tail[None, :] - A1[:, None] * mu[None, :]

    return (
        w_top,
        np.asarray(w_tail, float).ravel(),
        np.asarray(mu, float).ravel(),
    )

def _run_simple_cokriging_core(
    *,
    z: np.ndarray,
    y: np.ndarray,
    Xz: np.ndarray,
    Xy: np.ndarray,
    XT: np.ndarray,
    mz: float,
    my: float,
    mu_z: Optional[float],
    sd_z: float,
    standardize: bool,
    assemble_system_fn,
    assemble_conditioning_fn=None,
    assemble_rhs_fn=None,
    target_batch_size: Optional[int] = None,
    distance_type: str,
    max_neighbors_primary: Optional[int],
    max_neighbors_secondary: Optional[int],
    balltree_leaf_size: int,
    check_positive_definite: bool,
    return_weights: bool,
    show_progress: bool = True,
):
    """
    Run the global or local two-variable simple cokriging solve.

    The supplied assemble_system_fn must return:
        K, k0, sigma2_primary

    for the requested primary samples, secondary samples, and targets.

    Notes
    -----
    In global mode, if assemble_conditioning_fn and assemble_rhs_fn are both
    supplied, the conditioning matrix is factorized once and the targets may be
    processed in batches through repeated right-hand-side solves. Otherwise the
    full K / k0 system is assembled and solved in one step.

    Local mode selects primary and secondary neighborhoods independently for
    each target, then assembles and solves a local block system per target.

    Local neighbor selection is performed on the original cleaned coordinates.
    Any anisotropy transform is applied later during covariance assembly.
    This matches the example-style workflow more closely.
    """
    nz = z.size
    ny = y.size
    m = XT.shape[0]

    use_local_z = (max_neighbors_primary is not None) and (int(max_neighbors_primary) < nz)
    use_local_y = (max_neighbors_secondary is not None) and (int(max_neighbors_secondary) < ny)
    use_local = use_local_z or use_local_y

    if not use_local:
        if assemble_conditioning_fn is None or assemble_rhs_fn is None:
            K, k0, sigma2_primary = assemble_system_fn(Xz, Xy, XT)
            Wt = _solve_sck_system(K, k0, check_positive_definite=check_positive_definite)

            W = Wt.T
            wz = W[:, :nz]
            wy = W[:, nz:]

            est_std = mz + (wz @ (z - mz)) + (wy @ (y - my))
            var_std = np.maximum(
                sigma2_primary - np.einsum("ij,ij->i", W, k0.T),
                0.0,
            )

            if standardize:
                est = mu_z + sd_z * est_std
                var = (sd_z ** 2) * var_std
            else:
                est = est_std
                var = var_std

            if return_weights:
                return est, var, wz, wy
            return est, var

        K, sigma2_primary = assemble_conditioning_fn(Xz, Xy)
        cf = _factor_cokriging_matrix(
            K,
            check_positive_definite=check_positive_definite,
        )
        del K

        if target_batch_size is None:
            batch_size = m
        else:
            batch_size = int(target_batch_size)
            if batch_size <= 0:
                raise ValueError("target_batch_size must be None or a positive integer.")

        est = np.empty(m, dtype=float)
        var = np.empty(m, dtype=float)

        if return_weights:
            wz = np.empty((m, nz), dtype=float)
            wy = np.empty((m, ny), dtype=float)

        batch_iter = range(0, m, batch_size)
        if show_progress and batch_size < m:
            batch_iter = tqdm(batch_iter, desc="Processing estimate batches")

        for j0 in batch_iter:
            j1 = min(j0 + batch_size, m)

            k0 = assemble_rhs_fn(Xz, Xy, XT[j0:j1])
            Wt = _solve_sck_from_factor(cf, k0)
            W = Wt.T

            wz_b = W[:, :nz]
            wy_b = W[:, nz:]

            est_std_b = mz + (wz_b @ (z - mz)) + (wy_b @ (y - my))
            var_std_b = np.maximum(
                sigma2_primary - np.einsum("ij,ij->i", W, k0.T),
                0.0,
            )

            if standardize:
                est[j0:j1] = mu_z + sd_z * est_std_b
                var[j0:j1] = (sd_z ** 2) * var_std_b
            else:
                est[j0:j1] = est_std_b
                var[j0:j1] = var_std_b

            if return_weights:
                wz[j0:j1, :] = wz_b
                wy[j0:j1, :] = wy_b

        if return_weights:
            return est, var, wz, wy
        return est, var

    if use_local_z:
        nn_z = query_nearest_neighbors_balltree(
            Xz, XT,
            distance_type=distance_type,
            max_neighbors=max_neighbors_primary,
            balltree_leaf_size=balltree_leaf_size,
        )
    else:
        nn_z = [np.arange(nz)] * m

    if use_local_y:
        nn_y = query_nearest_neighbors_balltree(
            Xy, XT,
            distance_type=distance_type,
            max_neighbors=max_neighbors_secondary,
            balltree_leaf_size=balltree_leaf_size,
        )
    else:
        nn_y = [np.arange(ny)] * m

    est = np.empty(m, dtype=float)
    var = np.empty(m, dtype=float)

    if return_weights:
        if use_local_z:
            wz_full = [None] * m
        else:
            wz_full = np.zeros((m, nz), dtype=float)

        if use_local_y:
            wy_full = [None] * m
        else:
            wy_full = np.zeros((m, ny), dtype=float)
    else:
        wz_full = None
        wy_full = None

    if show_progress:
        j_iter = tqdm(range(m), desc="Processing estimate")
    else:
        j_iter = range(m)

    for j in j_iter:
        idx_z = np.asarray(nn_z[j], dtype=int)
        idx_y = np.asarray(nn_y[j], dtype=int)

        Xz_loc = Xz[idx_z]
        Xy_loc = Xy[idx_y]
        z_loc = z[idx_z]
        y_loc = y[idx_y]
        XT_j = XT[j:j+1]

        K, k0, sigma2_primary = assemble_system_fn(Xz_loc, Xy_loc, XT_j)
        Wt = _solve_sck_system(K, k0, check_positive_definite=check_positive_definite)

        w = Wt[:, 0]
        nloc_z = idx_z.size
        wz = w[:nloc_z]
        wy = w[nloc_z:]

        est_std_j = mz + (wz @ (z_loc - mz)) + (wy @ (y_loc - my))
        var_std_j = max(sigma2_primary - float(w @ k0[:, 0]), 0.0)

        if standardize:
            est[j] = mu_z + sd_z * est_std_j
            var[j] = (sd_z ** 2) * var_std_j
        else:
            est[j] = est_std_j
            var[j] = var_std_j

        if return_weights:
            if use_local_z:
                wz_full[j] = (idx_z.copy(), wz.copy())
            else:
                wz_full[j, :] = wz

            if use_local_y:
                wy_full[j] = (idx_y.copy(), wy.copy())
            else:
                wy_full[j, :] = wy

    if return_weights:
        return est, var, wz_full, wy_full
    return est, var

def simple_cokriging(
    primary_values,
    primary_coords,
    secondary_values,
    secondary_coords,
    targets,
    *,
    covariance_mode: str = "direct",
    primary_model: Optional[dict] = None,
    secondary_model: Optional[dict] = None,
    cross_model: Optional[dict] = None,
    structures: Optional[Sequence[dict]] = None,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    standardize: bool = False,
    mean_primary: Union[str, float] = "sample",
    mean_secondary: Union[str, float] = "sample",
    jitter: float = 1e-10,
    chunk_cols: Optional[int] = None,
    max_neighbors_primary: Optional[int] = None,
    max_neighbors_secondary: Optional[int] = None,
    target_batch_size: Optional[int] = None,
    balltree_leaf_size: int = 40,
    check_positive_definite: bool = True,
    return_weights: bool = False,
    show_progress: bool = True,
):
    """
    Two-variable simple cokriging for a primary variable Z and a secondary variable Y.

    Parameters
    ----------
    primary_values, secondary_values : array_like
        Observed primary and secondary sample values.
    primary_coords, secondary_coords : array_like
        Coordinates associated with the primary and secondary samples.
    targets : array_like
        Prediction locations for the primary variable.
    covariance_mode : {'direct', 'lmc'}, default 'direct'
        Covariance representation used to assemble the cokriging system.

        - 'direct' uses primary_model, secondary_model, and cross_model directly.
        - 'lmc' uses a Linear Model of Coregionalization supplied through
          structures.
    primary_model, secondary_model, cross_model : dict or None
        Direct covariance specifications used when covariance_mode='direct'.
    structures : sequence of dict or None
        LMC structures used when covariance_mode='lmc'.
    distance_type : {'geographic', 'euclidean', 'cartesian', 'angular'}
        Distance convention passed to pairwise_distances(...).
    projection : str, default 'WGS84'
        Projection / ellipsoid argument passed through where relevant.
    rotation_matrix : dict or None, default None
        Optional anisotropy specification.

        - In covariance_mode='direct', this must be a single planar rotation
          dict with keys {'azimuth', 'a_max', 'a_min'}.
        - In covariance_mode='lmc', this may be a per-structure mapping with one
          entry for each LMC structure.

        Rotation is supported only for planar 2D coordinates with
        distance_type='euclidean' or 'cartesian'.
    standardize : bool, default False
        If True, the primary and secondary sample values are standardized
        internally using their own sample means and sample standard deviations,
        and the final estimates / variances are back-transformed to the primary
        scale.

        Important: this option standardizes the sample values only. It does not
        automatically rescale primary_model, secondary_model, cross_model, or
        LMC structures. Therefore standardize=True is theoretically consistent
        only when the supplied covariance representation is already expressed on
        the same standardized working scale (for example correlation/unit-
        variance models, normalized variogram forms, or LMC structures on the
        correlation scale).
    mean_primary, mean_secondary : {'sample', 'zero'} or float, default 'sample'
        Means used only when standardize=False.
    jitter : float, default 1e-10
        Nonnegative diagonal stabilization used during covariance assembly.
    chunk_cols : int or None, default None
        Optional chunk size passed to pairwise_distances(...).
    max_neighbors_primary, max_neighbors_secondary : int or None, default None
        Optional local neighborhoods for the primary and secondary samples. If
        omitted, a global solve is used.
    target_batch_size : int or None, default None
        Optional number of targets processed per batch in the global solve.

        This is used only when no local neighborhood is active. In that case
        the conditioning matrix is factorized once and the target-side right-
        hand side is processed in batches. If None, all targets are processed
        in one batch.
    balltree_leaf_size : int, default 40
        Leaf size passed to the BallTree-based neighbor search in local mode.
    check_positive_definite : bool, default True
        If True, raise an error when the assembled block covariance matrix is not
        numerically positive definite.
    return_weights : bool, default False
        If True, also return the primary and secondary cokriging weights.
    show_progress : bool, default True
        If True, show a progress bar in local mode and in global mode when the
        targets are processed in multiple batches.

    Returns
    -------
    est : (m,) ndarray
        Simple cokriging estimates of the primary variable at the targets.
    var : (m,) ndarray
        Simple cokriging variances of the primary variable at the targets.
    wz, wy : optional
        Weight outputs for the primary and secondary conditioning data.

        For each variable separately:
        - if that variable uses a global neighborhood, the returned weights are
          a dense matrix of shape (m, n_variable);
        - if that variable uses a local neighborhood, the returned weights are
          a list of length m where
              w[j] = (idx_j, w_j)
          stores only the weights associated with the selected neighbors for
          target j.

        Therefore mixed local/global runs may return one dense matrix and one
        list-of-tuples.

    Notes
    -----
    In local mode, neighborhood selection is performed on the original cleaned
    coordinates. Any optional anisotropy transform is applied only during
    covariance evaluation, not during BallTree neighbor selection.

    In global mode, target_batch_size may be used to reuse a single
    factorization of the conditioning matrix while processing the target-side
    right-hand side in batches.

    The assembled covariance system must already be on the same working scale as
    the values passed to the solver. In particular, if standardize=True, the
    supplied direct/LMC model specifications are assumed already to correspond to
    that standardized working scale.
    """
    prep = _prepare_simple_cokriging_inputs(
        primary_values,
        primary_coords,
        secondary_values,
        secondary_coords,
        targets,
        distance_type=distance_type,
        standardize=standardize,
        mean_primary=mean_primary,
        mean_secondary=mean_secondary,
    )

    dt = prep["distance_type"]
    covariance_mode = str(covariance_mode).lower().strip()

    if covariance_mode == "direct":
        _validate_simple_cokriging_runtime_options(
            covariance_mode=covariance_mode,
            distance_type=dt,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            max_neighbors_primary=max_neighbors_primary,
            max_neighbors_secondary=max_neighbors_secondary,
            target_batch_size=target_batch_size,
            balltree_leaf_size=balltree_leaf_size,
            jitter=jitter,
        )

        if primary_model is None or secondary_model is None or cross_model is None:
            raise ValueError(
                "For covariance_mode='direct', primary_model, secondary_model, and cross_model are required."
            )

        def assemble_conditioning_fn(Xz_, Xy_):
            return _assemble_sck_conditioning_matrix(
                Xz_, Xy_,
                primary_model=primary_model,
                secondary_model=secondary_model,
                cross_model=cross_model,
                distance_type=dt,
                projection=projection,
                rotation_matrix=rotation_matrix,
                jitter=jitter,
                chunk_cols=chunk_cols,
            )

        def assemble_rhs_fn(Xz_, Xy_, XT_):
            return _assemble_sck_rhs(
                Xz_, Xy_, XT_,
                primary_model=primary_model,
                cross_model=cross_model,
                distance_type=dt,
                projection=projection,
                rotation_matrix=rotation_matrix,
                chunk_cols=chunk_cols,
            )

        def assemble_system_fn(Xz_, Xy_, XT_):
            K_, sigma2_primary_ = assemble_conditioning_fn(Xz_, Xy_)
            k0_ = assemble_rhs_fn(Xz_, Xy_, XT_)
            return K_, k0_, sigma2_primary_

    elif covariance_mode == "lmc":
        if structures is None:
            raise ValueError(
                "For covariance_mode='lmc', structures must be provided."
            )

        _validate_simple_cokriging_runtime_options(
            covariance_mode=covariance_mode,
            distance_type=dt,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            max_neighbors_primary=max_neighbors_primary,
            max_neighbors_secondary=max_neighbors_secondary,
            target_batch_size=target_batch_size,
            balltree_leaf_size=balltree_leaf_size,
            jitter=jitter,
            n_lmc_structures=len(structures),
        )

        def assemble_conditioning_fn(Xz_, Xy_):
            return _assemble_lmc_sck_conditioning_matrix(
                Xz_, Xy_,
                structures=structures,
                distance_type=dt,
                projection=projection,
                rotation_matrix=rotation_matrix,
                jitter=jitter,
                chunk_cols=chunk_cols,
            )

        def assemble_rhs_fn(Xz_, Xy_, XT_):
            return _assemble_lmc_sck_rhs(
                Xz_, Xy_, XT_,
                structures=structures,
                distance_type=dt,
                projection=projection,
                rotation_matrix=rotation_matrix,
                chunk_cols=chunk_cols,
            )

        def assemble_system_fn(Xz_, Xy_, XT_):
            K_, sigma2_primary_ = assemble_conditioning_fn(Xz_, Xy_)
            k0_ = assemble_rhs_fn(Xz_, Xy_, XT_)
            return K_, k0_, sigma2_primary_

    else:
        raise ValueError("covariance_mode must be 'direct' or 'lmc'.")

    return _run_simple_cokriging_core(
        z=prep["z"],
        y=prep["y"],
        Xz=prep["Xz"],
        Xy=prep["Xy"],
        XT=prep["XT"],
        mz=prep["mz"],
        my=prep["my"],
        mu_z=prep["mu_z"],
        sd_z=prep["sd_z"],
        standardize=prep["standardize"],
        assemble_system_fn=assemble_system_fn,
        assemble_conditioning_fn=assemble_conditioning_fn,
        assemble_rhs_fn=assemble_rhs_fn,
        target_batch_size=target_batch_size,
        distance_type=dt,
        max_neighbors_primary=max_neighbors_primary,
        max_neighbors_secondary=max_neighbors_secondary,
        balltree_leaf_size=balltree_leaf_size,
        check_positive_definite=check_positive_definite,
        return_weights=return_weights,
        show_progress=show_progress,
    )

def simple_collocated_cokriging(
    primary_values,
    primary_coords,
    collocated_secondary_values,
    targets,
    *,
    markov_model: str = "mm1",
    rho0: Optional[float] = None,
    primary_values_for_rho0=None,
    secondary_values_for_rho0=None,
    rho0_method: str = "pearson",
    primary_model: Optional[dict] = None,
    secondary_model: Optional[dict] = None,
    residual_model: Optional[dict] = None,
    secondary_values_for_standardization=None,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    standardize: bool = True,
    mean_primary: Union[str, float] = "sample",
    mean_secondary: Union[str, float] = "sample",
    jitter: float = 1e-10,
    chunk_cols: Optional[int] = None,
    max_neighbors_primary: Optional[int] = None,
    target_batch_size: Optional[int] = None,
    balltree_leaf_size: int = 40,
    check_positive_definite: bool = True,
    return_weights: bool = False,
    show_progress: bool = True,
):
    """
    Simple collocated cokriging (SCCK) with Markov model I or II.

    SCCK uses neighboring primary samples Z(x_i) and one collocated secondary
    datum Y(u) at each target u.

    Parameters
    ----------
    primary_values : array_like
        Observed primary sample values.
    primary_coords : array_like
        Coordinates of the primary samples.
    collocated_secondary_values : array_like
        Secondary values collocated with the target locations.
    targets : array_like
        Target coordinates.
    markov_model : {'mm1', 'mm2'}, default 'mm1'
        Markov closure used for SCCK.

        - 'mm1' uses rho_z(h) as the primary direct shape and
          rho_zy(h) = rho0 * rho_z(h).
        - 'mm2' uses rho_y(h) and rho_r(h) to reconstruct
          rho_z(h) = rho0**2 * rho_y(h) + (1 - rho0**2) * rho_r(h),
          and rho_zy(h) = rho0 * rho_y(h).
    rho0 : float or None, default None
        Collocated primary-secondary correlation. If None, it is estimated from
        primary_values_for_rho0 and secondary_values_for_rho0.
    primary_values_for_rho0, secondary_values_for_rho0 : array_like or None
        Paired colocated values used to estimate rho0 when needed.
    rho0_method : {'pearson', 'uncentered'}, default 'pearson'
        Estimator used when rho0 is inferred.
    primary_model : dict or None
        Direct primary covariance/correlation model rho_z(h), required for MM1.
    secondary_model : dict or None
        Direct secondary covariance/correlation model rho_y(h), required for MM2.
    residual_model : dict or None
        Direct residual covariance/correlation model rho_r(h), required for MM2.
    secondary_values_for_standardization : array_like or None, default None
        Reference secondary values used to compute the secondary mean/scale that
        places collocated target-side secondary data on the SCCK working scale.

        When standardize=False and mean_secondary='sample', these values define
        the reference secondary mean. If omitted, secondary_values_for_rho0 is
        preferred when available; otherwise collocated_secondary_values is used.
    distance_type : {'geographic', 'euclidean', 'cartesian', 'angular'}
        Distance convention passed to pairwise_distances(...).
    projection : str, default 'WGS84'
        Projection / ellipsoid argument passed through where relevant.
    rotation_matrix : dict or None, default None
        Optional planar anisotropy specification with keys
        {'azimuth', 'a_max', 'a_min'}.
    standardize : bool, default True
        If True, the primary samples and collocated secondary values are placed
        on the standardized SCCK working scale internally, and the final SCCK
        estimate/variance are back-transformed to the primary scale.

        If False, the inputs are assumed already to be on the SCCK working scale
        (typically standardized or normal-score transformed with unit variance).
        This is not a general raw-unit SCCK mode.
    mean_primary, mean_secondary : {'sample', 'zero'} or float, default 'sample'
        Means used only when standardize=False.
    jitter : float, default 1e-10
        Nonnegative diagonal stabilization.
    chunk_cols : int or None, default None
        Optional chunk size passed to pairwise_distances(...).
    max_neighbors_primary : int or None, default None
        Optional primary neighborhood size. If omitted, a global solve is used.
    target_batch_size : int or None, default None
        Optional number of targets processed per batch in the global solve.

        This is used only when no local primary neighborhood is active. In that
        case the target-invariant SCCK neighbor block is factorized once and the
        target-dependent collocated terms are processed in batches. If None, all
        targets are processed in one batch.
    balltree_leaf_size : int, default 40
        Leaf size passed to the BallTree-based neighbor search in local mode.
    check_positive_definite : bool, default True
        If True, raise an error when the assembled SCCK matrix is not numerically
        positive definite.
    return_weights : bool, default False
        If True, also return the primary-data weights and collocated-secondary
        weights.
    show_progress : bool, default True
        If True, show a progress bar in local mode and in global mode when the
        targets are processed in multiple batches.

    Returns
    -------
    est : (m,) ndarray
        SCCK estimates of the primary variable at the targets.
    var : (m,) ndarray
        SCCK variances on the primary scale when standardize=True, otherwise on
        the supplied SCCK working scale.
    wz : optional
        If return_weights=True and a global solve is used, this is a dense
        weight matrix of shape (m, nz).

        If return_weights=True and a local solve is used, this is a list of
        length m where:
            wz[j] = (idx_z_j, wz_j)
        containing only the nonzero local primary weights for target j.
    wy0 : (m,) ndarray, optional
        Collocated-secondary weights, returned only when return_weights=True.

    Notes
    -----
    This implementation is direct-only. LMC remains available through
    simple_cokriging(...) and the associated LMC helpers, not through SCCK.

    In global mode, target_batch_size may be used to reuse a single
    factorization of the target-invariant SCCK neighbor block while processing
    the target-dependent collocated terms in batches. In local mode,
    target_batch_size is ignored.
    """
    markov_model = str(markov_model).lower().strip()

    if markov_model not in ("mm1", "mm2"):
        raise ValueError("markov_model must be 'mm1' or 'mm2'.")

    if markov_model == "mm1":
        if primary_model is None:
            raise ValueError("MM1 requires primary_model.")

    if markov_model == "mm2":
        if secondary_model is None:
            raise ValueError("MM2 requires secondary_model for rho_y(h).")
        if residual_model is None:
            raise ValueError("MM2 requires residual_model for rho_r(h).")
        # primary_model is intentionally ignored in MM2.

    dt = str(distance_type).lower().strip()
    if dt == "geographical":
        dt = "geographic"

    z_raw, Xz = _clean_values_and_coords(
        primary_values,
        primary_coords,
        distance_type=dt,
        name="primary",
    )
    XT = _coerce_coordinates(targets, dt, "targets")

    if standardize:
        z, mu_z, sd_z = _standardize_values(z_raw, name="primary")
    else:
        z = z_raw.copy()
        mu_z = None
        sd_z = 1.0

    prep = {
        "distance_type": dt,
        "z": z,
        "Xz": Xz,
        "XT": XT,
        "mu_z": mu_z,
        "sd_z": sd_z,
        "standardize": bool(standardize),
    }

    _validate_simple_cokriging_runtime_options(
        covariance_mode="direct",
        distance_type=prep["distance_type"],
        rotation_matrix=rotation_matrix,
        chunk_cols=chunk_cols,
        max_neighbors_primary=max_neighbors_primary,
        max_neighbors_secondary=None,
        target_batch_size=target_batch_size,
        balltree_leaf_size=balltree_leaf_size,
        jitter=jitter,
        n_lmc_structures=None,
    )

    rho0 = _resolve_rho0(
        rho0,
        primary_values_for_rho0=primary_values_for_rho0,
        secondary_values_for_rho0=secondary_values_for_rho0,
        method=rho0_method,
    )

    if standardize:
        if secondary_values_for_standardization is None:
            if secondary_values_for_rho0 is not None:
                secondary_values_for_standardization = secondary_values_for_rho0
            else:
                raise ValueError(
                    "For simple_collocated_cokriging(..., standardize=True), "
                    "provide secondary_values_for_standardization or "
                    "secondary_values_for_rho0."
                )
    else:
        if secondary_values_for_standardization is None:
            if secondary_values_for_rho0 is not None:
                secondary_values_for_standardization = secondary_values_for_rho0
            else:
                secondary_values_for_standardization = collocated_secondary_values

    y0, _, _ = _prepare_collocated_secondary_at_targets(
        collocated_secondary_values,
        prep["XT"],
        secondary_training_values=secondary_values_for_standardization,
        standardize=standardize,
    )

    if not standardize:
        mz = _resolve_simple_mean(mean_primary, z, var_name="primary")
        my = _resolve_simple_mean(
            mean_secondary,
            secondary_values_for_standardization,
            var_name="secondary",
        )

    z = prep["z"]
    Xz = prep["Xz"]
    XT = prep["XT"]
    mu_z = prep["mu_z"]
    sd_z = prep["sd_z"]

    nz = z.size
    m = XT.shape[0]

    use_local = (max_neighbors_primary is not None) and (int(max_neighbors_primary) < nz)

    est = np.empty(m, dtype=float)
    var = np.empty(m, dtype=float)

    if return_weights:
        if use_local:
            wz_full = [None] * m
        else:
            wz_full = np.empty((m, nz), dtype=float)
        wy0 = np.empty(m, dtype=float)
    else:
        wz_full = None
        wy0 = None

    if not use_local:
        A = _assemble_scck_base_conditioning_matrix(
            Xz,
            markov_model=markov_model,
            rho0=rho0,
            primary_model=primary_model,
            secondary_model=secondary_model,
            residual_model=residual_model,
            distance_type=prep["distance_type"],
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            jitter=jitter,
        )
        cf = _factor_cokriging_matrix(
            A,
            check_positive_definite=check_positive_definite,
        )
        del A

        if target_batch_size is None:
            batch_size = m
        else:
            batch_size = int(target_batch_size)
            if batch_size <= 0:
                raise ValueError("target_batch_size must be None or a positive integer.")

        batch_iter = range(0, m, batch_size)
        if show_progress and batch_size < m:
            batch_iter = tqdm(batch_iter, desc="Processing SCCK batches")

        tail_diag = 1.0 + jitter

        for j0 in batch_iter:
            j1 = min(j0 + batch_size, m)

            rz0_b, cy0_b = _assemble_scck_target_blocks(
                Xz,
                XT[j0:j1],
                markov_model=markov_model,
                rho0=rho0,
                primary_model=primary_model,
                secondary_model=secondary_model,
                residual_model=residual_model,
                distance_type=prep["distance_type"],
                projection=projection,
                rotation_matrix=rotation_matrix,
                chunk_cols=chunk_cols,
            )

            wz_t, wy_b = _solve_batched_collocated_simple_from_factor(
                cf,
                rz0_b,
                cy0_b,
                rhs_tail=rho0,
                tail_diag=tail_diag,
            )
            wz_b = wz_t.T

            if standardize:
                est_work_b = wz_b @ z + wy_b * y0[j0:j1]
                var_work_b = np.maximum(
                    1.0 - np.einsum("ij,ij->j", wz_t, rz0_b) - wy_b * float(rho0),
                    0.0,
                )
                est[j0:j1] = mu_z + sd_z * est_work_b
                var[j0:j1] = (sd_z ** 2) * var_work_b
            else:
                est[j0:j1] = mz + wz_b @ (z - mz) + wy_b * (y0[j0:j1] - my)
                var[j0:j1] = np.maximum(
                    1.0 - np.einsum("ij,ij->j", wz_t, rz0_b) - wy_b * float(rho0),
                    0.0,
                )

            if return_weights:
                wz_full[j0:j1, :] = wz_b
                wy0[j0:j1] = wy_b

        if return_weights:
            return est, var, wz_full, wy0
        return est, var

    nn_z = query_nearest_neighbors_balltree(
        Xz, XT,
        distance_type=prep["distance_type"],
        max_neighbors=max_neighbors_primary,
        balltree_leaf_size=balltree_leaf_size,
    )
    j_iter = tqdm(range(m), desc="Processing SCCK") if show_progress else range(m)

    for j in j_iter:
        idx_z = np.asarray(nn_z[j], dtype=int)

        K, r = _assemble_scck_system_one_target(
            Xz[idx_z],
            XT[j],
            markov_model=markov_model,
            rho0=rho0,
            primary_model=primary_model,
            secondary_model=secondary_model,
            residual_model=residual_model,
            distance_type=prep["distance_type"],
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            jitter=jitter,
        )

        w = _solve_sck_system(
            K,
            r[:, None],
            check_positive_definite=check_positive_definite,
        )[:, 0]

        wz = w[:-1]
        wy = w[-1]

        if standardize:
            est_work = wz @ z[idx_z] + wy * y0[j]
            var_work = max(1.0 - float(w @ r), 0.0)

            est[j] = mu_z + sd_z * est_work
            var[j] = (sd_z ** 2) * var_work
        else:
            est[j] = mz + wz @ (z[idx_z] - mz) + wy * (y0[j] - my)
            var[j] = max(1.0 - float(w @ r), 0.0)

        if return_weights:
            wz_full[j] = (idx_z.copy(), wz.copy())
            wy0[j] = wy

    if return_weights:
        return est, var, wz_full, wy0
    return est, var

def intrinsic_collocated_cokriging(
    primary_values,
    primary_coords,
    secondary_values,
    secondary_coords,
    collocated_secondary_values,
    targets,
    *,
    markov_model: str = "mm1",
    rho0: Optional[float] = None,
    primary_values_for_rho0=None,
    secondary_values_for_rho0=None,
    rho0_method: str = "pearson",
    primary_model: Optional[dict] = None,
    secondary_model: Optional[dict] = None,
    residual_model: Optional[dict] = None,
    secondary_values_for_standardization=None,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    standardize: bool = True,
    mean_primary: Union[str, float] = "sample",
    mean_secondary: Union[str, float] = "sample",
    jitter: float = 1e-10,
    chunk_cols: Optional[int] = None,
    max_neighbors_primary: Optional[int] = None,
    max_neighbors_secondary: Optional[int] = None,
    target_batch_size: Optional[int] = None,
    balltree_leaf_size: int = 40,
    check_positive_definite: bool = True,
    return_weights: bool = False,
    show_progress: bool = True,
):
    """
    Intrinsic collocated cokriging (ICCK) with Markov model I or II.

    ICCK uses three sources of information at each target:

        1. neighboring primary samples Z(x_i),
        2. neighboring secondary samples Y(y_j),
        3. one collocated secondary datum Y(u) at the target.

    As in SCCK, the primary-secondary cross terms are approximated through a
    Markov-type closure controlled by markov_model and rho0.

    Parameters
    ----------
    primary_values, secondary_values : array_like
        Observed primary and secondary sample values.
    primary_coords, secondary_coords : array_like
        Coordinates associated with the primary and secondary samples.
    collocated_secondary_values : array_like
        Secondary values collocated with the target locations.
    targets : array_like
        Target coordinates.
    markov_model : {'mm1', 'mm2'}, default 'mm1'
        Markov closure used for ICCK.

        - 'mm1' uses rho_z(h) as the primary direct shape and
          rho_zy(h) = rho0 * rho_z(h), with rho_y(h) = rho_z(h).
        - 'mm2' uses rho_y(h) and rho_r(h) to reconstruct
          rho_z(h) = rho0**2 * rho_y(h) + (1 - rho0**2) * rho_r(h),
          and rho_zy(h) = rho0 * rho_y(h).
    rho0 : float or None, default None
        Collocated primary-secondary correlation. If None, it is estimated from
        primary_values_for_rho0 and secondary_values_for_rho0.
    primary_values_for_rho0, secondary_values_for_rho0 : array_like or None
        Paired colocated values used to estimate rho0 when needed.
    rho0_method : {'pearson', 'uncentered'}, default 'pearson'
        Estimator used when rho0 is inferred.
    primary_model : dict or None
        Direct primary covariance/correlation model rho_z(h), required for MM1.
        Under MM1, the same model is also used for rho_y(h), consistent with the
        intrinsic Markov I closure.
    secondary_model : dict or None
        Direct secondary covariance/correlation model rho_y(h), required for MM2.
    residual_model : dict or None
        Direct residual covariance/correlation model rho_r(h), required for MM2.
    secondary_values_for_standardization : array_like or None, default None
        Reference secondary values defining the secondary location/scale used by
        ICCK.

        - When standardize=True, these values define the standardization
          transform applied to both neighboring and collocated secondary data.
        - When standardize=False and mean_secondary='sample', these values
          define the reference secondary mean. If omitted, the neighboring
          secondary sample values are used.
    distance_type : {'geographic', 'euclidean', 'cartesian', 'angular'}
        Distance convention passed to pairwise_distances(...).
    projection : str, default 'WGS84'
        Projection / ellipsoid argument passed through where relevant.
    rotation_matrix : dict or None, default None
        Optional planar anisotropy specification with keys
        {'azimuth', 'a_max', 'a_min'}.
    standardize : bool, default True
        If True, ICCK is carried out on the standardized working scale and the
        final estimate/variance are back-transformed to the primary scale.

        If False, the supplied primary, neighboring secondary, and collocated
        secondary values are assumed already to be on the ICCK working scale
        (typically standardized or normal-score transformed with unit variance).
        This is not a general raw-unit ICCK mode.
    mean_primary, mean_secondary : {'sample', 'zero'} or float, default 'sample'
        Means used only when standardize=False.
    jitter : float, default 1e-10
        Nonnegative diagonal stabilization.
    chunk_cols : int or None, default None
        Optional chunk size passed to pairwise_distances(...).
    max_neighbors_primary, max_neighbors_secondary : int or None, default None
        Optional local neighborhood sizes for the primary and secondary samples.
    target_batch_size : int or None, default None
        Optional number of targets processed per batch in the global solve.

        This is used only when no local neighborhood is active. In that case
        the target-invariant ICCK neighbor block is factorized once and the
        target-dependent collocated terms are processed in batches. If None, all
        targets are processed in one batch.
    balltree_leaf_size : int, default 40
        Leaf size passed to the BallTree-based neighbor search in local mode.
    check_positive_definite : bool, default True
        If True, raise an error when the assembled ICCK matrix is not
        numerically positive definite.
    return_weights : bool, default False
        If True, also return the primary weights, secondary weights, and the
        collocated-secondary weight at each target.
    show_progress : bool, default True
        If True, show a progress bar in local mode and in global mode when the
        targets are processed in multiple batches.

    Returns
    -------
    est : (m,) ndarray
        ICCK estimates of the primary variable.
    var : (m,) ndarray
        ICCK variances on the primary scale when standardize=True, otherwise on
        the supplied ICCK working scale.
    wz, wy : optional
        Weight outputs for the neighboring primary and neighboring secondary
        conditioning data.

        For each variable separately:
        - if that variable uses a global neighborhood, the returned weights are
          a dense matrix of shape (m, n_variable);
        - if that variable uses a local neighborhood, the returned weights are
          a list of length m where
              w[j] = (idx_j, w_j)
          stores only the weights associated with the selected neighbors for
          target j.

        Therefore mixed local/global runs may return one dense matrix and one
        list-of-tuples.
    wy0 : (m,) ndarray, optional
        Collocated-secondary weights, returned only when return_weights=True.

    Notes
    -----
    This implementation is direct-only.

    ICCK reduces to a local or global block solve depending on the chosen
    neighborhood sizes. As with SCCK, standardize=True is the recommended
    working mode.

    In global mode, target_batch_size may be used to reuse a single
    factorization of the target-invariant ICCK neighbor block while processing
    the target-dependent collocated terms in batches. In local mode,
    target_batch_size is ignored.
    """
    markov_model = str(markov_model).lower().strip()

    if markov_model not in ("mm1", "mm2"):
        raise ValueError("markov_model must be 'mm1' or 'mm2'.")

    if markov_model == "mm1":
        if primary_model is None:
            raise ValueError("MM1 requires primary_model.")

    if markov_model == "mm2":
        if secondary_model is None:
            raise ValueError("MM2 requires secondary_model for rho_y(h).")
        if residual_model is None:
            raise ValueError("MM2 requires residual_model for rho_r(h).")
        # primary_model is intentionally ignored in MM2.

    dt = str(distance_type).lower().strip()
    if dt == "geographical":
        dt = "geographic"

    z_raw, Xz = _clean_values_and_coords(
        primary_values, primary_coords, distance_type=dt, name="primary"
    )
    y_raw, Xy = _clean_values_and_coords(
        secondary_values, secondary_coords, distance_type=dt, name="secondary"
    )
    XT = _coerce_coordinates(targets, dt, "targets")

    if standardize:
        z, mu_z, sd_z = _standardize_values(z_raw, name="primary")

        if secondary_values_for_standardization is None:
            if secondary_values_for_rho0 is not None:
                secondary_values_for_standardization = secondary_values_for_rho0
            else:
                secondary_values_for_standardization = y_raw

        mu_y_ref, sd_y_ref = _compute_standardization_stats(
            secondary_values_for_standardization,
            name="secondary",
        )
        y = (y_raw - mu_y_ref) / sd_y_ref
    else:
        z = z_raw.copy()
        y = y_raw.copy()
        mu_z = None
        sd_z = 1.0
        if secondary_values_for_standardization is None:
            secondary_values_for_standardization = y_raw

    prep = {
        "distance_type": dt,
        "z": z,
        "y": y,
        "Xz": Xz,
        "Xy": Xy,
        "XT": XT,
        "mu_z": mu_z,
        "sd_z": sd_z,
        "standardize": bool(standardize),
    }

    _validate_simple_cokriging_runtime_options(
        covariance_mode="direct",
        distance_type=prep["distance_type"],
        rotation_matrix=rotation_matrix,
        chunk_cols=chunk_cols,
        max_neighbors_primary=max_neighbors_primary,
        max_neighbors_secondary=max_neighbors_secondary,
        target_batch_size=target_batch_size,
        balltree_leaf_size=balltree_leaf_size,
        jitter=jitter,
        n_lmc_structures=None,
    )

    rho0 = _resolve_rho0(
        rho0,
        primary_values_for_rho0=primary_values_for_rho0,
        secondary_values_for_rho0=secondary_values_for_rho0,
        method=rho0_method,
    )

    y0, _, _ = _prepare_collocated_secondary_at_targets(
        collocated_secondary_values,
        prep["XT"],
        secondary_training_values=secondary_values_for_standardization,
        standardize=standardize,
    )

    if not standardize:
        mz = _resolve_simple_mean(mean_primary, z, var_name="primary")
        my = _resolve_simple_mean(
            mean_secondary,
            secondary_values_for_standardization,
            var_name="secondary",
        )

    z = prep["z"]
    y = prep["y"]
    Xz = prep["Xz"]
    Xy = prep["Xy"]
    XT = prep["XT"]
    mu_z = prep["mu_z"]
    sd_z = prep["sd_z"]

    nz = z.size
    ny = y.size
    m = XT.shape[0]

    use_local_z = (max_neighbors_primary is not None) and (int(max_neighbors_primary) < nz)
    use_local_y = (max_neighbors_secondary is not None) and (int(max_neighbors_secondary) < ny)
    use_local = use_local_z or use_local_y

    est = np.empty(m, dtype=float)
    var = np.empty(m, dtype=float)

    if return_weights:
        if use_local_z:
            wz_full = [None] * m
        else:
            wz_full = np.empty((m, nz), dtype=float)

        if use_local_y:
            wy_full = [None] * m
        else:
            wy_full = np.empty((m, ny), dtype=float)

        wy0 = np.empty(m, dtype=float)
    else:
        wz_full = None
        wy_full = None
        wy0 = None

    if not use_local:
        A = _assemble_icck_base_conditioning_matrix(
            Xz,
            Xy,
            markov_model=markov_model,
            rho0=rho0,
            primary_model=primary_model,
            secondary_model=secondary_model,
            residual_model=residual_model,
            distance_type=prep["distance_type"],
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            jitter=jitter,
        )
        cf = _factor_cokriging_matrix(
            A,
            check_positive_definite=check_positive_definite,
        )
        del A

        if target_batch_size is None:
            batch_size = m
        else:
            batch_size = int(target_batch_size)
            if batch_size <= 0:
                raise ValueError("target_batch_size must be None or a positive integer.")

        batch_iter = range(0, m, batch_size)
        if show_progress and batch_size < m:
            batch_iter = tqdm(batch_iter, desc="Processing ICCK batches")

        tail_diag = 1.0 + jitter

        for j0 in batch_iter:
            j1 = min(j0 + batch_size, m)

            b0, c0 = _assemble_icck_target_blocks(
                Xz,
                Xy,
                XT[j0:j1],
                markov_model=markov_model,
                rho0=rho0,
                primary_model=primary_model,
                secondary_model=secondary_model,
                residual_model=residual_model,
                distance_type=prep["distance_type"],
                projection=projection,
                rotation_matrix=rotation_matrix,
                chunk_cols=chunk_cols,
            )

            wtop_t, wyc_b = _solve_batched_collocated_simple_from_factor(
                cf,
                b0,
                c0,
                rhs_tail=rho0,
                tail_diag=tail_diag,
            )

            W_b = wtop_t.T
            wz_b = W_b[:, :nz]
            wy_b = W_b[:, nz:]

            if standardize:
                est_work_b = wz_b @ z + wy_b @ y + wyc_b * y0[j0:j1]
                var_work_b = np.maximum(
                    1.0 - np.einsum("ij,ij->j", wtop_t, b0) - wyc_b * float(rho0),
                    0.0,
                )
                est[j0:j1] = mu_z + sd_z * est_work_b
                var[j0:j1] = (sd_z ** 2) * var_work_b
            else:
                est[j0:j1] = (
                    mz
                    + wz_b @ (z - mz)
                    + wy_b @ (y - my)
                    + wyc_b * (y0[j0:j1] - my)
                )
                var[j0:j1] = np.maximum(
                    1.0 - np.einsum("ij,ij->j", wtop_t, b0) - wyc_b * float(rho0),
                    0.0,
                )

            if return_weights:
                wz_full[j0:j1, :] = wz_b
                wy_full[j0:j1, :] = wy_b
                wy0[j0:j1] = wyc_b

        if return_weights:
            return est, var, wz_full, wy_full, wy0
        return est, var

    if use_local_z:
        nn_z = query_nearest_neighbors_balltree(
            Xz, XT,
            distance_type=prep["distance_type"],
            max_neighbors=max_neighbors_primary,
            balltree_leaf_size=balltree_leaf_size,
        )
    else:
        nn_z = [np.arange(nz)] * m

    if use_local_y:
        nn_y = query_nearest_neighbors_balltree(
            Xy, XT,
            distance_type=prep["distance_type"],
            max_neighbors=max_neighbors_secondary,
            balltree_leaf_size=balltree_leaf_size,
        )
    else:
        nn_y = [np.arange(ny)] * m

    j_iter = tqdm(range(m), desc="Processing ICCK") if show_progress else range(m)

    for j in j_iter:
        idx_z = np.asarray(nn_z[j], dtype=int)
        idx_y = np.asarray(nn_y[j], dtype=int)

        K, r = _assemble_icck_system_one_target(
            Xz[idx_z],
            Xy[idx_y],
            XT[j],
            markov_model=markov_model,
            rho0=rho0,
            primary_model=primary_model,
            secondary_model=secondary_model,
            residual_model=residual_model,
            distance_type=prep["distance_type"],
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            jitter=jitter,
        )

        w = _solve_sck_system(
            K,
            r[:, None],
            check_positive_definite=check_positive_definite,
        )[:, 0]

        nloc_z = idx_z.size
        nloc_y = idx_y.size

        wz = w[:nloc_z]
        wy = w[nloc_z:nloc_z+nloc_y]
        wyc = w[-1]

        if standardize:
            est_work = wz @ z[idx_z] + wy @ y[idx_y] + wyc * y0[j]
            var_work = max(1.0 - float(w @ r), 0.0)

            est[j] = mu_z + sd_z * est_work
            var[j] = (sd_z ** 2) * var_work
        else:
            est[j] = (
                mz
                + wz @ (z[idx_z] - mz)
                + wy @ (y[idx_y] - my)
                + wyc * (y0[j] - my)
            )
            var[j] = max(1.0 - float(w @ r), 0.0)

        if return_weights:
            if use_local_z:
                wz_full[j] = (idx_z.copy(), wz.copy())
            else:
                wz_full[j, :] = wz

            if use_local_y:
                wy_full[j] = (idx_y.copy(), wy.copy())
            else:
                wy_full[j, :] = wy

            wy0[j] = wyc

    if return_weights:
        return est, var, wz_full, wy_full, wy0
    return est, var

def ordinary_cokriging(
    primary_values,
    primary_coords,
    secondary_values,
    secondary_coords,
    targets,
    *,
    covariance_mode: str = "direct",
    primary_model: Optional[dict] = None,
    secondary_model: Optional[dict] = None,
    cross_model: Optional[dict] = None,
    structures: Optional[Sequence[dict]] = None,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    standardize: bool = True,
    jitter: float = 1e-10,
    chunk_cols: Optional[int] = None,
    max_neighbors_primary: Optional[int] = None,
    max_neighbors_secondary: Optional[int] = None,
    target_batch_size: Optional[int] = None,
    balltree_leaf_size: int = 40,
    check_positive_definite: bool = True,
    return_weights: bool = False,
    show_progress: bool = True,
):
    """
    Single-constraint ordinary cokriging (OCK) for a primary variable Z and a
    secondary variable Y.

    The ordinary unbiasedness constraint is imposed across all cokriging weights:

        sum(w_primary) + sum(w_secondary) = 1

    Parameters
    ----------
    primary_values, secondary_values : array_like
        Observed primary and secondary sample values.
    primary_coords, secondary_coords : array_like
        Coordinates associated with the primary and secondary samples.
    targets : array_like
        Prediction locations for the primary variable.
    covariance_mode : {'direct', 'lmc'}, default 'direct'
        Covariance representation used to assemble the cokriging system.

        - 'direct' uses primary_model, secondary_model, and cross_model directly.
        - 'lmc' uses a Linear Model of Coregionalization supplied through
          structures.
    primary_model, secondary_model, cross_model : dict or None
        Direct covariance specifications used when covariance_mode='direct'.
    structures : sequence of dict or None
        LMC structures used when covariance_mode='lmc'.
    distance_type : {'geographic', 'euclidean', 'cartesian', 'angular'}
        Distance convention passed to pairwise_distances(...).
    projection : str, default 'WGS84'
        Projection / ellipsoid argument passed through where relevant.
    rotation_matrix : dict or None, default None
        Optional anisotropy specification.

        - In covariance_mode='direct', this must be a single planar rotation
          dict with keys {'azimuth', 'a_max', 'a_min'}.
        - In covariance_mode='lmc', this may be a per-structure mapping with one
          entry for each LMC structure.
    standardize : bool, default True
        If True, the primary and secondary sample values are standardized
        internally before ordinary cokriging and the final estimates /
        variances are back-transformed to the primary scale.

        Important: this option standardizes the sample values only. It does not
        automatically rescale primary_model, secondary_model, cross_model, or
        LMC structures. Therefore standardize=True is theoretically consistent
        only when the supplied covariance representation is already expressed on
        the same standardized working scale (for example correlation/unit-
        variance models, normalized variogram forms, or LMC structures on the
        correlation scale).

        This is the recommended mode for the single-constraint ordinary
        cokriging formulation when the covariance model is already on that
        common working scale.
    jitter : float, default 1e-10
        Nonnegative diagonal stabilization used during covariance assembly.
    chunk_cols : int or None, default None
        Optional chunk size passed to pairwise_distances(...).
    max_neighbors_primary, max_neighbors_secondary : int or None, default None
        Optional local neighborhoods for the primary and secondary samples.
    target_batch_size : int or None, default None
        Optional number of targets processed per batch in the global solve.

        This is used only when no local neighborhood is active. In that case
        the conditioning matrix is factorized once and the target-side right-
        hand side is processed in batches. If None, all targets are processed
        in one batch.
    balltree_leaf_size : int, default 40
        Leaf size passed to the BallTree-based neighbor search in local mode.
    check_positive_definite : bool, default True
        If True, raise an error when the assembled block covariance matrix is
        not numerically positive definite.
    return_weights : bool, default False
        If True, also return the primary and secondary cokriging weights.
    show_progress : bool, default True
        If True, show a progress bar in local mode and in global mode when the
        targets are processed in multiple batches.

    Returns
    -------
    est : (m,) ndarray
        Ordinary cokriging estimates of the primary variable at the targets.
    var : (m,) ndarray
        Ordinary cokriging variances of the primary variable at the targets.
    wz, wy : optional
        Weight outputs for the primary and secondary conditioning data.

        For each variable separately:
        - if that variable uses a global neighborhood, the returned weights are
          a dense matrix of shape (m, n_variable);
        - if that variable uses a local neighborhood, the returned weights are
          a list of length m where
              w[j] = (idx_j, w_j)
          stores only the weights associated with the selected neighbors for
          target j.

        Therefore mixed local/global runs may return one dense matrix and one
        list-of-tuples.

    Notes
    -----
    When standardize=False, the user is responsible for supplying variables on a
    common scale for which a single shared ordinary-mean constraint is sensible.

    More generally, the assembled covariance system must already be on the same
    working scale as the values passed to the solver.

    In global mode, target_batch_size may be used to reuse a single
    factorization of the conditioning matrix while processing the target-side
    right-hand side in batches.
    """
    prep = _prepare_simple_cokriging_inputs(
        primary_values,
        primary_coords,
        secondary_values,
        secondary_coords,
        targets,
        distance_type=distance_type,
        standardize=standardize,
        mean_primary="sample",
        mean_secondary="sample",
    )

    dt = prep["distance_type"]
    covariance_mode = str(covariance_mode).lower().strip()

    if covariance_mode == "direct":
        _validate_simple_cokriging_runtime_options(
            covariance_mode=covariance_mode,
            distance_type=dt,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            max_neighbors_primary=max_neighbors_primary,
            max_neighbors_secondary=max_neighbors_secondary,
            target_batch_size=target_batch_size,
            balltree_leaf_size=balltree_leaf_size,
            jitter=jitter,
        )

        if primary_model is None or secondary_model is None or cross_model is None:
            raise ValueError(
                "For covariance_mode='direct', primary_model, secondary_model, and cross_model are required."
            )

        def assemble_conditioning_fn(Xz_, Xy_):
            return _assemble_sck_conditioning_matrix(
                Xz_, Xy_,
                primary_model=primary_model,
                secondary_model=secondary_model,
                cross_model=cross_model,
                distance_type=dt,
                projection=projection,
                rotation_matrix=rotation_matrix,
                jitter=jitter,
                chunk_cols=chunk_cols,
            )

        def assemble_rhs_fn(Xz_, Xy_, XT_):
            return _assemble_sck_rhs(
                Xz_, Xy_, XT_,
                primary_model=primary_model,
                cross_model=cross_model,
                distance_type=dt,
                projection=projection,
                rotation_matrix=rotation_matrix,
                chunk_cols=chunk_cols,
            )

        def assemble_system_fn(Xz_, Xy_, XT_):
            K_, sigma2_primary_ = assemble_conditioning_fn(Xz_, Xy_)
            k0_ = assemble_rhs_fn(Xz_, Xy_, XT_)
            return K_, k0_, sigma2_primary_

    elif covariance_mode == "lmc":
        if structures is None:
            raise ValueError("For covariance_mode='lmc', structures must be provided.")

        _validate_simple_cokriging_runtime_options(
            covariance_mode=covariance_mode,
            distance_type=dt,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            max_neighbors_primary=max_neighbors_primary,
            max_neighbors_secondary=max_neighbors_secondary,
            target_batch_size=target_batch_size,
            balltree_leaf_size=balltree_leaf_size,
            jitter=jitter,
            n_lmc_structures=len(structures),
        )

        def assemble_conditioning_fn(Xz_, Xy_):
            return _assemble_lmc_sck_conditioning_matrix(
                Xz_, Xy_,
                structures=structures,
                distance_type=dt,
                projection=projection,
                rotation_matrix=rotation_matrix,
                jitter=jitter,
                chunk_cols=chunk_cols,
            )

        def assemble_rhs_fn(Xz_, Xy_, XT_):
            return _assemble_lmc_sck_rhs(
                Xz_, Xy_, XT_,
                structures=structures,
                distance_type=dt,
                projection=projection,
                rotation_matrix=rotation_matrix,
                chunk_cols=chunk_cols,
            )

        def assemble_system_fn(Xz_, Xy_, XT_):
            K_, sigma2_primary_ = assemble_conditioning_fn(Xz_, Xy_)
            k0_ = assemble_rhs_fn(Xz_, Xy_, XT_)
            return K_, k0_, sigma2_primary_

    else:
        raise ValueError("covariance_mode must be 'direct' or 'lmc'.")

    return _run_ordinary_cokriging_core(
        z=prep["z"],
        y=prep["y"],
        Xz=prep["Xz"],
        Xy=prep["Xy"],
        XT=prep["XT"],
        mu_z=prep["mu_z"],
        sd_z=prep["sd_z"],
        standardize=prep["standardize"],
        assemble_system_fn=assemble_system_fn,
        assemble_conditioning_fn=assemble_conditioning_fn,
        assemble_rhs_fn=assemble_rhs_fn,
        target_batch_size=target_batch_size,
        distance_type=dt,
        max_neighbors_primary=max_neighbors_primary,
        max_neighbors_secondary=max_neighbors_secondary,
        balltree_leaf_size=balltree_leaf_size,
        check_positive_definite=check_positive_definite,
        return_weights=return_weights,
        show_progress=show_progress,
    )


def ordinary_collocated_cokriging(
    primary_values,
    primary_coords,
    collocated_secondary_values,
    targets,
    *,
    markov_model: str = "mm1",
    rho0: Optional[float] = None,
    primary_values_for_rho0=None,
    secondary_values_for_rho0=None,
    rho0_method: str = "pearson",
    primary_model: Optional[dict] = None,
    secondary_model: Optional[dict] = None,
    residual_model: Optional[dict] = None,
    secondary_values_for_standardization=None,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    standardize: bool = True,
    jitter: float = 1e-10,
    chunk_cols: Optional[int] = None,
    max_neighbors_primary: Optional[int] = None,
    target_batch_size: Optional[int] = None,
    balltree_leaf_size: int = 40,
    check_positive_definite: bool = True,
    return_weights: bool = False,
    show_progress: bool = True,
):
    """
    Single-constraint ordinary collocated cokriging (OCCK) with Markov model I
    or II.

    OCCK uses neighboring primary samples Z(x_i) and one collocated secondary
    datum Y(u) at each target u, with the ordinary unbiasedness constraint

        sum(w_primary) + w_collocated_secondary = 1

    Parameters
    ----------
    primary_values : array_like
        Observed primary sample values.
    primary_coords : array_like
        Coordinates of the primary samples.
    collocated_secondary_values : array_like
        Secondary values collocated with the target locations.
    targets : array_like
        Target coordinates.
    markov_model : {'mm1', 'mm2'}, default 'mm1'
        Markov closure used for OCCK.

        - 'mm1' uses rho_z(h) as the primary direct shape and
          rho_zy(h) = rho0 * rho_z(h).
        - 'mm2' uses rho_y(h) and rho_r(h) to reconstruct
          rho_z(h) = rho0**2 * rho_y(h) + (1 - rho0**2) * rho_r(h),
          and rho_zy(h) = rho0 * rho_y(h).
    rho0 : float or None, default None
        Collocated primary-secondary correlation. If None, it is estimated from
        primary_values_for_rho0 and secondary_values_for_rho0.
    primary_values_for_rho0, secondary_values_for_rho0 : array_like or None
        Paired colocated values used to estimate rho0 when needed.
    rho0_method : {'pearson', 'uncentered'}, default 'pearson'
        Estimator used when rho0 is inferred.
    primary_model : dict or None
        Direct primary covariance/correlation model rho_z(h), required for MM1.
    secondary_model : dict or None
        Direct secondary covariance/correlation model rho_y(h), required for MM2.
    residual_model : dict or None
        Direct residual covariance/correlation model rho_r(h), required for MM2.
    secondary_values_for_standardization : array_like or None, default None
        Reference secondary values defining the location/scale used to place the
        collocated target-side secondary data on the OCCK working scale when
        standardize=True.

        When standardize=False, these values are ignored except that they may be
        supplied for API consistency with SCCK.
    distance_type : {'geographic', 'euclidean', 'cartesian', 'angular'}
        Distance convention passed to pairwise_distances(...).
    projection : str, default 'WGS84'
        Projection / ellipsoid argument passed through where relevant.
    rotation_matrix : dict or None, default None
        Optional planar anisotropy specification with keys
        {'azimuth', 'a_max', 'a_min'}.
    standardize : bool, default True
        If True, the primary samples and collocated secondary values are placed
        on the standardized OCCK working scale internally, and the final OCCK
        estimate / variance are back-transformed to the primary scale.

        If False, the inputs are assumed already to be on a common OCCK working
        scale for which the single ordinary constraint is meaningful.
    jitter : float, default 1e-10
        Nonnegative diagonal stabilization.
    chunk_cols : int or None, default None
        Optional chunk size passed to pairwise_distances(...).
    max_neighbors_primary : int or None, default None
        Optional primary neighborhood size. If omitted, a global solve is used.
    target_batch_size : int or None, default None
        Optional number of targets processed per batch in the global solve.

        This is used only when no local primary neighborhood is active. In that
        case the target-invariant OCCK neighbor block is factorized once and the
        target-dependent collocated terms are processed in batches. If None, all
        targets are processed in one batch.
    balltree_leaf_size : int, default 40
        Leaf size passed to the BallTree-based neighbor search in local mode.
    check_positive_definite : bool, default True
        If True, raise an error when the assembled OCCK matrix is not numerically
        positive definite.
    return_weights : bool, default False
        If True, also return the primary-data weights and collocated-secondary
        weights.
    show_progress : bool, default True
        If True, show a progress bar in local mode and in global mode when the
        targets are processed in multiple batches.

    Returns
    -------
    est : (m,) ndarray
        OCCK estimates of the primary variable at the targets.
    var : (m,) ndarray
        OCCK variances on the primary scale when standardize=True, otherwise on
        the supplied OCCK working scale.
    wz : optional
        Primary-data weights.

        - If the primary neighborhood is global, this is a dense matrix of
          shape (m, nz).
        - If the primary neighborhood is local, this is a list of length m
          where
              wz[j] = (idx_z_j, wz_j)
          stores only the primary weights for the selected neighbors of target j.
    wy0 : (m,) ndarray, optional
        Collocated-secondary weights, returned only when return_weights=True.

    Notes
    -----
    This implementation is direct-only.

    In global mode, target_batch_size may be used to reuse a single
    factorization of the target-invariant OCCK neighbor block while processing
    the target-dependent collocated terms in batches. In local mode,
    target_batch_size is ignored.
    """
    markov_model = str(markov_model).lower().strip()

    if markov_model not in ("mm1", "mm2"):
        raise ValueError("markov_model must be 'mm1' or 'mm2'.")

    if markov_model == "mm1":
        if primary_model is None:
            raise ValueError("MM1 requires primary_model.")

    if markov_model == "mm2":
        if secondary_model is None:
            raise ValueError("MM2 requires secondary_model for rho_y(h).")
        if residual_model is None:
            raise ValueError("MM2 requires residual_model for rho_r(h).")

    dt = str(distance_type).lower().strip()
    if dt == "geographical":
        dt = "geographic"

    z_raw, Xz = _clean_values_and_coords(
        primary_values,
        primary_coords,
        distance_type=dt,
        name="primary",
    )
    XT = _coerce_coordinates(targets, dt, "targets")

    if standardize:
        z, mu_z, sd_z = _standardize_values(z_raw, name="primary")
    else:
        z = z_raw.copy()
        mu_z = None
        sd_z = 1.0

    _validate_simple_cokriging_runtime_options(
        covariance_mode="direct",
        distance_type=dt,
        rotation_matrix=rotation_matrix,
        chunk_cols=chunk_cols,
        max_neighbors_primary=max_neighbors_primary,
        max_neighbors_secondary=None,
        target_batch_size=target_batch_size,
        balltree_leaf_size=balltree_leaf_size,
        jitter=jitter,
        n_lmc_structures=None,
    )

    rho0 = _resolve_rho0(
        rho0,
        primary_values_for_rho0=primary_values_for_rho0,
        secondary_values_for_rho0=secondary_values_for_rho0,
        method=rho0_method,
    )

    if standardize:
        if secondary_values_for_standardization is None:
            if secondary_values_for_rho0 is not None:
                secondary_values_for_standardization = secondary_values_for_rho0
            else:
                raise ValueError(
                    "For ordinary_collocated_cokriging(..., standardize=True), "
                    "provide secondary_values_for_standardization or "
                    "secondary_values_for_rho0."
                )
    else:
        if secondary_values_for_standardization is None:
            secondary_values_for_standardization = collocated_secondary_values

    y0, _, _ = _prepare_collocated_secondary_at_targets(
        collocated_secondary_values,
        XT,
        secondary_training_values=secondary_values_for_standardization,
        standardize=standardize,
    )

    nz = z.size
    m = XT.shape[0]

    use_local = (max_neighbors_primary is not None) and (int(max_neighbors_primary) < nz)

    est = np.empty(m, dtype=float)
    var = np.empty(m, dtype=float)

    if return_weights:
        if use_local:
            wz_full = [None] * m
        else:
            wz_full = np.empty((m, nz), dtype=float)
        wy0 = np.empty(m, dtype=float)
    else:
        wz_full = None
        wy0 = None

    if not use_local:
        A = _assemble_scck_base_conditioning_matrix(
            Xz,
            markov_model=markov_model,
            rho0=rho0,
            primary_model=primary_model,
            secondary_model=secondary_model,
            residual_model=residual_model,
            distance_type=dt,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            jitter=jitter,
        )
        cf = _factor_cokriging_matrix(
            A,
            check_positive_definite=check_positive_definite,
        )
        del A

        if target_batch_size is None:
            batch_size = m
        else:
            batch_size = int(target_batch_size)
            if batch_size <= 0:
                raise ValueError("target_batch_size must be None or a positive integer.")

        batch_iter = range(0, m, batch_size)
        if show_progress and batch_size < m:
            batch_iter = tqdm(batch_iter, desc="Processing OCCK batches")

        tail_diag = 1.0 + jitter

        for j0 in batch_iter:
            j1 = min(j0 + batch_size, m)

            rz0_b, cy0_b = _assemble_scck_target_blocks(
                Xz,
                XT[j0:j1],
                markov_model=markov_model,
                rho0=rho0,
                primary_model=primary_model,
                secondary_model=secondary_model,
                residual_model=residual_model,
                distance_type=dt,
                projection=projection,
                rotation_matrix=rotation_matrix,
                chunk_cols=chunk_cols,
            )

            wz_t, wy_b, mu_b = _solve_batched_collocated_ordinary_from_factor(
                cf,
                rz0_b,
                cy0_b,
                rhs_tail=rho0,
                tail_diag=tail_diag,
            )
            wz_b = wz_t.T

            est_work_b = wz_b @ z + wy_b * y0[j0:j1]
            var_work_b = np.maximum(
                1.0 - np.einsum("ij,ij->j", wz_t, rz0_b) - wy_b * float(rho0) - mu_b,
                0.0,
            )

            if standardize:
                est[j0:j1] = mu_z + sd_z * est_work_b
                var[j0:j1] = (sd_z ** 2) * var_work_b
            else:
                est[j0:j1] = est_work_b
                var[j0:j1] = var_work_b

            if return_weights:
                wz_full[j0:j1, :] = wz_b
                wy0[j0:j1] = wy_b

        if return_weights:
            return est, var, wz_full, wy0
        return est, var

    nn_z = query_nearest_neighbors_balltree(
        Xz, XT,
        distance_type=dt,
        max_neighbors=max_neighbors_primary,
        balltree_leaf_size=balltree_leaf_size,
    )

    j_iter = tqdm(range(m), desc="Processing OCCK") if show_progress else range(m)

    for j in j_iter:
        idx_z = np.asarray(nn_z[j], dtype=int)

        K, r = _assemble_scck_system_one_target(
            Xz[idx_z],
            XT[j],
            markov_model=markov_model,
            rho0=rho0,
            primary_model=primary_model,
            secondary_model=secondary_model,
            residual_model=residual_model,
            distance_type=dt,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            jitter=jitter,
        )

        Wt, mu = _solve_ordinary_cokriging_system(
            K,
            r[:, None],
            check_positive_definite=check_positive_definite,
        )

        w = Wt[:, 0]
        mu_j = float(mu[0])

        wz = w[:-1]
        wy = w[-1]

        est_work = wz @ z[idx_z] + wy * y0[j]
        var_work = max(1.0 - float(w @ r) - mu_j, 0.0)

        if standardize:
            est[j] = mu_z + sd_z * est_work
            var[j] = (sd_z ** 2) * var_work
        else:
            est[j] = est_work
            var[j] = var_work

        if return_weights:
            wz_full[j] = (idx_z.copy(), wz.copy())
            wy0[j] = wy

    if return_weights:
        return est, var, wz_full, wy0
    return est, var


def ordinary_intrinsic_collocated_cokriging(
    primary_values,
    primary_coords,
    secondary_values,
    secondary_coords,
    collocated_secondary_values,
    targets,
    *,
    markov_model: str = "mm1",
    rho0: Optional[float] = None,
    primary_values_for_rho0=None,
    secondary_values_for_rho0=None,
    rho0_method: str = "pearson",
    primary_model: Optional[dict] = None,
    secondary_model: Optional[dict] = None,
    residual_model: Optional[dict] = None,
    secondary_values_for_standardization=None,
    distance_type: str = "euclidean",
    projection: str = "WGS84",
    rotation_matrix: Optional[dict] = None,
    standardize: bool = True,
    jitter: float = 1e-10,
    chunk_cols: Optional[int] = None,
    max_neighbors_primary: Optional[int] = None,
    max_neighbors_secondary: Optional[int] = None,
    target_batch_size: Optional[int] = None,
    balltree_leaf_size: int = 40,
    check_positive_definite: bool = True,
    return_weights: bool = False,
    show_progress: bool = True,
):
    """
    Single-constraint ordinary intrinsic collocated cokriging (OICCK) with
    Markov model I or II.

    OICCK uses:

        1. neighboring primary samples Z(x_i),
        2. neighboring secondary samples Y(y_j),
        3. one collocated secondary datum Y(u) at the target,

    with the ordinary unbiasedness constraint

        sum(w_primary) + sum(w_secondary) + w_collocated_secondary = 1

    Parameters
    ----------
    primary_values, secondary_values : array_like
        Observed primary and secondary sample values.
    primary_coords, secondary_coords : array_like
        Coordinates associated with the primary and secondary samples.
    collocated_secondary_values : array_like
        Secondary values collocated with the target locations.
    targets : array_like
        Target coordinates.
    markov_model : {'mm1', 'mm2'}, default 'mm1'
        Markov closure used for OICCK.

        - 'mm1' uses rho_z(h) as the primary direct shape and
          rho_zy(h) = rho0 * rho_z(h), with rho_y(h) = rho_z(h).
        - 'mm2' uses rho_y(h) and rho_r(h) to reconstruct
          rho_z(h) = rho0**2 * rho_y(h) + (1 - rho0**2) * rho_r(h),
          and rho_zy(h) = rho0 * rho_y(h).
    rho0 : float or None, default None
        Collocated primary-secondary correlation. If None, it is estimated from
        primary_values_for_rho0 and secondary_values_for_rho0.
    primary_values_for_rho0, secondary_values_for_rho0 : array_like or None
        Paired colocated values used to estimate rho0 when needed.
    rho0_method : {'pearson', 'uncentered'}, default 'pearson'
        Estimator used when rho0 is inferred.
    primary_model : dict or None
        Direct primary covariance/correlation model rho_z(h), required for MM1.
    secondary_model : dict or None
        Direct secondary covariance/correlation model rho_y(h), required for MM2.
    residual_model : dict or None
        Direct residual covariance/correlation model rho_r(h), required for MM2.
    secondary_values_for_standardization : array_like or None, default None
        Reference secondary values defining the location/scale used by OICCK.

        - When standardize=True, these values define the standardization
          transform applied to both neighboring and collocated secondary data.
        - When standardize=False, these values are ignored except that they may
          be supplied for API consistency with ICCK.
    distance_type : {'geographic', 'euclidean', 'cartesian', 'angular'}
        Distance convention passed to pairwise_distances(...).
    projection : str, default 'WGS84'
        Projection / ellipsoid argument passed through where relevant.
    rotation_matrix : dict or None, default None
        Optional planar anisotropy specification with keys
        {'azimuth', 'a_max', 'a_min'}.
    standardize : bool, default True
        If True, OICCK is carried out on the standardized working scale and the
        final estimate / variance are back-transformed to the primary scale.

        If False, the supplied values are assumed already to be on a common
        OICCK working scale for which the single ordinary constraint is
        meaningful.
    jitter : float, default 1e-10
        Nonnegative diagonal stabilization.
    chunk_cols : int or None, default None
        Optional chunk size passed to pairwise_distances(...).
    max_neighbors_primary, max_neighbors_secondary : int or None, default None
        Optional local neighborhood sizes for the primary and secondary samples.
    target_batch_size : int or None, default None
        Optional number of targets processed per batch in the global solve.

        This is used only when no local neighborhood is active. In that case
        the target-invariant OICCK neighbor block is factorized once and the
        target-dependent collocated terms are processed in batches. If None, all
        targets are processed in one batch.
    balltree_leaf_size : int, default 40
        Leaf size passed to the BallTree-based neighbor search in local mode.
    check_positive_definite : bool, default True
        If True, raise an error when the assembled OICCK matrix is not
        numerically positive definite.
    return_weights : bool, default False
        If True, also return the primary weights, secondary weights, and the
        collocated-secondary weight at each target.
    show_progress : bool, default True
        If True, show a progress bar in local mode and in global mode when the
        targets are processed in multiple batches.

    Returns
    -------
    est : (m,) ndarray
        OICCK estimates of the primary variable.
    var : (m,) ndarray
        OICCK variances on the primary scale when standardize=True, otherwise on
        the supplied OICCK working scale.
    wz, wy : optional
        Weight outputs for the neighboring primary and neighboring secondary
        conditioning data.

        For each variable separately:
        - if that variable uses a global neighborhood, the returned weights are
          a dense matrix of shape (m, n_variable);
        - if that variable uses a local neighborhood, the returned weights are
          a list of length m where
              w[j] = (idx_j, w_j)
          stores only the weights associated with the selected neighbors for
          target j.

        Therefore mixed local/global runs may return one dense matrix and one
        list-of-tuples.
    wy0 : (m,) ndarray, optional
        Collocated-secondary weights, returned only when return_weights=True.

    Notes
    -----
    This implementation is direct-only.

    In global mode, target_batch_size may be used to reuse a single
    factorization of the target-invariant OICCK neighbor block while processing
    the target-dependent collocated terms in batches. In local mode,
    target_batch_size is ignored.
    """
    markov_model = str(markov_model).lower().strip()

    if markov_model not in ("mm1", "mm2"):
        raise ValueError("markov_model must be 'mm1' or 'mm2'.")

    if markov_model == "mm1":
        if primary_model is None:
            raise ValueError("MM1 requires primary_model.")

    if markov_model == "mm2":
        if secondary_model is None:
            raise ValueError("MM2 requires secondary_model for rho_y(h).")
        if residual_model is None:
            raise ValueError("MM2 requires residual_model for rho_r(h).")

    dt = str(distance_type).lower().strip()
    if dt == "geographical":
        dt = "geographic"

    z_raw, Xz = _clean_values_and_coords(
        primary_values, primary_coords, distance_type=dt, name="primary"
    )
    y_raw, Xy = _clean_values_and_coords(
        secondary_values, secondary_coords, distance_type=dt, name="secondary"
    )
    XT = _coerce_coordinates(targets, dt, "targets")

    if standardize:
        z, mu_z, sd_z = _standardize_values(z_raw, name="primary")

        if secondary_values_for_standardization is None:
            if secondary_values_for_rho0 is not None:
                secondary_values_for_standardization = secondary_values_for_rho0
            else:
                secondary_values_for_standardization = y_raw

        mu_y_ref, sd_y_ref = _compute_standardization_stats(
            secondary_values_for_standardization,
            name="secondary",
        )
        y = (y_raw - mu_y_ref) / sd_y_ref
    else:
        z = z_raw.copy()
        y = y_raw.copy()
        mu_z = None
        sd_z = 1.0
        if secondary_values_for_standardization is None:
            secondary_values_for_standardization = collocated_secondary_values

    _validate_simple_cokriging_runtime_options(
        covariance_mode="direct",
        distance_type=dt,
        rotation_matrix=rotation_matrix,
        chunk_cols=chunk_cols,
        max_neighbors_primary=max_neighbors_primary,
        max_neighbors_secondary=max_neighbors_secondary,
        target_batch_size=target_batch_size,
        balltree_leaf_size=balltree_leaf_size,
        jitter=jitter,
        n_lmc_structures=None,
    )

    rho0 = _resolve_rho0(
        rho0,
        primary_values_for_rho0=primary_values_for_rho0,
        secondary_values_for_rho0=secondary_values_for_rho0,
        method=rho0_method,
    )

    y0, _, _ = _prepare_collocated_secondary_at_targets(
        collocated_secondary_values,
        XT,
        secondary_training_values=secondary_values_for_standardization,
        standardize=standardize,
    )

    nz = z.size
    ny = y.size
    m = XT.shape[0]

    use_local_z = (max_neighbors_primary is not None) and (int(max_neighbors_primary) < nz)
    use_local_y = (max_neighbors_secondary is not None) and (int(max_neighbors_secondary) < ny)
    use_local = use_local_z or use_local_y

    est = np.empty(m, dtype=float)
    var = np.empty(m, dtype=float)

    if return_weights:
        if use_local_z:
            wz_full = [None] * m
        else:
            wz_full = np.empty((m, nz), dtype=float)

        if use_local_y:
            wy_full = [None] * m
        else:
            wy_full = np.empty((m, ny), dtype=float)

        wy0 = np.empty(m, dtype=float)
    else:
        wz_full = None
        wy_full = None
        wy0 = None

    if not use_local:
        A = _assemble_icck_base_conditioning_matrix(
            Xz,
            Xy,
            markov_model=markov_model,
            rho0=rho0,
            primary_model=primary_model,
            secondary_model=secondary_model,
            residual_model=residual_model,
            distance_type=dt,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            jitter=jitter,
        )
        cf = _factor_cokriging_matrix(
            A,
            check_positive_definite=check_positive_definite,
        )
        del A

        if target_batch_size is None:
            batch_size = m
        else:
            batch_size = int(target_batch_size)
            if batch_size <= 0:
                raise ValueError("target_batch_size must be None or a positive integer.")

        batch_iter = range(0, m, batch_size)
        if show_progress and batch_size < m:
            batch_iter = tqdm(batch_iter, desc="Processing OICCK batches")

        tail_diag = 1.0 + jitter

        for j0 in batch_iter:
            j1 = min(j0 + batch_size, m)

            b0, c0 = _assemble_icck_target_blocks(
                Xz,
                Xy,
                XT[j0:j1],
                markov_model=markov_model,
                rho0=rho0,
                primary_model=primary_model,
                secondary_model=secondary_model,
                residual_model=residual_model,
                distance_type=dt,
                projection=projection,
                rotation_matrix=rotation_matrix,
                chunk_cols=chunk_cols,
            )

            wtop_t, wyc_b, mu_b = _solve_batched_collocated_ordinary_from_factor(
                cf,
                b0,
                c0,
                rhs_tail=rho0,
                tail_diag=tail_diag,
            )

            W_b = wtop_t.T
            wz_b = W_b[:, :nz]
            wy_b = W_b[:, nz:]

            est_work_b = wz_b @ z + wy_b @ y + wyc_b * y0[j0:j1]
            var_work_b = np.maximum(
                1.0 - np.einsum("ij,ij->j", wtop_t, b0) - wyc_b * float(rho0) - mu_b,
                0.0,
            )

            if standardize:
                est[j0:j1] = mu_z + sd_z * est_work_b
                var[j0:j1] = (sd_z ** 2) * var_work_b
            else:
                est[j0:j1] = est_work_b
                var[j0:j1] = var_work_b

            if return_weights:
                wz_full[j0:j1, :] = wz_b
                wy_full[j0:j1, :] = wy_b
                wy0[j0:j1] = wyc_b

        if return_weights:
            return est, var, wz_full, wy_full, wy0
        return est, var

    if use_local_z:
        nn_z = query_nearest_neighbors_balltree(
            Xz, XT,
            distance_type=dt,
            max_neighbors=max_neighbors_primary,
            balltree_leaf_size=balltree_leaf_size,
        )
    else:
        nn_z = [np.arange(nz)] * m

    if use_local_y:
        nn_y = query_nearest_neighbors_balltree(
            Xy, XT,
            distance_type=dt,
            max_neighbors=max_neighbors_secondary,
            balltree_leaf_size=balltree_leaf_size,
        )
    else:
        nn_y = [np.arange(ny)] * m

    j_iter = tqdm(range(m), desc="Processing OICCK") if show_progress else range(m)

    for j in j_iter:
        idx_z = np.asarray(nn_z[j], dtype=int)
        idx_y = np.asarray(nn_y[j], dtype=int)

        K, r = _assemble_icck_system_one_target(
            Xz[idx_z],
            Xy[idx_y],
            XT[j],
            markov_model=markov_model,
            rho0=rho0,
            primary_model=primary_model,
            secondary_model=secondary_model,
            residual_model=residual_model,
            distance_type=dt,
            projection=projection,
            rotation_matrix=rotation_matrix,
            chunk_cols=chunk_cols,
            jitter=jitter,
        )

        Wt, mu = _solve_ordinary_cokriging_system(
            K,
            r[:, None],
            check_positive_definite=check_positive_definite,
        )

        w = Wt[:, 0]
        mu_j = float(mu[0])

        nloc_z = idx_z.size
        nloc_y = idx_y.size

        wz = w[:nloc_z]
        wy = w[nloc_z:nloc_z+nloc_y]
        wyc = w[-1]

        est_work = wz @ z[idx_z] + wy @ y[idx_y] + wyc * y0[j]
        var_work = max(1.0 - float(w @ r) - mu_j, 0.0)

        if standardize:
            est[j] = mu_z + sd_z * est_work
            var[j] = (sd_z ** 2) * var_work
        else:
            est[j] = est_work
            var[j] = var_work

        if return_weights:
            if use_local_z:
                wz_full[j] = (idx_z.copy(), wz.copy())
            else:
                wz_full[j, :] = wz

            if use_local_y:
                wy_full[j] = (idx_y.copy(), wy.copy())
            else:
                wy_full[j, :] = wy

            wy0[j] = wyc

    if return_weights:
        return est, var, wz_full, wy_full, wy0
    return est, var


def OCK(*args, **kwargs):
    """Convenience wrapper for ordinary_cokriging(...)."""
    return ordinary_cokriging(*args, **kwargs)


def OCCK(*args, **kwargs):
    """Convenience wrapper for ordinary_collocated_cokriging(...)."""
    return ordinary_collocated_cokriging(*args, **kwargs)


def OICCK(*args, **kwargs):
    """Convenience wrapper for ordinary_intrinsic_collocated_cokriging(...)."""
    return ordinary_intrinsic_collocated_cokriging(*args, **kwargs)


def OCCK_MM1(*args, **kwargs):
    """Convenience wrapper for ordinary_collocated_cokriging(..., markov_model='mm1')."""
    kwargs = dict(kwargs)
    kwargs["markov_model"] = "mm1"
    return ordinary_collocated_cokriging(*args, **kwargs)


def OCCK_MM2(*args, **kwargs):
    """Convenience wrapper for ordinary_collocated_cokriging(..., markov_model='mm2')."""
    kwargs = dict(kwargs)
    kwargs["markov_model"] = "mm2"
    return ordinary_collocated_cokriging(*args, **kwargs)


def OICCK_MM1(*args, **kwargs):
    """Convenience wrapper for ordinary_intrinsic_collocated_cokriging(..., markov_model='mm1')."""
    kwargs = dict(kwargs)
    kwargs["markov_model"] = "mm1"
    return ordinary_intrinsic_collocated_cokriging(*args, **kwargs)


def OICCK_MM2(*args, **kwargs):
    """Convenience wrapper for ordinary_intrinsic_collocated_cokriging(..., markov_model='mm2')."""
    kwargs = dict(kwargs)
    kwargs["markov_model"] = "mm2"
    return ordinary_intrinsic_collocated_cokriging(*args, **kwargs)

def SCCK_MM1(*args, **kwargs):
    """Convenience wrapper for simple_collocated_cokriging(..., markov_model='mm1')."""
    kwargs = dict(kwargs)
    kwargs["markov_model"] = "mm1"
    return simple_collocated_cokriging(*args, **kwargs)


def SCCK_MM2(*args, **kwargs):
    """Convenience wrapper for simple_collocated_cokriging(..., markov_model='mm2')."""
    kwargs = dict(kwargs)
    kwargs["markov_model"] = "mm2"
    return simple_collocated_cokriging(*args, **kwargs)


def ICCK_MM1(*args, **kwargs):
    """Convenience wrapper for intrinsic_collocated_cokriging(..., markov_model='mm1')."""
    kwargs = dict(kwargs)
    kwargs["markov_model"] = "mm1"
    return intrinsic_collocated_cokriging(*args, **kwargs)


def ICCK_MM2(*args, **kwargs):
    """Convenience wrapper for intrinsic_collocated_cokriging(..., markov_model='mm2')."""
    kwargs = dict(kwargs)
    kwargs["markov_model"] = "mm2"
    return intrinsic_collocated_cokriging(*args, **kwargs)
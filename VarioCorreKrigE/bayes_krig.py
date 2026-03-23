"""
Tools for empirical bayesian kriging using Simple or Ordinary Kriging
and their custom variants.
"""

import numpy as np
from scipy.stats import norm
from tqdm.auto import tqdm

from VarioCorreKrigE.skrig import (
    simple_kriging,
    simple_kriging_custom_corr,
    simple_kriging_custom_vario,
)

from VarioCorreKrigE.okrig import (
    ordinary_kriging,
    ordinary_kriging_custom_corr,
    ordinary_kriging_custom_vario,
)


def _normalize_param_samples(params):
    """
    Normalize built-in parameter samples into a list of dicts.

    Supported inputs
    ----------------
    1) dict[str, array_like] of length S for each parameter
    2) list/tuple of dicts of length S

    Returns
    -------
    params_samples : list[dict]
        List of length S where each entry is one parameter dictionary.
    """
    if params is None:
        return None

    if isinstance(params, dict):
        keys = list(params.keys())
        lengths = [len(np.asarray(params[k])) for k in keys]
        if len(set(lengths)) != 1:
            raise ValueError("All entries in params dict must have the same number of samples.")
        S = lengths[0]
        return [{k: np.asarray(params[k])[i] for k in keys} for i in range(S)]

    if isinstance(params, (list, tuple)):
        if not all(isinstance(p, dict) for p in params):
            raise TypeError("If params is a list/tuple, each element must be a dict.")
        return list(params)

    raise TypeError("params must be either a dict of sample arrays or a list/tuple of dicts.")


def _normalize_theta_samples(theta):
    """
    Normalize theta samples into a list of per-sample theta vectors/scalars.

    Supported inputs
    ----------------
    1) list/tuple of per-sample theta values:
         [theta_0, theta_1, ..., theta_{S-1}]
    2) tuple/list of parameter-wise arrays:
         (p1_samples, p2_samples, ..., pk_samples)
       where each array has length S
    3) 2D ndarray of shape (S, p)

    Returns
    -------
    theta_samples : list
        List of length S where each entry is one theta sample.
    """
    if theta is None:
        return None

    if isinstance(theta, np.ndarray):
        arr = np.asarray(theta)
        if arr.ndim == 2:
            return [arr[i, :] for i in range(arr.shape[0])]
        elif arr.ndim == 1:
            raise ValueError(
                "A 1D ndarray for theta is ambiguous in bayes_krig. "
                "Pass either a list of per-sample theta values, a tuple/list of parameter-wise arrays, "
                "or a 2D ndarray of shape (n_samples, n_parameters)."
            )
        else:
            raise ValueError("theta ndarray must be 2D if provided as a numpy array.")

    if isinstance(theta, (list, tuple)):
        if len(theta) == 0:
            raise ValueError("theta cannot be empty.")

        # Case A: already sample-wise
        first = theta[0]

        # list/tuple of dicts or vectors/scalars
        # detect parameter-wise arrays like:
        # (LE_samples, gammaE_samples, LA_samples, LS_samples, w_samples)
        all_arraylike = all(np.ndim(np.asarray(t)) > 0 for t in theta)
        if all_arraylike:
            lengths = [len(np.asarray(t)) for t in theta]
            same_len = len(set(lengths)) == 1

            # heuristic: parameter-wise if number of elements is small
            # and each element is a longer sample array
            if same_len and lengths[0] > len(theta):
                S = lengths[0]
                theta_arrays = [np.asarray(t) for t in theta]
                return [tuple(arr[i] for arr in theta_arrays) for i in range(S)]

        # otherwise assume already sample-wise
        return list(theta)

    raise TypeError(
        "theta must be a list/tuple of per-sample values, "
        "a tuple/list of parameter-wise arrays, or a 2D ndarray."
    )


def _sample_or_scalar(x, i):
    """
    Return x[i] if x is sample-varying, otherwise return x unchanged.
    """
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.ndim == 0:
        return x
    return arr[i]


def bayes_krig(
    values,
    *,
    coords=None,
    targets=None,
    krig_type="sk",
    model_family="variogram",
    model_type="exponential",
    distance_type="geographic",
    thr=None,
    quantile_levels=(0.05, 0.95),
    max_neighbors=None,
    mean=None,
    sigma2=None,
    custom_model_family=None,
    custom_kernel=None,
    blocks_nn=None,
    blocks_nt=None,
    params=None,
    theta=None,
    projection="WGS84",
    jitter=1e-10,
    chunk_cols=None,
    balltree_leaf_size=40,
    random_state=1234,
    progress=True,
):
    """
    Empirical-Bayesian kriging wrapper for SK / OK, built-in or custom kernels.

    This function propagates uncertainty in fitted kriging parameters by looping
    over posterior or bootstrap samples, computing one kriging prediction/variance
    pair per sample, and then combining them into posterior predictive summaries.

    Modes
    -----
    Built-in kriging
        Uses `simple_kriging(...)` or `ordinary_kriging(...)`.

        Required:
        - coords
        - targets
        - model_family
        - model_type
        - one of:
            * params : parameter samples
            * theta  : theta samples

    Custom kriging
        Uses one of:
        - simple_kriging_custom_corr
        - ordinary_kriging_custom_corr
        - simple_kriging_custom_vario
        - ordinary_kriging_custom_vario

        Required:
        - blocks_nn
        - blocks_nt
        - custom_kernel
        - custom_model_family in {'correlation', 'variogram'}
        - theta : posterior samples

        Additionally for custom correlation:
        - sigma2 must be provided (scalar or length-S samples)

        For custom variogram:
        - theta must contain everything the custom variogram kernel needs.
          By your current convention, the last two entries of theta should be
          c0 and b if the kernel is written that way.

    Parameters
    ----------
    values : array_like, shape (n,)
        Observed values.

    coords : array_like, optional
        Observation coordinates for built-in kriging only.

    targets : array_like, optional
        Target coordinates for built-in kriging only.

    krig_type : {'sk', 'ok'}, default 'sk'
        Kriging type:
        - 'sk' : simple kriging
        - 'ok' : ordinary kriging

    model_family : {'variogram', 'correlation'}, default 'variogram'
        Built-in model family. Used only when `custom_kernel is None`.

    model_type : str, default 'exponential'
        Built-in model name. Used only when `custom_kernel is None`.

    distance_type : {'euclidean', 'cartesian', 'geographic', 'angular'}, default 'geographic'
        Distance type passed to built-in kriging.

    thr : float, optional
        Threshold for exceedance probability P(Y > thr | D).
        If None, the sample median of `values` is used.

    quantile_levels : tuple(float, float), default (0.05, 0.95)
        Lower and upper posterior predictive quantiles.

    max_neighbors : int, optional
        Local-neighborhood size for built-in SK/OK.

    mean : {'gls', 'zero'} or float, optional
        Mean for simple kriging only.
        If None and `krig_type='sk'`, defaults to 'gls'.

    sigma2 : float or array_like, optional
        Variance scale.
        - Built-in correlation: optional scalar or length-S samples
        - Custom correlation: required scalar or length-S samples
        - Ignored for variogram models

    custom_model_family : {'correlation', 'variogram'}, optional
        Custom model family. Required when `custom_kernel` is provided.

    custom_kernel : callable, optional
        Custom kernel function for custom kriging.

    blocks_nn, blocks_nt : dict[str, ndarray], optional
        Precomputed custom distance/dissimilarity blocks.

    params : dict or list of dicts, optional
        Built-in parameter samples.

    theta : sample container, optional
        Posterior/bootstrap theta samples.

        Supported forms:
        - list/tuple of per-sample theta values
        - tuple/list of parameter-wise arrays
        - 2D ndarray of shape (n_samples, n_parameters)

    projection : str, default 'WGS84'
        Projection passed to built-in kriging.

    jitter : float, default 1e-10
        Numerical jitter.

    chunk_cols : int, optional
        Chunking for built-in pairwise distance calculations.

    balltree_leaf_size : int, default 40
        BallTree leaf size for built-in local kriging.

    random_state : int, default 1234
        Random seed used for posterior predictive draws.

    progress : bool, default True
        If True, show a progress bar.

    Returns
    -------
    posterior_out : dict
        {
            "mu":  all_mu,   # (S, m)
            "var": all_var,  # (S, m)
        }

    posterior_predictive : dict
        {
            "mu": pred,
            "var": var,
            "std": std_error,
            "within_var": within,
            "between_var": between,
        }

    probability_of_exceedance : dict
        {
            "poe": prob,
            "threshold": thr,
        }

    posterior_quantiles : dict
        {
            "quantiles": pred_q,
            "upper_quantile": pred_qupp,
            "lower_quantile": pred_qlow,
        }
    """

    z = np.asarray(values, float).ravel()
    X = None if coords is None else np.asarray(coords, float)
    XT = None if targets is None else np.asarray(targets, float)

    krig_type = krig_type.lower()
    if krig_type not in {"sk", "ok"}:
        raise ValueError("krig_type must be 'sk' or 'ok'")

    qlow, qupp = quantile_levels
    if not (0.0 < qlow < qupp < 1.0):
        raise ValueError("quantile_levels must satisfy 0 < qlow < qupp < 1")

    if thr is None:
        thr = float(np.median(z))

    is_custom = custom_kernel is not None

    # ------------------------------------------------------------------
    # choose solver and normalize samples
    # ------------------------------------------------------------------
    if is_custom:
        if blocks_nn is None or blocks_nt is None:
            raise ValueError("For custom kriging, provide blocks_nn and blocks_nt.")
        if theta is None:
            raise ValueError("For custom kriging, provide posterior theta samples.")
        if custom_model_family not in {"correlation", "variogram"}:
            raise ValueError("custom_model_family must be 'correlation' or 'variogram'")

        theta_samples = _normalize_theta_samples(theta)
        n_samples = len(theta_samples)

        if custom_model_family == "correlation":
            solver = simple_kriging_custom_corr if krig_type == "sk" else ordinary_kriging_custom_corr
            if sigma2 is None:
                raise ValueError("For custom correlation kriging, provide sigma2.")
        else:
            solver = simple_kriging_custom_vario if krig_type == "sk" else ordinary_kriging_custom_vario

        param_samples = None

    else:
        if X is None:
            raise ValueError("For built-in kriging, provide coords.")
        if XT is None:
            raise ValueError("For built-in kriging, provide targets.")
        if params is None and theta is None:
            raise ValueError("Provide either params or theta for built-in kriging.")
        if model_family is None or model_type is None:
            raise ValueError("For built-in kriging, provide model_family and model_type.")

        solver = simple_kriging if krig_type == "sk" else ordinary_kriging

        param_samples = _normalize_param_samples(params)
        theta_samples = _normalize_theta_samples(theta)

        if param_samples is not None:
            n_samples = len(param_samples)
        else:
            n_samples = len(theta_samples)

    if mean is None and krig_type == "sk":
        mean = "gls"

    all_mu = []
    all_var = []

    iterator = tqdm(range(n_samples), desc="Processing samples") if progress else range(n_samples)

    # ------------------------------------------------------------------
    # loop over posterior / bootstrap samples
    # ------------------------------------------------------------------
    for i in iterator:

        if is_custom:
            theta_i = theta_samples[i]

            if custom_model_family == "correlation":
                sigma2_i = _sample_or_scalar(sigma2, i)

                if krig_type == "sk":
                    est, var = solver(
                        values=z,
                        blocks_nn=blocks_nn,
                        blocks_nt=blocks_nt,
                        custom_kernel=custom_kernel,
                        theta=theta_i,
                        sigma2=sigma2_i,
                        mean=mean,
                        jitter=jitter,
                        return_weights=False,
                    )
                else:
                    est, var = solver(
                        values=z,
                        blocks_nn=blocks_nn,
                        blocks_nt=blocks_nt,
                        custom_kernel=custom_kernel,
                        theta=theta_i,
                        sigma2=sigma2_i,
                        jitter=jitter,
                        return_weights=False,
                    )

            else:  # custom variogram
                if krig_type == "sk":
                    est, var = solver(
                        values=z,
                        blocks_nn=blocks_nn,
                        blocks_nt=blocks_nt,
                        custom_kernel=custom_kernel,
                        theta=theta_i,
                        mean=mean,
                        jitter=jitter,
                        return_weights=False,
                    )
                else:
                    est, var = solver(
                        values=z,
                        blocks_nn=blocks_nn,
                        blocks_nt=blocks_nt,
                        custom_kernel=custom_kernel,
                        theta=theta_i,
                        jitter=jitter,
                        return_weights=False,
                    )

        else:
            params_i = None if param_samples is None else param_samples[i]
            theta_i = None if theta_samples is None else theta_samples[i]
            sigma2_i = _sample_or_scalar(sigma2, i)

            if krig_type == "sk":
                est, var = solver(
                    values=z,
                    coords=X,
                    targets=XT,
                    model_family=model_family,
                    model_type=model_type,
                    theta=theta_i,
                    params=params_i,
                    sigma2=sigma2_i,
                    distance_type=distance_type,
                    projection=projection,
                    mean=mean,
                    jitter=jitter,
                    chunk_cols=chunk_cols,
                    max_neighbors=max_neighbors,
                    balltree_leaf_size=balltree_leaf_size,
                    return_weights=False,
                )
            else:
                est, var = solver(
                    values=z,
                    coords=X,
                    targets=XT,
                    model_family=model_family,
                    model_type=model_type,
                    theta=theta_i,
                    params=params_i,
                    sigma2=sigma2_i,
                    distance_type=distance_type,
                    projection=projection,
                    jitter=jitter,
                    chunk_cols=chunk_cols,
                    max_neighbors=max_neighbors,
                    balltree_leaf_size=balltree_leaf_size,
                    return_weights=False,
                )

        all_mu.append(np.asarray(est, float))
        all_var.append(np.clip(np.asarray(var, float), 0.0, None))

    all_mu = np.stack(all_mu, axis=0)
    all_var = np.stack(all_var, axis=0)
    all_std = np.sqrt(all_var)

    # posterior predictive summaries
    pred = all_mu.mean(axis=0)
    within = all_var.mean(axis=0)
    between = ((all_mu - pred) ** 2).mean(axis=0)
    var = within + between
    std_error = np.sqrt(var)

    # exceedance probability: P(Y > thr | D)
    z_thr = (thr - all_mu) / np.where(all_std > 0.0, all_std, 1.0)
    component_prob = np.where(
        all_std > 0.0,
        1.0 - norm.cdf(z_thr),
        (all_mu > thr).astype(float),
    )
    prob = component_prob.mean(axis=0)

    # posterior predictive quantiles
    rng = np.random.default_rng(random_state)
    predictive_draws = rng.normal(loc=all_mu, scale=all_std)

    pred_qlow = np.quantile(predictive_draws, qlow, axis=0)
    pred_qupp = np.quantile(predictive_draws, qupp, axis=0)
    pred_q = {q: np.quantile(predictive_draws, q, axis=0) for q in quantile_levels}

    posterior_out = {
        "mu": all_mu,
        "var": all_var,
    }

    posterior_predictive = {
        "mu": pred,
        "var": var,
        "std": std_error,
        "within_var": within,
        "between_var": between,
    }

    probability_of_exceedance = {
        "poe": prob,
        "threshold": thr,
    }

    posterior_quantiles = {
        "quantiles": pred_q,
        "upper_quantile": pred_qupp,
        "lower_quantile": pred_qlow,
    }

    return posterior_out, posterior_predictive, probability_of_exceedance, posterior_quantiles

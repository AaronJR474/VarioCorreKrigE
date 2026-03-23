from pyproj import Geod
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import rasterio
from pyproj import CRS, Transformer
from pathlib import Path

# get parameters from family of models: correlation or variogram
def theta_from_params(params, model_type, family):
    """
    Convert a parameter dictionary into the callable parameter order expected by
    the built-in variogram or correlation models.

    Notes
    -----
    For correlation models, this accepts either:
      - explicit alpha, or
      - c0 and b, from which alpha = c0 / (c0 + b) is derived.
    """
    params = dict(params)

    if family == 'variogram':
        if model_type in ('spherical', 'exponential', 'gaussian', 'cubic'):
            order = ('r', 'c0', 'b')
            return [float(params[k]) for k in order]

        elif model_type == 'powered_exponential':
            order = ('r', 'c0', 'beta', 'b')
            return [float(params[k]) for k in order]

        elif model_type == 'matern':
            order = ('r', 'c0', 's', 'b')
            return [float(params[k]) for k in order]

        elif model_type in ('damped_cosine_angle', 'angular_dissimilarity'):
            order = ('c', 'c0', 'b')
            return [float(params[k]) for k in order]

        else:
            raise ValueError("Unknown model_type")

    elif family == 'correlation':
        if 'alpha' in params:
            alpha = float(params['alpha'])
        elif ('c0' in params) or ('b' in params):
            c0 = float(params.get('c0', 0.0))
            b = float(params.get('b', 0.0))
            s2 = c0 + b
            alpha = 1.0 if s2 <= 0.0 else c0 / s2
        else:
            alpha = 1.0

        if model_type in ('spherical', 'exponential', 'gaussian', 'cubic'):
            return [float(params['r']), alpha]

        elif model_type == 'powered_exponential':
            return [float(params['r']), float(params['beta']), alpha]

        elif model_type == 'matern':
            return [float(params['r']), float(params['s']), alpha]

        elif model_type in ('damped_cosine_angle', 'angular_dissimilarity'):
            return [float(params['c']), alpha]

        else:
            raise ValueError("Unknown model_type")

    else:
        raise ValueError("family must be 'variogram' or 'correlation'")


# Compute weights
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
        'inverse-linear weighting'            : w(h)=n_j * 1/(1+h/b)
        'exponential weighting'               : w(h)=n_j * exp(-h/b)
        'powered weighting'                   : w(h)=n_j * (1+h/b)^(-alpha)
        'linear weighting'                    : w(h)=n_j * ones(h)
        'inverse-linear squared weighting'    : w(h)=n_j / h^2
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
            raise ValueError("inverse-linear weighting requires weight_params with b > 0.")
        w = n_j / (1.0 + h_lag / b)

    elif weight_type == 'exponential weighting':
        if b is None or b <= 0:
            raise ValueError("exponential weighting requires weight_params with b > 0.")
        w = n_j * np.exp(-h_lag / b)

    elif weight_type == 'powered weighting':
        if b is None or b <= 0 or alpha is None:
            raise ValueError("powered weighting requires weight_params with b > 0 and alpha.")
        w = n_j * (1.0 + h_lag / b) ** (-alpha)

    elif weight_type == 'linear weighting':
        w = n_j * np.ones_like(h_lag, dtype=float)

    elif weight_type == 'inverse-linear squared weighting':
        w = np.where(h_lag > 0.0, n_j / h_lag ** 2, 0.0)

    elif weight_type is None or weight_type == 'ols':
        w = np.ones_like(h_lag, dtype=float)

    else:
        raise ValueError(
            "Invalid weight_type: choose None/'ols', 'inverse-linear weighting', "
            "'exponential weighting', 'powered weighting', "
            "'inverse-linear squared weighting' or 'linear weighting'."
        )

    return w

# function for computing angles and distances from a reference points e.g., Xeq can be an earthquake
def LatLongToPolar(Xst, Xeq):
    geod = Geod(ellps='WGS84')
    r = geod.inv(lons1 = Xeq[:, 1], lats1 = Xeq[:, 0],
        lons2 = Xst[:, 1], lats2 = Xst[:, 0])
    repi = r[2]/1000
    az = r[0]*np.pi/180
    az[az<0] += 2*np.pi
    Xp = np.vstack([repi, az]).T
    return Xp

# sample points from a geotiff
def sample_points_from_geotiff(file_path, target_latlon, band=1, epsg_code = 4326):
    """
    Sample a GeoTIFF (e.g., VS30) at target lat/lon points.

    Parameters
    ----------
    file_path : str | Path
        Path to the GeoTIFF (e.g., r'D:\\...\\combined_mvn_wgs84.tif').
    target_latlon : (m,2) array_like
        Columns [lat, lon] in degrees (EPSG:4326).
    band : int, default=1
        Band index to sample.
    epsg_code: int, default=4326
        coordinate system of lat/long: should match input geotiff

    Returns
    -------
    (m,1) ndarray
        Sampled values; np.nan where points fall outside the raster or hit nodata.
    """
    file_path = Path(file_path)
    target_latlon = np.asarray(target_latlon, float)
    lat = target_latlon[:, 0]
    lon = target_latlon[:, 1]

    with rasterio.open(file_path) as ds:
        if ds.crs is None:
            raise ValueError("Raster CRS is undefined. The GeoTIFF must have a valid CRS.")
        src_crs = CRS.from_epsg(epsg_code)
        if ds.crs == src_crs:
            x, y = lon, lat  # rasterio expects (x=lon, y=lat)
        else:
            transformer = Transformer.from_crs(src_crs, ds.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)

        # mask points outside bounds -> NaN
        b = ds.bounds
        inside = (x >= b.left) & (x <= b.right) & (y >= b.bottom) & (y <= b.top)

        out = np.full(lat.shape, np.nan, dtype=float)
        if np.any(inside):
            coords_in = list(zip(x[inside], y[inside]))
            vals = np.array([v[0] for v in ds.sample(coords_in, indexes=band)], dtype=float)

            # map nodata to NaN
            if ds.nodata is not None:
                vals = np.where(vals == ds.nodata, np.nan, vals)

            # Apply scale/offset if present
            try:
                scale = (ds.scales or [1.0])[band - 1]
                offset = (ds.offsets or [0.0])[band - 1]
                vals = vals * scale + offset
            except Exception:
                pass

            out[inside] = vals

    return out.reshape(-1, 1)

# plotting krig maps in matplotlib for sanity checks
def plot_krig_maps(
    LAT,
    LON,
    est,
    var,
    coords,
    values=None,
    *,
    figsize=(14, 6),
    dpi=200,
    cmap_est="viridis",
    cmap_var="magma",
    log_var=True,
    var_percentiles=(1, 99),
    est_percentiles=None,          # e.g. (1, 99) if you want robust scaling
    aspect_correction=True,
    mean_lat=None,                 # if None, computed from coords[:,0]
    show_points=True,
    point_size=25,
    point_alpha=0.95,
    point_edgecolor="k",
    point_linewidth=0.3,
    var_point_facecolors="none",
    var_point_alpha=0.8,
    var_point_linewidth=0.5,
    titles=("Kriging Estimate", "Kriging Variance"),
    cb_labels=("Estimate", "Kriging variance"),
    grid=True,
    grid_kwargs=None,
    legend=True,
    legend_loc="upper left",
):
    """
    Plot estimate + variance kriging maps with optional observation overlay.
    Works for both Simple Kriging and Ordinary Kriging (same inputs).

    Parameters
    ----------
    LAT, LON : 2D arrays
        Meshgrid arrays (same shape) in degrees.
    est, var : array-like
        Either 1D (ravelled, length = LAT.size) or 2D (LAT.shape).
    coords : (n,2) array
        Observation coordinates as [lat, lon] per row.
    values : (n,) array or None
        Observation values for coloring points on estimate map.
        If None, points are shown in a single color.

    Notes
    -----
    - Uses a shared normalization on the estimate panel (surface + points).
    - Variance panel defaults to log scaling (use log_var=False for linear).
    """

    LAT = np.asarray(LAT)
    LON = np.asarray(LON)
    coords = np.asarray(coords, float)

    est = np.asarray(est, float)
    var = np.asarray(var, float)

    # reshape if needed
    if est.ndim == 1:
        est = est.reshape(LAT.shape)
    if var.ndim == 1:
        var = var.reshape(LAT.shape)

    # estimate normalization (shared between surface and point colors)
    if est_percentiles is None:
        vmin_est = np.nanmin([np.nanmin(est), np.nanmin(values) if values is not None else np.nanmin(est)])
        vmax_est = np.nanmax([np.nanmax(est), np.nanmax(values) if values is not None else np.nanmax(est)])
    else:
        p_lo, p_hi = est_percentiles
        base = est if values is None else np.concatenate([est.ravel(), np.asarray(values, float).ravel()])
        vmin_est, vmax_est = np.nanpercentile(base, [p_lo, p_hi])

    est_norm = mpl.colors.Normalize(vmin=vmin_est, vmax=vmax_est)

    # variance normalization
    if var_percentiles is not None:
        p_lo, p_hi = var_percentiles
        vmin_var, vmax_var = np.nanpercentile(var, [p_lo, p_hi])
    else:
        vmin_var, vmax_var = np.nanmin(var), np.nanmax(var)

    if log_var:
        vmin_var = max(float(vmin_var), 1e-12)
        var_norm = mpl.colors.LogNorm(vmin=vmin_var, vmax=float(vmax_var))
    else:
        var_norm = mpl.colors.Normalize(vmin=float(vmin_var), vmax=float(vmax_var))

    # figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi, constrained_layout=True)

    if grid_kwargs is None:
        grid_kwargs = dict(linestyle="--", alpha=0.3)

    # ----- panel 1: estimate -----
    ax = axes[0]
    m0 = ax.pcolormesh(LON, LAT, est, shading="auto", cmap=plt.get_cmap(cmap_est), norm=est_norm)

    if show_points:
        if values is None:
            ax.scatter(
                coords[:, 1], coords[:, 0],
                s=point_size, c="k",
                edgecolor=point_edgecolor, linewidths=point_linewidth,
                alpha=point_alpha, label="observations"
            )
        else:
            ax.scatter(
                coords[:, 1], coords[:, 0],
                c=np.asarray(values, float),
                cmap=plt.get_cmap(cmap_est),
                norm=est_norm,
                s=point_size,
                edgecolor=point_edgecolor,
                linewidths=point_linewidth,
                alpha=point_alpha,
                label="observations",
            )

    cb0 = plt.colorbar(m0, ax=ax, pad=0.01, fraction=0.05)
    cb0.set_label(cb_labels[0])
    ax.set_title(titles[0])
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    if grid:
        ax.grid(True, **grid_kwargs)
    if legend and show_points:
        ax.legend(loc=legend_loc, frameon=True)

    # aspect correction for lon/lat plots
    if aspect_correction:
        if mean_lat is None:
            mean_lat = float(np.nanmean(coords[:, 0]))
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))

    # ----- panel 2: variance -----
    ax = axes[1]
    m1 = ax.pcolormesh(LON, LAT, var, shading="auto", cmap=plt.get_cmap(cmap_var), norm=var_norm)

    if show_points:
        ax.scatter(
            coords[:, 1], coords[:, 0],
            facecolors=var_point_facecolors,
            edgecolors=point_edgecolor,
            s=point_size,
            linewidths=var_point_linewidth,
            alpha=var_point_alpha,
            label="observations",
        )

    cb1 = plt.colorbar(m1, ax=ax, pad=0.01, fraction=0.05)
    cb1.set_label(cb_labels[1] + (" (log)" if log_var else ""))
    ax.set_title(titles[1])
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    if grid:
        ax.grid(True, **grid_kwargs)
    if legend and show_points:
        ax.legend(loc=legend_loc, frameon=True)

    if aspect_correction:
        if mean_lat is None:
            mean_lat = float(np.nanmean(coords[:, 0]))
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))

    return fig, axes
from pyproj import Geod
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from pathlib import Path

# get parameters from family of models: correlation or variogram
def theta_from_params(params, model_type, family):
    if family == 'variogram':
        if model_type in ('spherical','exponential','gaussian','cubic'):
            order = ('r','c0','b')
        elif model_type == 'powered_exponential':
            order = ('r','c0','beta','b')
        elif model_type == 'matern':
            order = ('r','c0','s','b')
        elif model_type in ('damped_cosine_angle','angular_dissimilarity'):
            order = ('c','c0','b')
        else:
            raise ValueError("Unknown model_type")
    elif family == 'correlation':
        if model_type in ('spherical','exponential','gaussian','cubic'):
            order = ('r','alpha')
        elif model_type == 'powered_exponential':
            order = ('r','beta','alpha')
        elif model_type == 'matern':
            order = ('r','nu','alpha')
        elif model_type in ('damped_cosine_angle','angular_dissimilarity'):
            order = ('c','alpha')
        else:
            raise ValueError("Unknown model_type")
    else:
        raise ValueError("family must be 'variogram' or 'correlation'")
    return [float(params[k]) for k in order]

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
def sample_points_from_geotiff(file_path, target_latlon, band=1):
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
        src_crs = CRS.from_epsg(4326)
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
"""
VarioCorreKrigE
---------------
Variograms, correlograms, and kriging (SK + SGS) with options for
distance families (euclidean, cartesian, geographic, angular) and custom kernels.
"""

# -----------------------------------------------------------------------------
# Version (written by setuptools-scm to _version.py at build/install time)
# -----------------------------------------------------------------------------
try:
    from ._version import version as __version__  # created by setuptools-scm
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

# -----------------------------------------------------------------------------
# Public API re-exports (safe, optional imports)
# -----------------------------------------------------------------------------
# Utilities (distances, helpers)
try:
    from .utils import (
        LatLongToPolar,
        theta_from_params,
        sample_points_from_geotiff
    )
except Exception:
    # keep import-time failures non-fatal; users can still import submodules directly
    pass

# Variogram fitting
try:
    from .variofit import (
        variofit,
        variofitmulti,
        VARIOGRAM_MODELS,
        make_init_and_bounds as make_vario_init_and_bounds,
        theta_from_params as theta_from_params_vario,
        pack_params as pack_params_vario,
    )
except Exception:
    pass

# Correlation fitting
try:
    from .correfit import (
        correfit,
        correfitmulti,
        CORRELATION_MODELS,
        make_init_and_bounds as make_corre_init_and_bounds,
        theta_from_params as theta_from_params_corre,
        pack_params as pack_params_corre,
    )
except Exception:
    pass

# Kriging & SGS
try:
    from .skrig import (
        simple_kriging,
        sgs_simple_kriging,
        simple_kriging_custom_corr,
        sgs_simple_kriging_custom_corr,
        build_covariance_nn_nt,              
        build_covariance_custom_correlation, 
        build_blocks_all,                     
    )
except Exception:
    pass

# -----------------------------------------------------------------------------
# What we export at the package top level
# -----------------------------------------------------------------------------
__all__ = [
    "__version__",
    # utils
    "LatLongToPolar", "theta_from_params", "sample_points_from_geotiff",
    # variograms
    "variofit", "variofitmulti", "VARIOGRAM_MODELS",
    "make_vario_init_and_bounds", "theta_from_params_vario", "pack_params_vario",
    # correlations
    "correfit", "correfitmulti", "CORRELATION_MODELS",
    "make_corre_init_and_bounds", "theta_from_params_corre", "pack_params_corre",
    # kriging
    "simple_kriging", "sgs_simple_kriging",
    "simple_kriging_custom_corr", "sgs_simple_kriging_custom_corr",
    "build_covariance_nn_nt", "build_covariance_custom_correlation",
    "kernel_MpEAS", "build_blocks_all",
]


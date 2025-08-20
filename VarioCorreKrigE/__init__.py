"""
VarioCorreKrigE
---------------
Variograms, correlograms, and kriging (SK + SGS) with options for
distance families (euclidean, cartesian, geographic, angular) and custom kernels.
"""

# ---------------------------------------------------------------------
# Version (written by setuptools-scm to _version.py at build/install time)
# ---------------------------------------------------------------------
try:
    from ._version import version as __version__  # created by setuptools-scm
except ImportError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]  # grow as we successfully import things

# ---------------------------------------------------------------------
# Utilities (distances, helpers)
# ---------------------------------------------------------------------
try:
    from .utils import (
        LatLongToPolar,
        theta_from_params,          # unified helper that knows both families
        sample_points_from_geotiff,
        compute_distance_weights,
    )
    # common spelling alias
    LatLonToPolar = LatLongToPolar
    __all__ += ["LatLongToPolar", "LatLonToPolar", "theta_from_params", "sample_points_from_geotiff", "compute_distance_weights"]
except ImportError:
    pass

# ---------------------------------------------------------------------
# Variogram fitting
# ---------------------------------------------------------------------
try:
    from .variofit import (
        variofit,
        variofitmulti,
        VARIOGRAM_MODELS,
        make_init_and_bounds as make_vario_init_and_bounds,
        theta_from_params as theta_from_params_vario,
        pack_params as pack_params_vario,
        spherical as spherical_vario,
        exponential as exponential_vario,
        gaussian as gaussian_vario,
        cubic as cubic_vario,
        powered_exponential as powered_exponential_vario,
        matern as matern_vario,
        damped_cosine_angle as damped_cosine_angle_vario,
        angular_dissimilarity as angular_dissimilarity_vario,
    )
    __all__ += [
        "variofit", "variofitmulti", "VARIOGRAM_MODELS",
        "make_vario_init_and_bounds", "theta_from_params_vario", "pack_params_vario",
        "spherical_vario", "exponential_vario", "gaussian_vario", "cubic_vario",
        "powered_exponential_vario", "matern_vario",
        "damped_cosine_angle_vario", "angular_dissimilarity_vario"
    ]
except ImportError:
    pass

# ---------------------------------------------------------------------
# Correlation fitting
# ---------------------------------------------------------------------
try:
    from .correfit import (
        correfit,
        correfitmulti,
        CORRELATION_MODELS,
        make_init_and_bounds as make_corre_init_and_bounds,
        theta_from_params as theta_from_params_corre,
        pack_params as pack_params_corre,
        spherical as spherical_corre,
        exponential as exponential_corre,
        gaussian as gaussian_corre,
        cubic as cubic_corre,
        powered_exponential as powered_exponential_corre,
        matern as matern_corre,
        damped_cosine_angle as damped_cosine_angle_corre,
        angular_dissimilarity as angular_dissimilarity_corre,
    )
    __all__ += [
        "correfit", "correfitmulti", "CORRELATION_MODELS",
        "make_corre_init_and_bounds", "theta_from_params_corre", "pack_params_corre",
        "spherical_corre", "exponential_corre", "gaussian_corre", "cubic_corre",
        "powered_exponential_corre", "matern_corre",
        "damped_cosine_angle_corre", "angular_dissimilarity_corre",
    ]
except ImportError:
    pass

# ---------------------------------------------------------------------
# Kriging & SGS
# ---------------------------------------------------------------------
try:
    from .skrig import (
        simple_kriging,
        sgs_simple_kriging,
        simple_kriging_custom_corr,
        sgs_simple_kriging_custom_corr,
        build_covariance_nn_nt,
        build_covariance_custom_correlation,
        build_blocks_all,
        kernel_MpEAS,  # <- include the custom kernel since you export it
    )
    __all__ += [
        "simple_kriging", "sgs_simple_kriging",
        "simple_kriging_custom_corr", "sgs_simple_kriging_custom_corr",
        "build_covariance_nn_nt", "build_covariance_custom_correlation",
        "build_blocks_all", "kernel_MpEAS",
    ]
except ImportError:
    pass

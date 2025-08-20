![Cover](https://github.com/AaronJR474/VarioCorreKrigE/blob/main/Examples/Data/output.png)
# VarioCorreKrigE: Variogram, Correlation and Krigging Estimation

This repository was created to perform the estimation of variogram and correlation models, which are essential for many engineering uses. In particular, variogram and correlation models form the backbone for Kriging, by which values can be estimated at unknown locations. There are several available packages (see References), whilst some are thorough and fairly robust, they are largely niche in most cases and not tailored to specific use cases without additional modifications. In this regard, this repository focuses mainly on engineering applications such as:

1) Spatial Correlation Models (SCMs) for parameter and residual estimation (e.g., Vs30, $`\delta W`$)
2) Kriging Estimates of values at unknown locations
3) Sequential Gaussian Simulation (SGS) for statistically robust estimates at unknown locations

Refer to [Theory](https://github.com/AaronJR474/VarioCorreKrigE/blob/main/theory.ipynb) for getting a quick overview of the process for estimating semivariance and correlation whereby models are developed. [Theory](https://github.com/AaronJR474/VarioCorreKrigE/blob/main/theory.ipynb) also provides an overview of the available kernels and some of the underlying assumptions required to conduct robust geostatistical analyses.

## Installation
### Option A
1. Clone the repository:
     ```bash
     git clone https://github.com/AaronJR474/VarioCorreKrigE.git
     cd VarioCorreKrigE
     ```

2. Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```
### Option B
  ```bash
  pip install git+ https://github.com/AaronJR474/VarioCorreKrigE.git
  ```

## Utilizing the package

The package is broken down into three main functions as follows:

1. `variofit.py`: Estimation of Semivariance and fitting of variogram models
     ```bash
       from VarioCorreKrigE.variofit import variofit
       from VarioCorreKrigE.variofit import VARIOGRAM_MODELS
     ```
2. `correfit.py`: Estimation of correlation coefficients and fitting of correlation models
     ```bash
       from VarioCorreKrigE.correfit import correfit
       from VarioCorreKrigE.correfit import CORRELATION_MODELS
     ```
3. `skrig.py`: Estimation of Simple Kriging (including Sequential Gaussian Simulation)
     ```bash
       from VarioCorreKrigE.skrig import simple_kriging, sgs_simple_kriging
       from VarioCorreKrigE.correfit import CORRELATION_MODELS
       from VarioCorreKrigE.variofit import VARIOGRAM_MODELS
       from VarioCorreKrigE.utils import theta_from_params, LatLongToPolar, sample_points_from_geotiff
     ```
In the folder [Examples](VarioCorreKrigE/Examples), there are detailed examples for each of the mentioned functions, including the ability to utilize custom correlation/variogram models not currently within the package. The examples also demonstrate the creation of custom covariance matrices and corresponding pairwise distances for a user-tailored Kriging analysis.

_Future versions will be extended to include_:
- _Ordinary and Universal Kriging_
- _Bayesian Markov Chain Monte Carlo estimation of Variogram/Correlation model parameters_

## References

### Learning Resources

> Myers, Donald. (1997). Multivariate geostatistics By Hans Wackernagel. Mathematical Geology. 29. 307-310. 10.1007/BF02769635.

> Pyrcz, M.J., 2024, Applied Geostatistics in Python: a Hands-on Guide with GeostatsPy [e-book]. Zenodo. doi:10.5281/zenodo.15169133

### Other Python Geostatistical Packages

> Mirko Mälicke, Egil Möller, Helge David Schneider, & Sebastian Müller. (2021, May 28). mmaelicke/scikit-gstat: A scipy flavoured geostatistical variogram analysis toolbox (Version v0.6.0). Zenodo. http://doi.org/10.5281/zenodo.4835779

> Müller, S., Schüler, L., Zech, A., and Heße, F.: GSTools v1.3: a toolbox for geostatistical modelling in Python, Geosci. Model Dev., 15, 3161–3182, https://doi.org/10.5194/gmd-15-3161-2022, 2022.

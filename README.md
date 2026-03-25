![Cover](https://github.com/AaronJR474/VarioCorreKrigE/blob/main/Examples/Data/output.png)
# VarioCorreKrigE: Variogram, Correlation and Kriging Estimation

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
     AND Optional for Bayesian Markov Chain Monte Carlo (MCMC) analysis:
     ```bash
     pip install numpyro >= 0.18.0
     ```
### Option B
  ```bash
  pip install git+https://github.com/AaronJR474/VarioCorreKrigE.git
  ```
  OR Optional for Bayesian Markov Chain Monte Carlo (MCMC) analysis:
  ```bash
  pip install "git+https://github.com/AaronJR474/VarioCorreKrigE.git#egg=VarioCorreKrigE[bayesmcmc]"
  ```

## Utilizing the package

The package is broken down into three main functions as follows:

1. `variofit.py`: Estimation of Semivariance and fitting of variogram models (including bootstrapping: "pair" or "point" resampling)
     ```bash
       from VarioCorreKrigE.variofit import variofit, variofitmulti
       from VarioCorreKrigE.variofit import VARIOGRAM_MODELS
     ```
     ![variofit_bootstrap](https://github.com/AaronJR474/VarioCorreKrigE/blob/main/Examples/Data/variofit_bootstrap.png)
     
2. `correfit.py`: Estimation of correlation coefficients and fitting of correlation models (including bootstrapping: "pair" or "point" resampling)
     ```bash
       from VarioCorreKrigE.correfit import correfit, correfitmulti
       from VarioCorreKrigE.correfit import CORRELATION_MODELS
     ```
     ![correfit_bootstrap](https://github.com/AaronJR474/VarioCorreKrigE/blob/main/Examples/Data/correfit_bootstrap.png)
   
4. `skrig.py`: Estimation of Simple Kriging with predefined or custom models (including Sequential Gaussian Simulation)
     ```bash
       from VarioCorreKrigE.skrig import simple_kriging, simple_kriging_custom_corr, sgs_simple_kriging, sgs_simple_kriging_custom_corr
       from VarioCorreKrigE.correfit import CORRELATION_MODELS
       from VarioCorreKrigE.variofit import VARIOGRAM_MODELS
     ```
     ![sgs](https://github.com/AaronJR474/VarioCorreKrigE/blob/main/Examples/Data/sgs.png)

5. `okrig.py`: Estimation of Ordinary Kriging with predefined or custom models
     ```bash
       from VarioCorreKrigE.skrig import ordinary_kriging, ordinary_kriging_custom_corr
       from VarioCorreKrigE.correfit import CORRELATION_MODELS
       from VarioCorreKrigE.variofit import VARIOGRAM_MODELS
     ```
     
6. `bayes_krig.py`: Empirical Bayesian Kriging (EBK) using either bootstrapped variograms/correlograms or specified posterior (see: [VarioCorreKrige_bayesmcmc_example.ipynb](VarioCorreKrigE/Examples/VarioCorreKrige_bayesmcmc_example.ipynb))
     ```bash
       from VarioCorreKrigE.bayes_krig import bayes_krig
     ```     
     ![EBK](https://github.com/AaronJR474/VarioCorreKrigE/blob/main/Examples/Data/EBK.png)
   
In the folder [Examples](VarioCorreKrigE/Examples), there are detailed examples for each of the mentioned functions, including the ability to utilize custom correlation/variogram models not currently within the package. The examples also demonstrate the creation of custom covariance matrices and corresponding pairwise distances for a user-tailored Kriging analysis.

_Future versions will be extended to include_:
- _Directional/Anisotropic variograms/correlograms_
- _Universal Kriging_
- _Bayesian Updating for Combining Conditional Distributions_

## References

### Learning Resources

> Myers, Donald. (1997). Multivariate geostatistics By Hans Wackernagel. Mathematical Geology. 29. 307-310. 10.1007/BF02769635.

> Pyrcz, M.J., 2024, Applied Geostatistics in Python: a Hands-on Guide with GeostatsPy [e-book]. Zenodo. doi:10.5281/zenodo.15169133

> Deutsch, J., 2015–2025, Geostatistics Lessons [online]. Available at: https://geostatisticslessons.com/.

### Other Python Geostatistical Packages

> Mirko Mälicke, Egil Möller, Helge David Schneider, & Sebastian Müller. (2021, May 28). mmaelicke/scikit-gstat: A scipy flavoured geostatistical variogram analysis toolbox (Version v0.6.0). Zenodo. http://doi.org/10.5281/zenodo.4835779

> Müller, S., Schüler, L., Zech, A., and Heße, F.: GSTools v1.3: a toolbox for geostatistical modelling in Python, Geosci. Model Dev., 15, 3161–3182, https://doi.org/10.5194/gmd-15-3161-2022, 2022.

## Disclaimer

This project is provided “as is” without any warranties or guarantees of any kind. Use of this code is at your own risk. The authors and contributors are not liable for any damages, data loss, or issues arising from the use, modification, or distribution of this software.

This repository may contain experimental, incomplete, or evolving features. Nothing here should be considered production‑ready unless explicitly stated.

You are responsible for reviewing and complying with all relevant licenses, data‑use agreements, and third‑party dependencies referenced in this project.

# VarioCorreKrigE: Variogram, Correlation and Krigging Estimation

This repository was created to perform the estimation of variogram and correlation models, which are essential for many engineering uses. In particular, variogram and correlation models form the backbone for Kriging, by which values can be estimated at unknown locations. Whilst there are several available packages (see References), whilst thorough and fairly robust, they are largely generic and not tailored to specific use cases without additional modifications. In this regard, this repository focuses mainly on engineering applications such as:

1) Spatial Correlation Models (SCMs) for parameter and residual estimation (e.g., Vs30, $`\delta W`$)
2) Kriging Estimates of values at unknown locations
3) Sequential Gaussian Simulation (SGS) for statistically robust estimates at unknown locations

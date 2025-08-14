# RobustSpline
R package RobustSpline. Thin-plate spline methods for functional data: (i) regression for functional predictors and scalar response, and (ii) robust (multivariate) functional location estimation. The random functions are supposed to be discretely observed, but their domains are allowed to be multi-dimensional. Both robust and non-robust fits are implemented.

To install the package *RobustSpline* in Windows you need to have R Tools installed on your computer, see 

https://cran.r-project.org/bin/windows/Rtools/

Then it is enough to run

```
#!R

library(devtools)
install_github("NagyStanislav/RobustSpline")

library(RobustSpline)
help(package="RobustSpline")
```

- Ioannis Kalogridis and Stanislav Nagy. (2025). Robust functional regression with discretely sampled predictors. _Under review._
- Ioannis Kalogridis and Stanislav Nagy. (2025). Robust multidimensional location estimation from discretely sampled functional data.

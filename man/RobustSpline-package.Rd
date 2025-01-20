\name{RobustSpline-package}
\alias{RobustSpline-package}
\alias{RobustSpline}
\docType{package}
\title{
\packageTitle{RobustSpline}
}
\description{
\packageDescription{RobustSpline}
}
\details{
Main functions in the package are:
\itemize{
\item{\link{IRLS}} for general Iteratively Reweighted Least Squares procedure in regression problems;
\item{\link{ridge}} for fast Ridge Regression with given penalty matrix; 
\item{\link{ts_reg}} for fitting robust thin-plate splines regression;
\item{\link{ts_ridge}} for fast fitting of non-robust thin-plate splines regression.
}
All major functions are implemented both in \code{R} and in \code{C++} for computational efficacy.
}
\author{
\packageAuthor{RobustSpline}

Maintainer: \packageMaintainer{RobustSpline}
}
\references{
Ioannis Kalogridis and Stanislav Nagy. (2023). Robust functional regression with discretely sampled predictors. \emph{Under review}.
}
\keyword{ thin-plate spline }
\keyword{ scalar-on-function regression }
\keyword{ robustness }

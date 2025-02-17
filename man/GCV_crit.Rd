% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/hello.R
\name{GCV_crit}
\alias{GCV_crit}
\title{Criteria used for Cross-Validation and for tuning parameter lambda}
\usage{
GCV_crit(resids, hats, custfun = NULL)
}
\arguments{
\item{resids}{A vector of residuals of length \code{n}.}

\item{hats}{A vector of hat values of length \code{n}.}

\item{custfun}{A custom function combining the residuals \code{resids}, the 
hat values \code{hats}, and possibly also the weights \code{weights}. The 
result of the function must be numeric.}

\item{weights}{A vector of weights given to the observations in the IRLS 
procedure. If not provided, all weights are taken to be equal to one.}
}
\value{
A named numerical vector of values. The length of the vector depends
on the input. The vector contains (some of) the values:
\itemize{
 \item{"AIC"}{ Akaike's information criterion given by 
 \code{mean(resids^2)+log(n)*mean(hats)}, where \code{n} is the length of
 both \code{resids} and \code{hats}.}
 \item{"GCV"}{ Leave-one-out cross-validation criterion given by
 \code{mean((resids^2)/((1-hats)^2))}.}
 \item{"GCV(tr)"}{ Modified leave-one-out cross-validation criterion 
 given by \code{mean((resids^2)/((1-mean(hats))^2))}.}
 \item{"BIC"}{ Bayes information criterion given by 
 \code{mean(resids^2)+2*mean(hats)}.}
 \item{"rGCV"}{ A robust version of \code{GCV} where mean is replaced
 by a robust M-estimator of scale of \code{resids/(1-hats)}, see 
 \link[robustbase]{scaleTau2} for details.}
 \item{"rGCV(tr)"}{ Modified version of a \code{rGCV} given by 
 a robust M-estimator of scale of \code{resids/(1-mean(hats))}.}
 \item{"custom"}{ The custom criterion given by function \code{custfun}. 
 Works only if \code{custfun} is part of the input.}
}
}
\description{
Several criteria commonly used for selection of the tuning parameter 
\code{lambda} in functions \link{IRLS} and \link{ridge}.
}
\details{
Function \code{custfun} can have either (i) two arguments 
corresponding to \code{resids} and \code{hats} if \code{weights} are not 
provided, or (ii) three arguments (\code{resids}, \code{hats}, 
\code{weights}) if \code{weights} is given. The output of the function must
be numeric.
}
\examples{
n = 50      # sample size
p = 10      # dimension of predictors
Z = matrix(rnorm(n*p),ncol=p) # design matrix
Y = Z[,1]   # response vector
lambda = 1  # tuning parameter for penalization
H = diag(p) # penalty matrix
type = "absolute" # absolute loss

# Run with the IRLS procedure
res = IRLS(Z, Y, lambda, H, type)
with(res,GCV_crit(resids,hat_values))
with(res,GCV_crit(resids,hat_values,custfun = function(r,h) 
    sum(r^2)))
    
# Run with the ridge function
res = ridge(Z, Y, lambda, H)
with(res,GCV_crit(resids,hat_values))
with(res,GCV_crit(resids,hat_values,custfun = function(r,h) 
    sum((r/(1-h))^2)))
}

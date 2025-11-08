#' Iteratively Reweighted Least Squares for robust functional regression
#'
#' Iteratively Reweighted Least Squares (IRLS) algorithm that is used to 
#' estimate a vector of regression parameters in a (possibly robust and 
#' penalized) linear regression model. Weights can be supplied as well.
#'
#' @param Z Data matrix of dimension \code{n}-times-\code{p}, where \code{n} is
#' the number of observations, \code{p} is the dimension.
#'
#' @param Y Vector of responses of length \code{n}.
#'
#' @param lambda Tuning parameter, a non-negative real number.
#'
#' @param H Penalty matrix of size \code{p}-times-\code{p} that
#' is used inside the quadratic term for penalizing estimated parameters.
#' 
#' @param type The type of the loss function used in the minimization problem.
#' Accepted are \code{type="absolute"} for the absolute loss \code{rho(t)=|t|}; 
#' \code{type="square"} for the square loss \code{rho(t)=t^2}; 
#' \code{type="Huber"} for the Huber loss \code{rho(t)=t^2/2} if 
#' \code{|t|<tuning} and \code{rho(t)=tuning*(|t|-tuning/2)} otherwise; and 
#' \code{type="logistic"} for the logistic loss 
#' \code{rho(t)=2*t + 4*log(1+exp(-t))-4*log(2)}.
#'
#' @param w Vector of length \code{n} of weights attached to the elements of 
#' \code{Y}. If \code{w=NULL} (default), a constant vector with values 
#' \code{1/n} is used.
#'
#' @param sc Scale parameter to be used in the IRLS. By default \code{sc=1}, 
#' that is no scaling is performed.
#'
#' @param resids.in Initialization of the vector of residuals used to launch 
#' the IRLS algorithm.
#' 
#' @param tuning A non-negative tuning constant for the absolute loss function 
#' (that is, \code{type="absolute"}). For \code{tuning = 0} the standard 
#' absolute loss \code{rho(t) = |t|} is used. For \code{tuning > 0}, the Huber 
#' loss is used, that is \code{rho(t)} is quadratic for \code{|t|<tuning} and 
#' linear for \code{|t|>=tuning}. The function is chosen so that \code{rho} 
#' is always continuously differentiable.
#' 
#' @param toler A small positive constant specifying the tolerance level for 
#' terminating the algorithm. The prcedure stops if the maximum absolute 
#' distance between the residuals in the previous iteration and the new 
#' residuals drops below \code{toler}.
#' 
#' @param imax Maximum number of allowed iterations of IRLS. 
#'
#' @param vrs Version of the algorhitm to be used. The program is prepared in
#' two versions: i) \code{vrs="C"} calls the \code{C++} version of the 
#' algorithm, programmed within the \code{RCppArmadillo} framework for
#' manipulating matrices. This is typically the fastest version. 
#' ii) \code{vrs="R"} calls the \code{R} version. The two versions may 
#' give slightly different results due to the differences in evaluating inverse
#' matrices. With \code{vrs="C"} one uses the function \code{solve} directly
#' from \code{Armadillo} library in \code{C++}; with \code{vrs="R"} the 
#' standard function \code{solve} from \code{R} package \code{base} is used 
#' with the option \code{tol = toler_solve}.
#'
#' @param toler_solve A small positive constant to be passed to function
#' \link[base]{solve} as argument \code{tol}. Used to handle numerically 
#' singular matrices whose inverses need to be approximated. By default set to
#' 1e-35.
#'
#' @details Especially for extremely small values of \code{lambda}, numerically
#' singular matrices must be inverted in the procedure. This may cause numerical
#' instabilites, and is the main cause for differences in results when using
#' \code{vrs="C"} and \code{vrs="R"}. In case when IRLS does not converge within
#' \code{imax} iterations, a warning is given.
#'
#' @return A list composed of:
#' \itemize{
#'  \item{"theta_hat"}{ A numerical matrix of size \code{p}-times-\code{1} of 
#'  estimated regression coefficients.}
#'  \item{"converged"}{ Indicator whether the IRLS procedure succefully 
#'  converged. Takes value 1 if IRLS converged, 0 otherwise.}
#'  \item{"ic"}{ Number of iterations needed to reach connvergence. If 
#'  \code{converged=0}, always \code{ic=imax}.}
#'  \item{"resids"}{ A numerical vecotor of length \code{n} containing the final
#'  set of residuals in the fit of \code{Y} on \code{Z}.}
#'  \item{"hat_values"}{ Diagonal terms of the (possibly penalized) hat matrix of
#'  the form \code{Z*solve(t(Z)*W*Z+n*lambda*H)*t(Z)*W}, where \code{W} 
#'  is the diagonal weight matrix in the final iteration of IRLS.}
#'  \item{"last_check"}{ The final maximum absolute difference between 
#'  \code{resids} and the residuals from the previous iteration. We have 
#'  \code{resids < toler} if and only if the IRLS converged (that is, 
#'  \code{converged=1}).}
#'  \item{"weights"}{ The vector of weights given to the observations in the 
#'  final iteration of IRLS. For squared loss (\code{type="square"}) this gives 
#'  a vector whose all elements are 2.}
#'  \item{"fitted"}{ Fitted values in the model. A vector of length \code{n} 
#'  correponding to the fits of \code{Y}.}
#' }
#'
#' @references
#' Ioannis Kalogridis and Stanislav Nagy. (2023). Robust functional regression 
#' with discretely sampled predictors. 
#' \emph{Under review}.
#'
#' Peter. J. Huber. (1981). Robust Statistics, \emph{New York: John Wiley.}
#'
#' @seealso \link{ridge} for a faster (non-robust) version of this
#' function with \code{type="square"}.
#'
#' @examples
#' n = 50      # sample size
#' p = 10      # dimension of predictors
#' Z = matrix(rnorm(n*p),ncol=p) # design matrix
#' Y = Z[,1]   # response vector
#' lambda = 1  # tuning parameter for penalization
#' H = diag(p) # penalty matrix
#' type = "absolute" # absolute loss
#' 
#' # Run the two versions of the IRLS procedure
#' res_C = IRLS(Z, Y, lambda, H, type, vrs="C")
#' res_R = IRLS(Z, Y, lambda, H, type, vrs="R")
#' # Check whether both versions converged after the same number of iterations
#' res_C$ic
#' res_R$ic
#' # Check the maximum absolute difference between the results
#' max(abs(res_C$theta_hat-res_R$theta_hat))
#' # Visualise the difference between the results
#' plot(res_C$theta_hat ~ res_R$theta_hat)

IRLS = function(Z, Y, lambda, H, type, w=NULL, sc = 1, 
                resids.in = rep(1,length(Y)), 
                tuning=NULL, toler=1e-7, imax=1000, vrs="C", 
                toler_solve=1e-35){
  
  IRLS_R <- function(Z, Y, lambda, H, type, w=NULL, sc, resids.in, 
                      tuning, toler, imax, toler_solve){
    n = length(Y)
    if(is.null(w)) w = rep(1/n,n)
    if(length(w)!=n) stop("Weights w must have the same length as Y.")
    ic = 0
    istop = 0
    clim = c(.5, 1, .5*tuning, 1)[type] # tail behavior constant for psiw
    while(istop == 0 & ic < imax){
      ic = ic + 1
      #
      Wdiag = c(w*psiw(resids.in/sc,type,tuning)/sc^2)
      naind = (is.na(Wdiag)) & (abs(resids.in)<tuning)
      Wdiag[naind] = w[naind]*1 # division 0/0
      naind = (is.na(Wdiag))
      Wdiag[naind] = w[naind]*clim/abs(resids.in*sc) 
      # if resids are too small exp(-resids) ~ Inf but in the limit always
      # psiw(t) ~ clim/abs(t)
      Z.s = scale(t(Z), center = FALSE, scale = 1/Wdiag);
      Z1 = Z.s%*%Z + lambda*H
      theta_new = solve(Z1, Z.s%*%Y, tol=toler_solve)
      resids1 <- c(Y - Z%*%theta_new)
      check = max(abs(resids1-resids.in))/sc 
      if(check < toler){istop=1}
      resids.in <- resids1
    }
    if(istop==0) warning(
      paste0("log(lambda) ",sprintf("%.3f",log(lambda)),
             ": Estimator did not converge."))
    hat.values = diag(hatm<-Z%*%solve(Z1, Z.s, tol=toler_solve))
    #
    return(list(theta_hat = theta_new,
                converged=istop,
                ic=ic,
                resids = resids.in, 
                hat_values = hat.values,
                last_check = check,
                weights = 2*Wdiag,
                fitted = hatm%*%Y))
  }
  
  n = length(Y)
  if(is.null(w)) w = rep(1/n,n)
  vrs = match.arg(vrs,c("C","R"))
  type = match.arg(type,c("square","absolute","Huber","logistic"))
  type = switch(type, absolute = 1, square = 2, Huber = 3, logistic = 4)
  if(type==3 & is.null(tuning)){
    tuning = 1.345
    # warning("Huber loss, setting constant to default 1.345.")
  }
  if(type==1 & is.null(tuning)){
    tuning = 1/100
    # warning("absolute loss, setting tuning to default 1/100.")    
  }
  if(is.null(tuning)) tuning = 1/100 # tuning for logistic regression
  #
  if(nrow(Z)!=length(Y)) 
    stop("Number of rows of Z must equal the lenght of Y.")
  if(nrow(H)!=ncol(Z))
    stop("H must be a square matrix with the same number of columns as Z.")
  if(ncol(H)!=ncol(Z))
    stop("H must be a square matrix with the same number of columns as Z.")
  if(tuning<0) stop("tuning must be a non-negative number.")
  if(lambda<0) stop("lambda must be a non-negative number.")
  if(sc<=0) stop("Scale estimator must be strictly positive.")
  if(vrs=="C"){
    rs = tryCatch(
      error = function(cnd){
        warning(paste0("Solve in C++ crashed, switching to R version, ",cnd))
        IRLS_R(Z, Y, lambda, H, type, w, sc, resids.in, 
               tuning, toler, imax, toler_solve)
      }, {
        IRLSC(Z, Y, lambda, H, type, w, sc, resids.in, 
              tuning, toler, imax)
      })
    return(rs)
  }
  
  if(vrs=="R") return(IRLS_R(Z, Y, lambda, H, type, w, sc, resids.in, 
                             tuning, toler, imax, toler_solve))
}

#' Fast Ridge Regression with given penalty matrix
#'
#' A (weighted) ridge regression estimator with a specified penalty matrix
#' in a linear regression model. The solution corresponds to the 
#' result of function \link{IRLS} with \code{type="square"}.
#'
#' @param Z Data matrix of dimension \code{n}-times-\code{p}, where \code{n} is
#' the number of observations, \code{p} is the dimension.
#'
#' @param Y Vector of responses of length \code{n}.
#'
#' @param lambda Tuning parameter, a non-negative real number.
#'
#' @param H Penalty matrix of size \code{p}-times-\code{p} that
#' is used inside the quadratic term for penalizing estimated parameters.
#' 
#' @param w Vector of length \code{n} of weights attached to the elements of 
#' \code{Y}. If \code{w=NULL} (default), a constant vector with values 
#' \code{1/n} is used.
#' 
#' @param vrs Version of the algorhitm to be used. The program is prepared in
#' two versions: i) \code{vrs="C"} calls the \code{C++} version of the 
#' algorithm, programmed within the \code{RCppArmadillo} framework for
#' manipulating matrices. This is typically the fastest version. 
#' ii) \code{vrs="R"} calls the \code{R} version. The two versions may 
#' give slightly different results due to the differences in evaluating inverse
#' matrices. With \code{vrs="C"} one uses the function \code{solve} directly
#' from \code{Armadillo} library in \code{C++}; with \code{vrs="R"} the 
#' standard function \code{solve} from \code{R} package \code{base} is used 
#' with the option \code{tol = toler_solve}.
#'
#' @param toler_solve A small positive constant to be passed to function
#' \link[base]{solve} as argument \code{tol}. Used to handle numerically 
#' singular matrices whose inverses need to be approximated. By default set to
#' 1e-35.
#'
#' @details Especially for extremely small values of \code{lambda}, numerically
#' singular matrices must be inverted in the procedure. This may cause numerical
#' instabilites, and is the main cause for differences in results when using
#' \code{vrs="C"} and \code{vrs="R"}. This function is equivalent with 
#' \link{IRLS} when used with the square loss \code{type="square"}, but faster
#' and more stable as it does not perform the iterative algorithm. Instead, it 
#' computes the estimator directly.
#'
#' @return A list composed of:
#' \itemize{
#'  \item{"theta_hat"}{ A numerical matrix of size \code{p}-times-\code{1} of 
#'  estimated regression coefficients.}
#'  \item{"resids"}{ A numerical vecotor of length \code{n} containing the final
#'  set of residuals in the fit of \code{Y} on \code{Z}.}
#'  \item{"hat_values"}{ Diagonal terms of the (penalized) hat matrix of
#'  the form \code{Z*solve(t(Z)*Z + n*lambda*H)*t(Z)}.}
#'  \item{"fitted"}{ Fitted values in the model. A vector of length \code{n} 
#'  correponding to the fits of \code{Y}.}
#' }
#' 
#' @seealso \link{IRLS} for a robust version of this
#' function.
#'
#'
#' @examples
#' n = 50      # sample size
#' p = 10      # dimension of predictors
#' Z = matrix(rnorm(n*p),ncol=p) # design matrix
#' Y = Z[,1]   # response vector
#' lambda = 1  # tuning parameter for penalization
#' H = diag(p) # penalty matrix
#' 
#' res_C = ridge(Z, Y, lambda, H, vrs="C")
#' res_R = ridge(Z, Y, lambda, H, vrs="R")
#' # Check the maximum absolute difference between the results
#' max(abs(res_C$theta_hat-res_R$theta_hat))
#' # Visualise the difference between the results
#' plot(res_C$theta_hat ~ res_R$theta_hat)
#' 
#' # Compare the output with function IRLS
#' res_IRLS = IRLS(Z, Y, lambda, H, type="square")
#' max(abs(res_C$theta_hat-res_IRLS$theta_hat))

ridge = function(Z, Y, lambda, H, w=NULL, vrs="C", toler_solve=1e-35){
  
  n = length(Y)
  if(is.null(w)) w = rep(1/n,n)
  vrs = match.arg(vrs,c("C","R"))
  if(nrow(Z)!=length(Y)) 
    stop("Number of rows of Z must equal the lenght of Y.")
  if(nrow(H)!=ncol(Z))
    stop("H must be a square matrix with the same number of columns as Z.")
  if(ncol(H)!=ncol(Z))
    stop("H must be a square matrix with the same number of columns as Z.")
  if(lambda<0) stop("lambda must be a non-negative number.")
  # if(sc<=0) stop("Scale estimator must be strictly positive.")
  if(vrs=="C") return(ridgeC(Z, Y, lambda, H, w))
  
  ridge_R <- function(Z, Y, lambda, H, w=NULL, toler_solve){
    n = length(Y)
    th = solve(t(Z)%*%diag(w)%*%Z+lambda*H,t(Z)%*%diag(w)%*%Y,
               tol=toler_solve)
    hat = Z%*%solve(t(Z)%*%diag(w)%*%Z+lambda*H,t(Z)%*%diag(w),tol=toler_solve)
    resid = Y - Z%*%th
    return(list(theta_hat = th,
                resids = resid, 
                hat_values = diag(hat)))
  }
  
  if(vrs=="R") return(ridge_R(Z, Y, lambda, H, w, toler_solve))
}

#' Weight function for the IRLS algorithm
#'
#' Returns a vector of weights given by \code{psi(t)/(2*t)}, where 
#' \code{psi} is the derivative of the loss function \code{rho}. 
#'
#' @param t Vector of input values of length \code{n}.
#'
#' @param type Integer code for the type of loss function. Accepted are
#' \code{type=1} for the absolute loss \code{rho(t)=|t|}; \code{type=2} for
#' the square loss \code{rho(t)=t^2}; \code{type=3} for the Huber loss
#' \code{rho(t)=t^2/2} if \code{|t|<tuning} and 
#' \code{rho(t)=tuning*(|t|-tuning/2)} otherwise; and \code{type=4} for the
#' logistic loss \code{rho(t)=2*t + 4*log(1+exp(-t))-4*log(2)}.
#'
#' @param tuning Tuning parameter, a non-negative real number. For the absolute
#' this should be a small number that 'smooths' out the numerical effects of
#' the kink of \code{rho} near the origin (by default \code{tuning = 1/100}. 
#' For the Huber loss \code{tuning} is the constant to be used in the function
#' (by default \code{tuning = 1.345}. For \code{type=2} or \code{type=4} this 
#' constant is not used. 
#'
#'
#' @return A numerical vector of values.
#'
#' @examples
#' curve(psiw(x,type=1,tuning=0.1),-5,5) # absolute loss with tuning
#' curve(psiw(x,type=2),-5,5) # square loss
#' curve(psiw(x,type=3),-5,5) # Huber loss
#' curve(psiw(x,type=4),-5,5) # logistic loss

psiw = function(t,type,tuning=NULL){
  # Type is now only the code 1-4
  if(type==3 & is.null(tuning)){
    tuning = 1.345
    # warning("Huber loss, setting constant to default 1.345.")
  }
  if(type==1 & is.null(tuning)){
    tuning = 1/100
    # warning("absolute loss, setting tuning to default 1/100.")    
  }
  if(is.null(tuning)) tuning = 0
  return(psiwC(t,type,tuning))
}

#' Loss functions
#'
#' Several typically used loss functions \code{rho}. 
#'
#' @param t Vector of input values of length \code{n}.
#'
#' @param type Integer code for the type of loss function. Accepted are
#' \code{type="absolute"} for the absolute loss \code{rho(t)=|t|}; 
#' \code{type="square"} for the square loss \code{rho(t)=t^2}; 
#' \code{type="Huber"} for the Huber loss \code{rho(t)=t^2/2} if 
#' \code{|t|<Hk} and \code{rho(t)=Hk*(|t|-Hk/2)} otherwise; and 
#' \code{type="logistic"} for the logistic loss 
#' \code{rho(t)=2*t + 4*log(1+exp(-t))-4*log(2)}.
#'
#' @param Hk Tuning parameter, a non-negative real number, affects only
#' the Huber loss (by default \code{Hk = 1.345}.
#'
#' @return A numerical vector of values.
#'
#' @examples
#' curve(rho(x,type="absolute"),-5,5) # absolute loss with tuning
#' curve(rho(x,type="square"),-5,5) # square loss
#' curve(rho(x,type="Huber"),-5,5) # Huber loss
#' curve(rho(x,type="logistic"),-5,5) # logistic loss

rho = function(t,type,Hk=1.345){          # loss function
  type = match.arg(type,c("square","absolute","Huber","logistic"))
  if(Hk<0) stop("Hk must be a non-negative number.")
  if(type=="absolute") return(abs(t))
  if(type=="square") return(abs(t)^2)
  if(type=="Huber"){
    return((abs(t)<=Hk)*(t^2/2) + (abs(t)>Hk)*(Hk*(abs(t)-Hk/2)))
  }
  if(type=="logistic") return(2*t + 4*log(1+exp(-t))-4*log(2))
}

#' Generate a dataset for scalar-on-function linear regression
#'
#' Generates the data according to the model 
#' \code{Y = alpha0 + int beta0(t)*X(t) dt + epsilon}, where \code{X} are the 
#' regressor functions, \code{alpha0} is a real parameter, \code{beta0} is a 
#' functional parameter, and \code{epsilon} are independent error terms. 
#' The domain of \code{X} and \code{beta0} can be one-dimensional, or also
#' higher-dimensional. The functional data \code{X} are assumed to be observed
#' only in an incomplete grid of \code{p} points in its domain.
#'
#' @param alpha0 A real value of the intercept parameter.
#'
#' @param beta0 A function parameter. A real function whose arguments are
#' \code{d}-dimensional vectors.
#'
#' @param n Sample size.
#'
#' @param d Dimension of the domain of \code{X} and \code{beta0}.
#' 
#' @param p The size of the discretization grid where the functions \code{X} are
#' observed. The observation points are sampled randomly from the full grid of
#' all \code{p1^d} points in the domain.
#' 
#' @param bfX A list of basis functions for \code{X}. Each element of \code{bfX}
#' contains the list of values of a basis function when evaluated in a complete
#' grid of observations, that is \code{p1^d} real values.
#' 
#' @param bcX A matrix of basis coefficients that are combined with the basis
#' functions from \code{bfX} to get the regressor functions \code{X}. The matrix
#' must be of dimensions \code{n}-times-\code{K}, where \code{n} is the sample
#' size, and \code{K} is the length of the list \code{bfX}.
#' 
#' @param sd.noiseEps Standard deviation for the noise component \code{eps}. 
#' Noise is generated to be centered Gaussian, independent of \code{X}.
#' 
#' @param obs_only Indictar of whether to generate the response \code{Y}
#' the functions \code{X} and \code{beta0} should be used in the complete 
#' grid of size \code{p1^d} (\code{obs_only=FALSE}), or only in the grid 
#' of size \code{p} where \code{X} is observed (\code{obs_only=FALSE}). By
#' default set to \code{FALSE}.
#' 
#' @param p1 The complete size of the discretization grid where the 
#' (practically unobservable) complete version of the function \code{X} is 
#' known. \code{p1} corresponds to the complete grid in dimension 1; if 
#' \code{d>1}, the same grid is replicated also in the other dimensions and the
#' resulting size of the domain is \code{p1^d}. By default, 
#' \code{p1=101}.
#' 
#' @param sd.noiseX Standard deviation of an additional noise component that is
#' added to the regressors \code{X}. The final regressor functions are obtained
#' by taking a linear combination of basis functions from \code{bfX} weighted
#' by the coefficients from \code{bcX}, and then adding indepenent centered 
#' normal noise to each of the \code{p1^d} of its values with standard
#' deviation \code{sd.noiseX}. By default, \code{sd.noiseX = 0}.
#' 
#' @return A list composed of several elements. Some of these elements are
#' assumed to be observed; some are given for post-estimation diagnostic 
#' purposes. The elements that are assumed to be observed are:
#' \itemize{
#'  \item{"tgrid"}{ A complete grid of observation points in one dimension, 
#'  replicated in each dimension if \code{d>1}. An equidistant grid of 
#'  \code{p1} points in the interval [0,1].}
#'  \item{"X"}{ Matrix of observed values of \code{X} of size 
#'  \code{n}-times-\code{p}, one row per observation, columns corresponding to the 
#'  positions in the rows of \code{tobs}.}
#'  \item{"Y"}{ A vector of observed responses, size \code{n}.}
#'  \item{"tobs"}{ Domain locations for the observed points of \code{X}. Matrix
#'  of size \code{p}-times-\code{d}, one row per domain point.}
#' }
#' Additional data for diagnostic purposes that is not assumed to be observed:
#' \itemize{
#'  \item{"Xfull"}{ A \code{d+1}-dimensional array of completely observed 
#'  values of functions \code{X} of size 
#'  \code{n}-times-\code{p1}-...-times-\code{p1}. The first dimension is for 
#'  observations, additional ones for dimensions of the
#'  domain, i.e. \code{Xfull[2,4,5]} stands for the function value of function 2
#'  at the domain points \code{(tgrid[4],tgrid[5])} for \code{d=2}.}
#'  \item{"betafull"}{ A \code{d}-dimensional array of the values of the 
#'  regression function \code{beta0} of size \code{p1}-times-...-times-\code{p1}.
#'  Interpretation of values as for the functions of \code{Xfull}.}
#'  \item{"tfull"}{ A flattened matrix of the full grid of all observation 
#'  points. A matrix of size \code{p1^d}-times-\code{d}, each row 
#'  corersponding to the domain vector of a single point where \code{Xfull} and
#'  \code{betafull} are evaluated.}
#'  \item{"betafull0"}{ A vector of all evaluated values of the function 
#'  \code{beta0} in the complete grid of \code{p1^d} observation points. A 
#'  numerical vector of length \code{p1^d}, each value corresponding to a 
#'  row of \code{tfull}. Idential to a flattened version of \code{betafull}.}
#'  \item{"truemean"}{ A vector of length \code{n} of the true conditional mean 
#'  values of Y given the i-th regressor from \code{X}, without the intercept
#'  \code{alpha0}.}
#'  \item{"eps"}{ Vector of erros used in the regression. Naturally, 
#'  \code{Y = truemean + eps + alpha0}.}
#'  \item{"d"}{ Dimension of the domain of regressors.}
#' }
#'
#' @examples
#' d = 1       # dimension of domain
#' n = 50      # sample size
#' p = 10      # number of observed points in domain
#' p1 = 101 # size of the complete grid
#' alpha0 = 3  # intercept
#' beta0 = function(t) t^2 # regression coefficient function
#' sd.noiseEps = 0.1 # noise standard deviation
#' tgrid = seq(0,1,length=p1) # full grid of observation points
#' 
#' # Generating the basis functions for X
#' basis_fun = function(t,k) return(sin(2*pi*k*t)+t^k)
#' 
#' K = 5 # number of basis functions used in the expansion
#' bfX = list()
#' for(k in 1:K) bfX[[k]] = basis_fun(tgrid,k)
#' bcX = matrix(rnorm(n*K,mean=3,sd=5),ncol=K)  # basis coefficients
#' 
#' gen = generate(alpha0, beta0, n, d, p, bfX, bcX, sd.noiseEps)
#' 
#' # plot the first regressor function
#' plot(gen$Xfull[1,]~tgrid,type="l",main="Predictor X",lwd=2)
#' 
#' # plot the true mean values against the responses
#' plot(gen$truemean+alpha0, gen$Y)
#' abline(a=0,b=1)
#' 
#' # Check that the response is modelled as alpha0 + truemean + eps
#' all.equal(gen$Y,alpha0 + gen$truemean + gen$eps)
#' 
#' # Check that the truemean is an approximation to the integral of X and beta0
#' all.equal(apply(gen$Xfull,1,function(x) mean(x*gen$betafull)),gen$truemean)
#' 
#' # Check that betafull0 is flattened version of betafull
#' all.equal(c(gen$betafull), gen$betafull0)

generate = function(alpha0, beta0, n, d, p,  
         bfX, bcX, sd.noiseEps, obs_only = FALSE, p1 = 101, sd.noiseX=0){
  
  pfull = p1^d  # complete grid of observations
  tgrid = seq(0,1,length=p1)
  if(pfull<p1) stop("Less points in full discretization that p")
  #
  K = length(bfX) # number of basis functions in the expansion of X
  
  # generate full covariates X
  if(d==1){
    ifull = 1:p1
    tfull = matrix(tgrid,ncol=1)
    Xfull = array(rnorm(pfull*n,sd=sd.noiseX),
                  dim=c(n,p1))
    for(i in 1:n) for(k in 1:K) Xfull[i,] = 
      Xfull[i,] + bfX[[k]]*bcX[i,k]
    # predictors are iid normally distributed noise plus signal
    # plot(Xfull[1,]~tgrid,type="l",main="Predictor X",lwd=2)
  }
  if(d==2){
    ifull = expand.grid(1:p1,1:p1)
    tfull = expand.grid(tgrid,tgrid)
    Xfull = array(rnorm(pfull*n,sd=sd.noiseX),
                  dim=c(n,p1,p1)) # predictors
    for(i in 1:n) for(k in 1:K) Xfull[i,,] = 
      Xfull[i,,] + bfX[[k]]*bcX[i,k]
    # image(Xfull[1,,],main="Predictor X")
  }
  if(d>2){
    if(K>0) warning("For d>2 the basis in X is ignored, X is just noise.")
    ifull = expand.grid(replicate(d,1:p1,simplify=FALSE))
    tfull = expand.grid(replicate(d,tgrid,simplify=FALSE))
    Xfull = array(rnorm(pfull*n,sd=sd.noiseX),
                  dim=c(n,rep(p1,d))) # predictors
  }
  
  if(d==1){
    betafull = Vectorize(beta0)(tgrid)
    betafull0 = betafull
  }
  if(d==2){
    betafull0 = apply(tfull,1,beta0)
    betafull = matrix(betafull0,nrow=p1)
  }
  if(d>2){
    betafull0 = apply(tfull,1,beta0)
    betafull = array(betafull0,dim=rep(p1,d))
  }
  
  ###
  # Generating discretely observed data
  ###
  tind = sort(sample(1:pfull,p))
  tobs = tfull[tind,,drop=FALSE] # observed points
  
  # Observed values of X, only at the p points in tobs
  if(d==1){
    X = Xfull[,tind]
    # beta = betafull[tind]
  }
  if(d==2){
    Xff = matrix(Xfull,nrow=n)  # flattened X matrix
    X = Xff[,tind]              # observed points of X
    # beta = betafull0[tind]      # beta0 at observed points
  }
  if(d>2){
    Xff = matrix(Xfull,nrow=n)
    X = Xff[,tind]
    # beta = betafull0[tind]
  }
  
  # plot(apply(Xfull,1,function(x) mean(x*betafull))~
  #        apply(X,1,function(x) mean(x*beta)),
  #      xlab="Approximated integrals from discrete functions",
  #      ylab="Exact integrals")
  # abline(a=0,b=1)
  
  ###
  # model
  ###
  eps = rnorm(n,sd=sd.noiseEps)          # random errors
  
  if(obs_only){
  truemean = apply(X,1,function(x) mean(x*betafull0))
  } else {
  truemean = apply(Xfull,1,function(x) mean(x*betafull))
  }
  Y = alpha0 + truemean + eps

  return(list(tgrid=tgrid, X=X,Y=Y,tobs=tobs,
              Xfull=Xfull,betafull=betafull,betafull0=betafull0,
              tfull=tfull,
              truemean=truemean,eps=eps,d=d))
}

#' Radial functions for thin-plate splines
#'
#' The function \code{eta_{m,d}} involved in the definition of a 
#' \code{d}-dimensional thin-plate spline of order \code{m}.
#'
#' @param x Vector of input values of length \code{n}.
#' 
#' @param d Dimension of the domain, positive integer.
#' 
#' @param m Order of the spline, positive integer.
#'
#' @return A numerical vector of values.
#'
#' @examples
#' curve(eta(x,d=1,m=3),-5,5) 

eta = function(x,d,m){
  if(d%%2==0){
    x[x==0] = 1 # log(t) will be changed from -Infty to 0
    return((-1)^{m+1+d/2}/(2^{2*m-1}*pi^{d/2}*
                             factorial(m-1)*factorial(m-d/2))*
             x^{2*m-d}*log(x))
  }
  if(d%%2==1) return(gamma(d/2-m)/(2^{2*m}*pi^{d/2}*factorial(m-1))*
                       x^{2*m-d})
}

#' Pre-processing raw data for thin-plate multivariate location estimation
#'
#' Given the discretely observed functional data as an input, pre-processes the 
#' dataset to a form suitable for fitting thin-plate spline functional location
#' estimation.
#'
#' @param Y Matrix of observed values of functional data \code{Y} of size 
#'  \code{p}-times-\code{n}, one column per functional observation, 
#'  rows corresponding to the positions in the rows of \code{tobs}. If matrix
#'  \code{Y} does not contain any missing values \code{NA}, it is also 
#'  possible to be supplied as a long vector of length \code{p*n}, stacked by
#'  columns of the matrix \code{Y}. If \code{Y} contains missing values 
#'  \code{NA}, it must be a matrix.
#'  
#' @param tobs Domain locations for the observed points of \code{X}. Matrix
#'  of size \code{p}-times-\code{d}, one row per domain point.
#' 
#' @param r Order of the thin-plate spline, positive integer.
#' 
#' @return A list of values:
#' \itemize{
#'  \item{"Y"}{ A vector of length \code{m} of all the reponses that are not 
#'  missing. If \code{Y} was a matrix without missing values, this is simply 
#'  \code{c(Y)}; if \code{Y} was a vector, this is directly \code{Y}. In case
#'  \code{Y} contained missing values, this is the elements of \code{c(Y)} that
#'  are not \code{NA}.}
#'  \item{"Z"}{ A matrix of size \code{m}-times-\code{p} for fitting the 
#'  penalized robust regression model for thin-plate coefficients.}
#'  \item{"H"}{ A penalty matrix of size \code{p}-times-\code{p} used for
#'  fitting the location estimation model with thin-plate splines.}
#'  \item{"Q"}{ A matrix containing the null space of the rows of Phi of size
#'  \code{p}-times-\code{p-M}, where \code{M} is the number of monomials
#'  used for the construction of the thin-plate spline. This matrix is used to
#'  pass from parameters gamma to parametrization chi.}
#'  \item{"Omega"}{ A matrix of size \code{p}-times-\code{p} containing the 
#'  \link{eta}-transformed matrix of inter-point distances in \code{tobs}.}
#'  \item{"Phi"}{ A matrix of size \code{p}-times-\code{M} corresponding to the
#'  monomial part of the thin-plate spline.}
#'  \item{"w"}{ A vector of weights of length \code{m}. Each weight corresponds
#'  to \code{1/(n*m[i])}, where \code{m[i]} is the number of observations for 
#'  the \code{i}-th function.}
#'  \item{"degs"}{ A matrix of size \code{M}-times-\code{d} with degrees of the
#'  monomials in each dimension per row. Used for the construction of 
#'  \code{Phi}.}
#'  \item{"tobs"}{ The same as the input parameter \code{tobs}, for later use.}
#'  \item{"M"}{ Number of all monomials used, is equal to 
#'  \code{choose(m+d-1,d)}.}
#'  \item{"r"}{ Order of the spline, positive integer.}
#'  \item{"p"}{ Number of observed time points, positive integer.}
#'  \item{"d"}{ Dimension of domain, positive integer.}
#'  \item{"n"}{ Sample size, positive integer.}
#' }
#'
#' @examples
#' d = 1  # dimension of the domain
#' m = 50 # number of observation times per function
#' tobs = matrix(runif(m*d), ncol=d) # matrix of observation times
#' n = 20 # sample size
#' truemeanf = function(x) 10+15*x[1]^2 # true mean function
#' truemean = apply(tobs,1,truemeanf) # discretized values of the true mean
#' Y = replicate(n, truemean + rnorm(m)) # matrix of discrete functiona data, size p*n
#' tsp = ts_preprocess_location(Y, tobs, 2) # preprocessing matrices 
#' 
#' lambda0 = 1e-5 # regularization parameter
#' res_IRLS = IRLS(Z = tsp$Z, Y = tsp$Y, lambda = lambda0, H = tsp$H, type = "square")
#' res_ridge = ridge(Z = tsp$Z, Y = tsp$Y, lambda = lambda0, H = tsp$H)
#' # resulting estimates of the parameters theta, using IRLS and ridge
#' 
#' # testing that ridge and IRLS (square) give the same results
#' all.equal(res_IRLS$theta_hat, res_ridge$theta_hat)
#' 
#' res = res_ridge
#' resf = transform_theta_location(res$theta_hat, tsp)
#' 
#' plot(rep(tobs,n), c(Y), cex=.2, pch=16)
#' points(tobs, truemean, pch=16, col="orange")
#' points(tobs, resf$beta_hat, col=2, pch=16)

ts_preprocess_location = function(Y, tobs, r){
  
  # Y Observed values of functional data of size \code{p}-times-\code{n}
  # tobs Domain locations for the observed points of size \code{p}-times-\code{d}
  # r Order of the thin-plate spline, positive integer.
  
  p = nrow(tobs)    # number of distinct observation points
  d = ncol(tobs)    # dimension of the domain
  
  ### preparing the vector of effective values of the response
  if(sum(is.na(Y))==0){ 
    # if the design of observation is full, also vector Y is accepted
    if(is.matrix(Y)){
      if(nrow(Y)!=p) 
        stop("If Y is matrix, its number of rows must equal 
             the number of columns of tobs.")
      n = ncol(Y)
      Y = c(Y) # stack into a large vector of length p*n
    } else {
      n = length(Y)/p
      if(n%%1!=0) stop("The length of Y is not divisible by the number
                       of columns of tobs.")
    }
    w = rep(p,n) # vector of numbers of observations per function (length n)
    m = p*n      # total number of available observations
    inds = rep(1:p,n) # list of indices of active (observed) points, concatenated
  } else { # if the design of observation is not full, 
    if(!is.matrix(Y)) 
      stop("If Y contains NA's, only matrix form of Y is accepted.")
    
    if(nrow(Y)!=p) 
      stop("If Y is matrix, its number of rows must equal 
             the number of columns of tobs.")
    
    inds = apply(Y,2,function(x) which(!is.na(x))) 
    # list of indices of active (observed) points, one list per function
    inds = unlist(inds) # concatenated list of indices, length m
    w = apply(Y, 2, function(x) sum(!is.na(x))) 
    # vector of numbers of observations per function (length n)
    Y = Y[,w>0,drop=FALSE]
    # deleting functions that are never observed
    n = ncol(Y)
    m = sum(w) # total number of available observations
    Y = c(Y)   # stacking Y by columns
    Y = Y[!is.na(Y)] # vector of length m of clean observed values
  }
  
  M = choose(r+d-1,d)
  if(p-M<=0) stop(paste("p must be larger than",M))
  if(2*r<=d) stop(paste("r must be larger than",ceiling(d/2)))
  
  # Monomials phi of order <r
  allcom = expand.grid(replicate(d, 0:(r-1), simplify=FALSE))
  degs = as.matrix(allcom[rowSums(allcom)<r,,drop=FALSE])
  M = nrow(degs)
  if(M!=choose(r+d-1,d)) stop("Error in degrees of polynomials")
  
  # Fast matrix of Euclidean distances
  Em = matrix(sqrt(rowSums(
    apply(tobs,2,function(x) outer(x,x,"-")^2))),nrow=nrow(tobs))
  
  # Matrix Omega for the penalty term
  Omega = eta(Em,d,r)
  
  # Matrix Phi, monomials evaluated at tobs
  Phi = apply(degs,1,function(x) apply(t(tobs)^x,2,prod))
  # Phi = matrix(nrow=p,ncol=M)
  # for(i in 1:p) for(j in 1:M) Phi[i,j] = prod(tobs[i,]^degs[j,])
  
  # Null space of the rows of Phi p-(p-M)
  Q = MASS::Null(Phi)
  
  # Test that t(Phi)%*%Q = 0
  if(max(abs(t(Phi)%*%Q))>1e-4) 
    stop("Problem in computing the null space")
  if(ncol(Q)!=p-M) 
    stop("Problem in computing the null space")
  
  A = Omega%*%Q
  H = matrix(0, nrow=p, ncol=p)
  H[1:(p-M),1:(p-M)] = t(Q)%*%A
  Z = cbind(Omega%*%Q, Phi) # matrix p-times-p
  Z = Z[inds,] # matrix m-times-p
  # Z = do.call("rbind", replicate(n, Z, simplify = FALSE)) # 
  # replicates the matrix Z n-times, and then builds a final matrix
  
  # weights attached to observations (length m)
  w = rep(1/(w*n),w)
  
  return(list(Y=Y, Z=Z, H=H, Q=Q, 
              Omega=Omega, Phi=Phi, w=w, 
              degs=degs, tobs=tobs, M=M, r=r, p=p, d=d, n=n))          
}

#' Prediction for thin-plate multivariate location estimation
#'
#' Given the estimated thin-plate location spline from function 
#' \link{ts_location}, this function predicts the spline function values
#' at new observation points.
#'
#' @param tobs Domain locations for the observed points from which the location 
#' spline was estimated. Matrix of size \code{p1}-times-\code{d}, one row per 
#' domain point.
#' 
#' @param tobsnew Domain locations for the new points where the location 
#' spline is to be estimated. Matrix of size \code{p2}-times-\code{d}, one 
#' row per domain point.
#' 
#' @param gamma Vector \code{gamma} determining the first batch of parameters
#' of the thin-plate spline. Typically outcome of \code{gamma_hat} from 
#' functions \link{ts_location} or \link{transform_theta_location}.
#' 
#' @param delta Vector \code{delta} determining the second batch of parameters
#' of the thin-plate spline. Typically outcome of \code{delta_hat} from 
#' functions \link{ts_location} or \link{transform_theta_location}.
#' 
#' @param r Order of the thin-plate spline, positive integer.
#' 
#' @return A numerical vector of thin-plate spline values of length \code{p2}, 
#' corresponding to the rows of the matrix \code{tobsnew}.
#'
#' @examples
#' d = 1   # dimension of the domain
#' m = 10 # number of points per curve
#' tobs = matrix(runif(m*d), ncol=d)  # location of obsevation points
#' n = 500                            # sample size
#' truemeanf1 = function(x)   # true location function
#'   cos(4*pi*x[1])
#' truemean = apply(tobs,1,truemeanf1) # discretized values of the true location
#' Y = replicate(n, truemean + rnorm(m)) # a matrix of functional data, size m*n
#' 
#' # introduce NAs
#' obsprob = 0.5 # probability of observing a point
#' B = matrix(rbinom(n*m,1,obsprob),ncol=n)
#' for(i in 1:m) for(j in 1:n) if(B[i,j]==0) Y[i,j] = NA
#' 
#' # thin-plate spline fitting
#' res = ts_location(Y, tobs=tobs, r=2, type="square", method="ridge")
#' 
#' jcv = 3 # cross-validation criterion chosen
#' 
#' plot(rep(tobs,n), c(Y), cex=.2, pch=16, ann=FALSE)
#' title("True/estimated location function")
#' points(tobs, truemean, pch=16, col="orange")
#' fullt = matrix(seq(0,1,length=101),ncol=1)
#' lines(fullt, apply(fullt,1,truemeanf1), col="orange", lwd=2)
#' points(tobs, res$beta_hat[,jcv], col=2, pch=16)
#' 
#' # prediction
#' tobsnew = matrix(seq(0,1,length=501),ncol=1)
#' preds = ts_predict_location(tobs, tobsnew, res$gamma_hat[,jcv], 
#'   res$delta_hat[,jcv], r=2)
#' lines(tobsnew, preds, col=2, lwd=2)
#' legend("topleft",c("data","true","estimated"),
#'        pch=16, col=c(1,"orange",2))

ts_predict_location = function(tobs, tobsnew, gamma, delta, r){
  
  # tobs Domain locations for the observed points of size \code{p}-times-\code{d}
  # r Order of the thin-plate spline, positive integer.
  
  p = nrow(tobs)    # number of distinct observation points
  d = ncol(tobs)    # dimension of the domain
  
  if(d!=ncol(tobsnew)) stop("Dimension mismatch in tobs and tobsnew.")
  if(length(gamma)!=p) stop("The length of gamma does not equal nrow(tobs).")

  M = choose(r+d-1,d)
  if(length(delta)!=M) stop("The length of delta does not match.")
  if(p-M<=0) stop(paste("p must be larger than",M))
  if(2*r<=d) stop(paste("r must be larger than",ceiling(d/2)))
  
  # Monomials phi of order <r
  allcom = expand.grid(replicate(d, 0:(r-1), simplify=FALSE))
  degs = as.matrix(allcom[rowSums(allcom)<r,,drop=FALSE])
  M = nrow(degs)
  if(M!=choose(r+d-1,d)) stop("Error in degrees of polynomials")
  
  # Fast matrix of Euclidean distances
  Em = distnAB(tobsnew, tobs, nrow(tobsnew), nrow(tobs), ncol(tobs))
  
  # Matrix Omega for the penalty term
  Omega = eta(Em,d,r)
  
  # Matrix Phi, monomials evaluated at tobs
  Phi = apply(degs,1,function(x) apply(t(tobsnew)^x,2,prod))
  
  return(Omega%*%gamma + Phi%*%delta)        
}

#' Pre-processing raw data for thin-plate spline regression
#'
#' Given the data generated by \link{generate} as an input, pre-processes the 
#' dataset to a form suitable for fitting thin-plate spline regression.
#'
#' @param X Matrix of observed values of \code{X} of size 
#'  \code{n}-times-\code{p}, one row per observation, columns corresponding to the 
#'  positions in the rows of \code{tobs}.
#'  
#' @param tobs Domain locations for the observed points of \code{X}. Matrix
#'  of size \code{p}-times-\code{d}, one row per domain point.
#' 
#' @param m Order of the thin-plate spline, positive integer.
#' 
#' @param int_weights Indicator whether the integrals functions are to be 
#' approximated only as a mean of function values (\code{int_weights=FALSE}), or
#' whether they should be computed as weighted sums of function values 
#' (\code{int_weights=TRUE}). In the latter case, weights are computed as
#' proportional to the respective areas of the Voronoi tesselation of the domain
#' \code{I} (works for dimension \code{d==1} or \code{d==2}). For dimension 
#' \code{d==1}, this is equivalent to the length of intervals associated 
#' to adjacent observation points. For \code{d>2}, no weighting is performed.
#' 
#' @param I.method Applicable only if \code{int_weights==TRUE}. 
#' Input method for the complete domain \code{I}
#' where the Voronoi tesselation for obtaining the weights is evaluated. 
#' Takes a value \code{"box"}
#' for \code{I} the smallest axes-aligned box in the domain, or \code{"chull"}
#' for \code{I} being a convex hull of points in the domain. By default set to 
#' \code{"chull"}. In dimension \code{d=1}, the two methods \code{"box"}
#' and \code{"chull"} are equivalent.
#' 
#' @param I Applicable only if \code{int_weights==TRUE}. A set of points 
#' specifying the complete domain \code{I} of the functional data.
#' In general, can be a \code{q}-times-\code{d} matrix, where \code{q}
#' is the number of points and \code{d} is the dimension. The matrix \code{I}
#' then specifies the point from which to compute the domain \code{I}: 
#' (a) If \code{method.I=="chull"}, the domain is the convex hull of \code{I}; 
#' (b) If \code{method.I=="box"}, the domain is the smallest axes-aligned box
#' that contains \code{I}. If \code{I} is \code{NULL} (by default), then 
#' \code{I} is taken to be the same as \code{x}. If \code{I.method=="box"}, 
#' \code{I} can be specified also by a pair of real values \code{a<b}, 
#' in which case we take \code{I} to be the axis-aligned sqare \code{[a,b]^d}.
#'
#' @return A list of values:
#' \itemize{
#'  \item{"Z"}{ A matrix of size \code{n}-times-\code{p+1} for regression 
#'  fitting. Corresponds to transformed regressors \code{X}.}
#'  \item{"H"}{ A penalty matrix of size \code{p+1}-times-\code{p+1} used for
#'  fitting the regression with thin-plate splines.}
#'  \item{"Q"}{ A matrix containing the null space of the rows of Phi of size
#'  \code{p}-times-\code{p-M}, where \code{M} is the number of monomials
#'  used for the construction of the thin-plate spline. This matrix is used to
#'  pass from parameters gamma to parametrization chi.}
#'  \item{"Omega"}{ A matrix of size \code{p}-times-\code{p} containing the 
#'  \link{eta}-transformed matrix of inter-point distances in \code{tobs}.}
#'  \item{"Phi"}{ A matrix of size \code{p}-times-\code{M} corresponding to the
#'  monomial part of the thin-plate spline.}
#'  \item{"degs"}{ A matrix of size \code{M}-times-\code{d} with degrees of the
#'  monomials in each dimension per row. Used for the construction of 
#'  \code{Phi}.}
#'  \item{"tobs"}{ The same as the input parameter \code{tobs}, for later use.}
#'  \item{"M"}{ Number of all monomials used, is equal to 
#'  \code{choose(m+d-1,d)}.}
#'  \item{"m"}{ Order of the spline, positive integer.}
#'  \item{"p"}{ Number of observed time points, positive integer.}
#'  \item{"d"}{ Dimension of domain, positive integer.}
#'  \item{"n"}{ Sample size, positive integer.}
#' }
#'
#' @examples
#' d = 1       # dimension of domain
#' n = 50      # sample size
#' p = 10      # number of observed points in domain
#' p1 = 101 # size of the complete grid
#' alpha0 = 3  # intercept
#' beta0 = function(t) t^2 # regression coefficient function
#' sd.noiseEps = 0.1 # noise standard deviation
#' tgrid = seq(0,1,length=p1) # full grid of observation points
#' basis_fun = function(t,k) return(sin(2*pi*k*t)+t^k)
#' K = 5 # number of basis functions used in the expansion
#' bfX = list()
#' for(k in 1:K) bfX[[k]] = basis_fun(tgrid,k)
#' bcX = matrix(rnorm(n*K,mean=3,sd=5),ncol=K)  # basis coefficients
#' 
#' # Generate the raw data
#' gen = generate(alpha0, beta0, n, d, p, bfX, bcX, sd.noiseEps)
#' 
#' # Preprocess the data
#' X = gen$X; tobs = gen$tobs
#' m = 2; # degree of the thin-spline used
#' tspr = ts_preprocess(X,tobs,m)
#' 
#' # Thin-plate spline ridge regression
#' ridge(tspr$Z,gen$Y,1e-3,tspr$H)

ts_preprocess = function(X, tobs, m, int_weights=TRUE, 
                         I.method = "chull", I=NULL){
  n = nrow(X)
  p = ncol(X)
  d = ncol(tobs)
  
  if(p!=nrow(tobs)) 
    stop("Number of columns of X must equal number of rows of t")
  
  M = choose(m+d-1,d)
  if(p-M<=0) stop(paste("p must be larger than",M))
  if(2*m<=d) stop(paste("m must be larger than",ceiling(d/2)))
  
  # Establishing the weights for integration in one-dimensional domain
  if(int_weights) 
    integral_weights = vorArea(x = tobs, I.method=I.method, I=I) else
      integral_weights = rep(1/p,p)
  # The domain is taken to be the set [0,1]^d
  
  # if(d==1 & int_weights){
  #   if(any(order(tobs)!=1:p))
  #     stop("For numerical integration, values of tobs must be ordered.")
  #   mids = c(0,(tobs[-1]+tobs[-p])/2,1) # midpoints between adjacent intervals
  #   integral_weights = diff(mids)
  #   if(sum(integral_weights)!=1)
  #     stop("Problem with computing integrating weights.")
  #   # weights given in the one-dimensional integration to the points tobs
  # } else integral_weights = rep(1/p,p)
  # # For multidimensional domain the weights are taken to be equal
  
  # Monomials phi of order <m
  allcom = expand.grid(replicate(d, 0:(m-1), simplify=FALSE))
  degs = as.matrix(allcom[rowSums(allcom)<m,,drop=FALSE])
  M = nrow(degs)
  if(M!=choose(m+d-1,d)) stop("Error in degrees of polynomials")
  
  # Fast matrix of Euclidean distances
  Em = matrix(sqrt(rowSums(
    apply(tobs,2,function(x) outer(x,x,"-")^2))),nrow=nrow(tobs))
  
  # Matrix Omega for the penalty term
  Omega = eta(Em,d,m)
  
  # Matrix Phi, monomials evaluated at tobs
  Phi = apply(degs,1,function(x) apply(t(tobs)^x,2,prod))
  # Phi = matrix(nrow=p,ncol=M)
  # for(i in 1:p) for(j in 1:M) Phi[i,j] = prod(tobs[i,]^degs[j,])
  
  # Null space of the rows of Phi p-(p-M)
  Q = MASS::Null(Phi)
  
  # Test that t(Phi)%*%Q = 0
  if(max(abs(t(Phi)%*%Q))>1e-4) 
    stop("Problem in computing the null space")
  if(ncol(Q)!=p-M) 
    stop("Problem in computing the null space")
  
  A = Omega%*%Q
  B = t(Q)%*%A
  Int_W = diag(integral_weights)
  H = cbind(0,rbind(0,
                    cbind(t(A)%*%Int_W%*%A + B, t(A)%*%Int_W%*%Phi),
                    cbind(t(Phi)%*%Int_W%*%A, t(Phi)%*%Int_W%*%Phi)))
  Z = cbind(1,X%*%Int_W%*%Omega%*%Q,X%*%Int_W%*%Phi)
  
  return(list(Z=Z, H=H, Q=Q, Omega=Omega, Phi=Phi, 
              degs=degs, tobs=tobs, M=M, m=m, p=p, d=d, n=n,
              w=integral_weights))
}

#' Area of the cells in Voronoi tesselation
#'
#' For one-dimensional or two-dimensional sets of points, 
#' finds the Voronoi tesselation of the data and returns
#' a vector of volumes (lengths or areas) of all the Voronoi 
#' cells.
#'
#' @param x The dataset whose Voronoi tesselation should be considered.
#' \code{p}-times-\code{d} matrix, one row per point in 
#' \code{d}-dimensional space. If \code{d==1}, the elements of \code{x} must
#' be sorted.
#'
#' @param I.method Input method for the complete domain \code{I}
#' where the Voronoi tesselation is evaluated. Takes a value \code{"box"}
#' for \code{I} the smallest axes-aligned box, or \code{"chull"}
#' for \code{I} being a convex hull of points. By default set to 
#' \code{"chull"}. In dimension \code{d=1}, the two methods \code{"box"}
#' and \code{"chull"} are equivalent.
#' 
#' @param I A set of points specifying the complete domain \code{I}.
#' In general, can be a \code{q}-times-\code{d} matrix, where \code{q}
#' is the number of points and \code{d} is the dimension. The matrix \code{I}
#' then specifies the point from which to compute the domain \code{I}: 
#' (a) If \code{method.I=="chull"}, the domain is the convex hull of \code{I}; 
#' (b) If \code{method.I=="box"}, the domain is the smallest axes-aligned box
#' that contains \code{I}. If \code{I} is \code{NULL} (by default), then 
#' \code{I} is taken to be the same as \code{x}. If \code{I.method=="box"}, 
#' \code{I} can be specified also by a pair of real values \code{a<b}, 
#' in which case we take \code{I} to be the axis-aligned sqare \code{[a,b]^d}.
#' 
#' @param scale Should the resulting volumes be scaled to sum to \code{1}? By
#' default set to \code{TRUE}.
#' 
#' @param plot A logical indicator of whether the resulting Voronoi cells should
#' be plotted. By default \code{FALSE}.
#'
#' @return A numerical vector of length \code{p} of volumes of the Voronoi cells
#' corresponding to the rows of \code{x}. If \code{scale==TRUE} (default), this
#' vector is scaled so that the sum of its elements is \code{1}. If some of the
#' elements of this vector is numerically zero (that is, some of the Voronoi 
#' cells are empty), a warning is given.
#'
#' @examples
#' p = 50      # number of observed points in domain
#' x = runif(p) # x-coordinates of the points
#' y = runif(p) # y-coordinates of the points
#' 
#' # Voronoi areas in the plane:
#' # (a) I is the smallest box containing the data
#' vorArea(cbind(x,y), I.method="box",plot=TRUE)
#' 
#' # (b) I is the box [0,1]^2
#' vorArea(cbind(x,y),I.method="box",I=c(0,1),plot=TRUE)
#' 
#' # (c) I is the convex hull of the data
#' vorArea(cbind(x,y),I.method="chull",plot=TRUE)
#' 
#' # Voronoi areas in the real line:
#' # (a) I is the smallest interval containing the data
#' vorArea(x = matrix(sort(x),ncol=1),I.method="chull",plot=TRUE)
#' # (b) I = [0,1]
#' vorArea(x = matrix(sort(x),ncol=1),I.method="box",I=c(0,1),plot=TRUE)
#' # (c) I = [-1,1]
#' vorArea(x = matrix(sort(x),ncol=1),I.method="chull",I=matrix(c(-1,1),ncol=1),plot=TRUE)

vorArea = function(x, I.method = "chull", I=NULL, scale = TRUE, plot = FALSE){
  # input:
  # x: p-times-d matrix, one row per time point in R^d
  # I.method: "box" for I being the smallest axes-aligned box
  #           "chull" for I being the convex hull of I
  # I: q-times-d matrix, specifying the observation window I
  #    If I is NULL, then I = x.
  #    If I.method="box", I is a pair of points a<b, and I=[a,b]^d
  # output:
  # vector of length p of areas of the Voronoi cells
  # scaled to have total sum 1
  
  if(!is.matrix(x)) stop("Points in the domain x must be a matrix of dimension p-times-d.")
  d = ncol(x)
  p = nrow(x) 
  if(d>2){
    warning("vorArea currently implemented only for dimensions d<=2, output is only a vector of the same value 1/p.")
    return(rep(1/p,length=p))
  }
  I.method = match.arg(I.method, c("box","chull"))
  if(is.null(I)) I = x
  if(I.method=="box"){
    if(!is.matrix(I)) I = matrix(I,nrow=2,ncol=d)
    rng = apply(I,2,range)
    bnds = lapply(seq_len(ncol(rng)), function(i) rng[,i])
    I = expand.grid(bnds)
    # I = expand.grid(replicate(d, 0:1, simplify=FALSE))  
  }
  #
  if(ncol(I)!=d) stop("The dimensions of x and I in vorArea do not match.")
  #
  if(d==1){
    I = apply(I,2,range)
    if(I[1,1]>=I[2,1]) stop("The points defining the domain I must be ordered.")
    tobs = x[,1]
    if(any(order(tobs)!=1:p))
      stop("For numerical integration, values of tobs must be ordered.")
    mids = c(I[1,1],(tobs[-1]+tobs[-p])/2,I[2,1]) # midpoints between adjacent intervals
    mids[mids<=I[1,1]] = I[1,1]
    mids[mids>=I[2,1]] = I[2,1]
    ret = diff(mids)
    #
    if(plot){
      plot(rep(0,p) ~ tobs, pch = 19, cex=.5)
      abline(h=0,lty=3)
      points(rep(0,length(mids)) ~ mids, pch=3, col=2)
    }
    # if(sum(integral_weights)!=1)
    #  stop("Problem with computing integrating weights.")
    # weights given in the one-dimensional integration to the points tobs
    if(any(ret<=1e-25)) warning("Some Voronoi cells are of zero area, possibly conv(tobs) is not a subset of I.")
    if(scale) ret = ret/sum(ret) # scaling for total sum 1
    return(unname(ret))
  }
  #
  if(d==2){
    # deldir package
    I = I[chull(I),]
    I = list(x=I[,1],y=I[,2])
    #
    tesselation = tryCatch(
      error = function(cnd){
        warning(paste0("vorArea failed, jittering data, ",cnd))
        deldir(x[,1], x[,2])}, {
        deldir(jitter(x[,1]), jitter(x[,2]))
      })
      # deldir(x[,1], x[,2])
    tiles <- tile.list(tesselation, clipp=I)
    # indices of points whose cells are non-empty
    inds = sapply(tiles,function(x) x$ptNum)
    areas = sapply(tiles,function(x) x$area)
    #
    if(plot) plot(tiles, pch = 19)
    #
    ret = rep(0,p)
    ret[inds] = areas
    if(any(ret<=1e-25)) warning("Some Voronoi cells are of zero area, possibly conv(tobs) is not a subset of I.")
    if(scale) ret = ret/sum(ret) # scaling for total sum 1
    return(unname(ret))
  }
}

#' Transform the vector of estimated coefficients for location estimation
#'
#' Splits the vector of raw estimated coefficients (output of functions 
#' \link{IRLS} or \link{ridge}) after performing \link{ts_preprocess_locationi} 
#' into parts interpretable in the setup of thin-plate spline location 
#' estimation.
#'
#' @param theta Output vector of raw results of length \code{p} from function
#' \link{IRLS} or \link{ridge}.
#'
#' @param tspr Output of \link{ts_preprocess_location}.
#'
#' @return A list of estimated parameters:
#' \itemize{
#'  \item{"xi_hat"}{ Estimate of \code{xi}, the part of the parameters that
#'  correspond to matrix \code{Omega} when transformed by \code{Q}. A vector
#'  of length \code{(p-M)}.}
#'  \item{"delta_hat"}{ Estimate of \code{delta}, the part of the parameters that
#'  correspond to matrix \code{Phi}. A vector of length \code{M}, the number of
#'  monomials used for the construction of the thin-plate spline.}
#'  \item{"gamma_hat"}{ Estimate of \code{gamma}, the part of the parameters that
#'  correspond to matrix \code{Omega}. It holds true that 
#'  \code{gamma_hat = Q*xi_hat}. A vector of length \code{p}.}
#'  \item{"beta_hat"}{ Estimate of the location parameter \code{mu} 
#'  evaluated at the \code{p} points from \code{tobs}, where \code{Y} was
#'  observed.}
#' }
#'
#' @examples
#' d = 1  # dimension of the domain
#' m = 50 # number of observation times per function
#' tobs = matrix(runif(m*d), ncol=d) # matrix of observation times
#' n = 20 # sample size
#' truemeanf = function(x) 10+15*x[1]^2 # true mean function
#' truemean = apply(tobs,1,truemeanf) # discretized values of the true mean
#' Y = replicate(n, truemean + rnorm(m)) # matrix of discrete functiona data, size p*n
#' tsp = ts_preprocess_location(Y, tobs, 2) # preprocessing matrices 
#' 
#' lambda0 = 1e-5 # regularization parameter
#' res_IRLS = IRLS(Z = tsp$Z, Y = tsp$Y, lambda = lambda0, H = tsp$H, type = "square")
#' res_ridge = ridge(Z = tsp$Z, Y = tsp$Y, lambda = lambda0, H = tsp$H)
#' # resulting estimates of the parameters theta, using IRLS and ridge
#' 
#' # testing that ridge and IRLS (square) give the same results
#' all.equal(res_IRLS$theta_hat, res_ridge$theta_hat)
#' 
#' res = res_ridge
#' resf = transform_theta_location(res$theta_hat, tsp)
#' 
#' plot(rep(tobs,n), c(Y), cex=.2, pch=16)
#' points(tobs, truemean, pch=16, col="orange")
#' points(tobs, resf$beta_hat, col=2, pch=16)

transform_theta_location = function(theta, tspr){
  p = tspr$p
  M = tspr$M
  Q = tspr$Q
  Omega = tspr$Omega
  Phi = tspr$Phi
  xihat=theta[1:(p-M)]
  dhat=theta[(p-M+1):p]
  ghat = Q%*%xihat
  return(list(xi_hat=xihat,
              delta_hat=dhat,
              gamma_hat=ghat,
              beta_hat = c(Omega%*%ghat + Phi%*%dhat)))
}

#' Transform the vector of estimated regression coefficients
#'
#' Splits the vector of raw estimated coefficients (output of functions 
#' \link{IRLS} or \link{ridge}) into parts interpretable in the setup of 
#' thin-plate spline regression setup.
#'
#' @param theta Output vector of raw results of length \code{p+1} from function
#' \link{IRLS} or \link{ridge}.
#'
#' @param tspr Output of \link{ts_preprocess}.
#'
#' @return A list of estimated parameters:
#' \itemize{
#'  \item{"alpha_hat"}{ Estimate of \code{alpha0}, a numerical value.}
#'  \item{"xi_hat"}{ Estimate of \code{xi}, the part of the parameters that
#'  correspond to matrix \code{Omega} when transformed by \code{Q}. A vector
#'  of length \code{(p-M)}.}
#'  \item{"delta_hat"}{ Estimate of \code{delta}, the part of the parameters that
#'  correspond to matrix \code{Phi}. A vector of length \code{M}, the number of
#'  monomials used for the construction of the thin-plate spline.}
#'  \item{"gamma_hat"}{ Estimate of \code{gamma}, the part of the parameters that
#'  correspond to matrix \code{Omega}. It holds true that 
#'  \code{gamma_hat = Q*xi_hat}. A vector of length \code{p}.}
#'  \item{"beta_hat"}{ Estimate of the regression function \code{beta0} 
#'  evaluated at the \code{p} points from \code{tobs}, where \code{X} was
#'  observed.}
#' }
#'
#' @examples
#' d = 1       # dimension of domain
#' n = 50      # sample size
#' p = 10      # number of observed points in domain
#' p1 = 101 # size of the complete grid
#' alpha0 = 3  # intercept
#' beta0 = function(t) t^2 # regression coefficient function
#' sd.noiseEps = 0.1 # noise standard deviation
#' tgrid = seq(0,1,length=p1) # full grid of observation points
#' basis_fun = function(t,k) return(sin(2*pi*k*t)+t^k)
#' K = 5 # number of basis functions used in the expansion
#' bfX = list()
#' for(k in 1:K) bfX[[k]] = basis_fun(tgrid,k)
#' bcX = matrix(rnorm(n*K,mean=3,sd=5),ncol=K)  # basis coefficients
#' 
#' # Generate the raw data
#' gen = generate(alpha0, beta0, n, d, p, bfX, bcX, sd.noiseEps)
#' 
#' # Preprocess the data
#' X = gen$X; Y = gen$Y; tobs = gen$tobs
#' m = 2; # degree of thin-spline
#' tspr = ts_preprocess(X,tobs,m)
#' 
#' # Thin-plate spline ridge regression
#' res = ridge(tspr$Z,Y,1e-3,tspr$H)
#' 
#' # Transform the coefficients
#' transform_theta(res$theta_hat, tspr)

transform_theta = function(theta,tspr){
  p = tspr$p
  M = tspr$M
  Q = tspr$Q
  Omega = tspr$Omega
  Phi = tspr$Phi
  xihat=theta[2:(p-M+1)]
  dhat=theta[(p-M+2):(p+1)]
  ghat = Q%*%xihat
  return(list(alpha_hat=theta[1],xi_hat=xihat,
              delta_hat=dhat,gamma_hat=ghat,
              beta_hat = c(Omega%*%ghat + Phi%*%dhat)))
}

#' Reconstruct and visualise the slope function beta0 and its estimate
#'
#' Based on the thin-splines estimates of \code{beta0} obtained from 
#' functions \link{IRLS} or \link{ridge}, this function reconstructs the 
#' complete estimator of the regression function \code{beta0} on its full 
#' domain. It also visualises the results. 
#'
#' @param ts_prep Output of the function \link{ts_preprocess}.
#' 
#' @param theta Estimated coefficients, output of \link{IRLS} or \link{ridge}
#' after passed through \link{transform_theta}. Either \code{theta} or
#' \code{lambda} must be provided. If \code{theta} is given, \code{lambda}
#' is disregarded.
#'
#' @param lambda Tuning parameter, a non-negative real number. Either \code{theta} or
#' \code{lambda} must be provided. If \code{theta} is not given, \link{IRLS}
#' is fitted with parameter \code{lambda}.
#'
#' @param Y Vector of responses of length \code{n}.
#' 
#' @param type The type of the loss function used in the minimization problem.
#' Accepted are \code{type="absolute"} for the absolute loss \code{rho(t)=|t|}; 
#' \code{type="square"} for the square loss \code{rho(t)=t^2}; 
#' \code{type="Huber"} for the Huber loss \code{rho(t)=t^2/2} if 
#' \code{|t|<tuning} and \code{rho(t)=tuning*(|t|-tuning/2)} otherwise; and 
#' \code{type="logistic"} for the logistic loss 
#' \code{rho(t)=2*t + 4*log(1+exp(-t))-4*log(2)}.
#' 
#' @param p1 Number of equidistant discretization points in the interval [0,1]
#' in each dimension \code{d} of the domain. The total number of discretization
#' points at which the estimator of \code{beta0} will be generated is 
#' \code{p1^d}.
#' 
#' @param betafull The true function \code{beta0} to be plotted along its
#' estimator. A \code{d}-dimensional array of the values of the 
#' regression function \code{beta0} of size \code{p1}-times-...-times-\code{p1}.
#' Interpretation of values as for the functions of \code{Xfull} in function
#' \link{generate}. Can be provided directly as the output of \link{generate}.
#' 
#' @param main Title of the displayed plot.
#' 
#' @return A plot with the true and the estimated function \code{beta0}, and
#' the corresponding output of the function \link{IRLS}.
#'
#' @examples
#' d = 1       # dimension of domain
#' n = 50      # sample size
#' p = 10      # number of observed points in domain
#' p1 = 101 # size of the complete grid
#' alpha0 = 3  # intercept
#' beta0 = function(t) t^2 # regression coefficient function
#' sd.noiseEps = 0.1 # noise standard deviation
#' tgrid = seq(0,1,length=p1) # full grid of observation points
#' basis_fun = function(t,k) return(sin(2*pi*k*t)+t^k)
#' K = 5 # number of basis functions used in the expansion
#' bfX = list()
#' for(k in 1:K) bfX[[k]] = basis_fun(tgrid,k)
#' bcX = matrix(rnorm(n*K,mean=3,sd=5),ncol=K)  # basis coefficients
#' 
#' # Generate the raw data
#' gen = generate(alpha0, beta0, n, d, p, bfX, bcX, sd.noiseEps)
#' 
#' # Preprocess the data
#' X = gen$X; Y = gen$Y; tobs = gen$tobs
#' m = 2   # order of thin-plate splines
#' tspr = ts_preprocess(X,tobs,m)
#' 
#' # Reconstruction by specifying regression method
#' reconstruct(tspr, theta = NULL,lambda = 1e-3,Y,"absolute",
#'   p1=p1, betafull=gen$betafull, main="L1 regression")
#'   
#' # Reconstruction by using theta from previously fitted model
#' res = IRLS(tspr$Z,Y,1e-3,tspr$H,"absolute")
#' reconstruct(ts_prep = tspr, theta = transform_theta(res$theta_hat,tspr), 
#' lambda=NULL, Y = Y, type = "absolute", 
#' p1 = p1, betafull = gen$betafull)

reconstruct = function(ts_prep = NULL, 
                       theta=NULL, lambda=NULL,
                       Y=NULL, type=NULL,
                       p1 = 101,
                       betafull=NULL,main=NULL){
  tgrid = seq(0,1,length=p1)
  # plots the complete estimated function beta
  if(is.null(theta) & is.null(lambda)) 
    stop("Either theta or lambda must be given.")
  if(is.null(ts_prep)) stop("ts_prep must be given.")
  if(is.null(theta)){
    if(any(is.null(ts_prep$Z),is.null(Y),is.null(ts_prep$H),is.null(type)))
      stop("Z, Y, H, and type must be provided if theta is to be estimated.")
    # estimate directly
    res = IRLS(ts_prep$Z,Y,lambda,ts_prep$H,type)
    theta = transform_theta(res$theta_hat,ts_prep)
  } else res = NULL
  
  d = ts_prep$d
  if(d>2) stop("Visualisation only for d<=2.")
  #
  tfull = expand.grid(replicate(d,tgrid,simplify=FALSE))
  #
  tobs=ts_prep$tobs
  m = ts_prep$m
  degs = ts_prep$degs
  
  # Fast matrix of Euclidean distances
  if(d==1) Emfull = abs(outer(tfull[,1],tobs[,1],"-"))
  if(d==2) Emfull = sqrt(outer(tfull[,1],tobs[,1],"-")^2 + 
                           outer(tfull[,2],tobs[,2],"-")^2)
  
  # Matrix Omega for the penalty term
  Omegafull = eta(Emfull,d,m)
  # Matrix Phi, monomials evaluated at t
  Phifull = apply(degs,1,function(x) apply(t(tfull)^x,2,prod))
  #
  b_hatfull0 = c(Omegafull%*%theta$gamma_hat + Phifull%*%theta$delta_hat)
  b_hatfull = matrix(b_hatfull0,nrow=p1)
  
  if(d==1){
    if(!is.null(betafull)){
      plot(c(b_hatfull,betafull)~c(tgrid,tgrid),type="n",
         xlab="t",ylab=expression(beta[0]),main=main)
      lines(betafull~tgrid,lwd=2)
      lines(b_hatfull~tgrid,col=2,lwd=2)
      legend("topleft",c("True","Estimated"),col=1:2,lwd=2)
    } else {
      plot(c(b_hatfull)~c(tgrid),type="n",
           xlab="t",ylab=expression(beta[0]),main=main)
      lines(b_hatfull~tgrid,col=2,lwd=2)
      legend("topleft",c("Estimated"),col=2,lwd=2)
    }
    rug(tobs)
  }
  if(d==2){
    if(!is.null(betafull)) op = par(mfrow=c(1,2))
    image(b_hatfull,main=main)
    if(!is.null(betafull)) image(betafull,main="True beta")
    if(!is.null(betafull)) par(op)
  }
  return(res)
}

#' Cross-Validation for the IRLS procedure
#'
#' Provides the cross-validation indices from \link{GCV_crit} in
#' conjunction with the \link{IRLS} function directly as an argument of the
#' parameter \code{lambda}. 
#'
#' @param lambda A candidate parameter value; non-negative real number.
#'
#' @param Z Data matrix of dimension \code{n}-times-\code{p}, where \code{n} is
#' the number of observations, \code{p} is the dimension.
#'
#' @param Y Vector of responses of length \code{n}.
#'
#' @param H Penalty matrix of size \code{p}-times-\code{p} that
#' is used inside the quadratic term for penalizing estimated parameters.
#' 
#' @param type The type of the loss function used in the minimization problem.
#' Accepted are \code{type="absolute"} for the absolute loss \code{rho(t)=|t|}; 
#' \code{type="square"} for the square loss \code{rho(t)=t^2}; 
#' \code{type="Huber"} for the Huber loss \code{rho(t)=t^2/2} if 
#' \code{|t|<tuning} and \code{rho(t)=tuning*(|t|-tuning/2)} otherwise; and 
#' \code{type="logistic"} for the logistic loss 
#' \code{rho(t)=2*t + 4*log(1+exp(-t))-4*log(2)}.
#' 
#' @param sc Scale parameter to be used in the IRLS. By default \code{sc=1}, 
#' that is no scaling is performed.
#' 
#' @param vrs Version of the algorhitm to be used in function \link{IRLS}; 
#' either \code{vrs="C"} for the \code{C++} version, or \code{vrs="R"} for the 
#' \code{R} version. Both should give (nearly) identical results, see 
#' \link{IRLS}.
#' 
#' @param custfun A custom function combining the residuals \code{resids} and
#' the hat values \code{hats}. The result of the function must be numeric, 
#' see \link{GCV_crit}.
#' 
#' @param resids.in Initialization of the vector of residuals used to launch 
#' the IRLS algorithms. Optional.
#' 
#' @param toler A small positive constant specifying the tolerance level for 
#' terminating the algorithm. The prcedure stops if the maximum absolute 
#' distance between the residuals in the previous iteration and the new 
#' residuals drops below \code{toler}.
#' 
#' @param imax Maximum number of allowed iterations of IRLS. 
#'
#' @details Function \code{custfun} has two arguments 
#' corresponding to \code{resids} and \code{hats}. The output of the function 
#' must be numeric. 
#' 
#' @return A named numerical vector of values. The length of the vector depends
#' on the input. The vector contains the values:
#' \itemize{
#'  \item{"AIC"}{ Akaike's information criterion given by 
#'  \code{mean(resids^2)+log(n)*mean(hats)}, where \code{n} is the length of
#'  both \code{resids} and \code{hats}.}
#'  \item{"GCV"}{ Leave-one-out cross-validation criterion given by
#'  \code{mean((resids^2)/((1-hats)^2))}.}
#'  \item{"GCV(tr)"}{ Modified leave-one-out cross-validation criterion 
#'  given by \code{mean((resids^2)/((1-mean(hats))^2))}.}
#'  \item{"BIC"}{ Bayes information criterion given by 
#'  \code{mean(resids^2)+2*mean(hats)}.}
#'  \item{"rGCV"}{ A robust version of \code{GCV} where mean is replaced
#'  by a robust M-estimator of scale of \code{resids/(1-hats)}, see 
#'  \link[robustbase]{scaleTau2} for details.}
#'  \item{"rGCV(tr)"}{ Modified version of a \code{rGCV} given by 
#'  a robust M-estimator of scale of \code{resids/(1-mean(hats))}.}
#'  \item{"custom"}{ The custom criterion given by function \code{custfun}. 
#'  Works only if \code{custfun} is part of the input.}
#' }
#'
#' @examples
#' n = 50      # sample size
#' p = 10      # dimension of predictors
#' Z = matrix(rnorm(n*p),ncol=p) # design matrix
#' Y = Z[,1]   # response vector
#' lambda = 1  # tuning parameter for penalization
#' H = diag(p) # penalty matrix
#' type = "absolute" # absolute loss
#' 
#' # Run with the IRLS procedure
#' res = IRLS(Z, Y, lambda, H, type)
#' with(res,GCV_crit(resids,hat_values))
#' with(res,GCV_crit(resids,hat_values,custfun = function(r,h) sum(r^2)))
#'     
#' GCV(lambda,Z,Y,H,type,custfun = function(r,h) sum(r^2))

GCV <- function(lambda, Z, Y, H, type, sc = 1, 
                vrs="C", custfun=NULL, 
                resids.in = rep(1,length(Y)),
                toler=1e-7, imax=1000){
  # Generalized cross-validation
  ncv = 6
  if(!is.null(custfun)) ncv = 7
  vrs = match.arg(vrs, c("C", "R"))
  fit.r <- IRLS(Z, Y, lambda, H, type, sc = sc, vrs=vrs, 
                resids.in = resids.in,
                toler=toler, imax=imax)
  GCV.scores <- GCV_crit(fit.r$resids,fit.r$hat_values,
                           custfun=custfun)
  return(c(GCV.scores, fit.r$converged, fit.r$ic))
}

#' Cross-validation for location estimation
#'
#' Provides the cross-validation indices from \link{GCV_crit} in
#' conjunction with the \link{IRLS} function and \link{ridge} function directly 
#' as an argument of the parameter \code{lambda}. Applicable for location
#' estimation and function \link{ts_location}.
#'
#' @param lambda A candidate parameter value; non-negative real number.
#'
#' @param Z Data matrix of dimension \code{n}-times-\code{p}, where \code{n} is
#' the number of observations, \code{p} is the dimension.
#'
#' @param Y Vector of responses of length \code{n}.
#'
#' @param H Penalty matrix of size \code{p}-times-\code{p} that
#' is used inside the quadratic term for penalizing estimated parameters.
#' 
#' @param type The type of the loss function used in the minimization problem.
#' Accepted are \code{type="absolute"} for the absolute loss \code{rho(t)=|t|}; 
#' \code{type="square"} for the square loss \code{rho(t)=t^2}; 
#' \code{type="Huber"} for the Huber loss \code{rho(t)=t^2/2} if 
#' \code{|t|<tuning} and \code{rho(t)=tuning*(|t|-tuning/2)} otherwise; and 
#' \code{type="logistic"} for the logistic loss 
#' \code{rho(t)=2*t + 4*log(1+exp(-t))-4*log(2)}.
#' 
#' @param w Vector of length \code{n} of weights attached to the elements of 
#' \code{Y}. If \code{w=NULL} (default), a constant vector with values 
#' \code{1/n} is used. 
#'
#' @param vrs Version of the algorhitm to be used in function \link{IRLS}; 
#' either \code{vrs="C"} for the \code{C++} version, or \code{vrs="R"} for the 
#' \code{R} version. Both should give (nearly) identical results, see 
#' \link{IRLS} and \link{ridge}.
#' 
#' @param method A method for estimating the fit. Possible options are 
#' \code{"IRLS"} for the IRLS algorithm, or \code{"ridge"} for ridge regression.
#' Ridge is applicable only if \code{type="square"}; this method is much faster,
#' but provides only a non-robust fit.
#'
#' @param custfun A custom function combining the residuals \code{resids} and
#' the hat values \code{hats}. The result of the function must be numeric, 
#' see \link{GCV_crit}.
#'
#' @details Function \code{custfun} has two arguments 
#' corresponding to \code{resids} and \code{hats}. The output of the function 
#' must be numeric. 
#' 
#' @return A named numerical vector of values. The length of the vector depends
#' on the input. The vector contains the values:
#' \itemize{
#'  \item{"AIC"}{ Akaike's information criterion given by 
#'  \code{mean(resids^2)+log(n)*mean(hats)}, where \code{n} is the length of
#'  both \code{resids} and \code{hats}.}
#'  \item{"GCV"}{ Leave-one-out cross-validation criterion given by
#'  \code{mean((resids^2)/((1-hats)^2))}.}
#'  \item{"GCV(tr)"}{ Modified leave-one-out cross-validation criterion 
#'  given by \code{mean((resids^2)/((1-mean(hats))^2))}.}
#'  \item{"BIC"}{ Bayes information criterion given by 
#'  \code{mean(resids^2)+2*mean(hats)}.}
#'  \item{"rGCV"}{ A robust version of \code{GCV} where mean is replaced
#'  by a robust M-estimator of scale of \code{resids/(1-hats)}, see 
#'  \link[robustbase]{scaleTau2} for details.}
#'  \item{"rGCV(tr)"}{ Modified version of a \code{rGCV} given by 
#'  a robust M-estimator of scale of \code{resids/(1-mean(hats))}.}
#'  \item{"custom"}{ The custom criterion given by function \code{custfun}. 
#'  Works only if \code{custfun} is part of the input.}
#' }
#'
#' @examples
#' n = 50      # sample size
#' p = 10      # dimension of predictors
#' Z = matrix(rnorm(n*p),ncol=p) # design matrix
#' Y = Z[,1]   # response vector
#' lambda = 1  # tuning parameter for penalization
#' H = diag(p) # penalty matrix
#' type = "absolute" # absolute loss
#' 
#' # Run with the IRLS procedure
#' res = IRLS(Z, Y, lambda, H, type)
#' with(res,GCV_crit(resids,hat_values))
#' with(res,GCV_crit(resids,hat_values,custfun = function(r,h) 
#'     sum(r^2)))
#'     
#' w = rep(1/n,n)
#' GCV_location(lambda,Z,Y,H,type,w,custfun = function(r,h) sum(r^2))

GCV_location <- function(lambda, Z, Y, H, type, w, vrs="C", 
                         method="IRLS",
                         custfun=NULL, 
                         resids.in = rep(1,length(Y)),
                         toler=1e-7, imax=1000){
  
  method = match.arg(method,c("IRLS", "ridge"))
  type = match.arg(type,c("square","absolute","Huber","logistic"))
  if(method=="ridge" & type!="square") 
    stop("method 'ridge' available only for type 'square'.")
  
  # Generalized cross-validation
  ncv = 6
  if(!is.null(custfun)) ncv = 7
  vrs = match.arg(vrs, c("C", "R"))
  
  if(method=="IRLS"){
    fit.r <- IRLS(Z, Y, lambda, H, type=type, w=w, vrs=vrs, 
                  resids.in = resids.in,
                  toler=toler, imax=imax)
    GCV.scores <- GCV_crit(fit.r$resids,fit.r$hat_values,
                           custfun=custfun)
    return(c(GCV.scores, fit.r$converged, fit.r$ic))
  }
  if(method=="ridge"){
    fit.r <- ridge(Z, Y, lambda, H, w=w, vrs=vrs)
    GCV.scores <- GCV_crit(fit.r$resids,fit.r$hat_values,custfun=custfun)
    return(c(GCV.scores, 1, 0))
  }
}

#' Cross-validation for Ridge Regression
#'
#' Provides the cross-validation indices from \link{GCV_crit} in
#' conjunction with the \link{ridge} function directly as an argument of the
#' parameter \code{lambda}. 
#'
#' @param lambda A candidate parameter value; non-negative real number.
#'
#' @param Z Data matrix of dimension \code{n}-times-\code{p}, where \code{n} is
#' the number of observations, \code{p} is the dimension.
#'
#' @param Y Vector of responses of length \code{n}.
#'
#' @param H Penalty matrix of size \code{p}-times-\code{p} that
#' is used inside the quadratic term for penalizing estimated parameters.
#' 
#' @param vrs Version of the algorhitm to be used in function \link{ridge}; 
#' either \code{vrs="C"} for the \code{C++} version, or \code{vrs="R"} for the 
#' \code{R} version. Both should give (nearly) identical results, see 
#' \link{ridge}.
#'
#' @param custfun A custom function combining the residuals \code{resids}, the 
#' hat values \code{hats}. The result of the function must be numeric, see
#' \link{GCV_crit}.
#'
#' @details Function \code{custfun} has two arguments 
#' corresponding to \code{resids} and \code{hats}. The output of the function 
#' must be numeric. Additional parameters passed to \code{ridge} may be passed
#' to the function.
#'
#' @return A named numerical vector of values. The length of the vector depends
#' on the input. The vector contains the values:
#' \itemize{
#'  \item{"AIC"}{ Akaike's information criterion given by 
#'  \code{mean(resids^2)+log(n)*mean(hats)}, where \code{n} is the length of
#'  both \code{resids} and \code{hats}.}
#'  \item{"GCV"}{ Leave-one-out cross-validation criterion given by
#'  \code{mean((resids^2)/((1-hats)^2))}.}
#'  \item{"GCV(tr)"}{ Modified leave-one-out cross-validation criterion 
#'  given by \code{mean((resids^2)/((1-mean(hats))^2))}.}
#'  \item{"BIC"}{ Bayes information criterion given by 
#'  \code{mean(resids^2)+2*mean(hats)}.}
#'  \item{"rGCV"}{ A robust version of \code{GCV} where mean is replaced
#'  by a robust M-estimator of scale of \code{resids/(1-hats)}, see 
#'  \link[robustbase]{scaleTau2} for details.}
#'  \item{"rGCV(tr)"}{ Modified version of a \code{rGCV} given by 
#'  a robust M-estimator of scale of \code{resids/(1-mean(hats))}.}
#'  \item{"custom"}{ The custom criterion given by function \code{custfun}. 
#'  Works only if \code{custfun} is part of the input.}
#' }
#'
#' @examples
#' n = 50      # sample size
#' p = 10      # dimension of predictors
#' Z = matrix(rnorm(n*p),ncol=p) # design matrix
#' Y = Z[,1]   # response vector
#' lambda = 1  # tuning parameter for penalization
#' H = diag(p) # penalty matrix
#'     
#' # Run with the ridge function
#' res = ridge(Z, Y, lambda, H)
#' with(res,GCV_crit(resids,hat_values))
#' with(res,GCV_crit(resids,hat_values,custfun = function(r,h) 
#'     sum((r/(1-h))^2)))
#'     
#' GCV_ridge(lambda,Z,Y,H,custfun = function(r,h) sum((r/(1-h))^2))

GCV_ridge <- function(lambda,Z,Y,H,vrs="C",custfun=NULL){
  # Generalized cross-validation for ridge
  vrs = match.arg(vrs, c("C", "R"))
  fit.r <- ridge(Z,Y,lambda,H,vrs=vrs)
  GCV.scores <- GCV_crit(fit.r$resids, fit.r$hat_values, custfun=custfun)
  return(GCV.scores)
}

#' Criteria used for cross-validation and for tuning parameter lambda
#'
#' Several criteria commonly used for selection of the tuning parameter 
#' \code{lambda} in functions \link{IRLS} and \link{ridge}. 
#'
#' @param resids A vector of residuals of length \code{n}.
#'
#' @param hats A vector of hat values of length \code{n}.
#' 
#' @param custfun A custom function combining the residuals \code{resids} and 
#' the hat values \code{hats}. The 
#' result of the function must be numeric.
#'
#' @details Function \code{custfun} has two arguments 
#' corresponding to \code{resids} and \code{hats}. The output of the function 
#' must be numeric.
#'
#' @return A named numerical vector of values. The length of the vector depends
#' on the input. The vector contains (some of) the values:
#' \itemize{
#'  \item{"AIC"}{ Akaike's information criterion given by 
#'  \code{mean(resids^2)+2*mean(hats)}, where \code{n} is the length of
#'  both \code{resids} and \code{hats}.}
#'  \item{"GCV"}{ Leave-one-out cross-validation criterion given by
#'  \code{mean((resids^2)/((1-hats)^2))}.}
#'  \item{"GCV(tr)"}{ Modified leave-one-out cross-validation criterion 
#'  given by \code{mean((resids^2)/((1-mean(hats))^2))}.}
#'  \item{"BIC"}{ Bayes information criterion given by 
#'  \code{mean(resids^2)+log(n)*mean(hats)}.}
#'  \item{"rGCV"}{ A robust version of \code{GCV} where mean is replaced
#'  by a robust M-estimator of scale of \code{resids/(1-hats)}, see 
#'  \link[robustbase]{scaleTau2} for details.}
#'  \item{"rGCV(tr)"}{ Modified version of a \code{rGCV} given by 
#'  a robust M-estimator of scale of \code{resids/(1-mean(hats))}.}
#'  \item{"custom"}{ The custom criterion given by function \code{custfun}. 
#'  Works only if \code{custfun} is part of the input.}
#' }
#'
#' @examples
#' n = 50      # sample size
#' p = 10      # dimension of predictors
#' Z = matrix(rnorm(n*p),ncol=p) # design matrix
#' Y = Z[,1]   # response vector
#' lambda = 1  # tuning parameter for penalization
#' H = diag(p) # penalty matrix
#' type = "absolute" # absolute loss
#' 
#' # Run with the IRLS procedure
#' res = IRLS(Z, Y, lambda, H, type)
#' with(res,GCV_crit(resids,hat_values))
#' with(res,GCV_crit(resids,hat_values,custfun = function(r,h) 
#'     sum(r^2)))
#'     
#' # Run with the ridge function
#' res = ridge(Z, Y, lambda, H)
#' with(res,GCV_crit(resids,hat_values))
#' with(res,GCV_crit(resids,hat_values,custfun = function(r,h) 
#'     sum((r/(1-h))^2)))

GCV_crit = function(resids, hats, custfun=NULL){
  n = length(resids)
  GCVs = rep(NA,7)
  names(GCVs) = c("AIC","GCV","GCV(tr)","BIC","rGCV","rGCV(tr)",
                  "custom")
  GCVs[1:6] = c(
    mean(resids^2)+2*mean(hats),
    mean((resids^2)/((1-hats)^2)),
    mean((resids^2)/((1-mean(hats))^2)),
    mean(resids^2)+log(n)*mean(hats),
    robustbase::scaleTau2(resids/(1-hats), c2 = 5),
    robustbase::scaleTau2(resids/(1-mean(hats)), c2 = 5)
  )
  if(is.null(custfun)){
    return(GCVs[1:6])
  } else {
    GCVs[7] = custfun(resids,hats)
      return(GCVs) 
  }
}

#' k-fold cross-validation for the IRLS procedure
#'
#' Provides the k-fold version of the cross-validation procedure in
#' conjunction with the \link{IRLS} function directly as an argument of the
#' parameter \code{lambda}. 
#'
#' @param lambda A candidate parameter value; non-negative real number.
#'
#' @param Z Data matrix of dimension \code{n}-times-\code{p}, where \code{n} is
#' the number of observations, \code{p} is the dimension.
#'
#' @param Y Vector of responses of length \code{n}.
#'
#' @param H Penalty matrix of size \code{p}-times-\code{p} that
#' is used inside the quadratic term for penalizing estimated parameters.
#' 
#' @param type The type of the loss function used in the minimization problem.
#' Accepted are \code{type="absolute"} for the absolute loss \code{rho(t)=|t|}; 
#' \code{type="square"} for the square loss \code{rho(t)=t^2}; 
#' \code{type="Huber"} for the Huber loss \code{rho(t)=t^2/2} if 
#' \code{|t|<tuning} and \code{rho(t)=tuning*(|t|-tuning/2)} otherwise; and 
#' \code{type="logistic"} for the logistic loss 
#' \code{rho(t)=2*t + 4*log(1+exp(-t))-4*log(2)}.
#'
#' @param k Number of folds to consider, positive integer. By default
#' set to 5.
#' 
#' @param vrs Version of the algorhitm to be used in function \link{IRLS}; 
#' either \code{vrs="C"} for the \code{C++} version, or \code{vrs="R"} for the 
#' \code{R} version. Both should give (nearly) identical results, see 
#' \link{IRLS}.
#'
#' @details Cross-validation based on the median of squared residuals obtained
#' in each fold of the data.
#'
#' @return A numerical of medians of squared residuals for each fold of the 
#' data. A numerical vector of \code{k} non-negative values.
#'
#' @examples
#' n = 50      # sample size
#' p = 10      # dimension of predictors
#' Z = matrix(rnorm(n*p),ncol=p) # design matrix
#' Y = Z[,1]   # response vector
#' lambda = 1  # tuning parameter for penalization
#' H = diag(p) # penalty matrix
#' type = "absolute" # absolute loss
#' 
#' kCV(lambda,Z,Y,H,type,k=5)

kCV = function(lambda,Z,Y,H,type,k=5,vrs="C"){
  # k-fold cross-validation
  vrs = match.arg(vrs, c("C", "R"))
  n = length(Y)
  if(n<k) stop("For k-fold cross-validation we need n>=k.")
  #
  nk = floor(n/k) # number of functions in each batch but the last
  batchi = c(rep(1:k,each=nk),rep(k,n-nk*k)) 
  # batch index for each function
  crit = rep(NA,k)
  for(ki in 1:k){
    bi = (batchi==ki)
    Ze = Z[!bi,] # Z for estimation
    Ye = Y[!bi]  # Y for estimation
    Zt = Z[bi,]  # Z for testing
    Yt = Y[bi]   # Y for testing
    fit.r <- IRLS(Ze,Ye,lambda,H,type,vrs=vrs)
    rest = Yt - Zt%*%fit.r$theta_hat # residuals for the testing part
    # if(fit.r$converged==0) rest = rep(Inf,length(Yt)) # If IRLS did not converge
    crit[ki] = median(rest^2) # robustbase::scaleTau2(rest^2, c2 = 5)
    # criterion for ki-th batch
  }
  return(crit) # criterion for each batch
} 

#' Robust thin-plate splines regression
#'
#' Fits a (potentially robust) thin-plates spline in a scalar-on-function 
#' regression problem with discretely observed predictors. The tuning parameter
#' \code{lambda} is selected using a specified cross-validation criterion.
#'
#' @param X Matrix of observed values of \code{X} of size 
#'  \code{n}-times-\code{p}, one row per observation, columns corresponding to the 
#'  positions in the rows of \code{tobs}.
#'
#' @param Y Vector of responses of length \code{n}.
#'  
#' @param tobs Domain locations for the observed points of \code{X}. Matrix
#'  of size \code{p}-times-\code{d}, one row per domain point.
#' 
#' @param m Order of the thin-plate spline, positive integer.
#'
#' @param type The type of the loss function used in the minimization problem.
#' Accepted are \code{type="absolute"} for the absolute loss \code{rho(t)=|t|}; 
#' \code{type="square"} for the square loss \code{rho(t)=t^2}; 
#' \code{type="Huber"} for the Huber loss \code{rho(t)=t^2/2} if 
#' \code{|t|<tuning} and \code{rho(t)=tuning*(|t|-tuning/2)} otherwise; and 
#' \code{type="logistic"} for the logistic loss 
#' \code{rho(t)=2*t + 4*log(1+exp(-t))-4*log(2)}.
#' 
#' @param jcv A numerical indicator of the cross-validation method used to 
#' select the tuning parameter \code{lambda}. The criteria are always 
#' based on the residuals (\code{resids}) and hat values (\code{hats}) in
#' the fitted models. Possible values are:
#' \itemize{
#'  \item{"all"}{ All the criteria below are considered.}
#'  \item{"AIC"}{ Akaike's information criterion given by 
#'  \code{mean(resids^2)+log(n)*mean(hats)}, where \code{n} is the length of
#'  both \code{resids} and \code{hats}.}
#'  \item{"GCV"}{ Leave-one-out cross-validation criterion given by
#'  \code{mean((resids^2)/((1-hats)^2))}.}
#'  \item{"GCV(tr)"}{ Modified leave-one-out cross-validation criterion 
#'  given by \code{mean((resids^2)/((1-mean(hats))^2))}.}
#'  \item{"BIC"}{ Bayes information criterion given by 
#'  \code{mean(resids^2)+2*mean(hats)}.}
#'  \item{"rGCV"}{ A robust version of \code{GCV} where mean is replaced
#'  by a robust M-estimator of scale of \code{resids/(1-hats)}, see 
#'  \link[robustbase]{scaleTau2} for details.}
#'  \item{"rGCV(tr)"}{ Modified version of a \code{rGCV} given by 
#'  a robust M-estimator of scale of \code{resids/(1-mean(hats))}.}
#'  \item{"custom"}{ The custom criterion given by function \code{custfun}. 
#'  Works only if \code{custfun} is part of the input.}
#'  }
#'  
#' @param sc Scale parameter to be used in the IRLS. By default \code{sc=1}, 
#' that is no scaling is performed.
#' 
#' @param vrs Version of the algorhitm to be used in function \link{ridge}; 
#' either \code{vrs="C"} for the \code{C++} version, or \code{vrs="R"} for the 
#' \code{R} version. Both should give (nearly) identical results, see 
#' \link{IRLS}.
#' 
#' @param plotCV Indicator of whether a plot of the evaluated cross-validation 
#' criteria as a function of \code{lambda} should be given.
#' 
#' @param lambda_grid An optional grid for select \code{lambda} from. By default
#' this is set to be an exponential of a grid of 51 equidistant values
#' in the interval from -28 to -1. 
#'
#' @param custfun A custom function combining the residuals \code{resids} and
#' the hat values \code{hats}. The result of the function must be numeric, see 
#' \link{GCV_crit}.
#' 
#' @param int_weights Indicator whether the integrals functions are to be 
#' approximated only as a mean of function values (\code{int_weights=FALSE}), or
#' whether they should be computed as weighted sums of function values 
#' (\code{int_weights=TRUE}). In the latter case, weights are computed as
#' proportional to the respective areas of the Voronoi tesselation of the domain
#' \code{I} (works for dimension \code{d==1} or \code{d==2}). For dimension 
#' \code{d==1}, this is equivalent to the length of intervals associated 
#' to adjacent observation points. For \code{d>2}, no weighting is performed.
#' 
#' @param I.method Applicable only if \code{int_weights==TRUE}. 
#' Input method for the complete domain \code{I}
#' where the Voronoi tesselation for obtaining the weights is evaluated. 
#' Takes a value \code{"box"}
#' for \code{I} the smallest axes-aligned box in the domain, or \code{"chull"}
#' for \code{I} being a convex hull of points in the domain. By default set to 
#' \code{"chull"}. In dimension \code{d=1}, the two methods \code{"box"}
#' and \code{"chull"} are equivalent.
#' 
#' @param I Applicable only if \code{int_weights==TRUE}. A set of points 
#' specifying the complete domain \code{I} of the functional data.
#' In general, can be a \code{q}-times-\code{d} matrix, where \code{q}
#' is the number of points and \code{d} is the dimension. The matrix \code{I}
#' then specifies the point from which to compute the domain \code{I}: 
#' (a) If \code{method.I=="chull"}, the domain is the convex hull of \code{I}; 
#' (b) If \code{method.I=="box"}, the domain is the smallest axes-aligned box
#' that contains \code{I}. If \code{I} is \code{NULL} (by default), then 
#' \code{I} is taken to be the same as \code{x}. If \code{I.method=="box"}, 
#' \code{I} can be specified also by a pair of real values \code{a<b}, 
#' in which case we take \code{I} to be the axis-aligned sqare \code{[a,b]^d}.
#' 
#' @param resids.in Initialization of the vector of residuals used to launch 
#' the IRLS algorithms. Optional.
#' 
#' @param toler A small positive constant specifying the tolerance level for 
#' terminating the algorithm. The prcedure stops if the maximum absolute 
#' distance between the residuals in the previous iteration and the new 
#' residuals drops below \code{toler}.
#' 
#' @param imax Maximum number of allowed iterations of IRLS. 
#'
#' @param tolerGCV A small positive constant specifying the tolerance level for 
#' terminating the algorithm when the cross-validation is performed. 
#' The prcedure stops if the maximum absolute 
#' distance between the residuals in the previous iteration and the new 
#' residuals drops below \code{tolerGCV}.
#' 
#' @param imaxGCV Maximum number of allowed iterations of IRLS in 
#' cross-validation procedures.
#' 
#' @param echo An indicator whether diagnostic messages should be printed. Use
#' only when modifying the content of the function.
#' 
#' @return The output differs depending whether \code{jcv="all"} or 
#' not. If a specific cross-validation method is selected (that is, 
#' \code{jcv} is not \code{"all"}), a list is returned:
#'  \itemize{
#'  \item{"lambda"}{ The selected tuning parameter \code{lambda} that minimizes
#'  the chosen cross-validation criterion.}
#'  \item{"fitted"}{ A vector of \code{n} fitted values using the tuning 
#'  parameter \code{lambda}.}
#'  \item{"theta_hat"}{ A numerical matrix of size \code{p+1}-times-\code{1} of 
#'  estimated regression coefficients from \link{IRLS}.}
#'  \item{"beta_hat"}{ Estimate of the regression function \code{beta0} 
#'  evaluated at the \code{p} points from \code{tobs}, where \code{X} was
#'  observed.}
#'  \item{"alpha_hat"}{ Estimate of \code{alpha0}, a numerical value.}
#'  \item{"hat_values"}{ Diagonal terms of the (possibly penalized) hat 
#'  matrix of the form \code{Z*solve(t(Z)*W*Z+n*lambda*H)*t(Z)*W}, 
#'  where \code{W} is the diagonal weight matrix in the final iteration 
#'  of \link{IRLS}.}
#'  \item{"weights"}{ The vector of weights given to the observations in the 
#'  final iteration of \link{IRLS}. For squared loss (\code{type="square"})
#'  this gives a vector whose all elements are 2.}
#'  \item{"converged"}{ Indicator whether the \link{IRLS} procedure succefully 
#'  converged. Takes value 1 if IRLS converged, 0 otherwise.}
#' }
#' In case when \code{jcv="all"}, all these values are given for each 
#' cross-validation method considered. For \code{lambda}, \code{alpha_hat},
#' and \code{converged} provides a list of length 6 or 7 (depending on 
#' whether \code{custfun} is specified); for \code{fitted}, \code{beta_hat},
#' \code{hat_values}, and \code{weights} it gives a matrix with 6 or 7 
#' columns, each corresponding to one cross-validation method. 
#'
#' @references
#' Ioannis Kalogridis and Stanislav Nagy. (2023). Robust functional regression 
#' with discretely sampled predictors. 
#' \emph{Under review}.
#'
#' @seealso \link{ts_ridge} for a faster (non-robust) version of
#' this method applied with \code{type="square"}.
#'
#' @examples
#' n = 50      # sample size
#' p = 10      # dimension of predictors
#' X = matrix(rnorm(n*p),ncol=p) # design matrix
#' Y = X[,1]   # response vector
#' tobs = matrix(sort(runif(p)),ncol=1)
#' type = "absolute" # absolute loss
#' 
#' res = ts_reg(X, Y, tobs, m = 2, type = type, jcv = "all", plotCV = TRUE)

ts_reg = function(X, Y, tobs, m, type, jcv = "all", 
                  sc = 1, vrs="C", 
                  plotCV=FALSE, lambda_grid=NULL,
                  custfun=NULL, int_weights=TRUE, 
                  I.method = "chull", I=NULL, 
                  resids.in = rep(1,length(Y)),
                  toler=1e-7, imax=1000,
                  tolerGCV=toler, imaxGCV=imax,
                  echo = FALSE){
  
  jcv = match.arg(jcv,c("all", "AIC", "GCV", "GCV(tr)", "BIC", "rGCV", 
                        "rGCV(tr)", "custom"))
  if(jcv=="all") jcv = 0
  if(jcv=="AIC") jcv = 1
  if(jcv=="GCV") jcv = 2
  if(jcv=="GCV(tr)") jcv = 3
  if(jcv=="BIC") jcv = 4
  if(jcv=="rGCV") jcv = 5
  if(jcv=="rGCV(tr)") jcv = 6
  if(jcv=="custom") jcv = 7
  if(jcv==7 & is.null(custfun)) stop("With custom cross-validation, 
                                    cusfun must be provided.")
  
  # pre-processing for thin-plate splines
  tspr = ts_preprocess(X,tobs,m, int_weights=int_weights, 
                       I.method = I.method, I = I)
  # attach(tspr)
  Z = tspr$Z; H = tspr$H; Q = tspr$Q; Omega = tspr$Omega; Phi = tspr$Phi;
  degs = tspr$degs; M = tspr$M; m = tspr$m; p = tspr$p; d = tspr$d; n = tspr$n;
  
  if(is.null(lambda_grid)){
    # define grid for search for lambda
    rho1 = -28  # search range minimium exp(rho1)
    rho2 = -1   # search range maximum exp(rho2)
    lambda_length = 51
    lambda_grid = exp(c(-Inf,seq(rho1,rho2,length=lambda_length-1)))
    } else {
    if(!is.numeric(lambda_grid)) 
      stop("Grid for lambda values must contain numeric values.")
    if(any(lambda_grid<0)) 
      stop("Grid for lambda values must contain non-negative 
           values.")
    lambda_length = length(lambda_grid)
    }
  GCVfull <- Vectorize(
    function(x) GCV(x,
                    Z = Z, Y = Y, H = H, type=type, 
                    sc = sc, 
                    vrs=vrs,
                    custfun = custfun, 
                    resids.in = resids.in,
                    toler=tolerGCV, imax=imaxGCV))(lambda_grid)
  ncv = nrow(GCVfull)-2
  GCVconverged = GCVfull[ncv+1,]
  GCVic = GCVfull[ncv+2,]
  
  if(echo) print(paste(c("Numbers of iterations in IRLS", 
                       GCVic), collapse=", "))
  
  GCVfull = GCVfull[1:ncv,]
  cvnames = c("AIC","GCV","GCV(tr)","BIC","rGCV","rGCV(tr)",
              "custom")
  if(jcv==0) rownames(GCVfull) = cvnames[1:ncv] # if all the criteria are used 
  
  if(plotCV){
    if(jcv == 0){
      par(mfrow=c(3,2))
      for(i in 1:ncv){
        plot(log(GCVfull[i,])~log(lambda_grid),type="l",
             lwd=2,
             xlab=expression(log(lambda)),
             ylab="CV criterion",
             main = rownames(GCVfull)[i])
        points(log(GCVfull[i,])~log(lambda_grid),
          cex = 1-GCVconverged, col="red", pch=16)
        # abline(h=log(GCVfull[i,1]),lty=2)
        abline(v=log(lambda_grid[which.min(GCVfull[i,])]),lty=2)
      }
      par(mfrow=c(1,1))  
    } else {
      plot(log(GCVfull[jcv,])~log(lambda_grid),type="l",
         lwd=2, xlab=expression(log(lambda)),
         ylab="CV criterion",
         main = cvnames[jcv])
      points(log(GCVfull[jcv,])~log(lambda_grid),
             cex = 1-GCVconverged, col="red", pch=16)
      # abline(h=log(GCVfull[jcv,1]),lty=2)
      abline(v=log(lambda_grid[which.min(GCVfull[jcv,])]),lty=2)
    }
  }
  
  lopt = lambda_grid[apply(GCVfull,1,which.min)]
  
  if(jcv>0){
    lambda = lopt[jcv] # lambda parameter selected
    #
    res = IRLS(Z,Y,lambda,H,type,vrs=vrs,sc=sc, 
               resids.in = resids.in, 
               toler=toler, imax=imax)
    res_ts = transform_theta(res$theta_hat,tspr)
    return(list(lambda = lambda,
                fitted = res$fitted, 
                theta_hat = res$theta_hat,
                beta_hat = res_ts$beta_hat, 
                alpha_hat = (res$theta_hat)[1],
                hat_values = res$hat_values,
                weights = res$weights, 
                converged = res$converged))
  } else {
    n = length(Y)
    fitted = matrix(nrow=n,ncol=ncv)
    betahat = matrix(nrow=nrow(tobs), ncol=ncv)
    thetahat = matrix(nrow=nrow(tobs)+1, ncol=ncv)
    alphahat = rep(NA,ncv)
    hatvals = matrix(nrow=n, ncol=ncv)
    weights = matrix(nrow=n, ncol=ncv)
    converged = rep(NA,ncv)
    for(jcv in 1:ncv){
      lambda = lopt[jcv] # lambda parameter selected
      #
      res = IRLS(Z,Y,lambda,H,type,vrs=vrs,sc=sc, 
                 resids.in = resids.in,
                 toler=toler, imax=imax)
      res_ts = transform_theta(res$theta_hat,tspr)
      fitted[,jcv] = res$fitted
      thetahat[,jcv] = res$theta_hat
      betahat[,jcv] = res_ts$beta_hat
      alphahat[jcv] = (res$theta_hat)[1]
      hatvals[,jcv] = res$hat_values
      weights[,jcv] = res$weights
      converged[jcv] = res$converged
    }
    return(list(lambda = lopt,
                fitted = fitted, 
                theta_hat = thetahat,
                beta_hat = betahat,
                alpha_hat = alphahat,
                hat_values = hatvals,
                weights = weights, 
                converged = converged))
  }
}

#' Robust thin-plate splines location estmation for functional data
#'
#' Provides a (potentially robust) thin-plates spline location estimator for 
#' discretely observed functional data. The functional data do not need to be 
#' observed on a common grid (that is, \code{NA} values in the matrix \code{Y}
#' below are allowed). The tuning parameter \code{lambda} is selected using a 
#' specified cross-validation criterion.
#'
#' @param Y Matrix of observed values of functional data \code{Y} of size 
#'  \code{p}-times-\code{n}, one column per observation, rows corresponding to 
#'  the positions in the rows of \code{tobs}. Functions observed on different 
#'  grids can be provided by including missing values \code{NA} in the matrix.
#'
#' @param tobs Domain locations for the observed points of \code{Y}. Matrix
#'  of size \code{p}-times-\code{d}, one row per domain point.
#' 
#' @param r Order of the thin-plate spline, positive integer.
#'
#' @param type The type of the loss function used in the minimization problem.
#' Accepted are \code{type="absolute"} for the absolute loss \code{rho(t)=|t|}; 
#' \code{type="square"} for the square loss \code{rho(t)=t^2}; 
#' \code{type="Huber"} for the Huber loss \code{rho(t)=t^2/2} if 
#' \code{|t|<tuning} and \code{rho(t)=tuning*(|t|-tuning/2)} otherwise; and 
#' \code{type="logistic"} for the logistic loss 
#' \code{rho(t)=2*t + 4*log(1+exp(-t))-4*log(2)}.
#' 
#' @param jcv A numerical indicator of the cross-validation method used to 
#' select the tuning parameter \code{lambda}. The criteria are always 
#' based on the residuals (\code{resids}) and hat values (\code{hats}) in
#' the fitted models. Possible values are:
#' \itemize{
#'  \item{"all"}{ All the criteria below are considered.}
#'  \item{"AIC"}{ Akaike's information criterion given by 
#'  \code{mean(resids^2)+log(n)*mean(hats)}, where \code{n} is the length of
#'  both \code{resids} and \code{hats}.}
#'  \item{"GCV"}{ Leave-one-out cross-validation criterion given by
#'  \code{mean((resids^2)/((1-hats)^2))}.}
#'  \item{"GCV(tr)"}{ Modified leave-one-out cross-validation criterion 
#'  given by \code{mean((resids^2)/((1-mean(hats))^2))}.}
#'  \item{"BIC"}{ Bayes information criterion given by 
#'  \code{mean(resids^2)+2*mean(hats)}.}
#'  \item{"rGCV"}{ A robust version of \code{GCV} where mean is replaced
#'  by a robust M-estimator of scale of \code{resids/(1-hats)}, see 
#'  \link[robustbase]{scaleTau2} for details.}
#'  \item{"rGCV(tr)"}{ Modified version of a \code{rGCV} given by 
#'  a robust M-estimator of scale of \code{resids/(1-mean(hats))}.}
#'  \item{"custom"}{ The custom criterion given by function \code{custfun}. 
#'  Works only if \code{custfun} is part of the input.}
#'  }
#'  
#' @param vrs Version of the algorhitm to be used in function \link{ridge}; 
#' either \code{vrs="C"} for the \code{C++} version, or \code{vrs="R"} for the 
#' \code{R} version. Both should give (nearly) identical results, see 
#' \link{IRLS}.
#' 
#' @param method A method for estimating the fit. Possible options are 
#' \code{"IRLS"} for the IRLS algorithm, or \code{"ridge"} for ridge regression.
#' Ridge is applicable only if \code{type="square"}; this method is much faster,
#' but provides only a non-robust fit.
#' 
#' @param plotCV Indicator of whether a plot of the evaluated cross-validation 
#' criteria as a function of \code{lambda} should be given.
#' 
#' @param lambda_grid An optional grid for select \code{lambda} from. By default
#' this is set to be an exponential of a grid of \code{lambda_length} 
#' equidistant values in the interval from -28 to -1.
#' 
#' @param lambda_length Number of elements in the grid of values \code{lambda}. 
#' By default chosen to be 51. 
#'
#' @param custfun A custom function combining the residuals \code{resids} and
#' the hat values \code{hats}. The result of the function must be numeric, see 
#' \link{GCV_crit}.
#' 
#' @return The output differs depending whether \code{jcv="all"} or 
#' not. If a specific cross-validation method is selected (that is, 
#' \code{jcv} is not \code{"all"}), a list is returned:
#'  \itemize{
#'  \item{"lambda"}{ The selected tuning parameter \code{lambda} that minimizes
#'  the chosen cross-validation criterion.}
#'  \item{"fitted"}{ A vector of fitted values using the tuning 
#'  parameter \code{lambda}. The length of this vector equals \code{m}, the 
#'  number of non-missing values in the matrix \code{Y}. If \code{Y} does not
#'  contain missing values, the length of this vector is \code{n*p}.}
#'  \item{"theta_hat"}{ A numerical matrix of size \code{p}-times-\code{1} of 
#'  estimated regression coefficients from \link{IRLS}.}
#'  \item{"beta_hat"}{ The final location estimate \code{beta} 
#'  evaluated at the \code{p} points from \code{tobs}, where \code{Y} was
#'  observed. A numerical vector of length \code{p}.}
#'  \item{"hat_values"}{ Diagonal terms of the (possibly penalized and weighted)
#'  hat matrix of the form \code{Z*solve(t(Z)*W*Z+n*lambda*H)*t(Z)*W}, 
#'  where \code{W} is the diagonal weight matrix in the final iteration 
#'  of \link{IRLS}. A numerical vector of length \code{m}.}
#'  \item{"weights"}{ The vector of weights given to the observations in the 
#'  final iteration of \link{IRLS}. A numerical vector of length \code{m}.}
#'  \item{"converged"}{ Indicator whether the \link{IRLS} procedure succefully 
#'  converged. Takes value 1 if IRLS converged, 0 otherwise.}
#' }
#' In case when \code{jcv="all"}, all these values are given for each 
#' cross-validation method considered. For \code{lambda} and \code{converged} 
#' provides a list of length 6 or 7 (depending on whether \code{custfun} is 
#' specified); for \code{fitted}, \code{beta_hat},
#' \code{hat_values}, and \code{weights} it gives a matrix with 6 or 7 
#' columns, each corresponding to one cross-validation method. 
#'
#' @references
#' Ioannis Kalogridis and Stanislav Nagy. (2025). Robust multidimensional 
#' location estimation from discretely sampled functional data. 
#' \emph{Under review}.
#'
#' @examples
#' d = 1                              # dimension of domain
#' m = 50                             # number of observation points
#' tobs = matrix(runif(m*d), ncol=d)  # location of obsevation points
#' n = 20                             # sample size
#' truemeanf = function(x)            # true location function 
#'   cos(4*pi*x[1])
#' truemean = apply(tobs,1,truemeanf) # discretized values of the true location
#' Y = replicate(n, truemean + rnorm(m)) # a matrix of functional data, size m*n
#' 
#' # introduce NAs
#' obsprob = 0.2                      # probability of a point being observed
#' B = matrix(rbinom(n*m,1,obsprob),ncol=n)
#' for(i in 1:m) for(j in 1:n) if(B[i,j]==0) Y[i,j] = NA
#' 
#' res = ts_location(Y, tobs=tobs, r=2, type="square", plotCV=TRUE)

ts_location = function(Y, tobs, r, type, 
                       jcv = "all", vrs="C", method="IRLS",
                       plotCV=FALSE, lambda_grid=NULL,
                       lambda_length = 51, custfun=NULL,
                       resids.in = rep(1,length(Y)),
                       toler=1e-7, imax=1000,
                       tolerGCV=toler, imaxGCV=imax,
                       echo = FALSE){
  
  method = match.arg(method,c("IRLS", "ridge"))
  type = match.arg(type,c("square","absolute","Huber","logistic"))
  if(method=="ridge" & type!="square") 
    stop("method 'ridge' available only for type 'square'.")
  
  jcv = match.arg(jcv,c("all", "AIC", "GCV", "GCV(tr)", "BIC", "rGCV", 
                        "rGCV(tr)", "custom"))
  if(jcv=="all") jcv = 0
  if(jcv=="AIC") jcv = 1
  if(jcv=="GCV") jcv = 2
  if(jcv=="GCV(tr)") jcv = 3
  if(jcv=="BIC") jcv = 4
  if(jcv=="rGCV") jcv = 5
  if(jcv=="rGCV(tr)") jcv = 6
  if(jcv=="custom") jcv = 7
  if(jcv==7 & is.null(custfun)) stop("With custom cross-validation, 
                                    cusfun must be provided.")
  
  # pre-processing for thin-plate splines
  tspr = ts_preprocess_location(Y, tobs, r)
  # attach(tspr)
  Y = tspr$Y; 
  Z = tspr$Z; H = tspr$H; Q = tspr$Q; Omega = tspr$Omega; Phi = tspr$Phi;
  w = tspr$w;
  degs = tspr$degs; M = tspr$M; r = tspr$r; 
  p = tspr$p; d = tspr$d; n = tspr$n;
  
  if(is.null(lambda_grid)){
    # define grid for search for lambda
    rho1 = -28  # search range minimium exp(rho1)
    rho2 = -1   # search range maximum exp(rho2)
    if(is.null(lambda_length)) lambda_length = 51
    lambda_grid = exp(c(-Inf,seq(rho1,rho2,length=lambda_length-1)))
  } else {
    if(!is.numeric(lambda_grid)) 
      stop("Grid for lambda values must contain numeric values.")
    if(any(lambda_grid<0)) 
      stop("Grid for lambda values must contain non-negative 
           values.")
    lambda_length = length(lambda_grid)
  }
  
  GCVfull <- Vectorize(
    function(x) GCV_location(x,
                    Z = Z, Y = Y, H = H, type=type, w=w, vrs=vrs,
                    method = method,
                    custfun = custfun,
                    resids.in = resids.in,
                    toler=tolerGCV, imax=imaxGCV))(lambda_grid)
  ncv = nrow(GCVfull)-2
  GCVconverged = GCVfull[ncv+1,]
  GCVic = GCVfull[ncv+2,]
  
  if(echo) print(paste(c("Numbers of iterations in IRLS", 
                         GCVic), collapse=", "))
  
  GCVfull = GCVfull[1:ncv,]
  cvnames = c("AIC","GCV","GCV(tr)","BIC","rGCV","rGCV(tr)",
              "custom")
  if(jcv==0) rownames(GCVfull) = cvnames[1:ncv] # if all the criteria are used 
  
  if(plotCV){
    if(jcv == 0){
      par(mfrow=c(3,2))
      for(i in 1:ncv){
        plot(log(GCVfull[i,])~log(lambda_grid),type="l",
             lwd=2,
             xlab=expression(log(lambda)),
             ylab="CV criterion",
             main = rownames(GCVfull)[i])
        points(log(GCVfull[i,])~log(lambda_grid),
               cex = 1-GCVconverged, col="red", pch=16)
        abline(v=log(lambda_grid[which.min(GCVfull[i,])]),lty=2)
      }
      par(mfrow=c(1,1))  
    } else {
      plot(log(GCVfull[jcv,])~log(lambda_grid),type="l",
           lwd=2, xlab=expression(log(lambda)),
           ylab="CV criterion",
           main = cvnames[jcv])
      points(log(GCVfull[jcv,])~log(lambda_grid),
             cex = 1-GCVconverged, col="red", pch=16)
      abline(v=log(lambda_grid[which.min(GCVfull[jcv,])]),lty=2)
    }
  }
  
  lopt = lambda_grid[apply(GCVfull,1,which.min)]
  
  if(jcv>0){
    lambda = lopt[jcv] # lambda parameter selected
    #
    if(method=="IRLS") 
      res = IRLS(Z,Y,lambda,H,type=type,w=w,vrs=vrs,sc=1, 
                 resids.in = resids.in, 
                 toler=toler, imax=imax)
    if(method=="ridge")
      res = ridge(Z,Y,lambda,H,w=w,vrs=vrs)
    res_ts = transform_theta_location(res$theta_hat,tspr)
    if(method=="IRLS") return(
      list(lambda = lambda,
           fitted = res$fitted, 
           theta_hat = res$theta_hat,
           beta_hat = res_ts$beta_hat,
           gamma_hat = res_ts$gamma_hat,
           delta_hat = res_ts$delta_hat,
           hat_values = res$hat_values,
           weights = res$weights, 
           converged = res$converged))
    if(method=="ridge") return(
      list(lambda = lambda,
           fitted = res$fitted, 
           theta_hat = res$theta_hat,
           beta_hat = res_ts$beta_hat,
           gamma_hat = res_ts$gamma_hat,
           delta_hat = res_ts$delta_hat,
           hat_values = res$hat_values))
  } else {
    n = length(Y)
    fitted = matrix(nrow=n,ncol=ncv)
    betahat = matrix(nrow=nrow(tobs), ncol=ncv)
    gammahat = matrix(nrow=nrow(tobs),ncol=ncv)
    deltahat = matrix(nrow=choose(d+r-1,d),ncol=ncv)
    thetahat = matrix(nrow=nrow(tobs), ncol=ncv)
    hatvals = matrix(nrow=n, ncol=ncv)
    weights = matrix(nrow=n, ncol=ncv)
    converged = rep(NA,ncv)
    for(jcv in 1:ncv){
      lambda = lopt[jcv] # lambda parameter selected
      #
      if(method=="IRLS")
        res = IRLS(Z,Y,lambda,H,type=type,w=w,vrs=vrs,sc=1, 
                   resids.in = resids.in, 
                   toler=toler, imax=imax)
      if(method=="ridge")
        res = ridge(Z,Y,lambda,H,w=w,vrs=vrs)
      res_ts = transform_theta_location(res$theta_hat,tspr)
      fitted[,jcv] = res$fitted
      thetahat[,jcv] = res$theta_hat
      betahat[,jcv] = res_ts$beta_hat
      gammahat[,jcv] = res_ts$gamma_hat
      deltahat[,jcv] = res_ts$delta_hat
      hatvals[,jcv] = res$hat_values
      if(method=="IRLS"){
        weights[,jcv] = res$weights
        converged[jcv] = res$converged
      }
    }
    return(list(lambda = lopt,
                fitted = fitted, 
                theta_hat = thetahat,
                beta_hat = betahat,
                gamma_hat = gammahat,
                delta_hat = deltahat,
                hat_values = hatvals,
                weights = weights, 
                converged = converged))
  }
}

#' (Non-robust) Thin-plate splines regression
#'
#' Fits a (non-robust) thin-plates spline in a scalar-on-function 
#' regression problem with discretely observed predictors. The tuning parameter
#' \code{lambda} is selected using a specified cross-validation criterion.
#'
#' @param X Matrix of observed values of \code{X} of size 
#'  \code{n}-times-{p}, one row per observation, columns corresponding to the 
#'  positions in the rows of \code{tobs}.
#'
#' @param Y Vector of responses of length \code{n}.
#'  
#' @param tobs Domain locations for the observed points of \code{X}. Matrix
#'  of size \code{p}-times-\code{d}, one row per domain point.
#' 
#' @param m Order of the thin-plate spline, positive integer.
#'
#' @param jcv A numerical indicator of the cross-validation method used to 
#' select the tuning parameter \code{lambda}. The criteria are always 
#' based on the residuals (\code{resids}) and hat values (\code{hats}) in
#' the fitted models. Possible values are:
#' \itemize{
#'  \item{"all"}{ All the criteria below are considered.}
#'  \item{"AIC"}{ Akaike's information criterion given by 
#'  \code{mean(resids^2)+log(n)*mean(hats)}, where \code{n} is the length of
#'  both \code{resids} and \code{hats}.}
#'  \item{"GCV"}{ Leave-one-out cross-validation criterion given by
#'  \code{mean((resids^2)/((1-hats)^2))}.}
#'  \item{"GCV(tr)"}{ Modified leave-one-out cross-validation criterion 
#'  given by \code{mean((resids^2)/((1-mean(hats))^2))}.}
#'  \item{"BIC"}{ Bayes information criterion given by 
#'  \code{mean(resids^2)+2*mean(hats)}.}
#'  \item{"rGCV"}{ A robust version of \code{GCV} where mean is replaced
#'  by a robust M-estimator of scale of \code{resids/(1-hats)}, see 
#'  \link[robustbase]{scaleTau2} for details.}
#'  \item{"rGCV(tr)"}{ Modified version of a \code{rGCV} given by 
#'  a robust M-estimator of scale of \code{resids/(1-mean(hats))}.}
#'  \item{"custom"}{ The custom criterion given by function \code{custfun}. 
#'  Works only if \code{custfun} is part of the input.}
#'  }
#'  
#' @param vrs Version of the algorhitm to be used in function \link{ridge}; 
#' either \code{vrs="C"} for the \code{C++} version, or \code{vrs="R"} for the 
#' \code{R} version. Both should give (nearly) identical results, see 
#' \link{IRLS}.
#' 
#' @param plotCV Indicator of whether a plot of the evaluated cross-validation 
#' criteria as a function of \code{lambda} should be given.
#' 
#' @param lambda_grid An optional grid for select \code{lambda} from. By default
#' this is set to be an exponential of a grid of 51 equidistant values
#' in the interval from -28 to -1. 
#'
#' @param custfun A custom function combining the residuals \code{resids} and
#' the hat values \code{hats}. The result of the function must be numeric, see 
#' \link{GCV_crit}.
#' 
#' @param int_weights Indicator whether the integrals functions are to be 
#' approximated only as a mean of function values (\code{int_weights=FALSE}), or
#' whether they should be computed as weighted sums of function values 
#' (\code{int_weights=TRUE}). In the latter case, weights are computed as
#' proportional to the respective areas of the Voronoi tesselation of the domain
#' \code{I} (works for dimension \code{d==1} or \code{d==2}). For dimension 
#' \code{d==1}, this is equivalent to the length of intervals associated 
#' to adjacent observation points. For \code{d>2}, no weighting is performed.
#' 
#' @param I.method Applicable only if \code{int_weights==TRUE}. 
#' Input method for the complete domain \code{I}
#' where the Voronoi tesselation for obtaining the weights is evaluated. 
#' Takes a value \code{"box"}
#' for \code{I} the smallest axes-aligned box in the domain, or \code{"chull"}
#' for \code{I} being a convex hull of points in the domain. By default set to 
#' \code{"chull"}. In dimension \code{d=1}, the two methods \code{"box"}
#' and \code{"chull"} are equivalent.
#' 
#' @param I Applicable only if \code{int_weights==TRUE}. A set of points 
#' specifying the complete domain \code{I} of the functional data.
#' In general, can be a \code{q}-times-\code{d} matrix, where \code{q}
#' is the number of points and \code{d} is the dimension. The matrix \code{I}
#' then specifies the point from which to compute the domain \code{I}: 
#' (a) If \code{method.I=="chull"}, the domain is the convex hull of \code{I}; 
#' (b) If \code{method.I=="box"}, the domain is the smallest axes-aligned box
#' that contains \code{I}. If \code{I} is \code{NULL} (by default), then 
#' \code{I} is taken to be the same as \code{x}. If \code{I.method=="box"}, 
#' \code{I} can be specified also by a pair of real values \code{a<b}, 
#' in which case we take \code{I} to be the axis-aligned sqare \code{[a,b]^d}.
#' 
#' @details Function gives a faster (non-iterative) version of the solution
#' of \link{ts_reg} when \code{type="square"} is used. This corresponds to 
#' the ridge version of an estimator.
#'
#' @return The output differs depending whether \code{jcv="all"} or 
#' not. If a specific cross-validation method is selected (that is, 
#' \code{jcv} is not \code{"all"}), a list is returned:
#'  \itemize{
#'  \item{"lambda"}{ The selected tuning parameter \code{lambda} that minimizes
#'  the chosen cross-validation criterion.}
#'  \item{"fitted"}{ A vector of \code{n} fitted values using the tuning 
#'  parameter \code{lambda}.}
#'  \item{"theta_hat"}{ A numerical matrix of size \code{p+1}-times-\code{1} of 
#'  estimated regression coefficients from \link{ridge}.}
#'  \item{"beta_hat"}{ Estimate of the regression function \code{beta0} 
#'  evaluated at the \code{p} points from \code{tobs}, where \code{X} was
#'  observed.}
#'  \item{"alpha_hat"}{ Estimate of \code{alpha0}, a numerical value.}
#'  \item{"hat_values"}{ Diagonal terms of the (possibly penalized) hat 
#'  matrix of the form \code{Z*solve(t(Z)*W*Z+n*lambda*H)*t(Z)*W}, 
#'  where \code{W} is the diagonal weight matrix in the final iteration 
#'  of \link{IRLS}.}
#' }
#' In case when \code{jcv="all"}, all these values are given for each 
#' cross-validation method considered. For \code{lambda} and \code{alpha_hat},
#' provides a list of length 6 or 7 (depending on 
#' whether \code{custfun} is specified); for \code{fitted}, \code{beta_hat},
#' and \code{hat_values} it gives a matrix with 6 or 7 
#' columns, each corresponding to one cross-validation method. 
#'
#' @seealso \link{ts_reg} for a robust version of this method.
#'
#' @references
#' Ioannis Kalogridis and Stanislav Nagy. (2023). Robust functional regression 
#' with discretely sampled predictors. 
#' \emph{Under review}.
#'
#' @examples
#' n = 50      # sample size
#' p = 10      # dimension of predictors
#' X = matrix(rnorm(n*p),ncol=p) # design matrix
#' Y = X[,1]   # response vector
#' tobs = matrix(sort(runif(p)),ncol=1)
#' 
#' res = ts_ridge(X, Y, tobs, m = 2, jcv = "all", plotCV = TRUE)

ts_ridge = function(X, Y, tobs, m, jcv = "all", vrs="C", 
                  plotCV=FALSE, lambda_grid=NULL,
                  custfun=NULL, int_weights=TRUE, 
                  I.method = "chull", I=NULL){
  jcv = match.arg(jcv,c("all", "AIC", "GCV", "GCV(tr)", "BIC", "rGCV", 
                        "rGCV(tr)", "custom"))
  if(jcv=="all") jcv = 0
  if(jcv=="AIC") jcv = 1
  if(jcv=="GCV") jcv = 2
  if(jcv=="GCV(tr)") jcv = 3
  if(jcv=="BIC") jcv = 4
  if(jcv=="rGCV") jcv = 5
  if(jcv=="rGCV(tr)") jcv = 6
  if(jcv=="custom") jcv = 7
  if(jcv==7 & is.null(custfun)) stop("With custom cross-validation, 
                                    cusfun must be provided.")
  
  # pre-processing for thin-plate splines
  tspr = ts_preprocess(X,tobs,m, int_weights=int_weights, 
                       I.method = I.method, I = I)
  # attach(tspr)
  Z = tspr$Z; H = tspr$H; Q = tspr$Q; Omega = tspr$Omega; Phi = tspr$Phi;
  degs = tspr$degs; M = tspr$M; m = tspr$m; p = tspr$p; d = tspr$d; n = tspr$n;
  
  if(is.null(lambda_grid)){
    # define grid for search for lambda
    rho1 = -28  # search range minimium exp(rho1)
    rho2 = -1   # search range maximum exp(rho2)
    lambda_length = 51
    lambda_grid = exp(c(-Inf,seq(rho1,rho2,length=lambda_length-1)))
  } else {
    if(!is.numeric(lambda_grid)) 
      stop("Grid for lambda values must contain numeric values.")
    if(any(lambda_grid<0)) 
      stop("Grid for lambda values must contain non-negative 
           values.")
    lambda_length = length(lambda_grid)
  }
  GCVfull <- Vectorize(
    function(x) GCV_ridge(x,
                    Z = Z, Y = Y, H = H, vrs=vrs,
                    custfun = custfun))(lambda_grid)
  ncv = nrow(GCVfull)
  cvnames = c("AIC","GCV","GCV(tr)","BIC","rGCV","rGCV(tr)",
              "custom")
  if(jcv==0) rownames(GCVfull) = cvnames[1:ncv]  
  
  if(plotCV){
    if(jcv == 0){
      par(mfrow=c(3,2))
      for(i in 1:ncv){
        plot(log(GCVfull[i,])~log(lambda_grid),type="l",
             lwd=2,
             xlab=expression(log(lambda)),
             ylab="CV criterion",
             main=rownames(GCVfull)[i])
        # abline(h=log(GCVfull[i,1]),lty=2)
        abline(v=log(lambda_grid[which.min(GCVfull[i,])]),lty=2)
      }
      par(mfrow=c(1,1))  
    } else {
      plot(log(GCVfull[jcv,])~log(lambda_grid),type="l",
           lwd=2, xlab=expression(log(lambda)),
           ylab="CV criterion",
           main = cvnames[jcv])
      # abline(h=log(GCVfull[jcv,1]),lty=2)
      abline(v=log(lambda_grid[which.min(GCVfull[jcv,])]),lty=2)
    }
  }
  lopt = lambda_grid[apply(GCVfull,1,which.min)]
  
  if(jcv>0){
    lambda = lopt[jcv] # lambda parameter selected
    #
    res = ridge(Z,Y,lambda,H,vrs=vrs)
    res_ts = transform_theta(res$theta_hat,tspr)
    return(list(lambda = lambda,
                fitted = res$fitted, 
                theta_hat = res$theta_hat,
                beta_hat = res_ts$beta_hat, 
                alpha_hat = (res$theta_hat)[1],
                hat_values = res$hat_values,
                resids = res$resids))
  } else {
    n = length(Y)
    fitted = matrix(nrow=n,ncol=ncv)
    thetahat = matrix(nrow=nrow(tobs)+1, ncol=ncv)
    betahat = matrix(nrow=nrow(tobs), ncol=ncv)
    alphahat = rep(NA,ncv)
    hatvals = matrix(nrow=n, ncol=ncv)
    resids = matrix(nrow=n, ncol=ncv)
    for(jcv in 1:ncv){
      lambda = lopt[jcv] # lambda parameter selected
      #
      res = ridge(Z,Y,lambda,H,vrs=vrs)
      res_ts = transform_theta(res$theta_hat,tspr)
      fitted[,jcv] = res$fitted
      thetahat[,jcv] = res$theta_hat
      betahat[,jcv] = res_ts$beta_hat
      alphahat[jcv] = (res$theta_hat)[1]
      hatvals[,jcv] = res$hat_values
      resids[,jcv] = res$resids
    }
    return(list(lambda = lopt,
                fitted = fitted,
                theta_hat = thetahat,
                beta_hat = betahat,
                alpha_hat = alphahat,
                hat_values = hatvals,
                resids = resids))
  }
}

#' Interpolating functional data using thin-plate splines
#'
#' Given a discretely (and possibly) irregularly observed sample of functional
#' data, this function performs thin-plate spline interpolation of each 
#' functional datum. As an output, a matrix or function values of densely
#' observed interpolated functional data in a common grid of the domain 
#' is obtained.
#' 
# p.out = number of function values in the interpolated function in each d
# I = c(a,b) for a complete domain of the form of the square [a,b]^d
#'
#'
#' @param Xtobs A list of length \code{n}, each element of \code{Xtobs} is a 
#' list with two elements corresponding to the \code{i}-th function: 
#' (a) \code{X} A numerical vector of length \code{p} of observed values of 
#' the \code{i}-th function, where \code{p} is the number of observations for
#' this function. (b) \code{tobs} Domain locations for the observed points of 
#' the \code{i}-th function corresponding to \code{X}. Matrix of size 
#' \code{p}-times-\code{d}, one row per domain point. Here, \code{d} is the 
#' dimension of the domain of \code{X}.
#' 
#' @param r Order of the thin-plate spline, positive integer.
#' 
#' @param p.out Number of equi-distant points in which the densely observed
#' thin-plate spline interpolation of each function should be evaluated. In
#' dimension \code{d}, a total of \code{p.out^d} points are evaluated in the 
#' hyper-cube \code{[I[1],I[2]]^d}, and at these points, 
#' each function is evaluated.
#' 
#' @param I Two numbers \code{I[1]<I[2]} determining the box 
#' \code{[I[1],I[2]]^d} where all the interpolated functions are evaluated. 
#' 
#' @param solve.method Indicator of which solver of systems of linear equations
#' to use for the inversion of the matrices in the interpolation procedure. 
#' Possible options are \code{"C"} for the solver from \code{Armadillo} library
#' in \code{C++}, or \code{R} for the function \link[base]{solve} from \code{R}. 
#' 
#' @return A list of values:
#' \itemize{
#'  \item{"X"}{ A matrix of size \code{n}-times-\code{p.out^d} of the function
#'  values of the densely evaluated interpolated data. One function per row.}
#'  \item{"tobs"}{ A matrix of size \code{p.out^d}-times-\code{d} with all the
#'  coodinates of the points where the function values are evaluated.
#'  }
#'  }
#'
#' @examples
#' 
#' # Construct the irregularly observed functional data
#' Xtobs = list()
#' 
#' # Number of observations and dimension of the domain
#' n = 10
#' d = 1
#'
#'# Generating functions as random noise
#'for(i in 1:n){
#'  p = 10 + rpois(1, 5) # Number of observed points for the i-th function
#'  tobs = matrix(runif(p*d),ncol=d) # Points in the domain
#'  X = rnorm(p,sd=0.05) # Observed function values
#'  Xtobs[[i]] = list(X=X, tobs=tobs)
#'}
#'  
#'# Thin-plate spline interpolation of order r=4
#'intr = ts_interpolate(Xtobs, r = 4)
#'  
#'# Visalization of the i-th function
#'i = 1
#'if(d==1){
#'  plot(Xtobs[[i]]$X~Xtobs[[i]]$tobs[,1])  
#'  lines(intr$X[i,]~intr$tobs[,1])
#'}
#'if(d==2){
#'  rgl::plot3d(Xtobs[[i]]$tobs[,1], Xtobs[[i]]$tobs[,2], Xtobs[[i]]$X)
#'  rgl::points3d(intr$tobs[,1], intr$tobs[,2], intr$X[i,],col="red",add=TRUE)
#'}

ts_interpolate = function(Xtobs, r, p.out = 101, I=c(0,1), 
                          solve.method=c("C","R")){
  
  solve.method = match.arg(solve.method)
  n = length(Xtobs)
  d = ncol(Xtobs[[1]]$tobs)
  
  tobs.out1 = seq(I[1],I[2],length=p.out)
  tobs.out = as.matrix(expand.grid(replicate(d, tobs.out1, simplify=FALSE)))
  p.out = nrow(tobs.out)  
  res = matrix(nrow=n, ncol=p.out)
  
  for(i in 1:n){
    X = Xtobs[[i]]$X
    tobs = Xtobs[[i]]$tobs
    p = length(X)
    if(d!=ncol(tobs))
      stop("Not all functions have the same dimension of the domain")
    
    if(p!=nrow(tobs)) 
      stop("Number of columns of X must equal number of rows of t")
    
    M = choose(r+d-1,d)
    if(p-M<=0) stop(paste("p must be larger than",M))
    if(2*r<=d) stop(paste("r must be larger than",ceiling(d/2)))
    
    # Monomials phi of order <r
    allcom = expand.grid(replicate(d, 0:(r-1), simplify=FALSE))
    degs = as.matrix(allcom[rowSums(allcom)<r,,drop=FALSE])
    M = nrow(degs)
    if(M!=choose(r+d-1,d)) stop("Error in degrees of polynomials")
    
    # Fast matrix of Euclidean distances between tobs
    Em = matrix(sqrt(rowSums(
      apply(tobs,2,function(x) outer(x,x,"-")^2))),nrow=nrow(tobs))
    
    # Matrix Omega for the penalty term
    Omega = eta(Em,d,r)
    
    # Matrix Phi, monomials evaluated at tobs
    Phi = apply(degs,1,function(x) apply(t(tobs)^x,2,prod))
    
    A = rbind(cbind(Omega, Phi),cbind(t(Phi), matrix(0, nrow=M, ncol=M)))
    b = c(X,rep(0,M))
    if(solve.method=="R") coefs = solve(A,b, tol = 1e-21)
    if(solve.method=="C") coefs = c(solveC(A,matrix(b,ncol=1)))
    #
    gamma = coefs[1:p]
    delta = coefs[-(1:p)]
    
    # Matrices Omega and Phi for evaluating interpolated function
    Em.out = distnAB(tobs.out,tobs,p.out,p,d)
    Omega.out = eta(Em.out,d,r)
    Phi.out = apply(degs,1,function(x) apply(t(tobs.out)^x,2,prod))
    #
    res[i,] = Omega.out%*%gamma + Phi.out%*%delta
  }
  return(list(X=res, tobs = tobs.out))
}

#' Quantile regression based on FPCA of Kato (2012)
#'
#' Fits a (robust) quantile regression estimator in a scalar-on-function 
#' regression problem with discretely observed predictors. The estimator is 
#' based on quantile regression applied in the context of functional principal
#' components. The tuning parameter \code{J} (the number of principal 
#' compontents chosen) is selected using cross-validation.
#'
#' @param X Matrix of observed values of \code{X} of size 
#'  \code{n}-times-{p}, one row per observation, columns corresponding to the 
#'  positions in the rows of \code{t}.
#'
#' @param Y Vector of responses of length \code{n}.
#'  
#' @param t Domain locations for the observed points of \code{X}. A numerical
#'  vector of length \code{p}.
#' 
#' @param tau Order of the quantile to be used in the estimator. By default, 
#' \code{tau=0.5}, which corresponds to the regression median.
#' 
#' @param Kmax Maximum number of functional principal components to be 
#' considered. By default, set to \code{Kmax=10}.
#' 
#' @param n_fine Number of equi-spaced points in the domain for the complete
#' function. By default, set to \code{n_fine=200}.
#'
#' @return A list composed of
#' \itemize{
#'  \item{"fitted_Y"}{ A vector of \code{n} fitted values.}
#'  \item{"alpha_hat"}{ Estimate of \code{alpha0}, a numerical value.}
#'  \item{"beta_hat"}{ Estimate of the regression function \code{beta0} 
#'  evaluated at the \code{p} points from \code{tobs}, where \code{X} was
#'  observed.}
#'  \item{"gcv_vals"}{ Observed values of the generalized cross-validation 
#'  criteria for the selection of the dimension. A numerical vector of length
#'  \code{Kmax}.}
#'  \item{"best_m"}{ Selected dimension using generalized cross-validation.}
#' }
#'
#' @references
#' Kato, K. (2012). Estimation in functional linear quantile regression. 
#' \emph{The Annals of Statistics}, 40(6), 3108-3136.
#'
#' @examples
#' n = 50      # sample size
#' p = 10      # dimension of predictors
#' X = matrix(rnorm(n*p),ncol=p) # design matrix
#' Y = X[,1]   # response vector
#' tobs = sort(runif(p)) 
#' 
#' (res = qr_fpca_piecewise_fine(X, Y, tobs))

qr_fpca_piecewise_fine <- function(X, Y, t, tau = 0.5, K_max = 10, n_fine = 200) {
  
  # Piecewise constant interpolating function 

  piecewise_constant <- function(t_obs, x_obs) {
    
    # t_obs is a vector containing the discretization points
    # x_obs is a vector containing the values to be interpolated
    
    function(t) {
      sapply(t, function(tt) {
        if(tt >= t_obs[length(t_obs)]) {
          return(x_obs[length(x_obs)])
        } else {
          j <- max(which(t_obs <= tt))
          return(x_obs[j])
        }
      })
    }
  }
  
  n <- nrow(X) # number of observations
  p <- ncol(X) # number of discretization points
  
  # X is the n x p predictor matrix
  # t is a vector containing the discretization points (the grid is assumed to be common)
  # tau is the quantile to be estimated (by default 0.5 corresponding to the median)
  # K_max is the number of candidate eigenfunctions to keep
  # n_fine is the number of points that the random functions will be evaluated on after interpolation
  
  # Create piecewise constant functions for each one of the X_i
  X_func <- lapply(1:n, function(i) piecewise_constant(t, X[i,]))
  t_fine <- seq(min(t), max(t), length.out = n_fine) # equispaced discretization
  
  dt <- diff(c(t_fine, t_fine[n_fine] + (t_fine[n_fine]-t_fine[n_fine-1]))) # adjust the rightmost point
  
  # Evaluate curves on fine grid
  X_fine <- t(sapply(X_func, function(f) f(t_fine)))  # n x n_fine
  
  # Mean function on fine grid
  mean_fun_fine <- colMeans(X_fine)
  
  # Covariance matrix (kernel) on fine grid
  C <- matrix(0, n_fine, n_fine)
  for(i in 1:n) {
    Xc <- X_fine[i,] - mean_fun_fine
    C <- C + (Xc %*% t(Xc)) * outer(dt, dt)
  }
  C <- C / n
  
  # Eigen-decomposition
  eig <- eigen(C, symmetric = TRUE)
  lambda <- eig$values # eigenvalues in descending order
  phi_fine_all <- eig$vectors  # all eigenfunctions
  phi_fine_all <- phi_fine_all / sqrt(dt) # renormalize for orthonormality in L2
  
  
  # Riemann sums for all FPCA scores up to K_max
  xi_all <- matrix(0, n, K_max)
  for(i in 1:n) {
    for(k in 1:K_max) {
      xi_all[i,k] <- sum((X_fine[i,] - mean_fun_fine) * phi_fine_all[,k] * dt)
    }
  }
  
  # GCV selection of number of eigenfunctions
  
  gcv_vals <- numeric(K_max)
  for(m in 1:K_max) {
    xiK <- xi_all[,1:m, drop=FALSE]
    fit <- rq(Y ~ xiK, tau = tau) # Quantile regression with the rq function of the quantreg package
    fitted_Y <- fitted(fit) # fitted values
    rho <- function(r, tau) r * (tau - as.numeric(r < 0)) # check loss function
    gcv_vals[m] <- sum(rho(Y - fitted_Y, tau)) / (n - (m + 1)) # GCV criterion, see Section 4 Kato (2012)
  }
  
  best_m <- which.min(gcv_vals) # minimizer of the GCV criterion
  
  # Final fit using best_m
  xiK <- xi_all[,1:best_m, drop=FALSE]
  fit <- rq(Y ~ xiK, tau = tau)
  beta_hat <- coef(fit)[-1] # exclude intercept
  alpha_hat <- coef(fit)[1] # intercept
  
  # Functional coefficient on fine grid (linear combination of the eigenfunctions)
  beta_fun_fine <- phi_fine_all[,1:best_m] %*% beta_hat
  
  # Convert to original t
  beta_fun <- sapply(t, function(tt) beta_fun_fine[max(which(t_fine <= tt))])
  mean_fun <- sapply(t, function(tt) mean_fun_fine[max(which(t_fine <= tt))])
  phi <- matrix(0, nrow=length(t), ncol=best_m)
  for(k in 1:best_m) {
    phi[,k] <- sapply(t, function(tt) phi_fine_all[max(which(t_fine <= tt)),k])
  }
  
  # Fitted scalar responses
  fitted_Y <- alpha_hat + xiK %*% beta_hat
  
  return(list(
    mean_fun = mean_fun,
    eigenfunctions = phi,
    eigenvalues = lambda[1:best_m],
    scores = xiK,
    qr_fit = fit,
    fitted_Y = fitted_Y,
    alpha_hat = alpha_hat,
    beta_fun = beta_fun,
    t = t,
    gcv_vals = gcv_vals,
    best_m = best_m
  ))
}
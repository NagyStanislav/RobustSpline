// ==============================================================================
// File: freg.cpp
// Purpose: Functional regression with robust estimation for RobustSpline
// 
// This module implements iteratively reweighted least squares (IRLS) algorithms
// and quadratic programming solvers for robust regression with thin-plate splines.
// It provides multiple loss functions (absolute, square, Huber, logistic) and
// supports both ridge regularization and QP-based optimization via OSQP.
// 
// Authors: Stanislav Nagy, Michele Cavazzutti
// Email: michele.cavazzutti@matfyz.cuni.cz, stanislav.nagy@matfyz.cuni.cz
// Date: 31.08.2025
// 
// The Rcpp::export attribute makes these functions available to R code.
// ==============================================================================

# define ARMA_WARN_LEVEL 1   // Suppress warnings about nearly singular matrices

# include <RcppArmadillo.h>
#include "./osqp/include/public/osqp.h"
#include <vector>
#include <limits>
#include <cstring>
#include <algorithm>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// ==============================================================================
/// Compute weights for iteratively reweighted least squares (IRLS) algortihm
/// 
/// This function computes the weights psi(t)/(2*t) used in IRLS algorithms
/// to implement robust regression/location with various loss functions. The weights
/// control the influence of each observation during iterative estimation.
/// 
/// @param t          [in] arma::vec - Standardized residuals (typically residuals/scale)
/// @param type       [in] int       - Loss function type:
///                                    1 = Absolute/quantile loss
///                                    2 = Squared loss (OLS)
///                                    3 = Huber loss
///                                    4 = Logistic loss
/// @param alpha      [in] double    - Quantile level for asymmetric losses (used in type 1)
///                                    alpha in (0,1), default 0.5 for median
/// @param tuning     [in] double    - Tuning parameter for loss function:
///                                    type 1: threshold for Huberization (0 = no tuning)
///                                    type 3: Huber's k parameter (0 = absolute loss)
///                                    other types: unused
/// 
/// @return           arma::vec      - Weight vector, same length as t
///                                    weights >= 0, unbounded for small |t|
/// 
/// @details
/// - Type 1 (Absolute): weights are alpha/(2*|t|) with optional Huberization
/// - Type 2 (Square): constant weights = 1 (standard OLS)
/// - Type 3 (Huber): smooth transition between quadratic (|t| <= k) and linear (|t| > k)
/// - Type 4 (Logistic): smooth S-shaped weighting with exponential saturation
/// - Numerical safeguards: uses R_PosInf when t \approx 0 (type 1), caps exp(t) at +- 500
/// 
/// @warning Large |t| values may produce very small weights; consider residual scaling.
/// 
/// @example
/// // In R: compute weights for absolute loss with alpha=0.5 (median regression)
/// // resids <- c(-2, -1, 0, 1, 2)
/// // w <- psiwC(resids, type=1, alpha=0.5, tuning=0)
// [[Rcpp::export()]]
arma::vec psiwC (arma::vec t, int type, double alpha, double tuning)
{
  int n = t.n_elem;
  arma::vec res(n);
  
  // Type 1: Quantile/absolute loss (robust)
  if(type == 1){
    if(tuning == 0){
      // Pure absolute loss without regularization
      for(int i=0; i<n; i++){
        if(t(i)>0) res(i) = alpha/(2*t(i)); // 0.5/t(i);
        if(t(i)<0) res(i) = (alpha-1)/(2*t(i)); // -0.5/t(i);
        if(t(i)==0) res(i) = R_PosInf; 
      }
    } else {
      // Huberized absolute loss: smooth near zero to avoid non-diffrentiability
      for(int i=0; i<n; i++){
        if(abs(t(i))<=tuning){
          res(i) = alpha/(2*tuning); // 0.5/tuning;
        } else {
          if(t(i)>0) res(i) = alpha/(2*t(i)); // 0.5/t(i);
          if(t(i)<0) res(i) = (alpha-1)/(2*t(i)); // -0.5/t(i);
        }
      }
    }
    return(res);
  }

  // Type 2: Squared loss (standard OLS)
  if(type == 2){
    res.ones();
    return(res);
  }
  
  // Type 3: Huber loss (robust, smooth transition)
  if(type == 3){
    if(tuning == 0){
      // Pure absolute loss (similar to type 1 without alpha parameter)
      for(int i=0; i<n; i++){
        if(t(i)>0) res(i) = 0.5/t(i);
        if(t(i)<0) res(i) = -0.5/t(i);
        if(t(i)==0) res(i) = R_PosInf;
      } 
    } else {
      // Huber's robust loss with tuning parameter k
      // Quadratic for |t| ≤ k, linear for |t| > k
      for(int i=0; i<n; i++){
        if(abs(t(i))<=tuning){
          res(i) = 0.5;
        } else { 
          res(i) = 0.5*tuning/abs(t(i)); 
        }      
      }      
    }
    return(res);      
  }
  
  // Type 4: Logistic loss (smooth S-shaped weighting)
  if(type == 4){
    for(int i=0; i<n; i++){
      if(t(i)==0){
        res(i) = 0.5; 
      } else {
        if(abs(t(i))>=500){
	  res(i) = 1/abs(t(i)); // if abs(t) is too large, exp(t) might give
	  // numerically Inf, but in limit we have that psiw converges to 1/abs(t)
        } else {
	  // Full logistic formula: psi(t)/(2t)
	  res(i)=(-2*exp(-t(i))/(1+exp(-t(i)))+1)/t(i);
        }
      }
    }
    return(res);
  }
  
  // Unknown type: return zeros
  res.zeros();
  return(res);
}

// ==============================================================================
/// Iteratively reweighted least squares (IRLS) with regularization - signle lambda
/// 
/// Implements the IRLS algorithm for robust regression/location with a thin-plate spline
/// design matrix Z, penalty matrix H (e.g., from thin-plate spline basis), and
/// a single regularization parameter lambda. The algorithm iteratively updates
/// weights based on residuals until convergence.
/// 
/// The optimization problem solved is:
///   minimize: sum(w_i * rho((Y_i - Z*theta)_i/sc)) + lambda * theta^T H theta
/// 
/// where rho is the loss function (absolute, Huber, etc.) and w_i are prior weights.
/// 
/// @param Z          [in] arma::mat      - Design matrix, dimensions (n \times p)
/// @param Y          [in] arma::vec      - Response vector, length n
/// @param lambda     [in] double         - Regularization parameter (lambda >= 0)
///                                         Controls smoothness/complexity trade-off
/// @param H          [in] arma::mat      - Penalty matrix, dimensions (p \times p)
///                                         Usually from thin-plate spline basis
/// @param type       [in] int            - Loss function type (1-4, see psiwC)
/// @param alpha      [in] double         - Quantile level for asymmetric losses
/// @param W          [in] arma::vec      - Prior weights, length n (e.g., from design)
/// @param sc         [in] double         - Residual scale parameter (for standardization)
/// @param residsin   [in/out] arma::vec  - Initial residuals, updated each iteration
/// @param tuning     [in] double         - Tuning parameter for Huber-like loss functions
/// @param toler      [in] double         - Convergence tolerance (relative change in residuals)
/// @param imax       [in] int            - Maximum number of iterations
/// 
/// @return Rcpp::List with elements:
///   - theta_hat     (arma::vec): Final parameter estimates, length p
///   - converged     (int):       1 if converged, 0 otherwise
///   - ic            (int):       Number of iterations completed
///   - resids        (arma::vec): Final residuals, length n
///   - hat_values    (arma::vec): Diagonal of hat matrix (leverage), length n
///   - last_check    (double):    Final convergence criterion value
///   - weights       (arma::vec): Final IRLS weights (2*w), length n
///   - fitted        (arma::vec): Fitted values Z*theta_hat, length n
/// 
/// @details
/// The IRLS iteration:
/// 1. Compute weights w_i = psi(residuals_i/sc) / (2*residuals_i/sc)
/// 2. Solve weighted normal equations: (Z^T W Z + lambda*H)*theta = Z^T W Y
/// 3. Update residuals = Y - Z*theta
/// 4. Check convergence: max(|resids_new - resids_old|/sc) < toler
/// 5. Repeat until convergence or imax iterations
/// 
/// @warning
/// - (Relatively) Slow convergence is possible with very robust losses (type 1, 3)
/// - Matrix singularity warnings suppressed (ARMA_WARN_LEVEL=1)
/// - May not converge for ill-conditioned problems
/// - Implementation for absolute/quantile loss (type = 3) doesn't solve the real
///   absolute/quantile problem, but a Huber-like approximation of it.
///   See the quadratic programming solver for an exact solution of the problem
/// 
/// @see psiwC, solveC, ridgeC
/// 
/// @example
/// // In R: robust quantile regression with thin-plate spline
/// // Z <- model_matrix        # Design from TPS basis
/// // Y <- response_vector
/// // H <- penalty_matrix      # From TPS
/// // result <- IRLSC(Z, Y, lambda=0.1, H, type=1, alpha=0.5, 
/// //                 W=rep(1,n), sc=mad(Y), resids=Y, tuning=0, 
/// //                 toler=1e-4, imax=20)
// [[Rcpp::export()]]
Rcpp::List IRLSC (arma::mat Z, arma::vec Y, double lambda, arma::mat H,
		  int type, double alpha, arma::vec W, double sc,
		  arma::vec residsin, double tuning, double toler, int imax)
{
  
  int ic = 0, istop = 0;
  int n = Y.n_elem, p1 = Z.n_cols;

  // Precompute transpose once
  arma::vec w(n);
  arma::mat Zt = trans(Z), ZW(p1,n), Z1(p1,p1); // Zt is of dimension p1  \times n
  arma::mat hat_mat(n,n); // hat matrix for hat values
  arma::vec theta_new(p1), resids1(n), hat_diag(n), fitted(n);
  double check = 0;

  // IRLS loop
  while(istop == 0 && ic < imax){
    ic++;

    // Update weights: w = W * psi(residuals/sc) / (sc^{2})
    w = W % psiwC(residsin/sc, type, alpha, tuning); // element-wise product
    w = w/(sc*sc); // scaling affects only the weights

    // Build weighted normal equation matrix: Z^T W Z
    for(int j1=0; j1<n; j1++){
      ZW.col(j1) = Zt.col(j1)*w(j1); // Column-wise: Zt[,j1] * w[j1]
    }

    // Solve: (Z^T W Z + lambda*H) * theta = Z^T W Y
    Z1 = ZW*Z + lambda*H;
    theta_new = solve(Z1, ZW*Y);

    // Update residuals
    resids1 = Y - Z*theta_new;

    // Check convergence: relative change in residuals
    check = max(abs(resids1-residsin)/sc); 
    if(check < toler){
      istop = 1;
    }
    
    residsin = resids1;
  }

  // Warn if non-convergence
  if(istop==0) Rcpp::warning("Estimator did not converge.");

  // Compute hat matrix and its diagonal (leverage values)
  hat_mat = Z*solve(Z1, ZW);
  fitted = hat_mat*Y;
  hat_diag = hat_mat.diag();
  
  return Rcpp::List::create(Rcpp::Named("theta_hat") = theta_new,
                            Rcpp::Named("converged") = istop,
                            Rcpp::Named("ic") = ic,
                            Rcpp::Named("resids") = residsin,
                            Rcpp::Named("hat_values") = hat_diag,
                            Rcpp::Named("last_check") = check,
                            Rcpp::Named("weights") = 2*w,
                            Rcpp::Named("fitted") = fitted
			    );
}

// ==============================================================================
/// Iteratively reweighted least squares (IRLS) with regularization - multiple lambdas
/// 
/// This function extends IRLSC to handle multiple regularization parameters in a single call.
/// Useful for cross-validation or regularization path estimation. Each lambda value
/// solves the same robust regression/location problem with different smoothness penalties.
/// 
/// @param Z          [in] arma::mat      - Design matrix, dimensions (n \times p)
/// @param Y          [in] arma::vec      - Response vector, length n
/// @param lambda     [in] arma::vec      - Multiple regularization parameters, length nlambda
/// @param H          [in] arma::mat      - Penalty matrix, dimensions (p \times p)
/// @param type       [in] int            - Loss function type (1-4, see psiwC)
/// @param alpha      [in] double         - Quantile level for asymmetric losses
/// @param W          [in] arma::vec      - Prior weights, length n
/// @param sc         [in] double         - Residual scale parameter
/// @param tuning     [in] double         - Tuning parameter for Huber-like loss function
/// @param toler      [in] double         - Convergence tolerance
/// @param imax       [in] int            - Maximum number of iterations
/// 
/// @return Rcpp::List with elements:
///   - theta_hat     (arma::mat): Parameter estimates, dimensions (p \times nlambda)
///                                Column k contains estimates for lambda[k]
///   - converged     (arma::vec): Convergence status per lambda, length nlambda
///   - ic            (arma::vec): Iterations per lambda, length nlambda
///   - resids        (arma::mat): Final residuals, dimensions (n \times nlambda)
///   - hat_values    (arma::mat): Leverage values, dimensions (n \times nlambda)
///   - fitted        (arma::mat): Fitted values, dimensions (n \times nlambda)
/// 
/// @details
/// - Runs nlambda independent IRLS sequences
/// - Each lambda value produces a complete solution path element
/// - Useful for regularization parameter selection via cross-validation
/// - More efficient than calling IRLSC repeatedly (avoids recomputation of matrices)
/// - Same considerations about exactness and convergence of the single-lamba IRLS algorithm hold
/// 
/// @note Uses faster Arma solve option (arma::solve_opts::fast) for speed
/// 
/// @see IRLSC, psiwC
/// 
/// @example
/// // In R: compute regularization path with 5 lambda values
/// // lambdas <- c(0.001, 0.01, 0.1, 1, 10)
/// // result <- IRLSCmult(Z, Y, lambdas, H, type=1, alpha=0.5,
/// //                     W=rep(1,n), sc=mad(Y), tuning=0, 
/// //                     toler=1e-4, imax=20)
/// // theta_path <- result$theta_hat
// [[Rcpp::export()]]
Rcpp::List IRLSCmult (arma::mat Z, arma::vec Y, arma::vec lambda, arma::mat H,
		      int type, double alpha, arma::vec W, double sc, double tuning, double toler, int imax)
{
  int n = Y.n_elem, p1 = Z.n_cols, nlambda = lambda.n_elem;
  
  // Status tracking vectors
  arma::vec istop(nlambda, arma::fill::zeros), ic(nlambda, arma::fill::zeros);

  // Working arrays
  arma::vec w(n), resids1(n);
  arma::mat Zt = trans(Z), ZW(p1,n), Z1(p1,p1); // Zt is of dimension p1-times-n
  arma::mat hat_mat(n,n); // hat matrix for hat values

  // Output matrices (results stored per lambda)
  arma::mat resids(n,nlambda, arma::fill::ones);
  arma::mat hat_diag(n,nlambda, arma::fill::zeros);
  arma::mat fitted(n,nlambda, arma::fill::zeros);
  arma::mat theta_new(p1,nlambda, arma::fill::zeros);
  
  double check = 0;

  // Loop over each lambda value
  for(int k=0; k<nlambda; k++){
    // IRLS iteration for lambda[k]
    while(istop(k) == 0 && ic(k) < imax){
      ic(k)++;

      // Update weights based on current residuals for this lambda
      w = W % psiwC(resids.col(k)/sc, type, alpha, tuning); // element-wise product
      w = w/(sc*sc); // scaling affects only the weights

      // Build weighted normal equations
      for(int j1=0; j1<n; j1++){
        ZW.col(j1) = Zt.col(j1)*w(j1);        
      }

      // Solve: (Z^T W Z + lambda[k]*H) * theta = Z^T W Y
      Z1 = ZW*Z + lambda(k)*H;
      theta_new.col(k) = solve(Z1, ZW*Y, arma::solve_opts::fast);

      // Update residuals
      resids1 = Y - Z*theta_new.col(k);

      // Check convergence
      check = max(abs(resids1-resids.col(k)))/sc; 
      if(check < toler){
        istop(k) = 1;
      }
      
      resids.col(k) = resids1;
    }
    
    // Compute hat matrix and leverage for this lambda
    hat_mat = Z*solve(Z1, ZW, arma::solve_opts::fast);
    fitted.col(k) = hat_mat*Y;
    hat_diag.col(k) = hat_mat.diag();
  }
  
  return Rcpp::List::create(Rcpp::Named("theta_hat") = theta_new,
                            Rcpp::Named("converged") = istop,
                            Rcpp::Named("ic") = ic,
                            Rcpp::Named("resids") = resids,
                            Rcpp::Named("hat_values") = hat_diag,
                            Rcpp::Named("fitted") = fitted
			    );
}

// ==============================================================================
/// Ridge regression/location estimation with weighted least squares (non-robust)
/// 
/// Solves the standard ridge regression/location problem with optional prior weights.
/// Unlike IRLSC, this function performs a single (non-iterative) weighted
/// least squares step with no robustification. Useful as a baseline or for
/// parametric starting values.
/// 
/// The optimization problem is:
///   minimize: sum(W_i * (Y_i - Z*theta)_i²) + lambda * theta^T H theta
/// 
/// @param Z          [in] arma::mat  - Design matrix, dimensions (n \times p)
/// @param Y          [in] arma::vec  - Response vector, length n
/// @param lambda     [in] double     - Ridge regularization parameter (lambda >= 0)
/// @param H          [in] arma::mat  - Penalty matrix, dimensions (p \times p)
/// @param W          [in] arma::vec  - Prior weights, length n (typically all ones or
///                                     frequency weights from data design)
/// 
/// @return Rcpp::List with elements:
///   - theta_hat     (arma::vec): Parameter estimates, length p
///   - resids        (arma::vec): Residuals Y - Z*theta_hat, length n
///   - hat_values    (arma::vec): Diagonal of weighted hat matrix, length n
///   - fitted        (arma::vec): Fitted values Z*theta_hat, length n
/// 
/// @details
/// Solves in closed form: theta = (Z^T W Z + lambda*H)^(-1) Z^T W Y
/// No iteration: single linear algebra operation.
/// Faster than IRLS but lacks robustness to outliers.
/// 
/// @note
/// - W must be positive; no checks are performed
/// - Requires non-singular (Z^T W Z + lambda*H) matrix
/// - Useful for penalized least squares, elastic net-style regression
/// 
/// @see IRLSC, IRLSCmult
/// 
/// @example
/// // In R: standard ridge regression with thin-plate spline
/// // Z <- model_matrix
/// // Y <- response_vector
/// // result <- ridgeC(Z, Y, lambda=0.1, H, W=rep(1,n))
/// // coef <- result$theta_hat
// [[Rcpp::export()]]
Rcpp::List ridgeC (arma::mat Z, arma::vec Y, double lambda, arma::mat H, arma::vec W)
{
  int n = Y.n_elem, p1 = Z.n_cols;
  arma::mat Zt = trans(Z); // Zt is of dimension p1-times-n
  arma::mat hat_mat(n,n); // hat matrix for hat values
  arma::vec theta_new(p1), residsin(n), hat_diag(n), fitted(n);

  // Solve: (Z^T W Z + lambda*H) * theta = Z^T W Y
  theta_new = solve(Zt*arma::diagmat(W)*Z + lambda*H, Zt*arma::diagmat(W)*Y);

  // Compute fitted values and residuals
  fitted = Z*theta_new;
  residsin = Y - fitted;

  // Hat matrix diagonal (leverage values)
  hat_mat = Z*solve(Zt*arma::diagmat(W)*Z + lambda*H, Zt*arma::diagmat(W));
  hat_diag = hat_mat.diag();
  
  return Rcpp::List::create(Rcpp::Named("theta_hat") = theta_new,
                            Rcpp::Named("resids") = residsin,
                            Rcpp::Named("hat_values") = hat_diag,
                            Rcpp::Named("fitted") = fitted
			    );
}

// ==============================================================================
// ==============================================================================
// Quadratic programming section
// ==============================================================================
// ==============================================================================

// ==============================================================================
/// Helper: Extract sparse matrix in CSC (Compressed Sparse Column) format
/// 
/// Converts an Armadillo sparse matrix to OSQP-compatible CSC representation.
/// OSQP requires explicit row indices and column pointers for efficient solver setup.
/// 
/// @param S          [in] arma::sp_mat    - Sparse matrix to convert
/// @param x_out      [out] std::vector<OSQPFloat> - Non-zero values
/// @param i_out      [out] std::vector<OSQPInt>   - Row indices
/// @param p_out      [out] std::vector<OSQPInt>   - Column pointers (length n_cols+1)
/// 
/// @details
/// CSC format: columns are stored contiguously, column pointers define boundaries.
/// Example: 3×3 matrix [[1,0,2],[0,3,0],[4,0,5]]
/// - Values: [1, 4, 3, 2, 5]
/// - Row idx: [0, 2, 1, 0, 2]
/// - Col ptr: [0, 2, 3, 5]  (column j starts at index p[j], ends at p[j+1]-1)
/// 
/// @note Efficiency: Direct memory mapping avoids copying.
static void spmat_to_osqp_csc(const arma::sp_mat& S,
			      std::vector<OSQPFloat>& x,
			      std::vector<OSQPInt>& i,
			      std::vector<OSQPInt>& p
			      ) {
  // Copy non-zero values
    x.assign(S.values, S.values + S.n_nonzero);

    // Copy and cast row indices
    i.resize(S.n_nonzero);
    for(arma::uword k = 0; k < S.n_nonzero; ++k) {
        i[k] = static_cast<OSQPInt>(S.row_indices[k]);
    }

    // Copy and cast column pointers
    p.resize(S.n_cols + 1);
    for(arma::uword k = 0; k <= S.n_cols; ++k) {
        p[k] = static_cast<OSQPInt>(S.col_ptrs[k]);
    }
}

// ==============================================================================
/// Robust Huber regression/location estimation via quadratic programming (OSQP)
/// 
/// Solves the penalized Huber loss problem using the OSQP (Operator Splitting
/// Quadratic Program) solver. This approach reformulates the problem as a QP
/// to leverage advanced optimization algorithms.
/// 
/// The Huber loss is:
///   L_delta(t) = 0.5*t²          if |t| ≤ delta  (quadratic region - smooth)
///            delta*|t| - 0.5*delta^2 if |t| > delta (linear region - robust)
/// 
/// The QP formulation uses slack variables [theta, a, t] where:
///   - theta: regression coefficients (p dimensional)
///   - a:     auxiliary slack variables for lower bound (n dimensional)
///   - t:     auxiliary slack variables for upper bound (n dimensional)
/// 
/// Optimization:
///   minimize: 0.5*theta^T (H) theta + w^T (alpha*a + (1-alpha)*t)
///   subject to: Z*theta + a - t = Y,  a ≥ 0, t ≥ 0
/// 
/// where alpha = 0.5 for symmetric Huber loss.
/// 
/// @param Z          [in] arma::sp_mat - Sparse design matrix, dimensions (n \times p)
/// @param Y          [in] arma::vec    - Response vector, length n
/// @param H          [in] arma::mat    - Penalty matrix, dimensions (p \times p)
/// @param w          [in] arma::vec    - Weight vector, length n (usually all ones)
/// @param delta      [in] double       - Huber loss threshold delta (default: 1.345)
///                                       Controls quadratic-linear transition
/// @param eps_abs    [in] double       - Absolute tolerance for OSQP algorithm
/// @param eps_rel    [in] double       - Dual tolerance for OSQP algorithm
/// 
/// @return Rcpp::List with elements:
///   - theta_hat     (arma::vec): Regression coefficients, length p
///   - fitted        (arma::vec): Fitted values Z*theta_hat, length n
///   - resids        (arma::vec): Residuals Y - fitted, length n
///   - converged     (bool):      True if OSQP solver converged
/// 
/// @details
/// Constraint setup (3n constraints total):
///   1. Z*theta + a - t = Y       (n constraints: residual decomposition)
///   2. Z*theta - a + t = Y       (n constraints: mirror for symmetry)
///   3. t ≥ 0                     (n non-negativity constraints on slack)
/// 
/// P matrix structure (block diagonal):
///   [H + eps*I_p    0         0       ]
///   [0         w + eps*I_n   0       ]
///   [0              0      eps*I_n   ]
/// 
/// @warning
/// - delta = 1.345 is recommended for robust estimation
/// - Sparse Z required for efficiency; dense matrices will be slow
/// - May produce different results than IRLS due to different numerical algorithms (although converging to the same result asymptotically)
/// 
/// @see QuantileQpC (quantile QP solver)
/// 
/// @example
/// // In R: robust Huber regression /location estimation with thin-plate spline basis
/// // Z_sparse <- as(Z, "dgCMatrix")
/// // Y <- response_vector
/// // H <- penalty_matrix
/// // result <- HuberQpC(Z_sparse, Y, H, w=rep(1,n), delta=1.345)
// [[Rcpp::export]]
Rcpp::List HuberQpC(const arma::sp_mat& Z, 
                    const arma::vec& Y, 
                    const arma::mat& H, 
                    const arma::vec& w, 
                    double delta = 1.345,
		    double eps_abs = 1e-6,
		    double eps_rel = 1e-6
		    )
{
  int n = static_cast<int>(Z.n_rows);
  int p = static_cast<int>(Z.n_cols);
  int n_vars = p + 2 * n;
  int n_cons = 3 * n;

  // Build Hessian P (block diagonal structure)
  arma::sp_mat Psp(n_vars, n_vars);

  // Block (0,0): H + eps*I_p (regularization)
  Psp.submat(0, 0, p - 1, p - 1) = arma::sp_mat(H);
  for(int i = 0; i < p; ++i) {
    Psp(i, i) += 1e-12;  // Regularization for numerical stability
  }
  for(int i = 0; i < n; ++i) {
    Psp(p + i, p + i) = w(i) + 1e-12; // Block (p:p+n, p:p+n): w_i + eps*I_n (weights on auxiliary variables)
    Psp(p + n + i, p + n + i) = 1e-12; // Block (p+n:p+2n, p+n:p+2n): eps*I_n (regularization on slack)
  }

  // Keep only upper triangle for OSQP (symmetric matrix optimization)
  Psp = arma::trimatu(Psp);

  // Build linear term q
  arma::vec qvec = arma::zeros<arma::vec>(n_vars);
  // q[p+n:p+2n] = w * delta (cost on slack variable t)
  for(int i = 0; i < n; ++i) {
    qvec(p + n + i) = w(i) * delta;          // Peso * delta solo su t
  }

  // Build constraint matrix A
  // Constraints: [Z, I, -I; -Z, -I, I; 0, 0, I] * [theta, a, t] vs [Y, -Y, 0]
  arma::sp_mat sp_In = arma::speye<arma::sp_mat>(n, n);
  arma::sp_mat Asp(n_cons, n_vars);

  // Constraint block 1: Z*theta + a - t = Y
  Asp.submat(0, 0, n-1, p-1) = Z;
  Asp.submat(0, p, n-1, p+n-1) = sp_In;
  Asp.submat(0, p+n, n-1, n_vars-1) = sp_In;

  // Constraint block 2: -Z*theta - a + t ≥ -Y (equivalently, Z*theta + a - t ≤ Y, handled via bounds)
  Asp.submat(n, 0, 2*n-1, p-1) = -Z;
  Asp.submat(n, p, 2*n-1, p+n-1) = -sp_In;
  Asp.submat(n, p+n, 2*n-1, n_vars-1) = sp_In;

  // Constraint block 3: t ≥ 0 (represented as constraint on t variables)
  Asp.submat(2*n, p+n, 3*n-1, n_vars-1) = sp_In;

  // Lower and upper bounds for constraints
  arma::vec lvec(n_cons);
  lvec.subvec(0, n-1)        = Y;     // Z*theta + a - t = Y
  lvec.subvec(n, 2*n-1)      = -Y;    // -Z*theta - a + t = -Y
  lvec.subvec(2*n, 3*n-1).zeros();    // t ≥ 0
    
  arma::vec uvec(n_cons);
  uvec.fill(OSQP_INFTY); // Upper bounds: all constraints <= infty

  // Convert to OSQP CSC format
  std::vector<OSQPFloat> Px, Ax;
  std::vector<OSQPInt> Pi, Pp, Ai, Ap;
  spmat_to_osqp_csc(Psp, Px, Pi, Pp);
  spmat_to_osqp_csc(Asp, Ax, Ai, Ap);

  // Setup OSQP solver
  OSQPCscMatrix* P_osqp = OSQPCscMatrix_new(n_vars, n_vars, (OSQPInt)Px.size(), Px.data(), Pi.data(), Pp.data());
  OSQPCscMatrix* A_osqp = OSQPCscMatrix_new(n_cons, n_vars, (OSQPInt)Ax.size(), Ax.data(), Ai.data(), Ap.data());
    
  OSQPSettings* settings = (OSQPSettings*)malloc(sizeof(OSQPSettings));
  if(settings) osqp_set_default_settings(settings);
  settings->verbose = 0;
  settings->eps_abs = eps_abs; 
  settings->eps_rel = eps_rel;

  // Solve
  OSQPSolver* solver = nullptr;
  osqp_setup(&solver, P_osqp, qvec.memptr(), A_osqp, lvec.memptr(), uvec.memptr(), n_cons, n_vars, settings);
  osqp_solve(solver);

  // Check convergence status
  bool converged = false;
  if (solver && solver->info) {
    int status = solver->info->status_val;
    converged = (status == OSQP_SOLVED || status == OSQP_SOLVED_INACCURATE);
  }

  // Extract solution
  arma::vec theta = arma::zeros<arma::vec>(p);
  if (solver && solver->solution) {
    for (int i = 0; i < p; ++i) theta(i) = solver->solution->x[i];
  }

  // Cleanup
  osqp_cleanup(solver);
  free(P_osqp); free(A_osqp); free(settings);

  // Compute fitted values and residuals
  arma::vec fitted_vals = Z * theta;
  arma::vec residuals = Y - fitted_vals;

  return Rcpp::List::create(
			    Rcpp::Named("theta_hat") = theta,
			    Rcpp::Named("fitted") = fitted_vals,
			    Rcpp::Named("resids") = residuals,
			    Rcpp::Named("converged") = converged
			    );
}

// ==============================================================================
/// Robust quantile regression/location estimation via quadratic programming (OSQP)
/// 
/// Solves the quantile (asymmetric) loss problem using OSQP.
/// This approach reformulates quantile regression as a QP for use of advanced
/// optimization algorithms.
/// 
/// The quantile loss is:
///   L_α(t) = alpha*t        if t ≥ 0  (penalizes under-prediction more if alpha > 0.5)
///            (alpha-1)*|t|  if t < 0  (penalizes over-prediction more if alpha < 0.5)
/// 
/// The QP formulation uses slack variables [theta, a, t] where:
///   - theta: regression coefficients (p dimensional)
///   - a:     auxiliary slack variables for positive residuals (n dimensional)
///   - t:     auxiliary slack variables for negative residuals (n dimensional)
/// 
/// Optimization:
///   minimize: 0.5*theta^T (2*lambda*H) theta + alpha*w^T a + (1-alpha)*w^T t
///   subject to: Z*theta + a - t = Y,  a ≥ 0, t ≥ 0
/// 
/// where alpha is the quantile level.
/// 
/// @param Z          [in] arma::sp_mat  - Sparse design matrix, dimensions (n \times p)
/// @param Y          [in] arma::vec     - Response vector, length n
/// @param lambda     [in] double        - Regularization parameter (lambda >=  0)
/// @param H          [in] arma::mat     - Penalty matrix, dimensions (p \times p)
/// @param w          [in] arma::vec     - Weight vector, length n
/// @param alpha      [in] double        - Quantile level in (0,1) (default: 0.5 for median)
///                                        0.1 = 10th percentile, 0.9 = 90th percentile
/// @param eps_abs    [in] double        - Absolute tolerance for OSQP algorithm
/// @param eps_rel    [in] double        - Dual tolerance for OSQP algorithm
/// 
/// @return Rcpp::List with elements:
///   - theta_hat     (arma::vec): Quantile regression coefficients, length p
///   - fitted        (arma::vec): Fitted values Z*theta_hat, length n
///   - resids        (arma::vec): Residuals Y - fitted, length n
///   - converged     (bool):      True if OSQP solver converged
/// 
/// @details
/// Constraint setup (3n constraints):
///   1. Z*theta + a - t = Y       (residual decomposition: a - t = residuals)
///   2. a ≥ 0                     (positive slack)
///   3. t ≥ 0                     (negative slack)
/// 
/// Cost function exploits asymmetry:
///   - If alpha=0.5 (median): symmetric cost, alpha*w = (1-alpha)*w
///   - If alpha=0.9 (90th %ile): under-prediction penalized 9x more
///   - If alpha=0.1 (10th %ile): over-prediction penalized 9x more
/// 
/// P matrix: [2*lambda*H + eps*I_p,  0,      0    ]
///           [0,                  eps*I_n,  0    ]
///           [0,                  0,    eps*I_n ]
/// 
/// @note
/// - lambda controls smoothness; larger λ produces smoother fits
/// - Sparse Z required for efficiency
/// - Quantile regression robust to outliers by design (asymmetric weighting)
/// - Default alpha=0.5 gives median regression
/// 
/// @see HuberQpC (companion Huber regression solver)
/// 
/// @example
/// // In R: quantile regression at 25th percentile with regularization
/// // Z_sparse <- as(Z, "dgCMatrix")
/// // Y <- response_vector
/// // result <- QuantileQpC(Z_sparse, Y, lambda=0.1, H, 
/// //                       w=rep(1,n), alpha=0.25)
/// // coef_q25 <- result$theta_hat
// [[Rcpp::export]]
Rcpp::List QuantileQpC(const arma::sp_mat& Z, 
                       const arma::vec& Y, 
                       double lambda, 
                       const arma::mat& H, 
                       const arma::vec& w, 
                       double alpha = 0.5,
		       double eps_abs = 1e-6,
		       double eps_rel = 1e-6
		       )
{
  int n = static_cast<int>(Z.n_rows);
  int p = static_cast<int>(Z.n_cols);
  int n_vars = p + 2 * n; // Variables: [theta (p), a (n), t (n)]
  int n_cons = 3 * n;     // Constraints: 3n total

  // 1. Build Hessian P (block diagonal)
  arma::sp_mat Psp(n_vars, n_vars);

  // Block (0,0): 2*lambda*H + eps*I_p
  arma::mat H_p = 2.0 * lambda * H;
  H_p.diag() += 1e-6; // Regularization
  Psp.submat(0, 0, p-1, p-1) = arma::sp_mat(H_p);

  // Blocks (p:p+2n, p:p+2n): eps*I on slack variables
  for(int j = p; j < n_vars; ++j) Psp(j, j) = 1e-6;

  // Keep upper triangle only
  Psp = arma::trimatu(Psp);

  // 2. Build linear term q (asymmetric quantile weighting)
  arma::vec qvec = arma::zeros<arma::vec>(n_vars);
  qvec.subvec(p, p + n - 1) = alpha * w; // Cost on positive residuals (a)
  qvec.subvec(p + n, n_vars - 1) = (1.0 - alpha) * w;  // Cost on negative residuals (t)

  // 3. Build constraint matrix A
  arma::sp_mat sp_In = arma::speye<arma::sp_mat>(n, n);
  arma::sp_mat Asp(n_cons, n_vars);

  // Constraint 1: Z*theta + a - t = Y
  Asp.submat(0, 0, n-1, p-1) = Z;
  Asp.submat(0, p, n-1, p+n-1) = sp_In;
  Asp.submat(0, p+n, n-1, n_vars-1) = -sp_In;

  // Constraint 2: a ≥ 0 (enforced via lower bound)
  Asp.submat(n, p, 2*n-1, p+n-1) = sp_In;

  // Constraint 3: t ≥ 0 (enforced via lower bound)
  Asp.submat(2*n, p+n, 3*n-1, n_vars-1) = sp_In;

  // Constraint bounds
  arma::vec lvec = arma::join_cols(Y, arma::zeros<arma::vec>(2 * n));
  arma::vec uvec = lvec;
  uvec.subvec(n, 3 * n - 1).fill(OSQP_INFTY); // Upper bounds: a, t unconstrained

  // 4. Convert to OSQP CSC format
  std::vector<OSQPFloat> Px, Ax;
  std::vector<OSQPInt> Pi, Pp, Ai, Ap;
  spmat_to_osqp_csc(Psp, Px, Pi, Pp);
  spmat_to_osqp_csc(Asp, Ax, Ai, Ap);

  // // 5. Setup OSQP solver
  OSQPCscMatrix* P_osqp = OSQPCscMatrix_new(n_vars, n_vars, (OSQPInt)Px.size(), Px.data(), Pi.data(), Pp.data());
  OSQPCscMatrix* A_osqp = OSQPCscMatrix_new(n_cons, n_vars, (OSQPInt)Ax.size(), Ax.data(), Ai.data(), Ap.data());
    
  OSQPSettings* settings = (OSQPSettings*)malloc(sizeof(OSQPSettings));
  if(settings) osqp_set_default_settings(settings);
  settings->verbose = 0;
  settings->eps_abs = eps_abs; 
  settings->eps_rel = eps_rel; 

  // 6. Solve
  OSQPSolver* solver = nullptr;
  osqp_setup(&solver, P_osqp, qvec.memptr(), A_osqp, lvec.memptr(), uvec.memptr(), n_cons, n_vars, settings);
  osqp_solve(solver);

  // 7. Check convergence
  bool converged = false;
  if (solver && solver->info) {
    int status = solver->info->status_val;
    converged = (status == OSQP_SOLVED || status == OSQP_SOLVED_INACCURATE);
  }

  // 8. Extract solution
  arma::vec theta = arma::zeros<arma::vec>(p);
  if (solver && solver->solution) {
    for (int i = 0; i < p; ++i) theta(i) = solver->solution->x[i];
  }

  // 9. Cleanup
  osqp_cleanup(solver);
  free(P_osqp); free(A_osqp); free(settings);

  // 10. Compute fitted values and residuals
  arma::vec fitted_vals = Z * theta;
  arma::vec residuals = Y - fitted_vals;

  return Rcpp::List::create(
			    Rcpp::Named("theta_hat") = theta,
			    Rcpp::Named("fitted") = fitted_vals,
			    Rcpp::Named("resids") = residuals,
			    Rcpp::Named("converged") = converged
			    );
}

// ==============================================================================
// File: dist.cpp
// Purpose: Distance computation utilities for the RobustSpline package
// 
// This module provides efficient distance matrix calculations using Rcpp and
// Armadillo. It includes functions for computing pairwise Euclidean distances
// within a single dataset and between two datasets, as well as matrix solving.
// 
// The Rcpp::export attribute makes these functions available to R code.
// ==============================================================================

// Include headers for Rcpp integration with Armadillo linear algebra library
#include "RcppArmadillo.h"
#include <Rcpp.h>

// Use Rcpp namespace for simplified syntax
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

// ==============================================================================
/// Compute pairwise Euclidean distances within a dataset
/// 
/// This function calculates a symmetric distance matrix containing the Euclidean
/// distances between all pairs of rows in the input matrix X. The matrix is
/// computed efficiently by only calculating the upper triangle and mirroring
/// values for the lower triangle.
/// 
/// @param X        [in] arma::mat - Input data matrix of dimensions (n x d),
///                                   where n is the number of observations and
///                                   d is the number of features (dimensions)
/// @param n        [in] int       - Number of rows in matrix X (observations)
/// 
/// @return         arma::mat      - Symmetric distance matrix of dimensions (n x n),
///                                   where element (i,j) contains the Euclidean
///                                   distance between rows i and j of X
/// 
/// @details The distance is computed as: sqrt(sum((X[i] - X[j])^2))
///          The function exploits symmetry to reduce computation time by ~50%.
///          Suitable for moderately sized datasets (n < 10,000 recommended).
///          If n is chosen smaller than the number of rows in X, the
///          remaining rows will be ignored in computation
/// 
/// @example
/// // In R: compute distances between 5 observations in 3D space
/// // X <- matrix(rnorm(15), nrow = 5)
/// // D <- distn(X, 5)
// [[Rcpp::export]]
arma::mat distn(arma::mat X, int n)
{
  // Initialize n x n distance matrix with zeros
  arma::mat res(n,n, arma::fill::zeros);

  // Compute pairwise distances for upper triangle (i < j)
  // Lower triangle is filled via symmetry
  for(int i = 0; i < n; i++){
    for(int j = i+1; j < n; j++){
      // Euclidean distance: sqrt(sum of squared differences)
      res(i,j) = res(j,i) = sqrt(sum(pow(X.row(i) - X.row(j),2)));
    }
  }
  return res;
}

// ==============================================================================
/// Compute Euclidean distances between two datasets
/// 
/// This function calculates the distance matrix between all rows of matrix A
/// and all rows of matrix B. Useful for computing distances between a training
/// set and test set, or between two different groups of observations.
/// 
/// @param A        [in] arma::mat - First data matrix of dimensions (m x d),
///                                   where m is the number of observations
/// @param B        [in] arma::mat - Second data matrix of dimensions (n x d),
///                                   where n is the number of observations
/// @param m        [in] int       - Number of rows in matrix A
/// @param n        [in] int       - Number of rows in matrix B
/// 
/// @return         arma::mat      - Distance matrix of dimensions (m x n),
///                                   where element (i,j) contains the Euclidean
///                                   distance between row i of A and row j of B
/// 
/// @details The distance is computed as: sqrt(sum((A[i] - B[j])^2))
///          Result matrix is not necessarily symmetric (unless A and B are identical).
///          Suitable for cross-distance computations in thin-plate spline fitting.
/// 
/// @example
/// // In R: compute distances from 4 training points to 3 test points
/// // A <- matrix(rnorm(12), nrow = 4)  # 4 observations, 3 dimensions
/// // B <- matrix(rnorm(9), nrow = 3)   # 3 observations, 3 dimensions
/// // D <- distnAB(A, B, 4, 3)       # Result: 4 x 3 matrix
// [[Rcpp::export]]
arma::mat distnAB(arma::mat A, arma::mat B, int m, int n)
{
  // Initialize m x n distance matrix with zeros
  arma::mat res(m,n, arma::fill::zeros);

  // Compute all pairwise distances between rows of A and rows of B
  for(int i = 0; i < m; i++){
    for(int j = 0; j < n; j++){
      // Euclidean distance between row i of A and row j of B
      res(i,j) = sqrt(sum(pow(A.row(i) - B.row(j),2)));
    }
  }
  return res;
}

// ==============================================================================
/// Solve a system of linear equations
/// 
/// This function provides a wrapper around Armadillo's solve function to expose
/// it to R. It solves the linear system A*X = B for X using efficient matrix
/// decomposition methods (LU or other appropriate algorithms).
/// 
/// @param A        [in] arma::mat - Coefficient matrix (must be square and non-singular)
/// @param B        [in] arma::mat - Right-hand side matrix or vector
/// 
/// @return         arma::mat      - Solution matrix X such that A*X = B
/// 
/// @throws         May throw an exception if A is singular or near-singular
/// 
/// @details This is used internally for solving the thin-plate spline normal equations.
///          Armadillo automatically selects the best decomposition method based on
///          matrix properties (LU, Cholesky, etc.).
/// 
/// @note    For numerical stability, consider using regularization for ill-conditioned systems.
/// 
/// @example
/// // In R: solve a simple 2x2 system
/// // A <- matrix(c(2, 1, 1, 2), nrow = 2)
/// // B <- matrix(c(1, 2), nrow = 2)
/// // X <- solveC(A, B)  # Solves A*X = B
// [[Rcpp::export]]
arma::mat solveC(arma::mat A, arma::mat B)
{
    // solve function implemented in armadillo
    return(solve(A, B));
}

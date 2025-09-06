#include <RcppArmadillo.h>
#include <gurobi_c++.h>

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::vec huber_qp_gurobi_penalized(const arma::mat& X, const arma::vec& y, const arma::mat& H, double delta = 1.345) {
  size_t n = X.n_rows;
  size_t p = X.n_cols;
  if (y.n_elem != n) {
    throw std::invalid_argument("Length of y must equal number of rows in X");
  }
  if (H.n_rows != p || H.n_cols != p) {
    throw std::invalid_argument("H must be p x p, where p is ncol(X)");
  }
  
  size_t nvars = p + 2 * n;
  size_t ncons = 3 * n;
  double epsilon = 1e-6;
  
  try {
    GRBEnv env = GRBEnv();
    env.set(GRB_IntParam_OutputFlag, 0);
    
    GRBModel model = GRBModel(env);
    
    // Add variables with LB = -Inf
    GRBVar vars[nvars];
    for (size_t i = 0; i < nvars; ++i) {
      vars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    }
    
    // Set linear objective: delta for each t_i
    for (size_t i = p + n; i < nvars; ++i) {
      vars[i].set(GRB_DoubleAttr_Obj, delta);
    }
    
    // Build quadratic objective
    GRBQuadExpr qobj;
    
    // For beta: Q_beta = (1/2) * H + (epsilon / 2) * I_p
    arma::mat Q_beta = 0.5 * H + (0.5 * epsilon) * arma::eye(p, p);
    for (size_t i = 0; i < p; ++i) {
      qobj.addTerm(Q_beta(i, i), vars[i], vars[i]);
      for (size_t j = i + 1; j < p; ++j) {
        qobj.addTerm(2 * Q_beta(i, j), vars[i], vars[j]);
      }
    }
    
    // For a: diagonal (0.5 + epsilon / 2) for each a_i
    double q_a = 0.5 + 0.5 * epsilon;
    for (size_t i = p; i < p + n; ++i) {
      qobj.addTerm(q_a, vars[i], vars[i]);
    }
    
    // For t: diagonal (epsilon / 2) for each t_i
    double q_t = 0.5 * epsilon;
    for (size_t i = p + n; i < nvars; ++i) {
      qobj.addTerm(q_t, vars[i], vars[i]);
    }
    
    model.setObjective(qobj, GRB_MINIMIZE);
    
    // Add constraints
    for (size_t i = 0; i < n; ++i) {
      size_t a_idx = p + i;
      size_t t_idx = p + n + i;
      
      // Constraint 1: t_i + X_i^T beta + a_i >= y_i
      GRBLinExpr expr1 = vars[t_idx] + vars[a_idx];
      for (size_t j = 0; j < p; ++j) {
        expr1 += X(i, j) * vars[j];
      }
      model.addConstr(expr1 >= y(i));
      
      // Constraint 2: t_i - X_i^T beta - a_i >= -y_i
      GRBLinExpr expr2 = vars[t_idx] - vars[a_idx];
      for (size_t j = 0; j < p; ++j) {
        expr2 -= X(i, j) * vars[j];
      }
      model.addConstr(expr2 >= -y(i));
      
      // Constraint 3: t_i >= 0
      model.addConstr(vars[t_idx] >= 0);
    }
    
    // Optimize
    model.optimize();
    
    if (model.get(GRB_IntAttr_Status) != GRB_OPTIMAL) {
      throw std::runtime_error("Gurobi solver failed to find optimal solution");
    }
    
    // Extract beta
    arma::vec beta(p);
    for (size_t i = 0; i < p; ++i) {
      beta(i) = vars[i].get(GRB_DoubleAttr_X);
    }
    
    return beta;
  } catch (GRBException e) {
    Rcpp::stop("Gurobi error code: %d\nReason: %s", e.getErrorCode(), e.getMessage());
  } catch (...) {
    Rcpp::stop("Unknown error in Gurobi optimization");
  }
}
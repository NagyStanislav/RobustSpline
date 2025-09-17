# define ARMA_WARN_LEVEL 1  // Turns off warnings about inverses of nearly singular matrices
# include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Stanislav Nagy
// nagy@karlin.mff.cuni.cz
// 31.08.2025

// [[Rcpp::export()]]
arma::vec psiwC (arma::vec t, int type, double tuning){
// function that gives the weights psi(t)/(2*t) (with 2 in the denominator)
// for reweighting in the IRLS algorithm

// type: the type of the loss function used. Encoding:
//       1 for the absolute loss
//       2 for the square loss
//       3 for the Huber loss, in this case tuning is the constant in the loss
//       4 for the logistic loss
  int n = t.n_elem;
  arma::vec res(n);
  // absolute loss, with slight Huberization at constant tuning
  // for numerical reasons
  if(type == 1){
    if(tuning == 0){  // if tuning == 0 => absolute loss without tuning
      for(int i=0; i<n; i++){
        if(t(i)>0) res(i) = 0.5/t(i);
        if(t(i)<0) res(i) = -0.5/t(i);
        if(t(i)==0) res(i) = R_PosInf; 
      }
    } else {
      for(int i=0; i<n; i++){
        if(abs(t(i))<=tuning){
          res(i) = 0.5/tuning;
        } else {
          if(t(i)>0) res(i) = 0.5/t(i);
          if(t(i)<0) res(i) = -0.5/t(i);
        }
      }
    }
    return(res);
  }
  // square loss
  if(type == 2){
    res.ones();
    return(res);
  }
  // Huber loss
  if(type == 3){
    if(tuning == 0){ // tuning == 0 => absolute loss
      for(int i=0; i<n; i++){
        if(t(i)>0) res(i) = 0.5/t(i);
        if(t(i)<0) res(i) = -0.5/t(i);
        if(t(i)==0) res(i) = R_PosInf;
      } 
    } else {
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
  // logistic loss
  if(type == 4){
    for(int i=0; i<n; i++){
      if(t(i)==0){
        res(i) = 0.5; 
      } else {
        if(abs(t(i))>=500){
        res(i) = 1/abs(t(i)); // if abs(t) is too large, exp(t) might give
        // numerically Inf, but in limit we have that psiw converges to 1/abs(t)
        } else {
        res(i)=(-2*exp(-t(i))/(1+exp(-t(i)))+1)/t(i);
        }
      }
    }
  return(res);
  }
  // If type is unknown returns zeros
  res.zeros();
  return(res);
}

// [[Rcpp::export()]]
Rcpp::List IRLSC (arma::mat Z, arma::vec Y, double lambda, arma::mat H,
  int type, arma::vec W, double sc, arma::vec residsin, double tuning, double toler, int imax){
  
  int ic = 0, istop = 0;
  int n = Y.n_elem, p1 = Z.n_cols;
  arma::vec w(n);
  arma::mat Zt = trans(Z), ZW(p1,n), Z1(p1,p1); // Zt is of dimension p1-times-n
  arma::mat hat_mat(n,n); // hat matrix for hat values
  arma::vec theta_new(p1), resids1(n), hat_diag(n), fitted(n);
  double check = 0;
  
  while(istop == 0 && ic < imax){
    ic++;
    w = W % psiwC(residsin/sc, type, tuning); // element-wise product
    w = w/(sc*sc); // scaling affects only the weights
    for(int j1=0; j1<n; j1++){
      ZW.col(j1) = Zt.col(j1)*w(j1);        
    } // scaled matrix t(Z)%*%diag(W)
    Z1 = ZW*Z + lambda*H;
    theta_new = solve(Z1, ZW*Y); // need to solve the numerical inverse issue
    resids1 = Y - Z*theta_new;
    check = max(abs(resids1-residsin))/sc; 
    if(check < toler){
      istop = 1;
    }
    // Rcpp::Rcout << "It. " << ic << " Check: " << check << " Theta : \n" << theta_new << "\n";
    residsin = resids1;
  }
  if(istop==0) Rcpp::warning("Estimator did not converge.");
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
                            Rcpp::Named("fitted") = fitted);
}

// [[Rcpp::export()]]
Rcpp::List IRLSCmult (arma::mat Z, arma::vec Y, arma::vec lambda, arma::mat H,
  int type, arma::vec W, double sc, double tuning, double toler, int imax){

  int n = Y.n_elem, p1 = Z.n_cols, nlambda = lambda.n_elem;  
  // Rcpp::Rcout << nlambda << std::endl;
  arma::vec istop(nlambda, arma::fill::zeros), ic(nlambda, arma::fill::zeros);
  arma::vec w(n), resids1(n);
  arma::mat Zt = trans(Z), ZW(p1,n), Z1(p1,p1); // Zt is of dimension p1-times-n
  arma::mat hat_mat(n,n); // hat matrix for hat values
  arma::mat resids(n,nlambda, arma::fill::ones), hat_diag(n,nlambda, arma::fill::zeros), fitted(n,nlambda, arma::fill::zeros);
  arma::mat theta_new(p1,nlambda, arma::fill::zeros); 
  double check = 0;
  
  for(int k=0; k<nlambda; k++){
    while(istop(k) == 0 && ic(k) < imax){
      ic(k)++;
      w = W % psiwC(resids.col(k)/sc, type, tuning); // element-wise product
      w = w/(sc*sc); // scaling affects only the weights
      for(int j1=0; j1<n; j1++){
        ZW.col(j1) = Zt.col(j1)*w(j1);        
      } // scaled matrix t(Z)%*%diag(W)
      Z1 = ZW*Z + lambda(k)*H;
      theta_new.col(k) = solve(Z1, ZW*Y, arma::solve_opts::fast); // need to solve the numerical inverse issue
      resids1 = Y - Z*theta_new.col(k);
      check = max(abs(resids1-resids.col(k)))/sc; 
      if(check < toler){
        istop(k) = 1;
      }
      resids.col(k) = resids1;
    }
    // if(istop(k)==0) Rcpp::warning("Estimator did not converge.");
    hat_mat = Z*solve(Z1, ZW, arma::solve_opts::fast);
    fitted.col(k) = hat_mat*Y;
    hat_diag.col(k) = hat_mat.diag();
  }
  return Rcpp::List::create(Rcpp::Named("theta_hat") = theta_new,
                            Rcpp::Named("converged") = istop,
                            Rcpp::Named("ic") = ic,
                            Rcpp::Named("resids") = resids,
                            Rcpp::Named("hat_values") = hat_diag,
                            Rcpp::Named("fitted") = fitted);
}

// [[Rcpp::export()]]
Rcpp::List ridgeC (arma::mat Z, arma::vec Y, double lambda, arma::mat H, arma::vec W){
  
  int n = Y.n_elem, p1 = Z.n_cols;
  arma::mat Zt = trans(Z); // Zt is of dimension p1-times-n
  arma::mat hat_mat(n,n); // hat matrix for hat values
  arma::vec theta_new(p1), residsin(n), hat_diag(n), fitted(n);
  
  theta_new = solve(Zt*arma::diagmat(W)*Z + lambda*H, Zt*arma::diagmat(W)*Y);
  fitted = Z*theta_new;
  residsin = Y - fitted;
  hat_mat = Z*solve(Zt*arma::diagmat(W)*Z + lambda*H, Zt*arma::diagmat(W));
  hat_diag = hat_mat.diag();
  return Rcpp::List::create(Rcpp::Named("theta_hat") = theta_new,
                            Rcpp::Named("resids") = residsin,
                            Rcpp::Named("hat_values") = hat_diag,
                            Rcpp::Named("fitted") = fitted);
}
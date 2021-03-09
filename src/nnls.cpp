// 3/4/2021 Zach DeBruine <zach.debruine@vai.org
// github.com/zdebruine/amf
//
// Thank you for browsing the source code!
// Please raise issues and feature/support requests on github.

#define ARMA_NO_DEBUG

#ifdef _OPENMP
#include <omp.h>
#endif

//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

#include <Rcpp.h>
#include <R.h>

// for microbenchmarking purposes
//[[Rcpp::export]]
arma::vec voidfunc(const arma::mat& a, const arma::vec& b) {
  return(b);
}

//[[Rcpp::export]]
arma::vec armasolve(const arma::mat& a, const arma::vec& b) {
  return(arma::solve(a, b, arma::solve_opts::likely_sympd + arma::solve_opts::fast));
}

// ***************************************************************************************
// SOLVE 2-VARIABLE SYSTEMS

// given sympd a and vector b of length 2, solve by inversion/substitution
arma::vec nnls2(const arma::mat& a, const arma::vec& b, const unsigned int L0 = 2) {

  arma::vec x = arma::zeros(2);
  if (L0 == 1) {
    if (b(0) > b(1)) x(0) = b(0) / a(0, 0);
    else x(1) = b(1) / a(1, 1);
  } else {
    double d = a(0, 0) * a(1, 1) - std::pow(a(0, 1), 2);
    x(0) = (a(1, 1) * b(0) - a(0, 1) * b(1)) / d;
    x(1) = (a(0, 0) * b(1) - a(0, 1) * b(0)) / d;
    if (x(0) < 0) {
      x(0) = 0;
      x(1) = b(1) / a(1, 1);
    } else if (x(1) < 0) {
      x(1) = 0;
      x(0) = b(0) / a(0, 0);
    }
  }
  return(x);
}

// ***************************************************************************************
// FAST + COORDINATE DESCENT NNLS

// Coordinate Descent NNLS (actually bounded variable constraints)
// general framework adapted from Xihui Lin, github linxihui/NNLM/src/scd_ls_update.cpp
arma::vec cdnnls_core(const arma::mat& a, arma::vec& b, arma::vec& x,
  const arma::vec& feasible_set, unsigned int maxit, const double tol, const double min,
  const double max) {

  double tol_i = 1 + tol;
  while (maxit-- > 0 && tol_i > tol) {
    tol_i = 0;
    for (unsigned int i = 0; i < x.n_elem; ++i) {
      if (feasible_set(i) == 1) {
        double xi = x(i) - b(i) / a(i, i);
        // lower bounds check
        if (xi < min) xi = min;
        // upper bounds check
        else if (max != min && xi > max) xi = max;
        if (xi != x(i)) {
          // update gradient
          b += a.col(i) * (xi - x(i));
          // calculate tolerance for this value, update iteration tolerance if needed
          double tol_xi = 2 * std::abs(x(i) - xi) / (xi + x(i) + 1e-16);
          if (tol_xi > tol_i) tol_i = tol_xi;
          x(i) = xi;
        }
      }
    }
  }
  return(x);
}

// subroutine for multinomial cdnnls constraints in "cdnnls"
// return closest number in "values" to x, "values" is ascending and of at least length 2
double round_to_values(const double x, const arma::vec& values) {
  double r = values(0);
  double r_err = std::abs(x - values(0));
  double i_err;
  // ascending search loop, breaks after passing the value of x
  for (unsigned int i = 1; i < values.n_elem; i++) {
    i_err = std::abs(x - values(i));
    if (i_err < r_err) {
      r_err = i_err;
      r = values(i);
    }
    if (values(i) > x) break;
  }
  return(r);
}

// given sympd a, vector b, and initial vector x, use coordinate descent to refine x
// the bulk of this function handles multinomial constraints
// the core of this function relies on the above subroutine (cdnnls_core)
arma::vec cdnnls(const arma::mat& a, arma::vec& b, arma::vec& x, const unsigned int maxit,
  const double tol, const double min, const double max, const arma::vec& values) {

  x = cdnnls_core(a, b, x, arma::ones(x.n_elem), maxit, tol, min, max);
  if (any(values != 0)) {
    // apply multinomial constraints one at a time by matching the largest value
    // to the closest value in the multinomial array and move that value to the 
    // feasible set. Repeat procedure until all values in x are within the 
    // multinomial distribution.
    arma::vec feasible_set = arma::ones(x.n_elem);
    for (unsigned int i = 0; i < x.n_elem; i++) {
      // get index of maximum unfixed value in x
      unsigned int max_index = 0;
      double ind_max = 0;
      for (unsigned int j = 0; j < x.n_elem; j++) {
        if (feasible_set(j) == 1 && x(j) > ind_max) {
          ind_max = x(j);
          max_index = j;
        }
      }
      if (x(max_index) == values.min()) break;
      feasible_set(max_index) = 0;
      x(max_index) = round_to_values(x(max_index), values);
      x = cdnnls_core(a, b, x, feasible_set, maxit, tol, min, max);
    }
  }
  return(x);
}

// finds the minimum index in x not equal to zero
unsigned int min_unbound_index(const arma::vec& x) {
  double min_x = max(x);
  unsigned int min_index = 0;
  for (unsigned int i = 0; i < x.n_elem; ++i) {
    if (x(i) != 0 && x(i) < min_x) {
      min_index = i;
      min_x = x(i);
    }
  }
  return(min_index);
}

//[[Rcpp::export]]
arma::vec trimls(const arma::mat& a, const arma::vec& b, arma::vec& x) {
  // sometimes convex optimization assigns positive values to variables that would actually
  // give a better value if set to zero. Here we "scan" variables beginning with the lowest
  // weighted, proceeding until it is not advantageous to trim one more

  bool trim_more = true;
  double current_err = sum(arma::square(a * x - b));

  while (trim_more) {
    arma::vec x_new = x;
    // get index of minimum non-zero value in x
    unsigned int min_index = min_unbound_index(x);
    // set that value to zero and re-solve, calculate error
    x_new(min_index) = 0;
    arma::uvec ind = find(x_new > 0);
    x_new.elem(ind) = armasolve(a.submat(ind, ind), b.elem(ind));
    double x_new_err = sum(arma::square(a * x_new - b));

    // if error is better, keep it
    if (x_new_err < current_err) {
      x = x_new;
      trim_more = true;
      current_err = x_new_err;
    } else trim_more = false;
  }
  return(x);
}

// given sympd a and vector b, solve system of equations using FAST
//[[Rcpp::export]]
arma::vec fastnnls(const arma::mat& a, const arma::vec& b, const bool trim = true) {
  arma::vec x = armasolve(a, b);
  // repeat refinement of non-zero indices until solutions in nz are strictly positive
  while (any(x < 0)) {
    // keep values greater than 0 in the unconstrained/non-zero set
    arma::uvec nz = find(x > 0);
    // set all other values to zero
    x.zeros();
    // solve unconstrained least squares on unconstrained values
    x.elem(nz) = armasolve(a.submat(nz, nz), b.elem(nz));
  }

  // see if the removal of the minimum values in the solution would improve it
  // this is a common pitfall of convex optimization nnls
  if (trim) x = trimls(a, b, x);
  return(x);
}

// given sympd a and vector b, solve using FAST and refine using cd (if requested)
arma::vec fastcdnnls(const arma::mat& a, arma::vec b, const unsigned int maxit, const double tol,
  const double min, const double max, const arma::vec& values,
  const bool trim = true) {

  // FAST nnls
  arma::vec x = fastnnls(a, b, false);

  // coordinate descent nnls
  arma::vec b0 = a * x - b;
  arma::vec x_cd = cdnnls(a, b0, x, maxit, tol, min, max, values);

  // if the x_cd solution is better than the FAST solution, keep it
  x_err = sum(arma::square(a * x - b));
  x_cd_err = sum(arma::square(a * x_cd - b));
  if (sum(values) > 0 || min != max || min != 0 || x_cd_err < x_err) x = x_cd;

  if (trim) {
    x = trimls(a, b, x);

    // after trimming, rerun coordinate descent
    arma::vec b0 = a * x - b;
    arma::vec x_cd = cdnnls(a, b0, x, maxit, tol, min, max, values);

    // if the x_cd solution is better than the previous solution, keep it
    x_err = sum(arma::square(a * x - b));
    x_cd_err = sum(arma::square(a * x_cd - b));
    if (sum(values) > 0 || min != max || min != 0 || x_cd_err < x_err) x = x_cd;
  }

  return(x);
}

// ***************************************************************************************
// L0 NNLS FUNCTIONS

// Find exact L0 = 1 solution given sympd a and vector b
arma::vec L01nnls(const arma::mat& a, const arma::vec& b) {
  arma::vec x = arma::zeros(b.n_elem);
  arma::vec solutions = b / a.diag();
  arma::vec errors = arma::zeros(solutions.n_elem);
  for (unsigned int j = 0; j < solutions.n_elem; ++j) {
    errors(j) = sum(arma::square(b - a.col(j) * solutions(j)));
  }
  unsigned int best_err = errors.index_min();
  x(best_err) = solutions(best_err);
  return(x);
}

// If !add, binds to zero the non-zero coef in x that increases error least when bound to zero
// if add, unbinds from zero the zero coef in x that decreases error most when unbound
arma::vec change_feasible_set(const arma::mat& a, const arma::vec& b, arma::vec& x,
  const bool add = true) {
  double best_err = 0;
  arma::vec x_new = x;
  for (unsigned int i = 0; i < x.n_elem; ++i) {
    // calculate error for each index if it were set to zero
    bool use_index = (add) ? x(i) == 0 : x(i) != 0;
    if (use_index) {
      arma::vec x2 = x;
      x2(i) = (add) ? 1 : 0;
      arma::uvec ind = find(x2 > 0);
      x2.elem(ind) = armasolve(a.submat(ind, ind), b.elem(ind));
      double xi_err = sum(arma::square(a * x2 - b));
      if (xi_err < best_err || best_err == 0) {
        best_err = xi_err;
        x_new = x2;
      }
    }
  }
  x = x_new;
  return(x);
}

// finds the substitution of a non-zero/zero value pair in x that decreases error most (if any)
arma::uvec find_best_substitution(const arma::mat& a, const arma::vec& b, arma::vec& x) {
  arma::vec x_new(x.n_elem);
  arma::vec x2 = x;
  double current_err = sum(arma::square(a * x - b));
  for (unsigned int i = 0; i < x.n_elem; ++i) {
    if (x(i) != 0) {
      for (unsigned int j = 0; j < x.n_elem; ++j) {
        if (x(j) == 0) {
          // test whether substituting j for i in the feasible set is beneficial
          x_new = x;
          x_new(i) = 0;
          x_new(j) = 1;
          arma::uvec ind = find(x_new > 0);
          x_new(ind) = armasolve(a.submat(ind, ind), b.elem(ind));
          x_new.elem(find(x_new < 0)).zeros();
          double x_new_err = sum(arma::square(a * x_new - b));
          if (x_new_err < current_err) {
            x2 = x_new;
            current_err = x_new_err;
          }
        }
      }
    }
  }
  return(x2);
}

// iteratively set the variable that gives the smallest error to zero until L0 is satisfied, (not
// agglomerative), or iteratively unbinds the variable from zero that gives the best improvement
// in error until L0 is satisfied (agglomerative)
// [[Rcpp::export]]
arma::vec L0partial(const arma::mat& a, const arma::vec& b, const unsigned int L0,
  const bool substitution = false, const bool agglomerative = false) {
  // given a solution, try removing any single variable, re-solve
  arma::vec x = (agglomerative) ? L01nnls(a, b) : fastnnls(a, b);
  unsigned int x_nz;
  if (!agglomerative) {
    arma::vec x_nnz = nonzeros(x);
    x_nz = x_nnz.n_elem;
  } else x_nz = x.n_elem - 1;

  // now repeat truncation/agglomeration procedure (with substitution?) until L0 is 
  // satisfied
  while (x_nz != L0) {
    // find the variable in x that gives the solution with the least error when 
    // removed (not agglomerative) or added (agglomerative)
    x = (agglomerative) ? change_feasible_set(a, b, x, true) : change_feasible_set(a, b, x, false);
    // now try substituting values to reduce mean squared error
    if (substitution) {
      arma::vec x2 = x;
      while (x2 != x) {
        // for each unbound variable, find the best possible substitution. Pick the best overall.
        // repeat until a better substitution cannot be found.
        x = x2;
        x2 = find_best_substitution(a, b, x);
      }
    }
    (agglomerative) ? ++x_nz : --x_nz;
  }
  return(x);
}

// simple remove the smallest coefficient from a solution, re-solve, and repeat until L0 is
// satisfied. Generally not a good idea, but fast in practice and useful for clustering 
// applications where the L0 path is very clear-cut.
//[[Rcpp::export]]
arma::vec L0truncate(const arma::mat& a, const arma::vec& b, const unsigned int L0) {
  arma::vec x = fastnnls(a, b);
  arma::vec x_nnz = nonzeros(x);
  unsigned int x_nz = x_nnz.n_elem;
  while (x_nz > L0) {
    Rcpp::checkUserInterrupt();
    unsigned int min_index = min_unbound_index(x);
    // set minimum non-zero value in x to zero
    x(min_index) = 0;
    arma::uvec ind = find(x > 0);
    // solve 
    x.elem(ind) = armasolve(a.submat(ind, ind), b.elem(ind));
    x_nnz = nonzeros(x);
    x_nz = x_nnz.n_elem;
  }
  return(x);
}

// master partial L0nnls function
// path = 0: truncate
// path = 1: agglomerative
// path = 2: divisive
// path = 3: agglomerative + divisive (take best solution)
// path = 4: agglomerative with substitution
// path = 5: divisive with substitution 
// path = 6: agglomerative + divisive with substitution (take best solution)
arma::vec L0nnls(const arma::mat& a, const arma::vec& b, const unsigned int L0,
  const unsigned int path = 3) {
  if (L0 == 1) return(L01nnls(a, b));
  else if (path == 0) return(L0truncate(a, b, L0));
  else if (path == 1) return(L0partial(a, b, L0, false, true));
  else if (path == 2) return(L0partial(a, b, false, false));
  else if (path == 4) return(L0partial(a, b, true, true));
  else if (path == 5) return(L0partial(a, b, true, false));

  // find both the agglomerative and divisive solution, return the best one
  arma::vec x_agg = L0partial(a, b, L0, path == 6, true);
  arma::vec x_div = L0partial(a, b, L0, path == 6, false);
  double x_agg_err = sum(arma::square(a * x_agg - b));
  double x_div_err = sum(arma::square(a * x_div - b));
  if (x_agg_err < x_div_err) return(x_agg);
  else return(x_div);
}

// subroutine in allcombinations (see immediately below)
// "n" gives the maximum value in a sequence of integers (i.e. 1:10)
// "L0" gives the number of integers in unique sequences (0 < L0 <= n)
// finds all unique combinations of sequence compositions
arma::mat combinations(int n, int L0) {
  std::vector<bool> v(n);
  std::fill(v.end() - L0, v.end(), true);
  arma::vec x;
  do {
    for (int i = 0; i < n; ++i) {
      if (v[i]) {
        x.resize(x.n_elem + 1);
        x(x.n_elem - 1) = i;
      }
    }
  } while (std::next_permutation(v.begin(), v.end()));
  arma::mat y(x);
  y.reshape(L0, x.n_elem / L0);
  return(y);
}

// subroutine in exact nnls
// for a sequence of integers from 1 to "n", get all combinations of unique sequence compositions
// "r" is the maximum cardinality (aka L0) of returned combinations
// code is quite slow, but speed isn't necessarily the point of exact nnls
//[[Rcpp::export]]
arma::umat allcombinations(unsigned int n, unsigned int min_L0 = 1, unsigned int max_L0 = 0) {
  if (max_L0 == 0) max_L0 = n;
  arma::field<arma::umat> res(max_L0 - min_L0 + 1);
  for (unsigned int L0 = min_L0; L0 <= max_L0; ++L0)
    res(L0 - 1) = arma::conv_to<arma::umat>::from(combinations(n, L0));
  // get total number of columns in all field matrices
  unsigned int tot_cols = 0;
  for (unsigned int i = 0; i < res.n_elem; ++i)
    tot_cols += res(i).n_cols;
  // create a new matrix of dimensions n x tot_cols
  arma::umat res_mat(n, tot_cols, arma::fill::zeros);
  // convert indices in the field to 1s in res_mat
  unsigned int res_mat_ind = 0;
  arma::uvec ind_vec(1);
  for (unsigned int i = 0; i < res.n_elem; ++i) {
    arma::umat L0_ind = res(i);
    for (unsigned int j = 0; j < L0_ind.n_cols; ++j) {
      for (unsigned int L0 = 0; L0 < L0_ind.n_rows; ++L0) {
        res_mat(L0_ind(L0, j), res_mat_ind) = 1;
      }
      res_mat_ind++;
    }
  }
  return(res_mat);
}

// exact nnls
// sets is a matrix of column vectors where each vector gives a possible active set to be tested
// values in sets are either 0 or 1. Matrices in this format are generated by allcombinations().
//[[Rcpp::export]]
arma::vec exactnnls(const arma::mat& a, arma::vec b, arma::umat& sets,
  const bool cd = true, const unsigned int maxit = 50, const double tol = 1e-8,
  const bool mse = true) {

  double best_err = sum(b);
  if (mse) best_err = sum(arma::square(b));
  arma::vec x = arma::zeros(b.n_elem);
  for (unsigned int i = 0; i < sets.n_cols; ++i) {
    arma::uvec ind = find(sets.col(i) == 1);
    arma::vec xi = arma::zeros(b.n_elem);
    xi.elem(ind) = arma::solve(a.submat(ind, ind), b.elem(ind),
      arma::solve_opts::likely_sympd + arma::solve_opts::fast);
    xi.elem(find(xi < 0)).zeros();
    double xi_err;
    if (mse) xi_err = sum(arma::square(b - a * xi));
    else xi_err = sum(arma::abs(b - a * xi));
    if (cd) {
      // try refining the solution by coordinate descent (within the feasible set limits)
      arma::vec unbound = arma::conv_to<arma::vec>::from(sets.col(i));
      arma::vec b0 = a * xi - b;
      arma::vec xi2 = cdnnls_core(a, b0, xi, unbound, maxit, tol, 0, 0);
      double xi2_err;
      if (mse) xi2_err = sum(arma::square(b - a * xi2));
      else xi2_err = sum(arma::abs(b - a * xi2));
      if (xi2_err < xi_err) xi = xi2;
    }
    // if the solution gives the best error discovered so far, save it!
    if (xi_err < best_err) {
      x = xi;
      best_err = xi_err;
    }
  }
  return(x);
}

// same as exactnnls, but returns all solution with absolute/squared errors, parallelized across
// columns
// should not be used as a subroutine in parallelized routines
//[[Rcpp::export]]
Rcpp::List exactnnls_fullpath(const arma::mat& a, arma::vec b, arma::umat& sets,
  const bool cd = true, const unsigned int maxit = 500, const double tol = 1e-10,
  const unsigned int trace = 1000, const bool verbose = true) {

  arma::mat x(sets.n_rows, sets.n_cols, arma::fill::zeros);
  arma::mat sqerr = x;
  arma::vec mse(sets.n_cols);

  if (verbose) Rprintf("%10s | %10s \n--------------------------\n", "solved", "remaining");
  for (unsigned int i = 0; i < sets.n_cols; ++i) {
    if (i % trace == 0) {
      Rcpp::checkUserInterrupt();
      if (verbose) Rprintf("%10d | %10d \n", i, sets.n_cols - i);
    }
    arma::uvec ind = find(sets.col(i) == 1);
    arma::vec xi = arma::zeros(b.n_elem);
    xi.elem(ind) = solve(a.submat(ind, ind), b.elem(ind),
      arma::solve_opts::likely_sympd + arma::solve_opts::fast);
    xi.elem(find(xi < 0)).zeros();
    double xi_mse = sum(arma::square(b - a * xi));
    if (cd) {
      // try refining the solution by coordinate descent (within the feasible set limits)
      arma::vec unbound = arma::conv_to<arma::vec>::from(sets.col(i));
      arma::vec xi2 = cdnnls_core(a, b, xi, unbound, maxit, tol, 0, 0);
      double xi2_mse = sum(arma::square(b - a * xi2));
      if (xi2_mse < xi_mse) xi = xi2;
    }
    x.col(i) = xi;
    arma::vec residuals = a * xi - b;
    sqerr.col(i) = arma::square(residuals);
    mse(i) = arma::mean(sqerr.col(i));
  }

  return(Rcpp::List::create(
    Rcpp::Named("x") = x,
    Rcpp::Named("sq_err") = sqerr,
    Rcpp::Named("mse") = mse,
    Rcpp::Named("unbound") = sets));
}

// *************************************************************************************************
// This function adaptively calls the subroutines above
//[[Rcpp::export]]
arma::mat c_sparsels(arma::mat& A, arma::sp_mat& B, arma::mat X, const arma::vec& values,
  unsigned int threads = 0, const double min = 0, const double max = 0,
  const unsigned int maxit = 50, const double tol = 1e-8, double L0 = 0, double L1 = 0,
  double L2 = 0, double PE = 0, const bool exact = false, const bool cd = true,
  const bool mse = false) {

  if (A.n_rows == B.n_rows) inplace_trans(A);

  // if a is not symmetric positive definite, take cross-product
  bool sympd = true;
  arma::mat a(A.n_rows, A.n_rows);
  if (A.n_rows != A.n_cols || !A.is_sympd()) {
    a = A * A.t();
    sympd = false;
  } else a = A;

  if (L2 != 0 || PE != 0) {
    double diag_mean = sum(a.diag()) / a.n_rows;
    double L2penalty = L2 * diag_mean;
    double PEpenalty = PE * diag_mean;
    a += PEpenalty;
    a.diag() += L2penalty - PEpenalty;
  }

  if (sum(sum(X)) == 0) X = arma::mat(A.n_rows, B.n_cols, arma::fill::zeros);
  else {
    // initial X has been provided, verify correct dimensions
    if (a.n_rows != X.n_rows)
      Rcpp::stop("The number of rows in X must be the same as the edge length in 'a'");
    if (X.n_cols != B.n_cols)
      Rcpp::stop("The number of columns in X must be the same as the number of columns in B");
  }

  // don't be doing multithreading on more threads than there are tasks
  if (threads == 0) threads = omp_get_num_procs();
  if (B.n_cols < threads) threads = B.n_cols;

  if (L0 == 0 || L0 > a.n_rows) L0 = a.n_rows;

  // DECIDE WHAT TYPE OF NNLS TO RUN
  // default mode: FAST nnls without refinement by coordinate descent
  arma::umat sets;
  unsigned int mode = 0;
  if (sum(sum(X)) != 0) mode = 3; // use a warm start: coordinate descent
  // 2-variable solution
  else if (a.n_rows == 2 && sum(values) == 0 && max == min && min == 0) mode = 1;
  else if (L0 == 1) mode = 4; // L0 = 1: exact L0 = 1 nnls
  else if (exact || L0 < a.n_rows) {
    mode = 5;
    sets = allcombinations(a.n_rows, 1, L0);
  }
  // cd is requested or required: FAST + CD
  else if (cd || sum(values) > 0 || min != 0 || max != min) mode = 2;

#pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (unsigned int i = 0; i < X.n_cols; ++i) {
    if (mode == 3) {
      arma::vec x = X.col(i);
      arma::vec b0(x.n_rows);
      if (!sympd) b0 = A * B.col(i);
      else b0 = B.col(i);
      if (L1 != 0) b0 -= (L1 * b0.max());
      arma::vec b = a * x - b0;
      X.col(i) = cdnnls(a, b, x, maxit, tol, min, max, values);
    } else {
      arma::vec b(A.n_rows);
      if (!sympd) b = A * B.col(i);
      else b = B.col(i);
      if (L1 != 0) b += (L1 * b.max());
      if (mode == 0) X.col(i) = fastnnls(a, b);
      else if (mode == 1) X.col(i) = nnls2(a, b, L0);
      else if (mode == 2) X.col(i) = fastcdnnls(a, b, maxit, tol, min, max, values);
      else if (mode == 4) X.col(i) = L01nnls(a, b);
      else if (mode == 5) X.col(i) = exactnnls(a, b, sets, cd, maxit, tol, mse);
    }
  }
  return(X);
}

//[[Rcpp::export]]
arma::mat c_sparselm(arma::mat& W, arma::sp_mat& A, arma::mat H, const arma::vec& values,
  unsigned int threads = 0, const double min = 0, const double max = 0,
  const unsigned int maxit = 50, const double tol = 1e-8, double L0 = 0, double L1 = 0,
  double L2 = 0, double PE = 0, const bool exact = false, const bool cd = true,
  const bool mse = false) {

  if (W.n_rows == A.n_rows) inplace_trans(W);

  // if a is not symmetric positive definite, take cross-product
  a = W * W.t();
  if (L2 != 0 || PE != 0) {
    double diag_mean = sum(a.diag()) / a.n_rows;
    double L2penalty = L2 * diag_mean;
    double PEpenalty = PE * diag_mean;
    a += PEpenalty;
    a.diag() += L2penalty - PEpenalty;
  }

  if (sum(sum(H)) == 0) H = arma::mat(W.n_rows, A.n_cols, arma::fill::zeros);
  else {
    // initial H has been provided, verify correct dimensions
    if (a.n_rows != H.n_rows)
      Rcpp::stop("The number of rows in H must be the same as the edge length in 'a'");
    if (H.n_cols != A.n_cols)
      Rcpp::stop("The number of columns in H must be the same as the number of columns in A");
  }

  // don't be doing multithreading on more threads than there are systems to solve
  if (threads == 0) threads = omp_get_num_procs();
  if (A.n_cols < threads) threads = A.n_cols;

  if (L0 == 0 || L0 > a.n_rows) L0 = a.n_rows;

  // DECIDE WHAT TYPE OF NNLS TO RUN
  // default mode: FAST nnls without refinement by coordinate descent
  arma::umat sets;
  unsigned int mode = 0;
  if (sum(sum(H)) != 0) mode = 3; // use a warm start: coordinate descent
  // 2-variable solution
  else if (a.n_rows == 2 && sum(values) == 0 && max == min && min == 0) mode = 1;
  else if (L0 < a.n_rows && !exact) mode = 4;
  else if (exact || L0 < a.n_rows) {
    mode = 5;
    sets = allcombinations(a.n_rows, 1, L0);
  }
  // cd is requested or required: FAST + CD
  else if (cd || sum(values) > 0 || min != 0 || max != min) mode = 2;

  if (mode == 3) {
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int i = 0; i < H.n_cols; ++i) {
      arma::vec x = H.col(i);
      arma::vec b0(x.n_rows);
      if (!sympd) b0 = W * A.col(i);
      else b0 = A.col(i);
      if (L1 != 0) b0 -= (L1 * b0.max());
      arma::vec b = a * x - b0;
      H.col(i) = cdnnls(a, b, x, maxit, tol, min, max, values);
    }
  } else {
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int i = 0; i < H.n_cols; ++i) {
      arma::vec b(W.n_rows);
      if (!sympd) b = W * A.col(i);
      else b = A.col(i);
      if (L1 != 0) b += (L1 * b.max());
      if (mode == 0) H.col(i) = fastnnls(a, b);
      else if (mode == 1) H.col(i) = nnls2(a, b, L0);
      else if (mode == 2) H.col(i) = fastcdnnls(a, b, maxit, tol, min, max, values);
      else if (mode == 4) H.col(i) = L01nnls(a, b);
      else if (mode == 5) H.col(i) = exactnnls(a, b, sets, cd, maxit, tol, mse);
    }
  }
  return(H);
}
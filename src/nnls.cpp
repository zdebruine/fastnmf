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
// the bulk of this function handles L0 truncation and multinomial constraints
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

// given sympd a and vector b, solve system of equations using FAST
arma::vec fastnnls(const arma::mat& a, const arma::vec& b) {
    arma::vec x = solve(a, b, arma::solve_opts::likely_sympd + arma::solve_opts::fast);
    // repeat refinement of non-zero indices until solutions in nz are strictly positive
    while (any(x < 0)) {
        // keep values greater than 0 in the unconstrained/non-zero set
        arma::uvec nz = find(x > 0);
        // set all other values to zero
        x.zeros();
        // solve unconstrained least squares on unconstrained values
        x.elem(nz) = solve(a.submat(nz, nz), b.elem(nz), arma::solve_opts::likely_sympd +
            arma::solve_opts::fast);
    }
    return(x);
}

// given sympd a and vector b, solve using FAST and refine using cd (if requested)
arma::vec fastcdnnls(const arma::mat& a, arma::vec b, const unsigned int maxit, const double tol,
    const double min, const double max, const arma::vec& values) {

    arma::vec x = fastnnls(a, b);
    b = a * x - b;
    x = cdnnls(a, b, x, maxit, tol, min, max, values);
    return(x);
}

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

// Find exact L0 = 1 solution given sympd a and vector b
arma::vec L01nnls(const arma::mat& a, const arma::vec& b) {
    arma::vec x = arma::zeros(b.n_elem);
    arma::vec solutions = b / a.diag();
    arma::vec errors = arma::zeros(solutions.n_elem);
    for (unsigned int j = 0; j < solutions.n_elem; ++j) {
        errors(j) = sum(abs(b - a.col(j) * solutions(j)));
    }
    unsigned int best_err = errors.index_min();
    x(best_err) = solutions(best_err);
    return(x);
}

// subroutine in allcombinations (see immediately below)
// "n" gives the maximum value in a sequence of integers (i.e. 1:10)
// "k" gives the number of integers in unique sequences (0 < k <= n)
// finds all unique combinations of sequence compositions
arma::mat combinations(int n, int k) {
    std::vector<bool> v(n);
    std::fill(v.end() - k, v.end(), true);
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
    y.reshape(k, x.n_elem / k);
    return(y);
}

// subroutine in exact nnls
// for a sequence of integers from 1 to "n", get all combinations of unique sequence compositions
// "r" is the maximum cardinality (aka L0) of returned combinations
// code is quite slow, but speed isn't necessarily the point of exact nnls
arma::umat allcombinations(unsigned int n, unsigned int r = 0) {
    if (r == 0) r = n;
    arma::field<arma::umat> res(r);
    for (unsigned int k = 1; k <= r; k++)
        res(k - 1) = arma::conv_to<arma::umat>::from(combinations(n, k));
    // get total number of columns in all field matrices
    unsigned int tot_cols = 0;
    for (unsigned int i = 0; i < r; i++)
        tot_cols += res(i).n_cols;
    // create a new matrix of dimensions n x tot_cols
    arma::umat res_mat(n, tot_cols, arma::fill::zeros);
    // convert indices in the field to 1s in res_mat
    unsigned int res_mat_ind = 0;
    arma::uvec ind_vec(1);
    for (unsigned int i = 0; i < r; i++) {
        arma::umat k_ind = res(i);
        for (unsigned int j = 0; j < k_ind.n_cols; j++) {
            for (unsigned int k = 0; k < k_ind.n_rows; k++) {
                res_mat(k_ind(k, j), res_mat_ind) = 1;
            }
            res_mat_ind++;
        }
    }
    return(res_mat);
}

// exact nnls
// sets is a matrix of column vectors where each vector gives a possible active set to be tested
// values in sets are either 0 or 1. Matrices in this format are generated by allcombinations().
arma::vec exactnnls(const arma::mat& a, const arma::vec& b, const arma::umat& sets) {
    double best_err = sum(b);
    arma::vec x = arma::zeros(b.n_elem);
    for (unsigned int i = 0; i < sets.n_cols; ++i) {
        arma::uvec ind = find(sets.col(i) == 1);
        arma::vec xi = arma::zeros(b.n_elem);
        xi.elem(ind) = solve(a.submat(ind, ind), b.elem(ind),
            arma::solve_opts::likely_sympd + arma::solve_opts::fast);
        if (all(xi.elem(ind) >= 0)) {
            // if the solution yields exclusively positive values, check error
            double xi_err = sum(abs(b - a * xi));
            if (xi_err < best_err) {
                // if the solution gives the best error discovered so far, save it!
                x = xi;
                best_err = xi_err;
            }
        }
    }
    return(x);
}

// *************************************************************************************************
// This function adaptively calls the subroutines above
//[[Rcpp::export]]
arma::mat c_nnls(arma::mat& A, arma::sp_mat& B, arma::mat X, const arma::vec& values,
    unsigned int threads = 0, const double min = 0, const double max = 0,
    const unsigned int maxit = 50, const double tol = 1e-8, double L0 = 0, double L1 = 0,
    double L2 = 0, double PE = 0, const bool exact = false, const bool cd = true) {

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

    // make sure input won't throw a logical error and crash the session
    if (B.n_cols == A.n_cols && B.n_rows != A.n_cols) B = B.t();
    if (B.n_rows != A.n_cols) Rcpp::stop("dimensions of A and B are incompatible!");

    if (sum(sum(X)) == 0) X = arma::mat(A.n_rows, B.n_cols, arma::fill::zeros);
    else {
        if (X.n_cols == A.n_cols && X.n_rows == B.n_cols) inplace_trans(X);
        else if (X.n_rows != A.n_cols || X.n_cols != B.n_cols)
            Rcpp::stop("dimensions of initial X is incompatible with A and B");
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
        sets = allcombinations(a.n_rows, L0);
    }
    // cd is requested or required: FAST + CD
    else if (cd || sum(values) > 0 || min != 0 || max != min) mode = 2;

#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int i = 0; i < X.n_cols; ++i) {
        if (mode == 3) {
            arma::vec x = X.col(i);
            arma::vec b0(x.n_rows);
            if (!sympd) b0 = a * B.col(i);
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
            else if (mode == 5) X.col(i) = exactnnls(a, b, sets);
        }
    }
    return(X);
}

arma::mat diagnorm(arma::mat& x, arma::vec& d, const unsigned int threads) {
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int i = 0; i < x.n_cols; ++i)
        x.col(i) /= d;
    return(x);
}

double calc_tol(arma::mat& h1, arma::mat& h2, const unsigned int threads) {
    double tol = 0;
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int i = 0; i < h1.n_cols; ++i)
        tol += arma::mean(arma::abs(h1.col(i) - h2.col(i)) / (h1.col(i) + h2.col(i) + 1e-15));
    return(tol);
}

arma::mat diagmult(arma::mat x, arma::vec d, const unsigned int threads) {
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int i = 0; i < x.n_cols; ++i)
        x.col(i) %= d;
    return(x);
}

//[[Rcpp::export]]
Rcpp::List c_nmf(arma::sp_mat& A, const unsigned int k, const double min_w = 0,
    const double min_h = 0, const double max_w = 0, const double max_h = 0,
    const unsigned int cd_maxit = 50, const double cd_tol = 1e-8, const double L0_w = 0,
    const double L0_h = 0, const double L1_w = 0, const double L1_h = 0,
    const double L2_w = 0, const double L2_h = 0, const double PE_w = 0,
    const double PE_h = 0, const bool exact_h = false, const bool exact_w = false,
    const bool cd_w = true, const bool cd_h = true, unsigned int maxit = 100,
    const double tol = 0.01, const unsigned int threads = 0, const bool diag = true,
    const unsigned int seed = 0, const bool verbose = true) {

    if (seed != 0) arma::arma_rng::set_seed(seed);
    arma::mat wt(k, A.n_rows, arma::fill::randu);
    arma::vec d = arma::ones(k);
    arma::mat emptymat = arma::mat(1, 1).zeros();
    arma::vec emptyvec = arma::vec(1).zeros();
    arma::mat h = c_nnls(wt, A, emptymat, emptyvec, threads, min_h, max_h, cd_maxit, cd_tol,
        L0_h, L1_h, L2_h, PE_h, exact_h, cd_h);
    arma::sp_mat At = A.t();
    double tol_it = tol + 1;

    if (verbose) Rprintf("\n%10s | %10s\n----------------------------\n", "iter", "rel tol");

    arma::mat h_new = h;
    unsigned int it = 1;
    for (; it <= maxit && tol_it > tol; ++it) {
        Rcpp::checkUserInterrupt();
        // update wt
        // optionally subsample features
        arma::mat hd = diagmult(h, d, threads);
        wt = c_nnls(hd, At, emptymat, emptyvec, threads, min_w, max_w,
            cd_maxit, cd_tol, L0_w, L1_w, L2_w, PE_w, exact_w, cd_w);

        // update h
        // optionally subsample samples
        arma::mat wtd = diagmult(wt, d, threads);
        h_new = c_nnls(wtd, A, emptymat, emptyvec, threads, min_h, max_h,
            cd_maxit, cd_tol, L0_h, L1_h, L2_h, PE_h, exact_h, cd_h);

        // calculate tolerance between h and h_new
        tol_it = calc_tol(h, h_new, threads) / h.n_cols;
        h = h_new;

        if (verbose) Rprintf("%10d | %10.4f\n", it, tol_it);

        // scale factors in "w" and "h" to sum to 1, update diagonal
        if (diag) {
            arma::vec wt_sums = sum(wt.t());
            arma::vec h_sums = sum(h.t());
            wt = diagnorm(wt, wt_sums, threads);
            h = diagnorm(h, h_sums, threads);
            d %= (wt_sums % h_sums);
        }
    }

    // sort factors by diagonal weights
    inplace_trans(wt);
    if(diag){
        arma::uvec indices = sort_index(d, "descend");
        d = d.elem(indices);
        wt = wt.cols(indices);
        h = h.rows(indices);
    }

    return(Rcpp::List::create(
        Rcpp::Named("w") = wt,
        Rcpp::Named("d") = d,
        Rcpp::Named("h") = h,
        Rcpp::Named("tol") = tol_it,
        Rcpp::Named("iter") = it));
}
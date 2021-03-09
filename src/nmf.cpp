// 3/6/2021 Zach DeBruine <zach.debruine@vai.org
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

// LOSS OF FACTORIZATION
double calc_penalty(const arma::mat& X, const double L1, const double L2, const double PE) {
    double penalty = 0;
    if (L1 != 0) penalty += L1 * arma::sum(arma::sum(X));
    if (L2 != PE) penalty += 0.5 * (L2 - PE) * arma::sum(arma::sum(arma::square(X)));
    if (PE != 0) penalty += 0.5 * PE * arma::sum(arma::sum(X * X.t()));
    return(penalty);
}

double calc_loss(const arma::field<mat> WtH, const arma::vec& D, arma::sp_mat& A,
    const arma::vec L1, const arma::vec L2, const arma::vec PE,
    const unsigned int threads = 0, const bool mse) {

    double tot_loss = 0;
    if (L1(0) > 0 || L2(0) > 0 || PE(0) > 0)
        tot_loss += calc_penalty(WtH(0).t(), L1(0), L2(0), PE(0));
    if (L1(1) > 0 || L2(1) > 0 || PE(1) > 0)
        tot_loss += calc_penalty(WtH(1), L1(1), L2(1), PE(1));

    arma::mat WD = WtH(0) * diagmat(D);
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int j = 0; j < A.n_cols; ++j) {
        arma::vec WDH_j = WD * WtH(1).col(j);
        arma::sp_mat::col_iterator it = A.begin_col(j);
        arma::sp_mat::col_iterator it_end = A.end_col(j);
        for (; it != it_end; ++it) WDH_j(it.row()) -= *it;
        if (mse) tot_loss += arma::sum(arma::square(WDH_j));
        else tot_loss += arma::sum(arma::abs(WDH_j));
    }
    return(tot_loss / A.n_elem);
}

// TOLERANCE OF CONSECUTIVE ITERATIONS
double calc_tol(const arma::field<mat> WtH1, const arma::field<mat> WtH2,
    const unsigned int threads) {

    // calculate relative change in W
    double tol_W = 0;
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int i = 0; i < WtH1(0).n_cols; ++i)
        tol_W += arma::sum(arma::square(arma::abs(WtH1(0).col(i) - WtH2(0).col(i)) /
            (WtH1(0).col(i) + WtH2(0).col(i) + 1e-15)));
    tol_W /= WtH1(0).n_elem;

    // calculate relative change in H
    double tol_H = 0;
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int i = 0; i < WtH1(1).n_cols; ++i)
        tol_H += arma::sum(arma::square(arma::abs(WtH1(1).col(i) - WtH2(1).col(i)) /
            (WtH1(1).col(i) + WtH2(1).col(i) + 1e-15)));
    tol_H /= WtH1(1).n_elem;

    // calculate relative change in W multiplied by relative change in H
    double tol = tol_H * tol_W;
    return(tol);
}

// BASE NNLS ALGORITHM
arma::vec base_nnls(const arma::mat& a, arma::vec& b, const bool cd, unsigned int maxit,
    const double tol) {

    // initial unbounded least squares
    arma::vec x = solve(a, b, arma::solve_opts::likely_sympd + arma::solve_opts::fast);

    // FAST NNLS
    while (any(x < 0)) {
        arma::uvec nz = find(x > 0);
        x.zeros();
        x.elem(nz) = arma::solve(a.submat(nz, nz), b.elem(nz),
            arma::solve_opts::likely_sympd + arma::solve_opts::fast);
    }

    // Coordinate Descent NNLS
    if (cd) {
        b = a * x - b;
        double tol_i = 1 + tol;
        double xi;
        double tol_xi;
        while (maxit-- > 0 && tol_i > tol) {
            tol_i = 0;
            for (unsigned int i = 0; i < x.n_elem; ++i) {
                xi = x(i) - b(i) / a(i, i);
                if (xi < 0) xi = 0;
                if (xi != x(i)) {
                    b += a.col(i) * (xi - x(i));
                    tol_xi = 2 * std::abs(x(i) - xi) / (xi + x(i) + 1e-16);
                    if (tol_xi > tol_i) tol_i = tol_xi;
                    x(i) = xi;
                }
            }
        }
    }
    return(x);
}

arma::mat calc_a(const arma::mat& X, const double L2, const double PE) {
    arma::mat a = X * X.t();
    if (L2 != 0 || PE != 0) {
        double diag_mean = sum(a.diag()) / a.n_rows;
        double L2penalty = L2 * diag_mean;
        double PEpenalty = PE * diag_mean;
        a += PEpenalty;
        a.diag() += L2penalty - PEpenalty;
    }
    return(a);
}

arma::mat multiply_by_D(arma::mat& X, const arma::vec& D, const unsigned int threads) {
    arma::mat XD = X;
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int i = 0; i < XD.n_cols; ++i) XD.col(i) %= D;
    return(X);
}

arma::mat calc_b(const arma::mat& Wt, const arma::mat& HD, const arma::sp_mat& At,
    const unsigned int threads) {

    arma::mat b_Wt = Wt;
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int i = 0, i < WtH(0).n_cols; ++i)
        b_Wt.col(i) = HD * At.col(i);
    return(b_Wt);
}

// INNER ITERATIONS FOR LEAST SQUARES UPDATES
arma::field<mat> inner_loop(arma::field<mat>& WtH, const arma::vec& D, const arma::sp_mat& A,
    const arma::sp_mat& At, const unsigned int inner_maxit, const double inner_tol,
    const bool cd, const unsigned int cd_maxit, const double cd_tol, const arma::vec L1,
    const arma::vec L2, const arma::vec PE) {

    double inner_tol_it = inner_tol_it + 1;
    arma::mat b_Wt;
    arma::mat b_H;
    // the entire purpose of this inner loop is to avoid recalculation of b_Wt and b_H
    // and try to "peek ahead" to refine the solution without this expensive step

    for (unsigned int it = 0; it < inner_maxit && inner_tol_it < inner_tol; ++it) {
        // calculate a for Wt update
        arma::mat a_Wt = calc_a(WtH(1), L2(0), PE(0));

        // calculate H multiplied by the diagonal
        arma::mat HD = multiply_by_D(WtH(1), D, threads);

        // calculate b for Wt updates if inner_iter != 0
        if(it > 0) b_Wt = calc_b(WtH(0), HD, At, threads);

        arma::field<mat> WtH_new = WtH;
        // update Wt 
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (unsigned int i = 0, i < WtH(0).n_cols; ++i)
            WtH_new(0).col(i) = base_nnls(a_Wt, b_Wt.col(i), cd, cd_maxit, cd_tol);

        // calculate a for H update
        arma::mat a_H = calc_a(WtH(0), L2(1), PE(1));

        // calculate Wt multiplied by the diagonal
        arma::mat WtD = multiply_by_D(WtH(0), D, threads);

        // calculate b for H update if inner_iter != 0
        if(it > 0) b_H = calc_b(WtH(1), WtD, A, threads);

        // update H
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (unsigned int i = 0, i < WtH(1).n_cols; ++i)
            WtH_new(1).col(i) = base_nnls(a_H, b_H.col(i), cd, cd_maxit, cd_tol);

        // calculate tolerance between WtH_new and WtH
        inner_tol_it = calc_tol(WtH, WtH_new, threads);
        WtH = WtH_new;
    }

    return(WtH);
}

// NMF FRAMEWORK
//[[Rcpp::export]]
Rcpp::List c_nmf(arma::sp_mat& A, const unsigned int k,
    const arma::vec L0, const arma::vec L1, const arma::vec L2, const arma::vec PE,
    const bool cd, const unsigned int cd_maxit, const double cd_tol, const unsigned int maxit,
    const double tol, const unsigned int threads, const bool diag,
    const unsigned int verbose = 2, const bool path = false, const unsigned int maxit_inner,
    const double tol_inner, const bool loss, const bool low_mem = false) {

    // there's a bit of bonus functionality here, but the idea is simple:
    // 1. randomly initialize W
    // 2. update H
    // 3. alternating updates of W and H until the (change in W)*(change in H) dips below tol
    // 3b. inner iterations repeat least squares without updating the right-hand side of the system
    // 4. with each outer iteration, rows in Wt and H are scaled by a diagonal to sum to 1

    // the idea of repeating inner iterations without updating the right-hand side of the system is 
    // similar to projected gradient, but does not suffer from the same amount of instability and
    // is more easily controlled.

    // ---------------------------------------------------------------------------------------

    // store a pre-transposed copy of A to avoid the repeated cost of transposition
    // low_mem turns on optional storing of a transposed A matrix
    arma::sp_mat At = A.t();
    //    if (!low_mem) At = A.t();

        // the field WtH stores wt and h at 0/1 indices
    arma::field<mat> WtH(2);
    arma::vec D = arma::ones(k);

    // initialize wt randomly
    WtH(0) = arma::mat(k, A.n_rows, arma::fill::randu);
    WtH(1) = arma::mat(k, A.n_cols, arma::fill::zeros);

    // initialize wt by solving for h once
    WtH = inner_loop(WtH, D, A, At, inner_maxit, inner_tol, cd, cd_maxit, cd_tol, L1, L2, PE);
    arma::field<mat> WtH_new = WtH;

    double tol_it = tol + 1;

    // structures for recording full solution path
    arma::cube  path_W(A.n_rows, k, maxit);
    arma::mat   path_D(k, maxit);
    arma::cube  path_H(k, A.n_cols, maxit, arma::fill::zeros);
    arma::vec   path_tol(maxit);
    arma::vec   path_loss(maxit);
    arma::vec   path_loss_tol(maxit);

    if (loss) loss_it = mean(mean(square(A - wt.t() * h)));

    // work on a progress bar against log tol (verbose = 1)
    if (verbose == 2) Rprintf("%10s | %10s \n----------------------\n", "iter", "tol");

    unsigned int it = 1;
    for (; it <= maxit && tol_it > tol; ++it) {
        Rcpp::checkUserInterrupt();
        // update WtH
        WtH_new = inner_loop(WtH, D, A, At, inner_maxit, inner_tol, cd, cd_maxit, cd_tol,
            L1, L2, PE);

        // calculate tolerance between WtH and WtH_new
        tol_it = calc_tol(WtH, WtH_new, threads);
        WtH = WtH_new;

        // scale factors in W and H to sum to 1, update diagonal
        if (diag) {
            arma::vec Wt_sums = sum(WtH(0).t());
            arma::vec H_sums = sum(WtH(1));

            // normalize rows in Wt to Wt_sums
#pragma omp parallel for num_threads(threads) schedule(dynamic)
            for (unsigned int i = 0; i < WtH(0).n_cols; ++i) WtH(0).col(i) /= Wt_sums;

            // normalize rows in H to H_sums
#pragma omp parallel for num_threads(threads) schedule(dynamic)
            for (unsigned int i = 0; i < WtH(1).n_cols; ++i) WtH(1).col(i) /= H_sums;

            // adjust scaling diagonal to reflect changes in Wt_sums and H_sums
            D %= (Wt_sums % H_sums);
        }

        if (loss) {
            loss_it_prev = loss_it;
            loss_it = c_loss_nmf(WtH(0).t(), D, WtH(1), A, L1, L2, PE, threads);
            loss_tol = 2 * (std::abs(loss_it - loss_it_prev) / (loss_it + loss_it_prev + 1e-15));
        }

        // record intermediate solutions if path = true
        if (path) {
            path_W.slice(it - 1) = wt.t();
            path_D.col(it - 1) = d;
            path_H.slice(it - 1) = h;
            path_tol(it - 1) = tol_it;
            path_loss(it - 1) = loss;
            path_loss_tol(it - 1) = loss_tol;
        }

        // verbose updates
        if (verbose == 2) Rprintf("%10d | %10.5f \n", it, tol_it);
    }

    if (!full_path) {
        return(Rcpp::List::create(
            Rcpp::Named("w") = WtH(0).t(),
            Rcpp::Named("d") = D,
            Rcpp::Named("h") = WtH(1),
            Rcpp::Named("tol") = tol_it,
            Rcpp::Named("loss") = loss,
            Rcpp::Named("loss_tol") = loss_tol,
            Rcpp::Named("iter") = it
        ));
    } else {
        return(Rcpp::List::create(
            Rcpp::Named("w") = path_W,
            Rcpp::Named("d") = path_D.cols(0, it - 2),
            Rcpp::Named("h") = path_H,
            Rcpp::Named("tol") = path_tol(arma::span(0, it - 2)),
            Rcpp::Named("loss") = path_loss(arma::span(0, it - 2)),
            Rcpp::Named("loss_tol") = path_loss_tol(arma::span(0, it - 2))
        ));
    }
}
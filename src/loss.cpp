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

double add_penalty(const arma::mat& X, const double L1, const double L2, const double PE) {
    double penalty = 0;
    if (L1 != 0) penalty += L1 * arma::sum(arma::sum(X));
    if (L2 != PE) penalty += 0.5 * (L2 - PE) * arma::sum(arma::sum(arma::square(X)));
    if (PE != 0) penalty += 0.5 * PE * arma::sum(arma::sum(X * X.t()));
    return(penalty);
}

//[[Rcpp::export]]
Rcpp::List c_loss_rcpp(const arma::mat& w, const arma::vec& d, const arma::mat& h,
    arma::sp_mat& A, const arma::vec L1, const arma::vec L2, const arma::vec PE,
    const unsigned int threads = 0, const bool mse) {

    double penalty = 0;
    if (L1(0) > 0 || L2(0) > 0 || PE(0) > 0) penalty += add_penalty(w, L1(0), L2(0), PE(0));
    if (L1(1) > 0 || L2(1) > 0 || PE(1) > 0) penalty += add_penalty(h, L1(1), L2(1), PE(1));
    penalty /= A.n_elem;

    // calculate total loss for all samples
    arma::vec losses = arma::zeros(A.n_cols);
    arma::mat wd = w * diagmat(d);
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int j = 0; j < A.n_cols; ++j) {
        arma::vec wdh_j = wd * h.col(j);
        arma::sp_mat::col_iterator it = A.begin_col(j);
        arma::sp_mat::col_iterator it_end = A.end_col(j);
        for (; it != it_end; ++it) wdh_j(it.row()) -= *it;
        if (mse) losses(j) = arma::mean(arma::square(wdh_j));
        else losses(j) = arma::mean(arma::abs(wdh_j));
    }
    double net_loss = arma::mean(losses);

    return(Rcpp::List::create(
        Rcpp::Named("loss") = net_loss,
        Rcpp::Named("penalty") = penalty,
        Rcpp::Named("tot_loss") = net_loss + penalty,
        Rcpp::Named("sample_losses") = losses));
}
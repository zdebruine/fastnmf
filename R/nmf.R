#' @title Non-negative matrix factorization
#'
#' @description NMF by alternating constrained least squares with fast solvers and an efficient 
#' stopping criteria.
#'
#' @details
#' NMF solves the equation \deqn{A = WH} by minimizing \eqn{A - WH}. In NMF by alternating least 
#' squares, this objective is implicitly minimized and is thus never explicitly calculated. Rather,
#' progress towards the objective is measured indirectly by the rate of change in \eqn{H} between 
#' consecutive iterations.
#'
#' ## Model diagonalization
#' By default, \code{amf::nmf} scales columns in \eqn{w} and rows in \eqn{h} to sum to 1 by 
#' introducing a "scaling diagonal":
#' \deqn{A = WDH}
#' This is important in several ways:
#' * **interpretability**: it is easy to see which features (or samples) are most important in a
#' factor, not as a proportion of the total signal explained by the model, but as a proportion of 
#' contribution to that specific factor/signal.
#' * **regularization**: regularization is performed on the normalized matrices, thus regularization
#' in a diagonalized model penalizes signals equally and not as a proportion of their relative
#' contribution towards overall signal.
#' * **multiple model alignment**: factorizations from multiple random starts rarely align.
#' However, after sorting along the diagonal, they are much more likely to naturally align.
#' 
#' ## Model regularization
#' Regularization, especially sparsifying regularization (i.e. L0, L1, PE) challenges the discovery
#' of the best model solution. It may be most useful to first solve the unpenalized model, and then 
#' add regularizations one at a time.
#' 
#' ## Exact solutions
#' The challenges of exact nnls have been reviewed in \code{link{nnls}}. Exact NNLS is NP-hard 
#' because it considers every possible feasible set.
#'
#' ## NNLS solvers used
#' At present, only cold-start solvers are used (i.e. FAST, FAST + CD, nnls2, L0 = 1, and exact). 
#' In the future, support will be added for warm-start solvers if they are faster.
#'
#' @param A matrix of features (rows) by samples (columns) to be factorized. Provide in dense 
#' or sparse format.
#' @param k rank of factorization
#' @param min lower bound on solution, usually 0 (default). Vector of two for \code{c(w, h)}
#' @param max upper bound on solution, if applicable (NULL, default). Vector of two for 
#' \code{c(w, h)}.
#' @param L0 cardinality of least squares solutions, if other than full-rank (NULL, default). If 1,
#' an efficient exact solver is used. If 2 or greater, all possible feasible sets are sampled using 
#' exact NNLS (NP-hard). Specify with caution, see details. Vector of two for \code{c(w, h)}
#' @param L1 LASSO penalty to be subtracted from right-hand side of coefficients in least squares 
#' updates, scaled to the maximum value in the right-hand side of each least squares problem. 
#' Vector of two for \code{c(w, h)}.
#' @param L2 Ridge regression penalty to be added to diagonal of \eqn{a}. Scaled to mean diagonal 
#' value in coefficient matrix. Vector of two for \code{c(w, h)}.
#' @param PE Pattern extraction penalty to be added to off-diagonal values in least squares 
#' updates, scaled to mean diagonal value of the coefficient matrix. Vector of two for \code{c(w, h)
#' }.
#' @param exact calculate exact solution? See \code{\link{nnls}} for details. Use with caution. 
#' Vector of two for \code{c(w, h)}.
#' @param cd use coordinate descent to refine the solution? Vector of two for \code{c(w, h)}
#' @param maxit maximum number of alternating updates of \eqn{w} and \eqn{h}
#' @param maxit_cd in coordinate descent, maximum number of iterations
#' @param tol relative change in \eqn{h} between consecutive iterations at convergence
#' @param tol_cd in coordinate descent, distance of solution from maximum gradient residual at 
#' convergence
#' @param diag use a diagonal to scale columns in \eqn{w} and rows in \eqn{h} to sum to 1 (default 
#' \code{TRUE})
#' @param seed seed for random initialization of \eqn{w}
#' @param threads use column-wise parallelization for each update of \eqn{w} and \eqn{h}. Specify 
#' either 0 (all available threads, default) 1 (no parallelization), or any valid number.
#' @param verbose show tolerance for each iteration
#' @param ... additional parameters
#' @returns A list of \code{w}, \code{d}, and \code{h} giving the factorization result, and 
#' \code{iter} giving the number of iterations required to converge, and \code{tol} giving the 
#' relative change in \eqn{h} between the last two iterations.
#' @export
#' @seealso \code{\link{nnls}}
#' @md
#' @import Matrix RcppArmadillo
#' @importFrom methods as
#'
nmf <- function(A, k, min = c(0, 0), max = c(NULL, NULL), L0 = c(NULL, NULL), L1 = c(0, 0),
    L2 = c(0, 0), PE = c(0, 0), exact = c(FALSE, FALSE), cd = c(TRUE, TRUE), maxit = 100,
    maxit_cd = 100, tol = 0.01, tol_cd = 1e-8, diag = TRUE, seed = NULL, threads = 0,
    verbose = TRUE, ...) {

  if (is.null(seed)) seed <- 0
  if (is.null(L0[1])) L0[1] <- k
  if (is.null(L0[2])) L0[2] <- k
  if (is.null(max)) max <- c(min[1], min[2])
  A <- as(A, "dgCMatrix")
  
  check_sanity <- function(n, L0) {
    msg <- "You have requested an exact solution in a very large feasible set search space. 
  Please be sure you wish to proceed. If so, specify \"check_exact = FALSE\" in your function call."
    if (L0 != n) {
      if ((L0 > 10 && n > 15) || (L0 > 7 && n > 50) || (L0 > 5 && n > 100) || (L0 > 3 && n > 200) ||
          (L0 > 1 && n > 500)) stop(msg)
    } else if (n > 15) stop(msg)
  }
  
  if (any(exact == TRUE) || any(L0 != k)) {
    check_exact <- is.null(list(...)$check_exact)
    if (check_exact) check_sanity(k, L0[1])
    if (check_exact) check_sanity(k, L0[2])
  }
  return(c_nmf(A, k, min[1], min[2], max[1], max[2], maxit_cd, tol_cd, L0[1], L0[2], L1[1], L1[2],
  L2[1], L2[2], PE[1], PE[2], exact[1], exact[2], cd[1], cd[2], maxit, tol, threads, diag, seed,
  verbose))
}
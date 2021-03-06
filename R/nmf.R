#' @title Alternating matrix factorization
#'
#' @description Non-negative matrix factorization, or other constrained factorization, by
#' alternating constrained least squares with fast solvers and efficient stopping criteria.
#'
#' @details
#' NMF solves the equation \deqn{A = WH} by minimizing \eqn{A - WH}. In NMF by alternating least 
#' squares, this objective is implicitly minimized and is thus never explicitly calculated. Rather,
#' progress towards the objective is measured indirectly by the rate of change in \eqn{H} between 
#' consecutive iterations.
#'
#' ## Model diagonalization
#' By default, \code{nmf} scales columns in \eqn{w} and rows in \eqn{h} to sum to 1 by 
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
#' @param loss One of c(\code{NULL}, \code{"mse"}, \code{"mae"}) to either not calculate loss 
#' (which is computationally expensive) or to calculate mean squared or absolute error of the 
#' factorization every \code{trace} iterations (and potentially use loss as a convergence criteria 
#' by specifying \code{tol_loss}). NULL by default, because loss calculations are slower than the 
#' factorization itself.
#' @param tol_wh stop factorizing when the relative change in \eqn{h} multiplied by the relative
#' change in \eqn{w} between consecutive iterations falls below this value.
#' @param tol_loss stop factorizing when the relative change in loss between consecutive iterations 
#' falls below this value. If \code{loss} is not NULL, loss tolerance will be calculated.
#' @param trace calculate tolerances and record \code{full_path} solutions (if applicable) every 
#' trace iterations (default 1)
#' @param tol_cd in coordinate descent, distance of solution from maximum gradient residual at 
#' convergence
#' @param diag use a diagonal to scale columns in \eqn{w} and rows in \eqn{h} to sum to 1 (default 
#' \code{TRUE})
#' @param trace calculate applicable tolerances / losses every trace iterations, and check for 
#' convergence
#' @param seed seed for random initialization of \eqn{w}
#' @param threads use column-wise parallelization for each update of \eqn{w} and \eqn{h}. Specify 
#' either 0 (all available threads, default) 1 (no parallelization), or any valid number.
#' @param verbose show tolerance for each iteration
#' @param full_path return all intermediate models in addition to the final model (default FALSE)
#' @param tol_wh_switch_to_cd begin using coordinate descent solver exclusively at this tolerance,  
#' as opposed to cold-start consisting of FAST initialization followed by coordinate descent.
#' @param ... additional parameters
#' @returns An \code{\link{nmfmodel}} object with slots:
#' * \code{w}: matrix of \emph{features x factors} of dimensions \emph{m x k}
#' * \code{d}: diagonal scaling vector of length \emph{k}
#' * \code{h}: matrix of \emph{factors x samples} of dimensions \emph{k x n}
#' * \code{loss}: mean squared/absolute error of the model (if \code{loss} is not NULL)
#' * \code{tol_wh}: tolerance of change in \emph{w} multiplied by the change in \emph{h} between 
#' the last two consecutive iterations
#' * \code{tol_loss}: mean squared/absolute error of the model (if \code{loss} is not NULL)
#' * \code{it}: number of iterations used to solve the model
#' * \code{path}: a list of \code{\link{nmfmodel}} objects containing intermediate model states 
#' along the full solution path (if \code{full_path = TRUE})
#' @export
#' @seealso \code{\link{nnls}}, \code{\link{nmfmodel-class}}
#' @md
#' @import Matrix RcppArmadillo
#' @importFrom methods as
#'
nmf <- function(A, k, min = 0, max = NULL, L0 = NULL, L1 = 0,
    L2 = 0, PE = 0, exact = FALSE, cd = FALSE, trace = 1,
    maxit = 100, maxit_cd = 100, tol_cd = 1e-8, diag = TRUE, seed = NULL, threads = 0,
    verbose = TRUE, full_path = FALSE, tol_wh = 0.001, loss = NULL,
    tol_loss = 0.001, ...) {

  if (is.null(seed)) set.seed(seed)
  if (is.null(L0)) L0 <- c(k, k)
  if (is.null(L0[1])) L0[1] <- k
  if (is.null(L0[2])) L0[2] <- k
  if (is.null(min)) min <- c(0, 0)
  if (length(min) == 1) min <- c(min, min)
  if (is.null(max)) max <- c(min[1], min[2])
  if (is.null(max)) max <- c(min[1], min[2])
  if (length(min) == 1) min <- c(min, min)
  if (length(L1) == 1) L1 <- c(L1, L1)
  if (length(L2) == 1) L2 <- c(L2, L2)
  if (length(PE) == 1) PE <- c(PE, PE)
  if (length(exact) == 1) exact <- c(exact, exact)
  if (length(cd) == 1) cd <- c(cd, cd)
  A <- as(A, "dgCMatrix")
  if (is.null(loss)) loss <- 0 else if (loss == "mse") loss <- 1 else loss <- 2
  if (is.null(tol_loss)) tol_loss <- 0.001

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

  mod <- c_nmf(A, k, min[1], min[2], max[1], max[2], maxit_cd, tol_cd, L0[1], L0[2], L1[1], L1[2],
  L2[1], L2[2], PE[1], PE[2], exact[1], exact[2], cd[1], cd[2], maxit, tol_wh, threads, diag,
  verbose, full_path, loss, tol_loss, trace)

  if (!full_path) {
    result <- new("nmfmodel")
    result@w <- mod$w
    result@h <- mod$h
    result@d <- as.vector(mod$d)
    result@it <- mod$it
    result@tol_wh <- mod$tol_wh
    if (loss != 0) result@loss <- mod$loss
    if (tol_loss != 0) result@tol_loss <- mod$tol_loss
  } else {
    models <- list()
    for (i in 1:length(mod$it)) {
      models[[i]] <- new("nmfmodel")
      models[[i]]@w <- mod$w[,, i]
      models[[i]]@h <- mod$h[,, i]
      models[[i]]@d <- as.vector(mod$d[, i])
      models[[i]]@it = mod$it[i]
      models[[i]]@tol_wh <- mod$tol_wh[i]
      if (loss != 0) models[[i]]@loss <- mod$loss[i]
      if (tol_loss != 0) models[[i]]@tol_loss <- mod$tol_loss[i]
    }
    result <- models[[length(models)]]
    result@path <- models
  }
  if (mean(result@d) != 1) {
    # sort final model by diagonal value
    d_order <- order(result@d, decreasing = TRUE)
    result@w <- result@w[, d_order]
    result@h <- result@h[d_order,]
    result@d <- result@d[d_order]
  }
  return(result)
}

#' The "nmfmodel" class
#'
#' The nmfmodel object holds an nmf factor model and basic statistics computed about the model
#'
#' @slot w matrix of \emph{features x factors} of dimensions \emph{m x k} 
#' @slot d diagonal scaling vector of length \emph{k}
#' @slot h matrix of \emph{factors x samples} of dimensions \emph{k x n}
#' @slot loss mean squared/absolute error of the model
#' @slot tol_wh relative change in \eqn{h} across consecutive iterations multiplied by the 
#' relative change in \eqn{w} across consecutive iterations.
#' @slot tol_loss relative change in loss across consecutive iterations.
#' @slot it number of iterations used to solve the model
#' @slot path a list of \code{nmfmodel} objects containing intermediate model states along the full
#' solution path, if they were requested in \code{\link{nmf}}
#' @name nmfmodel
#' @aliases nmfmodel, nmfmodel-class
#' @exportClass nmfmodel
#' @seealso \code{\link{nmfmodel}}
#'
setClass("nmfmodel",
    representation(
      w = "matrix",
      d = "numeric",
      h = "matrix",
      loss = "numeric",
      tol_wh = "numeric",
      tol_loss = "numeric",
      it = "numeric",
      path = "list"),
    prototype(
      w = matrix(),
      d = numeric(),
      h = matrix(),
      loss = NA_real_,
      tol_wh = NA_real_,
      tol_loss = NA_real_,
      path = list()),
      validity = function(object) {
        msg <- NULL
        if (ncol(object@w) != nrow(object@h))
          msg <- c(msg, "ranks of 'w' and 'h' are not equal")
        if (ncol(object@w) != length(object@d) || ncol(object@h) != length(object@d))
          msg <- c(msg, "ranks of 'd' and 'w' and/or 'h' are not equal")
        if (!is.null(path) || length(path) > 0)
          if (any(sapply(path, class) != "nmfmodel"))
            msg <- c(msg, "path does not strictly contain objects of class 'nmfmodel'")
        if (is.null(msg)) TRUE else msg
      }
)
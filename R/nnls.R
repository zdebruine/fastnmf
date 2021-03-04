#' @title Non-negative least squares
#'
#' @description Solve a system of equations with non-negative constrained least squares. 
#'
#' @details
#' Solve the equation \eqn{ax = b} for \eqn{x} when \eqn{a} is symmetric positive definite.
#' Solves the equation \eqn{aa^Tx = ab_j} for all columns \eqn{j} in \eqn{b} when \eqn{a} 
#' is rectangular.
#'
#' ## Adaptive Solvers
#' \code{nnls} is an adaptive solver with several subroutines that may be called based on the
#' requirements of the solution. The entire solver is written in C++ using the Armadillo
#' library with OpenMP parallelization across columns in \eqn{b} (if \eqn{b} is a matrix). 
#' 
#' * **fast approximate solution trajectory** (FAST) non-negative least squares. 
#' Used to initialize coordinate descent if no initial \code{x_init} is provided, but also gives
#' a near-exact solution in well-conditioned systems with excellent speed. Set \code{cd = FALSE} 
#' to use only the FAST solver.
#' * **coordinate descent** (cd) finds the best solution discoverable by convex optimization 
#' given an initial start (either \eqn{x_init} or the result from FAST). The coordinate descent
#' solver also handles bounded variable constraints (\code{min}, \code{max}), and multinomial 
#' constraints (\code{values}). Coordinate descent may be disabled by setting \code{cd = FALSE} 
#' to use FAST only, but will only be disabled if \code{min = 0, max = NULL, values = NULL}.
#' * **singular L0**. Exact solutions with singular cardinality are found efficiently when L0 = 1
#' * **2-variable solutions** are found exactly by hard-coded inversion and substitution.
#' * **exact nnls**: NNLS is often not a perfectly convex problem, and thus the global solution
#' minimum is often not discoverable by convex optimization. Exact NNLS finds the true solution
#' by solving the solution for every possible feasible set. Note that the number of possible
#' solutions increases exponentially with the number of variables in the system (NP-hard). Exact 
#' NNLS is only reasonable for problems of up to ~15 variables or for correspondingly small L0 
#' cardinalities. Exact NNLS is the only supported method for finding L0 solutions, as methods for 
#' approximating L0 are very inaccurate.
#'
#' ## Input Requirements
#' There are two formats for input matrices. The first corresponds to the equation:
#' 
#' \deqn{ax = b}
#' 
#' where \eqn{a} is symmetric positive definite \emph{k x k}, and \emph{x} and \emph{b} are vectors
#' of length \emph{k} or equidimensional matrices with \emph{k} rows.
#' 
#' The second format corresponds to the classical alternating matrix factorization update problem:
#' 
#' \deqn{A = WH}
#' 
#' When updating \eqn{H}, we have:
#' 
#' \deqn{W^TWH = WA_j}
#' \deqn{a = W^TW}
#' \deqn{x = H}
#' \deqn{b = WA_j}
#'
#' for all columns \eqn{j} in \eqn{A}. 
#' In \code{nnls}, \eqn{A} may be specified as \eqn{b}, \eqn{W^T} as \eqn{a}, and \eqn{H} as 
#' \eqn{x_init}. Note that when \code{nnls} checks whether \eqn{W} is symmetric positive definite, it 
#' pivots to the above interpretation of parameters.
#' 
#' In the case above, \eqn{a} is of dimensions \emph{k x m}, \eqn{b} is of dimensions \emph{m x n}, 
#' and \eqn{x} is of dimensions {k x n}. \code{nnls} will try to rearrange \eqn{a}, \eqn{b}, and 
#' \eqn{x} to align along common edges if improper dimensions are provided.
#' 
#' The corresponding equation for updating \eqn{W} in block-pivoting is:
#' 
#' \deqn{HH^TW^T = HA^T_j}
#'
#' @param a symmetric positive definite \emph{k x k} matrix or rectangular \emph{k x m} matrix 
#' giving the coefficients of the linear system.
#' @param b vector or dense/sparse matrix giving right-hand side(s) of the linear system. If \eqn{a}
#' is symmetric positive definite, \eqn{b} must be a positive vector of length {k}. If \eqn{a} is
#' of dimensions \emph{k x m}, \emph{b} must be a dense/sparse matrix of dimensions \emph{m x n} or 
#' a vector of length \emph{m}.
#' @param x_init if provided, initial value for x. If \eqn{a} is symmetric positive definite, 
#' \eqn{x} must be of the same dimensions as \eqn{b}. If \eqn{a} is rectangular, \eqn{x} must be a 
#' matrix of dimensions \emph{k x n}. If a non-zero \eqn{x_init} is provided, coordinate descent 
#' will refine the solution immediately without initialization by FAST.
#' @param values multinomial array (including 0) to which variables in the solution will be 
#' constrained, if applicable (NULL, default)
#' @param min lower bound on solution, usually 0 (default)
#' @param max upper bound on solution, if applicable (NULL, default)
#' @param L0 cardinality of the solution, if other than full-rank (NULL, default). If 1, an 
#' efficient exact solver is used. If 2 or greater, all possible feasible sets are sampled using 
#' exact NNLS (NP-hard). Specify with caution, see section on exact NMF in details.
#' @param L1 LASSO penalty to be subtracted from \code{b}. Scaled to maximum value in each column 
#' of \code{b}.
#' @param L2 Ridge regression penalty to be added to diagonal of \code{a}. Scaled to mean diagonal 
#' value in \code{a}.
#' @param PE Pattern Extraction penalty to be added to off-diagonal values in \code{a}. Scaled 
#' to mean diagonal value in \code{a}.
#' @param threads If \code{b} is a matrix, apply column-wise parallelization by specifying either 0 
#' (all available threads, default) or a number other than 1.
#' @param maxit in coordinate descent, maximum number of iterations
#' @param tol in coordinate descent, distance of solution from maximum gradient residual at 
#' convergence
#' @param exact find an exact solution (default FALSE). Use with caution. Extremely slow and 
#' possibly intractable. See details. This is the only method for \code{L0} solving.
#' @param cd Should coordinate descent be used to refine the initial solution approximated by FAST?
#' By default this is true, but for well-conditioned systems FAST often gives a very good solution
#' and cd can be disabled for speed gains. See details.
#' @param ... additional parameters
#' @returns \eqn{x}, a vector or matrix containing constrained solutions to the linear system(s)
#' @export
#' @md
#' @import Matrix RcppArmadillo
#' @importFrom methods as
#' @examples
#' A <- matrix(runif(25*10), 25, 10)
#' B <- matrix(runif(25*1000)*0.1, 25, 1000)
#' X <- nnls(A, B)
#'
nnls <- function(a, b, x_init = NULL, min = 0, max = NULL, values = NULL, L0 = NULL, L1 = 0,
    L2 = 0, PE = 0, threads = 0, maxit = 100, tol = 1e-8, exact = FALSE, cd = TRUE, ...) {

  check_exact <- is.null(list(...)$check_exact)
  if (is.null(max)) max <- min
  if (is.null(values)) values <- c(0)
  if (is.null(x_init)) x_init <- matrix(0, 1, 1)
  if ("numeric" %in% class(b)) b <- as.matrix(b)
  if (!("dgcMatrix" %in% class(b))) b <- as(b, "dgCMatrix")
  if (is.null(L0)) L0 = 0
  if (check_exact && L0 != 0) {
    n <- ncol(a)
    msg <- "You have requested an exact solution in a very large feasible set search space. 
  Please be sure you wish to proceed. If so, specify \"check_exact = FALSE\" in your function call."
    if (L0 != n) {
      if ((L0 > 10 && n > 15) || (L0 > 7 && n > 50) || (L0 > 5 && n > 100) || (L0 > 3 && n > 200) ||
          (L0 > 1 && n > 500)) stop(msg)
    } else if (n > 15) stop(msg)
  }
  return(c_nnls(a, b, x_init, values, threads, min, max, maxit, tol, L0, L1, L2, PE, exact, cd))
}

check_sanity <- function(n, L0) {
}

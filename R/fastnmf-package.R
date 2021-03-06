#' fastnmf: fast non-negative matrix factorization
#'
#' @description
#' Fast and adaptive solvers for constrained non-negative least squares and matrix factorization by alternating constrained least squares.
#' 
#' @details 
#' The fastnmf package is built around two functions: \code{nnls} and \code{nmf}. 
#' 
#' \code{nnls} is an adaptive solver for non-negative least squares with unprecedented efficiency and flexibility.
#' \code{nmf} uses the \code{nnls} engine to run non-negative matrix factorization, making use of several new ideas to both increase speed and reduce iterations needed for convergence.
#'
#' @section Why is fastnmf fast?:
#' * thing 1
#' * thing 2
#' * thing 3
#'
#' @section Non-negative least squares:
#' d
#' 
#' @section Get started!:
#' Vignette
#'
#' @section Advanced tutorials:
#' * Vignette:
#' * Vignette:
#' * Vignette:
#'
#' @useDynLib fastnmf, .registration = TRUE
#' @docType package
#' @importFrom Rcpp evalCpp
#' @name fastnmf
#' @aliases fastnmf, fastnmf-package
#' @author Zach DeBruine
#' @md
#'
"_PACKAGE"
# sparselm
Sparse linear models for fast learning

## Sparse least squares
Non-negative, regularized, and near-exact L0-regularized least squares `(ax = b)` with new algorithms that give better and faster solutions than any other method.

## Sparse linear models
Factor model projection `(A = WH)` using sparse least squares with sparse matrix support. 

## Sparse matrix factorization
Non-negative or constrained matrix factorization by alternating least squares with several critical improvements for computational efficiency. Adaptive introduction of sparsifying regularizations to facilitate model equilibration and fast convergence towards the best discoverable solution.

All functions receive a well-documented R interface and a parallelized RcppArmadillo backend.

ISSUES:
Regularizations causing instability in nmf

SMALL TO DOs:
S4 "show" function for objects of class "nmfmodel"
S4 method for nmfmodel to wrap c_loss_sample data into an nmfmodel
S4 plot method for "nmfmodel" class that wraps mheatmap
Wrap detailed nmf/nnls parameters into S4 "nmfParams" and "nnlsParams" objects

LARGER TO DOs:
SNAIL NNLS for better L0 approximation

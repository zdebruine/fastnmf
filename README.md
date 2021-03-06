# fastnmf
Fast non-negative matrix factorization

3/4 initial commit

3/5 

ISSUES
Regularizations causing instability in nmf

SMALL ITEMS:
S4 "show" function for objects of class "nmfmodel"
S4 method for nmfmodel to wrap c_loss_sample data into an nmfmodel
S4 plot method for "nmfmodel" class
Wrap detailed nmf parameters into an S4 "nmfParams" object
add multinomial constraints to nmf

SATURDAY
- Write near-exact nnls to stochastically sample sparse active sets with gradient descent
- Add coordinate descent parameter to exact nnls, but restrict coordinate descent to the feasible set. cd = TRUE is already an option in nnls adaptive solver, integrate directly. Always check to see if the solution is improved.
- Write an nnlsmodel S4 object similar to nmfmodel.
- Add full solution path to nnlsmodel if detail = TRUE
- S4 method for nnlsmodel for rasterized UMAP of solutions. PCA -> RcppAnnoy nearest neighbors -> uwot::UMAP -> ggplot2 raster

Major storylines:
 - NNLS is not convex, all existing methods are assume perfect convexity (Exact NNLS)
 - L0=1 truncation projects hard clusterings.
 - NMF with L0 = 1 gives a clustering model (like k-means) but is faster
 - Benchmarking: AMF is faster than all other NMF implementations (MLPACK), LaPack etc.
 - Better convergence criteria in AMF
 - Regularization on diagonalized models is better

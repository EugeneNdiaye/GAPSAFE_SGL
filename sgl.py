# Author: Eugene Ndiaye
#         Olivier Fercoq
#         Alexandre Gramfort
#         Joseph Salmon
# GAP Safe Screening Rules for Sparse-Group Lasso.
# firstname.lastname@telecom-paristech.fr

import numpy as np
from sgl_fast import bcd_fast
from sgl_tools import build_lambdas, precompute_norm, precompute_DGST3

NO_SCREEN = 0
STATIC_SAFE = 1
DYNAMIC_SAFE = 2
DST3 = 3
GAPSAFE_SEQUENTIAL = 4
GAPSAFE = 5


def sgl_path(X, y, size_groups, omega, screen, beta_init=None, lambdas=None,
             tau=0.5, lambda2=0, max_iter=30000, f=10, eps=1e-4):
    """Compute Sparse-Group-Lasso path with block coordinate descent

    The Sparse-Group-Lasso optimization solves:

    f(beta) + lambda_1 Omega(beta) + 0.5 * lambda_2 norm(beta,2)^2
    where f(beta) = 0.5 * norm(y - X beta,2)^2 and
    Omega(beta) = tau norm(beta,1) + (1 - tau) * sum_g omega_g * norm{beta_g,2}

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication.

    y : ndarray, shape = (n_samples,)
        Target values

    size_groups : ndarray, shape = (n_groups,)
        List of sizes of the different groups
        (n_groups are the number of groups considered).

    omega : ndarray, shape = (n_groups,)
        List of the weight of the different groups: n_groups are the number
        of groups considered.

    screen : integer
        Screening rule to be used: it must be choosen in the following list

        NO_SCREEN = 0 : Standard method

        STATIC_SAFE = 1 : Use static safe screening rule
            cf. El Ghaoui, L., Viallon, V., and Rabbani, T.
            "Safe feature elimination in sparse supervised learning".
            J. Pacific Optim., 2012.

        DYNAMIC_SAFE = 2 : Use dynamic safe screening rule
            cf. Bonnefoy, A., Emiya, V., Ralaivola, L., and Gribonval, R.
            "Dynamic Screening: Accelerating First-Order Al-
            gorithms for the Lasso and Group-Lasso".
            IEEE Trans. Signal Process., 2015.

        DST3 = 3 : Adaptation of the DST3 safe screening rules
            cf.  Xiang, Z. J., Xu, H., and Ramadge, P. J.,
            "Learning sparse representations of high dimensional data on large
            scale dictionaries". NIPS 2011

        GAPSAFE_SEQUENTIAL = 4 : Proposed safe screening rule using duality gap in a sequential way.

        GAPSAFE = 5 : Proposed safe screening rule using duality gap in both a sequential and dynamic way.

    beta_init : array, shape (n_features, ), optional
        The initial values of the coefficients.

    lambdas : ndarray
        List of lambdas where to compute the models.

    tau : float, optional
        Parameter that make a tradeoff between l1 and l1_2 penalties

    f : float, optional
        The screening rule will be execute at each f pass on the data

    eps : float, optional
        Prescribed accuracy on the duality gap.

    Returns
    -------
    coefs : array, shape (n_features, n_alphas)
        Coefficients along the path.

    dual_gaps : array, shape (n_alphas,)
        The dual gaps at the end of the optimization for each alpha.

    lambdas : ndarray
        List of lambdas where to compute the models.

    screening_sizes_groups : array, shape (n_alphas,)
        Number of active groups.

    screening_sizes_features : array, shape (n_alphas,)
        Number of active variables.

    n_iters : array-like, shape (n_alphas,)
        The number of iterations taken by the block coordinate descent
        optimizer to reach the specified accuracy for each lambda.

    """
    n_groups = len(size_groups)
    g_start = np.zeros(n_groups, order='F', dtype=np.intc)
    for i in range(1, n_groups):
        g_start[i] = size_groups[i - 1] + g_start[i - 1]

    if lambdas is None:
        lambdas, imax = build_lambdas(X, y, omega, size_groups, g_start)

    # Useful precomputation
    norm_X, norm_X_g, nrm2_y = precompute_norm(X, y, size_groups, g_start)

    if screen == DST3:
        if lambdas is not None:
            _, imax = build_lambdas(X, y, omega, size_groups, g_start)

        nDST3, norm2_nDST3, nDST3Ty = \
            precompute_DGST3(X, y, tau, omega, lambdas[0], imax, size_groups,
                             g_start)
        tau_w_star = tau + (1 - tau) * omega[imax]
        nTy_on_lambda_ = nDST3Ty / lambdas[0]

    else:  # We take arbitrary values since they are not used by others rules
        nDST3 = np.ones(1)
        norm2_nDST3 = 1
        nDST3Ty = 1
        tau_w_star = 1
        nTy_on_lambda_ = 1

    n_lambdas = len(lambdas)
    n_samples, n_features = X.shape
    lambda_max = lambdas[0]

    # Fortran-contiguous array are used to avoid useless copy of the data.
    X = np.asfortranarray(X)
    y = np.asfortranarray(y)
    size_groups = np.asfortranarray(size_groups, dtype=np.intc)
    norm_X = np.asfortranarray(norm_X)
    norm_X_g = np.asfortranarray(norm_X_g)
    omega = np.asfortranarray(omega)
    if beta_init is None:
        beta_init = np.zeros(n_features, order='F')
    else:
        beta_init = np.asfortranarray(beta_init)

    coefs = np.zeros((n_features, n_lambdas), order='F')
    residual = np.asfortranarray(y - np.dot(X, beta_init))
    XTR = np.asfortranarray(np.dot(X.T, residual))
    dual_scale_p = lambda_max  # good iif beta_init = 0

    dual_gaps = np.ones(n_lambdas)
    screening_sizes_features = np.zeros(n_lambdas)
    screening_sizes_groups = np.zeros(n_lambdas)
    n_iters = np.zeros(n_lambdas)

    for t in range(n_lambdas):

        if t == 0:
            lambda_prec = lambda_max
        else:
            lambda_prec = lambdas[t - 1]

        if screen == DST3:
            nTy_on_lambda_ = nDST3Ty / lambdas[t]

        model = bcd_fast(X, y, beta_init, XTR, residual,
                                           dual_scale_p, omega,
                                           n_samples, n_features, n_groups,
                                           size_groups, g_start, norm_X,
                                           norm_X_g, nrm2_y,
                                           tau, lambdas[t], lambda2,
                                           lambda_prec, lambda_max, max_iter,
                                           f, eps, screen, nDST3, norm2_nDST3,
                                           nTy_on_lambda_, tau_w_star)

        dual_scale_p, dual_gaps[t], n_active_groups, n_active_features, \
            n_iters[t] = model
        coefs[:, t] = beta_init
        screening_sizes_groups[t] = n_active_groups
        screening_sizes_features[t] = n_active_features

        if abs(dual_gaps[t]) > eps:
            print("Warning did not converge ... t = %s gap = %s eps = %s" %
                  (t, dual_gaps[t], eps))

    return (coefs, dual_gaps, lambdas, screening_sizes_groups,
            screening_sizes_features, n_iters)

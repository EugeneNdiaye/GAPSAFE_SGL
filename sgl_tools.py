# Author: Eugene Ndiaye
#         Olivier Fercoq
#         Alexandre Gramfort
#         Joseph Salmon
# GAP Safe Screening Rules for Sparse-Group Lasso.
# http://arxiv.org/abs/1602.06225
# firstname.lastname@telecom-paristech.fr

import numpy as np
from scipy.linalg import toeplitz


def vect_ST(u, x):
    """
        Vectorial soft-thresholding at level u.

    """
    return [np.sign(x[i]) * max(abs(x[i]) - u, 0) for i in range(len(x))]


def generate_data(n_samples, n_features, size_groups, rho=0.5):
    """ Data generation process with Toplitz like correlated features:
        this correspond to the synthetic dataset used in our paper
        "GAP Safe Screening Rules for Sparse-Group Lasso".

    """
    n_groups = len(size_groups)
    g_start = np.zeros(n_groups, order='F', dtype=np.intc)
    for i in range(1, n_groups):
        g_start[i] = size_groups[i - 1] + g_start[i - 1]

    # 10% of groups are actives
    gamma1 = int(np.ceil(n_groups * 0.1))
    selected_groups = np.random.random_integers(0, n_groups - 1, gamma1)
    true_beta = np.zeros(n_features)

    for i in selected_groups:

        begin = g_start[i]
        end = g_start[i] + size_groups[i]
        # 10% of features are actives
        gamma2 = int(np.ceil(size_groups[i] * 0.1))
        selected_features = np.random.random_integers(begin, end - 1, gamma2)

        ns = len(selected_features)
        s = 2 * np.random.rand(ns) - 1
        u = np.random.rand(ns)
        true_beta[selected_features] = np.sign(s) * (10 * u + (1 - u) * 0.5)

    vect = rho ** np.arange(n_features)
    covar = toeplitz(vect, vect)

    X = np.random.multivariate_normal(np.zeros(n_features), covar, n_samples)
    y = np.dot(X, true_beta) + 0.01 * np.random.normal(0, 1, n_samples)

    X /= np.sqrt(np.sum(X ** 2, axis=0))
    y /= np.linalg.norm(y, ord=2)

    return X, y


def epsilon_norm(x, alpha, R):
    """
        Compute the unique positive solution in z of the equation
        norm(ST(x, alpha * z), 2) = alpha * R. The epsilon-norm correspond to
        the case alpha = 1 - epsilon and R = epsilon.
        See our paper "GAP Safe Screening Rules for Sparse-Group Lasso".

    """
    if alpha == 0 and R != 0:
            return np.linalg.norm(x, ord=2) / R

    if R == 0:  # j0 = 0 iif R = 0 iif alpha = 1 in practice
        return np.linalg.norm(x, ord=np.inf) / alpha

    zx = np.abs(x)
    norm_inf = np.linalg.norm(x, ord=np.inf)
    I_inf = np.where(zx > alpha * (norm_inf) / (alpha + R))[0]
    n_inf = len(I_inf)
    zx = np.sort(zx[I_inf])[::-1]

    if norm_inf == 0:
        return 0

    if n_inf == 1:
        return zx[0]

    R2 = R ** 2
    alpha2 = alpha ** 2
    R2onalpha2 = R2 / alpha2
    a_k = S = S2 = 0

    for k in range(n_inf - 1):

        S += zx[k]
        S2 += zx[k] ** 2
        b_k = S2 / (zx[k + 1] ** 2) - 2 * S / zx[k + 1] + k + 1

        if a_k <= R2onalpha2 and R2onalpha2 < b_k:
            j0 = k + 1
            break
        a_k = b_k
    else:
        j0 = n_inf
        S += zx[n_inf - 1]
        S2 += zx[n_inf - 1] ** 2

    alpha_S = alpha * S
    j0alpha2_R2 = j0 * alpha2 - R2

    if (j0alpha2_R2 == 0):
        return S2 / (2 * alpha_S)

    delta = alpha_S ** 2 - S2 * j0alpha2_R2

    return (alpha_S - np.sqrt(delta)) / j0alpha2_R2


def precompute_norm(X, y, size_groups, g_start):
    """
        Precomputation of the norm and group's norm used in the algorithm.

    """
    nrm2_y = np.linalg.norm(y, ord=2) ** 2
    n, p = X.shape
    n_groups = len(size_groups)

    norm_X = [np.linalg.norm(X[:, j], ord=2) for j in range(p)]
    norm_X_g = [np.linalg.norm(X[:, g_start[i]:g_start[i] + size_groups[i]])
                for i in range(n_groups)]

    return norm_X, norm_X_g, nrm2_y


def precompute_DGST3(X, y, tau, omega, lambda_max, imax, size_groups, g_start):
    """
        Precomputation needed for DST3 screening rule.

    """
    g_max = range(g_start[imax], g_start[imax] + size_groups[imax])
    tgmax = (1 - tau) * omega[imax] / (tau + (1 - tau) * omega[imax])
    XgmaxTy_on_lambda_max = np.dot(X[:, g_max].T, y) / lambda_max
    treshold_n = (1 - tgmax) * epsilon_norm(XgmaxTy_on_lambda_max, 1 - tgmax,
                                            tgmax)

    if tau == 1:

        Xty = np.dot(X.T, y)
        nDST3 = X[:, np.argmax(Xty)]
        norm2_nDST3 = 1
        nDST3Ty = lambda_max

    else:

        STx = vect_ST(treshold_n, XgmaxTy_on_lambda_max)

        scale_n = tgmax * np.linalg.norm(STx, ord=2) + \
            (1 - tgmax) * np.linalg.norm(STx, ord=1)

        nDST3 = np.dot(X[:, g_max], STx) / scale_n
        norm2_nDST3 = np.linalg.norm(nDST3, ord=2)
        nDST3Ty = np.dot(nDST3.T, y)

    return nDST3, norm2_nDST3, nDST3Ty


def build_lambdas(X, y, omega, size_groups, g_start, n_lambdas=100, delta=3,
                  tau=0.5):
    """
        Compute a list of regularization parameters which decrease geometrically

    """
    eps_g = (1 - tau) * omega / (tau + (1 - tau) * omega)
    n_groups = len(size_groups)

    nrm = [epsilon_norm(
        np.dot(X[:, g_start[i]:g_start[i] + size_groups[i]].T, y),
        1 - eps_g[i], eps_g[i]) for i in range(n_groups)]

    nrm = np.array(nrm) / (tau + (1 - tau) * omega)
    imax = np.argmax(nrm)
    lambda_max = nrm[imax]

    lambdas = lambda_max * \
        10 ** (-delta * np.arange(n_lambdas) / float(n_lambdas - 1))

    return lambdas, imax

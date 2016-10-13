# Author: Eugene Ndiaye
#         Olivier Fercoq
#         Alexandre Gramfort
#         Joseph Salmon
# GAP Safe Screening Rules for Sparse-Group Lasso.
# http://arxiv.org/abs/1602.06225
# firstname.lastname@telecom-paristech.fr


from libc.math cimport fabs, sqrt
from libc.stdlib cimport qsort
from scipy.linalg.cython_blas cimport daxpy, ddot, dnrm2, dscal
import numpy as np
cimport numpy as np
cimport cython

cdef int NO_SCREENING = 0

cdef int STATIC_SAFE = 1
cdef int DYNAMIC_SAFE = 2
cdef int DST3 = 3

cdef int GAPSAFE_SEQ = 4
cdef int GAPSAFE = 5

cdef int GAPSAFE_SEQ_pp = 6
cdef int GAPSAFE_pp = 7


cdef inline double fmax(double x, double y) nogil:
    if x > y:
        return x
    return y


cdef inline double fsign(double f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0

cdef double abs_max(int n, double * a) nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef double m = fabs(a[0])
    cdef double d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


cdef double max(int n, double * a) nogil:
    """np.max(a)"""
    cdef int i
    cdef double m = a[0]
    cdef double d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m


# mimic rounding to zero
# cdef double near_zero(double a) nogil:
#     if fabs(a) <= 1e-14:
#         return 0
#     return a

cdef double near_zero(double a) nogil:
    if fabs(a) <= 1e-14:
        return fabs(a)
    return a

# Function to compute the primal value
cdef double primal_value(int n_samples, int n_features, int n_groups,
                         int * size_groups, int * g_start, double * omega,
                         double * residual, double * beta,
                         int * disabled_groups, int * disabled_features,
                         double residual_norm2, double norm_beta2,
                         double lambda_, double lambda2, double tau) nogil:

    cdef double group_norm = 0.
    cdef double l1_norm = 0.
    cdef double fval = 0.
    cdef int i = 0
    cdef int inc = 1

    # group_norm_beta = np.sum([linalg.norm(beta[u], ord=2)
    #                          for u in group_labels])
    if tau < 1.:
        for i in range(n_groups):
            if disabled_groups[i] == 1:
                continue

            group_norm += omega[i] * dnrm2(& size_groups[i],
                                           & beta[g_start[i]], & inc)

    if tau > 0:
        for i in range(n_features):
            if disabled_features[i] == 1:
                continue
            l1_norm += fabs(beta[i])

    fval = lambda_ * (tau * l1_norm + (1. - tau) * group_norm)

    if lambda2 != 0.:
        fval += 0.5 * (residual_norm2 + lambda2 * norm_beta2)
    else:
        fval += 0.5 * residual_norm2

    return fval


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dual(int n_samples, int n_features, double * residual,
                 double * beta, double * y, double dual_scale,
                 double residual_norm2, double beta_norm2,
                 double lambda_, double lambda2) nogil:

    cdef int i = 0
    cdef int inc = 1
    cdef double Ry = ddot(& n_samples, & residual[0], & inc, & y[0], & inc)

    dval = ((-0.5 * (lambda_ ** 2) * (residual_norm2 + lambda2 * beta_norm2)
              / (dual_scale ** 2)) + lambda_ * Ry / dual_scale)

    return dval


cdef double segment_project(double x, double a, double b) nogil:
    return -fmax(-b, -fmax(x, a))


cdef int compare_doubles(void * a, void * b) nogil:

    cdef double * da = <double * > a
    cdef double * db = <double * > b

    return (da[0] < db[0]) - (da[0] > db[0])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double epsilon_norm(int len_x, double * x, double alpha, double R,
                         double[:] zx) nogil:

    """
        Compute the solution in nu of the equation
        sum_i max(|x_i| - alpha * nu, 0)^2 = (nu * R)^2
    """

    # if alpha == 0 and R == 0:  # this case never happen
    #   return np.inf

    cdef int inc = 1

    if alpha == 0 and R != 0:
        return dnrm2(& len_x, x, & inc) / R

    # j0 = 0 iif R = 0
    if R == 0:
        return abs_max(len_x, x) / alpha

    cdef double R2 = R * R
    cdef double alpha2 = alpha * alpha
    cdef double delta = 0.
    cdef double R2onalpha2 = R2 / alpha2
    cdef double alpha2j0 = 0.
    cdef double j0alpha2_R2 = 0.
    cdef double alpha_S = 0.
    cdef double S = 0.
    cdef double S2 = 0.
    cdef double a_k = 0.
    cdef double b_k = 0.
    cdef int j0 = 0
    cdef int k = 0
    cdef int n_I = 0
    cdef double norm_inf = abs_max(len_x, x)
    cdef double ratio_ = alpha * (norm_inf) / (alpha + R)

    if norm_inf == 0:
        return 0

    for k in range(len_x):

        if fabs(x[k]) > ratio_:
            zx[n_I] = fabs(x[k])
            n_I += 1

    # zx = np.sort(zx)[::-1]
    qsort(& zx[0], n_I, sizeof(double), compare_doubles)

    if norm_inf == 0:
        return 0

    if n_I == 1:
        return zx[0]

    for k in range(n_I - 1):

        S += zx[k]
        S2 += zx[k] * zx[k]
        b_k = S2 / (zx[k + 1] * zx[k + 1]) - 2 * S / zx[k + 1] + k + 1

        if a_k <= R2onalpha2 and R2onalpha2 < b_k:
            j0 = k + 1
            break
    else:
        j0 = n_I
        S += zx[n_I - 1]
        S2 += zx[n_I - 1] * zx[n_I - 1]

    alpha_S = alpha * S
    alpha2j0 = alpha2 * j0

    if (alpha2j0 == R2):
        return S2 / (2 * alpha_S)

    j0alpha2_R2 = alpha2j0 - R2
    delta = alpha_S * alpha_S - S2 * j0alpha2_R2

    return (alpha_S - sqrt(delta)) / j0alpha2_R2


cdef double ST(double u, double x) nogil:
    return fsign(x) * fmax(fabs(x) - u, 0.)


cdef double dual_gap(int n_samples, int n_features, int n_groups,
                     int * size_groups, int * g_start, double * residual,
                     double * y, double * beta, double nrm2_y,
                     double * omega, double dual_scale,
                     int * disabled_groups, int * disabled_features,
                     double residual_norm2, double norm_beta2,
                     double lambda_, double lambda2, double tau) nogil:

    cdef double pobj = primal_value(n_samples, n_features, n_groups,
                                    size_groups, g_start, omega, residual,
                                    beta, disabled_groups, disabled_features,
                                    residual_norm2, norm_beta2,
                                    lambda_, lambda2, tau)

    cdef double dobj = dual(n_samples, n_features, residual,
                            beta, y, dual_scale,
                            residual_norm2, norm_beta2, lambda_, lambda2)

    cdef double gap_ = pobj - dobj
    return gap_


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void fscreen_sgl(double * beta, double * XTc, double[::1, :] X,
                      double * residual, double * XTR, int * disabled_features,
                      int * disabled_groups, int * size_groups,
                      double * norm2_X, double * norm2_X_g, int * g_start,
                      int n_groups, double r, int n_samples,
                      int * n_active_features, int * n_active_groups,
                      double tau, double * omega) nogil:

    cdef int i = 0
    cdef int j = 0
    cdef int len_g = 0
    cdef int g_end = 0
    cdef double norm_XTc_g = 0.
    cdef double r_normX_g = 0.
    cdef double r_normX_j = 0.
    cdef double ftest = 0.
    cdef int sphere_test_g = 0
    cdef int sphere_test_j = 0
    cdef int inc = 1

    # Safe rule for Group level
    for i in range(n_groups):

        if disabled_groups[i] == 1:
            continue

        g_end = g_start[i] + size_groups[i]
        r_normX_g = r * sqrt(norm2_X_g[i])

        if r_normX_g > (1. - tau) * omega[i] or tau == 1.:
            sphere_test_g = 0

        else:

            # norm_XTc_g = linalg.norm(XTc[g], ord=np.inf)
            norm_XTc_g = abs_max(size_groups[i], & XTc[g_start[i]])

            if norm_XTc_g <= tau:
                ftest = fmax(0., norm_XTc_g + r_normX_g - tau)

            else:
                ftest = 0.
                for j in range(size_groups[i]):
                    ftest += ST(tau, XTc[g_start[i] + j]) ** 2
                ftest = sqrt(ftest) + r_normX_g

            sphere_test_g = ftest < (1. - tau) * omega[i]

        if sphere_test_g:

            len_g = 0
            for j in range(g_start[i], g_end):

                if disabled_features[j] == 0:

                    if beta[j] != 0.:
                        # residual -= X[:, j] * (-beta[j])
                        daxpy(& n_samples, & beta[j], & X[0, j], & inc,
                              residual, & inc)
                        beta[j] = 0.

                    XTR[j] = 0.
                    disabled_features[j] = 1
                    len_g += 1

            disabled_groups[i] = 1
            n_active_groups[0] -= 1
            n_active_features[0] -= len_g

        # Safe rule for Feature level
        else:

            for j in range(g_start[i], g_end):

                if disabled_features[j] == 1:
                    continue

                r_normX_j = r * sqrt(norm2_X[j])
                if r_normX_j > tau or tau == 0:
                    continue

                if tau < 1.:
                    sphere_test_j = fabs(XTc[j]) + r_normX_j <= tau
                else:
                    sphere_test_j = fabs(XTc[j]) + r_normX_j < tau

                if sphere_test_j:

                    # Update residual
                    if beta[j] != 0.:
                        # residual -= X[:, j] * (beta[j] - beta_old[j])
                        daxpy(& n_samples, & beta[j], & X[0, j], & inc,
                              residual, & inc)
                        beta[j] = 0.

                    # we "set" x_j to zero since the j_th feature is inactive
                    XTR[j] = 0.
                    disabled_features[j] = 1
                    n_active_features[0] -= 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void prox_sgl(int n_samples, int n_features, int n_groups,
                   double * beta, double * beta_old, double[::1, :] X,
                   double * residual, double * XTR,
                   int * disabled_features, int * disabled_groups,
                   int * size_groups, double * norm2_X_g,
                   int * g_start, double lambda_, double lambda2,
                   double tau, double * omega) nogil:

    cdef int i = 0
    cdef int j = 0
    cdef int inc = 1
    cdef int g_end = 0
    cdef double mu_st = 0.
    cdef double mu_g = 0.
    cdef double L_g = 0.  # Lipschitz constants
    cdef double norm_beta_g = 0.
    cdef double scaling = 0.
    cdef double double_tmp = 0.

    for i in range(n_groups):

        if disabled_groups[i] == 1:
            continue

        L_g = norm2_X_g[i] + lambda2
        g_end = g_start[i] + size_groups[i]
        mu_st = tau * lambda_ / L_g

        norm_beta_g = 0.
        # coordinate wise soft tresholding
        for j in range(g_start[i], g_end):

            if disabled_features[j] == 1:
                continue

            beta_old[j] = beta[j]

            # XTR[j] = np.dot(X[:, j], residual) - lambda2 * beta
            double_tmp = ddot(
                & n_samples, & X[0, j], & inc, & residual[0], & inc)
            XTR[j] = double_tmp - lambda2 * beta[j]

            beta[j] = ST(mu_st, beta[j] + XTR[j] / L_g)
            norm_beta_g += beta[j] ** 2

        # group soft tresholding
        # norm_beta_g = linalg.norm(beta[g], ord=2)
        # norm_beta_g = dnrm2(& size_groups[i], & beta[g_start[i]], & inc)

        if norm_beta_g > 0.:

            norm_beta_g = sqrt(norm_beta_g)
            mu_g = (1. - tau) * omega[i] * lambda_ / L_g
            scaling = fmax(1. - mu_g / norm_beta_g, 0.)
            # beta[g] = scaling * beta[g]
            dscal(& size_groups[i], & scaling, & beta[g_start[i]], & inc)

        # Update residual
        for j in range(g_start[i], g_end):

            if disabled_features[j] == 1:
                continue

            if beta[j] != beta_old[j]:
                # residual -= X[:, j] * (beta[j] - beta_old[j])
                double_tmp = -beta[j] + beta_old[j]
                daxpy(& n_samples, & double_tmp, & X[0, j], & inc,
                      & residual[0], & inc)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dual_scaling(int n_samples, int n_features, int n_groups,
                         double * beta, double[::1, :] X, double * residual,
                         double * XTR, int * disabled_features,
                         int * disabled_groups, int * size_groups,
                         int * g_start, double lambda_, double lambda2,
                         double tau, double * omega, double[:] zx) nogil:

    cdef int i = 0
    cdef int j = 0
    cdef int inc = 1
    cdef double dual_scale = 0.
    cdef double double_tmp = 0.
    cdef double tg = 0.

    for i in range(n_groups):

        if disabled_groups[i] == 1:
            continue

        g_end = g_start[i] + size_groups[i]
        for j in range(g_start[i], g_end):

            if disabled_features[j] == 1:
                continue

            # XTR[j] = np.dot(X[:, j], residual) - lambda2 * beta
            double_tmp = ddot(& n_samples, & X[0, j], & inc,
                              & residual[0], & inc)
            XTR[j] = double_tmp - lambda2 * beta[j]

        tg = ((1. - tau) * omega[i]) / (tau + (1. - tau) * omega[i])
        double_tmp = epsilon_norm(size_groups[i], & XTR[g_start[i]],
                                  1. - tg, tg, zx)
        double_tmp /= tau + (1. - tau) * omega[i]
        dual_scale = fmax(double_tmp, dual_scale)

    dual_scale = fmax(lambda_, dual_scale)
    return dual_scale


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def bcd_fast(double[::1, :] X, double[:] y, double[:] beta,
             double[:] XTR, double[:] residual, double dual_scale,
             double[:] omega, int n_samples, int n_features, int n_groups,
             int[:] size_groups, int[:] g_start,
             double[:] norm2_X, double[:] norm2_X_g, double nrm2_y,
             double tau, double lambda_, double lambda2,
             double lambda_max, int max_iter, int f, double eps,
             int screen, double[:] n_DST3, double norm2_nDST3,
             double tau_w_star, int[:] disabled_features,
             int[:] disabled_groups, int wstr_plus):
    """
        Solve the sparse-group-lasso regression with elastic-net
        We minimize
        f(beta) + lambda_1 Omega(beta) + 0.5 * lambda_2 norm(beta, 2)^2
        where f(beta) = 0.5 * norm(y - X beta, 2)^2 and
        Omega(beta) = tau norm(beta, 1) +
                      (1 - tau) * sum_g omega_g * norm(beta_g, 2)
    """

    cdef int i = 0
    cdef int k = 0
    cdef int j = 0
    cdef int g_end = 0
    cdef int n_active_groups = n_groups
    cdef int n_active_features = n_features
    cdef int inc = 1

    cdef double r = 666.  # radius in the screening rules
    cdef double gap_t = 666.
    cdef double double_tmp = 0.

    # zx is a temporary array used in dual_scaling (without gil)
    cdef double[:] zx = np.zeros(n_features, order='F')

    cdef double tmp_DST3 = lambda_max / lambda_ - tau_w_star
    cdef double delta_DST3 = tmp_DST3 / norm2_nDST3
    cdef double norm2_y_on_lambda_minus_center = 0.
    cdef double residual_norm2 = dnrm2(& n_samples, & residual[0], & inc) ** 2
    cdef double norm_beta2 = dnrm2(& n_features, & beta[0], & inc) ** 2

    cdef double[:] beta_old = np.zeros(n_features, order='F')
    cdef double[:] XTc = np.zeros(n_features, order='F')
    cdef double[:] center = np.zeros(n_samples, order='F')

    with nogil:

        if wstr_plus == 0:

            for j in range(n_features):
                disabled_features[j] = 0
            for i in range(n_groups):
                disabled_groups[i] = 0

        if screen == DST3:

            for i in range(n_samples):
                center[i] = y[i] / lambda_ - delta_DST3 * n_DST3[i]

            # r is computed dynamically
            # XTc = np.dot(X.T, center)
            for j in range(n_features):
                XTc[j] = ddot(
                    & n_samples, & X[0, j], & inc, & center[0], & inc)

        if screen == DYNAMIC_SAFE:
            # center = y / lambda_
            for i in range(n_samples):
                center[i] = y[i] / lambda_
            # r is computed dynamically
            # XTc = np.dot(X.T, center)
            for j in range(n_features):
                XTc[j] = ddot(
                    & n_samples, & X[0, j], & inc, & center[0], & inc)

        if screen == STATIC_SAFE:

            r = fabs(1. / lambda_max - 1. / lambda_) * sqrt(nrm2_y)
            # center = y / lambda_
            for i in range(n_samples):
                center[i] = y[i] / lambda_

            # XTc = np.dot(X.T, center)
            for j in range(n_features):
                XTc[j] = ddot(
                    & n_samples, & X[0, j], & inc, & center[0], & inc)

        if screen == GAPSAFE_SEQ:
            # center = theta (previous optimal dual solution)
            gap_t = dual_gap(n_samples, n_features, n_groups,
                             & size_groups[0], & g_start[0],
                             & residual[0], & y[0], & beta[0], nrm2_y,
                             & omega[0], dual_scale, & disabled_groups[0],
                             & disabled_features[0], residual_norm2,
                             norm_beta2, lambda_, lambda2, tau)

            # r = sqrt(2 * gap_t) / lambda_
            r = sqrt(2 * near_zero(gap_t)) / lambda_
            for j in range(n_features):
                XTc[j] = (XTR[j] - lambda2 * beta[j]) / dual_scale

        if screen in [GAPSAFE_SEQ, STATIC_SAFE]:

            fscreen_sgl(
                & beta[0], & XTc[0], X, & residual[0], & XTR[0],
                & disabled_features[0], & disabled_groups[0],
                & size_groups[0], & norm2_X[0], & norm2_X_g[0],
                & g_start[0], n_groups, r, n_samples,
                & n_active_features, & n_active_groups, tau, & omega[0])

        for k in range(max_iter):

            if f != 0 and k % f == 0:

                # Compute dual point by dual scaling :
                # theta_k = residual / dual_scale
                dual_scale = dual_scaling(n_samples, n_features, n_groups,
                                          & beta[0], X, & residual[0],
                                          & XTR[0], & disabled_features[0],
                                          & disabled_groups[0],
                                          & size_groups[0], & g_start[0],
                                          lambda_, lambda2, tau, & omega[0], zx)

                residual_norm2 = dnrm2(& n_samples, & residual[0], & inc) ** 2
                norm_beta2 = dnrm2(& n_features, & beta[0], & inc) ** 2
                gap_t = dual_gap(n_samples, n_features, n_groups,
                                 & size_groups[0], & g_start[0],
                                 & residual[0], & y[0], & beta[0],
                                 nrm2_y, & omega[0], dual_scale,
                                 & disabled_groups[0],
                                 & disabled_features[0],
                                 residual_norm2, norm_beta2,
                                 lambda_, lambda2, tau)

                if gap_t <= eps * nrm2_y:
                    # print "boom bapp ---> ", near_zero(gap_t), gap_t
                    break

                if screen == GAPSAFE:
                    # center = theta_k
                    # r = sqrt(2 * gap_t) / lambda_
                    r = sqrt(2 * near_zero(gap_t)) / lambda_
                    for j in range(n_features):
                        XTc[j] = (XTR[j] - lambda2 * beta[j]) / dual_scale

                if screen == DYNAMIC_SAFE:
                    # center = y /lambda_
                    # r = ||theta_k - center||
                    if lambda_ == lambda_max:
                        r = 0.
                    else:
                        r = 0.
                        for i in range(n_samples):
                            r += (residual[i] / dual_scale - center[i]) ** 2
                        r = sqrt(r)

                if screen == DST3:
                    # center is precomputed
                    # r = sqrt(||theta_k - y/lambda||**2 -
                    #          ||y/lambda_ - center||**2)
                    if lambda_ == lambda_max:
                        r = 0.
                    else:
                        r = 0.
                        for i in range(n_samples):
                            r += (residual[i] / dual_scale - y[i] / lambda_) ** 2
                        r = sqrt(r - delta_DST3 * tmp_DST3)

                if screen in [GAPSAFE, DYNAMIC_SAFE, DST3]:

                    fscreen_sgl(
                        & beta[0], & XTc[0], X, & residual[0], & XTR[0],
                        & disabled_features[0], & disabled_groups[0],
                        & size_groups[0], & norm2_X[0], & norm2_X_g[0],
                        & g_start[0], n_groups, r, n_samples,
                        & n_active_features, & n_active_groups, tau,
                        & omega[0])

            prox_sgl(n_samples, n_features, n_groups,
                     & beta[0], & beta_old[0], X, & residual[0], & XTR[0],
                     & disabled_features[0], & disabled_groups[0],
                     & size_groups[0], & norm2_X_g[0],
                     & g_start[0], lambda_, lambda2, tau, & omega[0])

    return (dual_scale, gap_t, n_active_groups, n_active_features, k)

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

ctypedef np.float64_t DOUBLE

np.import_array()

NO_SCREEN = 0
STATIC_SAFE = 1
DYNAMIC_SAFE = 2
DST3 = 3
GAPSAFE_SEQUENTIAL = 4
GAPSAFE = 5


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
cdef double near_zero(double a) nogil:
    if fabs(a) <= 1e-14:
        return 0
    return a

# Function to compute the primal value
cdef double primal_value(   # Data
                            int n_samples,
                            int n_features,
                            int n_groups,
                            int * size_groups_data,
                            int * g_start_data,
                            double * omega_data,
                            double * residual_data,
                            double * beta_data,
                            int * disabled_groups_data,
                            int * disabled_features_data,
                            double residual_norm2,
                            double norm_beta2,
                            # Parameters
                            double lambda_, double lambda2,
                            double tau) nogil:

    cdef double group_norm = 0
    cdef double l1_norm = 0
    cdef double fval = 0
    cdef int i = 0
    cdef int inc = 1

    # group_norm_beta = np.sum([linalg.norm(beta[u], ord=2)
    #                          for u in group_labels])
    if tau < 1:
        for i in range(n_groups):
            if disabled_groups_data[i] == 1:
                continue

            group_norm += omega_data[i] * dnrm2(& size_groups_data[i],
                                                & beta_data[g_start_data[i]], & inc)

    if tau > 0:
        for i in range(n_features):
            if disabled_features_data[i] == 1:
                continue
            l1_norm += fabs(beta_data[i])

    fval = lambda_ * (tau * l1_norm + (1 - tau) * group_norm)

    if lambda2 != 0:
        fval += 0.5 * (residual_norm2 + lambda2 * norm_beta2)
    else:
        fval += 0.5 * residual_norm2

    return fval


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dual(int n_samples,
                 int n_features,
                 double * residual_data,
                 double * beta_data,
                 double * y_data,
                 double dual_scale,
                 double residual_norm2,
                 double beta_norm2,
                 double lambda_, double lambda2) nogil:

    cdef double Ry = 0
    cdef int i = 0

    for i in range(n_samples):
        Ry += residual_data[i] * y_data[i]

    if dual_scale != 0:
        dval = ((-0.5 * (lambda_ ** 2) * (residual_norm2 + lambda2 * beta_norm2)
               / (dual_scale ** 2)) + lambda_ * Ry / dual_scale)

    return dval


cdef double segment_project(double x, double a, double b) nogil:
    return -fmax(-b, -fmax(x, a))


cdef int compare_doubles(void * a, void * b) nogil:

    cdef DOUBLE * da = <DOUBLE * > a
    cdef DOUBLE * db = <DOUBLE * > b

    return (da[0] < db[0]) - (da[0] > db[0])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double epsilon_norm(int len_x, double * x_data, double alpha, double R):

    """
        Compute the solution in nu of the equation
        sum_i max(|x_i| - alpha * nu, 0)^2 = (nu * R)^2
    """

    # if alpha == 0 and R == 0: #this case never happen
    #   return np.inf

    cdef int inc = 1

    if alpha == 0 and R != 0:
        return dnrm2(& len_x, x_data, & inc) / R

    # j0 = 0 iif R = 0
    if R == 0:
        return abs_max(len_x, x_data) / alpha

    cdef double[:] zx = np.zeros(len_x + 1, order='F')
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
    cdef double norm_inf = abs_max(len_x, x_data)
    cdef double ratio_ = alpha * (norm_inf) / (alpha + R)

    with nogil:

        if norm_inf == 0:
            return 0

        for k in range(len_x):

            if fabs(x_data[k]) > ratio_:
                zx[n_I] = fabs(x_data[k])
                n_I += 1

        # zx = np.sort(zx)[::-1]
        qsort(& zx[0], n_I, sizeof(DOUBLE), compare_doubles)

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
    return fsign(x) * fmax(fabs(x) - u, 0)


cdef double group_dual_gap(int n_samples,
                           int n_features,
                           int n_groups,
                           int * size_groups_data,
                           int * g_start_data,
                           double * residual_data,
                           double * y_data,
                           double * beta_data,
                           double nrm2_y,
                           double * omega_data,
                           double dual_scale,
                           int * disabled_groups_data,
                           int * disabled_features_data,
                           double residual_norm2,
                           double norm_beta2,
                           double lambda_, double lambda2,
                           double tau) nogil:

    cdef double pobj = primal_value(n_samples, n_features, n_groups,
                                    size_groups_data, g_start_data, omega_data,
                                    residual_data, beta_data,
                                    disabled_groups_data,
                                    disabled_features_data,
                                    residual_norm2, norm_beta2,
                                    lambda_, lambda2, tau)

    cdef double dobj = dual(n_samples, n_features, residual_data,
                            beta_data, y_data, dual_scale,
                            residual_norm2, norm_beta2, lambda_, lambda2)

    cdef double gap_ = pobj - dobj
    return gap_


cdef void fscreen_sgl(double * beta_data,
                      double * XTc_data,
                      double * X_data_ptr,
                      double * residual_data,
                      double * XTR_data,
                      int * disabled_features_data,
                      int * disabled_groups_data,
                      int * size_groups_data,
                      double * norm_X_data,
                      double * norm_X_g_data,
                      int * g_start_data,
                      int n_groups,
                      double r,
                      int n_samples,
                      int * n_active_features,
                      int * n_active_groups,
                      double tau,
                      double * omega_data) nogil:

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

        if disabled_groups_data[i] == 1:
            continue

        r_normX_g = r * norm_X_g_data[i]
        if r_normX_g > (1 - tau) * omega_data[i] and tau != 1:
            sphere_test_g = 0

        else:
            g_end = g_start_data[i] + size_groups_data[i]

            # norm_XTc_g = linalg.norm(XTc[g], ord=np.inf)
            norm_XTc_g = abs_max(size_groups_data[i],
                                 & XTc_data[g_start_data[i]])

            if norm_XTc_g <= tau:
                ftest = fmax(0, norm_XTc_g + r * norm_X_g_data[i] - tau)

            else:
                ftest = 0
                for j in range(size_groups_data[i]):
                    ftest += ST(tau, XTc_data[g_start_data[i] + j]) ** 2
                ftest = sqrt(ftest) + r_normX_g

            sphere_test_g = ftest < (1 - tau) * omega_data[i]

        if sphere_test_g:

            len_g = 0
            for j in range(g_start_data[i], g_end):

                if disabled_features_data[j] == 0:

                    if beta_data[j] != 0:
                        # residual -= X[:, j] * (-beta[j])
                        daxpy(& n_samples, & beta_data[j],
                              X_data_ptr + j * n_samples, & inc, residual_data,
                              & inc)
                        beta_data[j] = 0

                    disabled_features_data[j] = 1
                    len_g += 1

            disabled_groups_data[i] = 1
            n_active_groups[0] -= 1
            n_active_features[0] -= len_g

        # Safe rule for Feature level
        else:
            for j in range(g_start_data[i], g_end):

                if disabled_features_data[j] == 1:
                    continue

                r_normX_j = r * norm_X_data[j]
                if r_normX_j > tau and tau != 0:
                    continue

                if tau < 1:
                    sphere_test_j = fabs(XTc_data[j]) + r_normX_j <= tau
                else:
                    sphere_test_j = fabs(XTc_data[j]) + r_normX_j < tau

                if sphere_test_j:

                    # Update residual
                    if beta_data[j] != 0:
                        # residual -= X[:, j] * (beta[j] - beta_old[j])
                        daxpy(& n_samples, & beta_data[j],
                              X_data_ptr + j * n_samples, & inc, residual_data,
                              & inc)
                        beta_data[j] = 0

                    # we "set" x_j to zero since the j_th feature is inactive
                    XTR_data[j] = 0
                    disabled_features_data[j] = 1
                    n_active_features[0] -= 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void prox_sgl(int n_samples,
                   int n_features,
                   int n_groups,
                   double * beta_data,
                   double * beta_old,
                   double * X_data_ptr,
                   double * residual_data,
                   double * XTR_data,
                   int * disabled_features_data,
                   int * disabled_groups_data,
                   int * size_groups_data,
                   double * norm_X_data,
                   double * norm_X_g_data,
                   int * g_start_data,
                   double lambda_, double lambda2,
                   double tau,
                   double * omega_data) nogil:

    cdef int i = 0
    cdef int j = 0
    cdef double mu_st = 0
    cdef double mu_g = 0
    cdef double L_g = 0  # Lipschitz constants
    cdef int g_end = 0
    cdef double norm_beta_g = 0
    cdef double scaling = 0
    cdef double double_tmp = 0
    cdef int inc = 1

    for i in range(n_groups):

        if disabled_groups_data[i] == 1:
            continue

        L_g = norm_X_g_data[i] ** 2 + lambda2
        g_end = g_start_data[i] + size_groups_data[i]
        mu_st = tau * lambda_ / L_g

        # coordinate wise soft tresholding
        for j in range(g_start_data[i], g_end):

            if disabled_features_data[j] == 1:
                continue

            beta_old[j] = beta_data[j]

            # XTR[j] = np.dot(X[:, j], residual) - lambda2 * beta
            double_tmp = ddot(& n_samples, X_data_ptr + j * n_samples, & inc,
                              & residual_data[0], & inc)
            XTR_data[j] = double_tmp - lambda2 * beta_data[j]

            beta_data[j] = ST(mu_st, beta_data[j] + XTR_data[j] / L_g)

        # group soft tresholding
        # norm_beta_g = linalg.norm(beta[g], ord=2)
        norm_beta_g = dnrm2(& size_groups_data[i], & beta_data[g_start_data[i]],
                            & inc)

        if norm_beta_g > 0:

            mu_g = (1 - tau) * omega_data[i] * lambda_ / L_g
            scaling = fmax(1 - mu_g / norm_beta_g, 0)
            # beta[g] = scaling * beta[g]
            dscal(& size_groups_data[i], & scaling, & beta_data[g_start_data[i]],
                  & inc)

        # Update residual
        for j in range(g_start_data[i], g_end):

            if disabled_features_data[j] == 1:
                continue

            if beta_data[j] != beta_old[j]:
                # residual -= X[:, j] * (beta[j] - beta_old[j])
                double_tmp = -beta_data[j] + beta_old[j]
                daxpy(& n_samples, & double_tmp,
                      X_data_ptr + j * n_samples, & inc, & residual_data[0],
                      & inc)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dual_scaling(int n_samples,
                         int n_features,
                         int n_groups,
                         double * beta_data,
                         double * X_data_ptr,
                         double * residual_data,
                         double * XTR_data,
                         int * disabled_features_data,
                         int * disabled_groups_data,
                         int * size_groups_data,
                         int * g_start_data,
                         double lambda_,
                         double lambda2,
                         double tau,
                         double * omega_data) nogil:

    cdef int i = 0
    cdef int j = 0
    cdef int inc = 1
    cdef double dual_scale = 0
    cdef double double_tmp = 0
    cdef double tg = 0

    for i in range(n_groups):

        if disabled_groups_data[i] == 1:
            continue

        g_end = g_start_data[i] + size_groups_data[i]
        for j in range(g_start_data[i], g_end):

            if disabled_features_data[j] == 1:
                continue

            # XTR[j] = np.dot(X[:, j], residual) - lambda2 * beta
            double_tmp = ddot(& n_samples, X_data_ptr + j * n_samples, & inc,
                              & residual_data[0], & inc)
            XTR_data[j] = double_tmp - lambda2 * beta_data[j]

        tg = ((1 - tau) * omega_data[i]) / (tau + (1 - tau) * omega_data[i])
        with gil:
            double_tmp = epsilon_norm(size_groups_data[i],
                                      & XTR_data[g_start_data[i]], 1 - tg, tg)
        with nogil:
            double_tmp /= tau + (1 - tau) * omega_data[i]
            dual_scale = fmax(double_tmp, dual_scale)

    dual_scale = fmax(lambda_, dual_scale)
    return dual_scale


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def bcd_fast(double[::1, :] X, double[:] y, double[:] beta,
             double[:] XTR, double[:] residual, double dual_scale_prec,
             double[:] omega, int n_samples, int n_features, int n_groups,
             int[:] size_groups, int[:] g_start,
             double[:] norm_X, double[:] norm_X_g, double nrm2_y,
             double tau, double lambda_, double lambda2, double lambda_prec,
             double lambda_max, int max_iter, int f, double eps,
             int screen, double[:] n_DST3, double norm2_nDST3,
             double nTy_on_lambda_, double tau_w_star):
    """
        Solve the sparse-group-lasso regression with elastic-net
        We minimize
        f(beta) + lambda_1 Omega(beta) + 0.5 * lambda_2 norm(beta, 2)^2
        where f(beta) = 0.5 * norm(y - X beta, 2)^2 and
        Omega(beta) = tau norm(beta, 1) +
                      (1 - tau) * sum_g omega_g * norm(beta_g, 2)
    """

    cdef double * X_ptr = &X[0, 0]
    cdef int i = 0
    cdef int k = 0
    cdef int j = 0
    cdef int g_end = 0
    cdef int n_active_groups = n_groups
    cdef int n_active_features = n_features
    cdef int inc = 1

    cdef double dual_scale = 0.
    cdef double r = 0  # radius in the screening rules
    cdef double gap_t = 1
    cdef double double_tmp = 0

    cdef double mu_st = 0
    cdef double mu_g = 0
    cdef double tg = 0
    cdef double norm_beta_g = 0.

    cdef double cte_n = 0  # For DST3
    cdef double norm2_y_on_lambda_minus_center = 0.
    cdef double residual_norm2 = 0
    cdef double norm_beta2 = 0

    cdef double[:] beta_old = np.zeros(n_features, order='F')
    cdef double[:] XTc = np.zeros(n_features, order='F')
    cdef double[:] center = np.zeros(n_samples, order='F')
    cdef int[:] disabled_features = np.zeros(n_features,
                                             dtype=np.intc, order='F')
    cdef int[:] disabled_groups = np.zeros(n_groups, dtype=np.intc, order='F')

    if screen == DST3:

        cte_n = (nTy_on_lambda_ - tau_w_star) / (norm2_nDST3 ** 2)
        for i in range(n_samples):
            center[i] = y[i] / lambda_ - cte_n * n_DST3[i]
        norm2_y_on_lambda_minus_center = cte_n * (nTy_on_lambda_ - tau_w_star)
        # r is computed dynamically
        # XTc = np.dot(X.T, center)
        for j in range(n_features):
            XTc[j] = ddot(& n_samples, X_ptr + j * n_samples, & inc, & center[0],
                          & inc)

    if screen == DYNAMIC_SAFE:
        # center = y / lambda_
        for i in range(n_samples):
            center[i] = y[i] / lambda_
        # r is computed dynamically
        # XTc = np.dot(X.T, center)
        for j in range(n_features):
            XTc[j] = ddot(& n_samples, X_ptr + j * n_samples, & inc, & center[0],
                          & inc)

    if screen == STATIC_SAFE:

        r = abs(1 / lambda_max - 1 / lambda_) * sqrt(nrm2_y)
        # center = y / lambda_
        for i in range(n_samples):
            center[i] = y[i] / lambda_

        # XTc = np.dot(X.T, center)
        for j in range(n_features):
            XTc[j] = ddot(& n_samples, X_ptr + j * n_samples, & inc, & center[0],
                          & inc)

    if screen == GAPSAFE_SEQUENTIAL:
        # center = theta_k
        # compute l2 norm of preceeding residual
        residual_norm2 = dnrm2( & n_samples, & residual[0], & inc) ** 2
        norm_beta2 = dnrm2( & n_features, & beta[0], & inc) ** 2
        gap_t = group_dual_gap(n_samples, n_features, n_groups,
                               & size_groups[0], & g_start[0],
                               & residual[0], & y[0], & beta[0], nrm2_y,
                               & omega[0], dual_scale_prec,
                               & disabled_groups[0],
                               & disabled_features[0], residual_norm2,
                               norm_beta2, lambda_, lambda2, tau)

        # r = sqrt(2 * gap_t) / lambda_
        r = sqrt(2 * near_zero(gap_t)) / lambda_
        for j in range(n_features):
            XTc[j] = (XTR[j] - lambda2 * beta[j]) / dual_scale_prec

    if screen in [GAPSAFE_SEQUENTIAL, STATIC_SAFE]:

        fscreen_sgl(& beta[0], & XTc[0], & X_ptr[0], & residual[0], & XTR[0],
                    & disabled_features[0], & disabled_groups[0],
                    & size_groups[0], & norm_X[0], & norm_X_g[0], & g_start[0],
                    n_groups, r, n_samples, & n_active_features,
                    & n_active_groups, tau, & omega[0])

    for k in range(max_iter):

        if f != 0 and k % f == 0:

            # Compute dual point by dual scaling :
            # theta_k = residual / dual_scale
            dual_scale = dual_scaling(n_samples, n_features, n_groups,
                                      & beta[0], X_ptr, & residual[0],
                                      & XTR[0], & disabled_features[0],
                                      & disabled_groups[0], & size_groups[0],
                                      & g_start[0], lambda_, lambda2, tau,
                                      & omega[0])

            residual_norm2 = dnrm2( & n_samples, & residual[0], & inc) ** 2
            norm_beta2 = dnrm2( & n_features, & beta[0], & inc) ** 2
            gap_t = group_dual_gap(n_samples, n_features, n_groups,
                                   & size_groups[0], & g_start[0],
                                   & residual[0], & y[0], & beta[0],
                                   nrm2_y, & omega[0], dual_scale,
                                   & disabled_groups[0],
                                   & disabled_features[0],
                                   residual_norm2, norm_beta2,
                                   lambda_, lambda2, tau)

            if gap_t <= eps:
                break

            if (screen == GAPSAFE):
                # center = theta_k
                # r = sqrt(2 * gap_t) / lambda_
                r = sqrt(2 * near_zero(gap_t)) / lambda_
                for j in range(n_features):
                    XTc[j] = (XTR[j] - lambda2 * beta[j]) / dual_scale

            if (screen == DYNAMIC_SAFE):
                # center = y /lambda_
                # r = ||theta_k - center||
                if lambda_ == lambda_max:
                    r = 0
                else:
                    r = 0.
                    for i in range(n_samples):
                        r += (residual[i] / dual_scale - center[i]) ** 2
                    r = sqrt(r)

            if screen == DST3:
                # center is precomputed
                # r = sqrt(||theta_k - y/lambda||**2 - ||y/lambda_ -
                # center||**2)
                if lambda_ == lambda_max:
                    r = 0
                else:
                    double_tmp = 0.
                    for i in range(n_samples):
                        double_tmp += (residual[i] / dual_scale - y[i] / lambda_) ** 2
                    r = sqrt(double_tmp - norm2_y_on_lambda_minus_center)

            if (screen in [GAPSAFE, DYNAMIC_SAFE, DST3]):

                fscreen_sgl(& beta[0], & XTc[0], & X_ptr[0], & residual[0],
                            & XTR[0], & disabled_features[0],
                            & disabled_groups[0], & size_groups[0],
                            & norm_X[0], & norm_X_g[0],
                            & g_start[0], n_groups, r, n_samples,
                            & n_active_features, & n_active_groups, tau,
                            & omega[0])

        prox_sgl(n_samples, n_features, n_groups,
                 & beta[0], & beta_old[0], X_ptr, & residual[0], & XTR[0],
                 & disabled_features[0], & disabled_groups[0],
                 & size_groups[0], & norm_X[0], & norm_X_g[0], & g_start[0],
                 lambda_, lambda2, tau, & omega[0])

    return (dual_scale, gap_t, n_active_groups, n_active_features, k)

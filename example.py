# Author: Eugene Ndiaye (The padawan :-D)
#         Olivier Fercoq
#         Alexandre Gramfort
#         Joseph Salmon
# GAP Safe Screening Rules for Sparse-Group Lasso.
# firstname.lastname@telecom-paristech.fr

import numpy as np
from sgl_tools import generate_data
from sgl import sgl_path
import time
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

n_samples = 100
n_features = 300
size_group = 30  # all groups have size = size_group

size_groups = size_group * np.ones(n_features / size_group, order='F',
                                   dtype=np.intc)
X, y = generate_data(n_samples, n_features, size_groups, rho=0.4)
omega = np.sqrt(size_groups)

NO_SCREEN = 0
STATIC_SAFE = 1
DYNAMIC_SAFE = 2
DST3 = 3
GAPSAFE_SEQUENTIAL = 4
GAPSAFE = 5

screenings = [NO_SCREEN, STATIC_SAFE, DYNAMIC_SAFE, DST3, GAPSAFE_SEQUENTIAL,
              GAPSAFE]
screenings_names = ["NO SCREENING", "STATIC SAFE", "DYNAMIC SAFE", "DST3",
                    "GAP SAFE SEQUENTIAL", "GAP SAFE"]

eps_ = range(2, 8, 2)
times = np.zeros((len(screenings), len(eps_)))
for ieps, eps in enumerate(eps_):
    for iscreening, screening in enumerate(screenings):

        begin = time.time()

        coefs, dual_gaps, lambdas, screening_size_groups, \
            screening_size_features, n_iters = \
            sgl_path(X, y, size_groups, omega, screening,
                     eps=10 ** (-eps))

        duration = time.time() - begin
        times[iscreening, ieps] = duration


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 20,
          'legend.fontsize': 15,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': True,
          'text.latex.preamble': r'\usepackage{amsmath}',
          'figure.figsize': (8, 6)}
plt.rcParams.update(params)
plt.style.use('ggplot')

df = pd.DataFrame(times.T, columns=screenings_names)
fig, ax = plt.subplots(1, 1, figsize=(9, 6))
df.plot(kind='bar', ax=ax, rot=0)
plt.xticks(range(len(eps_)), [r"$%s$" % (np.str(t)) for t in eps_])
plt.xlabel(r"$-\log_{10}\text{(duality gap)}$", fontsize=20)
plt.ylabel(r"$\text{Time (s)}$", fontsize=20)
plt.grid(color='w')
leg = plt.legend(loc='best')
plt.show()

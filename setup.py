# Author: Eugene Ndiaye (The padawan :-D)
#         Olivier Fercoq
#         Alexandre Gramfort
#         Joseph Salmon
# GAP Safe Screening Rules for Sparse-Group Lasso.
# firstname.lastname@telecom-paristech.fr

from distutils.core import setup, Extension
import numpy as np
from Cython.Build import cythonize

print np.get_include()

extensions = [Extension("*", ["*.pyx"], include_dirs=[np.get_include()])]

setup(
    ext_modules=cythonize(extensions)
)

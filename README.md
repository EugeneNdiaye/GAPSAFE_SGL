# GAP Safe Screening Rules for Sparse-Group-Lasso
See http://arxiv.org/abs/1602.06225

Please, use the latest version in https://github.com/EugeneNdiaye/Gap_Safe_Rules

In this repository, we propose an efficient implementation to solve the
Sparse-Group-Lasso (with optional elastic net regularization) using a block
coordinate descent algorithm with safe screening rules.

Examples on synthetic dataset are presented in examples.ipynb (example.py for a
pure python version).

This package has the following requirements:

- Python (version 2.7)
- Cython
- Numpy (tested with version 0.16)
- Scipy (at least version 0.16.1)

We recommend to install or update anaconda (at least version 0.16.1).

The compilation proceed as follows:

- $ cython sgl_fast.pyx
- $ python setup.py build_ext --inplace

# Nonnegative Matrix Factorization (NMF) Methods

Python implementation of Nonnegative Matrix Factorization (NMF) methods including:
1. Nonnegative Unimodal Matrix Factorization (NUMF); to learn more about this 
method please refer to https://ieeexplore.ieee.org/iel7/9413349/9413350/09414631.pdf. The `numf_toy_example.ipynb` notebook runs NUMF and Multigrid versions of NUMF on a toy example.
2. NMF with Sum-of-Norms Clustering Regularization (NMF-SON); to learn more 
about this method please refer to https://uwaterloo.ca/computational-mathematics/sites/ca.computational-mathematics/files/uploads/files/waqas_bin_hamed_research_paper.pdf. 
Accelerated versions of the algorithm are also provided. The `hu_nmf_son.ipynb` 
notebook runs all NMF-SON implementation for Hyperspectral Unmixing.

### Installation

This package is still in development, so you need to install it as an experimental package. To install:

1. Clone repository.
2. Navigate to the main directory `nmf_method`.
3. Run `pip install -r requirements.txt` to install dependencies. Note that the packages required to run the notebooks are commented.
4. Run `pip install -e .`.

### Notes

- This package is a work in progress. Apologizes for any bugs.
- The `experimental` directory contains files related to ongoing improvements to the NMF methods and the package. 
- Please feel free to email me at waqasbinhamed@gmail.com for any concerns related to this package.

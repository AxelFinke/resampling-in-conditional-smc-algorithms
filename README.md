# Resampling in Conditional SMC Algorithms

Code for reproducing the results in the paper "Resampling in Conditional SMC Algorithms".

## Reference

To be added when available.

## Repository Structure

This repository is divided into two parts.

1. The folder `binary_model/` contains code for evaluating the invariant distribution 
   of the iterated conditional sequential Monte Carlo (iterated CSMC) algorithm in a 
   binary state-space model.
2. The folder `gaussian_model/` contains code for running the iterated CSMC algorithm 
   to illustrate the bias of naive implementations of conditional resampling in a simple 
   linear-Gaussian state-space model.

Each folder contains subfolders `cpp/` and `r/` holding C++ and R code, respectively.

## Dependencies

### R
- R (>= 4.5.2)
- Rcpp (>= 1.0.14)
- RcppArmadillo (>= 14.4.2.1)
- tidyverse (>= 2.0.0)
- cowplot (>= 1.1.3)
- magrittr (>= 2.0.3)
- here (>= 1.0.1)
- tikzDevice (>= 0.12.6)

### C++
- GCC (>= 15.2.1)
- Armadillo (bundled with RcppArmadillo 14.4.2.1)

## Installation

Clone the repository and install the required R packages:

```r
install.packages(c("Rcpp", "RcppArmadillo", "tidyverse", "cowplot", 
                   "magrittr", "here", "tikzDevice"))
```

## Usage / Reproducing the Results

To reproduce the results, run `binary_model/r/run.R` or `gaussian_model/r/run.R` in R. 
The following flags at the top of each script control execution:

```r
recompute_results <- TRUE  # set to TRUE to regenerate simulation results
recompute_figures <- TRUE  # set to TRUE to regenerate figures
```

## Licence

This project is licensed under the GNU General Public License v3.0 — see the 
[LICENSE](LICENSE) file for details.

## Third-Party Code

Parts of this code are adapted from the 
[chopthin](https://cran.r-project.org/web/packages/chopthin/index.html) R package 
by Axel Gandy, which is licensed under the GNU General Public License v3.0.

## Contact

Axel Finke  
axel.finke@ncl.ac.uk

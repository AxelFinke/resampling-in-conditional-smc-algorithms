## ========================================================================== ##
## SETUP PATHS
## ========================================================================== ##

path_to_cpp <- here::here("script", "cpp")

## ========================================================================== ##
## LOAD PACKAGE
## ========================================================================== ##

library("Rcpp")

## ========================================================================== ##
## SETUP JUST-IN-TIME COMPILER
## ========================================================================== ##

R_COMPILE_PKGS <- TRUE
R_ENABLE_JIT <- 3

## ========================================================================== ##
## SETUP RCPP
## ========================================================================== ##

Sys.setenv(
  "PKG_CXXFLAGS" = paste0(
    "-Wall -std=c++11 -I\"", path_to_cpp, "\" -I/usr/include -fopenmp -O3 -ffast-math -fno-finite-math-only -march=native"
  )
)
Sys.setenv("PKG_LIBS" = "-fopenmp")
# note that the option -ffast-math causes the compiler to ignore NaNs or Infs;
# the option -fno-finite-math-only should circumvent this

sourceCpp(
  file.path(path_to_cpp, "variational_resampling.cpp"),
  rebuild = TRUE,
  verbose = FALSE
)



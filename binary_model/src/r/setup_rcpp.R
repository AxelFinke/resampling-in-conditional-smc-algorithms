library("Rcpp")
library("here")

Sys.setenv("PKG_CXXFLAGS" = paste("-Wall -std=c++17 -I\"", here("binary_model", "src", "cpp"), "\" -I/usr/include -lCGAL -fopenmp -O3 -ffast-math -fno-finite-math-only -march=native", sep = ""))
Sys.setenv("PKG_LIBS" = "-fopenmp")
# note that the option -ffast-math causes the compiler to ignore NaNs or Infs; the option -fno-finite-math-only should circumvent this



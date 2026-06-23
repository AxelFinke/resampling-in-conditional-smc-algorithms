# Copyright (C) 2026 Axel Finke
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


library("Rcpp")
library("here")

Sys.setenv("PKG_CXXFLAGS" = paste("-Wall -std=c++17 -I\"", here("gaussian_model", "src", "cpp"), "\" -I/usr/include -lCGAL -fopenmp -O3 -ffast-math -fno-finite-math-only -march=native", sep = ""))
Sys.setenv("PKG_LIBS" = "-fopenmp")
# note that the option -ffast-math causes the compiler to ignore NaNs or Infs; the option -fno-finite-math-only should circumvent this



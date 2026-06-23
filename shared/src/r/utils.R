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



library("here")
library("tidyverse")
library("cowplot")
library("tikzDevice")
options( 
  tikzDocumentDeclaration = c(
    "\\documentclass[11pt]{article}",
    "\\usepackage{amssymb, amsmath, graphicx, mathtools, mathdots, stmaryrd}",
    "\\usepackage{tikz}" 
  )
)


save_figure <- function(
  plot,
  fig_path,
  fig_name,
  fig_type = "pdf",
  width = 8.5,
  height = 5,
  fig_res = "600",
  units = "in",
  onefile = FALSE,
  sleep = 0,
  dev_off = TRUE
) {
  
  if (dev_off == TRUE) {
    file <- file.path(fig_path, paste0(fig_name, ".", fig_type))
    if (fig_type == "pdf") {
      pdf(file = file, height = height, width = width, onefile = FALSE)
    } else if (fig_type == "png") {
      png(file = file, height = height, width = width, units = units, res = fig_res)
    } else if (fig_type == "tex") {
      tikz(file = file, standAlone = FALSE, width = width, height = height)
    }
  }
  
  print(plot)
  
  Sys.sleep(sleep)
  
  if (dev_off == TRUE) {
    dev.off()
  }
}


resample_type_levels <- c(
  "multinomial",
  "residual_multinomial",
  "stratified",
  "systematic",
  "chopthin_balanced",
  "chopthin_balanced_reordered",
  "chopthin_2015",
  "chopthin_2016",
  "naive_systematic_i",
  "naive_residual_multinomial_i",
  "naive_systematic_ii",
  "naive_residual_multinomial_ii"
)

resample_type_labels <- c(
  "Multinomial",
  "Residual-\nmultinomial",
  "Stratified",
  "Systematic",
  "Chopthin\n(balanced)",
  "Chopthin\n(balanced; reordered)",
  "Chopthin\n(2015)",
  "Chopthin\n(2016)",
  "Naive\nsystematic I",
  "Naive residual-\nmultinomial I",
  "Naive\nsystematic II",
  "Naive residual-\nmultinomial II"
)


path_selection_type_levels <- c(
  "ancestor_tracing",
  "backward_sampling",
  "ancestor_sampling"
)
path_selection_type_labels <- c(
  "Ancestor tracing",
  "Backward sampling",
  "Ancestor sampling"
)



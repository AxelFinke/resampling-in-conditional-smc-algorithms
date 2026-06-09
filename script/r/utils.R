library("here")
library("tidyverse")

library("tikzDevice")
options( 
  tikzDocumentDeclaration = c(
    "\\documentclass[11pt]{article}",
    "\\usepackage{amssymb, amsmath, graphicx, mathtools, mathdots, stmaryrd}",
    "\\usepackage{tikz}" 
  )
)


fig_type_default <- "tex"
fig_path_default <- here("tex", "arxiv")
# width_default  <- 9.3
# height_default <- 5.4

width_default  <- 8.5
height_default <- 5
height_small <- 3


save_figure <- function(
  plot,
  fig_path = fig_path_default,
  fig_name,
  fig_type = fig_type_default,
  width = width_default,
  height = height_default,
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

library("tidyverse")
library("magrittr")
library("Rcpp")
library("here")
library("tikzDevice")
library("xtable")
options(
  tikzDocumentDeclaration = c(
    "\\documentclass[11pt]{article}",
    "\\usepackage{amssymb, amsmath, graphicx, mathtools, mathdots, stmaryrd}",
    "\\usepackage{tikz}"
  )
)


source(here::here("script", "r", "setup.R"))

# Recreates ggplot colours.
gg_colour_hue <- function(n) {
  hues <- seq(from = 15, to = 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

# The names and colours for the chosen resampling schemes:
resample_types <- c("multinomial", "stratified", "systematic", "variational", "variational_alt",  "multinomial_truncated", "stratified_truncated",  "systematic_truncated")
resample_type_names <- c("multinomial", "stratified", "systematic", "variational", "weighted variational",  "multinomial (truncated)", "stratified (truncated)",  "systematic (truncated)")
col_aux <- gg_colour_hue(3)
resample_type_colours <- col_aux[c(3, 3, 3, 1, 2, 1, 2, 3)]

ess_resampling_threshold <- 1
base_size_default <- 10
alpha_fill_default <- 0.25


fig_type_default <- "tex" # default figure file format
fig_path_default <- here("tex", "fig") # default path for saving figures

width_default = 6.5
height_default = 3.8

width_large = 8.4
height_large = 5.5

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
# Normalises weights in log-space.
normalise_exp <- function(log_w) {
  log_w_max <- max(log_w)
  log_Z <- log_w_max + log(sum(exp(log_w - log_w_max)))
  return(log_w - log_Z)
}


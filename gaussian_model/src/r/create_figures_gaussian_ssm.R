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
source(here("shared", "src",  "r", "utils.R"))

resampling_scheme_names <- c(
  "Multinomial",
  "Systematic",
  "Stratified",
  "Chopthin",
  "Resdidual-\nmultinomial",
  "Na\\\"ive\nsystematic I",
  "Na\\\"ive residual-\nmultinomial I",
  "Na\\\"ive\nsystematic II",
  "Na\\\"ive residual-\nmultinomial II"
)

# Load data
# ---------------------------------------------------------------------------- #

simulation_no_to_plot <- 1
time_to_plot <- 2

data_name <- "n_iterations_without_burnin_1e+06_thinning_10_A_1_B_1_C_1_simulation_no_"


pathToOutput <- "/home/axel/Dropbox/research/output/cpp/monte-carlo-rcpp/projects/ensemble/linear/debug"
mm <- simulation_no_to_plot

df <- bind_rows(
  read_rds(file.path(pathToOutput, paste0("output_states_", data_name, mm, ".Rds"))) %>%
  mutate(resample_type = factor(resample_type, levels = resampling_scheme_names)) %>%
#     mutate(simulation_no = mm) %>%
  mutate(
    group = factor(
      paste0(n_particles, "_", as.numeric(backward_sampling_type)),
      levels = c("2_0", "3_0", "2_1", "3_1"),
      labels = c("$N = 2$", "$N = 3$", "$N = 2$\nbackward sampling", "$N = 3$\nbackward sampling")
    )
  )
)
moments <-
  tibble(
#       simulation_no = mm,
    true_mean = read_rds(file.path(pathToOutput, paste0("mu_", data_name, mm, ".Rds"))) %>% as.numeric(),
    true_var =  read_rds(file.path(pathToOutput, paste0("sigma_", data_name, mm, ".Rds"))) %>% diag()
  )

observations <- read_rds(file.path(pathToOutput, paste0("observations_", data_name, mm, ".Rds")))


# Draw figures
# ---------------------------------------------------------------------------- #


rel_widths_default <- c(1, 0.15)

mu <- moments %>% pull(true_mean) %>% .[time_to_plot]
sigma <-  moments %>% pull(true_var) %>% .[time_to_plot] %>% sqrt()
true_density <- function(x) {dnorm(x, mean = mu, sd = sigma)}
alpha_lines <- c(0.05, 1)
col_lines <- c("red", "black")
label_lines <- c("CSMC approximation", "True marginal distribution")
df %>%
  slice_sample(prop = 0.5) %>%
  filter(resample_type %in% resampling_scheme_names[6:7]) %>%
  filter(time == time_to_plot) %>%
  ggplot(mapping = aes(x = state, group = repeat_no)) +
  geom_line(stat = "density", colour = col_lines[1], alpha = alpha_lines[1]) +
  geom_function(fun = true_density, colour = col_lines[2], alpha = alpha_lines[2], inherit.aes = FALSE) +
  theme_classic() +
  geom_line(
    data = tibble(x = c(0, 0), y = c(-1, -1), type = label_lines),
    mapping = aes(x = x, y = y, colour = type, alpha = type),
    inherit.aes = FALSE
  ) +
  coord_cartesian(xlim = mu + 3 * c(-sqrt(sigma), sqrt(sigma)), ylim = c(0, 0.65)) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_colour_manual(values = col_lines, breaks = label_lines) +
  scale_alpha_manual(values = alpha_lines, breaks = label_lines) +
  labs(x = paste0("State at time $t = ", time_to_plot, "$"), y = "Density", colour = "", alpha = "") +
  guides(colour = guide_legend(override.aes = list(alpha = 1))) +
  facet_grid(cols = vars(group), rows = vars(resample_type)) -> p_gaussian_ssm




# Export figures
# ---------------------------------------------------------------------------- #

p_gaussian_ssm %>% save_figure(
  height = height_small,
  fig_name = "gaussian_ssm"
)




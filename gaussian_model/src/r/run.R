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


library("tidyverse")
library("magrittr")
library("here")

source(here("gaussian_model", "src", "r", "setup_rcpp.R"))
sourceCpp(
  here("gaussian_model", "src", "cpp", "main.cpp"),
  rebuild = TRUE,
  verbose = FALSE
)
source(here("shared", "src",  "r", "utils.R"))

rng_seed <- 1

fig_path <- here("gaussian_model", "fig")
out_path <- here("gaussian_model", "out")

recompute_results <- FALSE
recompute_figures <- TRUE



# Illustrating the bias of the "naive" implementations of conditional
# resampling, e.g., of conditional systematic and conditional
# residual-multinomial resampling.
# --------------------------------------------------------------------------- #


observations <- c(0.75, 0.62, 1.91, 0.12) %>% as.matrix() %>% t()

if (recompute_results) {

  set.seed(rng_seed)

  n_iterations <- 10^6
  burnin <- 19^4
  n_replicates <- 25

  run_simulation_study_csmc_linear_gaussian_state_space_model(
    observations = observations,
    n_iterations = n_iterations + burnin,
    n_replicates = n_replicates,
    n_particles_vec = c(2, 3),
    ess_resampling_thresholds = c(1.0),
    resample_types = c(
      "naive_systematic_i",
      "naive_residual_multinomial_i"
    ),
    path_selection_types = c(
      "ancestor_tracing",
      "backward_sampling",
      "ancestor_sampling"
    ),
    times_to_store = 1:4,
    dimensions_to_store = rep(1, times = 4),
    rng_seed = rng_seed
  ) %>% as_tibble() -> bias_of_naive_implementations_tbl


   bias_of_naive_implementations_tbl %>% write_rds(
    file.path(
      out_path,
      paste0(
        "bias_of_naive_implementations_tbl",
        "_n_observations_", length(observations),
        "_n_replicates_", n_replicates ,
        "_seed_", rng_seed,
        ".Rds"
      )
    )
  )

}


if (recompute_results) {

  ground_truth <- evaluate_ground_truth_linear_gaussian_state_space_model(observations)

  bias_of_naive_implementations_tbl %>%
    filter(iteration > burnin) %>% # remove burnin
    mutate(
      resample_type = factor(
        resample_type,
        levels = resample_type_levels,
        labels = resample_type_labels
      )
    ) %>%
    mutate(
      path_selection_type = case_when(
        path_selection_type == "ancestor_tracing" ~ "",
        path_selection_type == "backward_sampling" ~ "\nbackward sampling",
        path_selection_type == "ancestor_sampling" ~ "\nancestor sampling",
      )
    ) %>%
    mutate(aux = paste0("$N = ", n_particles, "$", path_selection_type)) %>%
    filter(time == 2) %>%
    filter(dimension == 1) %>%
    ggplot(mapping = aes(x = state, group = replicate)) +
    geom_density(
      mapping = aes(colour = "CSMC approximation"),
      key_glyph = "path"
    ) +
    geom_function(
      data = data.frame(x = NA),
      mapping = aes(colour = "True marginal distribution"),
      fun = dnorm, args = list(mean = ground_truth$means_smoothing[2], sd = sqrt(ground_truth$variances_smoothing[2])),
      inherit.aes = FALSE
    ) +
    facet_grid(resample_type ~ aux) +
    labs(x = "State at time $t = 2$", y = "Density", colour = NULL) +
    theme_classic() +
    scale_colour_manual(
      name = NULL,
      values = c(
        "True marginal distribution" = "black",
        "CSMC approximation" = alpha("red", 0.5)
      )
    ) -> p_gaussian_ssm

#   p_gaussian_ssm %>% save_figure(
#     height = height_small,
#     fig_path = fig_path,
#     fig_name = "gaussian_ssm"
#   )



}






# Comparing the performance of standard "unconditional" resampling schemes
# NOTE: This is currently not shown in the paper.
# --------------------------------------------------------------------------- #

if (recompute_results) {

  set.seed(rng_seed)

  n_data_sets <- 10^6
  n_observations <- 5
  n_replicates <- 1
  n_particles_vec <- c(1000)

  run_simulation_study_smc_linear_gaussian_state_space_model(
    n_data_sets = n_data_sets,
    n_observations = n_observations,
    n_replicates = n_replicates,
    n_particles_vec = n_particles_vec,
    ess_resampling_thresholds = c(1.0),
    resample_types = c(
      "multinomial",
      "stratified",
      "systematic",
      "residual_multinomial",
      "chopthin_balanced",
      "chopthin_balanced_reordered",
      "chopthin_2015",
      "chopthin_2016"
    ),
    n_benchmark_particles = 0,
    rng_seed = rng_seed
  ) %>% as_tibble() -> performance_of_resampling_tbl

  performance_of_resampling_tbl %>% write_rds(
    file.path(
      out_path,
      paste0(
        "performance_of_resampling_",
        "_n_data_sets_" , n_data_sets,
        "_n_observations_", n_observations,
        "_n_replicates_", n_replicates ,
        "_n_particles_vec_", n_particles_vec,
        "_seed_", rng_seed,
        ".Rds"
      )
    )
  )
}

if (recompute_figures) {

  read_rds(
    file.path(
      out_path, "performance_of_resampling_n_data_sets_1e+06_n_observations_5_n_replicates_1_n_particles_vec_1000_seed_1.Rds"
    )
  ) %>%
    mutate(
      resample_type = factor(
        resample_type,
        levels = resample_type_levels,
        labels = resample_type_labels
      )
    ) %>%
    mutate(error = abs(relative_likelihood_estimate - 1)) %>%
  #   mutate(error = (relative_likelihood_estimate - 1)^2) %>%
    group_by(resample_type, n_particles, ess_resampling_threshold) %>%
    summarise(mean = mean(error), sd = sd(error), se = sd / sqrt(n())) %>%
    ungroup() %>%
    ggplot(mapping = aes(x = resample_type, y = mean)) +
    geom_col() + #(width = 0.7) +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.2) +
  #   facet_grid(n_particles ~ ess_resampling_threshold) +
  #   geom_hline(yintercept = 1) +
    coord_cartesian(ylim = c(0.06, 0.0675)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
    theme_classic() +
    labs(x = NULL, y = "Mean-absolute error")






}











# Autocorrelation of the CSMC chain for different (conditional) resampling
# schemes. NOTE: This is currently not shown in the paper.
# --------------------------------------------------------------------------- #



if (recompute_results) {

  set.seed(rng_seed)

  n_data_sets <- 1000
  n_iterations <- 10000
  n_replicates <- 1
  burnin <- 1000
  n_particles_vec <- c(5)
  n_observations <- 10

  run_simulation_study_csmc_acf_linear_gaussian_state_space_model(
    n_data_sets = n_data_sets,
    n_observations = n_observations,
    n_iterations = n_iterations + burnin, ## 10^6 + 10^5,
    n_replicates = n_replicates,
    n_particles_vec = n_particles_vec,
    ess_resampling_thresholds = 1.0,
    resample_types = c(
      "multinomial",
      "residual_multinomial",
      "stratified",
      "systematic",
      "chopthin_balanced",
      "chopthin_balanced_reordered",
      "chopthin_2015"
    ),
    path_selection_types = c(
      "ancestor_tracing",
      "backward_sampling",
      "ancestor_sampling"
    ),
    times_to_store = 1:n_observations,
    dimensions_to_store = rep(1, times = n_observations),
    lag_max = 100,
    burnin = burnin,
    rng_seed = rng_seed
  ) %>% as_tibble() -> acf_tbl

  acf_tbl %>% write_rds(
    file.path(
      out_path,
      paste0(
        "acf",
        "_n_data_sets_" , n_data_sets,
        "_n_observations_", n_observations,
        "_n_replicates_", n_replicates ,
        "_n_iterations_", n_iterations,
        "_burnin_", burnin,
        "_n_particles_vec_", n_particles_vec,
        "_seed_", rng_seed,
        ".Rds"
      )
    )
  )

}


if (recompute_figures) {


  read_rds(
    file.path(
      out_path,
      "acf_n_data_sets_1000_n_observations_10_n_replicates_1_n_iterations_10000_burnin_1000_n_particles_vec_5_seed_1.Rds"
    )
  ) %>%
  #   filter(resample_type %in% c("chopthin_balanced", "chopthin_balanced_reordered")) %>%
  #   filter(resample_type != "chopthin_balanced_reordered") %>%
    mutate(
      resample_type = factor(
        resample_type,
        levels = resample_type_levels,
        labels = resample_type_labels
      )
    ) %>%
    mutate(
      path_selection_type = factor(
        path_selection_type,
        levels = path_selection_type_levels,
        labels = path_selection_type_labels
      )
    ) %>%
    group_by(time, dimension, lag, n_particles, replicate, resample_type, ess_resampling_threshold, path_selection_type) %>%
    summarise(acf = mean(acf)) %>%
    filter(time == 1) %>%
    ggplot(
      mapping = aes(
        x = lag,
        y = acf,
        colour = resample_type#,
  #       group = interaction(data_set, replicate)
      )
    ) +
    geom_line() +
    theme_classic() +
    scale_x_continuous(expand = 0) +
    scale_y_continuous(expand = 0) +
  #   coord_cartesian(xlim = c(0, 5)) +
  #   facet_grid(path_selection_type~ess_resampling_threshold) +
    facet_wrap(~path_selection_type) +
  labs(x = "Lag", y = "Autocorrelation", colour = NULL)


}

rm(list = ls()) # clears the workspace


source(here::here("script", "r", "helper_functions.R"))


# Runs the simulation study
# Plot the output
# ============================================================================ #

# NOTE: this script generates the figures for the simulated-data
# state-space model experiments using the results from the saved .Rds files.
# To also re-generate those results, run the script
# run_simulation_study_state_space_models.R first and then set the
# filename_addendum appropriately so that the results can be found
# by this script.

# Addition for the file names.
filename_addendum <- paste0(
  "_n_replicates_", 20,
  "_n_data_sets_", 20,
  "_n_observations_", 50,
  "_n_particles_", 1000,
  "_n_backward_sampling_particles_", 500,
  "_rng_seed_", 1
)

models <- c(
  "linear_gaussian_state_space_model",
  "stochastic_volatility_model",
  "nonlinear_state_space_model",
  "lorenz_state_space_model"
)

model_names <- c(
  "linear Gaussian",
  "stochastic volatility",
  "nonlinear",
  "Lorenz-63"
)




models_to_plot <- models %>% .[1:4] # select which of these models to plot

results_all <- tibble()
for (model in models_to_plot) {
  results_all %<>% bind_rows(
    read_rds(here("data", model, paste0("results", filename_addendum, ".Rds"))) %>%
      mutate(model = model)
  )
}

results_all %<>%
  mutate(use_smoothing_weights = if_else(use_smoothing_weights, "smoothing weights", "standard weights")) %>%
  mutate(
    use_smoothing_weights = factor(
      use_smoothing_weights,
      levels = c("standard weights", "smoothing weights")
    )
  ) %>%
  mutate(
    filtering_or_smoothing = case_when(
      metric %in% c("mse_filtering", "mse_star_filtering", "squared_mahalanobis_distance_filtering") ~ "filtering",
      metric %in% c("mse_smoothing_ancestor_tracing", "mse_star_smoothing_ancestor_tracing", "squared_mahalanobis_distance_smoothing_ancestor_tracing") ~ "smoothing\n(ancestor tracing)",
      metric %in% c("mse_smoothing_backward_sampling", "mse_star_smoothing_backward_sampling", "squared_mahalanobis_distance_smoothing_backward_sampling") ~ "smoothing\n(backward sampling)",
      TRUE ~ NA_character_
    )
  ) %>%
  mutate(
    filtering_or_smoothing = factor(
      filtering_or_smoothing,
      levels = c("filtering", "smoothing\n(ancestor tracing)", "smoothing\n(backward sampling)")
    )
  ) %>%
  mutate(
    metric = case_when(
      metric %in% c("mse_filtering", "mse_smoothing_ancestor_tracing", "mse_smoothing_backward_sampling") ~ "mse",
      metric %in% c("mse_star_filtering", "mse_star_smoothing_ancestor_tracing", "mse_star_smoothing_backward_sampling") ~ "mse_star",
      metric %in% c("squared_mahalanobis_distance_filtering", "squared_mahalanobis_distance_smoothing_ancestor_tracing", "squared_mahalanobis_distance_smoothing_backward_sampling") ~ "squared_mahalanobis_distance",
      TRUE ~ metric
    )
  ) %>%
  mutate(
    model_cat = case_when(
      model == models[1] ~ model_names[1],
      model == models[2] ~ model_names[2],
      model == models[3] ~ model_names[3],
      model == models[4] ~ model_names[4],
      TRUE ~ model
    )
  ) %>%
  mutate(
    model_cat = factor(
      model_cat,
      levels = model_names
    )
  ) %>%
  mutate(
    resample_type_cat = case_when(
      resample_type == resample_types[1] ~ resample_type_names[1],
      resample_type == resample_types[2] ~ resample_type_names[2],
      resample_type == resample_types[3] ~ resample_type_names[3],
      resample_type == resample_types[4] ~ resample_type_names[4],
      resample_type == resample_types[5] ~ resample_type_names[5],
      TRUE ~ resample_type
    )
  ) %>%
  mutate(
    resample_type_cat = factor(
      resample_type_cat,
      levels = resample_type_names
    )
  )





results_all %>%
  select(!c(dimension, time, filtering_or_smoothing)) %>%
  filter(metric == "relative_likelihood_estimation_error") %>%
  ggplot(mapping = aes(y = resample_type_cat, x = value + 1, colour = resample_type_cat, fill = resample_type_cat)) + # the "+1" is needed because we have stored $\hat{Z}/Z - 1$
  geom_boxplot(
    show.legend = FALSE,
    alpha = alpha_fill_default
  ) +
  geom_vline(linetype = "dashed", xintercept = 1) +
  theme_classic(base_size = base_size_default) +
  coord_cartesian(xlim = c(0, 2.5)) +
  labs(y = "Resampling scheme", x = "Relative likelihood estimate") +
  facet_grid(cols = vars(model_cat), rows = vars(use_smoothing_weights)) +
  scale_colour_manual(breaks = resample_type_names, values = resample_type_colours) +
  scale_fill_manual(breaks = resample_type_names, values = resample_type_colours) +
  scale_x_continuous(labels = scales::comma) +
  scale_y_discrete(limits = rev, labels = scales::label_wrap(10)) -> p_likelihood_estimates

  p_likelihood_estimates %>% save_figure(fig_name = "likelihood_estimates", height = 3)





for (model_to_plot in models_to_plot) {

  if (model_to_plot == "lorenz_state_space_model") {
    state_dimension <- 3
  } else {
    state_dimension <- 1
  }

  results_all %>%
    filter(model == model_to_plot) %>%
    filter(metric == "mse") %>%
    select(!c(dimension, metric, model)) %>%
    group_by(data_set, replicate, resample_type_cat, use_smoothing_weights, filtering_or_smoothing) %>%
    summarise(time_average_value = mean(value, na.rm = TRUE) / state_dimension) %>%
    ggplot(mapping = aes(y = resample_type_cat, x = time_average_value, colour = resample_type_cat, fill = resample_type_cat)) +
    geom_boxplot(
      show.legend = FALSE,
      alpha = alpha_fill_default
    ) +
    theme_classic(base_size = base_size_default) +
    labs(y = "Resampling scheme", x = "Average mean-square error") +
    facet_grid(cols = vars(filtering_or_smoothing), rows = vars(use_smoothing_weights), scales = "free_x") +
    scale_colour_manual(breaks = resample_type_names, values = resample_type_colours) +
    scale_fill_manual(breaks = resample_type_names, values = resample_type_colours) +
    scale_y_discrete(limits = rev, labels = scales::label_wrap(10)) -> p_mse

    if (model_to_plot %in% c("nonlinear_state_space_model", "lorenz_state_space_model")) {
      p_mse <- p_mse + scale_x_log10(guide = "axis_logticks", labels = scales::comma)
    } else {
      p_mse <- p_mse + scale_x_continuous(labels = scales::comma)
    }

  p_mse %>% save_figure(fig_name = paste0("mse_", model_to_plot), height = 3)


  results_all %>%
    filter(model == model_to_plot) %>%
    filter(metric == "squared_mahalanobis_distance") %>%
    select(!c(dimension, metric, model)) %>%
    group_by(data_set, replicate, resample_type_cat, use_smoothing_weights, filtering_or_smoothing) %>%
    summarise(time_average_value = mean(value, na.rm = TRUE) / state_dimension) %>%
    ggplot(mapping = aes(y = resample_type_cat, x = time_average_value, colour = resample_type_cat, fill = resample_type_cat)) +
    geom_boxplot(
      show.legend = FALSE,
      alpha = alpha_fill_default
    ) +
    theme_classic(base_size = base_size_default) +
    labs(y = "Resampling scheme", x = "Average squared Mahalanobis distance") +
    geom_vline(linetype = "dashed", xintercept = 1) +
    facet_grid(cols = vars(filtering_or_smoothing), rows = vars(use_smoothing_weights), scales = "free_x") +
    scale_colour_manual(breaks = resample_type_names, values = resample_type_colours) +
    scale_fill_manual(breaks = resample_type_names, values = resample_type_colours) +
    scale_x_continuous(labels = scales::comma) +
    scale_y_discrete(limits = rev, labels = scales::label_wrap(10)) -> p_smd

    if (model_to_plot %in% c("nonlinear_state_space_model", "lorenz_state_space_model")) {
      p_smd <- p_smd + scale_x_log10(guide = "axis_logticks", labels = scales::comma)
    } else {
      p_smd <- p_smd + scale_x_continuous(labels = scales::comma)
    }

   p_smd %>% save_figure(fig_name = paste0("smd_", model_to_plot), height = 3)
}




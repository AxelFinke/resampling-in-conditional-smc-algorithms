rm(list = ls()) # clears the workspace
set.seed(1) # the seed for the random numbers used by R (e.g. used for the initial parameter values)



source(here::here("script", "r", "helper_functions.R"))
source(here::here("script", "r", "gaussian_model.R"))

NN <- c(100, 1000, 10000) # number of particles
MM <- NN # number of resampled particles
n_simulations <- 1000

x_grid <- seq(from = -3.5, to = 3.5, length = 1000)

resample_types <- c("multinomial", "stratified", "systematic", "variational", "variational_alt")

simulate_results <- FALSE # if TRUE, then the results are re-generated; if FALSE, then the saved .Rds results are used

if (simulate_results) {

  simulations <- 1:n_simulations
  df <- tibble()
  for (nn in seq_along(NN)) {
    print(paste0("nn: ", nn, " out of ", length(NN)))
    for (mm in seq_along(MM)) {

      print(paste0("mm: ", mm, " out of ", length(MM)))
      for (resample_type in resample_types) {
        df_aux_1 <- tibble()
        for (simulation in simulations) {
          x <- sample_from_proposal(n = NN[nn])
          aux <- resample_cpp(evaluate_log_potential_function(x) %>% normalise_exp() %>% exp(), MM[mm], resample_type)
          df_aux_1 %<>% bind_rows(
            tibble(
              simulation = simulation,
              resampled_particles = x[aux$parent_indices],
              resampled_weights = aux$resampled_weights,
              from = min(x[aux$parent_indices]),
              to = max(x[aux$parent_indices])
            )
          )
        }
        # Global values for "from" and "to" across all simulations for this configuration
        from <- df_aux_1 %>% pull(from) %>% min()
        to <- df_aux_1 %>% pull(to) %>% max()

        density_list <- vector("list", length(simulations))
        for (ss in seq_along(simulations)) {
          df_aux_2 <- df_aux_1 %>%
            filter(simulation == simulations[ss])
          density_list[[ss]] <- density(
            df_aux_2$resampled_particles,
            weights = df_aux_2$resampled_weights, #### / sum(df_aux_2$resampled_weights),
            from = from,
            to = to
          )
        }
        y_mean <- sapply(density_list, function(d) approx(d$x, d$y, x_grid, yleft = 0, yright = 0)$y) %>% rowMeans
        y_median <- sapply(density_list, function(d) approx(d$x, d$y, x_grid, yleft = 0, yright = 0)$y) %>% apply(1, median, na.rm = TRUE)


        df %<>% bind_rows(
          tibble(
            N = NN[nn],
            M = MM[mm],
            from = from,
            to = to,
            x = x_grid,
            y_mean = y_mean,
            y_median = y_median,
            resample_type = resample_type
          )
        )

      }
    }
  }

  df %>% write_rds(here::here("data", "truncation_bias.Rds"))

} else {

  df <- read_rds(here::here("data", "truncation_bias.Rds"))

}



df %>%
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
  ) %>%
  mutate(N = paste0("$N = ", N, "$")) %>%
  mutate(M = paste0("$M = ", M, "$")) %>%
  select(!c(from, to)) %>%
  group_by(N, M, x, resample_type_cat) %>%
  summarise(mean_density = mean(y_mean, na.rm = TRUE)) -> df_plot


df_plot %>%
  ggplot(
    mapping = aes(
      x = x,
      y = mean_density,
      group = resample_type_cat,
      colour = resample_type_cat
    )
  ) +
  geom_line(data = df_plot %>% filter(!(resample_type_cat == resample_type_names[5]))) +
  geom_line(data = df_plot %>% filter(resample_type_cat == resample_type_names[5])) +
  geom_function(fun = evaluate_target_density, inherit.aes = FALSE, colour = "black", n = 1000) +
  labs(x = "$x$", y = "Density", colour = "Resampling scheme") +
  coord_cartesian(xlim = c(-3, 3), ylim = c(0, 1 / 2)) +
  theme_classic(base_size = base_size_default) +
  facet_grid(cols = vars(N), rows = vars(M)) +
  scale_colour_manual(breaks = resample_type_names[1:5], values = resample_type_colours[1:5]) -> p_truncation_bias

  p_truncation_bias %>% save_figure(fig_name = "truncation_bias", height = 5)

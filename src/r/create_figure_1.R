rm(list = ls()) # clears the workspace
set.seed(1) # the seed for the random numbers used by R (e.g. used for the initial parameter values)

source(here::here("script", "r", "helper_functions.R"))
source(here::here("script", "r", "gaussian_model.R"))


N <- 1000 # number of particles
M <- 1000 # number of resampled particles
n_simulations <- 1000

resample_types <- c("multinomial", "stratified", "systematic", "variational", "variational_alt")

x_grid <- seq(from = -3.5, to = 3.5, length = 1000)

simulate_results <- FALSE # if TRUE, then the results are re-generated; if FALSE, then the saved .Rds results are used

if (simulate_results) {

  df <- tibble()
  df_avg <- tibble() # holds the average of the kernel density estimates

  for (resample_type in resample_types) {

    print(paste0("resample_type: ", resample_type))

    df_aux_1 <- tibble()
    for (ss in 1:n_simulations) {
      x <- sample_from_proposal(n = N)
      aux <- resample_cpp(evaluate_log_potential_function(x) %>% normalise_exp() %>% exp(), M, resample_type)

      from <- min(x[aux$parent_indices])
      to <- max(x[aux$parent_indices])

      df_aux_1 %<>% bind_rows(
        tibble(
          simulation = ss,
          resampled_particles = x[aux$parent_indices],
          resampled_weights = aux$resampled_weights,
          from = from,
          to = to
        )
      )
    }
    # Global values for "from" and "to" across all simulations for this configuration
    from <- df_aux_1 %>% pull(from) %>% min()
    to <- df_aux_1 %>% pull(to) %>% max()

    density_list <- vector("list", n_simulations)
    for (ss in 1:n_simulations) {
      df_aux_2 <- df_aux_1 %>%
        filter(simulation == ss)
      density_list[[ss]] <- density(
        df_aux_2$resampled_particles,
        weights = df_aux_2$resampled_weights, ##/ sum(df_aux_2$resampled_weights),
        from = from,
        to = to
      )

      dens <- density(
        df_aux_2$resampled_particles,
        weights = df_aux_2$resampled_weights, ### / sum(df_aux_2$resampled_weights),
        from = from,
        to = to
      )
      df %<>% bind_rows(
        tibble(
          simulation = ss,
          from = from,
          to = to,
          x = x_grid,
          y = approx(dens$x, dens$y, x_grid, yleft = 0, yright = 0)$y,
          resample_type = resample_type
        )
      )
    }
    y_mean <- sapply(density_list, function(d) approx(d$x, d$y, x_grid, yleft = 0, yright = 0)$y) %>% rowMeans
    y_median <- sapply(density_list, function(d) approx(d$x, d$y, x_grid, yleft = 0, yright = 0)$y) %>% apply(1, median, na.rm = TRUE)

    df_avg %<>% bind_rows(
      tibble(
        from = from,
        to = to,
        x = x_grid,
        y_mean = y_mean,
        y_median = y_median,
        resample_type = resample_type
      )
    )
  }

  df %>% write_rds(here::here("data", "df_weighting_bias.Rds"))
  df_avg %>% write_rds(here::here("data", "df_avg_weighting_bias.Rds"))

} else {

  df <- read_rds(here::here("data", "df_weighting_bias.Rds"))
  df_avg <- read_rds(here::here("data", "df_avg_weighting_bias.Rds"))

}

n_simulations <- df %>% pull(simulation) %>% max()

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
  group_by(resample_type_cat, x) %>%
  summarise(y_min = quantile(y, probs = 0.05, na.rm = TRUE), y_max = quantile(y, probs = 0.95, na.rm = TRUE)) %>%
  ungroup() %>%
  ggplot(
    mapping = aes(
      x = x,
#       y = y,
      ymin = y_min,
      ymax = y_max,
      group = resample_type_cat,
      fill = resample_type_cat
    )
  ) +
  geom_ribbon(alpha = 0.2, show.legend = FALSE) +
  geom_line(
    data = df_avg %>%
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
        ),
    mapping = aes(
      x = x,
      y = y_mean,
      colour = resample_type_cat
    ),
    inherit.aes = FALSE,
    show.legend = FALSE,
    linewidth = 1
  ) +
  geom_function(fun = evaluate_target_density, inherit.aes = FALSE, colour = "black", n = 10000) +
  labs(x = "$x$", y = "Density") +
  coord_cartesian(xlim = c(-3, 3), ylim = c(0, 0.45)) +
  scale_colour_manual(breaks = resample_type_names[1:5], values = resample_type_colours[1:5]) +
  scale_fill_manual(breaks = resample_type_names[1:5], values = resample_type_colours[1:5]) +
  theme_classic(base_size = base_size_default) +
  facet_grid(cols = vars(resample_type_cat)) -> p_weighting_bias

  p_weighting_bias %>% save_figure(fig_name = "weighting_bias", height = 2.4)


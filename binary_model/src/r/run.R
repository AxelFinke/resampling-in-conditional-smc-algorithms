
cd "/home/axel/code/resampling-in-conditional-smc-algorithms"

R

library("here")
library("tidyverse")

source(here("binary_model", "src", "r", "setup_rcpp.R"))
sourceCpp(
  here("binary_model", "src", "cpp", "main.cpp"),
  rebuild = TRUE,
  verbose = FALSE
)
source(here("shared", "src",  "r", "utils.R"))


fig_path <- here("binary_model", "fig")
out_path <- here("binary_model", "out")

recompute_results <- FALSE
recompute_figures <- TRUE




if (recompute_results) {

  n_grid_coordinates <- 3 ### 26
  n_particles <- c(2, 3)
  n_observations <- 5

  evaluate_naive_conditional_resampling_bias(
    n_particles,
    n_observations,
    1:n_grid_coordinates / (n_grid_coordinates + 1), # epsilon
    1:n_grid_coordinates / (n_grid_coordinates + 1), # delta
    c(
      "multinomial",
      "residual_multinomial",
      "systematic",
      "naive_systematic_i",
      "naive_residual_multinomial_i",
      "naive_systematic_ii",
      "naive_residual_multinomial_ii"
    ),
    c("ancestor_tracing", "backward_sampling")
  ) %>% as_tibble() -> df


  df %>% write_rds(
    file.path(
      out_path,
      paste0(
        "2026-06-22_binary_state_space_model_",
        n_grid_coordinates,
        "_grid_coordinates.Rds"
      )
    )
  )

}




if (recompute_figures) {

  # Load data
  # ---------------------------------------------------------------------------- #

  # read_rds(here("data", "2024-04-12_binary_state_space_model_26_grid_coordinates.Rds")) %>%
  df %>% #######################################TODO remove this and uncomment previous line
    filter(n_particles %in% c(2, 3)) %>%
    filter(n_observations == 5) %>%
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
    mutate(group = paste0("$N = ", n_particles, "$", path_selection_type)) -> df

  # Draw figures
  # ---------------------------------------------------------------------------- #

  xbreaks_default <- c(0.25, 0.5, 0.75)
  ybreaks_default <- xbreaks_default
  rel_widths_default <- c(1, 0.15)


  df %>%
    ggplot(mapping = aes(x = delta, y = epsilon, fill = tvd)) +
    geom_tile() +
    scale_x_continuous(expand = c(0, 0), breaks = xbreaks_default) +
    scale_y_continuous(expand = c(0, 0), breaks = ybreaks_default) +
    scale_fill_viridis_c(trans = "log10") +
    # scale_fill_gradient(trans = "log10", low = "white", high = "blue") +
    facet_grid(rows = vars(group), cols = vars(resample_type)) +
    labs(
      x = "$\\delta$",
      y = "$\\varepsilon$",
      fill = "$\\lVert \\pi_{T} - \\pi_{T}^{\\text{\\textsc{csmc}}} \\rVert_{\\text{\\textsc{tv}}}$",
    ) +
    theme_classic() +
    theme(
      legend.margin = margin(0, -133, 0, 0),
      legend.box.margin = margin(2, -133, 2, 2)
    ) -> p_bias_log_scale

  # p_bias_log_scale


  df %>%
    ggplot(mapping = aes(x = delta, y = epsilon, fill = tvd)) +
    geom_tile() +
    scale_x_continuous(expand = c(0, 0), breaks = xbreaks_default) +
    scale_y_continuous(expand = c(0, 0), breaks = ybreaks_default) +
    # scale_fill_viridis_c(trans = "log10", option = "A") +
    scale_fill_viridis_c() +
    facet_grid(rows = vars(group), cols = vars(resample_type)) +
    labs(
      x = "$\\delta$",
      y = "$\\varepsilon$",
      fill = "$\\lVert \\pi_{T} - \\pi_{T}^{\\text{\\textsc{csmc}}} \\rVert_{\\text{\\textsc{tv}}}$",
    ) +
    theme_classic() +
    theme(
      legend.margin = margin(0, -133, 0, 0),
      legend.box.margin = margin(2, -133, 2, 2)
    ) -> p_bias_linear_scale

  # p_bias_linear_scale

  df %>%
    ggplot(mapping = aes(x = delta, y = epsilon, fill = autocorrelation)) +
    geom_tile() +
    scale_x_continuous(expand = c(0, 0), breaks = xbreaks_default) +
    scale_y_continuous(expand = c(0, 0), breaks = ybreaks_default) +
    # scale_fill_continuous(limits = c(0, 1)) +
    scale_fill_viridis_c(limits = c(0, 1)) +
    facet_grid(rows = vars(group), cols = vars(resample_type)) +
    labs(
      x = "$\\delta$",
      y = "$\\varepsilon$",
      fill = "Autocorrelation",
    ) +
    theme_classic() -> p_autocorrelation




  # Export figures
  # ---------------------------------------------------------------------------- #

  l_bias_linear_scale <- get_legend(p_bias_linear_scale)
  p_bias_linear_scale <- p_bias_linear_scale + theme(legend.position = "none")
  l_bias_log_scale <- get_legend(p_bias_log_scale)
  p_bias_log_scale <- p_bias_log_scale + theme(legend.position = "none")
  l_autocorrelation <- get_legend(p_autocorrelation)
  p_autocorrelation <- p_autocorrelation + theme(legend.position = "none")

  ggdraw(
    plot_grid(
      p_bias_linear_scale, l_bias_linear_scale,
      ncol = 2, rel_widths = rel_widths_default
    )
  ) -> bias_linear_scale

  ggdraw(
    plot_grid(
      p_bias_log_scale, l_bias_log_scale,
      ncol = 2, rel_widths = rel_widths_default
    )
  ) -> bias_log_scale

  ggdraw(
    plot_grid(
      p_autocorrelation, l_autocorrelation,
      ncol = 2, rel_widths = rel_widths_default
    )
  ) -> autocorrelation

  bias_linear_scale %>% save_figure(
    fig_path = fig_path,
    fig_name = "toy_model_bias_linear_scale"
  )

  bias_log_scale %>% save_figure(
    fig_path = fig_path,
    fig_name = "toy_model_bias_log_scale"
  )

  autocorrelation %>% save_figure(
    fig_path = fig_path,
    fig_name = "toy_model_autocorrelation"
  )





}

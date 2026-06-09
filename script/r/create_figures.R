library("here")
library("tidyverse")
library("cowplot")
source(here("script", "r", "utils.R"))


resampling_scheme_names <- c(
  "Multinomial", 
  "Systematic", 
  "Resdidual-\nmultinomial",
  "Na\\\"ive\nsystematic I",
  "Na\\\"ive residual-\nmultinomial I",
  "Na\\\"ive\nsystematic II",
  "Na\\\"ive residual-\nmultinomial II"
)

# Load data
# ---------------------------------------------------------------------------- #

read_rds(here("data", "2024-04-12_binary_state_space_model_26_grid_coordinates.Rds")) %>%
# df %>% #######################################TODO remove this and uncomment previous line
  filter(N %in% c(2, 3)) %>%
  filter(T == 5) %>%
  mutate(
    group = factor(
      paste0(N, "_", as.numeric(use_backward_sampling)), 
      levels = c("2_0", "3_0", "2_1", "3_1"),
      labels = c("$N = 2$", "$N = 3$", "$N = 2$\nbackward sampling", "$N = 3$\nbackward sampling")
    )
  ) %>%
  mutate(resampling_scheme = factor(resampling_scheme, levels = 0:6, labels = resampling_scheme_names)) -> df

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
  facet_grid(rows = vars(group), cols = vars(resampling_scheme)) +
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
  facet_grid(rows = vars(group), cols = vars(resampling_scheme)) +
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
  facet_grid(rows = vars(group), cols = vars(resampling_scheme)) +
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
  fig_name = "toy_model_bias_linear_scale"
)

bias_log_scale %>% save_figure(
  fig_name = "toy_model_bias_log_scale"
)

autocorrelation %>% save_figure(
  fig_name = "toy_model_autocorrelation"
)





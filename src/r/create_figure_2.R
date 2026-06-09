rm(list = ls()) # clears the workspace
set.seed(1) # the seed for the random numbers used by R (e.g. used for the initial parameter values)


source(here::here("script", "r", "helper_functions.R"))
source(here::here("script", "r", "gaussian_model.R"))

n_simulations <- 10000

NN <- c(100, 1000, 10000)  # number of particles
MM <- NN # number of resampled particles
alpha <- 1
beta_grid <- exp(seq(from = log(0.0001), to = log(1), length = 200000))
resample_types <- c("multinomial", "stratified", "systematic", "variational")

simulate_results <- FALSE # if TRUE, then the results are re-generated; if FALSE, then the saved .Rds results are used

if (simulate_results) {

  is_truncated <- matrix(NA, length(beta_grid), n_simulations)
  df <- tibble()

  for (nn in seq_along(NN)) {
    print(paste0("nn: ", nn, " out of ", length(NN)))
    for (mm in seq_along(MM)) {
      print(paste0("mm: ", mm, " out of ", length(MM)))
      for (resample_type in resample_types) {
        print(paste0("resample_type: ", resample_type))
        for (ss in 1:n_simulations) {

          log_w <- sample_from_proposal(n = NN[nn]) %>% evaluate_log_potential_function()
          aux <- resample_cpp(exp(normalise_exp(log_w)), MM[mm], resample_type)
          end.time <- Sys.time()

          # Binary segmentation to efficiently compute whether or not the resampled distribution
          # has all its potential functions above the threshold $\beta$ or not.
          check_truncation <- function(idx) {if_else(any(log_w[aux$parent_indices] < log(beta_grid[idx])), FALSE, TRUE)}

          a <- 1
          c <- length(beta_grid)
          b <- floor((a + c) / 2)
          if (check_truncation(a) == FALSE) {
  #           print("truncation for no boundaries")
            is_truncated[, ss] <- FALSE
          } else if (check_truncation(c) == TRUE) {
  #           print("truncation for all boundaries")
            is_truncated[, ss] <- TRUE
          } else {
            while (b != a) {
              if (check_truncation(b) == TRUE) { # i.e., if there is already truncation at Position $b$ so that we need to search in right segment
  #                 print("truncation at b")
                a <- b
              } else { # i.e., if there is no truncation at the half-way point so that we need to search in left segment
  #                   print("no truncation at b")
                c <- b
              }
              b <- floor((a + c) / 2)
            }
            is_truncated[, ss] <- FALSE
            is_truncated[1:a, ss] <- TRUE
          }

        }
        df %<>% bind_rows(
          tibble(
            N = NN[nn],
            M = MM[mm],
            alpha = alpha,
            beta = beta_grid,
            space_truncation = sqrt(- log(beta_grid) * 2 / lambda),
            resample_type = resample_type,
            probability_type = "empirical estimate",
            probability = rowMeans(is_truncated) %>% as.numeric()
          )
        )
      }
    }
  }


  first_moments <- evaluate_moment_cpp(beta_grid, 1, lambda)
  second_moments <- evaluate_moment_cpp(beta_grid, 2, lambda)

  for (nn in seq_along(NN)) {
    print(paste0("nn: ", nn, " out of ", length(NN)))
    for (mm in seq_along(MM)) {
      print(paste0("mm: ", mm, " out of ", length(MM)))
      df %<>% bind_rows(
        tibble(
          N = NN[nn],
          M = MM[mm],
          alpha = alpha,
          beta = beta_grid,
          space_truncation = sqrt(- log(beta_grid) * 2 / lambda),
          resample_type = "variational",
          probability_type = "lower bound",
          probability = evaluate_lower_bound_vectorised(NN[nn], MM[mm], first_moments, second_moments, alpha = 1) %>% as.numeric()
        )
      )
    }
  }

  df %>% write_rds(here::here("data", "bounds.Rds"))

} else {

  df <- read_rds(here::here("data", "bounds.Rds"))

}

beta_grid_subset <- beta_grid[floor(seq(from = 1, to = length(beta_grid), length = 200))]

# Subsample the data frame to ensure that the LaTeX compilation does not run out of memory.
df %>%
  filter(probability <= 1) %>%
  group_by(N, M) %>%
  filter(resample_type != "variational") %>%
  filter(probability_type == "empirical estimate") %>%
  filter(beta %in% beta_grid_subset) -> df_others

df %>%
  filter(probability <= 1) %>%
  group_by(N, M) %>%
  filter(resample_type == "variational") %>%
  filter(probability_type == "empirical estimate") %>%
  filter((probability > 0.99) | (probability < 0.01)) %>%
  filter(beta %in% beta_grid_subset) -> df_no_lower_bound_1


df %>%
  filter(probability <= 1) %>%
  group_by(N, M) %>%
  filter(resample_type == "variational") %>%
  filter(probability_type == "empirical estimate") %>%
  filter((probability <= 0.99) & (probability >= 0.01)) %>%
  slice_sample(n = 200)  -> df_no_lower_bound_2
#   slice(seq(from = 1, to = n(), by = 3)) -> df_no_lower_bound_2

df %>%
  filter(probability <= 1) %>%
  group_by(N, M) %>%
  filter(resample_type == "variational") %>%
  filter(probability_type == "lower bound") %>%
  filter(probability > 0.99) %>%
  filter(beta %in% beta_grid_subset) -> df_lower_bound_1

df %>%
  filter(probability <= 1) %>%
  group_by(N, M) %>%
  filter(resample_type == "variational") %>%
  filter(probability_type == "lower bound") %>%
  filter(probability <= 0.99) %>%
  slice_sample(n = 200)  -> df_lower_bound_2

df_others %>%
  bind_rows(df_no_lower_bound_1) %>%
  bind_rows(df_no_lower_bound_2) %>%
  bind_rows(df_lower_bound_1) %>%
  bind_rows(df_lower_bound_2) %>%
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
  ggplot(
    mapping = aes(
      x = space_truncation,
      y = probability,
      group = interaction(resample_type_cat, probability_type),
      colour = resample_type_cat,
      linetype = probability_type
    )
  ) +
  geom_line() +
  coord_cartesian(
    ylim = c(0, 1)
  ) +
  theme_classic(base_size = base_size_default) +
  facet_grid(cols = vars(N), rows = vars(M)) +
  scale_linetype_manual(breaks = c("empirical estimate", "lower bound"), values = c("solid", "dashed")) +
  scale_colour_manual(breaks = resample_type_names[1:4], values = resample_type_colours[1:4]) +
  labs(
    x = "$x^*$",#\n(i.e., $\\tilde{\\pi}^M$ does not have mass outside of $[x^*, x^*]$)",
    y = "Probability",
    colour = "Resampling scheme",
    linetype = "Truncation probability"
  ) -> p_lower_bound

  p_lower_bound %>% save_figure(fig_name = "bounds", height = 4.5)

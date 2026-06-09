library("here")
library("tidyverse")

source(here("script", "r", "setup_rcpp.R"))
sourceCpp(
  here("script", "cpp", "toy_example.cpp"),
  rebuild = TRUE,
  verbose = FALSE
)

resampling_scheme_names <- c(
  "multinomial", # 0
  "systematic", # 1
  "resdidual-multinomial", #2
  "naive systematic I", #3
  "naive residual-multinomial I", #4
  "naive systematic II", #5
  "naive residual-multinomial II" #6
)

n_grid_coordinates <- 26

aux <- evaluate_naive_conditional_resampling_bias(
  c(2, 3), # number of particles
  c(5), # number of observations
  1:n_grid_coordinates / (n_grid_coordinates + 1), # epsilon
  1:n_grid_coordinates / (n_grid_coordinates + 1), # delta
  0:6, # the resampling schemes; 0: "multinomial"; 1: "systematic"; 2: "residual-multinomial"; 3: "naive systematic I"; 4: "naive residual-multinomial I"; 5: "naive systematic II"; 6: "naive residual-multinomial II",
  0:1 # should backward sampling be used? 0: "no"; 1: "yes"
)

df <- tibble(
  N = aux$N %>% as.numeric(),
  T = aux$T %>% as.numeric(),
  epsilon = aux$epsilon %>% as.numeric(),
  delta = aux$delta %>% as.numeric(),
  resampling_scheme = aux$resample_type %>% as.numeric(),
  use_backward_sampling = aux$backward_sampling_type %>% as.logical(),
  forward_kl = aux$forward_kl %>% as.numeric(),
  reverse_kl = aux$reverse_kl %>% as.numeric(),
  tvd = aux$tvd %>% as.numeric(),
  autocorrelation = aux$autocorrelation %>% as.numeric(),
)

df %>% write_rds(
  here(
    "data", 
    paste0(
      "2024-04-12_binary_state_space_model_", 
      n_grid_coordinates,
      "_grid_coordinates.Rds"
    )
  )
)

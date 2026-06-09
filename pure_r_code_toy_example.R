# Toy binary state-space model with $T = 2$ time steps.
# NOTE: The code currently seems to fail if delta or epsilon are equal to 0 or 1.

# TODO: implement the density of the SMC algorithm to verify that the "naive" resampling schemes are correct!

library("tidyverse")
library("magrittr")
# library("gtools") # needs to be installed! provides the function permutations()

epsilon <- 0.8 # 0.41
delta <- 0.25 # 0.37
n_particles <- 2

space_paths <- list(c(0, 0), c(1, 1), c(0, 1), c(1, 0))

# Evaluates the log of the initial density $M_1$.
evaluate_log_initial_density <- function(x, epsilon) {
  log((1 - epsilon) * (x == 0) + epsilon * (x == 1))
}
# Evaluates the log of the transition density $M_t$, for $t > 1$.
evaluate_log_transition_density <- function(x_new, x_old, epsilon) {
  log((1 - epsilon) * (x_new == x_old) + epsilon * (x_new == 1 - x_old)) 
}
# Evaluates the log of the potential function $G_t$.
evaluate_log_potential_function <- function(x, delta = delta) {
  log((1 - delta) * (x == 0) + delta * (x == 1))
}
# Evaluates the log of the unnormalised target density.
evaluate_log_unnormalised_target_density <- function(x, epsilon, delta) {
  evaluate_log_initial_density(x[1], epsilon = epsilon) +
    evaluate_log_transition_density(x_new = x[2], x_old = x[1], epsilon = epsilon) + 
    evaluate_log_potential_function(x[1], delta = delta) +
    evaluate_log_potential_function(x[2], delta = delta)
}
# Evaluates the log of the normalising constant.
evaluate_log_normalising_constant <- function(epsilon, delta) {
#   normalising_constant <- 0
#   for (pathe in space_paths) {
#     normalising_constant <- normalising_constant + 
#       exp(
#         evaluate_log_unnormalised_target_density(
#           x = path, 
#           epsilon = epsilon, 
#           delta = delta
#         )
#       )
#   }
  # Direct calculation (gives the same answer as the above loop):
  normalising_constant <- (1 - epsilon)^2 * (1 - delta)^2 + 
    epsilon * (1 - epsilon) * delta^2 +
    (1 - epsilon) * epsilon * (1 - delta) * delta +
    epsilon^2 * delta * (1 - delta)
  
  return(log(normalising_constant))
}


# Evaluates the log of the unnormalised target density.
evaluate_log_target_density <- function(x, epsilon, delta) {
  evaluate_log_unnormalised_target_density(x = x, epsilon = epsilon, delta = delta) -
    evaluate_log_normalising_constant(epsilon = epsilon, delta = delta)
}
# Evaluates the log of the initial index density $\lambda_1$.
evaluate_log_initial_lambda <- function(k, n_particles) {-log(n_particles)}
# Evaluates the log of the (unconditional) resampling density $\rho_{t-1}$.
evaluate_log_resampling_density <- function(a, W, resampling_scheme) {

  M <- length(a)

  if (resampling_scheme == "multinomial") {

    log_rho <- sum(log(W[a]))

  } else if (resampling_scheme == "systematic") {

    if (!is.unsorted(a)) {

      is_inside_support <- TRUE
      # lower and upper bounds on the uniform random variable
      # implied by each stratum:
      u_lower <- rep(NA, times = M)
      u_upper <- rep(NA, times = M)

      Q <- M * cumsum(W)

      m <- 1

      while (is_inside_support && (m <= M)) {

        if (a[m] > 1) {
          u_lower[m] <- max(Q[a[m] - 1], m - 1) - (m - 1)
        } else {
          u_lower[m] <- 0
        }
        u_upper[m] <- min(Q[a[m]], m) - (m - 1)

        if ((u_lower[m] >= 0) && (u_upper[m] <= 1)) {
          m <- m + 1
        } else {
          is_inside_support <- FALSE
        }

      }

      if (is_inside_support && (min(u_upper) > max(u_lower))) {
        log_rho <- log(min(u_upper) - max(u_lower))
      } else {
        log_rho <- -Inf
      }

    } else {
      log_rho <- -Inf
    }

  } else if (resampling_scheme == "residual-multinomial") {

    MW <- M * W
    MW_floor <- floor(MW)
    alpha <- rep(1:length(W), times = MW_floor)
    M0 <- sum(MW_floor)

    if (all(a[1:M0] == alpha)) {
      if (M0 < M) {
        log_v <- log(MW - MW_floor) - log(M - M0)
        log_rho <- sum(log_v[a[(M0 + 1):M]])
      } else {
        log_rho <- 0
      }
    } else {
      log_rho <- -Inf
    }
  }

  return(log_rho)

}
# Evaluates the log of the product of the index density $\lambda_t$ and
# the conditional resampling density $\rho_{t-1}^{-k_t}$.
evaluate_log_conditional_resampling_density <- function(
  a, # vector of ancestor indices
  k_new, # index of the reference path at time $t$
  k_old, # index of the reference path at time $t - 1$
  W, # vector of self-normalised weights
  resampling_scheme # resampling scheme
) {

  N <- length(W)
  if (a[k_new] == k_old) {
    is_inside_support <- TRUE
  } else {
    print("WARNING: a[k_new] != k_old!")
    is_inside_support <- FALSE
  }
    
  if (is_inside_support) {

    if (resampling_scheme == "multinomial") {

      log_density <- evaluate_log_resampling_density(
        a = a[setdiff(1:N, k_new)],
        W = W,
        resampling_scheme = "multinomial"
      ) - log(N)
      
    } else if (resampling_scheme == "systematic") {

      if (!is.unsorted(a)) {
        log_density <- evaluate_log_resampling_density(
          a = a,
          W = W,
          resampling_scheme = "systematic"
        ) - log(N * W[k_old])

      } else {
        is_inside_support <- FALSE
      }

    
    } else if (resampling_scheme == "naive systematic") {

      if ((k_new == 1) && !is.unsorted(a[2:N])) {
        log_density <- evaluate_log_resampling_density(
          a = a[2:N],
          W = W,
          resampling_scheme = "systematic"
        )
      } else {
        is_inside_support <- FALSE
      }

    } else if (resampling_scheme == "residual-multinomial") {

      log_density <- evaluate_log_resampling_density(
        a = a,
        W = W,
        resampling_scheme = "residual-multinomial"
      ) - log(N * W[k_old])
    
    } else if (resampling_scheme == "naive residual-multinomial") {
      
      if (k_new == 1) { # assuming that the naive resampling scheme puts the reference path in Position 1
        log_density <- evaluate_log_resampling_density(
          a = a[2:N],
          W = W,
          resampling_scheme = "residual-multinomial"
        )
      } else {
        is_inside_support <- FALSE
      }

    }
  }
  
  if (is_inside_support) {
    return(log_density)
  } else {
    return(-Inf)
  }
  
}
# Normalises weights in log-space
normalise_weights_in_log_space <- function(log_w) {
  log_w_max = max(log_w)
  log_Z = log_w_max + log(sum(exp(log_w - log_w_max)))
  return(log_w - log_Z)
}
# Evaluates the log of the extended target density of the
# conditional SMC algorithm
evaluate_log_extended_target_density_csmc <- function(
  path_out, # output reference path, i.e., $\mathbf{x}_{1:2}'$
  path_in, # input reference path, i.e., $\mathbf{x}_{1:2}$
  particles1, # particles at time 1, i.e., $\mathbf{x}_1^{1:2}$
  particles2, # particles at time 2, i.e., $\mathbf{x}_2^{1:2}$
  a, # vector of ancestor indices for time 1, i.e., $a_1^{1:2}$
  k, # particle indices of the input reference path, i.e., $k_{1:2}$
  l, # particle indices of the output reference path, i.e., $l_{1:2}$
  resampling_scheme,
  epsilon,
  delta
) {

  if (
    (particles1[k[1]] == path_in[1]) &&
    (particles2[k[2]] == path_in[2]) &&
    (particles1[l[1]] == path_out[1]) &&
    (particles2[l[2]] == path_out[2]) &&
    (a[k[2]] == k[1]) &&
    (a[l[2]] == l[1])
  ) {

    log_W1 <- evaluate_log_potential_function(particles1, delta = delta) %>%
      normalise_weights_in_log_space()
    log_W2 <- evaluate_log_potential_function(particles2, delta = delta)  %>%
      normalise_weights_in_log_space()
    n_particles <- length(particles1)

    log_density <- evaluate_log_initial_lambda(k = k[1], n_particles = n_particles) +
      evaluate_log_conditional_resampling_density(
        a = a,
        k_new = k[2],
        k_old = k[1],
        W = exp(log_W1),
        resampling_scheme = resampling_scheme
      ) +
      evaluate_log_initial_density(particles1, epsilon = epsilon) %>% sum() -
      evaluate_log_initial_density(particles1[k[1]], epsilon = epsilon) +
      evaluate_log_transition_density(x_new = particles2, x_old = particles1[a], epsilon = epsilon) %>% sum() -
      evaluate_log_transition_density(x_new = particles2[k[2]], x_old = particles1[k[1]], epsilon = epsilon) +
      log_W2[l[2]]

  } else {

    log_density <- -Inf

  }

  return(log_density)

}




# Evaluates a single entry of the transition matrix
evaluate_transition_matrix_component <- function(
  path_out, # output reference path, i.e., $\mathbf{x}_{1:2}'$
  path_in, # input reference path, i.e., $\mathbf{x}_{1:2}$
  n_particles,
  resampling_scheme,
  epsilon,
  delta
) {

  # The space of possible values for the vector of all ancestor indices
  # at a particular time step, stored as a list. For instance, if
  # n_particles = 2, then the below command is equivalent to
  # space_indices <- list(c(1, 1), c(2, 2), c(1, 2), c(2, 1))
  space_indices <- gtools::permutations(
    n = n_particles, r = n_particles, v = 1:n_particles, repeats.allowed = TRUE
  ) %>%
    asplit(MARGIN = 1)

  # The space of possible values for the vector of all particles
  # at a particular time step, stored as a list. For instance, if
  # n_particles = 2, then the below command is equivalent to
  # space_particles <- list(c(0, 0), c(1, 1), c(0, 1), c(1, 0))
  space_particles <- gtools::permutations(
    n = 2, r = n_particles, v = 0:1, repeats.allowed = TRUE
  ) %>%
    asplit(MARGIN = 1)

  transition_density <- 0

  for (k in space_indices) {
    for (l in space_indices) {
      for (a in space_indices) {
        for (particles1 in space_particles) {
          for (particles2 in space_particles) {
            transition_density <- transition_density +
              exp(
                evaluate_log_extended_target_density_csmc(
                  path_out = path_out,
                  path_in = path_in,
                  particles1 = particles1,
                  particles2 = particles2,
                  a = a,
                  k = k,
                  l = l,
                  resampling_scheme = resampling_scheme,
                  epsilon = epsilon,
                  delta = delta
                )
              )
          }
        }
      }
    }
  }
  
  return(transition_density)
}





# Evaluates the transition matrix.
# The order of the states is: c(0, 0), c(1, 1), c(0, 1), c(1, 0),
# where each vector represents the states at time 1 and time 2
evaluate_transition_matrix <- function(
  n_particles,
  resampling_scheme,
  epsilon,
  delta
) {

  # The transition matrix;
  transition_matrix <- matrix(NA, nrow = 4, ncol = 4)

  for (i in seq_along(space_paths)) {
    for (j in seq_along(space_paths)) {
      transition_matrix[i, j] <- evaluate_transition_matrix_component(
        path_out = space_paths[[j]],
        path_in = space_paths[[i]],
        n_particles = n_particles,
        resampling_scheme = resampling_scheme,
        epsilon = epsilon,
        delta = delta
      )
    }
  }

  invariant_density <- eigen(t(transition_matrix))$vectors[, 1] %>% abs()
  invariant_density <- invariant_density / sum(invariant_density)


  return(
    list(
      transition_matrix = transition_matrix,
      invariant_density = invariant_density
    )
  )
}


# We now evaluate the invariant distribution at all four possible states).
# The order of these states is: c(0, 0), c(1, 1), c(0, 1), c(1, 0),
# invariant_density_multinomial <- evaluate_transition_matrix(
#   n_particles = n_particles,
#   resampling_scheme = "multinomial",
#   epsilon = epsilon,
#   delta = delta
# )$invariant_density
#
# invariant_density_systematic <- evaluate_transition_matrix(
#   n_particles = n_particles,
#   resampling_scheme = "systematic",
#   epsilon = epsilon,
#   delta = delta
# )$invariant_density
#
# invariant_density_residual_multinomial <- evaluate_transition_matrix(
#   n_particles = n_particles,
#   resampling_scheme = "residual-multinomial",
#   epsilon = epsilon,
#   delta = delta
# )$invariant_density
#
# invariant_density_naive_systematic <- evaluate_transition_matrix(
#   n_particles = n_particles,
#   resampling_scheme = "naive systematic",
#   epsilon = epsilon,
#   delta = delta
# )$invariant_density
#
#
# invariant_density_naive_residual_multinomial <- evaluate_transition_matrix(
#   n_particles = n_particles,
#   resampling_scheme = "naive residual-multinomial",
#   epsilon = epsilon,
#   delta = delta
# )$invariant_density
# 
# 
# 
# print(invariant_density)
# print(invariant_density_multinomial)
# print(invariant_density_systematic)
# print(invariant_density_residual_multinomial)
# print(invariant_density_naive_systematic)
# print(invariant_density_naive_residual_multinomial)


# Computes the Kullback--Leibler divergence from
# one distribution $p$ to another distribution $q$.
# It is assumed these are defined on a finite space,
# i.e., that they are represented as finite vectors of
# probabilities.
compute_kl_divergence <- function(p, q) {
  return(sum(p * (log(p) - log(q))))
}

N_PARTICLES <- c(2, 3)
EPSILON <- seq(from = 0.01, to = 0.99, length = 10)
DELTA   <- seq(from = 0.01, to = 0.99, length = 10)
RESAMPLING_SCHEME <- c("multinomial", "systematic", "residual-multinomial", "naive systematic", "naive residual-multinomial")

results_tbl <- tibble()

for (epsilon in EPSILON) {
  for (delta in DELTA) {

    # Calculate exact target density
    p <- rep(NA, times = length(space_paths))
    for (i in seq_along(space_paths)) {
      p[i] <- evaluate_log_target_density(space_paths[[i]], epsilon = epsilon, delta = delta) %>% exp()
    }

    for (n_particles in N_PARTICLES) {
      for (resampling_scheme in RESAMPLING_SCHEME) {

        q <- invariant_density_naive_residual_multinomial <- evaluate_transition_matrix(
          n_particles = n_particles,
          resampling_scheme = resampling_scheme,
          epsilon = epsilon,
          delta = delta
        )$invariant_density

        results_tbl %<>% bind_rows(
          tibble(
            resampling_scheme = resampling_scheme,
            n_particles = n_particles,
            epsilon = epsilon,
            delta = delta,
            kl_divergence = compute_kl_divergence(p = p, q = q)
          ) %>% mutate(resampling_scheme = factor(resampling_scheme, levels = RESAMPLING_SCHEME)) # conversion to factors for convenience to ensure the panels are plotted in the desired order later
        )
      }
    }
  }
}

results_tbl %>% write_rds("/home/axel/Dropbox/Apps/Overleaf/conditional_resampling_in_smc_algorithms/r_code/data/2024-04-06_results_tbl.Rds")

n_particles <- 2

results_tbl %>%
  filter(n_particles == n_particles) %>%
  ggplot(mapping = aes(x = delta, y = epsilon, fill = kl_divergence)) +
  geom_tile() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  facet_wrap(~resampling_scheme) +
  labs(
    x = "Delta",
    y = "Epsilon",
    fill = "KL(pi || pi_CSMC)",
    title = paste0("Different conditional resampling schemes\nin the toy binary state-space model\nwith N = ", n_particles, " particles; pi is the target;\npi_CSMC is the invariant distribution of the algorithm")
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold")
  )


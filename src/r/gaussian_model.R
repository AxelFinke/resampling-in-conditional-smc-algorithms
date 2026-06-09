lambda <- 0.8 # for lambda = 0.8, we obtain (up to rescaling) the same experiment as in Kviman et al. (2024, Section 5.1)

# Evaluates the log-potential function
evaluate_log_potential_function <- function(x) {- lambda * x^2 / 2}
# Evaluates the target density:
evaluate_target_density <- function(x) {dnorm(x)}
# Samples from the proposal distribution.
sample_from_proposal <- function(n = 1) {rnorm(n = n, sd = 1 / sqrt(1 - lambda))}


my_const <- 1 / 2 # either 1/2 or 1
# Evaluates the mean of the number of offspring obtained from a single particle.
evaluate_moment <- function(beta, k = 1) {

  compute_boundary <- function(ii) { sqrt(-2 / lambda * (log(beta) + log(ii +  my_const) + 1))}
  II <- floor(exp(-log(beta) - 1) - my_const)
  F <- pnorm(compute_boundary(1:II), sd = 1 / sqrt(1 - lambda))
  val <- 2 * sum((1:II)^k * (F - c(F[-1], 1 / 2)))
  return(val)
}


# Evaluates the theoretical lower bound on the probability of no particle with
# log-potential lower than log(beta) surviving the resampling procedure.
evaluate_lower_bound <- function(N, M, alpha = 1, beta) {
  first_moment <- evaluate_moment(beta, 1)
  second_moment <- evaluate_moment(beta, 2)
  if (is.finite(first_moment) && is.finite(second_moment) && (first_moment > alpha * M / N)) {
    theta <- alpha * M / (N * first_moment)
    aux <- (1 - theta)^2 * first_moment^2
#     return((1 - theta)^2 * first_moment^2 / second_moment) # the standard Paley--Zygmund bound (rather loose)
    return(aux / ((second_moment - first_moment^2) / N + aux)) # the sharpended (via the Cauchy--Schwarz inequality) Paley--Zygmund bound
  } else {
    return(NA)
  }
}

# Evaluates the theoretical lower bound on the probability of no particle with
# log-potential lower than log(beta) surviving the resampling procedure.
evaluate_lower_bound_vectorised <- function(N, M, first_moments, second_moments, alpha = 1) {
  aux <- (1 - alpha * M / (N * first_moments))^2 * first_moments^2
  val <- aux / ((second_moments - first_moments^2) / N + aux)
  val[!is.finite(first_moments) | !is.finite(second_moments) | (first_moments <= alpha * M / N)] <- NA
  return(val)
}

#ifndef __RESAMPLING_H
#define __RESAMPLING_H

#include <iostream>
#include "helper_functions.h"

/// Specifiers for various resampling algorithms.
enum Resample_type
{
  RESAMPLE_MULTINOMIAL = 0,
  RESAMPLE_STRATIFIED,
  RESAMPLE_SYSTEMATIC,
  RESAMPLE_VARIATIONAL,
  RESAMPLE_VARIATIONAL_ALT,
  RESAMPLE_MULTINOMIAL_TRUNCATED,
  RESAMPLE_STRATIFIED_TRUNCATED,
  RESAMPLE_SYSTEMATIC_TRUNCATED
};
/// Converts a Resample_type object to a std::string.
std::string convert_resample_type_to_string(const Resample_type& resample_type) {
  switch(resample_type) {
    case RESAMPLE_MULTINOMIAL: return "multinomial";
    case RESAMPLE_STRATIFIED: return "stratified";
    case RESAMPLE_SYSTEMATIC: return "systematic";
    case RESAMPLE_VARIATIONAL: return "variational";
    case RESAMPLE_VARIATIONAL_ALT: return "variational_alt";
    case RESAMPLE_MULTINOMIAL_TRUNCATED: return "multinomial_truncated";
    case RESAMPLE_STRATIFIED_TRUNCATED: return "stratified_truncated";
    case RESAMPLE_SYSTEMATIC_TRUNCATED: return "systematic_truncated";
    default: return std::string();
  }
}
/// Converts a string to a Resample_type object.
Resample_type convert_string_to_resample_type(const std::string& resample_type) {
  if (resample_type == "multinomial") {
    return Resample_type::RESAMPLE_MULTINOMIAL;
  } else if (resample_type == "stratified") {
    return Resample_type::RESAMPLE_STRATIFIED;
  } else if (resample_type == "systematic") {
    return Resample_type::RESAMPLE_SYSTEMATIC;
  } else if (resample_type == "variational") {
    return Resample_type::RESAMPLE_VARIATIONAL;
  } else if (resample_type == "variational_alt") {
    return Resample_type::RESAMPLE_VARIATIONAL_ALT;
  } else if (resample_type == "multinomial_truncated") {
    return Resample_type::RESAMPLE_MULTINOMIAL_TRUNCATED;
  } else if (resample_type == "stratified_truncated") {
    return Resample_type::RESAMPLE_STRATIFIED_TRUNCATED;
  } else if (resample_type == "systematic_truncated") {
    return Resample_type::RESAMPLE_SYSTEMATIC_TRUNCATED;
  } else {
    std::cout << "WARNING: Resampling scheme is not implemented!" << std::endl;
    return Resample_type::RESAMPLE_MULTINOMIAL;
  }
}

/// Performs multinomial resampling.
void resample_multinomial(
  arma::uvec& parent_indices,
  arma::colvec& log_resampled_weights,
  const arma::colvec& normalised_weights,
  const unsigned int n_resampled_particles
) {

  parent_indices.set_size(n_resampled_particles);
  log_resampled_weights.set_size(n_resampled_particles);
  log_resampled_weights.fill(-std::log(n_resampled_particles));

  arma::colvec u;
  u.randu(n_resampled_particles + 1);
  u = - arma::cumsum(arma::log(u)); // samples from an exponential distribution
  arma::colvec T = u(arma::span(0, n_resampled_particles - 1)) / u(n_resampled_particles);

  // arma::colvec T = u.sort(); /// NOTE: rather than sorting, we can generate (M+1) exponential RVs and take the cumulative sum of the first M of these divided by the sum of all (M+1) of these! This gives O(max(N, M)) complexity! See, e.g., https://djalil.chafai.net/blog/2014/06/03/back-to-basics-order-statistics-of-exponential-distribution/

  arma::colvec Q = arma::cumsum(normalised_weights);
  unsigned int i = 0;
  unsigned int j = 0;

  while (j < n_resampled_particles)
  {
    if (T(j) <= Q(i))
    {
      parent_indices(j) = i;
      ++j;
    }
    else
    {
      ++i;
    }
  }
}
/// Performs stratified resampling.
void resample_stratified(
  arma::uvec& parent_indices,
  arma::colvec& log_resampled_weights,
  const arma::colvec& normalised_weights,
  const unsigned int n_resampled_particles
) {

  parent_indices.set_size(n_resampled_particles);
  log_resampled_weights.set_size(n_resampled_particles);
  log_resampled_weights.fill(-std::log(n_resampled_particles));

  arma::colvec u;
  u.randu(n_resampled_particles);
  arma::colvec T = (arma::linspace(0, n_resampled_particles - 1, n_resampled_particles) + u) / n_resampled_particles;
  arma::colvec Q = arma::cumsum(normalised_weights);
  unsigned int i = 0;
  unsigned int j = 0;

  while (j < n_resampled_particles)
  {
    if (T(j) <= Q(i))
    {
      parent_indices(j) = i;
      ++j;
    }
    else
    {
      ++i;
    }
  }
}
/// Performs systematic resampling.
void resample_systematic(
  arma::uvec& parent_indices,
  arma::colvec& log_resampled_weights,
  const arma::colvec& normalised_weights,
  const unsigned int n_resampled_particles
) {

  parent_indices.set_size(n_resampled_particles);
  log_resampled_weights.set_size(n_resampled_particles);
  log_resampled_weights.fill(-std::log(n_resampled_particles));

  arma::colvec T = (arma::linspace(0, n_resampled_particles - 1, n_resampled_particles) + arma::randu()) / n_resampled_particles;
  arma::colvec Q = arma::cumsum(normalised_weights);
  unsigned int i = 0;
  unsigned int j = 0;

  while (j < n_resampled_particles)
  {
    if (T(j) <= Q(i))
    {
      parent_indices(j) = i;
      ++j;
    }
    else
    {
      ++i;
    }
  }
}
/// Performs variational resampling.
/// NOTE: Depending on the complexity of armadillo's index_max() function,
/// this may only be an $O(N^2)$ implementation rather than the possible $O(N log(N))$
/// implementation via heap sort suggested in Kviman et al. (2024).
void resample_variational(
  arma::uvec& parent_indices,
  arma::colvec& log_resampled_weights,
  const arma::colvec& normalised_weights,
  const unsigned int n_resampled_particles,
  const bool is_weighted
) {

  unsigned int N = normalised_weights.size();
  parent_indices.set_size(n_resampled_particles);
  log_resampled_weights.set_size(n_resampled_particles);

  arma::uvec k(N); // holds the total number of offspring for each particle
  k.zeros();
  arma::colvec logC = arma::log(normalised_weights);
  unsigned int imax;

  for (unsigned int m = 0; m < n_resampled_particles; ++m) {
    imax = logC.index_max();
    parent_indices(m) = imax;
    k(imax) = k(imax) + 1;
    logC(imax) = k(imax) * std::log(k(imax)) - (k(imax) + 1) * std::log(k(imax) + 1) + std::log(normalised_weights(imax));
  }


  if (is_weighted) {

    double sum_w = 0;
    for (unsigned int n = 0; n < N; ++n) {
      if (k(n) > 0) {
        sum_w += normalised_weights(n);
      }
    }
    // The division by sum_w is needed to correct for the fact that particles with sufficiently small unnormalised weights cannot survive the resampling procedure here
    for (unsigned int m = 0; m < n_resampled_particles; ++m) {
      log_resampled_weights(m) = std::log(normalised_weights(parent_indices(m))) - std::log(sum_w * k(parent_indices(m)));
    }
  } else {
    log_resampled_weights.fill(-std::log(n_resampled_particles));
  }
}
/// Performs resampling.
void resample(
  arma::uvec& parent_indices,
  arma::colvec& log_resampled_weights,
  const arma::colvec& normalised_weights,
  const unsigned int n_resampled_particles, // number of resampled particles
  const Resample_type& resample_type
) {

  unsigned int N = normalised_weights.size();
  arma::colvec w_aux;

  if (resample_type == RESAMPLE_MULTINOMIAL_TRUNCATED ||
      resample_type == RESAMPLE_STRATIFIED_TRUNCATED ||
      resample_type == RESAMPLE_SYSTEMATIC_TRUNCATED ) {
    w_aux = normalised_weights;
    for (unsigned int n = 0; n < N; ++n) {
      if (normalised_weights(n) < 1.0 / N) {
        w_aux <- 0;
      }
    }
    w_aux = arma::exp(normalise_exp(arma::log(w_aux)));
  }

  if (resample_type == RESAMPLE_MULTINOMIAL) {
    resample_multinomial(parent_indices, log_resampled_weights, normalised_weights, n_resampled_particles);
  } else if (resample_type == RESAMPLE_STRATIFIED) {
    resample_stratified(parent_indices, log_resampled_weights, normalised_weights, n_resampled_particles);
  } else if (resample_type == RESAMPLE_SYSTEMATIC) {
    resample_systematic(parent_indices, log_resampled_weights, normalised_weights, n_resampled_particles);
  } else if (resample_type == RESAMPLE_VARIATIONAL) {
    resample_variational(parent_indices, log_resampled_weights, normalised_weights, n_resampled_particles, false);
  } else if (resample_type == RESAMPLE_VARIATIONAL_ALT) {
    resample_variational(parent_indices, log_resampled_weights, normalised_weights, n_resampled_particles, true);
  } else if (resample_type == RESAMPLE_MULTINOMIAL_TRUNCATED) {
    resample_multinomial(parent_indices, log_resampled_weights, w_aux, n_resampled_particles);
  } else if (resample_type == RESAMPLE_STRATIFIED_TRUNCATED) {
    resample_stratified(parent_indices, log_resampled_weights, w_aux, n_resampled_particles);
  } else if (resample_type == RESAMPLE_SYSTEMATIC_TRUNCATED) {
    resample_systematic(parent_indices, log_resampled_weights, w_aux, n_resampled_particles);
  }
}

#endif

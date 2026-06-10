#include <RcppArmadillo.h>

#include "State_space_models.h"
#include "simulation_study.h"

// Enable C++14 via this plugin
// [[Rcpp::plugins("cpp14")]]



////////////////////////////////////////////////////////////////////////////////
// Performs resampling for a number of different resampling schemes.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

Rcpp::List resample(
  const arma::colvec& normalised_weights, // original normalised weights
  const unsigned int n_resampled_particles, // number of resampled particles
  const std::string& str_resample_type // resampling scheme (e.g., "multinomial", "stratified", "systematic")
) {

  arma::uvec parent_indices;
  arma::colvec log_resampled_weights;

  resample(
    parent_indices,
    log_resampled_weights,
    normalised_weights,
    n_resampled_particles,
    convert_string_to_resample_type(str_resample_type)
  );

  return Rcpp::List::create(
    Rcpp::Named("parent_indices") = parent_indices + 1, // switch to 1-indexing for use in R
    Rcpp::Named("log_resampled_weights") = log_resampled_weights
  );
}

////////////////////////////////////////////////////////////////////////////////
// Performs conditional resampling for a number of different resampling schemes.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

Rcpp::List conditional_resample(
  const arma::colvec& normalised_weights, // original normalised weights
  const unsigned int n_resampled_particles, // number of resampled particles
  const unsigned int single_parent_index, // the single parent index upon which we condition
  const std::string& str_resample_type // resampling scheme ("multinomial", "stratified", "systematic")
) {

  arma::uvec parent_indices;
  arma::colvec log_resampled_weights;
  unsigned int particle_index;

  conditional_resample(
    parent_indices,
    particle_index,
    log_resampled_weights,
    normalised_weights,
    n_resampled_particles,
    single_parent_index,
    convert_string_to_resample_type(str_resample_type)
  );

  return Rcpp::List::create(
    Rcpp::Named("parent_indices") = parent_indices + 1, // switch to 1-indexing for use in R
    Rcpp::Named("log_resampled_weights") = log_resampled_weights,
    Rcpp::Named("particle_index") = particle_index
  );
}

////////////////////////////////////////////////////////////////////////////////
// Runs the simulation study for the linear-Gaussian state-space model.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List run_simulation_study_linear_gaussian_state_space_model_cpp(
  const unsigned int n_data_sets,
  const unsigned int n_replicates,
  const unsigned int n_observations,
  const unsigned int n_particles,
  const unsigned int n_backward_sampling_particles,
  const unsigned int n_benchmark_particles,
  const unsigned int n_benchmark_backward_sampling_particles,
  const double ess_resampling_threshold,
  const std::vector<bool>& use_smoothing_weights,
  const std::vector<std::string>& resample_types,
  const unsigned int rng_seed
) {

  std::mt19937 rng(rng_seed);

  Linear_gaussian_state_space_model model;
  model.set_n_observations(n_observations);
  Results results;

  run_simulation_study<Linear_gaussian_state_space_model>(
    results,
    model,
    n_data_sets,
    n_replicates,
    n_particles,
    n_backward_sampling_particles,
    n_benchmark_particles,
    n_benchmark_backward_sampling_particles,
    ess_resampling_threshold,
    use_smoothing_weights,
    resample_types,
    true, // likelihood, filtering and smoothing distributions are analytically tractable in this model
    true, // use simulated data sampled from the model
    true, // approximate the filtering distributions
    true // use ancestor tracing and backward sampling (note that this can be costly if n_particles is large)
  );

  return Rcpp::List::create( // simply convert this to a data frame using as_tibble() in R
    Rcpp::Named("data_set") = results.data_set_,
    Rcpp::Named("replicate") = results.replicate_,
    Rcpp::Named("time") = results.time_,
    Rcpp::Named("dimension") = results.dimension_,
    Rcpp::Named("use_smoothing_weights") = results.use_smoothing_weights_,
    Rcpp::Named("resample_type") = results.resample_type_,
    Rcpp::Named("metric") = results.metric_,
    Rcpp::Named("value") = results.value_
  );
}

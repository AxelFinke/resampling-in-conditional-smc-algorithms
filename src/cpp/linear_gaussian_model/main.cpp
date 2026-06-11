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
  Rcpp::List args = Rcpp::List()
) {

  /// (default) parameter for controlling the ratio of the
  /// resampled weights in chopthin resampling.
  double beta = args.containsElementNamed("beta") ? (double)args["beta"] : 5.828427;

  arma::uvec parent_indices;
  arma::colvec log_resampled_weights;

  resample(
    parent_indices,
    log_resampled_weights,
    normalised_weights,
    n_resampled_particles,
    convert_string_to_resample_type(str_resample_type),
    beta
  );

  /// (default) parameter for controlling the ratio of the
  /// resampled weights in chopthin resampling.
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
  const std::string& str_resample_type, // resampling scheme ("multinomial", "stratified", "systematic")
  Rcpp::List args = Rcpp::List()
) {

  double beta = args.containsElementNamed("beta") ? (double)args["beta"] : 5.828427;

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
    convert_string_to_resample_type(str_resample_type),
    beta
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
Rcpp::List run_simulation_study_linear_gaussian_state_space_model(
  const arma::mat& observations,
  const unsigned int n_particles,
  const unsigned int n_iterations,
  const unsigned int n_replicates,
  const std::vector<std::string>& resample_types,
  const std::vector<std::string>& path_selection_types,
  const arma::uvec& times_to_store,
  const arma::uvec& dimensions_to_store, // must be of the same length as times_to_store
  const double ess_resampling_threshold,
  const unsigned int rng_seed
) {

  std::mt19937 rng(rng_seed);

  Linear_gaussian_state_space_model model;
  Results results;

  run_simulation_study<Linear_gaussian_state_space_model>(
    results,
    model,
    n_particles,
    n_iterations,
    n_replicates,
    resample_types,
    path_selection_types,
    ess_resampling_threshold,
    beta,
  );

  return Rcpp::List::create( // simply convert this to a data frame using as_tibble() in R
    Rcpp::Named("time") = results.time_,
    Rcpp::Named("dimension") = results.dimension_,
    Rcpp::Named("iteration") = results.iteration_,
    Rcpp::Named("replicate") = results.replicate_,
    Rcpp::Named("resample_type") = results.resample_type_,
    Rcpp::Named("path_selection_type") = results.path_selection_type_,
    Rcpp::Named("value") = results.value_
  );
}

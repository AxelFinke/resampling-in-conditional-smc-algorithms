// Copyright (C) 2026 Axel Finke
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.


#include <RcppArmadillo.h>

#include "State_space_models.h"
#include "simulation_study.h"

// Enable C++17 via this plugin
// [[Rcpp::plugins("cpp17")]]

////////////////////////////////////////////////////////////////////////////////
// Performs resampling for a number of different resampling schemes.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

Rcpp::List resample(
  const arma::colvec& normalised_weights, // original normalised weights
  const unsigned int n_resampled_particles, // number of resampled particles
  const std::string& str_resample_type // resampling scheme (e.g., "multinomial", "stratified", "systematic", "residual_multinomial", ...)
) {

  arma::uvec parent_indices;
  arma::colvec log_resampled_weights;

  resampling::unconditional(
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
  const std::string& str_resample_type // resampling scheme ("multinomial", "stratified", "systematic", ...)
) {

  arma::uvec parent_indices;
  arma::colvec log_resampled_weights;
  unsigned int particle_index;

  resampling::conditional(
    parent_indices,
    particle_index,
    log_resampled_weights,
    normalised_weights,
    n_resampled_particles,
    single_parent_index - 1,
    convert_string_to_resample_type(str_resample_type)
  );

  return Rcpp::List::create(
    Rcpp::Named("parent_indices") = parent_indices + 1, // switch to 1-indexing for use in R
    Rcpp::Named("log_resampled_weights") = log_resampled_weights,
    Rcpp::Named("particle_index") = particle_index
  );
}


////////////////////////////////////////////////////////////////////////////////
// Runs the simulation study for standard ("unconditional") SMC
// the linear-Gaussian state-space model.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List evaluate_ground_truth_linear_gaussian_state_space_model(
  const arma::mat& observations
) {

  Linear_gaussian_state_space_model model;
  model.set_observations(observations);

  arma::mat means_filtering;
  arma::cube variances_filtering;
  arma::mat means_smoothing;
  arma::cube variances_smoothing;

 double log_likelihood = model.evaluate_log_likelihood_and_moments(
   means_filtering,
   variances_filtering,
   means_smoothing,
   variances_smoothing
  );

  return Rcpp::List::create(
    Rcpp::Named("log_likelihood") = log_likelihood,
    Rcpp::Named("means_filtering") = means_filtering,
    Rcpp::Named("variances_filtering") = variances_filtering,
    Rcpp::Named("means_smoothing") = means_smoothing,
    Rcpp::Named("variances_smoothing") = variances_smoothing
  );
}

////////////////////////////////////////////////////////////////////////////////
// Runs the simulation study for standard ("unconditional") SMC
// the linear-Gaussian state-space model.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List run_simulation_study_smc_linear_gaussian_state_space_model(
  const unsigned int n_data_sets,
  const unsigned int n_observations,
  const unsigned int n_replicates,
  const arma::uvec& n_particles_vec,
  const arma::colvec& ess_resampling_thresholds,
  const std::vector<std::string>& resample_types,
  const unsigned int n_benchmark_particles,
  const unsigned int rng_seed
) {

  std::mt19937 rng(rng_seed);

  Linear_gaussian_state_space_model model;
  model.set_n_observations(n_observations);
  Results_smc results;

  run_simulation_study_smc<Linear_gaussian_state_space_model>(
    results,
    model,
    n_data_sets,
    n_replicates,
    n_particles_vec,
    ess_resampling_thresholds,
    resample_types,
    n_benchmark_particles
  );

  return Rcpp::List::create( // simply convert this to a data frame using as_tibble() in R
    Rcpp::Named("data_set") = results.data_set_,
    Rcpp::Named("replicate") = results.replicate_,
    Rcpp::Named("n_particles") = results.n_particles_,
    Rcpp::Named("ess_resampling_threshold") = results.ess_resampling_threshold_,
    Rcpp::Named("resample_type") = results.resample_type_,
    Rcpp::Named("relative_likelihood_estimate") = results.relative_likelihood_estimate_
  );
}


////////////////////////////////////////////////////////////////////////////////
// Runs the simulation study for conditional SMC
// the linear-Gaussian state-space model.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List run_simulation_study_csmc_linear_gaussian_state_space_model(
  const arma::mat& observations,
  const unsigned int n_iterations,
  const unsigned int n_replicates,
  const arma::uvec& n_particles_vec,
  const arma::colvec& ess_resampling_thresholds,
  const std::vector<std::string>& resample_types,
  const std::vector<std::string>& path_selection_types,
  const arma::uvec& times_to_store,
  const arma::uvec& dimensions_to_store, // must be of the same length as times_to_store
  const unsigned int rng_seed
) {

  std::mt19937 rng(rng_seed);

  Linear_gaussian_state_space_model model;
  model.set_observations(observations);
  Results_csmc results;

  run_simulation_study_csmc<Linear_gaussian_state_space_model>(
    results,
    model,
    n_iterations,
    n_replicates,
    n_particles_vec,
    ess_resampling_thresholds,
    resample_types,
    path_selection_types,
    times_to_store,
    dimensions_to_store
  );

  return Rcpp::List::create( // simply convert this to a data frame using as_tibble() in R
    Rcpp::Named("time") = results.time_,
    Rcpp::Named("dimension") = results.dimension_,
    Rcpp::Named("iteration") = results.iteration_,
    Rcpp::Named("replicate") = results.replicate_,
    Rcpp::Named("n_particles") = results.n_particles_,
    Rcpp::Named("ess_resampling_threshold") = results.ess_resampling_threshold_,
    Rcpp::Named("resample_type") = results.resample_type_,
    Rcpp::Named("path_selection_type") = results.path_selection_type_,
    Rcpp::Named("state") = results.state_
  );
}




////////////////////////////////////////////////////////////////////////////////
// Runs the simulation study for conditional SMC
// the linear-Gaussian state-space model.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List run_simulation_study_csmc_acf_linear_gaussian_state_space_model(
  const unsigned int n_data_sets,
  const unsigned int n_observations,
  const unsigned int n_iterations,
  const unsigned int n_replicates,
  const arma::uvec& n_particles_vec,
  const arma::colvec& ess_resampling_thresholds,
  const std::vector<std::string>& resample_types,
  const std::vector<std::string>& path_selection_types,
  const arma::uvec& times_to_store,
  const arma::uvec& dimensions_to_store, // must be of the same length as times_to_store
  const unsigned int lag_max,
  const unsigned int burnin,
  const unsigned int rng_seed
) {

  std::mt19937 rng(rng_seed);

  Linear_gaussian_state_space_model model;
  model.set_n_observations(n_observations);
  Results_csmc_acf results;

  run_simulation_study_csmc_acf<Linear_gaussian_state_space_model>(
    results,
    model,
    n_data_sets,
    n_iterations,
    n_replicates,
    n_particles_vec,
    ess_resampling_thresholds,
    resample_types,
    path_selection_types,
    times_to_store,
    dimensions_to_store,
    lag_max,
    burnin
  );


  return Rcpp::List::create( // simply convert this to a data frame using as_tibble() in R
    Rcpp::Named("data_set") = results.data_set_,
    Rcpp::Named("time") = results.time_,
    Rcpp::Named("dimension") = results.dimension_,
    Rcpp::Named("lag") = results.lag_,
    Rcpp::Named("replicate") = results.replicate_,
    Rcpp::Named("n_particles") = results.n_particles_,
    Rcpp::Named("ess_resampling_threshold") = results.ess_resampling_threshold_,
    Rcpp::Named("resample_type") = results.resample_type_,
    Rcpp::Named("path_selection_type") = results.path_selection_type_,
    Rcpp::Named("acf") = results.acf_
  );
}


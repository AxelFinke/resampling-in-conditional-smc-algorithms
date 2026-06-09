#include <RcppArmadillo.h>

#include "State_space_models.h"
#include "simulation_study.h"

// Enable C++14 via this plugin
// [[Rcpp::plugins("cpp14")]]

/// Evaluates the first or second moment of the number of
/// offspring obtained from a single particle.
double evaluate_moment(
  const double beta,
  const unsigned int k,
  const double lambda,
  const double sd
) {

  double my_const = 1.0;
  double I_aux = std::exp(-std::log(beta) - 1) - my_const;
  unsigned int I;

  if (I_aux >= 1) {
    I = std::floor(I_aux);
    arma::colvec F(I);
    for (unsigned int i = 0; i < I; ++i) {
      F(i) = R::pnorm(
        std::sqrt(-2.0 / lambda * (std::log(beta) + std::log(i +  my_const + 1) + 1)),
        0,
        sd,
        true, false
      );
    }

    double val = 0;
    if (I > 1) {
      for (unsigned int i = 0; i < I - 1; ++i) {
        val += std::pow(i + 1, k) * (F(i) - F(i + 1));
      }
    }
    val += std::pow(I, k) * (F(I - 1) - 0.5);

    return 2 * val;

  } else {
    return - std::numeric_limits<double>::infinity();
  }
}


////////////////////////////////////////////////////////////////////////////////
// Evaluates the first or second moments of the number of
// offspring obtained from a single particle for different
// bounds $\beta$.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
arma::colvec evaluate_moment_cpp(
  const arma::colvec& beta,
  const unsigned int k,
  const double lambda
) {

  double sd = 1 / std::sqrt(1 - lambda);
  arma::colvec moments(beta.size());
  for (unsigned int j = 0; j < beta.size(); ++j) {
    moments(j) = evaluate_moment(beta(j), k, lambda, sd);
  }
  return moments;
}

////////////////////////////////////////////////////////////////////////////////
// Performs multinomial, stratified, systematic, variational (original) or
// variational (bias reduced) resampling.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]

Rcpp::List resample_cpp(
  const arma::colvec& normalised_weights, // normalised weights
  const unsigned int n_resampled_particles, // number of resampled particles
  const std::string& str_resample_type // resampling scheme ("multinomial", "stratified", "systematic", "variational" or "variational_alt"
) {

  arma::uvec parent_indices;
  arma::colvec log_resampled_weights;

  resample(
    parent_indices,
    log_resampled_weights,
    normalised_weights, // the original normalised weights
    n_resampled_particles,
    convert_string_to_resample_type(str_resample_type)
  );

  return Rcpp::List::create(
    Rcpp::Named("parent_indices") = parent_indices + 1, // switch to 1-indexing for use in R
    Rcpp::Named("resampled_weights") = arma::exp(log_resampled_weights)
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
////////////////////////////////////////////////////////////////////////////////
// Runs the simulation study for the stochastic volatility model.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List run_simulation_study_stochastic_volatility_model_cpp(
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

  Stochastic_volatility_model model;
  model.set_n_observations(n_observations);
  Results results;

  run_simulation_study<Stochastic_volatility_model>(
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
    false, // likelihood, filtering and smoothing distributions are not analytically tractable in this model
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
////////////////////////////////////////////////////////////////////////////////
// Runs the simulation study for the nonlinear state-space model.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List run_simulation_study_nonlinear_state_space_model_cpp(
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

  Nonlinear_state_space_model model;
  model.set_n_observations(n_observations);
  Results results;

  run_simulation_study<Nonlinear_state_space_model>(
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
    false, // likelihood, filtering and smoothing distributions are not analytically tractable in this model
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
////////////////////////////////////////////////////////////////////////////////
// Runs the simulation study for the Lorenz-63 state-space model.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List run_simulation_study_lorenz_state_space_model_cpp(
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

  Lorenz_state_space_model model;
  model.set_n_observations(n_observations);
  Results results;

  run_simulation_study<Lorenz_state_space_model>(
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
    false, // likelihood, filtering and smoothing distributions are not analytically tractable in this model
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


////////////////////////////////////////////////////////////////////////////////
// Runs the simulation study for the stochastic volatility model.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List run_simulation_study_stochastic_volatility_model_real_data_cpp(
  const arma::mat& observations,
  const double phi,
  const double sigma,
  const double beta,
  const unsigned int n_replicates,
  const unsigned int n_particles,
  const unsigned int n_backward_sampling_particles,
  const unsigned int n_benchmark_particles,
  const unsigned int n_benchmark_backward_sampling_particles,
  const double ess_resampling_threshold,
  const std::vector<bool>& use_smoothing_weights,
  const std::vector<std::string>& resample_types,
  const bool approximate_filtering_distributions, // should the filtering distributions be approximated?
  const bool approximate_smoothing_distributions, // if false, then no ancestor tracing/backward sampling is performed
  const unsigned int rng_seed
) {

  std::mt19937 rng(rng_seed);

  Stochastic_volatility_model model;
  model.set_model_parameters(phi, sigma, beta);
  model.set_observations(observations);
  Results results;

  run_simulation_study<Stochastic_volatility_model>(
    results,
    model,
    1, // n_data_sets,
    n_replicates,
    n_particles,
    n_backward_sampling_particles,
    n_benchmark_particles,
    n_benchmark_backward_sampling_particles,
    ess_resampling_threshold,
    use_smoothing_weights,
    resample_types,
    false, // likelihood, filtering and smoothing distributions are not analytically tractable in this model
    false, // do not use simulated data sampled from the model
    approximate_filtering_distributions,
    approximate_smoothing_distributions
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



#ifndef __PARTICLE_FILTER_H
#define __PARTICLE_FILTER_H

#include "resampling.h"

/// Specifiers for various path-selection algorithms.
enum class Path_selection_type {
  ANCESTOR_TRACING = 0,
  BACKWARD_SAMPLING,
  ANCESTOR_SAMPLING
};
/// Converts a Path_selection_type object to a std::string.
std::string convert_path_selection_type_to_string(const Path_selection_type& path_selection_type) {
  switch(path_selection_type) {
    case ANCESTOR_TRACING: return "ancestor_tracing";
    case BACKWARD_SAMPLING: return "backward_sampling";
    case ANCESTOR_SAMPLING: return "ancestor_sampling";
    default: return std::string();
  }
}
/// Converts a string to a Path_selection_type object.
Path_selection_type convert_string_to_resample_type(const std::string& path_selection_type) {
  if (path_selection_type == "multinomial") {
    return Path_selection_type::ANCESTOR_TRACING;
  } else if (path_selection_type == "stratified") {
    return Path_selection_type::BACKWARD_SAMPLING;
  } else if (path_selection_type == "systematic") {
    return Path_selection_type::ANCESTOR_SAMPLING;
  } else {
    std::cout << "WARNING: Path selection type is not implemented!" << std::endl;
    return Path_selection_type::ANCESTOR_TRACING;
  }
}

/// Particle filter class.
template <class Model> class Particle_filter
{

public:

  /// Constructor.
  Particle_filter(
    const Model& model
  ) :
    model_(model)
  {
    ess_resampling_threshold_ = 1.0; // by default, the particle filter resamples at every step
    resample_type_ = RESAMPLE_STRATIFIED; // by default, the particle filter uses stratified resampling
    path_selection_type_ = ANCESTOR_TRACING; // by default, the conditional particle filter uses ancestor tracing
  }
  /// Specifies the number of particles.
  void set_n_particles(const unsigned int n_particles) {n_particles_ = n_particles;}
  /// Specifies the effective-sample-size based resampling threshold.
  void set_ess_resampling_threshold(const double ess_resampling_threshold) {ess_resampling_threshold_ = ess_resampling_threshold;}
  /// Specifies the resampling scheme.
  void set_resample_type(const Resample_type& resample_type) {resample_type_ = resample_type;}
  /// Specifies the path selection method.
  void set_path_selection_type(const Path_selection_type& path_selection_type) {path_selection_type_ = path_selection_type;}
  /// Runs the particle filter and returns the log-marginal likelihood estimate.
  double run_particle_filter();
  /// Runs the conditional particle filter (with backward sampling or ancestor tracing);
  std::vector<arma::colvec> run_conditional_particle_filter(
    const std::vector<arma::colvec>& x // the current reference path
  );


private:

  const Model& model_;
  unsigned int n_particles_;
  double ess_resampling_threshold_;
  Resample_type resample_type_;
  path_selection_type path_selection_type_;

  std::vector<std::vector<arma::colvec>> particles_;
  arma::mat self_normalised_weights_;
  arma::umat parent_indices_;

  /// Performs ancestor tracing to return one particle trajectory;
  std::vector<arma::colvec> run_ancestor_tracing() const;
  /// Performs backward sampling to return one particle trajectory;
  std::vector<arma::colvec> run_backward_sampling() const;
  /// Samples from the backward kernel at time $t$ (given a state from time $t + 1$).
  unsigned int sample_from_backward_kernel(const unsigned int t, const arma::colvec& x) const {
    arma::colvec log_w_aux(n_particles_);
    for (unsigned int n = 0; n < n_particles_; ++n) {
      log_w_aux(n) = std::log(self_normalised_weights_(n, t)) +
        model_.evaluate_log_transition_density(t + 1, x, particles_[t][n]);
    }
    return sample_int(arma::exp(normalise_exp(log_w_aux)));
  }

};

/// Runs the particle filter.
template <class Model>
double Particle_filter<Model>::run_particle_filter() {

  // std::cout << "PF start" << std::endl;
  unsigned int T = model_.get_n_observations();
  unsigned int N = n_particles_;
  double log_likelihood_estimate = 0;

  arma::colvec log_observation_densities(N);
  arma::colvec log_unnormalised_weights(N);
  arma::uvec parent_indices(N);

  parent_indices_.set_size(N, T - 1);
  self_normalised_weights_.set_size(N, T);

  // std::cout << "PF: resize particles_" << std::endl;
  particles_.resize(T);
  for (unsigned int t = 0; t < T; ++t) {
    particles_[t].resize(N);
  }

  // std::cout << "PF: start: sample particles" << std::endl;
  for (unsigned int n = 0; n < N; ++n) {
    particles_[0][n] = model_.sample_from_initial_distribution();
    log_observation_densities(n) = model_.evaluate_log_observation_density(0, particles_[0][n]);
    log_unnormalised_weights(n) = log_observation_densities(n) - std::log(N);
  }
  self_normalised_weights_.col(0) = arma::exp(normalise_exp(log_likelihood_estimate, log_unnormalised_weights));

  for (unsigned int t = 1; t < T; ++t) {

    double ess = compute_effective_sample_size(self_normalised_weights_.col(t - 1));

    if (ess < ess_resampling_threshold_ * N) {
      resample(parent_indices, log_unnormalised_weights, self_normalised_weights_.col(t - 1), N, resample_type_);
    } else {
      parent_indices = arma::linspace<arma::uvec>(0, N - 1, N);
      log_unnormalised_weights = arma::log(self_normalised_weights_.col(t - 1));
    }
    parent_indices_.col(t - 1) = parent_indices;

    for (unsigned int n = 0; n < N; ++n) {

      particles_[t][n] = model_.sample_from_transition_equation(t, particles_[t - 1][parent_indices_(n, t - 1)]);
      log_observation_densities(n) = model_.evaluate_log_observation_density(t, particles_[t][n]);
      log_unnormalised_weights(n) = log_unnormalised_weights(n) + log_observation_densities(n);
    }
    self_normalised_weights_.col(t) = arma::exp(normalise_exp(log_likelihood_estimate, log_unnormalised_weights));
  }

  return log_likelihood_estimate;
}

/// Runs the conditional particle filter.
template <class Model>
std::vector<arma::colvec>& Particle_filter<Model>::run_conditional_particle_filter(
  const std::vector<arma::colvec>& x // the current reference path
) {

  // std::cout << "PF start" << std::endl;
  unsigned int T = model_.get_n_observations();
  unsigned int N = n_particles_;

  arma::colvec log_observation_densities(N);
  arma::colvec log_unnormalised_weights(N);
  arma::uvec parent_indices(N);

  parent_indices_.set_size(N, T - 1);
  self_normalised_weights_.set_size(N, T);

  // std::cout << "PF: resize particles_" << std::endl;
  particles_.resize(T);
  for (unsigned int t = 0; t < T; ++t) {
    particles_[t].resize(N);
  }

  std::vector<unsigned int> particle_indices(T); // indices of the reference path
  particle_indices[0] = arma::randi<int>(arma::distr_param(0, N - 1));

  // std::cout << "PF: start: sample particles" << std::endl;
  for (unsigned int n = 0; n < N; ++n) {
    if (n == particle_indices[0]) {
      particles_[0][n] = x[0];
    } else {
      particles_[0][n] = model_.sample_from_initial_distribution();
    }
    log_observation_densities(n) = model_.evaluate_log_observation_density(0, particles_[0][n]);
    log_unnormalised_weights(n) = log_observation_densities(n) - std::log(N);
  }
  self_normalised_weights_.col(0) = arma::exp(normalise_exp(log_unnormalised_weights));

  for (unsigned int t = 1; t < T; ++t) {

    double ess = compute_effective_sample_size(self_normalised_weights_.col(t - 1));

    if (ess < ess_resampling_threshold_ * N) {
      unsigned int k;
      unsigned int a;

      if (path_selection_type_ == ANCESTOR_SAMPLING) {
        a = particle_indices[t - 1];
      } else {
        a = sample_from_backward_kernel(t - 1, x[t]);
      }

      conditional_resample(parent_indices, k, log_unnormalised_weights, self_normalised_weights_.col(t - 1), N, a, resample_type_);
      particle_indices[t] = k;
    } else {
      particle_indices[t] = particle_indices[t - 1];
      parent_indices = arma::linspace<arma::uvec>(0, N - 1, N);
      log_unnormalised_weights = arma::log(self_normalised_weights_.col(t - 1));
    }
    parent_indices_.col(t - 1) = parent_indices;

    for (unsigned int n = 0; n < N; ++n) {

      if (n == particle_indices[t]) {
        particles_[t][n] = x[t];
      } else {
        particles_[t][n] = model_.sample_from_transition_equation(t, particles_[t - 1][parent_indices_(n, t - 1)]);
      }
      log_observation_densities(n) = model_.evaluate_log_observation_density(t, particles_[t][n]);
      log_unnormalised_weights(n) = log_unnormalised_weights(n) + log_observation_densities(n);
    }
    self_normalised_weights_.col(t) = arma::exp(normalise_exp(log_unnormalised_weights));
  }

  if (path_selection_type_ == BACKWARD_SAMPLING) {
    return run_backward_sampling();
  } else {
    return run_ancestor_tracing();
  }
}

/// Generates a trajectory via backward sampling.
template <class Model>
std::vector<arma::colvec> Particle_filter<Model>::run_backward_sampling() const {

  unsigned int T = model_.get_n_observations();
  // unsigned int N = n_particles_;
  unsigned int b;
  // arma::colvec log_w_aux(N);

  std::vector<arma::colvec> x(T);
  b = sample_int(self_normalised_weights_.col(T - 1));
  x[T - 1] = particles_[T - 1][b];

  for (unsigned int t = T - 2; t != static_cast<unsigned>(-1); t--) {
    b = sample_from_backward_kernel(t, x[t + 1]);
    // for (unsigned int n = 0; n < N; ++n) {
    //   log_w_aux(n) = std::log(self_normalised_weights_(n, t)) +
    //     model_.evaluate_log_transition_density(t + 1, x[t + 1], particles_[t][n]);
    // }
    // b = sample_int(arma::exp(normalise_exp(log_w_aux)));
    x[t] = particles_[t][b];
  }
  return x;
}
/// Generates a trajectory via ancestor tracing.
template <class Model>
std::vector<arma::colvec> Particle_filter<Model>::run_ancestor_tracing() const {

  unsigned int T = model_.get_n_observations();
  unsigned int N = n_particles_;
  unsigned int b;

  std::vector<arma::colvec> x(T);
  b = sample_int(self_normalised_weights_.col(T - 1));
  x[T - 1] = particles_[T - 1][b];
  for (unsigned int t = T - 2; t != static_cast<unsigned>(-1); t--) {
    b = parent_indices_(b, t);
    x[t] = particles_[t][b];
  }
  return x;
}


#endif

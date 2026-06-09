
#ifndef __PARTICLE_FILTER_H
#define __PARTICLE_FILTER_H

#include "resampling.h"

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
    use_smoothing_weights_ = false;
  }
  /// Returns the estimate of the log-likelihood.
  double get_log_likelihood_estimate() const {return log_likelihood_estimate_;}
  /// Specifies the number of particles.
  void set_n_particles(const unsigned int n_particles) {n_particles_ = n_particles;}
  /// Specifies the effective-sample-size based resampling threshold.
  void set_ess_resampling_threshold(const double ess_resampling_threshold) {ess_resampling_threshold_ = ess_resampling_threshold;}
  /// Specifies the resampling scheme.
  void set_resample_type(const Resample_type& resample_type) {resample_type_ = resample_type;}
  /// Specifies whether the "smoothing weights" from Kviman et al. (2024)
  /// should be used for resampling. Note that these induce bias.
  void set_use_smoothing_weights(const bool use_smoothing_weights) {use_smoothing_weights_ = use_smoothing_weights;}
  /// Runs the particle filter and returns the log-likelihood_estimate.
  double run_particle_filter(
    const unsigned int n_particles,
    const double ess_resampling_threshold,
    const Resample_type& resample_type
  ) {
    set_n_particles(n_particles);
    set_ess_resampling_threshold(ess_resampling_threshold);
    set_resample_type(resample_type);
    run_particle_filter_base();
    return get_log_likelihood_estimate();
  }
  /// Generates smoothing trajectories via ancestor tracing.
  void run_ancestor_tracing();
  /// Generates smoothing trajectories via backward sampling.
  void run_backward_sampling(const unsigned int n_backward_sampling_trajectories);
  /// Computes the mean-square error (averaged across dimensions) for
  /// the marginal filtering distribution at some time $t$.
  double compute_mse_filtering(const unsigned int t, const arma::colvec& ground_truth_mean) const;
  /// Computes the mean-square error (averaged across dimensions) for
  /// the marginal smoothing distribution at some time $t$.
  double compute_mse_smoothing(const unsigned int t, const arma::colvec& ground_truth_mean) const;
  /// Computes the squared Mahalanobis distance for
  /// the marginal filtering distribution at some time $t$.
  double compute_squared_mahalanobis_distance_filtering(const unsigned int t, const arma::colvec& ground_truth_mean, const arma::mat& ground_truth_variance) const;
  /// Computes the squared Mahalanobis distance for
  /// the marginal smoothing distribution at some time $t$.
  double compute_squared_mahalanobis_distance_smoothing(const unsigned int t, const arma::colvec& ground_truth_mean, const arma::mat& ground_truth_variance) const;
  /// Computes the estimates of the means and variances of the
  /// marginal filtering distributions at all time steps.
  void estimate_moments_filtering(arma::mat& means_filtering, arma::cube& variances_filtering) {
    unsigned int D = particles_[0][0].size();
    unsigned int T = particles_.size();
    means_filtering.set_size(D, T);
    variances_filtering.set_size(D, D, T);
    for (unsigned int t = 0; t < T; ++t) {
      means_filtering.col(t) = estimate_mean_filtering(t);
      variances_filtering.slice(t) = estimate_variance_filtering(t);
    }
  }
  /// Computes the estimates of the means and variances of the
  /// marginal smoothing distributions at all time steps.
  void estimate_moments_smoothing(arma::mat& means_smoothing, arma::cube& variances_smoothing) {
    unsigned int D = smoothing_trajectories_[0][0].size();
    unsigned int T = smoothing_trajectories_.size();
    means_smoothing.set_size(D, T);
    variances_smoothing.set_size(D, D, T);
    for (unsigned int t = 0; t < T; ++t) {
      means_smoothing.col(t) = estimate_mean_smoothing(t);
      variances_smoothing.slice(t) = estimate_variance_smoothing(t);
    }
  }

private:

  const Model& model_;
  unsigned int n_particles_;
  double ess_resampling_threshold_;
  Resample_type resample_type_;

  bool use_smoothing_weights_; // this induces bias!

  std::vector<std::vector<arma::colvec>> particles_;
  std::vector<std::vector<arma::colvec>> smoothing_trajectories_;
  arma::colvec smoothing_trajectory_weights_;
  arma::mat self_normalised_weights_;
  arma::umat parent_indices_;
  arma::colvec ess_;

  double log_likelihood_estimate_;

  /// Runs the particle filter.
  void run_particle_filter_base();
  /// Estimates the mean of the marginal filtering distribution at time $t$.
  arma::colvec estimate_mean_filtering(const unsigned int t) const {
    unsigned int D = particles_[t][0].size();
    arma::colvec out(D);
    out.zeros();
    for (unsigned int n = 0; n < particles_[t].size(); ++n) {
      out = out + self_normalised_weights_(n, t) * particles_[t][n];
    }
    return(out);
  }
  /// Estimates the mean of the marginal smoothing distribution at time $t$.
  arma::colvec estimate_mean_smoothing(const unsigned int t) const {
    unsigned int D = smoothing_trajectories_[t][0].size();
    arma::colvec out(D);
    out.zeros();
    for (unsigned int m = 0; m < smoothing_trajectories_[t].size(); ++m) {
      out = out + smoothing_trajectory_weights_(m) * smoothing_trajectories_[t][m];
    }
    return(out);
  }
  /// Estimates the variance of the marginal filtering distribution at time $t$.
  arma::mat estimate_variance_filtering(const unsigned int t) const {
    unsigned int D = particles_[t][0].size();

    arma::colvec mean_filtering = estimate_mean_filtering(t);
    arma::colvec z;

    arma::mat out(D, D);
    out.zeros();

    for (unsigned int n = 0; n < particles_[t].size(); ++n) {
      z = particles_[t][n] - mean_filtering;
      out = out + self_normalised_weights_(n, t) * z * z.t();
    }
    return(out);
  }
  /// Estimates the variance of the marginal smoothing distribution at time $t$.
  arma::mat estimate_variance_smoothing(const unsigned int t) const {
    unsigned int D = smoothing_trajectories_[t][0].size();

    arma::colvec mean_smoothing = estimate_mean_smoothing(t);
    arma::colvec z;

    arma::mat out(D, D);
    out.zeros();

    for (unsigned int m = 0; m < smoothing_trajectories_[t].size(); ++m) {
      z = smoothing_trajectories_[t][m] - mean_smoothing;
      out = out + smoothing_trajectory_weights_(m) * z * z.t();
    }
    return(out);
  }
};

/// Runs the particle filter.
template <class Model>
void Particle_filter<Model>::run_particle_filter_base() {

  // std::cout << "PF start" << std::endl;
  unsigned int T = model_.get_n_observations();
  unsigned int N = n_particles_;
  log_likelihood_estimate_ = 0;

  arma::colvec log_observation_densities(N);
  arma::colvec log_unnormalised_weights(N);
  arma::uvec parent_indices(N);

  parent_indices_.set_size(N, T - 1);
  self_normalised_weights_.set_size(N, T);
  ess_.set_size(T);

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
  self_normalised_weights_.col(0) = arma::exp(normalise_exp(log_likelihood_estimate_, log_unnormalised_weights));

  arma::mat log_unnormalised_smoothing_weights;
  if (use_smoothing_weights_) {
    log_unnormalised_smoothing_weights.set_size(N, T);
    for (unsigned int n = 0; n < N; ++n) {
      log_unnormalised_smoothing_weights(n, 0) = model_.evaluate_log_initial_density(particles_[0][n]) + log_observation_densities(n);
    }
  }

  for (unsigned int t = 1; t < T; ++t) {

    ess_(t - 1) = compute_effective_sample_size(self_normalised_weights_.col(t - 1));

    if (ess_(t - 1) < ess_resampling_threshold_ * N) {
      if (use_smoothing_weights_) {
        resample(parent_indices, log_unnormalised_weights,  arma::exp(normalise_exp(log_unnormalised_smoothing_weights.col(t - 1))), N, resample_type_);
      } else {
        resample(parent_indices, log_unnormalised_weights, self_normalised_weights_.col(t - 1), N, resample_type_);
      }

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

    self_normalised_weights_.col(t) = arma::exp(normalise_exp(log_likelihood_estimate_, log_unnormalised_weights));

    if (use_smoothing_weights_) {
      for (unsigned int n = 0; n < N; ++n) {
        log_unnormalised_smoothing_weights(n, t) = log_unnormalised_smoothing_weights(parent_indices_(n, t - 1), t - 1) + model_.evaluate_log_transition_density(t, particles_[t][n], particles_[t - 1][parent_indices_(n, t - 1)]) + log_observation_densities(n);
      }
    }

  }

  ess_(T - 1) = compute_effective_sample_size(self_normalised_weights_.col(T - 1));

}
/// Generates smoothing trajectories via backward sampling.
template <class Model>
void Particle_filter<Model>::run_backward_sampling(const unsigned int n_backward_sampling_trajectories) {

  unsigned int T = model_.get_n_observations();
  unsigned int N = n_particles_;
  unsigned int M = n_backward_sampling_trajectories;
  unsigned int b;
  arma::colvec log_w_aux(N);

  smoothing_trajectories_.resize(T);
  for (unsigned int t = 0; t < T; ++t) {
    smoothing_trajectories_[t].resize(M);
  }
  for (unsigned int m = 0; m < M; ++m) {
    b = sample_int(self_normalised_weights_.col(T - 1));
    smoothing_trajectories_[T - 1][m] = particles_[T - 1][b];
    for (unsigned int t = T - 2; t != static_cast<unsigned>(-1); t--)
    {
      for (unsigned int n = 0; n < N; ++n)
      {
        log_w_aux(n) = std::log(self_normalised_weights_(n, t)) + model_.evaluate_log_transition_density(t + 1, smoothing_trajectories_[t + 1][m], particles_[t][n]);
      }
      b = sample_int(arma::exp(normalise_exp(log_w_aux)));

      smoothing_trajectories_[t][m] = particles_[t][b];
    }
  }
  smoothing_trajectory_weights_.set_size(n_backward_sampling_trajectories);
  smoothing_trajectory_weights_.fill(1.0 / n_backward_sampling_trajectories);

}
/// Generates smoothing trajectories via ancestor tracing.
template <class Model>
void Particle_filter<Model>::run_ancestor_tracing() {

  unsigned int T = model_.get_n_observations();
  unsigned int N = n_particles_;
  unsigned int b;

  smoothing_trajectories_.resize(T);
  for (unsigned int t = 0; t < T; ++t) {
    smoothing_trajectories_[t].resize(N);
  }
  for (unsigned int m = 0; m < N; ++m) {
    b = m;
    smoothing_trajectories_[T - 1][m] = particles_[T - 1][b];
    for (unsigned int t = T - 2; t != static_cast<unsigned>(-1); t--)
    {
      b = parent_indices_(b, t);
      smoothing_trajectories_[t][m] = particles_[t][b];
    }
  }
  smoothing_trajectory_weights_ = self_normalised_weights_.col(T - 1);
}
/// Computes the mean-square error (averaged across dimensions) for
/// the marginal filtering distribution at some time $t$.
template <class Model>
double Particle_filter<Model>::compute_mse_filtering(const unsigned int t, const arma::colvec& ground_truth_mean) const {
  double mse = 0;
  arma::colvec z;
  for (unsigned int n = 0; n < particles_[t].size(); ++n) {
    z = particles_[t][n] - ground_truth_mean;
    mse += self_normalised_weights_(n, t) * arma::dot(z, z);
  }
  return mse;
}
/// Computes the mean-square error (averaged across dimensions) for
/// the marginal smoothing distribution at some time $t$.
template <class Model>
double Particle_filter<Model>::compute_mse_smoothing(const unsigned int t, const arma::colvec& ground_truth_mean) const {
  double mse = 0;
  arma::colvec z;
  for (unsigned int m = 0; m < smoothing_trajectories_[t].size(); ++m) {
    z = smoothing_trajectories_[t][m] - ground_truth_mean;
    mse += smoothing_trajectory_weights_(m) * arma::dot(z, z);
  }
  return mse;
}
/// Computes the squared Mahalanobis distance for
/// the marginal filtering distribution at some time $t$.
template <class Model>
double Particle_filter<Model>::compute_squared_mahalanobis_distance_filtering(const unsigned int t, const arma::colvec& ground_truth_mean, const arma::mat& ground_truth_variance) const {
  double smd = 0;
  arma::colvec z;
  arma::mat rooti = arma::trans(arma::inv(arma::trimatl(arma::chol(ground_truth_variance, "lower"))));
  for (unsigned int n = 0; n < particles_[t].size(); ++n) {
    z = rooti * (particles_[t][n] - ground_truth_mean);
    smd += self_normalised_weights_(n, t) * arma::dot(z, z);
  }
  return smd;
}
/// Computes the squared Mahalanobis distance for
/// the marginal smoothing distribution at some time $t$.
template <class Model>
double Particle_filter<Model>::compute_squared_mahalanobis_distance_smoothing(const unsigned int t, const arma::colvec& ground_truth_mean, const arma::mat& ground_truth_variance) const {
  double smd = 0;
  arma::colvec z;
  arma::mat rooti = arma::trans(arma::inv(arma::trimatl(arma::chol(ground_truth_variance, "lower"))));
  for (unsigned int m = 0; m < smoothing_trajectories_[t].size(); ++m) {
    z = rooti * (smoothing_trajectories_[t][m] - ground_truth_mean);
    smd += smoothing_trajectory_weights_(m) * arma::dot(z, z);
  }
  return smd;
}

#endif

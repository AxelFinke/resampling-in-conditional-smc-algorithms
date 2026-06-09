
#ifndef __SIMULATION_STUDY_H
#define __SIMULATION_STUDY_H


#include "Particle_filter.h"


/// Class for holding the results of the simulation study
/// in such a way that they can be used as the columns of a
/// tidy data frame.
class Results {

public:

  std::vector<unsigned int> data_set_;
  std::vector<unsigned int> replicate_;
  std::vector<int> time_;
  std::vector<int> dimension_;
  std::vector<bool> use_smoothing_weights_;
  std::vector<std::string> resample_type_;
  std::vector<std::string> metric_;
  std::vector<double> value_;

  unsigned int data_set_current_;
  unsigned int replicate_current_;
  int time_current_;
  int dimension_current_;
  bool use_smoothing_weights_current_;
  std::string resample_type_current_;

  /// Initialises the vectors.
  void initialise(const unsigned int K) {
    data_set_.reserve(K);
    replicate_.reserve(K);
    time_.reserve(K);
    dimension_.reserve(K);
    use_smoothing_weights_.reserve(K);
    resample_type_.reserve(K);
    metric_.reserve(K);
    value_.reserve(K);
  }
  /// Stores a value.
  void store(const std::string& metric, const double value) {
    data_set_.push_back(data_set_current_);
    replicate_.push_back(replicate_current_);
    time_.push_back(time_current_);
    dimension_.push_back(dimension_current_);
    use_smoothing_weights_.push_back(use_smoothing_weights_current_);
    resample_type_.push_back(resample_type_current_);
    metric_.push_back(metric);
    value_.push_back(value);
  }

private:

};
/// Simulation study that runs a particle filter for a given state-space model
/// multiple times independently -- using a different data set sampled from the
/// model each time -- for different tuning parameters.
template <class Model> void run_simulation_study(
  Results& results,
  Model& model,
  const unsigned int n_data_sets,
  const unsigned int n_replicates,
  const unsigned int n_particles,
  const unsigned int n_backward_sampling_particles,
  const unsigned int n_benchmark_particles,
  const unsigned int n_benchmark_backward_sampling_particles,
  const double ess_resampling_threshold,
  const std::vector<bool>& use_smoothing_weights,
  const std::vector<std::string>& resample_types,
  const bool model_has_analytical_solution,
  const bool simulate_data,
  const bool approximate_filtering_distributions, // should the filtering distributions be approximated?
  const bool approximate_smoothing_distributions // if false, then we only approximate the filtering distributions
) {

  unsigned int T = model.get_n_observations();
  unsigned int D = model.get_state_dimension();
  unsigned int N = n_particles;
  unsigned int R = resample_types.size();
  unsigned int S = use_smoothing_weights.size();

  Particle_filter<Model> pf(model);

  if (approximate_filtering_distributions && approximate_smoothing_distributions) {
    // total number of values to be stored, i.e., for each simulation and resampling scheme,
    // we run the particle filter twice -- once with and once without the "smoothing" weights --
    // and each time we compute, the log-likelihood estimate as well as the MSE, MSE*
    // and Mahalanobis distance for both the marginal filtering and marginal smoothing
    // distributions at each time point, where MSE* refers to the mean square error relative
    // to the true state; and both MSE and MSE* refer to the average across all dimensions.
    results.initialise(n_replicates * n_data_sets * R * S * (1 + T * 9));
  } else if (approximate_filtering_distributions && !approximate_smoothing_distributions) {
    results.initialise(n_replicates * n_data_sets * R * S * (1 + T * 3));
  } else if (!approximate_filtering_distributions && approximate_smoothing_distributions) {
    results.initialise(n_replicates * n_data_sets * R * S * (1 + T * 6));
  } else if (!approximate_filtering_distributions && !approximate_smoothing_distributions) {
    results.initialise(n_replicates * n_data_sets * R * S);
  } else {
    std::cout << "ERROR in approximate_filtering_distributions or approximate_smoothing_distributions!" << std::endl;
  }
  results.dimension_current_ = -1;

  arma::mat means_filtering(D, T);
  arma::mat means_smoothing(D, T);
  arma::cube variances_filtering(D, D, T);
  arma::cube variances_smoothing(D, D, T);
  double log_likelihood;

  for (unsigned int i = 0; i < n_data_sets; ++i) {

    std::cout << "Progress in terms of data sets: " << static_cast<int>(static_cast<double>(i) / n_data_sets * 100) << " %" << std::endl;
    results.data_set_current_ = i;

    if (simulate_data) {
      model.simulate_data();
    }

    if (model_has_analytical_solution) {

      log_likelihood = model.evaluate_log_likelihood_and_moments(means_filtering, variances_filtering, means_smoothing, variances_smoothing);

    } else {
      pf.set_use_smoothing_weights(false);

      std::cout << "run_simulation_study: run reference PF" << std::endl;
      pf.run_particle_filter(n_benchmark_particles, 0.9, Resample_type::RESAMPLE_STRATIFIED); // Benchmark particle filter to obtain approximate ground truth
      log_likelihood = pf.get_log_likelihood_estimate();

      if (approximate_filtering_distributions) {
        std::cout << "run_simulation_study: estimate true filtering/variances" << std::endl;
        pf.estimate_moments_filtering(means_filtering, variances_filtering);
      }
      if (approximate_smoothing_distributions) {
        std::cout << "run_simulation_study: run reference PF's backward sampling" << std::endl;
        pf.run_backward_sampling(n_backward_sampling_particles);
        std::cout << "run_simulation_study: estimate true smoothing means/variances" << std::endl;
        pf.estimate_moments_smoothing(means_smoothing, variances_smoothing);
      }

    }

    for (unsigned int j = 0; j < n_replicates; ++ j) {

      std::cout << "Progress in terms of replicates for the current data set: " << static_cast<int>(static_cast<double>(j) / n_replicates * 100) << " %" << std::endl;
      results.replicate_current_ = j;

      std::cout << "Benchmark logZ: " << log_likelihood << std::endl;

      for (unsigned int r = 0; r < R; ++r) {

        results.resample_type_current_ = resample_types[r];

        for (unsigned int s = 0; s < S; ++s) { // use smoothing weights (1) or not (0)?

          results.use_smoothing_weights_current_ = use_smoothing_weights[s];
          pf.set_use_smoothing_weights(use_smoothing_weights[s]);
          pf.run_particle_filter(N, ess_resampling_threshold, convert_string_to_resample_type(resample_types[r]));

          results.time_current_ = -1;

          std::cout << "Estimated logZ for: " << resample_types[r] << ": " << pf.get_log_likelihood_estimate() << std::endl;
          results.store("relative_likelihood_estimation_error",  std::expm1(pf.get_log_likelihood_estimate() - log_likelihood));

          if (approximate_filtering_distributions) {
            for (unsigned int t = 0; t < T; ++t) {
              results.time_current_ = t;
              results.store("mse_filtering", pf.compute_mse_filtering(t, means_filtering.col(t)));
              results.store("mse_star_filtering", pf.compute_mse_filtering(t, model.get_state(t)));
              results.store("squared_mahalanobis_distance_filtering", pf.compute_squared_mahalanobis_distance_filtering(t, means_filtering.col(t), variances_filtering.slice(t)));
            }
          }

          if (approximate_smoothing_distributions) {
            pf.run_ancestor_tracing();

            for (unsigned int t = 0; t < T; ++t) {
              results.time_current_ = t;
              results.store("mse_smoothing_ancestor_tracing", pf.compute_mse_smoothing(t, means_smoothing.col(t)));
              results.store("mse_star_smoothing_ancestor_tracing", pf.compute_mse_smoothing(t, model.get_state(t)));
              results.store("squared_mahalanobis_distance_smoothing_ancestor_tracing", pf.compute_squared_mahalanobis_distance_smoothing(t, means_smoothing.col(t), variances_smoothing.slice(t)));
            }

            pf.run_backward_sampling(n_backward_sampling_particles);

            for (unsigned int t = 0; t < T; ++t) {
              results.time_current_ = t;
              results.store("mse_smoothing_backward_sampling", pf.compute_mse_smoothing(t, means_smoothing.col(t)));
              results.store("mse_star_smoothing_backward_sampling", pf.compute_mse_smoothing(t, model.get_state(t)));
              results.store("squared_mahalanobis_distance_smoothing_backward_sampling", pf.compute_squared_mahalanobis_distance_smoothing(t, means_smoothing.col(t), variances_smoothing.slice(t)));
            }
          }
        }
      }
    }
  }
}



#endif

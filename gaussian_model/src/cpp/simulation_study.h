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


#ifndef __SIMULATION_STUDY_H
#define __SIMULATION_STUDY_H

#include "Particle_filter.h"



/// Class for holding the results of the simulation study
/// in such a way that they can be used as the columns of a
/// tidy data frame.
class Results_smc {

public:

  std::vector<unsigned int> data_set_;
  std::vector<unsigned int> replicate_;
  // std::vector<int> time_;
  // std::vector<int> dimension_;
  std::vector<int> n_particles_;
  std::vector<double> ess_resampling_threshold_;
  std::vector<std::string> resample_type_;
  std::vector<std::string> metric_;
  std::vector<double> relative_likelihood_estimate_;

  /// Stores a value.
  void store(
    // const unsigned int time,
    // const unsigned int dimension,
    // const unsigned int iteration,
    const unsigned int data_set,
    const unsigned int replicate,
    const unsigned int n_particles,
    const double ess_resampling_threshold,
    const std::string& resample_type,
    const double relative_likelihood_estimate
 ) {
    // time_.push_back(time);
    // dimension_.push_back(dimension);
    // iteration_.push_back(iteration);
    data_set_.push_back(data_set);
    replicate_.push_back(replicate);
    n_particles_.push_back(n_particles);
    ess_resampling_threshold_.push_back(ess_resampling_threshold);
    resample_type_.push_back(resample_type);
    relative_likelihood_estimate_.push_back(relative_likelihood_estimate);
  }

private:

};
/// Simulation study that runs a particle filter for a given state-space model
/// multiple times independently -- using a different data set sampled from the
/// model each time -- for different tuning parameters.
template <class Model> void run_simulation_study_smc(
  Results_smc& results,
  Model& model,
  const unsigned int n_data_sets,
  const unsigned int n_replicates,
  const arma::uvec& n_particles_vec,
  // const unsigned int n_backward_sampling_particles,
  // const unsigned int n_benchmark_backward_sampling_particles,
  const arma::colvec& ess_resampling_thresholds,
  const std::vector<std::string>& resample_types,
  const unsigned int n_benchmark_particles
  // const bool approximate_filtering_distributions, // should the filtering distributions be approximated?
  // const bool approximate_smoothing_distributions // if false, then we only approximate the filtering distributions
) {


  Particle_filter<Model> pf(model);
  double log_likelihood;

  for (const unsigned int data_set : arma::regspace<arma::uvec>(1, n_data_sets)) {

    std::cout << "Progress in terms of data sets: " << static_cast<int>(static_cast<double>(data_set) / n_data_sets * 100) << " %" << std::endl;

    model.simulate_data();

    if (model.has_analytical_solution()) {

      unsigned int T = model.get_n_observations();
      unsigned int D = model.get_state_dimension();

      // The following are not really used here. Only needed if we also approximate
      // filtering/smoothing distributions.
      arma::mat means_filtering(D, T);
      arma::mat means_smoothing(D, T);
      arma::cube variances_filtering(D, D, T);
      arma::cube variances_smoothing(D, D, T);

      log_likelihood = model.evaluate_log_likelihood_and_moments(means_filtering, variances_filtering, means_smoothing, variances_smoothing);

    } else {

      std::cout << "run_simulation_study: run reference PF" << std::endl;
      pf.set_n_particles(n_benchmark_particles);
      pf.set_ess_resampling_threshold(0.9);
      pf.set_resample_type(Resample_type::STRATIFIED);
      log_likelihood = pf.run_particle_filter(); // obtains benchmark results

    }

    for (const std::string& resample_type : resample_types) {

      // std::cout << "resample_type: " << resample_type << std::endl;
      pf.set_resample_type(convert_string_to_resample_type(resample_type));

      for (const double ess_resampling_threshold : ess_resampling_thresholds) {

        // std::cout << "ess_resampling_threshold: " << ess_resampling_threshold << std::endl;
        pf.set_ess_resampling_threshold(ess_resampling_threshold);

        for (const unsigned int replicate : arma::regspace<arma::uvec>(1, n_replicates)) {

          for (const unsigned int n_particles : n_particles_vec) {

            pf.set_n_particles(n_particles);

            // std::cout << "run_particle_filter" << std::endl;
            double log_likelihood_estimate = pf.run_particle_filter();
            // std::cout << "run_ancestor_tracing" << std::endl;

            results.store(
              // time, dimension,
              data_set,
              // iteration,
              replicate,
              n_particles,
              ess_resampling_threshold,
              resample_type,
              std::exp(log_likelihood_estimate - log_likelihood)
            );

          }
        }
      }
    }
  }
}








/// Class for holding the results of the simulation study
/// in such a way that they can be used as the columns of a
/// tidy data frame.
class Results_csmc {

public:

  std::vector<int> time_;
  std::vector<int> dimension_;
  std::vector<int> iteration_;
  std::vector<int> replicate_;
  std::vector<int> n_particles_;
  std::vector<double> ess_resampling_threshold_;
  std::vector<std::string> resample_type_;
  std::vector<std::string> path_selection_type_;
  std::vector<double> state_;

  /// Stores a value.
  void store(
    const unsigned int time,
    const unsigned int dimension,
    const unsigned int iteration,
    const unsigned int replicate,
    const unsigned int n_particles,
    const double ess_resampling_threshold,
    const std::string& resample_type,
    const std::string& path_selection_type,
    const double state
  ) {
    time_.push_back(time);
    dimension_.push_back(dimension);
    iteration_.push_back(iteration);
    replicate_.push_back(replicate);
    n_particles_.push_back(n_particles);
    ess_resampling_threshold_.push_back(ess_resampling_threshold);
    resample_type_.push_back(resample_type);
    path_selection_type_.push_back(path_selection_type);
    state_.push_back(state);
  }

private:

};
template <class Model> void run_simulation_study_csmc(
  Results_csmc& results,
  Model& model,
  const unsigned int n_iterations,
  const unsigned int n_replicates,
  const arma::uvec& n_particles_vec,
  const arma::colvec& ess_resampling_thresholds,
  const std::vector<std::string>& resample_types,
  const std::vector<std::string>& path_selection_types,
  const arma::uvec& times_to_store,
  const arma::uvec& dimensions_to_store // must be of the same length as times_to_store
) {

  if (times_to_store.size() != dimensions_to_store.size()) {
    std::cout << "Error: times_to_store and dimensions_to_store must be of the same length!" << std::endl;
  }

  Particle_filter<Model> pf(model);



  for (const std::string& resample_type : resample_types) {

    std::cout << "resample_type: " << resample_type << std::endl;

    pf.set_resample_type(convert_string_to_resample_type(resample_type));

    for (const std::string& path_selection_type : path_selection_types) {

      std::cout << "path_selection_type: " << path_selection_type << std::endl;

      pf.set_path_selection_type(convert_string_to_path_selection_type(path_selection_type));

      for (const double ess_resampling_threshold : ess_resampling_thresholds) {

        pf.set_ess_resampling_threshold(ess_resampling_threshold);
        std::cout << "ess_resampling_threshold: " << ess_resampling_threshold << std::endl;

        for (const unsigned int replicate : arma::regspace<arma::uvec>(1, n_replicates)) {

          for (const unsigned int n_particles : n_particles_vec) {

            // std::cout << "n_particles: " << n_particles << std::endl;

            pf.set_n_particles(n_particles);

            // std::cout << "run_particle_filter" << std::endl;
            pf.run_particle_filter();
            // std::cout << "run_ancestor_tracing" << std::endl;
            std::vector<arma::colvec> x = pf.run_ancestor_tracing(); // the reference path

            for (const unsigned int iteration : arma::regspace<arma::uvec>(1, n_iterations)) {

              // std::cout << "iteration: " << iteration << std::endl;
              x = pf.run_conditional_particle_filter(x);
              for (unsigned int k = 0; k < times_to_store.size(); ++k) {
                unsigned int time = times_to_store(k);
                unsigned int dimension = dimensions_to_store(k);
                results.store(
                  time,
                  dimension,
                  iteration,
                  replicate,
                  n_particles,
                  ess_resampling_threshold,
                  resample_type,
                  path_selection_type,
                  x[time - 1](dimension - 1)
                );
              }

            }

          }
        }

      // std::cout << "Progress in terms of resample types: " << static_cast<int>(static_cast<double>(i) / n_data_sets * 100) << " %" << std::endl;

      }
    }
  }

}















/// Class for holding the results of the simulation study
/// in such a way that they can be used as the columns of a
/// tidy data frame.
class Results_csmc_acf {

public:

  std::vector<int> time_;
  std::vector<int> dimension_;
  std::vector<int> lag_;
  std::vector<int> data_set_;
  std::vector<int> replicate_;
  std::vector<int> n_particles_;
  std::vector<double> ess_resampling_threshold_;
  std::vector<std::string> resample_type_;
  std::vector<std::string> path_selection_type_;
  std::vector<double> acf_;

  /// Stores a value.
  void store(
    const unsigned int time,
    const unsigned int dimension,
    const unsigned int lag,
    const unsigned int data_set,
    const unsigned int replicate,
    const unsigned int n_particles,
    const double ess_resampling_threshold,
    const std::string& resample_type,
    const std::string& path_selection_type,
    const double acf
  ) {
    time_.push_back(time);
    dimension_.push_back(dimension);
    lag_.push_back(lag);
    data_set_.push_back(data_set);
    replicate_.push_back(replicate);
    n_particles_.push_back(n_particles);
    ess_resampling_threshold_.push_back(ess_resampling_threshold);
    resample_type_.push_back(resample_type);
    path_selection_type_.push_back(path_selection_type);
    acf_.push_back(acf);
  }

private:

};
template <class Model> void run_simulation_study_csmc_acf(
  Results_csmc_acf& results,
  Model& model,
  const unsigned int n_data_sets,
  const unsigned int n_iterations,
  const unsigned int n_replicates,
  const arma::uvec& n_particles_vec,
  const arma::colvec& ess_resampling_thresholds,
  const std::vector<std::string>& resample_types,
  const std::vector<std::string>& path_selection_types,
  const arma::uvec& times_to_store,
  const arma::uvec& dimensions_to_store, // must be of the same length as times_to_store
  const unsigned int lag_max,
  const unsigned int burnin
) {

  if (times_to_store.size() != dimensions_to_store.size()) {
    std::cout << "Error: times_to_store and dimensions_to_store must be of the same length!" << std::endl;
  }

  Particle_filter<Model> pf(model);

  for (const unsigned int data_set : arma::regspace<arma::uvec>(1, n_data_sets)) {

    std::cout << "Progress in terms of data sets: " << static_cast<int>(static_cast<double>(data_set) / n_data_sets * 100) << " %" << std::endl;

    model.simulate_data();

    for (const std::string& resample_type : resample_types) {

      // std::cout << "resample_type: " << resample_type << std::endl;

      pf.set_resample_type(convert_string_to_resample_type(resample_type));

      for (const std::string& path_selection_type : path_selection_types) {

        // std::cout << "path_selection_type: " << path_selection_type << std::endl;

        pf.set_path_selection_type(convert_string_to_path_selection_type(path_selection_type));

        for (const double ess_resampling_threshold : ess_resampling_thresholds) {

          pf.set_ess_resampling_threshold(ess_resampling_threshold);
          // std::cout << "ess_resampling_threshold: " << ess_resampling_threshold << std::endl;

          for (const unsigned int replicate : arma::regspace<arma::uvec>(1, n_replicates)) {

            for (const unsigned int n_particles : n_particles_vec) {

              // std::cout << "n_particles: " << n_particles << std::endl;

              pf.set_n_particles(n_particles);

              // std::cout << "run_particle_filter" << std::endl;
              pf.run_particle_filter();
              // std::cout << "run_ancestor_tracing" << std::endl;
              std::vector<arma::colvec> x = pf.run_ancestor_tracing(); // the reference path

              std::vector<std::vector<arma::colvec>> x_full;

              for (const unsigned int iteration : arma::regspace<arma::uvec>(1, n_iterations)) {
                // std::cout << "iteration: " << iteration << std::endl;
                x = pf.run_conditional_particle_filter(x);

                if (iteration > burnin) {
                  x_full.push_back(x);
                }
              }

              for (unsigned int k = 0; k < times_to_store.size(); ++k) {
                unsigned int time = times_to_store(k);
                unsigned int dimension = dimensions_to_store(k);

                arma::colvec z(x_full.size());
                for (unsigned int i = 0; i < z.size(); ++i) {
                  z(i) = x_full[i][time - 1](dimension - 1);
                }

                arma::colvec acf_vec = acf(z, lag_max);

                for (const unsigned int lag : arma::regspace<arma::uvec>(0, lag_max)) {
                results.store(
                  time,
                  dimension,
                  lag,
                  data_set,
                  replicate,
                  n_particles,
                  ess_resampling_threshold,
                  resample_type,
                  path_selection_type,
                  acf_vec(lag)
                );
                }
              }

            }
          }

        // std::cout << "Progress in terms of resample types: " << static_cast<int>(static_cast<double>(i) / n_data_sets * 100) << " %" << std::endl;

        }
      }
    }
  }

}



#endif

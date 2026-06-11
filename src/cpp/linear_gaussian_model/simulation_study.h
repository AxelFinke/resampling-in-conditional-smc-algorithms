
#ifndef __SIMULATION_STUDY_H
#define __SIMULATION_STUDY_H


#include "Particle_filter.h"


/// Class for holding the results of the simulation study
/// in such a way that they can be used as the columns of a
/// tidy data frame.
class Results {

public:


  std::vector<int> time_;
  std::vector<int> dimension_;
  std::vector<int> iteration_;
  std::vector<int> replicate_;
  std::vector<std::string> resample_type_;
  std::vector<std::string> path_selection_type_;
  std::vector<double> value_;

  /// Initialises the vectors.
  void initialise(const unsigned int K) {
    time_.reserve(K);
    dimension_.reserve(K);
    iteration_.reserve(K);
    replicate_.reserve(K);
    resample_type_.reserve(K);
    path_selection_type_.reserve(K);
    value_.reserve(K);
  }
  /// Stores a value.
  void store(
    const unsigned int time,
    const unsigned int dimension,
    const unsigned int iteration,
    const unsigned int replicate,
    const string& resample_type,
    const string& path_selection_type,
    const double value,
  ) {
    time_.push_back(time);
    dimension_.push_back(dimension);
    iteration_.push_back(iteration);
    replicate_.push_back(replicate);
    resample_type_.push_back(resample_type);
    path_selection_type_.push_back(path_selection_type);
    value_.push_back(value);
  }

private:

};
template <class Model> void run_simulation_study(
  Results& results,
  Model& model,
  const unsigned int n_particles,
  const unsigned int n_iterations,
  const unsigned int n_replicates,
  const std::vector<std::string>& resample_types,
  const std::vector<std::string>& path_selection_types,
  const arma::uvec& times_to_store,
  const arma::uvec& dimensions_to_store, // must be of the same length as times_to_store
  const double ess_resampling_threshold
) {

  if (times_to_store.size() != dimensions_to_store.size()) {
    std::cout << "Error: times_to_store and dimensions_to_store must be of the same length!" << std::endl;
  }

  unsigned int T = model.get_n_observations();
  unsigned int D = model.get_state_dimension();
  unsigned int N = n_particles;
  unsigned int R = resample_types.size();
  unsigned int P = path_selection_types.size();

  Particle_filter<Model> pf(model);
  pf.set_n_particles(n_particles);
  pf.set_ess_resampling_threshold(ess_resampling_threshold);


  for (const std::string& resample_type : resample_types) {

    pf.set_resample_type(convert_string_to_resample_type(resample_type));

    for (const std::string& path_selection_type : path_selection_types) {

      pf.set_path_selection_type(convert_string_to_path_selection_type(path_selection_type));

      for (const unsigned int replicate : arma::regspace<arma::uvec>(0, n_replicates - 1)) {

        pf.run_particle_filter();
        std::vector<arma::colvec> x = pf.run_ancestor_tracing(); // the reference path

        for (const unsigned int iteration : arma::regspace<arma::uvec>(0, n_iterations - 1)) {

          x = pf.run_conditional_particle_filter(x);
          for (unsigned int k = 0; k < times_to_store.size(); ++k) {
            unsigned int time = times_to_store(k);
            unsigned int dimension = dimensions_to_store(k);
            store(time, dimension, iteration, replicate, resample_type, path_selection_type, x[time](dimension));
          }

        }


      }

    // std::cout << "Progress in terms of resample types: " << static_cast<int>(static_cast<double>(i) / n_data_sets * 100) << " %" << std::endl;



    }
  }
}



#endif

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


#include "model.h"
#include "algorithm.h"

// [[Rcpp::plugins(cpp17)]]

////////////////////////////////////////////////////////////////////////////////
// Computes the bias induced by the naive implementations of 
// conditional resampling schemes as well as the lag-1 autocorrelation of the
// initial states
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List evaluate_naive_conditional_resampling_bias(
  const arma::uvec& n_particles_values,
  const arma::uvec& n_observations_values,
  const arma::colvec& epsilon_values,
  const arma::colvec& delta_values,
  const std::vector<std::string>& resample_type_values,
  const std::vector<std::string>& path_selection_type_values
) {
  
 // unsigned int M =
 //   n_particles_values.size() * n_observations_values.size() *
 //   epsilon_values.size() * delta_values.size() *
 //   resample_type_values.size() * path_selection_type_values.size();
  
 std::vector<unsigned int> n_particles_full;
 std::vector<unsigned int> n_observations_full;
 std::vector<double> epsilon_full;
 std::vector<double> delta_full;
 std::vector<std::string> resample_type_full;
 std::vector<std::string> path_selection_type_full;
 std::vector<double> forward_kld_full;
 std::vector<double> reverse_kld_full;
 std::vector<double> tvd_full;
 std::vector<double> autocorrelation_full;
  
 arma::colvec p, log_p; // true (log-)target density
 arma::colvec q, log_q; // (log-)invariant density
 arma::mat P; // transition matrix
 
 // unsigned int m = 0;
 
 for (const auto& T: n_observations_values) {
   
   std::vector<std::vector<unsigned int>> path_space = get_path_space(T);
   log_p.set_size(path_space.size());
   log_q.set_size(path_space.size());
   P.set_size(path_space.size(), path_space.size());
   
   for (const auto& epsilon: epsilon_values) {
     for (const auto& delta: delta_values) {
         
       // Compute true log-target density.
       for (unsigned int i = 0; i < path_space.size(); ++i) {
         log_p(i) = evaluate_log_unnormalised_target_density(path_space[i], epsilon, delta);
       }
       log_p = normalise_vector_in_log_space(log_p);
       p = arma::exp(log_p);

       for (const auto& N: n_particles_values) {
         for (const auto& resample_type: resample_type_values) {
           for (const auto& path_selection_type: path_selection_type_values) {

             // std::cout << "Loop " << m << " of " << M << std::endl;
             std::cout <<
             "N: " << N <<
             "; T: " << T <<
             "; epsilon: " << epsilon <<
             "; delta: " << delta <<
             "; resampling scheme: " << resample_type <<
             "; use backward sampling? " << path_selection_type <<
             std::endl;
             
             evaluate_log_invariant_density(
               log_q,
               P,
               path_space,
               N,
               resample_type,
               path_selection_type,
               epsilon,
               delta
             );
             

             q = arma::exp(log_q);

             n_particles_full.push_back(N);
             n_observations_full.push_back(T);
             epsilon_full.push_back(epsilon);
             delta_full.push_back(delta);
             resample_type_full.push_back(resample_type);
             path_selection_type_full.push_back(path_selection_type);

             forward_kld_full.push_back(compute_kl_divergence(p, q));
             reverse_kld_full.push_back(compute_kl_divergence(q, p));
             tvd_full.push_back(compute_tv_distance(p, q));
             autocorrelation_full.push_back(compute_autocorrelation(q, P, path_space, 0, 1));

             // m++;
           }
         }
       }
     }
   }
 }
   
 return Rcpp::List::create(
   Rcpp::Named("n_particles") = n_particles_full,
   Rcpp::Named("n_observations") = n_observations_full,
   Rcpp::Named("epsilon") = epsilon_full, 
   Rcpp::Named("delta") = delta_full, 
   Rcpp::Named("resample_type") = resample_type_full, 
   Rcpp::Named("path_selection_type") = path_selection_type_full,
   Rcpp::Named("forward_kld") = forward_kld_full, 
   Rcpp::Named("reverse_kld") = reverse_kld_full,
   Rcpp::Named("tvd") = tvd_full, 
   Rcpp::Named("autocorrelation") = autocorrelation_full
 );
}



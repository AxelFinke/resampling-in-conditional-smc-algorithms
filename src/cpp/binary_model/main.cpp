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
  const arma::uvec& resample_type_values,
  const arma::uvec& backward_sampling_type_values
) {
  
 unsigned int M = 
   n_particles_values.size() * n_observations_values.size() * 
   epsilon_values.size() * delta_values.size() *
   resample_type_values.size() * backward_sampling_type_values.size();
  
 arma::uvec n_particles_full(M);
 arma::uvec n_observations_full(M);
 arma::colvec epsilon_full(M);
 arma::colvec delta_full(M);
 arma::uvec resample_type_full(M); 
 arma::uvec backward_sampling_type_full(M);
 arma::colvec forward_kld_full(M);
 arma::colvec reverse_kld_full(M);
 arma::colvec tvd_full(M);
 arma::colvec autocorrelation_full(M); 
  
 arma::colvec p, log_p; // true (log-)target density
 arma::colvec q, log_q; // (log-)invariant density
 arma::mat P; // transition matrix
 
 unsigned int m = 0;
 
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
           for (const auto& backward_sampling_type: backward_sampling_type_values) {

             std::cout << "Loop " << m << " of " << M << std::endl;
             std::cout <<
             "N: " << N <<
             "; T: " << T <<
             "; epsilon: " << epsilon <<
             "; delta: " << delta <<
             "; resampling scheme: " << resample_type <<
             "; use backward sampling? " << backward_sampling_type <<
             std::endl;
             
             evaluate_log_invariant_density(
               log_q,
               P,
               path_space,
               N,
               static_cast<ResampleType>(resample_type),
               static_cast<BackwardSamplingType>(backward_sampling_type),
               epsilon,
               delta
             );
             

             q = arma::exp(log_q);

             n_particles_full(m) = N;
             n_observations_full(m) = T;
             epsilon_full(m) = epsilon;
             delta_full(m) = delta;
             resample_type_full(m) = resample_type;
             backward_sampling_type_full(m) = backward_sampling_type;

             forward_kld_full(m) = compute_kl_divergence(p, q);
             reverse_kld_full(m) = compute_kl_divergence(q, p);
             tvd_full(m) = compute_tv_distance(p, q);
             autocorrelation_full(m) = compute_autocorrelation(q, P, path_space, 0, 1);

             m++;
           }
         }
       }
     }
   }
 }
   
 return Rcpp::List::create(
   Rcpp::Named("N") = n_particles_full, 
   Rcpp::Named("T") = n_observations_full, 
   Rcpp::Named("epsilon") = epsilon_full, 
   Rcpp::Named("delta") = delta_full, 
   Rcpp::Named("resample_type") = resample_type_full, 
   Rcpp::Named("backward_sampling_type") = backward_sampling_type_full,
   Rcpp::Named("forward_kld") = forward_kld_full, 
   Rcpp::Named("reverse_kld") = reverse_kld_full,
   Rcpp::Named("tvd") = tvd_full, 
   Rcpp::Named("autocorrelation") = autocorrelation_full
 );
}



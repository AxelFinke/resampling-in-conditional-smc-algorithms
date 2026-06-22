/// \file
/// \brief Some functions related to conditional SMC algorithms.
///
/// This file contains some generic functions for evaluating
/// the invariant density of a conditional SMC algorithm.

#ifndef __ALGORITHM_H
#define __ALGORITHM_H

#include "misc.h"

/// Specifiers for various path-selection algorithms.
enum class Path_selection_type {
  ANCESTOR_TRACING = 0,
  BACKWARD_SAMPLING,
  ANCESTOR_SAMPLING
};
/// Converts a Path_selection_type object to a std::string.
std::string convert_path_selection_type_to_string(const Path_selection_type& path_selection_type) {
  switch(path_selection_type) {
    case Path_selection_type::ANCESTOR_TRACING: return "ancestor_tracing";
    case Path_selection_type::BACKWARD_SAMPLING: return "backward_sampling";
    case Path_selection_type::ANCESTOR_SAMPLING: return "ancestor_sampling";
    default: return std::string();
  }
}
/// Converts a string to a Path_selection_type object.
Path_selection_type convert_string_to_path_selection_type(const std::string& path_selection_type) {
  if (path_selection_type == "ancestor_tracing") {
    return Path_selection_type::ANCESTOR_TRACING;
  } else if (path_selection_type == "backward_sampling") {
    return Path_selection_type::BACKWARD_SAMPLING;
  } else if (path_selection_type == "ancestor_sampling") {
    return Path_selection_type::ANCESTOR_SAMPLING;
  } else {
    std::cout << "WARNING: Path selection type is not implemented!" << std::endl;
    return Path_selection_type::ANCESTOR_TRACING;
  }
}

enum class Resample_type {
  MULTINOMIAL = 0,
  SYSTEMATIC,
  RESIDUAL_MULTINOMIAL,
  NAIVE_SYSTEMATIC_I,           // WARNING: This induces bias!
  NAIVE_RESIDUAL_MULTINOMIAL_I, // WARNING: This induces bias!
  NAIVE_SYSTEMATIC_II,          // WARNING: This induces bias if $N > 2$!
  NAIVE_RESIDUAL_MULTINOMIAL_II // WARNING: This induces bias if $N > 2$!
};

/// Converts a Resample_type object to a std::string.
std::string convert_resample_type_to_string(const Resample_type& resample_type) {
  switch(resample_type) {
    case Resample_type::MULTINOMIAL: return "multinomial";
    case Resample_type::SYSTEMATIC: return "systematic";
    case Resample_type::RESIDUAL_MULTINOMIAL: return "residual_multinomial";
    case Resample_type::NAIVE_SYSTEMATIC_I: return "naive_systematic_i";
    case Resample_type::NAIVE_RESIDUAL_MULTINOMIAL_I: return "naive_residual_multinomial_i";
    case Resample_type::NAIVE_SYSTEMATIC_II: return "naive_systematic_ii";
    case Resample_type::NAIVE_RESIDUAL_MULTINOMIAL_II: return "naive_residual_multinomial_ii";
    default: return std::string();
  }
}
/// Converts a string to a Resample_type object.
Resample_type convert_string_to_resample_type(const std::string& resample_type) {
  if (resample_type == "multinomial") {
    return Resample_type::MULTINOMIAL;
  } else if (resample_type == "systematic") {
    return Resample_type::SYSTEMATIC;
  } else if (resample_type == "residual_multinomial") {
    return Resample_type::RESIDUAL_MULTINOMIAL;
  } else if (resample_type == "naive_systematic_i") {
    return Resample_type::NAIVE_SYSTEMATIC_I;
  } else if (resample_type == "naive_residual_multinomial_i") {
    return Resample_type::NAIVE_RESIDUAL_MULTINOMIAL_I;
  } else if (resample_type == "naive_systematic_ii") {
    return Resample_type::NAIVE_SYSTEMATIC_II;
  } else if (resample_type == "naive_residual_multinomial_ii") {
    return Resample_type::NAIVE_RESIDUAL_MULTINOMIAL_II;
  } else {
    std::cout << "WARNING: Resampling scheme is not implemented!" << std::endl;
    return Resample_type::MULTINOMIAL;
  }
}


// Returns true if the index of the reference particle
// is always set to zero.
bool is_reference_particle_index_set_to_zero(
  const Resample_type resample_type
) {
  if ((resample_type == Resample_type::MULTINOMIAL) ||
      (resample_type == Resample_type::NAIVE_SYSTEMATIC_I) ||
      (resample_type == Resample_type::NAIVE_RESIDUAL_MULTINOMIAL_I) ||
      (resample_type == Resample_type::NAIVE_SYSTEMATIC_II) ||
      (resample_type == Resample_type::NAIVE_RESIDUAL_MULTINOMIAL_II)) {
    return true;
  } else {
    return false;
  }
}
// Returns "false" for a value of the reference particle
// index guaranteed to be outside of the support of the
// extended target distribution. This
// reduces the combinatorial burden of some of the calculations.
bool does_reference_particle_index_have_support(
  const unsigned int k_new,
  const Resample_type resample_type
) {

  if (is_reference_particle_index_set_to_zero(resample_type) && (k_new != 0)) {
    return false;
  } else {
    return true;
  }
}
// Returns "false" for values of the vector of ancestor indices 
// which are guaranteed to be outside of the support of the
// extended target distribution. This
// reduces the combinatorial burden of some of the calculations.
bool do_ancestor_indices_have_support(
  const std::vector<unsigned int>& a, // values of all N ancestor indices
  const Resample_type resample_type
) {

  bool has_support = true;

  arma::uvec a_aux;

  if (resample_type == Resample_type::SYSTEMATIC) {
    a_aux = arma::conv_to<arma::uvec>::from(a);
    if (!a_aux.is_sorted()) {
      has_support = false;
    }
  } else if ((resample_type == Resample_type::NAIVE_SYSTEMATIC_I) ||
             (resample_type == Resample_type::NAIVE_SYSTEMATIC_II) ||
             (resample_type == Resample_type::MULTINOMIAL)) { // to reduce the computational complexity, we exploit symmetry!
    a_aux = arma::conv_to<arma::uvec>::from(a);
    a_aux = a_aux(arma::span(1, a_aux.size() - 1));
    if (!a_aux.is_sorted()) {
      has_support = false;
    }
  } 
  return has_support;
}
// Returns the state space.
std::vector<unsigned int> get_particle_index_space(const unsigned int N) {
  return create_sequence(0, N - 1);
}
// Returns the space of all possible combinations 
// of input and output path particle indices.
std::vector<std::vector<unsigned int>> get_reference_particle_index_space(const unsigned int N) {

  std::vector<std::vector<unsigned int>> x(2);
  x[0] = get_particle_index_space(N);
  x[1] = get_particle_index_space(N);
  return compute_cartesian_product<unsigned int>(x);
}
// Returns the space of all possible combinations of particles at
// a single time step.
std::vector<std::vector<unsigned int>> get_particle_space(const unsigned int N) {

  std::vector<unsigned int> state_space = get_state_space();
  std::vector<std::vector<unsigned int>> x(N);
  for (unsigned int n = 0; n < N; ++n) {
    x[n] = state_space;
  }
  return compute_cartesian_product<unsigned int>(x);
}
// Returns the space of all possible combinations of ancestor indices
// at a single time step.
std::vector<std::vector<unsigned int>> get_ancestor_index_space(const unsigned int N) {

  std::vector<unsigned int> particle_index_space = get_particle_index_space(N);
  std::vector<std::vector<unsigned int>> x(N);
  for (unsigned int n = 0; n < N; ++n) {
    x[n] = particle_index_space;
  }
  return compute_cartesian_product<unsigned int>(x);
}
// Returns the space of all possible combinations of ancestor indices
// at a single time step except those which cannot have support
// under the resampling distribution.
std::vector<std::vector<unsigned int>> get_reduced_ancestor_index_space(
  const unsigned int N,
  const Resample_type resample_type
) {
  
  std::vector<std::vector<unsigned int>> ancestor_index_space = get_ancestor_index_space(N);
  std::vector<std::vector<unsigned int>> reduced_ancestor_index_space;
  for (unsigned int i = 0; i < ancestor_index_space.size(); ++i) {
    if (do_ancestor_indices_have_support(ancestor_index_space[i], resample_type)) {
      reduced_ancestor_index_space.push_back(ancestor_index_space[i]);
    }
  }
  return reduced_ancestor_index_space;
}
// Returns the space of all possible combinations of
// input and output path particle indices, particles, and
// ancestor indices at a single time step.
std::vector<std::vector<std::vector<unsigned int>>> get_extended_state_space(const unsigned int N) {

  std::vector<std::vector<std::vector<unsigned int>>> x(2);
  x[0] = get_reference_particle_index_space(N); // for the index of the input and output path
  x[1] = get_particle_space(N);

  return compute_cartesian_product<std::vector<unsigned int>>(x);
}
// Returns the space of all possible combinations of
// input and output path particle indices, and particles, 
// at a single time step but with certain combinations 
// are outside of the support of the extended target distribution removed.
std::vector<std::vector<std::vector<unsigned int>>> get_reduced_extended_state_space(
  const unsigned int N,
  const Resample_type resample_type
) {
  
  std::vector<std::vector<std::vector<unsigned int>>> extended_state_space = get_extended_state_space(N);
  std::vector<std::vector<std::vector<unsigned int>>> reduced_extended_state_space;

  std::vector<std::vector<unsigned int>> reduced_ancestor_index_space;
  for (unsigned int i = 0; i < extended_state_space.size(); ++i) {
    if (does_reference_particle_index_have_support(extended_state_space[i][0][0], resample_type)) {
      reduced_extended_state_space.push_back(extended_state_space[i]);
    }
  }
  return reduced_extended_state_space;
}
// Evaluates the log of the initial index density $\lambda_1(k_1)$.
double evaluate_log_initial_lambda(
  const unsigned int k_new,
  const unsigned int n_particles,
  const Resample_type resample_type
) {
  
  double log_density = -std::numeric_limits<double>::infinity();
  if (is_reference_particle_index_set_to_zero(resample_type) && (k_new == 0)) {
    if (k_new == 0) {
      log_density = 0.0;
    }
  } else {
    if (k_new <= n_particles) {
      log_density = -std::log(n_particles);
    }
  } 
  return log_density;
}
// Evaluates the log of the (unconditional) resampling density,
// $\log \rho_{t-1}(a_{t-1}^{1:N}; W_{t-1}^{1:N})$.
double evaluate_log_resampling_density(
  const arma::uvec& ancestor_indices,
  const arma::colvec& normalised_weights,
  const Resample_type resample_type
) {

  unsigned int N = normalised_weights.size();
  unsigned int M = ancestor_indices.size();
  double log_rho = -std::numeric_limits<double>::infinity();

  if (resample_type == Resample_type::MULTINOMIAL) {
    
    if (ancestor_indices.is_sorted()) { // helps reduce the combinatorial complexity
      log_rho = evaluate_log_multinomial_coefficient(ancestor_indices) +
        arma::accu(arma::log(normalised_weights.elem(ancestor_indices)));
    }

  } else if (resample_type == Resample_type::SYSTEMATIC) {

    if (ancestor_indices.is_sorted()) {

      bool has_support = true;
      // lower and upper bounds on the uniform random variable
      // implied by each stratum:
      arma::colvec u_lower(M);
      arma::colvec u_upper(M);

      arma::colvec Q = M * arma::cumsum(normalised_weights);

      unsigned int m = 0;

      while (has_support && (m < M)) {

        if (ancestor_indices(m) > 0) {
          u_lower(m) = std::max(Q(ancestor_indices(m) - 1), static_cast<double>(m)) - m;
        } else {
          u_lower(m) = 0;
        }
        u_upper(m) = std::min(Q(ancestor_indices(m)), static_cast<double>(m + 1)) - m;

        if ((u_lower(m) >= 0) && (u_upper(m) <= 1) && (u_upper(m) > u_lower(m))) {
          m++;
        } else {
          has_support = false;
        }

      }

      double u0 = u_lower.max();
      double u1 = u_upper.min();

      if (has_support && (u1 > u0)) {
        log_rho = std::log(u1 - u0);
      }
    }

  } else if (resample_type == Resample_type::RESIDUAL_MULTINOMIAL) {

    arma::colvec MW = M * normalised_weights;
    arma::uvec MW_floor = arma::conv_to<arma::uvec>::from(arma::floor(MW));

    // This deals with the problem which occurs if all
    // the elements of MW are 1 but have their floor computed as 0 due to
    // finite precision
    if (accu(MW_floor) == 0) {
      for (unsigned int n = 0; n < N; ++n) {
        if (std::abs(MW(n) - 1.0) < 0.0000000001) {
          MW_floor(n) = 1;
        }
      }
    }
    
    arma::uvec alpha = arma::conv_to<arma::uvec>::from(rep(arma::conv_to<arma::colvec>::from(create_sequence(0, N - 1)), MW_floor));
    unsigned int M0 = arma::sum(MW_floor);


    if (((M0 > 0) && (arma::all(ancestor_indices(arma::span(0, M0 - 1)) == alpha))) || (M0 == 0)) {
      if (M0 < M) {
        arma::colvec log_v = arma::log(MW - MW_floor) - std::log(M - M0);
        log_rho = arma::accu(log_v.elem(ancestor_indices(arma::span(M0, M - 1))));
      } else {
        log_rho = 0;
      }
    }
    // std::cout << "END: evaluate log residual-multinomial density!" << std::endl;
  }

  return log_rho;
}
// Evaluates the log of the product of the index density,
// $\log \lambda_t(k_t|k_{t-1}; W_{t-1}^{1:N})$ and
// the conditional resampling density,
// $\log \rho_{t-1}^{-k_t}(a_{t-1}^{-k_t}|a_t^{k_t}; W_{t-1}^{1:N})$.
double evaluate_log_conditional_resampling_density(
  const arma::uvec& ancestor_indices, //  vector of ancestor indices
  const unsigned int k_new, // index of the reference path at time $t$
  const unsigned int k_old, // index of the reference path at time $t - 1$
  const arma::colvec& normalised_weights, // vector of normalised weights
  const Resample_type resample_type // resampling scheme
) {


  double log_density = -std::numeric_limits<double>::infinity();

  if (ancestor_indices(k_new) == k_old) {

    unsigned int N = normalised_weights.size();

     if (resample_type == Resample_type::MULTINOMIAL) {

      if (k_new == 0) { // to reduce the combinatorial complexity, the reference path is put in Position 0
        log_density = evaluate_log_resampling_density(
          ancestor_indices(arma::span(1, N - 1)),
          normalised_weights,
          Resample_type::MULTINOMIAL
        );
      }

    } else if ((resample_type == Resample_type::SYSTEMATIC) || (resample_type == Resample_type::RESIDUAL_MULTINOMIAL)) {

      log_density = evaluate_log_resampling_density(
        ancestor_indices,
        normalised_weights,
        resample_type
      ) - std::log(N) - std::log(normalised_weights(k_old));

    } else if (resample_type == Resample_type::NAIVE_SYSTEMATIC_I) {

      if (k_new == 0) { // assuming that the naive resampling scheme puts the reference path in Position 0
        arma::colvec log_density_aux(N);
        arma::uvec ancestor_indices_aux = ancestor_indices;
        for (unsigned int n = 0; n < N; n++) {
          ancestor_indices_aux(0) = n;
          log_density_aux(n) = evaluate_log_resampling_density(
            ancestor_indices_aux,
            normalised_weights,
            Resample_type::SYSTEMATIC
          );
        }
        log_density = compute_normalising_constant_in_log_space(log_density_aux);
      }

    } else if (resample_type == Resample_type::NAIVE_RESIDUAL_MULTINOMIAL_I) {

      if (k_new == 0) { // assuming that the naive resampling scheme puts the reference path in Position 0
        arma::colvec log_density_aux(N);
        arma::uvec ancestor_indices_aux = ancestor_indices;
        for (unsigned int n = 0; n < N; n++) {
          ancestor_indices_aux(0) = n;
          log_density_aux(n) = evaluate_log_resampling_density(
            ancestor_indices_aux,
            normalised_weights,
            Resample_type::RESIDUAL_MULTINOMIAL
          );
        }
        log_density = compute_normalising_constant_in_log_space(log_density_aux);
      }

    } else if (resample_type == Resample_type::NAIVE_SYSTEMATIC_II) {

      if (k_new == 0) { // assuming that the naive resampling scheme puts the reference path in Position 0
        log_density = evaluate_log_resampling_density(
          ancestor_indices(arma::span(1, N - 1)),
          normalised_weights,
          Resample_type::SYSTEMATIC
        );
      }

    } else if (resample_type == Resample_type::NAIVE_RESIDUAL_MULTINOMIAL_II) {

      if (k_new == 0) { // assuming that the naive resampling scheme puts the reference path in Position 0
        log_density = evaluate_log_resampling_density(
          ancestor_indices(arma::span(1, N - 1)),
          normalised_weights,
          Resample_type::RESIDUAL_MULTINOMIAL
        );
      }

    }

  } else {

    std::cout << "WARNING: ancestor_indices(k_new) != k_old!" << std::endl;

  }
  return log_density;
}
// Evaluates the transition density of the extended target distribution
// of the conditional SMC algorithm.
double evaluate_log_extended_semigroup_transition_density(
  const unsigned int x_in_new, // input reference particle from the current time step
  const unsigned int x_in_old, // input reference particle from the previous time step
  const unsigned int x_out_new, // output reference particle from the current time step
  const unsigned int x_out_old, // output reference particle from the current time step
  const unsigned int k_new, // input reference particle index from the current time step
  const unsigned int k_old, // input reference particle index from the previous time step
  const unsigned int l_new, // output reference particle index from the current time step
  const unsigned int l_old, // output reference particle index from the previous time step
  const std::vector<unsigned int>& x_new, // values of all N particle from the current time step
  const std::vector<unsigned int>& x_old, // values of all N particle from the previous time step
  const std::vector<unsigned int>& a, // values of all N ancestor indices
  const Resample_type resample_type,
  const Path_selection_type path_selection_type,
  const double epsilon,
  const double delta,
  const bool is_final_time_step
) {

  double log_density = 0.0;

  if (
    (x_new[k_new] != x_in_new) ||
    (x_old[k_old] != x_in_old) ||
    (x_new[l_new] != x_out_new) ||
    (x_old[l_old] != x_out_old) ||
    (a[k_new] != k_old) ||
    ((path_selection_type == Path_selection_type::ANCESTOR_SAMPLING) && (a[l_new] != l_old))
  )
  {

    log_density = -std::numeric_limits<double>::infinity();

  } else {

    unsigned int N = x_old.size();
    arma::colvec log_w_old(N);
    arma::colvec log_w_new;

    for (unsigned int n = 0; n < N; ++n) {
      log_w_old(n) = evaluate_log_potential_function(x_old[n], delta);
    }
    log_w_old = normalise_vector_in_log_space(log_w_old);

    if (is_final_time_step) {
      log_w_new.set_size(N);
      for (unsigned int n = 0; n < N; ++n) {
        log_w_new(n) = evaluate_log_potential_function(x_new[n], delta);
      }
      log_w_new = normalise_vector_in_log_space(log_w_new);
      log_density += log_w_new[l_new];
    }

    log_density += evaluate_log_conditional_resampling_density(
      arma::conv_to<arma::uvec>::from(a),
      k_new, k_old, arma::exp(log_w_old),
      resample_type
    );

    for (unsigned int n = 0; n < N; ++n) {
      log_density += evaluate_log_transition_density(x_new[n], x_old[a[n]], epsilon);
    }
    log_density -= evaluate_log_transition_density(x_new[k_new], x_old[k_old], epsilon);

    if (path_selection_type == Path_selection_type::BACKWARD_SAMPLING) {
      arma::colvec log_unnormalised_backward_kernels(N);
      for (unsigned int n = 0; n < N; ++n) {
        log_unnormalised_backward_kernels(n) = log_w_old(n) +
          evaluate_log_transition_density(x_new[l_new], x_old[n], epsilon);
      }
      log_unnormalised_backward_kernels = normalise_vector_in_log_space(log_unnormalised_backward_kernels);
      log_density += log_unnormalised_backward_kernels(l_old);
    }

  }

  return log_density;
}
// Evaluates the initial density of the extended target distribution
// of the conditional SMC algorithm.
double evaluate_log_extended_semigroup_initial_density(
  const unsigned int x_in_new, // input reference particle from the current time step
  const unsigned int x_out_new, // output reference particle from the current time step
  const unsigned int k_new, // input reference particle index from the current time step
  const unsigned int l_new, // output reference particle index from the current time step
  const std::vector<unsigned int>& x_new, // values of all N particle from the current time step
  const Resample_type resample_type,
  const Path_selection_type path_selection_type,
  const double epsilon,
  const double delta,
  const bool is_final_time_step
) {

  double log_density;

  if ((is_reference_particle_index_set_to_zero(resample_type) && (k_new != 0)) ||
      (x_new[k_new] != x_in_new) || 
      (x_new[l_new] != x_out_new)) {

    log_density = -std::numeric_limits<double>::infinity();

  } else {

    unsigned int N = x_new.size();

    log_density = evaluate_log_initial_lambda(k_new, N, resample_type);

    arma::colvec log_w_new;
    if (is_final_time_step) {
      log_w_new.set_size(N);
      for (unsigned int n = 0; n < N; ++n) {
        log_w_new(n) = evaluate_log_potential_function(x_new[n], delta);
      }
      log_w_new = normalise_vector_in_log_space(log_w_new);
      log_density += log_w_new[l_new];
    }

    for (unsigned int n = 0; n < N; ++n) {
      log_density += evaluate_log_initial_density(x_new[n], epsilon);
    }
    log_density -= evaluate_log_initial_density(x_new[k_new], epsilon);

  }

  return log_density;
}
// Evaluates a single entry of the transition matrix
double evaluate_log_transition_matrix_component(
  const std::vector<unsigned int>& path_out, // output reference path, i.e., $\mathbf{x}_{1:2}'$
  const std::vector<unsigned int>& path_in, // input reference path, i.e., $\mathbf{x}_{1:2}$
  const unsigned int N, // number of particles
  const Resample_type resample_type,
  const Path_selection_type path_selection_type,
  const double epsilon,
  const double delta
) {


  unsigned int T = path_in.size();
  std::vector<std::vector<std::vector<unsigned int>>> X =
    get_reduced_extended_state_space(N, resample_type);

  arma::colvec beta_new(X.size());
  bool is_final_time_step;
  if (T > 1) {
    is_final_time_step = false;
  } else {
    is_final_time_step = true;
  }

  for (unsigned int i = 0; i < X.size(); ++i) {
    
    beta_new(i) = evaluate_log_extended_semigroup_initial_density(
      path_in[0], // input reference particle from the current time step
      path_out[0], // output reference particle from the current time step
      X[i][0][0], // input reference particle index from the current time step
      X[i][0][1], // output reference particle index from the current time step
      X[i][1], // values of all N particle from the current time step
      resample_type,
      path_selection_type,
      epsilon,
      delta,
      is_final_time_step
    );
  }

  if (T > 1) {

    std::vector<std::vector<unsigned int>> A =
      get_reduced_ancestor_index_space(N, resample_type);

    arma::colvec beta_old(X.size());
    arma::mat beta_aux(X.size(), A.size());

    for (unsigned int t = 1; t < T; ++t) {

      if (t == T - 1) {
        is_final_time_step = true;
      }

      beta_old.swap(beta_new);
      beta_new.fill(-std::numeric_limits<double>::infinity()); 
      beta_aux.fill(-std::numeric_limits<double>::infinity());
           
      for (unsigned int i = 0; i < X.size(); ++i) {

        if (!((path_in[t] != X[i][1][X[i][0][0]]) || (path_out[t] != X[i][1][X[i][0][1]]))) {
          
          for (unsigned int a = 0; a < A.size(); ++a) {
            
            for (unsigned int j = 0; j < X.size(); ++j) {
              
              beta_aux(j, a) = beta_old(j) + evaluate_log_extended_semigroup_transition_density(
                path_in[t], // input reference particle from the current time step
                path_in[t - 1], // input reference particle from the previous time step
                path_out[t], // output reference particle from the current time step
                path_out[t - 1], // output reference particle from the current time step
                X[i][0][0], // input reference particle index from the current time step
                X[j][0][0], // input reference particle index from the previous time step
                X[i][0][1], // output reference particle index from the current time step
                X[j][0][1], // output reference particle index from the previous time step
                X[i][1], // values of all N particles from the current time step
                X[j][1], // values of all N particles from the previous time step
                A[a], // values of all N ancestor indices
                resample_type,
                path_selection_type,
                epsilon,
                delta,
                is_final_time_step
              );
            }
          }
          beta_new(i) = compute_normalising_constant_in_log_space(arma::vectorise(beta_aux));
        }
      }
    }
  }

  return compute_normalising_constant_in_log_space(beta_new);
}
// Evaluates the invariant density of the CSMC algorithm.
void evaluate_log_invariant_density(
  arma::colvec& log_invariant_density,
  arma::mat& transition_matrix,
  const std::vector<std::vector<unsigned int>>& path_space,
  const unsigned int N, // number of particles
  const std::string& resample_type,
  const std::string& path_selection_type,
  const double epsilon,
  const double delta
) {

  unsigned int n_states = path_space.size();
  transition_matrix.set_size(n_states, n_states);

  for (unsigned int i = 0; i < n_states; ++i) {
    for (unsigned int j = 0; j < n_states; ++j) {
      transition_matrix(i, j) = std::exp(
        evaluate_log_transition_matrix_component(
          path_space[j],
          path_space[i],
          N,
          convert_string_to_resample_type(resample_type),
          convert_string_to_path_selection_type(path_selection_type),
          epsilon,
          delta
        )
      );
    }
  }

  arma::cx_vec eigval;
  arma::cx_mat eigvec;
  arma::eig_gen(eigval, eigvec, transition_matrix.t());

  arma::colvec eigval_aux = arma::abs(eigval - 1);
  unsigned int idx = eigval_aux.index_min();

  log_invariant_density = arma::log(arma::abs(eigvec.col(idx)));
  log_invariant_density = normalise_vector_in_log_space(log_invariant_density);
}

#endif

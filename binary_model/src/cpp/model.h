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


/// This file contains some functions specific to a toy
/// binary state-space model which..

#ifndef __MODEL_H
#define __MODEL_H

#include "misc.h"

// Returns the state space.
std::vector<unsigned int> get_state_space() {
  return {0, 1};
}
// Constructs a vector whose elements are elements of the path space.
std::vector<std::vector<unsigned int>> get_path_space(const unsigned int T) {

  std::vector<std::vector<unsigned int>> x(T);
  for (unsigned int t = 0; t < T; ++t) {
    x[t] = get_state_space();
  }
  return compute_cartesian_product<unsigned int>(x);
}
// Evaluates the log of the initial density, $\log M_1(x_1)$.
double evaluate_log_initial_density(const unsigned int x, const double epsilon) {
  if (x == 0 || x == 1) {
    return std::log((1 - epsilon) * (x == 0) + epsilon * (x == 1));
  } else {
    return -std::numeric_limits<double>::infinity();
  }
}
// Evaluates the log of the transition density, $\log M_t(x_t|x_{t-1})$, for $t > 1$.
double evaluate_log_transition_density(const unsigned int x_new, const unsigned int x_old, const double epsilon) {
  if (x_new == x_old) {
    return std::log(1 - epsilon);
  } else if (x_new == 1 - x_old) {
    return std::log(epsilon);
  } else {
    return -std::numeric_limits<double>::infinity();
  }
}
// Evaluates the log of the potential function, $\log G_t(x_t)$.
double evaluate_log_potential_function(const unsigned int x, const double delta) {
  if (x == 0) {
    return std::log(1 - delta);
  } else if (x == 1) {
    return std::log(delta);
  } else {
    return -std::numeric_limits<double>::infinity();
  }
}
// Evaluates the log of the product of the transition density and of the
// potential function, $\log M_t(x_t|x_{t-1}) + \log G_t(x_t)$.
double evaluate_log_semigroup_transition_density(
  const unsigned int x_new,
  const unsigned int x_old,
  const double epsilon,
  const double delta
) {
  return evaluate_log_transition_density(x_new, x_old, epsilon) +
    evaluate_log_potential_function(x_new, delta);
}
// Evaluates the log of the product of the initial density and of the
// potential function, $\log M_1(x_1) + \log G_1(x_1)$.
double evaluate_log_semigroup_initial_density(
  const unsigned int x,
  const double epsilon,
  const double delta
) {
  return evaluate_log_initial_density(x, epsilon) +
    evaluate_log_potential_function(x, delta);
}
// Evaluates the log of the unnormalised target density.
double evaluate_log_unnormalised_target_density(
  const std::vector<unsigned int>& path,
  const double epsilon,
  const double delta
) {
  if (path.size() > 0) {
    double log_density = evaluate_log_semigroup_initial_density(path[0], epsilon, delta);
    for (unsigned int t = 1; t < path.size(); ++t) {
      log_density += evaluate_log_semigroup_transition_density(path[t], path[t - 1], epsilon, delta);
    }
    return log_density;
  } else {
    return -std::numeric_limits<double>::infinity();
  }
}
// Evaluates the log of the normalising constant.
double evaluate_log_normalising_constant(
  const unsigned int T,
  const double epsilon,
  const double delta
) {

  std::vector<unsigned int> state_space = get_state_space();
  unsigned int K = state_space.size();
  arma::colvec alpha_new(K);
  arma::colvec alpha_old(K);
  arma::colvec aux(K);

  for (unsigned int k = 0; k < K; ++k) {
    alpha_new(k) = evaluate_log_semigroup_initial_density(state_space[k], epsilon, delta);
  }
  for (unsigned int t = 1; t < T; ++t) {
    alpha_old = alpha_new;
    for (unsigned int k = 0; k < K; ++k) {
      for (unsigned int l = 0; l < K; ++l) {
        aux(l) = alpha_old(l) +
          evaluate_log_semigroup_transition_density(state_space[k], state_space[l], epsilon, delta);
      }
      alpha_new(k) = compute_normalising_constant_in_log_space(aux);
    }
  }
  return compute_normalising_constant_in_log_space(alpha_new);
}
// Evaluates the log of the target density.
double evaluate_log_target_density(
  const std::vector<unsigned int>& path,
  const double epsilon,
  const double delta
) {
  return evaluate_log_unnormalised_target_density(path, epsilon, delta) -
    evaluate_log_normalising_constant(path.size(), epsilon, delta);
}


#endif

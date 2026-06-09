/// \file
/// \brief Some auxiliary functions..
///
/// This file contains some generic functions which are needed by the model
/// or conditional SMC algorithm.

#ifndef __MISC_H
#define __MISC_H

#include <RcppArmadillo.h>
#include <cmath>
#include <iostream>

// Counts the number of occurrences of the numbers $0, \dotsc, N - 1$
// in a given arma::uvec.
arma::uvec count_repeats(const arma::uvec& x) {
  // std::cout << "START: count_repeats()" << std::endl;
  unsigned int N = x.max() + 1;
  arma::uvec y(N);
  y.zeros();
  // std::cout << "x: " << x.t() << std::endl;
  // std::cout << "y: " << y.t() << std::endl;
  for (const auto & x_elem : x) {
    // std::cout << "x_elem: " << x_elem << std::endl;
    y(x_elem) = y(x_elem) + 1;
  }
  // std::cout << "END: count_repeats()" << std::endl;
  return y;
}
// Evaluates the log-multinomial coefficient.
double evaluate_log_multinomial_coefficient(const arma::uvec& x) {
  arma::uvec repeats = count_repeats(x);
  double y = std::lgamma(x.size() + 1);
  for (const auto & repeats_elem : repeats) {
    y -= std::lgamma(repeats_elem + 1);
  }
  return y;
}
// Reproduces R's rep() function.
arma::vec rep(const arma::vec& x, const arma::uvec& times) {
  if (x.n_elem != times.n_elem) {
    std::cerr << "Error: x and times must have the same length!" << std::endl;
    return arma::vec();
  }

  size_t total_length = arma::accu(times);
  arma::vec result(total_length);

  size_t pos = 0;
  for (size_t i = 0; i < x.n_elem; ++i) {
    for (unsigned int j = 0; j < times(i); ++j) {
      result(pos++) = x(i);
    }
  }

  return result;
}
// Creates a sequence from a to b (both a and b are included).
std::vector<unsigned int> create_sequence(const unsigned int a, const unsigned int b) {
  if (a > b) {
    std::cerr << "Error: a must be less than or equal to b!" << std::endl;
  }
  std::vector<unsigned int> seq(b - a + 1);
  for (unsigned int i = 0; i < seq.size(); ++i) {
    seq[i] = a + i;
  }
  return seq;
}
// Normalises weights in log-space.
arma::colvec normalise_vector_in_log_space(const arma::colvec& log_w) {
  if (!log_w.has_nan() && (log_w.max() == -std::numeric_limits<double>::infinity())) {
    arma::colvec y(log_w.size());
    y.fill(-std::numeric_limits<double>::infinity());
    return y;
  } else {
    double log_w_max = arma::max(log_w);
    double log_Z = log_w_max + std::log(arma::accu(arma::exp(log_w - log_w_max)));
    return log_w - log_Z;
  }
}
// Computes the log-normalising constant.
double compute_normalising_constant_in_log_space(const arma::colvec& log_w) {
  // std::cout << "START: compute normalising const in log-space!" << std::endl;
  // std::cout << "log_w(0): " << log_w(0) << std::endl;
  double y;
  if (!log_w.has_nan() && (log_w.max() == -std::numeric_limits<double>::infinity())) {
    y = -std::numeric_limits<double>::infinity();
  } else {
    double log_w_max = arma::max(log_w);
    y = log_w_max + std::log(arma::accu(arma::exp(log_w - log_w_max)));
  } 
  // std::cout << "END: compute normalising const in log-space!" << std::endl;
  return y;
}
// Computes the Cartesian product of an arbitrary number of vectors of integers.
// Proposed by user anumi, 2013-06-11, at https://stackoverflow.com/questions/5279051/how-can-i-create-the-cartesian-product-of-a-vector-of-vectors?noredirect=1&lq=1
template<typename T> 
std::vector<std::vector<T>> compute_cartesian_product(const std::vector<std::vector<T>>& in) {

  std::vector<std::vector<T>> results = {{}};
  
  for (const auto& new_values : in) {
    
    std::vector<std::vector<T>> next_results;
    
    for (const auto & result : results) {
      for (const auto & value : new_values) {
        next_results.push_back(result);
        next_results.back().push_back(value);
      }
    }
    results = std::move(next_results);
  }
  return results;
}
// Computes the Kullback--Leibler divergence from a finite
// distribution $q$ to a finite distribution $p$ (both are
// assumed to be represented as vectors of the same length).
double compute_kl_divergence(const arma::colvec& p, const arma::colvec& q) {
  return arma::accu(p % (arma::log(p) - arma::log(q)));
}
// Computes the total variation distance between a finite
// distribution $q$ and a finite distribution $p$ (both are
// assumed to be represented as vectors of the same length).
double compute_tv_distance(const arma::colvec& p, const arma::colvec& q) {
  return 0.5 * arma::accu(arma::abs(p - q));
}
// Computes the lag-one autocorrelation of the state at time $t$.
double compute_autocorrelation(
  const arma::colvec& invariant_density,
  const arma::mat& transition_matrix,
  const std::vector<std::vector<unsigned int>>& path_space,
  const unsigned int time,
  const unsigned int lag
) {
  
  double autocorrelation = 1;
  
  if (lag > 0) {
    
    arma::mat P = arma::powmat(transition_matrix, static_cast<int>(lag));
    double stationary_mean = 0.0;
    double stationary_variance = 0.0;
    double product_mean = 0.0;
    unsigned int n_states = path_space.size();
    
    for (unsigned int i = 0; i < n_states; ++i) {
      stationary_mean += path_space[i][time] * invariant_density(i);
      stationary_variance += path_space[i][time] * path_space[i][time] * invariant_density(i);
      for (unsigned int j = 0; j< n_states; ++j) {
        product_mean += path_space[i][time] * path_space[j][time] * invariant_density(i) * transition_matrix(i, j);
      }
    }
    stationary_variance -= stationary_mean * stationary_mean;
    autocorrelation = (product_mean - stationary_mean * stationary_mean) / stationary_variance;
  }
  
  return autocorrelation;
}

#endif

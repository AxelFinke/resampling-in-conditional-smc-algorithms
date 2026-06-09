#ifndef __RESAMPLING_H
#define __RESAMPLING_H

#include <random>
#include <iostream>
#include <vector>
#include <RcppArmadillo.h>
#include "misc.h"

/// Specifiers for various resampling algorithms.
enum class Resample_type {
  RESAMPLE_MULTINOMIAL = 0,
  RESAMPLE_STRATIFIED,
  RESAMPLE_SYSTEMATIC,
  RESAMPLE_RESIDUAL_MULTINOMIAL,
  RESAMPLE_NAIVE_SYSTEMATIC_I,           // WARNING: This induces bias!
  RESAMPLE_NAIVE_RESIDUAL_MULTINOMIAL_I, // WARNING: This induces bias!
  RESAMPLE_NAIVE_SYSTEMATIC_II,          // WARNING: This induces bias if $N > 2$!
  RESAMPLE_NAIVE_RESIDUAL_MULTINOMIAL_II // WARNING: This induces bias if $N > 2$!

};
/// Converts a Resample_type object to a std::string.
std::string convert_resample_type_to_string(const Resample_type& resample_type) {
  switch(resample_type) {
    case RESAMPLE_MULTINOMIAL: return "multinomial";
    case RESAMPLE_STRATIFIED: return "stratified";
    case RESAMPLE_SYSTEMATIC: return "systematic";
    case RESAMPLE_RESIDUAL_MULTINOMIAL: return "residual_multinomial";
    case RESAMPLE_NAIVE_SYSTEMATIC_I: return "naive_systematic_i";
    case RESAMPLE_NAIVE_RESIDUAL_MULTINOMIAL_I: return "naive_residual_multinomial_i";
    case RESAMPLE_NAIVE_SYSTEMATIC_II: return "naive_systematic_ii";
    case RESAMPLE_NAIVE_RESIDUAL_MULTINOMIAL_II: return "naive_residual_multinomial_ii";
    default: return std::string();
  }
}
/// Converts a string to a Resample_type object.
Resample_type convert_string_to_resample_type(const std::string& resample_type) {
  if (resample_type == "multinomial") {
    return Resample_type::RESAMPLE_MULTINOMIAL;
  } else if (resample_type == "stratified") {
    return Resample_type::RESAMPLE_STRATIFIED;
  } else if (resample_type == "systematic") {
    return Resample_type::RESAMPLE_SYSTEMATIC;
  } else if (resample_type == "residual_multinomial") {
    return Resample_type::RESAMPLE_RESIDUAL_MULTINOMIAL;
  } else if (resample_type == "naive_systematic_i") {
    return Resample_type::RESAMPLE_NAIVE_SYSTEMATIC_I;
  } else if (resample_type == "naive_residual_multinomial_i") {
    return Resample_type::RESAMPLE_NAIVE_RESIDUAL_MULTINOMIAL_I;
  } else if (resample_type == "naive_systematic_ii") {
    return Resample_type::RESAMPLE_NAIVE_SYSTEMATIC_II;
  } else if (resample_type == "naive_residual_multinomial_ii") {
    return Resample_type::RESAMPLE_NAIVE_RESIDUAL_MULTINOMIAL_II;
  } else {
    std::cout << "WARNING: Resampling scheme is not implemented!" << std::endl;
    return Resample_type::RESAMPLE_MULTINOMIAL;
  }
}

namespace resample {

  // In the following:
  //
  // W is the vector of normalised(!) weights ($\mathbf{W} = W^{1:N}$ in the paper).
  // M is the number of descendants ($M$ in the paper).
  // log_w_tilde are the log-resampled weights ($\tilde{w}$ in the paper).
  // k is the index of the current distinguished particle ($k$ in the paper).
  // a is the index of the ancestor of the current distinguished particle ($a$ in the paper).

  /// Efficiently performs stratified inverse-transform sampling.
  stratified_inverse_transform(
    arma::uvec& parent_indices,
    const arma::colvec& W,
    const unsigned int M,
    const arma::colvec T
  ) {
    parent_indices.set_size(M);
    arma::colvec Q = arma::cumsum(W);
    unsigned int i = 0;
    unsigned int j = 0;

    while (j < M) {
      if (T(j) <= Q(i)) {
        parent_indices(j) = i;
        ++j;
      } else {
        ++i;
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  /// Multinomial resampling.
  /////////////////////////////////////////////////////////////////////////////

  /// Performs multinomial resampling.
  void multinomial(
    arma::uvec& parent_indices,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M
  ) {
    log_w_tilde.set_size(M);
    log_w_tilde.fill(-std::log(M));
    arma::colvec u = arma::randu<arma::colvec>(M + 1);
    u = - arma::cumsum(arma::log(u)); // samples from an exponential distribution
    arma::colvec T = u(arma::span(0, M - 1)) / u(M);

    // arma::colvec T = u.sort(); /// NOTE: rather than sorting, we generate (M+1) exponential RVs and take the cumulative sum of the first M of these divided by the sum of all (M+1) of these! This gives O(max(N, M)) complexity! See, e.g., https://djalil.chafai.net/blog/2014/06/03/back-to-basics-order-statistics-of-exponential-distribution/

    stratified_inverse_transform(parent_indices, W, M, T);
  }

  /////////////////////////////////////////////////////////////////////////////
  /// Conditional multinomial resampling.
  /////////////////////////////////////////////////////////////////////////////

  /// Performs conditional multinomial resampling.
  void conditional_multinomial(
    arma::uvec& parent_indices,
    unsigned int& k, // particle index of the distinguished particle
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a
  ) {

    multinomial(parent_indices, log_w_tilde, W, M);
    k = arma::as_scalar(arma::randi(1, arma::distr_param(0, M - 1)));
    parent_indices(k) = a;
  }

  /////////////////////////////////////////////////////////////////////////////
  /// Systematic resampling.
  /////////////////////////////////////////////////////////////////////////////

  /// Performs systematic resampling given a particular uniform random number.
  void systematic_base (
    arma::uvec& parent_indices,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    unsigned int M,
    const double u // uniform random variable supplied by the user
  ) {

    log_w_tilde.set_size(M);
    log_w_tilde.fill(-std::log(M));
    arma::colvec T = (arma::linspace(0, M - 1, M) + u) / M;
    stratified_inverse_transform(parent_indices, W, M, T);
  }
  /// Performs systematic resampling.
  void systematic(
    arma::uvec& parent_indices,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M
  ) {

    systematic_base(parent_indices, log_w_tilde, W, M, arma::randu());
  }


  ////////////////////////////////////////////////////////////////////////////////
  // Conditional systematic resampling
  ////////////////////////////////////////////////////////////////////////////////

  /// Samples from the index distribution.
  void sample_from_index_distribution_systematic(
    double& u,
    unsigned int& k,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a
  ) {
    arma::colvec Q = arma::cumsum(M * W);

    // Determines the highest possible stratum for each cumulative particle weight
    arma::uvec bins = arma::conv_to<arma::uvec>::from(arma::ceil(Q) - 1);
    bins.elem(arma::find(bins == M)).fill(M - 1);
    arma::colvec w_aux; // stores the probabilities of sampling k from a particular stratum

    // First step: obtain the index of the distinguished particle at
    // the current step given the entire history of the particle system
    // and in particular, given that the parent of this particle has index a.

    double lb, ub;
    unsigned int k_aux, first_stratum, last_stratum, n_relevant_strata;

    if ((a == 0) && (bins(a) == 0)) {
      k = 0;
      lb = 0;
      ub = Q(0);
    } else if ((a > 0) && (bins(a) == bins(a - 1))) {
      k = bins(a);
      lb = Q(a - 1) - bins(a);
      ub = Q(a) - bins(a);
    } else if ((a == 0) && (bins(a) > 0)) {
      first_stratum = 0;
      last_stratum = bins(a);
      n_relevant_strata = last_stratum - first_stratum + 1;
      w_aux.ones(n_relevant_strata);
      w_aux(n_relevant_strata - 1) = Q(a) - bins(a);
      w_aux = arma::normalise(w_aux, 1);
      k_aux = sample_int(w_aux);

      if (k_aux == n_relevant_strata - 1) {
        lb = 0;
        ub = Q(a) - bins(a);
      } else {
        lb = 0;
        ub = 1;
      }
      k = k_aux;
    }
    else { // i.e. if (a > 0) & (bins(a) > bins(a - 1))
      first_stratum = bins(a - 1);
      last_stratum = bins(a);
      n_relevant_strata = last_stratum - first_stratum + 1;
      w_aux.ones(n_relevant_strata);
      w_aux(0) = bins(a - 1) - Q(a - 1) + 1;
      w_aux(n_relevant_strata - 1) = Q(a) - bins(a);
      w_aux = arma::normalise(w_aux, 1);
      k_aux = sample_int(w_aux);

      if (k_aux == 0) {
        lb = Q(a - 1) - bins(a - 1);
        ub = 1;
      } else if (k_aux == n_relevant_strata - 1) {
        lb = 0;
        ub = Q(a) - bins(a);
      } else {
        lb = 0;
        ub = 1;
      }
      k = k_aux + first_stratum;
    }
    u = lb + (ub - lb) * arma::randu();
  }

  /// Performs conditional systematic resampling.
  void conditional_systematic(
    arma::uvec& parent_indices,
    unsigned int& k,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a
  ) {
    double u;
    sample_from_index_distribution_systematic(u, k, W, a);
    systematic_base(parent_indices, log_w_tilde, W, M, u);
  }


  /////////////////////////////////////////////////////////////////////////////
  /// Stratified resampling.
  /////////////////////////////////////////////////////////////////////////////

  /// Performs stratified resampling given a particular vector of random numbers.
  void stratified_base (
    arma::uvec& parent_indices,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    unsigned int M,
    const arma::colvec& u // uniform random variables supplied by the user
  ) {
    log_w_tilde.set_size(M);
    log_w_tilde.fill(-std::log(M));
    arma::colvec T = (arma::linspace(0, M - 1, M) + u) / M;
    stratified_inverse_transform(parent_indices, W, M, T);
  }
  /// Performs stratified resampling.
  void stratified(
    arma::uvec& parent_indices,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M
  ) {
    stratified_base(parent_indices, W, M, arma::randu<arma::colvec>(M));
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Conditional systematic resampling
  ////////////////////////////////////////////////////////////////////////////////

  /// Samples from the index distribution.
  void sample_from_index_distribution_stratified(
    double& u,
    unsigned int& k,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a
  ) {
    sample_from_index_distribution_systematic(u, k, W, M, a);
  }
  /// Performs conditional stratified resampling.
  void conditional_stratified(
    arma::uvec& parent_indices,
    unsigned int& k,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a
  ) {

    double u0;
    sample_from_index_distribution_stratified(u0, k, W, a);
    arma::colvec u = arma::randu<arma::colvec>(M);
    u(k) = u0;
    stratified_base(parent_indices, log_w_tilde, W, M, u);
  }


  /////////////////////////////////////////////////////////////////////////////
  /// Residual-multinomial resampling.
  /////////////////////////////////////////////////////////////////////////////

  /// Calculates some auxiliary quantities needed within (conditional)
  /// residial-multinomial resampling.
  void calculate_residual_weights (
    arma::colvec& W_residual,
    unsigned int& M_residual,
    arma::uvec& parent_indices_deterministic,
    arma::colvec& MW,
    arma::uvec& MW_floor,
    const arma::colvec& W,
    unsigned int M
  ) {

    MW = M * W;
    MW_floor = arma::conv_to<arma::uvec>::from(arma::floor(MW));

    unsigned int M0 = arma::sum(MW_floor);
    M_residual = M - M0;
    parent_indices_deterministic.set_size(M0);

    unsigned int m = 0;
    unsigned int n = 0;
    unsigned int l = 0;

    while (m < M0) {
      l = 0;
      while (l < MW_floor(n)) {
        parent_indices_deterministic(m) = n;
        ++m;
        ++l;
      }
      ++n;
    }
    if (M > M0) {
      W_residual = (MW - MW_floor) / M_residual;
    }
  }

  /// Performs residual-multinomial resampling.
  void residual_multinomial_base(
    arma::uvec& parent_indices,
    const arma::colvec& W_residual,
    const unsigned int M_residual,
    const arma::uvec& parent_indices_deterministic
  ) {
    arma::uvec parent_indices_residual;
    arma::colvec log_w_tilde;
    multinomial(parent_indices_residual, log_w_tilde, W_residual, M_residual);
    parent_indices = arma::join_cols(parent_indices_deterministic, parent_indices_residual);
  }
  /// Performs residual-multinomial resampling.
  void residual_multinomial(
    arma::uvec& parent_indices,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    unsigned int M
  ) {
    arma::colvec W_residual, MW;
    unsigned int M_residual;
    arma::uvec parent_indices_deterministic, MW_floor;
    calculate_residual_weights(W_residual, M_residual, parent_indices_deterministic, MW, MW_floor, W, M);
    residual_multinomial_base(parent_indices, W_residual, M_residual, parent_indices_deterministic);
    log_w_tilde.set_size(M);
    log_w_tilde.fill(-std::log(M));
  }

  //////////////////////////////////////////////////////////////////////////////
  // Conditional residual-multinomial resampling
  //////////////////////////////////////////////////////////////////////////////

  /// Performs conditional residual-multinomial resampling.
  void conditional_residual_multinomial (
    arma::uvec& parent_indices,
    unsigned int& k,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a
  ) {
    double u = arma::randu();
    arma::colvec W_residual, MW;
    unsigned int M_residual, l;
    arma::uvec parent_indices_deterministic, parent_indices_residual, MW_floor;

    calculate_residual_weights(W_residual, M_residual, parent_indices_deterministic, MW, MW_floor, W, M);
    unsigned int M0 = parent_indices_deterministic.size();

    if (u * MW(a) <= MW_floor(a)) {

      l = arma::as_scalar(arma::randi(1, arma::distr_param(0, static_cast<int>(MW_floor(a) - 1))));
      if (a == 0) {
        k = l;
      } else {
        k = arma::accu(MW_floor(arma::span(0, a - 1))) + l;
      }

      if (M_residual > 0) {
        residualMultinomial(parent_indices_residual, W_residual, M_residual);
      } else {
        parent_indices_residual.set_size(0);
      }

    } else {

      if (M_residual > 0) {
        conditionalMultinomial(parent_indices_residual, l, W_residual, M_residual, a);
      } else {
        parent_indices_residual.set_size(0);
      }
      k = M0 + l;
    }
    parent_indices = arma::join_cols(parent_indices_deterministic, parent_indices_residual);
    log_w_tilde.set_size(M);
    log_w_tilde.fill(-std::log(M));
  }


  //////////////////////////////////////////////////////////////////////////////
  // Naive (and wrong!) conditional residual-multinomial resampling I
  //////////////////////////////////////////////////////////////////////////////

  /// Performs a naive (and wrong!) version of conditional
  /// residual-multinomial resampling
  void naive_conditional_residual_multinomial_i(
    arma::uvec& parent_indices,
    unsigned int& k,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a
  ) {

    k = 0;
    parent_indices.set_size(M);
    residualMultinomial(parent_indices, w, M);
    parent_indices(0) = a;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Naive (and wrong!) conditional residual-multinomial resampling II
  //////////////////////////////////////////////////////////////////////////////

  /// \brief Performs a naive (and wrong!) version of conditional
  /// residual-multinomial resampling
  void naive_conditional_residual_multinomial_ii(
    arma::uvec& parent_indices,
    unsigned int& k,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a
  ) {
    k = 0;
    arma::uvec aVec(1);
    aVec(0) = a;
    parent_indices.set_size(M - 1);
    residualMultinomial(parent_indices, w, M - 1);
    parent_indices = arma::join_cols(aVec, parent_indices);
  }


  ////////////////////////////////////////////////////////////////////////////////
  // Naive (and wrong!) conditional systematic resampling I
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Performs a naive (and wrong!) version of conditional systematic
  /// systematic resampling.
  void naive_conditional_systematic_i(
    arma::uvec& parent_indices,
    unsigned int& k,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a
  ) {
    k = 0;
    parent_indices.set_size(M);
    systematic(parent_indices, w, M);
    parent_indices(0) = a;

  }
  ////////////////////////////////////////////////////////////////////////////////
  // Naive (and wrong!) conditional systematic resampling II
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Performs a naive (and wrong!) version of conditional systematic
  /// systematic resampling.
  void naive_conditional_systematic_ii(
    arma::uvec& parent_indices,
    unsigned int& k,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a
  ) {

    k = 0;
    arma::uvec aVec(1);
    aVec(0) = a;
    parent_indices.set_size(M - 1);
    systematic(parent_indices, w, M - 1);
    parent_indices = arma::join_cols(aVec, parent_indices);
  }


  /// Performs resampling.
  void resample(
    arma::uvec& parent_indices,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const Resample_type& resample_type
  ) {

    if (resample_type == RESAMPLE_MULTINOMIAL) {
      multinomial(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == RESAMPLE_STRATIFIED) {
      stratified(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == RESAMPLE_SYSTEMATIC) {
      systematic(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == RESAMPLE_RESIDUAL_MULTINOMIAL) {
      residual_multinomial(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == RESAMPLE_NAIVE_SYSTEMATIC_I) {
      systematic(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == RESAMPLE_NAIVE_RESIDUAL_MULTINOMIAL_I) {
      residual_multinomial(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == RESAMPLE_NAIVE_SYSTEMATIC_II) {
      systematic(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == RESAMPLE_NAIVE_RESIDUAL_MULTINOMIAL_II) {
      residual_multinomial(parent_indices, log_w_tilde, W, M);
    }
  }

  /// Performs conditional resampling.
  void conditional_resample(
    arma::uvec& parent_indices,
    unsigned int& k,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a,
    const Resample_type& resample_type
  ) {

    if (resample_type == RESAMPLE_MULTINOMIAL) {
      conditional_multinomial(parent_indices, k, log_w_tilde, W, M, a);
    } else if (resample_type == RESAMPLE_STRATIFIED) {
      conditional_stratified(parent_indices, k, log_w_tilde, W, M, a);
    } else if (resample_type == RESAMPLE_SYSTEMATIC) {
      conditional_systematic(parent_indices, k, log_w_tilde, W, M, a);
    } else if (resample_type == RESAMPLE_RESIDUAL_MULTINOMIAL) {
      conditional_residual_multinomial(parent_indices, k, log_w_tilde, W, M, a);
    } else if (resample_type == RESAMPLE_NAIVE_SYSTEMATIC_I) {
      naive_conditional_systematic_i(parent_indices, k, log_w_tilde, W, M, a);
    } else if (resample_type == RESAMPLE_NAIVE_RESIDUAL_MULTINOMIAL_I) {
      naive_conditional_residual_multinomial_i(parent_indices, k, log_w_tilde, W, M, a);
    } else if (resample_type == RESAMPLE_NAIVE_SYSTEMATIC_II) {
      naive_conditional_systematic_ii(parent_indices, k, log_w_tilde, W, M, a);
    } else if (resample_type == RESAMPLE_NAIVE_RESIDUAL_MULTINOMIAL_II) {
      naive_conditional_residual_multinomial_ii(parent_indices, k, log_w_tilde, W, M, a);
    }
  }

}

#endif

#ifndef __RESAMPLING_H
#define __RESAMPLING_H

#include <random>
#include <iostream>
#include <vector>
#include <cmath>
#include <RcppArmadillo.h>
#include "misc.h"
#include "chopthin.h"

/// Specifiers for various resampling algorithms.
enum class Resample_type {
  MULTINOMIAL = 0,
  STRATIFIED,
  SYSTEMATIC,
  RESIDUAL_MULTINOMIAL,
  CHOPTHIN_BALANCED,
  CHOPTHIN_BALANCED_REORDERED,
  CHOPTHIN_2015,
  CHOPTHIN_2016, // WARNING: This does not have a known valid conditional version
  NAIVE_SYSTEMATIC_I,           // WARNING: This induces bias!
  NAIVE_RESIDUAL_MULTINOMIAL_I, // WARNING: This induces bias!
  NAIVE_SYSTEMATIC_II,          // WARNING: This induces bias if $N > 2$ (assuming $M = N$)!
  NAIVE_RESIDUAL_MULTINOMIAL_II // WARNING: This induces bias if $N > 2$ (assuming $M = N$)!
};
/// Converts a Resample_type object to a std::string.
std::string convert_resample_type_to_string(const Resample_type& resample_type) {
  switch(resample_type) {
    case Resample_type::MULTINOMIAL: return "multinomial";
    case Resample_type::STRATIFIED: return "stratified";
    case Resample_type::SYSTEMATIC: return "systematic";
    case Resample_type::RESIDUAL_MULTINOMIAL: return "residual_multinomial";
    case Resample_type::CHOPTHIN_BALANCED: return "chopthin_balanced";
    case Resample_type::CHOPTHIN_BALANCED_REORDERED: return "chopthin_balanced_reordered";
    case Resample_type::CHOPTHIN_2015: return "chopthin_2015";
    case Resample_type::CHOPTHIN_2016: return "chopthin_2016";
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
  } else if (resample_type == "stratified") {
    return Resample_type::STRATIFIED;
  } else if (resample_type == "systematic") {
    return Resample_type::SYSTEMATIC;
  } else if (resample_type == "residual_multinomial") {
    return Resample_type::RESIDUAL_MULTINOMIAL;
  } else if (resample_type == "chopthin_balanced") {
    return Resample_type::CHOPTHIN_BALANCED;
  } else if (resample_type == "chopthin_balanced_reordered") {
    return Resample_type::CHOPTHIN_BALANCED_REORDERED;
  } else if (resample_type == "chopthin_2015") {
    return Resample_type::CHOPTHIN_2015;
  } else if (resample_type == "chopthin_2016") {
    return Resample_type::CHOPTHIN_2016;
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

namespace resampling {

  // In the following:
  //
  // W is the vector of normalised(!) weights ($\mathbf{W} = W^{1:N}$ in the paper).
  // M is the number of descendants ($M$ in the paper).
  // log_w_tilde are the log-resampled weights ($\tilde{w}$ in the paper).
  // k is the index of the current distinguished particle ($k$ in the paper).
  // a is the index of the ancestor of the current distinguished particle ($a$ in the paper).

  /// Efficiently performs stratified inverse-transform sampling.
  void sample_inverse_transform(
    arma::uvec& parent_indices,
    const arma::colvec& W,
    const unsigned int M,
    const arma::colvec& T
  ) {
    parent_indices.set_size(M);
    arma::colvec Q = arma::cumsum(W);
    unsigned int i = 0;
    unsigned int j = 0;

    while (j < M && i < Q.n_elem) {
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
    u = - arma::cumsum(arma::log(u)); // samples from an Exponential(1) distribution
    arma::colvec T = u(arma::span(0, M - 1)) / u(M);

    // NOTE: to avoid the O(N \log(N)) complexity of sorting, we generate (M+1)
    // exponential RVs and take the cumulative sum of the first M of these
    // divided by the sum of all (M + 1) of these! This gives O(\max(N, M))
    // complexity! See, e.g.,
    // https://djalil.chafai.net/blog/2014/06/03/back-to-basics-order-statistics-of-exponential-distribution/

    sample_inverse_transform(parent_indices, W, M, T);
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
    const unsigned int M,
    const double u // uniform random variable supplied by the user
  ) {

    log_w_tilde.set_size(M);
    log_w_tilde.fill(-std::log(M));
    arma::colvec T = (arma::regspace(0, M - 1) + u) / M;
    sample_inverse_transform(parent_indices, W, M, T);
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
    sample_from_index_distribution_systematic(u, k, W, M, a);
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
    const unsigned int M,
    const arma::colvec& u // uniform random variables supplied by the user
  ) {
    log_w_tilde.set_size(M);
    log_w_tilde.fill(-std::log(M));
    arma::colvec T = (arma::regspace(0, M - 1) + u) / M;
    sample_inverse_transform(parent_indices, W, M, T);
  }
  /// Performs stratified resampling.
  void stratified(
    arma::uvec& parent_indices,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M
  ) {
    stratified_base(parent_indices, log_w_tilde, W, M, arma::randu<arma::colvec>(M));
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
    sample_from_index_distribution_stratified(u0, k, W, M, a);
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
    const unsigned int M
  ) {

    MW = M * W;
    MW_floor = arma::conv_to<arma::uvec>::from(arma::floor(MW));

    unsigned int M_deterministic = arma::sum(MW_floor);
    M_residual = M - M_deterministic;
    parent_indices_deterministic.set_size(M_deterministic);

    unsigned int m = 0;
    for (unsigned int n = 0; n < W.n_elem; ++n) {
      for (unsigned int l = 0; l < MW_floor(n); ++l) {
        parent_indices_deterministic(m++) = n;
      }
    }

    if (M_residual > 0) {
      W_residual = (MW - MW_floor) / M_residual;
    }
  }
  /// Performs residual-multinomial resampling (assuming that the
  /// residual weights have already been computed.
  void residual_multinomial_base(
    arma::uvec& parent_indices,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const arma::colvec& W_residual,
    const unsigned int M_residual,
    const arma::uvec& parent_indices_deterministic
  ) {

    arma::colvec log_w_tilde_aux;
    arma::uvec parent_indices_residual;
    if (M_residual > 0) {
      multinomial(parent_indices_residual, log_w_tilde_aux, W_residual, M_residual);
      parent_indices = arma::join_cols(parent_indices_deterministic, parent_indices_residual);
    } else {
      parent_indices = parent_indices_deterministic;
    }
    log_w_tilde.set_size(M);
    log_w_tilde.fill(-std::log(M));
  }
  /// Performs residual-multinomial resampling.
  void residual_multinomial(
    arma::uvec& parent_indices,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M
  ) {

    arma::colvec W_residual, MW, log_w_tilde_aux;
    arma::uvec parent_indices_deterministic, parent_indices_residual, MW_floor;
    unsigned int M_residual;

    calculate_residual_weights(W_residual, M_residual, parent_indices_deterministic, MW, MW_floor, W, M);
    residual_multinomial_base(parent_indices, log_w_tilde, W, M, W_residual, M_residual, parent_indices_deterministic);
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

    arma::colvec W_residual, MW;
    unsigned int M_residual, l;
    arma::uvec parent_indices_deterministic, parent_indices_residual, MW_floor;

    calculate_residual_weights(W_residual, M_residual, parent_indices_deterministic, MW, MW_floor, W, M);
    unsigned int M_deterministic = parent_indices_deterministic.size();

    if (arma::randu() * MW(a) <= MW_floor(a)) {

      residual_multinomial_base(parent_indices, log_w_tilde, W, M, W_residual, M_residual, parent_indices_deterministic);

      /// The following is equivalent to:
      /// arma::uvec indices = arma::find(parent_indices_deterministic == a);
      /// k = indices(std::floor(arma::randu() * indices.n_elem));
      /// but should be slightly more efficient.
      l = arma::as_scalar(arma::randi(1, arma::distr_param(0, static_cast<int>(MW_floor(a) - 1))));
      if (a == 0) {
        k = l;
      } else {
        k = arma::accu(MW_floor(arma::span(0, a - 1))) + l;
      }

    } else {

      if (M_residual > 0) {
        arma::colvec log_w_tilde_aux;
        conditional_multinomial(parent_indices_residual, l, log_w_tilde_aux, W_residual, M_residual, a);
        parent_indices = arma::join_cols(parent_indices_deterministic, parent_indices_residual);
      } else {
        parent_indices = parent_indices_deterministic;
      }
      k = M_deterministic + l;
      log_w_tilde.set_size(M);
      log_w_tilde.fill(-std::log(M));

    }
  }







  //////////////////////////////////////////////////////////////////////////////
  // Chopthin resampling
  //////////////////////////////////////////////////////////////////////////////

  /// Counts the number of offspring implied by the vector of parent indices.
  arma::uvec count_offspring(const arma::uvec& parent_indices, const unsigned int N) {
    arma::uvec n_offspring(N);
    unsigned int M = parent_indices.size();
    n_offspring.zeros();
    for (unsigned int m = 0; m < M; ++m) {
      unsigned int b = parent_indices(m);
      n_offspring(b) = n_offspring(b) + 1;
    }
    return n_offspring;
  }

  /// Performs chopthin resampling without yet determining the
  /// post-resampling weights.
  void chopthin_base(
    arma::uvec& parent_indices,
    arma::colvec& H,
    double& alpha,
    const arma::colvec& W,
    const unsigned int M,
    const bool reorder_weights,
    const double beta //= 5.828427
  ) {

    alpha = find_alpha(W, M, beta);

    if (reorder_weights) {
      arma::uvec idx_below = arma::find(W < alpha);
      arma::uvec idx_above = arma::find(W >= alpha);
      arma::uvec idx_combined = arma::join_cols(idx_below, idx_above);
      // arma::colvec W_reordered = W.elem(idx_combined);
      H = h(W, alpha, beta);
      arma::colvec V = H.elem(idx_combined) / M;
      arma::colvec log_w_tilde_aux;
      arma::uvec b;
      systematic(b, log_w_tilde_aux, V, M);
      parent_indices = idx_combined.elem(b);
    } else {
      H = h(W, alpha, beta);
      arma::colvec V = H / M;
      arma::colvec log_w_tilde_aux;
      systematic(parent_indices, log_w_tilde_aux, V, M);
    }


    // for (unsigned int m = 0; m < M; ++m) {
    //   unsigned int b = parent_indices(m);
    //   if (W(b) < alpha) {
    //     log_w_tilde(m) = alpha;
    //   } else {
    //     log_w_tilde(m) = W(b) / n_offspring(b);
    //   }
    // }
    // log_w_tilde = arma::log(log_w_tilde);


  }
  /// Performs a type of chopthin resampling which is a special case of
  /// modified-weight resampling (see manuscript).
  void chopthin_balanced(
    arma::uvec& parent_indices,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const double beta = 5.828427
  ) {

    double alpha;
    arma::colvec H;
    chopthin_base(parent_indices, H, alpha, W, M, false, beta);

    log_w_tilde = arma::log(W.elem(parent_indices)) - arma::log(H.elem(parent_indices));
  }
  /// Performs a type of chopthin resampling which is a special case of
  /// modified-weight resampling (see manuscript) but with additional re-ordering
  /// of the weights prior to resampling.
  void chopthin_balanced_reordered(
    arma::uvec& parent_indices,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const double beta = 5.828427
  ) {

    double alpha;
    arma::colvec H;
    chopthin_base(parent_indices, H, alpha, W, M, true, beta);

    log_w_tilde = arma::log(W.elem(parent_indices)) - arma::log(H.elem(parent_indices));
  }
  /// Performs chopthin resampling from Gandy and Lau (2015), e.g., Version 3 on arXiv.
  void chopthin_2015(
    arma::uvec& parent_indices,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const double beta = 5.828427
  ) {

    double alpha;
    arma::colvec H;
    chopthin_base(parent_indices, H, alpha, W, M, false, beta);

    unsigned int N = W.size();
    arma::uvec n_offspring = count_offspring(parent_indices, N);


    log_w_tilde.set_size(M);
    double log_alpha = std::log(alpha);
    for (unsigned int m = 0; m < M; ++m) {
      unsigned int b = parent_indices(m);
      if (W(b) < alpha) {
        log_w_tilde(m) = log_alpha;
      } else {
        log_w_tilde(m) = std::log(W(b)) - std::log(n_offspring(b));
      }
    }
  }
  /// Performs chopthin resampling from Gandy and Lau (2016), e.g., Version 5 on arXiv.
  void chopthin_2016(
    arma::uvec& parent_indices,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const double beta = 5.828427
  ) {

    double alpha;
    arma::colvec H;
    chopthin_base(parent_indices, H, alpha, W, M, true, beta);

    unsigned int N = W.size();
    arma::colvec h_frac(N);
    h_frac.zeros();
    double sl = 0.0;
    for (unsigned int n = 0; n < N; ++n) {
      if (W(n) >= alpha) {
        h_frac(n) = H(n) - std::floor(H(n));
      } else {
        sl += W(n);
      }
    }
    unsigned int cl = 0;
    for (unsigned int m = 0; m < M; ++m) {
      if (W(parent_indices(m)) < alpha) {
        cl++;
      }
    }
    double zeta = (sl - alpha * cl) / sum(h_frac);

    arma::uvec n_offspring = count_offspring(parent_indices, N);

    log_w_tilde.set_size(M);
    double log_alpha = std::log(alpha);
    for (unsigned int m = 0; m < M; ++m) {
      unsigned int b = parent_indices(m);
      if (W(b) < alpha) {
        log_w_tilde(m) = log_alpha;
      } else {
        log_w_tilde(m) = std::log(W(b) - zeta * h_frac(b)) - std::log(n_offspring(b));
      }
    }
  }


  /// Performs conditional chopthin resampling without yet determining the
  /// post-resampling weights.
  void conditional_chopthin_base(
    arma::uvec& parent_indices,
    unsigned int& k,
    arma::colvec& H,
    double& alpha,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a,
    const bool reorder_weights,
    const double beta
  ) {

    alpha = find_alpha(W, M, beta);

    if (reorder_weights) {

      arma::uvec idx_below = arma::find(W < alpha);
      arma::uvec idx_above = arma::find(W >= alpha);
      arma::uvec idx_combined = arma::join_cols(idx_below, idx_above);
      // arma::colvec W_reordered = W.elem(idx_combined);
      H = h(W, alpha, beta);
      arma::colvec V = H.elem(idx_combined) / M;
      arma::colvec log_w_tilde_aux;
      arma::uvec b;

      unsigned int N = W.size();
      arma::uvec idx_inv(N);
      for (arma::uword n = 0; n < N; ++n) {
        idx_inv(idx_combined(n)) = n;
      }

      if (W(a) < alpha) {
        conditional_systematic(b, k, log_w_tilde_aux, V, M, idx_inv(a));
        parent_indices = idx_combined.elem(b);
      } else {
        systematic(b, log_w_tilde_aux, V, M);
        parent_indices = idx_combined.elem(b);
        arma::uvec indices = arma::find(parent_indices == a);
        k = indices(std::floor(arma::randu() * indices.n_elem));
      }
    } else {
      H = h(W, alpha, beta);
      arma::colvec V = H / M;
      arma::colvec log_w_tilde_aux;

      if (W(a) < alpha) {
        conditional_systematic(parent_indices, k, log_w_tilde_aux, V, M, a);
      } else {
        systematic(parent_indices, log_w_tilde_aux, V, M);
        arma::uvec indices = arma::find(parent_indices == a);
        k = indices(std::floor(arma::randu() * indices.n_elem));
      }
    }





    // unsigned int N = W.size();
    // n_offspring.set_size(N);
    // n_offspring.zeros();
    // // log_w_tilde.set_size(M);
    // for (unsigned int m = 0; m < M; ++m) {
    //   unsigned int b = parent_indices(m);
    //   n_offspring(b) = n_offspring(b) + 1;
    // }
    // for (unsigned int m = 0; m < M; ++m) {
    //   unsigned int b = parent_indices(m);
    //   if (W(b) < alpha) {
    //     log_w_tilde(m) = alpha;
    //   } else {
    //     log_w_tilde(m) = W(b) / n_offspring(b);
    //   }
    // }
    // log_w_tilde = arma::log(log_w_tilde);
  }

  /// Performs the conditional version of a type of chopthin resampling
  /// which is a special case of modified-weight resampling (see manuscript).
  void conditional_chopthin_balanced(
    arma::uvec& parent_indices,
    unsigned int& k,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a,
    const double beta = 5.828427
  ) {

    double alpha;
    arma::colvec H;
    conditional_chopthin_base(parent_indices, k, H, alpha, W, M, a, false, beta);
    log_w_tilde = arma::log(W.elem(parent_indices)) - arma::log(H.elem(parent_indices));
  }
  /// Performs the conditional version of a type of chopthin resampling
  /// which is a special case of modified-weight resampling (see manuscript).
  void conditional_chopthin_balanced_reordered(
    arma::uvec& parent_indices,
    unsigned int& k,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a,
    const double beta = 5.828427
  ) {

    double alpha;
    arma::colvec H;
    conditional_chopthin_base(parent_indices, k, H, alpha, W, M, a, true, beta);
    log_w_tilde = arma::log(W.elem(parent_indices)) - arma::log(H.elem(parent_indices));
  }
  /// Performs the conditional version of the chopthin resampling scheme from
  /// Gandy and Lau (2015), e.g., Version 3 on arXiv.
  void conditional_chopthin_2015(
    arma::uvec& parent_indices,
    unsigned int& k,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a,
    const double beta = 5.828427
  ) {

    double alpha;
    arma::colvec H;
    conditional_chopthin_base(parent_indices, k, H, alpha, W, M, a, false, beta);
    log_w_tilde = arma::log(W.elem(parent_indices)) - arma::log(H.elem(parent_indices));

    unsigned int N = W.size();
    arma::uvec n_offspring = count_offspring(parent_indices, N);

    log_w_tilde.set_size(M);
    double log_alpha = std::log(alpha);
    for (unsigned int m = 0; m < M; ++m) {
      unsigned int b = parent_indices(m);
      if (W(b) < alpha) {
        log_w_tilde(m) = log_alpha;
      } else {
        log_w_tilde(m) = std::log(W(b)) - std::log(n_offspring(b));
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Naive (and wrong!) conditional residual-multinomial resampling I
  //////////////////////////////////////////////////////////////////////////////

  /// Performs a naive (and wrong!) version of conditional
  /// residual-multinomial resampling. Performs standard residual-multinomial
  /// resampling and then overwrites the first ancestor index.
  void naive_conditional_residual_multinomial_i(
    arma::uvec& parent_indices,
    unsigned int& k,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a
  ) {
    residual_multinomial(parent_indices, log_w_tilde, W, M);
    k = 0;
    parent_indices(0) = a;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Naive (and wrong!) conditional residual-multinomial resampling II
  //////////////////////////////////////////////////////////////////////////////

  /// Performs a naive (and wrong!) version of conditional
  /// residual-multinomial resampling. Sets the reference particle index to 0 and then
  /// performs standard residual-multinomial resampling for the remaining
  /// $M - 1$ particles.
  void naive_conditional_residual_multinomial_ii(
    arma::uvec& parent_indices,
    unsigned int& k,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a
  ) {
    k = 0;
    arma::uvec a_vec(1);
    a_vec(0) = a;
    parent_indices.set_size(M - 1);
    residual_multinomial(parent_indices, log_w_tilde, W, M - 1);
    parent_indices = arma::join_cols(a_vec, parent_indices);
    log_w_tilde.set_size(M);
    log_w_tilde.fill(-std::log(M));
  }


  ////////////////////////////////////////////////////////////////////////////////
  // Naive (and wrong!) conditional systematic resampling I
  ////////////////////////////////////////////////////////////////////////////////

  /// Performs a naive (and wrong!) version of conditional systematic
  /// systematic resampling. Performs standard systematic resampling
  /// and then overwrites the first ancestor index.
  void naive_conditional_systematic_i(
    arma::uvec& parent_indices,
    unsigned int& k,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a
  ) {
    systematic(parent_indices, log_w_tilde, W, M);
    k = 0;
    parent_indices(0) = a;

  }
  ////////////////////////////////////////////////////////////////////////////////
  // Naive (and wrong!) conditional systematic resampling II
  ////////////////////////////////////////////////////////////////////////////////

  /// Performs a naive (and wrong!) version of conditional systematic
  /// systematic resampling. Sets the reference particle index to 0 and then
  /// performs standard systematic resampling for the remaining $M - 1$ particles.
  void naive_conditional_systematic_ii(
    arma::uvec& parent_indices,
    unsigned int& k,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a
  ) {

    k = 0;
    arma::uvec a_vec(1);
    a_vec(0) = a;
    parent_indices.set_size(M - 1);
    systematic(parent_indices, log_w_tilde, W, M - 1);
    parent_indices = arma::join_cols(a_vec, parent_indices);
    log_w_tilde.set_size(M);
    log_w_tilde.fill(-std::log(M));
  }


  /// Performs resampling.
  void unconditional(
    arma::uvec& parent_indices,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const Resample_type& resample_type
  ) {

    if (resample_type == Resample_type::MULTINOMIAL) {
      multinomial(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == Resample_type::STRATIFIED) {
      stratified(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == Resample_type::SYSTEMATIC) {
      systematic(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == Resample_type::RESIDUAL_MULTINOMIAL) {
      residual_multinomial(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == Resample_type::CHOPTHIN_BALANCED) {
      chopthin_balanced(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == Resample_type::CHOPTHIN_BALANCED_REORDERED) {
      chopthin_balanced_reordered(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == Resample_type::CHOPTHIN_2015) {
      chopthin_2015(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == Resample_type::CHOPTHIN_2016) {
      chopthin_2016(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == Resample_type::NAIVE_SYSTEMATIC_I) {
      systematic(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == Resample_type::NAIVE_RESIDUAL_MULTINOMIAL_I) {
      residual_multinomial(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == Resample_type::NAIVE_SYSTEMATIC_II) {
      systematic(parent_indices, log_w_tilde, W, M);
    } else if (resample_type == Resample_type::NAIVE_RESIDUAL_MULTINOMIAL_II) {
      residual_multinomial(parent_indices, log_w_tilde, W, M);
    }
  }

  /// Performs conditional resampling.
  void conditional(
    arma::uvec& parent_indices,
    unsigned int& k,
    arma::colvec& log_w_tilde,
    const arma::colvec& W,
    const unsigned int M,
    const unsigned int a,
    const Resample_type& resample_type
  ) {

    if (resample_type == Resample_type::MULTINOMIAL) {
      conditional_multinomial(parent_indices, k, log_w_tilde, W, M, a);
    } else if (resample_type == Resample_type::STRATIFIED) {
      conditional_stratified(parent_indices, k, log_w_tilde, W, M, a);
    } else if (resample_type == Resample_type::SYSTEMATIC) {
      conditional_systematic(parent_indices, k, log_w_tilde, W, M, a);
    } else if (resample_type == Resample_type::RESIDUAL_MULTINOMIAL) {
      conditional_residual_multinomial(parent_indices, k, log_w_tilde, W, M, a);
    } else if (resample_type == Resample_type::CHOPTHIN_BALANCED) {
      conditional_chopthin_balanced(parent_indices, k, log_w_tilde, W, M, a);
    } else if (resample_type == Resample_type::CHOPTHIN_BALANCED_REORDERED) {
      conditional_chopthin_balanced(parent_indices, k, log_w_tilde, W, M, a);
    } else if (resample_type == Resample_type::CHOPTHIN_2015) {
      conditional_chopthin_2015(parent_indices, k, log_w_tilde, W, M, a);
    } else if (resample_type == Resample_type::CHOPTHIN_2016) {
      std::cout << "ERROR: It is not (yet) known whether the version of chopthin resampling from Gandy & Lau (2016) admits a valid conditional version! Hence, no conditional version is implemented!" << std::endl;
    } else if (resample_type == Resample_type::NAIVE_SYSTEMATIC_I) {
      naive_conditional_systematic_i(parent_indices, k, log_w_tilde, W, M, a);
    } else if (resample_type == Resample_type::NAIVE_RESIDUAL_MULTINOMIAL_I) {
      naive_conditional_residual_multinomial_i(parent_indices, k, log_w_tilde, W, M, a);
    } else if (resample_type == Resample_type::NAIVE_SYSTEMATIC_II) {
      naive_conditional_systematic_ii(parent_indices, k, log_w_tilde, W, M, a);
    } else if (resample_type == Resample_type::NAIVE_RESIDUAL_MULTINOMIAL_II) {
      naive_conditional_residual_multinomial_ii(parent_indices, k, log_w_tilde, W, M, a);
    }
  }

}

#endif

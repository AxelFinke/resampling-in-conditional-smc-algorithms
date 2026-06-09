#ifndef __STATE_SPACE_MODELS_H
#define __STATE_SPACE_MODELS_H

#include "helper_functions.h"

/// A linear-Gaussian state-space model.
class Linear_gaussian_state_space_model
{

public:

  /// Constructing the model using default parameters.
  Linear_gaussian_state_space_model() {
    set_default_model_parameters();
  }
  /// Returns the true latent state at some time $t$.
  arma::colvec get_state(const unsigned int t) const {return states_.col(t);}
  /// Returns the observation at some time $t$.
  arma::colvec get_observation(const unsigned int t) const {return observations_.col(t);}
  /// Returns the number of observations.
  unsigned int get_n_observations() const {return n_observations_;}
  /// Returns the dimension of the latent state.
  unsigned int get_state_dimension() const {return dx_;}
  /// Specifies the number of observations.
  void set_n_observations(const unsigned int n_observations) {n_observations_ = n_observations;}
  /// Specifies the observations.
  void set_observations(const arma::mat& observations) {
    observations_ = observations;
    n_observations_ = observations_.n_cols;
    states_.set_size(1, n_observations_);
    states_.fill(0);
  }
  /// Specifies the model parameters.
  void set_model_parameters(
    const arma::colvec& a,
    const arma::colvec& c,
    const arma::mat& A,
    const arma::mat& B,
    const arma::mat& C,
    const arma::mat& D,
    const arma::colvec& m0,
    const arma::colvec& C0
  ) {
    a_ = a;
    c_ = c;
    A_ = A;
    B_ = B;
    C_ = C;
    D_ = D;
    m0_ = m0;
    C0_ = C0;
    C0_root_ = arma::chol(C0, "lower");
    dx_ = A_.n_rows;
    dy_ = C_.n_rows;
  }
  /// Sets the model parameters to default values.
  void set_default_model_parameters() {
    a_.zeros(1);
    c_.zeros(1);
    A_.set_size(1, 1);
    A_(0, 0) = 0.95;
    B_.set_size(1, 1);
    B_(0, 0) = 0.5;
    C_.set_size(1, 1);
    C_.ones();
    D_.set_size(1, 1);
    D_.ones();
    m0_.set_size(1);
    m0_.ones();
    C0_.set_size(1, 1);
    C0_(0, 0) = B_(0, 0) * B_(0, 0) / (1.0 - A_(0, 0) * A_(0, 0));
    C0_root_ = arma::chol(C0_, "lower");
    dx_ = A_.n_rows;
    dy_ = C_.n_rows;
  }
  /// Samples from the initial distribution of the latent Markov chain.
  arma::colvec sample_from_initial_distribution() const {
    return m0_ + C0_root_ * arma::randn(dx_);
  }
  /// Samples from the transition kernel of the latent Markov chain.
  arma::colvec sample_from_transition_equation(const unsigned int t, const arma::colvec& x) const {
    return a_ + A_ * x + B_ * arma::randn(dx_);
  }
  /// Samples from the observation equation.
  arma::colvec sample_from_observation_equation(const unsigned int t, const arma::colvec& x) const {
    return c_ + C_ * x + D_ * arma::randn(dy_);
  }
  /// Evaluates the initial density of the latent Markov chain.
  double evaluate_log_initial_density(const arma::colvec& x) const {
    return evaluate_gaussian_density(x, m0_, C0_root_, true, true);
  }
  /// Evaluates the log-transition density of the latent Markov chain.
  double evaluate_log_transition_density(const unsigned int t, const arma::colvec& x_new, const arma::colvec& x_old) const {
    return evaluate_gaussian_density(x_new, a_ + A_ * x_old, B_, true, true);
  }
  /// Evaluates the log-observation density of the latent Markov chain.
  double evaluate_log_observation_density(const unsigned int t, const arma::colvec& x) const {
    return evaluate_gaussian_density(observations_.col(t), c_ + C_ * x, D_, true, true);
  }
  /// Simulates data from the model.
  void simulate_data() {
    states_.set_size(dx_, n_observations_);
    observations_.set_size(dy_, n_observations_);
    states_.col(0) = sample_from_initial_distribution();
    observations_.col(0) = sample_from_observation_equation(0, states_.col(0));
    for (unsigned int t = 1; t < n_observations_; ++t) {
      states_.col(t) = sample_from_transition_equation(t, states_.col(t - 1));
      observations_.col(t) = sample_from_observation_equation(t, states_.col(t));
    }
  }
  /// Evaluates the log-likelihood as well as the filtering and smoothing moments.
  double evaluate_log_likelihood_and_moments(
    arma::mat& means_filtering,
    arma::cube& variances_filtering,
    arma::mat& means_smoothing,
    arma::cube& variances_smoothing
  ) {

    arma::mat means_prediction;
    arma::cube variances_prediction;

    double log_likelihood = run_kalman_filter(
      means_prediction,
      variances_prediction,
      means_filtering,
      variances_filtering,
      a_, c_, A_, B_, C_, D_, m0_, C0_, observations_
    );

    run_kalman_smoother(
      means_smoothing,
      variances_smoothing,
      means_prediction,
      variances_prediction,
      means_filtering,
      variances_filtering,
      A_
    );
    return log_likelihood;
  }

private:

  unsigned int dx_, dy_, n_observations_;
  arma::mat observations_;
  arma::mat states_;
  arma::mat A_, B_, C_, D_, C0_, C0_root_;
  arma::colvec m0_, a_, c_;

};
/// The well known univariate stochastic volatility model.
class Stochastic_volatility_model
{

public:

  /// Constructing the model using default parameters.
  Stochastic_volatility_model() {
    set_default_model_parameters();
  }
  /// Returns the true latent state at some time $t$.
  arma::colvec get_state(const unsigned int t) const {return states_.col(t);}
  /// Returns the observation at some time $t$.
  arma::colvec get_observation(const unsigned int t) const {return observations_.col(t);}
  /// Returns the number of observations.
  unsigned int get_n_observations() const {return n_observations_;}
  /// Returns the dimension of the latent state.
  unsigned int get_state_dimension() const {return dx_;}
  /// Specifies the number of observations.
  void set_n_observations(const unsigned int n_observations) {n_observations_ = n_observations;}
  /// Specifies the observations.
  void set_observations(const arma::mat& observations) {
    observations_ = observations;
    n_observations_ = observations_.n_cols;
    states_.set_size(1, n_observations_);
    states_.fill(0);
  }
  /// Specifies the model parameters.
  void set_model_parameters(
    const double phi,
    const double sigma,
    const double beta
  ) {
    phi_ = phi;
    sigma_ = sigma;
    beta_ = beta;
    c_.set_size(1);
    c_.zeros();
    m0_.set_size(1);
    m0_.zeros();
    C0_root_.set_size(1, 1);
    C0_root_(0) = sigma / std::sqrt(1.0 - phi_ * phi_);
  }
  /// Sets the model parameters to values commonly used in the literature.
  void set_default_model_parameters() {
    phi_ = 0.91;
    sigma_ = 1;
    beta_ = 0.5;
    c_.set_size(1);
    c_.zeros();
    m0_.set_size(1);
    m0_.zeros();
    C0_root_.set_size(1, 1);
    C0_root_(0) = sigma_ / std::sqrt(1.0 - phi_ * phi_);
  }
  /// Samples from the initial distribution of the latent Markov chain.
  arma::colvec sample_from_initial_distribution() const {
    return m0_ + C0_root_ * arma::randn(dx_);
  }
  /// Samples from the transition kernel of the latent Markov chain.
  arma::colvec sample_from_transition_equation(const unsigned int t, const arma::colvec& x) const {
    return phi_ * x + sigma_ * arma::randn(dx_);
  }
  /// Samples from the observation equation.
  arma::colvec sample_from_observation_equation(const unsigned int t, const arma::colvec& x) {
    return beta_ * arma::exp(x / 2.0) * arma::randn(dy_);
  }
  /// Evaluates the initial density of the latent Markov chain.
  double evaluate_log_initial_density(const arma::colvec& x) const {
    return evaluate_gaussian_density(x, m0_, C0_root_, true, true);
  }
  /// Evaluates the log-transition density of the latent Markov chain.
  double evaluate_log_transition_density(const unsigned int t, const arma::colvec& x_new, const arma::colvec& x_old)  const {
    return evaluate_gaussian_density(x_new, phi_ * x_old, sigma_, true, true);
  }
  /// Evaluates the log-observation density of the latent Markov chain.
  double evaluate_log_observation_density(const unsigned int t, const arma::colvec& x) const {
    return evaluate_gaussian_density(observations_.col(t), c_, beta_ * arma::exp(x / 2.0), true, true);
  }
  /// Simulates data from the model.
  void simulate_data() {
    states_.set_size(dx_, n_observations_);
    observations_.set_size(dy_, n_observations_);
    states_.col(0) = sample_from_initial_distribution();
    observations_.col(0) = sample_from_observation_equation(0, states_.col(0));
    for (unsigned int t = 1; t < n_observations_; ++t) {
      states_.col(t) = sample_from_transition_equation(t, states_.col(t - 1));
      observations_.col(t) = sample_from_observation_equation(t, states_.col(t));
    }
  }
  /// Evaluates the log-likelihood as well as the filtering and smoothing moments.
  double evaluate_log_likelihood_and_moments(
    arma::mat& means_filtering,
    arma::cube& variances_filtering,
    arma::mat& means_smoothing,
    arma::cube& variances_smoothing
  ) {
    // NOTE: The likelihood and filtering/smoothing moments are not analytically tractable in thie model
    return - std::numeric_limits<double>::infinity();
  }


private:

  unsigned int dx_ = 1;
  unsigned int dy_ = 1;
  unsigned int n_observations_;
  arma::mat observations_;
  arma::mat states_;
  double phi_, sigma_, beta_;
  arma::mat C0_root_;
  arma::colvec m0_, c_;

};
/// The well known univariate non-linear state-space model.
class Nonlinear_state_space_model
{

public:

  /// Constructing the model using default parameters.
  Nonlinear_state_space_model() {
    set_default_model_parameters();
  }
  /// Returns the true latent state at some time $t$.
  arma::colvec get_state(const unsigned int t) const {return states_.col(t);}
  /// Returns the observation at some time $t$.
  arma::colvec get_observation(const unsigned int t) const {return observations_.col(t);}
  /// Returns the number of observations.
  unsigned int get_n_observations() const {return n_observations_;}
  /// Returns the dimension of the latent state.
  unsigned int get_state_dimension() const {return dx_;}
  /// Specifies the number of observations.
  void set_n_observations(const unsigned int n_observations) {n_observations_ = n_observations;}
  /// Specifies the observations.
  void set_observations(const arma::mat& observations) {
    observations_ = observations;
    n_observations_ = observations_.n_cols;
    states_.set_size(1, n_observations_);
    states_.fill(0);
  }
  /// Specifies the model parameters.
  void set_model_parameters(
    const double alpha_0,
    const double alpha_1,
    const double alpha_2,
    const double b,
    const double sigma_u,
    const double sigma_v
  ) {
    alpha_0_ = alpha_0;
    alpha_1_ = alpha_1;
    alpha_2_ = alpha_2;
    b_ = b;
    sigma_u_ = sigma_u;
    sigma_v_ = sigma_v;
    m0_.set_size(1);
    m0_.zeros();
    C0_root_.set_size(1, 1);
    C0_root_.ones();
  }
  /// Sets the model parameters to values commonly used in the literature.
  void set_default_model_parameters() {
    alpha_0_ = 0.5;
    alpha_1_ = 25;
    alpha_2_ = 8;
    b_ = 0.05;
    sigma_u_ = 1;
    sigma_v_ = std::sqrt(10);
    m0_.set_size(1);
    m0_.zeros();
    C0_root_.set_size(1, 1);
    C0_root_(0, 0) = std::sqrt(5);
  }
  /// Samples from the initial distribution of the latent Markov chain.
  arma::colvec sample_from_initial_distribution() const {
    return m0_ + C0_root_ * arma::randn(dx_);
  }
  /// Samples from the transition kernel of the latent Markov chain.
  arma::colvec sample_from_transition_equation(const unsigned int t, const arma::colvec& x) const {
    return evaluate_mean(t, x) + sigma_u_ * arma::randn(dx_);
  }
  /// Samples from the observation equation.
  arma::colvec sample_from_observation_equation(const unsigned int t, const arma::colvec& x) {
    return b_ * arma::pow(x, 2) + sigma_v_ * arma::randn(dy_);
  }
  /// Evaluates the initial density of the latent Markov chain.
  double evaluate_log_initial_density(const arma::colvec& x) const {
    return evaluate_gaussian_density(x, m0_, C0_root_, true, true);
  }
  /// Evaluates the log-transition density of the latent Markov chain.
  double evaluate_log_transition_density(const unsigned int t, const arma::colvec& x_new, const arma::colvec& x_old)  const {
    return evaluate_gaussian_density(x_new, evaluate_mean(t, x_old), sigma_u_, true, true);
  }
  /// Evaluates the log-observation density of the latent Markov chain.
  double evaluate_log_observation_density(const unsigned int t, const arma::colvec& x) const {
    return evaluate_gaussian_density(observations_.col(t), b_ * arma::pow(x, 2), sigma_v_, true, true);
  }
  /// Simulates data from the model.
  void simulate_data() {
    states_.set_size(dx_, n_observations_);
    observations_.set_size(dy_, n_observations_);
    states_.col(0) = sample_from_initial_distribution();
    observations_.col(0) = sample_from_observation_equation(0, states_.col(0));
    for (unsigned int t = 1; t < n_observations_; ++t) {
      states_.col(t) = sample_from_transition_equation(t, states_.col(t - 1));
      observations_.col(t) = sample_from_observation_equation(t, states_.col(t));
    }
  }
  /// Evaluates the log-likelihood as well as the filtering and smoothing moments.
  double evaluate_log_likelihood_and_moments(
    arma::mat& means_filtering,
    arma::cube& variances_filtering,
    arma::mat& means_smoothing,
    arma::cube& variances_smoothing
  ) {
    // NOTE: The likelihood and filtering/smoothing moments are not analytically tractable in thie model
    return - std::numeric_limits<double>::infinity();
  }

private:

  unsigned int dx_ = 1;
  unsigned int dy_ = 1;
  unsigned int n_observations_;
  arma::mat observations_;
  arma::mat states_;
  double alpha_0_, alpha_1_, alpha_2_, b_, sigma_u_, sigma_v_;
  arma::mat C0_root_;
  arma::colvec m0_;

  /// Evaluates the mean of the transition equation.
  arma::colvec evaluate_mean(const unsigned int t, const arma::colvec& x) const {
    return alpha_0_ * x + alpha_1_ * x / (1.0 + x(0) * x(0)) + alpha_2_ * std::cos(1.2 * t);
  }
};
/// The Lorenz 63 model.
class Lorenz_state_space_model
{

public:

  /// Constructing the model using default parameters.
  Lorenz_state_space_model() {
    set_default_model_parameters();
  }
  /// Returns the true latent state at some time $t$.
  arma::colvec get_state(const unsigned int t) const {return states_.col(t);}
  /// Returns the observation at some time $t$.
  arma::colvec get_observation(const unsigned int t) const {return observations_.col(t);}
  /// Returns the number of observations.
  unsigned int get_n_observations() const {return n_observations_;}
  /// Returns the dimension of the latent state.
  unsigned int get_state_dimension() const {return dx_;}
  /// Specifies the number of observations.
  void set_n_observations(const unsigned int n_observations) {
    n_observations_ = n_observations;
  }
  /// Specifies the observations.
  void set_observations(const arma::mat& observations) {
    observations_ = observations;
    n_observations_ = observations_.n_cols;
    states_.set_size(dx_, n_observations_);
    states_.fill(0);
  }
  /// Specifies the model parameters.
  void set_model_parameters(
    const double s,
    const double rho,
    const double beta,
    const double sigma_v
  ) {
    s_ = s;
    rho_ = rho;
    beta_ = beta;
    sigma_v_ = sigma_v;
    delta_ = 0.01;
    sigma_u_ = std::sqrt(delta_ / 2);
    m0_.set_size(dx_);
    m0_.zeros();
    C0_root_.set_size(dx_, dx_);
    C0_root_.eye();
  }
  /// Sets the model parameters to values commonly used in the literature.
  void set_default_model_parameters() {

    s_ = 10;
    rho_ = 28;
    beta_ = 8.0 / 3;
    sigma_v_ = 1;
    delta_ = 0.01;
    sigma_u_ = std::sqrt(delta_ / 2);
    m0_.set_size(dx_);
    m0_.zeros();
    C0_root_.set_size(dx_, dx_);
    C0_root_.eye();
  }
  /// Samples from the initial distribution of the latent Markov chain.
  arma::colvec sample_from_initial_distribution() const {
    return m0_ + C0_root_ * arma::randn(dx_);
  }
  /// Samples from the transition kernel of the latent Markov chain.
  arma::colvec sample_from_transition_equation(const unsigned int t, const arma::colvec& x) const {

    return evaluate_mean(t, x) + sigma_u_* arma::randn(dx_);
  }
  /// Samples from the observation equation.
  arma::colvec sample_from_observation_equation(const unsigned int t, const arma::colvec& x) {
    arma::colvec mean(1);
    mean(0) = x(0);
    return mean + sigma_v_ * arma::randn(dy_);
  }
  /// Evaluates the initial density of the latent Markov chain.
  double evaluate_log_initial_density(const arma::colvec& x) const {
    return evaluate_gaussian_density(x, m0_, C0_root_, true, true);
  }
  /// Evaluates the log-transition density of the latent Markov chain.
  double evaluate_log_transition_density(const unsigned int t, const arma::colvec& x_new, const arma::colvec& x_old)  const {

    return evaluate_gaussian_density(x_new, evaluate_mean(t, x_old), sigma_u_, true, true);
  }
  /// Evaluates the log-observation density of the latent Markov chain.
  double evaluate_log_observation_density(const unsigned int t, const arma::colvec& x) const {
    arma::colvec mean(1);
    mean(0) = x(0);
    return evaluate_gaussian_density(observations_.col(t), mean, sigma_v_, true, true);
  }
  /// Simulates data from the model.
  void simulate_data() {
    states_.set_size(dx_, n_observations_);
    observations_.set_size(dy_, n_observations_);
    states_.col(0) = sample_from_initial_distribution();
    observations_.col(0) = sample_from_observation_equation(0, states_.col(0));
    for (unsigned int t = 1; t < n_observations_; ++t) {
      states_.col(t) = sample_from_transition_equation(t, states_.col(t - 1));
      observations_.col(t) = sample_from_observation_equation(t, states_.col(t));
    }

  }
  /// Evaluates the log-likelihood as well as the filtering and smoothing moments.
  double evaluate_log_likelihood_and_moments(
    arma::mat& means_filtering,
    arma::cube& variances_filtering,
    arma::mat& means_smoothing,
    arma::cube& variances_smoothing
  ) {
    // NOTE: The likelihood and filtering/smoothing moments are not analytically tractable in this model
    return - std::numeric_limits<double>::infinity();
  }


private:

  unsigned int dx_ = 3;
  unsigned int dy_ = 1;
  unsigned int n_observations_;
  arma::mat observations_;
  arma::mat states_;
  double s_, rho_, beta_, sigma_u_, sigma_v_, delta_;
  arma::mat C0_root_;
  arma::colvec m0_;

  /// Evaluates the mean of the transition equation.
  arma::colvec evaluate_mean(const unsigned int t, const arma::colvec& x) const {
    arma::colvec x_new(dx_);
    x_new(0) = x(0) + s_ * (x(1) - x(0)) * delta_;
    x_new(1) = x(1) + (x(0) * (rho_ - x(2)) - x(1)) * delta_;
    x_new(2) = x(2) + (x(0) * x(1) - beta_ * x(2)) * delta_;
    return x_new;
  }
};

#endif

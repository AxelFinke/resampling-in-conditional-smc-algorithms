#ifndef __HELPER_FUNCTIONS_H
#define __HELPER_FUNCTIONS_H

#include <time.h>
#include <RcppArmadillo.h>

/// Global constant needed for evaluating Gaussian densities.
const double log2pi = std::log(2.0 * M_PI);
/// Samples a single value from a categorical distribution
/// (i.e. outputs a value in {0, ..., length(W)-1}).
unsigned int sample_int(const arma::colvec& W) {
  arma::uvec indices = arma::find(arma::cumsum(W) >= arma::randu(), 1, "first");
  return indices.is_empty() ? W.n_elem - 1 : indices(0);
}
/// Calculates the autorocorrelations in the same was as the function acf() in R.
arma::colvec acf(const arma::colvec& x, unsigned int lag_max) {
  unsigned int n = x.n_elem;
  arma::colvec x_centered = x - arma::mean(x);

  double c0 = arma::dot(x_centered, x_centered) / n;  // lag-0 autocovariance (variance)

  arma::colvec rho(lag_max + 1);
  for (unsigned int k = 0; k <= lag_max; ++k) {
    double ck = arma::dot(x_centered.head(n - k), x_centered.tail(n - k)) / n;
    rho(k) = ck / c0;
  }

  return rho;
}
/// Normalise a single distribution in log-space (returns
/// normalised weights in log space).
arma::vec normalise_exp(const arma::vec& log_w)
{
  double log_w_max = arma::max(log_w);
  double log_Z = log_w_max + std::log(arma::sum(arma::exp(log_w - log_w_max)));
  return(log_w - log_Z);
}
/// Normalises the weights and updates the normalising constant estimate.
arma::vec normalise_exp(double& log_Z, const arma::vec& log_w)
{
  double log_w_max = arma::max(log_w);
  double log_w_sum = log_w_max + std::log(arma::sum(arma::exp(log_w - log_w_max)));
  log_Z += log_w_sum;
  return(log_w - log_w_sum);
}
/// Computes the effective sample size.
double compute_effective_sample_size(const arma::colvec& self_normalised_weights) {
  return 1.0 / arma::dot(self_normalised_weights, self_normalised_weights);
}
/// Evaluates the density of a multivariate normal distribution.
double evaluate_gaussian_density(
  const arma::colvec& x,
  const arma::colvec& mean,
  const arma::mat& sigma,
  bool is_chol = false,
  bool logd = false
) {
  unsigned int dx = x.size();

  arma::mat precision_root; // upper-triangular precision root

  if (is_chol) {
    precision_root = arma::inv(arma::trimatl(sigma));
  } else {
    precision_root = arma::inv(arma::trimatl(arma::chol(sigma, "lower")));
  }

  double rootisum = arma::sum(log(precision_root.diag()));
  double constants = -0.5 * dx * log2pi;
  arma::colvec z = precision_root.t() * (x - mean);
  double out = constants - 0.5 * arma::sum(z % z) + rootisum;

  if (logd) {
    return out;
  } else {
    return std::exp(out);
  }
}
/// Evaluates the density of a multivariate normal distribution
/// under the assumption that the covariance matrix is a scaled identity matrix.
double evaluate_gaussian_density(
  const arma::colvec& x,
  const arma::colvec& mean,
  const double sigma,
  bool is_chol = false,
  bool logd = false
) {
  unsigned int dx = x.size();

  double precision_root; // Inverse root of the covariance matrix

  if (is_chol) {
    precision_root = 1.0 / sigma;
  } else {
    precision_root = 1.0 / std::sqrt(sigma);
  }
  double rootisum = dx * std::log(precision_root);
  double constants = -(static_cast<double>(dx) / 2.0) * log2pi;
  arma::colvec z = precision_root * (x - mean);
  double out = constants - 0.5 * arma::sum(z % z) + rootisum;

  if (logd) {
    return out;
  } else {
    return std::exp(out);
  }
}
/// Runs a Kalman filter
double run_kalman_filter(
  arma::mat& mP,
  arma::cube& CP,
  arma::mat& mU,
  arma::cube& CU,
  const arma::colvec& a,
  const arma::colvec& c,
  const arma::mat& A,
  const arma::mat& B,
  const arma::mat& C,
  const arma::mat& D,
  const arma::colvec& m0,
  const arma::mat& C0,
  const arma::mat& y
)
{
  unsigned int T    = y.n_cols; // number of time steps
  unsigned int dimX = A.n_rows;
  unsigned int dimY = y.n_rows;

  // Predictive mean and covariance matrix
  mP.set_size(dimX, T);
  CP.set_size(dimX, dimX, T);

  // Updated mean and covariance matrix
  mU.set_size(dimX, T);
  CU.set_size(dimX, dimX, T);

  // Mean and covariance matrix of the incremental likelihood
  arma::colvec mY(dimY);
  arma::mat CY(dimY, dimY);

  // Auxiliary quantities
  arma::mat kg;
  arma::mat Q = B * B.t();
  arma::mat R = D * D.t();

  double log_likelihood = 0;

  for (unsigned int t = 0; t < T; ++t) {

    // Prediction step
    if (t > 0) {
      mP.col(t) = A * mU.col(t-1) + a;
      CP.slice(t) = A * CU.slice(t-1) * A.t() + Q;
    }
    else
    {
      mP.col(0) = m0;
      CP.slice(0) = C0;
    }

    // Likelihood step
    mY = C * mP.col(t) + c;
    CY = C * CP.slice(t) * C.t() + R;

    // Update step
    // kg = (arma::solve(CY.t(), C * arma::trans(CP.slice(t)))).t();
    kg = arma::solve(CY, C * CP.slice(t)).t();
    mU.col(t)   = mP.col(t)   + kg * (y.col(t) - mY);
    CU.slice(t) = CP.slice(t) - kg * C * CP.slice(t);

    log_likelihood += evaluate_gaussian_density(y.col(t), mY, CY, false, true);
  }
  return log_likelihood;
}
/// Performs backward Kalman smoothing
void run_kalman_smoother(
  arma::mat& mS,
  arma::cube& CS,
  const arma::mat& mP,
  const arma::cube& CP,
  const arma::mat& mU,
  const arma::cube& CU,
  const arma::mat& A
)
{
  unsigned int T    = mU.n_cols; // number of time steps
  unsigned int dimX = A.n_rows;

  // Smoothed mean and covariance matrix
  mS.set_size(dimX, T);
  CS.set_size(dimX, dimX, T);

  arma::mat Ck(dimX, dimX);
  mS.col(T-1)   = mU.col(T-1);
  CS.slice(T-1) = CU.slice(T-1);

  for (unsigned int t=T-2; t != static_cast<unsigned>(-1); t--)
  {
    // Ck          = CU.slice(t) * A.t() * arma::inv(CP.slice(t+1));
    Ck = arma::solve(CP.slice(t + 1), A * CU.slice(t)).t();
    mS.col(t)   = mU.col(t)   + Ck * (mS.col(t+1)   - mP.col(t+1));
    CS.slice(t) = CU.slice(t) + Ck * (CS.slice(t+1) - CP.slice(t+1)) * Ck.t();
  }
}

#endif

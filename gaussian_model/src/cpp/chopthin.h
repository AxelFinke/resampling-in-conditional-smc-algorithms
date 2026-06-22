// Adapted from the chopthin R package (v0.2.1)
// Original author: Axel Gandy
// Source: https://cran.r-project.org/web/packages/chopthin/index.html
// Original code licensed under the GNU General Public License v3.0
// https://www.gnu.org/licenses/gpl-3.0.html
//
// This file is a derivative work and is distributed under the same license.
// Modifications: extracted and adapted find_a() (Algorithm 2) from
// chopthin_internal.h for standalone use.

#include <Rcpp.h>

int intrand(const int n) { return floor(unif_rand() * n); }
inline double myrunif() { return unif_rand(); }
#define chopthin_error(x) throw Rcpp::exception((x))

#include <vector>
#include <iterator>

// Algorithm 2 from Gandy & Lau (2016): fast determination of $\alpha$.
// Returns $\alpha > 0$ such that $\sum_{n=1}^N h_{\alpha, \beta}(W^n) = M$.
double find_alpha(
  const arma::colvec& W,
  unsigned int M,
  double beta = 5.828427
) {
  std::vector<double> vl = arma::conv_to<std::vector<double>>::from(W);
  std::vector<double> vu = vl;
  std::vector<double>::iterator vli, vui, j;

  double sl = 0, cm = 0, su = 0, cu = 0;
  double a, b;
  double sltmp, sutmp;
  int cutmp, cmtmp;

  while (vl.size() > 0 || vu.size() > 0) {

    if (vl.size() >= vu.size()) {
      a = vl[intrand(vl.size())];
      b = a * beta / 2.0;
    } else {
      b = vu[intrand(vu.size())];
      a = 2.0 * b / beta;
    }

    cmtmp = 0;
    sltmp = 0.0;
    for (vli = vl.begin(); vli != vl.end(); ++vli) {
      if (*vli <= a) sltmp += *vli;
      else cmtmp++;
    }

    sutmp = 0.0;
    cutmp = 0;
    for (vui = vu.begin(); vui != vu.end(); ++vui) {
      if (*vui >= b) { sutmp += (*vui); cutmp++; }
    }

    double h;
    if (a <= 0) {
      if (cm + cmtmp == 0)
        chopthin_error("No positive weights");
      else
        h = M + 1;
    } else {
      h = ((cm + cmtmp) - (cu + cutmp)) + (sl + sltmp) / a + (su + sutmp) / b;
    }

    if (h == M) {
      sl += sltmp; su += sutmp;
      cu += cutmp; cm += cmtmp;
      return a;
    }

    if (h > M) {

      sl += sltmp;

      // Remove elements <= a from vl.
      j = vl.begin();
      for (vli = vl.begin(); vli != vl.end(); ++vli)
        if (*vli > a) { *j = *vli; ++j; }
      vl.resize(j - vl.begin());

      // Remove elements <= b from vu.
      j = vu.begin();
      for (vui = vu.begin(); vui != vu.end(); ++vui)
        if (*vui > b) { *j = *vui; ++j; }
      vu.resize(j - vu.begin());

    } else {

      // Remove elements >= a from vl; cm is incremented
      // for each discarded element.
      j = vl.begin();
      for (vli = vl.begin(); vli != vl.end(); ++vli) {
        if (*vli < a) { *j = *vli; ++j; }
        else cm++;
      }
      vl.resize(j - vl.begin());


      su += sutmp;
      cu += cutmp;

      // Remove elements >= b from vu
      j = vu.begin();
      for (vui = vu.begin(); vui != vu.end(); ++vui)
        if (*vui < b) { *j = *vui; ++j; }
      vu.resize(j - vu.begin());
    }
  }

  return (sl + 2.0 * su / beta) / (M - cm + cu);
}

/// The function $h_{\alpha, \beta}$ used for obtaining the
/// transformed weights according to which chopthin resampling
/// performs systematic resampling.
double h(double W, double alpha, double beta) {
  double b = beta * alpha / 2.0;
  if      (W < alpha)  return W / alpha;
  else if (W < b)  return 1.0;
  else             return 2.0 * W / (beta * alpha);
}
/// Vectorisation of $h$.
arma::colvec h(const arma::colvec& W, double alpha, double beta) {
  unsigned int N = W.size();
  arma::colvec H(N);
  for (unsigned int n = 0; n < N; ++n) {
    H(n) = h(W(n), alpha, beta);
  }
  return H;
}

//
// double h_sum(const arma::colvec& W, double alpha, double eta) {
//   double total = 0.0;
//   for (double w : W)
//     total += h(w, alpha, eta);
//   return total;
// }
//
// double find_a(const arma::colvec& W, double eta, int M) {
//   arma::colvec vl = W;
//   arma::colvec vu = W;
//
//   double sl = 0.0, su = 0.0, cm = 0.0, cu = 0.0;
//   double a, b;
//
//   while (!vl.is_empty() || !vu.is_empty()) {
//
//     if (vl.n_elem >= vu.n_elem) {
//       int idx = arma::randi<arma::ivec>(1, arma::distr_param(0, (int)vl.n_elem - 1))(0);
//       a = vl(idx);
//       b = a * beta / 2.0;
//     } else {
//       int idx = arma::randi<arma::ivec>(1, arma::distr_param(0, (int)vu.n_elem - 1))(0);
//       b = vu(idx);
//       a = 2.0 * b / eta;
//     }
//
//     arma::uvec vl_lo = arma::find(vl <= a);
//     arma::uvec vl_hi = arma::find(vl >  a);
//     double sltmp = arma::sum(vl(vl_lo));
//     double cmtmp = (double)vl_hi.n_elem;
//
//     arma::uvec vu_hi = arma::find(vu >= b);
//     arma::uvec vu_lo = arma::find(vu <  b);
//     double sutmp = arma::sum(vu(vu_hi));
//     double cutmp = (double)vu_hi.n_elem;
//
//     double h = (cm + cmtmp) - (cu + cutmp)
//             + (sl + sltmp) / a
//             + (su + sutmp) / b;
//
//     if (h == M) {
//       sl += sltmp; su += sutmp;
//       cm += cmtmp; cu += cutmp;
//       return a;
//     }
//
//     if (h > M) {
//       sl += sltmp;
//       vl  = vl(vl_hi);
//       vu  = vu(arma::find(vu > b));
//     } else {
//       cm += cmtmp;
//       vl  = vl(arma::find(vl < a));
//       su += sutmp;
//       cu += cutmp;
//       vu  = vu(vu_lo);
//     }
//   }
//
//   return (sl + 2.0 * su / eta) / (M - cm + cu);
// }

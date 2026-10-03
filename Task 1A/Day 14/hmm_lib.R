# Gaussian HMM via Baum-Welch EM, written in base R (no depmixS4/MSwM:
# neither is installable here, CRAN is unreachable from this sandbox -
# checked directly, see r/README.md). This is a genuine cross-language
# check: an independently-written implementation of the same
# forward-backward / Baum-Welch mathematics Day 4's Python frequentist
# HMM (hmmlearn) uses, not a different algorithm standing in for it.
#
# Usage: Rscript hmm_baum_welch.R <K> <n_restarts> <seed>

forward_backward <- function(x, P, pi0, mu, sigma) {
  T <- length(x)
  K <- length(mu)
  log_emit <- matrix(0, T, K)
  for (k in 1:K) log_emit[, k] <- dnorm(x, mean = mu[k], sd = sigma[k], log = TRUE)

  log_alpha <- matrix(0, T, K)
  log_alpha[1, ] <- log(pi0) + log_emit[1, ]
  for (t in 2:T) {
    for (k in 1:K) {
      log_alpha[t, k] <- matrixStats_logSumExp(log_alpha[t - 1, ] + log(P[, k])) + log_emit[t, k]
    }
  }
  log_beta <- matrix(0, T, K)
  for (t in (T - 1):1) {
    for (k in 1:K) {
      log_beta[t, k] <- matrixStats_logSumExp(log(P[k, ]) + log_emit[t + 1, ] + log_beta[t + 1, ])
    }
  }
  loglik <- matrixStats_logSumExp(log_alpha[T, ])
  log_gamma <- log_alpha + log_beta - loglik
  gamma <- exp(log_gamma)

  # filtered probabilities (causal, forward-only, normalised at each step)
  log_filtered <- log_alpha - apply(log_alpha, 1, matrixStats_logSumExp)
  filtered <- exp(log_filtered)

  list(gamma = gamma, filtered = filtered, loglik = loglik, log_alpha = log_alpha,
       log_beta = log_beta, log_emit = log_emit)
}

matrixStats_logSumExp <- function(v) {
  m <- max(v)
  if (!is.finite(m)) return(-Inf)
  m + log(sum(exp(v - m)))
}

fit_hmm_em <- function(x, K, seed, max_iter = 200, tol = 1e-6) {
  set.seed(seed)
  T <- length(x)
  # init: random assignment-based moment init, similar in spirit to
  # hmmlearn's k-means-ish init, not identical
  idx <- sample(1:K, T, replace = TRUE)
  mu <- sapply(1:K, function(k) mean(x[idx == k]))
  sigma <- sapply(1:K, function(k) sd(x[idx == k]))
  sigma[is.na(sigma) | sigma <= 0] <- sd(x)
  P <- matrix(1 / K, K, K) + diag(K) * 0.5
  P <- P / rowSums(P)
  pi0 <- rep(1 / K, K)

  prev_ll <- -Inf
  for (iter in 1:max_iter) {
    fb <- tryCatch(forward_backward(x, P, pi0, mu, sigma), error = function(e) NULL)
    if (is.null(fb) || !is.finite(fb$loglik)) return(NULL)

    gamma <- fb$gamma
    log_alpha <- fb$log_alpha; log_beta <- fb$log_beta; log_emit <- fb$log_emit

    # xi: expected transition counts
    xi_sum <- matrix(0, K, K)
    for (t in 1:(T - 1)) {
      log_xi_t <- matrix(0, K, K)
      for (i in 1:K) for (j in 1:K) {
        log_xi_t[i, j] <- log_alpha[t, i] + log(P[i, j]) + log_emit[t + 1, j] + log_beta[t + 1, j] - fb$loglik
      }
      xi_sum <- xi_sum + exp(log_xi_t)
    }
    P_new <- xi_sum / rowSums(xi_sum)
    pi0_new <- gamma[1, ]
    mu_new <- colSums(gamma * x) / colSums(gamma)
    sigma_new <- sqrt(colSums(gamma * (x - matrix(mu_new, T, K, byrow = TRUE))^2) / colSums(gamma))
    sigma_new[sigma_new < 1e-6] <- 1e-6

    P <- P_new; pi0 <- pi0_new; mu <- mu_new; sigma <- sigma_new

    if (abs(fb$loglik - prev_ll) < tol) break
    prev_ll <- fb$loglik
  }
  list(P = P, pi0 = pi0, mu = mu, sigma = sigma, loglik = fb$loglik,
       filtered = fb$filtered, smoothed = fb$gamma, n_iter = iter)
}

fit_hmm_robust <- function(x, K, n_restarts = 10, base_seed = 42) {
  best <- NULL
  best_ll <- -Inf
  n_valid <- 0
  for (r in 1:n_restarts) {
    fit <- fit_hmm_em(x, K, seed = base_seed + r)
    if (is.null(fit)) next
    # reject collapsed states (< 1% occupancy by hard assignment), same
    # robustness criterion Day 4's Python fit_regime_hmm_robust uses
    hard <- apply(fit$smoothed, 1, which.max)
    occ <- table(factor(hard, levels = 1:K)) / length(hard)
    if (any(occ < 0.01)) next
    n_valid <- n_valid + 1
    if (fit$loglik > best_ll) { best <- fit; best_ll <- fit$loglik }
  }
  if (is.null(best)) stop("All restarts produced a collapsed state or failed to converge")
  list(fit = best, n_valid = n_valid, n_restarts = n_restarts)
}


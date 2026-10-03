# Split-conformal classification and Monte Carlo VaR/CVaR in base R.
# No `conformal` or `mlr3` package: CRAN is unreachable from this
# sandbox (see r/README.md). Both algorithms are simple and well
# defined enough to implement directly; this is, again, a genuine
# cross-language check of the same mathematics (Section A6.2, A13.2),
# not a package-equivalence check.

split_conformal_r <- function(cal_probs, y_cal, test_probs, alpha = 0.1) {
  # q_hat computed as the exact order statistic numpy's quantile(...,
  # method="higher") uses, not R's generic quantile(type=...)
  # interpolation: found a real, if modest (~0.01), discrepancy between
  # the two when first cross-checked against Python (q_hat 0.9843 here
  # vs 0.9934 there on the same Day 11 data), traced to this difference
  # in interpolation convention rather than a deeper algorithmic
  # mismatch, and fixed by matching numpy's formula directly:
  # sorted_scores[ceil(q_level * (n-1))] (0-indexed; +1 below for R's
  # 1-indexing).
  n <- length(y_cal)
  cal_scores <- 1 - cal_probs[cbind(1:n, y_cal)]
  q_level <- min(ceiling((n + 1) * (1 - alpha)) / n, 1.0)
  sorted_scores <- sort(cal_scores)
  idx <- ceiling(q_level * (n - 1)) + 1  # +1 for R's 1-indexing
  q_hat <- sorted_scores[idx]
  pred_sets <- test_probs >= (1 - q_hat)
  list(pred_sets = pred_sets, q_hat = q_hat)
}

regime_var_r <- function(paths, alpha = 0.05) {
  final <- paths[, ncol(paths)] - 1.0
  var <- as.numeric(quantile(final, alpha))
  cvar <- mean(final[final <= var])
  list(var = var, cvar = cvar)
}

simulate_mc_r <- function(P, mu, sigma, init_dist, horizon = 252, n_sims = 10000, seed = 42) {
  set.seed(seed)
  K <- length(mu)
  s0 <- sample(1:K, n_sims, replace = TRUE, prob = init_dist)
  states <- matrix(0L, n_sims, horizon)
  states[, 1] <- s0
  for (t in 2:horizon) {
    probs <- P[states[, t - 1], , drop = FALSE]
    cum <- t(apply(probs, 1, cumsum))
    u <- runif(n_sims)
    states[, t] <- max.col(u < cum, ties.method = "first")
  }
  mu_t <- matrix(mu[states], n_sims, horizon)
  sigma_t <- matrix(sigma[states], n_sims, horizon)
  eps <- matrix(rnorm(n_sims * horizon), n_sims, horizon)
  rets <- mu_t + sigma_t * eps
  paths <- exp(t(apply(log1p(rets), 1, cumsum)))
  list(paths = paths, states = states)
}

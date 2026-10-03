# Runs all three Python/R cross-validation checks (HMM, split-conformal,
# Monte Carlo VaR/CVaR) and writes cross_validation_results.json.
# Expects Python's comparison inputs already exported to this directory
# (see scripts/run_day14_crossval_export.py).
library(jsonlite)
source("hmm_lib.R")
source("conformal_and_var.R")

results <- list()

# --- 1. HMM ---
data <- read.csv("nifty_returns_full.csv")
x <- data$return
hmm_result <- fit_hmm_robust(x, K = 2, n_restarts = 10, base_seed = 42)
fit <- hmm_result$fit
write.csv(data.frame(date = data$X, fit$filtered), "r_hmm_filtered_k2.csv", row.names = FALSE)
results$hmm <- list(
  n_valid_restarts = hmm_result$n_valid, n_restarts = hmm_result$n_restarts,
  loglik = fit$loglik, mu_ann = as.list(fit$mu * 252), sigma_ann = as.list(fit$sigma * sqrt(252)),
  P_diag = as.list(diag(fit$P))
)
cat(sprintf("[1/3] HMM: R loglik=%.2f (K=2, %d restarts, %d valid)\n",
            fit$loglik, hmm_result$n_restarts, hmm_result$n_valid))

# --- 2. Split-conformal ---
cal_probs <- as.matrix(read.csv("cal_probs.csv", header = FALSE))
y_cal <- as.integer(read.csv("y_cal.csv", header = FALSE)[, 1])
eval_probs <- as.matrix(read.csv("eval_probs.csv", header = FALSE))
y_eval <- as.integer(read.csv("y_eval.csv", header = FALSE)[, 1])
cc <- split_conformal_r(cal_probs, y_cal, eval_probs, alpha = 0.1)
covered <- sapply(1:nrow(eval_probs), function(i) cc$pred_sets[i, y_eval[i]])
results$conformal <- list(q_hat = cc$q_hat, coverage = mean(covered),
                           mean_set_size = mean(rowSums(cc$pred_sets)))
cat(sprintf("[2/3] Split-conformal: R q_hat=%.6f coverage=%.4f mean_size=%.4f\n",
            cc$q_hat, mean(covered), mean(rowSums(cc$pred_sets))))

# --- 3. Monte Carlo VaR/CVaR ---
P <- as.matrix(read.csv("P_matrix.csv", header = FALSE))
mc_spec <- fromJSON("mc_spec.json")
mu <- mc_spec$drift_ann / 252
sigma <- mc_spec$vol_ann / sqrt(252)
init_dist <- mc_spec$init_dist
mc <- simulate_mc_r(P, mu, sigma, init_dist, horizon = 252, n_sims = 10000, seed = 42)
vc <- regime_var_r(mc$paths, alpha = 0.05)
final <- mc$paths[, 252] - 1.0
results$monte_carlo <- list(
  var_5pct = vc$var, cvar_5pct = vc$cvar, prob_negative = mean(final < 0),
  p90_lo = as.numeric(quantile(final, 0.05)), p90_hi = as.numeric(quantile(final, 0.95))
)
cat(sprintf("[3/3] Monte Carlo: R VaR=%.4f CVaR=%.4f P(neg)=%.4f\n", vc$var, vc$cvar, mean(final < 0)))

write(toJSON(results, auto_unbox = TRUE, pretty = TRUE), "cross_validation_results.json")
cat("DONE\n")

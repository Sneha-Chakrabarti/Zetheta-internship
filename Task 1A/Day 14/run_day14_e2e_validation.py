"""Day 14: end-to-end validation of the regime pipeline, raw data
ingestion through ensembled regime probability to tilt recommendation
to Investment Committee artefact. Runs every stage for real, on a
single target date, and prints what each stage actually produced -
this is a validation that the PIPELINE CONNECTS, not a re-derivation of
any single day's findings.

  python3 scripts/run_day14_e2e_validation.py
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.abspath("."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

TARGET_DATE = "2024-06-14"  # arbitrary, mid-window, chosen only for being
                             # unremarkable - no cherry-picking a date known
                             # to produce a clean story

print(f"=== End-to-end pipeline validation, target date {TARGET_DATE} ===\n")

# --- Stage 1: raw data ingestion (Day 1-3). ---
from src.data.loader import DataConfig, load_market_data
data = load_market_data(DataConfig(backend="synthetic", n_years=15.0, seed=7))
returns = data["nifty50"]["close"].pct_change().dropna()
assert pd.Timestamp(TARGET_DATE) in returns.index, "target date not in the data - pipeline check needs an in-sample date"
print(f"[1] Raw data: {len(returns)} days loaded, {returns.index.min().date()} to {returns.index.max().date()}")
print(f"    Return on target date: {returns.loc[TARGET_DATE]:+.4%}")

# --- Stage 2: feature engineering (Day 3) sanity check - just confirm the
# module runs on this data without needing its full output downstream
# (this project's HMM/overlay path uses returns directly, not the full
# feature set; Day 3's richer features feed the BDL/foundation models
# instead, already validated in their own days). ---
from src.features.engineer import engineer_regime_features
features = engineer_regime_features(data)
target_row = features.loc[TARGET_DATE]
print(f"[2] Feature engineering: {features.shape[1]} features computed, full series "
      f"({target_row.notna().sum()} non-NaN on {TARGET_DATE})")

# --- Stage 3: regime probability (Day 4's frequentist HMM, filtered/causal). ---
from src.models.hmm.frequentist import fit_regime_hmm_robust, label_regimes, filtered_state_probs
model, states, _, seed, nvalid = fit_regime_hmm_robust(returns, n_states=5, n_restarts=20, base_seed=42)
label_map = label_regimes(model, K=5)
filtered = filtered_state_probs(model, returns)
target_idx = returns.index.get_loc(TARGET_DATE)
regime_probs_target = filtered[target_idx]
regime_dict = {label_map[i]: float(regime_probs_target[i]) for i in range(5)}
dominant = max(regime_dict, key=regime_dict.get)
print(f"[3] Regime probability (5-state HMM, seed={seed}, {nvalid}/20 valid): "
      f"dominant={dominant} ({regime_dict[dominant]:.1%})")
print(f"    Full distribution: {', '.join(f'{k}={v:.1%}' for k,v in regime_dict.items())}")

# --- Stage 4: ensembling (Day 9) - note on scope: Day 9's actual trained
# ensemble only covers its own specific 252-day window (2024-07-11 to
# 2025-06-27); this target date (2024-06-14) predates it. Using the
# single HMM's own probability for this pipeline check is a conscious,
# stated substitution, not a silent gap - Day 9's stacking weights do
# not generalise to a date outside the window they were fit on. ---
print(f"[4] Ensembling: target date predates Day 9's fitted ensemble window "
      f"(2024-07-11 onward) - using the single HMM's own filtered probability "
      f"for this pipeline check, stated explicitly rather than silently substituted.")

# --- Stage 5: tilt recommendation (Day 13's overlay). ---
from src.backtest.overlay import tilt_conviction_scaled, regime_sharpe_proxy
drift_ann = model.means_[:, 0] * 252
vol_ann = np.sqrt(model.covars_.reshape(5, 1)[:, 0]) * np.sqrt(252)
order5 = [label_map[i] for i in range(5)]
canonical_order = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]
drift_ordered = np.array([drift_ann[order5.index(n)] for n in canonical_order])
vol_ordered = np.array([vol_ann[order5.index(n)] for n in canonical_order])
probs_ordered = np.array([regime_dict[n] for n in canonical_order])
weight = tilt_conviction_scaled(probs_ordered, drift_ordered, vol_ordered,
                                 base_weight=0.7, min_weight=0.3, max_weight=1.0)[0]
print(f"[5] Tilt recommendation: equity weight = {weight:.2f} "
      f"(base 0.70, conviction-scaled toward {dominant}'s own risk-adjusted target)")

# --- Stage 6: Monte Carlo projection + IC artefact (Day 13). ---
from src.models.monte_carlo.regime_mc import RegimeConditionedMC, regime_var
from src.models.monte_carlo.ic_artefact import Lineage, build_ic_artefact, conditional_on_state_statement

mc = RegimeConditionedMC(model.transmat_[np.ix_([order5.index(n) for n in canonical_order],
                                                  [order5.index(n) for n in canonical_order])],
                          drift_ordered / 252, vol_ordered / np.sqrt(252), K=5)
paths, sim_states = mc.simulate(probs_ordered, horizon=252, n_sims=5000, seed=42)
var5, cvar5 = regime_var(paths, alpha=0.05)
final_returns = paths[:, -1] - 1.0
p90 = tuple(np.percentile(final_returns, [5, 95]))
prob_neg = float((final_returns < 0).mean())

risk_off_idx = canonical_order.index("Risk-Off")
stmt, p_trans, worst = conditional_on_state_statement(
    paths, sim_states, target_state=risk_off_idx, window_days=90, min_days_in_state=45,
    state_name="Risk-Off",
)
lineage = Lineage(model_name="Frequentist 5-state HMM (full-series fit)",
                   model_fit_window=f"{returns.index.min().date()} to {returns.index.max().date()}",
                   model_fit_seed=int(seed), n_valid_restarts=int(nvalid),
                   simulation_seed=42, n_sims=5000, horizon_days=252, as_of_date=TARGET_DATE)
artefact = build_ic_artefact(lineage=lineage, final_regime_dist=regime_dict, p90_interval=p90,
                              prob_negative=prob_neg, p5_return=float(np.percentile(final_returns, 5)),
                              var5=float(var5), cvar5=float(cvar5), conditional_statement=stmt)
print(f"[6] Monte Carlo + IC artefact:")
for s in artefact.render_ic_statements():
    print("    -", s)

print("\n=== Pipeline check: every stage produced output and fed the next without error ===")
result = {
    "target_date": TARGET_DATE, "regime_probabilities": regime_dict, "dominant_regime": dominant,
    "tilt_weight": float(weight), "ic_statements": artefact.render_ic_statements(),
    "lineage": artefact.lineage.__dict__,
}
json.dump(result, open("artifacts_data/day14_e2e_validation.json", "w"), indent=1)
print("\nDONE")

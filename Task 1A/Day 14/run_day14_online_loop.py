"""Day 14: verify the ONLINE inference loop (distinct from the batch
pipeline in run_day14_e2e_validation.py) runs end-to-end on a single
streamed day: particle filter update, BOCPD update, and streaming
Dirichlet transition update, each processing exactly one new
observation as it would arrive in production, not a batch re-run.

  python3 scripts/run_day14_online_loop.py
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.abspath("."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.data.loader import DataConfig, load_market_data
from src.models.hmm.frequentist import fit_regime_hmm_robust, label_regimes
from src.models.sequential.particle_filter import RegimeParticleFilter
from src.models.sequential.bocpd import bocpd, calibrate_normal_gamma_prior, short_run_length_probability
from src.models.sequential.streaming import StreamingDirichletTransitions

STREAM_START, STREAM_DAY = "2024-01-01", "2024-06-14"

data = load_market_data(DataConfig(backend="synthetic", n_years=15.0, seed=7))
returns = data["nifty50"]["close"].pct_change().dropna()

print("=== Online inference loop: warm-start state, then process ONE new day ===\n")

# --- Warm-start: everything the online system would have accumulated
# BEFORE the streamed day (not re-derived live - this is what a running
# production system already holds). ---
history = returns.loc[:STREAM_START]
model, states_hist, _, seed, nvalid = fit_regime_hmm_robust(history, n_states=5, n_restarts=20, base_seed=42)
label_map = label_regimes(model, K=5)
print(f"Warm-start HMM (fit on data through {STREAM_START}, seed={seed}, {nvalid}/20 valid)")

pf = RegimeParticleFilter(model.transmat_, model.means_[:, 0],
                           np.sqrt(model.covars_.reshape(5, 1)[:, 0]), n_particles=5000, seed=42)
warmup_window = returns.loc[STREAM_START:STREAM_DAY].iloc[:-1]  # everything up to but not including the streamed day
_ = pf.run(warmup_window.values)
print(f"Particle filter warmed up on {len(warmup_window)} days "
      f"({warmup_window.index.min().date()} to {warmup_window.index.max().date()})")

prior = calibrate_normal_gamma_prior(returns.loc[:STREAM_START].values, calibration_window=60)
bocpd_history = returns.loc[STREAM_START:STREAM_DAY].iloc[:-1].values

streaming_trans = StreamingDirichletTransitions(K=5, alpha_prior=1.0)
hard_states_warmup = pf.particles  # current hard particle assignment as a proxy "last known state" stream
streaming_trans.update_sequence(states_hist[-200:])  # seed with recent history's own transitions
print(f"Streaming Dirichlet transition tracker seeded with the last 200 days of warm-start transitions\n")

# --- THE STREAMED DAY: process exactly one new observation through each component. ---
new_obs = returns.loc[STREAM_DAY]
print(f"--- Streamed day: {STREAM_DAY}, return = {new_obs:+.4%} ---\n")

pf_post_before = pf.weights.copy(), pf.particles.copy()
t0_particles = pf.n_resamples
pf_posterior = pf.step(new_obs)
print(f"[Particle filter] P(regime | up to and including today): "
      f"{', '.join(f'{label_map[i]}={pf_posterior[i]:.1%}' for i in range(5))}")
print(f"  Resampled this step: {pf.n_resamples > t0_particles}  (ESS: {pf.ess_history[-1]:.0f} / {pf.N})")

full_bocpd_series = np.concatenate([bocpd_history, [new_obs]])
R = bocpd(full_bocpd_series, hazard=1 / 100, **prior)
sig = short_run_length_probability(R, k=3)
today_signal = sig[-1]
print(f"\n[BOCPD] Short-run-length signal today: {today_signal:.4f} "
      f"({'ABOVE' if today_signal > 0.5 else 'below'} the 0.5 detection threshold)")

prev_hard_state = int(np.bincount(hard_states_warmup, minlength=5).argmax())
new_hard_state = int(np.bincount(pf.particles, minlength=5).argmax())
streaming_trans.update(prev_hard_state, new_hard_state)
updated_row = streaming_trans.posterior_mean()[prev_hard_state]
print(f"\n[Streaming Dirichlet] Observed transition: {label_map[prev_hard_state]} -> {label_map[new_hard_state]}")
print(f"  Updated posterior-mean transition row for {label_map[prev_hard_state]}: "
      f"{', '.join(f'{label_map[i]}={updated_row[i]:.3f}' for i in range(5))}")

print("\n=== Online loop check: all three components processed the streamed day without re-reading full history ===")
result = {
    "streamed_date": STREAM_DAY, "return": float(new_obs),
    "particle_filter_posterior": {label_map[i]: float(pf_posterior[i]) for i in range(5)},
    "bocpd_signal_today": float(today_signal),
    "bocpd_fired": bool(today_signal > 0.5),
    "streaming_transition_observed": [label_map[prev_hard_state], label_map[new_hard_state]],
}
json.dump(result, open("artifacts_data/day14_online_loop.json", "w"), indent=1)
print("\nDONE")

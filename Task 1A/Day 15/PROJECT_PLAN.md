# Project plan and status

Tracks progress against Part D (Section D2 day-by-day breakdown, Section D3
mandatory deliverables) of the task document. Updated as each piece lands.

Status key: `[ ]` not started, `[~]` in progress, `[x]` done.

## Day-by-day

- [~] Day 1: Domain orientation, environment setup
      - [x] Repo structure, README, requirement tiers, environment.yml
      - [x] Environment confirmation: core tier installed, 5/5 smoke tests
            pass (docs/day1_environment_confirmation.md)
      - [x] Data layer built: CSVBackend + synthetic regime-switching
            backend behind one `load_market_data()` interface
      - [ ] Data source confirmed (see "Open decisions" below)
- [x] Day 2: Data ingestion and exploration
      - [x] `notebooks/01_data_exploration.ipynb`: basic stats, price paths,
            return distributions, VIX path, FII/DII flows, rolling
            correlations, real-data cross-check (executed, 0 errors)
      - [x] `docs/day2_data_quality_report.md`: written report
      - [x] Found and fixed a volatility-ordering bug in the synthetic
            generator while checking notebook output against a claim
            before writing it (see report)
- [x] Day 3: Feature engineering (incl. TDA + GCN features)
      - [x] `src/features/engineer.py`: 31 tabular features, 7 groups
      - [x] `src/features/topology.py`: TDA persistence landscapes (A7.2)
      - [x] `src/features/sector_gnn.py`: segment-graph GCN embeddings (A7.3)
      - [x] `notebooks/02_feature_engineering.ipynb` (executed, 0 errors)
      - [x] `docs/day3_feature_engineering_notes.md`: what's implemented,
            what's skipped and why (breadth, real rates, DXY-INR need
            data outside the required panel), two claims that looked true
            visually but were checked numerically and one turned out
            partly wrong (see notes)
      - [x] 13/13 tests passing (`tests/test_features.py`)
- [x] Day 4: Frequentist HMM (hmmlearn)
      - [x] `src/models/hmm/frequentist.py`: fit, robust multi-restart fit,
            post-hoc labelling, duration/transition stats, BIC comparison
      - [x] `notebooks/03_frequentist_hmm.ipynb` (executed, 0 errors)
      - [x] `docs/day4_hmm_notes.md`: task doc's exact single-seed recipe
            demonstrably degenerates (2 redundant states + 1 empty state);
            multi-restart fitting added as the fix; BIC actually prefers
            3 states over 5, 7 states is unidentifiable (0/50 restarts);
            cross-checked against synthetic ground truth (42.2% exact
            match vs. 20% random) - all reported honestly, not smoothed
      - [x] 21/21 tests passing (`tests/test_hmm_frequentist.py` added)
- [x] Day 5: Bayesian HMM (PyMC)
      - [x] Found and proved Section A3.4's example code doesn't
            implement an HMM (P's posterior == its prior, verified
            numerically); fixed via forward-algorithm marginalisation
            (`src/models/hmm/bayesian.py`)
      - [x] Discovered background processes don't survive between tool
            calls in this sandbox; real timing forced a documented scope
            reduction (1-year window, 500+500 draws vs. spec's 15-year/
            2000+1000) - see docs/day5_bayesian_hmm_notes.md
      - [x] `notebooks/04_bayesian_hmm.ipynb` (executed on the
            `bayesian-env` kernel, 0 errors): R-hat ~1.00, 2/2000
            divergences on the reduced-budget run
      - [x] Ground truth check: Bayesian match rate (21.0%) is
            statistically indistinguishable from random (0.4 SE);
            frequentist full-window-sliced (29.4%) is genuinely better
            (3.7 SE) - cleaner MCMC diagnostics did not mean better
            regime recovery, caught before writing the opposite claim
      - [x] 6/6 tests passing (4 numpy-only in main env
            `tests/test_hmm_bayesian.py`, 2 pymc-dependent in
            `.venv-bayesian` `tests/test_hmm_bayesian_pymc.py`)
- [x] Day 6: Regime-switching VAR, univariate + multivariate
      - [x] `src/models/rsvar/baseline.py` (A8.2, statsmodels): found the
            same multi-seed instability pattern as Day 4/5 (4/5 seeds
            agree, 1 collapses); fixed statsmodels' column- vs.
            row-stochastic transition matrix mismatch; handled a boundary
            transition-probability crash gracefully
      - [x] `src/models/rsvar/bayesian_rsvar.py` (A8.3): found and fixed
            two real spec-code gaps (`pm.HiddenMarkovChain` doesn't
            exist; `LKJCholeskyCov(shape=(K,))` doesn't batch, verified
            both directly) via forward-algorithm marginalisation +
            K independent LKJCholeskyCov calls
      - [x] Found catastrophic R-hat (1.4-2.4, 0 divergences) after
            sampling, diagnosed as label-switching across chains, fixed
            with per-draw post-hoc relabelling (R-hat -> 1.00-1.18)
      - [x] Regime-conditional impulse responses + covariance with
            credible intervals delivered; intervals too wide for
            confident regime-dependent claims at this compute budget,
            reported honestly rather than emphasising point estimates
      - [x] Ground truth: 32.9% match, chance level (33.3% baseline) -
            attributed to compute-budget-limited posterior resolution
            (320 draws on a 3-regime, 6-d VAR), not a refutation of
            the approach
      - [x] `notebooks/05_rsvar.ipynb` (executed on `bayesian-env`, 0
            errors) + `docs/day6_rsvar_notes.md`
      - [x] 43/43 tests passing (36 main env incl. 5 numpy-only RS-VAR,
            2 pymc-dependent skipped there; 2 pymc-dependent pass in
            `.venv-bayesian`)
- [x] Day 7: Bayesian deep learning (MC dropout, variational BNN, ensemble)
      - [x] `src/models/bdl/`: MC Dropout (A4.2), variational BNN (A4.3),
            deep ensemble M=10 (A4.4), evaluation and data helpers;
            TensorFlow and TFP in `.venv-bayesian`
      - [x] 5-fold time-series CV with per-fold scaling, reported against
            majority and class-prior baselines. Result: no model beats
            "always Risk-On" on accuracy; the variational BNN beats the
            class prior on NLL in 5 of 5 folds; architectures not
            separable
      - [x] Found and fixed: ensemble members had active dropout (not
            deterministic); the spec's 30-epoch ensemble recipe overfits
            (worse than the class prior in 3 of 5 folds; fixed with
            validation-split early stopping); scaler look-ahead leakage;
            a double weight-sampling bug in `vi_predict`
      - [x] Epistemic/aleatoric decomposition (spec and entropy versions
            agree, Spearman 0.998 and 1.0); uncertainty detects wrong
            calls with AUROC 0.84 to 0.86 but about one in five missed
            stress days is missed confidently
      - [x] Synthetic true-regime check: entropy and soft stress
            probability both track true stress (AUROC 0.77 to 0.93),
            neither better; rests on only two stress episodes
      - [x] SHAP for five rule-chosen dates, additivity verified
      - [x] `notebooks/06_bayesian_deep_learning.ipynb` (0 errors),
            `docs/day7_bdl_notes.md`
      - [x] Tests: 11 evaluation/data tests (main env) plus 6 TensorFlow
            tests (`.venv-bayesian`); full main-env suite 44 passed,
            6 skipped
- [x] Day 8: Foundation model integration (Chronos + one more), as
      synthetic-pretrained re-implementations, NOT the real checkpoints
      - [x] `src/models/foundation/`: Chronos-style tokenised LM (116k
            parameters) and TimesFM-style patch decoder with quantile
            heads (105k), pretrained on a nine-family synthetic corpus;
            variational Bayesian head and `HybridRegimeModel` (A5.4)
      - [x] Pretraining learned real structure on trend, seasonal and
            level-shift series but almost none on return-like series;
            on the project's returns both match static baselines
      - [x] Sample-efficiency comparison, 7 training sizes, 3 subsets
            each. Pretrained embeddings rank true stress at AUROC about
            0.875 vs about 0.5 for the same architectures untrained. No
            advantage over engineered features (they are higher at n=50
            and n=100, level at large n). Heads on embeddings are
            poorly calibrated (NLL 0.3 above a class prior)
      - [x] Control: pretraining the TimesFM-style model without the
            regime-switching family gives the same AUROC (0.876). Not
            run for the Chronos-style model
      - [x] A one-line scale-free volatility ratio (AUROC 0.93 to 0.96)
            beats the embeddings (0.875)
      - [x] `notebooks/07_foundation_models.ipynb` (0 errors),
            `docs/day8_foundation_notes.md`
      - [x] Tests: 6 corpus tests (main env) plus 8 TensorFlow tests
            including causality and a target-leakage guard
            (`.venv-bayesian`)
- [x] Day 9: Model ensembling, stacking, WAIC/LOO selection
      - [x] Built filtered (causal) probability + pointwise log-likelihood
            functions for all three HMM/RS-VAR modules (none existed
            before; smoothed-only was leaking the future into any
            out-of-fold claim). Each verified: matches smoothing exactly
            at the final timestep, differs before it; pointwise log-lik
            sums to the model total
      - [x] `src/models/ensemble/`: labels (3-way collapse, forced by
            Day 6's RS-VAR being K=3-only), combine (BMA + stacking,
            A10.1/A10.2, one fix: explicit class count, the spec's
            inference from y_true silently breaks on an absent rare
            class), contract (A10.4 schema, unbuilt Day 10/11 fields
            explicitly None, not omitted)
      - [x] Assembled 7 members (Day 4-6 HMM/RS-VAR family, Day 7's 3
            BDL models, a re-run Day 8 foundation head) on a common
            252-day window; 3 of 7 are NOT genuinely out-of-fold on it
            (fit using data that includes this window) - stated
            prominently, not buried
      - [x] BMA/stacking evaluated via real 4-fold TimeSeriesSplit.
            A10.4's "ensemble beats every member" claim holds in 1 of 4
            tested cases (stacking vs. HMM labels, narrowly); against
            true regime, naive equal-weighting beats both BMA and
            stacking - reported as found, not smoothed over
      - [x] PSIS-LOO comparison (A10.3): installed ArviZ has dropped
            WAIC entirely (checked directly, hasattr false); RS-VAR's
            own LOO flagged unreliable (p_loo~205 near its ~197 raw
            parameters, 19/252 Pareto-k points bad/very-bad),
            independently corroborating Day 6's compute-limited-posterior
            finding from a different angle
      - [x] `notebooks/08_ensembling.ipynb` (0 errors, all claims
            verified against printed output) + `docs/day9_ensembling_notes.md`
      - [x] 77 passed, 7 skipped (main env, incl. 27 new tests for
            labels/combine/contract/filtered-probs/pointwise-loglik)
- [x] Day 10: Sequential/online inference (particle filter, BOCPD)
      - [x] `src/models/sequential/particle_filter.py`: spec's
            RegimeParticleFilter with propagation vectorised (spec's own
            per-particle rng.choice would need ~18.5M calls at N=5000
            over the full series, checked impractical). Validated
            against the exact forward algorithm (Day 9) on real data:
            mean TV=0.0095, 99.2% hard-label agreement
      - [x] `src/models/sequential/bocpd.py`: found and fixed two real
            bugs in the spec's own example. (1) Default priors imply
            variance 1.0, ~4 orders of magnitude above real return
            variance, silently defeating detection - fixed with
            `calibrate_normal_gamma_prior`. (2) `R[0,t]` (the obvious
            "changepoint probability" reading) is PROVEN (algebraically
            + numerically on random inputs) to equal the hazard rate
            exactly regardless of data - a pure identity of the
            recursion, not a detection signal. Corrected signals
            (`map_run_length`, `short_run_length_probability`) cleanly
            detect toy breaks with zero false positives
      - [x] Real-data validation: 3 of 5 detected event clusters on the
            real partial Nifty data independently confirmed via search
            (Aug 2015 Black Monday - already found Day 2; Nov 2016
            demonetisation; Sep 2019 corporate tax cut, "largest
            single-day Nifty gain in a decade"). Low recall (4.3%) on
            synthetic ground truth explained via Day 4's weak-state-
            separability finding, not left unexplained
      - [x] `src/models/sequential/streaming.py`: Dirichlet/Beta
            streaming updates, proved exact streaming==batch equivalence
            and order-invariance. Found a real, explained 0.196 gap vs.
            hmmlearn's own soft-EM transmat_ (hard-Viterbi-count vs.
            soft-EM estimators disagree on a low-occupancy state)
      - [x] Online/batch reconciliation diagnostic: the expected
            monotonic "staleness grows" pattern does NOT hold (checked,
            not assumed) - TV distance peaks at 40-80 days, not at the
            far end; traced to BOCPD events and HMM belief-revision
            being governed by different, poorly-correlated processes
      - [x] `notebooks/09_sequential_inference.ipynb` (0 errors, all
            claims verified against printed output; main `python3`
            kernel, no PyMC/TF needed) + `docs/day10_sequential_notes.md`
      - [x] 100 passed, 7 skipped (main env, incl. 23 new tests across
            particle filter, BOCPD, streaming)
- [x] Day 11: Conformal prediction and calibration
      - [x] `src/models/conformal/split_conformal.py`: split-conformal
            (A6.2) and APS (A6.3), verified over 50 trials that marginal
            coverage holds on average (not assumed from one split).
            Confirmed directly (not just cited) that plain
            split-conformal produces genuinely empty sets ~10% of the
            time, matching A6.3's own stated warning
      - [x] `src/models/conformal/mondrian.py`: class-conditional
            conformal (A6.5's required distribution-shift-robust
            method), chosen specifically to target the ~89%-one-class
            imbalance established since Day 7. Real coverage ranged
            0.53-1.00 on actual model data; traced to two causes (tiny
            per-class calibration samples, AND a genuine measured
            temporal shift in one model's own calibration quality for
            its majority class, mean nonconformity 0.008 in calibration
            vs. 0.163 in evaluation) rather than reported as one
            unexplained number
      - [x] `src/models/conformal/aci.py`: Adaptive Conformal Inference
            (A6.5). Found and verified a real limitation by deliberately
            testing under severe synthetic shift: alpha_t cannot push
            q_hat past max(calibration scores), so online adaptation
            alone cannot rescue coverage under severe enough shift
            (confirmed: coverage collapsed to 0.027 vs 0.9 target even
            as alpha_t saturated near its floor)
      - [x] `src/models/conformal/diagnostics.py`: reliability
            diagrams + ECE (A6.6). Found ECE against true regime is
            6-10x ECE against HMM training labels for every model,
            stated as a reminder that a calibration number is only
            meaningful relative to its label source
      - [x] Applied to 5 real models (Day 7's 3 BDL models, Day 9's
            foundation head, an equal-weight ensemble) on the 702-day
            held-out test window, both label sources
      - [x] Rolling 252-day coverage: base (uncalibrated top-1) swings
            0.37-1.00 over the window; all 3 conformal methods stay
            within 0.90-1.00 across the same window - the headline value
            proposition demonstrated on real data, not asserted
      - [x] `notebooks/10_conformal_calibration.ipynb` (0 errors, all
            claims verified against printed output, one caught and
            corrected mid-build) + `docs/day11_conformal_notes.md`
      - [x] 125 passed, 7 skipped (main env, incl. 25 new tests across
            split-conformal, Mondrian, ACI, calibration diagnostics)
- [x] Day 12: Case study research and replication (Part C)
      - [x] Chose 4 of 5 case studies (COVID 2020, IL&FS 2018, Taper
            Tantrum 2013, Election 2024); dropped the 2017-18 midcap
            case explicitly, its lesson substantially duplicates
            IL&FS, rather than silently omitting one
      - [x] Cross-checked every figure against independent sources
            rather than copying the task doc's own numbers. Found and
            reported real discrepancies: VIX peak (doc: 86.6; real
            data: 83.61), VIX on 28 Feb 2020 (doc: "above 25"; real
            data: 23.24)
      - [x] Central finding: re-ran Day 10's BOCPD on real data it had
            never been tested against. Confirmed (via Day 10's own
            prior results) it never fires near the actual IL&FS
            default date; explained why directly from real Nifty
            returns (gradual 2-month grind, daily moves within ~1.5
            std of normal, not a sharp break)
      - [x] Ran BOCPD fresh on real VIX for COVID and the 2024
            election: misses both at Day 10's validated 0.5 threshold.
            Found and explained the COVID miss precisely: the 60-day
            calibration window runs directly through the crisis onset,
            calibrating "normal" using the crisis itself. A lower,
            unvalidated threshold (0.10) catches both; quantified its
            cost (5.3% baseline flag rate) and reported it as
            unvalidated, not as a fix
      - [x] 4 real-data-grounded figures + 12-page PDF report (pandoc +
            xelatex; fixed a missing lmodern LaTeX package along the
            way) at `docs/day12_case_studies.{md,pdf}`. Every specific
            numeric claim re-verified against actual computed output
            before rendering
      - [x] No source files touched (pure research/documentation day);
            125 passed, 7 skipped, confirmed unaffected
- [x] Day 13: Backtesting, Monte Carlo, allocation overlay
      - [x] Central methodological decision: a valid 2019-2024 backtest
            needs point-in-time PARAMETERS, not just point-in-time
            probabilities. Reusing Day 4's full-series HMM would repeat
            the same look-ahead bias Day 9 flagged for its ensemble
            members. Refit on pre-2019 data alone: K=5 fails outright
            (0/30), K=3 degenerates into 2 single-day outlier states
            (35/35/2015 occupancy), K=2 is robust (30/30) - further,
            independent confirmation of Day 4's "simpler is more
            robust" finding, not a convenience fallback
      - [x] `src/models/monte_carlo/regime_mc.py`: A13.1/A13.2, with one
            real numerical fix (exact row-stochasticity before
            inverse-CDF state sampling, checked directly against a
            near-boundary random test case, not assumed negligible)
      - [x] `src/backtest/overlay.py`: tilt rules built around a real
            subtlety in this project's own regime taxonomy (Post-Shock
            has the 2nd-highest drift of any regime, 0.25, a recovery
            rally not continued stress). Certain Post-Shock gets weight
            1.0 from proper risk-adjusted rules, weight 0.3 (= certain
            Risk-Off) from a naive name-based rule kept to show the
            contrast. Found and fixed a real 1D/2D shape inconsistency
            bug in tilt_probability_weighted during testing
      - [x] Backtest 2019-2024 (synthetic price path, explicitly and
            repeatedly flagged as such, not real Nifty): all 3 overlay
            variants reduce drawdown and achieve positive IR vs
            buy-hold. Post-Shock point extended with a genuine 2nd
            finding: the full 5-state model's OWN ESTIMATED Post-Shock
            drift came out negative (-0.809), not the true +0.25,
            directly extending Day 4's weak-identification finding to a
            concrete downstream consequence (reported as illustrative,
            n=2 days only)
      - [x] `src/models/monte_carlo/ic_artefact.py`: A13.3 conditional
            statements generated from actual numbers (never
            hand-written), each with full traceable lineage (model, fit
            window, seed, valid-restart count, sim seed/count/horizon)
      - [x] `notebooks/11_monte_carlo_backtest.ipynb` (0 errors, all
            claims verified against printed output) +
            `docs/day13_monte_carlo_notes.md`
      - [x] 156 passed, 7 skipped (main env, incl. 31 new tests across
            regime_mc, overlay, backtest engine, IC artefact)
- [x] Day 14: Validation, polish, Python/R cross-checks
      - [x] R cross-validation (`r/`): checked directly, not assumed,
            that CRAN is unreachable here (named packages depmixS4,
            MSwM, bcp, changepoint, conformal, mlr3 are all
            uninstallable). Implemented the underlying algorithms
            directly in base R instead of substituting a different
            package - arguably a stronger cross-check (same math, two
            languages) than a packaged alternative would have been
      - [x] HMM cross-check: Python and R converge to DIFFERENT local
            optima on identical data (R best-of-10=12077.19 beats
            Python best-of-30=12073.49) - independent confirmation of
            Day 4's weak-identifiability finding from a second angle,
            not a bug. Resulting regime probabilities still agree
            reasonably (85.5% hard-label agreement)
      - [x] Conformal cross-check: found a real discrepancy (R's q_hat
            0.9843 vs Python's 0.9934 on identical Day 11 data), traced
            to a quantile-interpolation convention difference, fixed by
            matching numpy's exact order-statistic formula - after the
            fix, exact match to printed precision
      - [x] VaR/CVaR cross-check: given identical model spec, 10,000-
            path simulations in each language agree within 0.4-0.6pp,
            consistent with expected MC sampling noise
      - [x] End-to-end batch pipeline validation
            (raw data -> features -> regime probability -> tilt ->
            Monte Carlo -> IC artefact) run for real on a target date;
            every stage connects, one real edge case (IC artefact's
            "too few simulated paths" fallback) fired correctly in a
            live run, not just a synthetic test
      - [x] Online inference loop validated separately: particle
            filter + BOCPD + streaming Dirichlet update, warm-started
            then processing exactly ONE new streamed day, matching how
            a production system would actually operate
      - [x] Executive summary (`docs/EXECUTIVE_SUMMARY.md`), IC
            briefing template with a real worked example
            (`docs/IC_BRIEFING_TEMPLATE.md`), peer review self-audit
            (`docs/PEER_REVIEW.md`)
      - [x] Peer review found and corrected a real documentation
            imprecision: "7 skipped" had been described uniformly as
            "the torch gap" in recent PROJECT_PLAN entries; actually
            only 3 are torch-related, the other 4 are TF/PyMC tests
            skipped BY DESIGN in the main env. Verified directly: those
            4 files give 18 passed, 0 skipped in .venv-bayesian.
            Accurate total: 174 tests passing across both environments,
            3 genuine skips
      - [x] Visualisation polish: light spot-check (all 38 figures
            already share one style module), not an exhaustive re-audit
            - stated as such, not overclaimed
      - [x] `docs/day14_validation_notes.md` ties all of the above
            together with full detail
- [~] Day 15: Final submission and handover (partial - 3 of 6 bullets
      require real external access this environment does not have;
      stated honestly rather than claimed as done)
      - [ ] Transfer GitHub repository ownership to @ZethetaIntern: NOT
            DONE - no real external GitHub remote exists to transfer.
            Documented in `docs/EXIT_DOCUMENTATION.md`
      - [ ] Upload all deliverables to shared drive: NOT DONE - no
            shared drive connected to this session. Same doc
      - [x] Submit final report, code, and presentation: code is this
            repository (18 commits); report content exists across 14
            day-notes files mapped to the required structure in
            `docs/REPORT_INDEX.md` (exceeds 40 pages by volume; a
            single consolidated PDF is a stated remaining mechanical
            step, not a content gap); presentation is a real, built,
            validated 18-slide deck (`presentation/REGIME_LAB_PRESENTATION.pptx`,
            passes the pptx skill's validator, visually QA'd across
            title/content/table/dark-background slides)
      - [ ] Record 10-minute demonstration video: NOT literally
            recorded (no video/audio capability). Built
            `docs/VIDEO_WALKTHROUGH_SCRIPT.md` instead: a timestamped
            script from a REAL execution of Day 14's e2e pipeline
            script, every number in it verified against actual saved
            output (caught and fixed 2 inaccuracies while writing it:
            a wrong path count, 10,000 vs the actual 5,000, and an
            imprecise "under 1%" where the real value was exactly 0%)
      - [x] Complete exit documentation and knowledge transfer:
            `docs/EXIT_DOCUMENTATION.md` - states the 3 unfeasible
            items plainly, documents repo state, gives evidence-based
            next steps in priority order
      - [x] Also built: `docs/MODEL_CARD.md` (Deliverable 5,
            consolidating every member model's inputs/priors/version
            and all calibration/MCMC/reconciliation evidence from Days
            5-14 into one document) and discovered the project's own
            name is literally Part B's gamification framework (the
            technical Days 4-11 map onto its 9-level campaign almost
            exactly) - made explicit in the presentation and in
            `docs/VIDEO_WALKTHROUGH_SCRIPT.md`, not noticed until Day 15
      - [x] 156 passed, 7 skipped, confirmed unaffected (no src/
            changes; Day 15 is documentation and a presentation deck)

## Deliverables (Section D3)

- [ ] Deliverable 1: Main report (min. 40 pages, LaTeX)
- [ ] Deliverable 2: Python codebase
- [ ] Deliverable 3: R codebase
- [ ] Deliverable 4: Backtesting and simulation engine
- [ ] Deliverable 5: Model card, calibration and validation pack
- [ ] Deliverable 6: Final presentation (18 slides) + 10-minute demo video

## Open decisions

1. **Data source.** No CSV pack was provided. Searched GitHub (the only
   bulk-data-capable source reachable from this sandbox; NSE/RBI/AMFI/
   Yahoo Finance are all unreachable here) and found partial real data:
   Nifty 50 OHLC 2015-2019 and India VIX 2020-2026, saved under
   `data/raw/real_partial/`. Full 15-year coverage for Nifty 50, and any
   coverage at all for Nifty Midcap 100, Nifty Smallcap 100, 10Y Gilt,
   AAA-Gilt spread, FII/DII flows and SIP totals, was not obtainable this
   way; see `DATA_SOURCES.md` for the full search record and the
   recommended fix (run `jugaad-data`/`nseindiapy`/`yfinance` somewhere
   with normal internet access, e.g. the user's own machine, and drop the
   resulting files under `data/raw/`). `src/data/loader.py` supports a
   CSV backend and a synthetic regime-switching backend behind one
   interface, so model development proceeds on synthetic data now and
   switches to real data with a one-line config change whenever it
   arrives.
2. **Foundation model weights (Day 8).** `chronos-forecasting` and
   `timesfm` install from PyPI but pull pretrained weights from Hugging
   Face Hub at first use. If that host is unreachable here, the fallback
   is documented at that point in the plan rather than blocking earlier
   days.
3. **Report format.** Deliverable 1 (main report) and the model card are
   drafted in LaTeX rather than DOCX.
4. **Disk space is tight (started around 5-6GB free, not GB-scale
   headroom).** Plain `pip install torch` pulls a full CUDA stack
   (cuda-toolkit, nvidia-cudnn, nvidia-nccl, triton) that alone can
   exceed what's free. Fix used for Day 3: `pip install --no-deps torch`
   plus its small pure-Python deps (filelock, sympy, networkx, jinja2,
   fsspec, typing-extensions) by hand, which gives a working CPU-only
   torch at ~1GB instead of several GB. `torch_geometric` and
   `giotto-tda` installed normally (small, pure-Python-heavy wheels).
   Side effect: `giotto-tda` downgraded numpy 2.4.4 -> 1.26.4 and
   scikit-learn 1.8.0 -> 1.3.2; re-ran the full test suite after and
   everything still passed. Check `df -h /` before any heavy install on
   Day 7 (TensorFlow) and Day 8 (foundation model weights, which land in
   `~/.cache/huggingface` at multiple GB each).
5. **PyMC and giotto-tda cannot coexist in one environment, confirmed on
   Day 5, not a hypothetical.** `pytensor` (PyMC's backend) hard-requires
   `numpy>=2.0`; `giotto-tda==0.6.2` (the only version on PyPI)
   hard-requires `scikit-learn==1.3.2`, which requires `numpy<2.0`.
   Upgrading scikit-learn to work with numpy 2.x breaks giotto-tda at
   runtime (`TypeError: check_array() got an unexpected keyword argument
   'force_all_finite'`), verified directly, not inferred from pip's
   dependency warnings alone. Fix: a second, isolated venv at
   `.venv-bayesian/` (created with `python3 -m venv`) holds pymc, arviz,
   numpyro-if-needed, and a matching hmmlearn/scikit-learn, entirely
   separate from the main environment's torch/gtda/hmmlearn stack.
   Registered as the Jupyter kernel `bayesian-env` (vs. `python3` for
   every other notebook). Any future day needing both PyMC and TDA/GCN
   features in the same analysis will need to bridge the two via saved
   files (parquet/csv/netCDF), not a shared process.
6. **Background processes do not survive between separate tool calls in
   this sandbox, confirmed on Day 5.** A `nohup long_job &` launched in
   one command is gone by the next command, even with the process
   protected from SIGHUP - something about how each tool call's process
   context is torn down kills it regardless. This ruled out a
   background-and-poll pattern for anything long-running; expensive
   computations (e.g. Day 5's MCMC sampling) must either fit inside a
   single tool call's ~280s practical budget or be split into multiple
   complete, independently-resumable invocations (one per chunk of work,
   each saving its result to disk before returning) run across multiple
   tool calls. Relevant again for Day 6-9's heavier model fits and Day 13's
   Monte Carlo engine - budget for it before assuming a long job can just
   run in the background.
7. **This environment has exactly 1 CPU core** (`os.cpu_count() == 1`).
   PyMC's automatic BLAS-core-per-worker allocation divides by the
   requested core count and raises `ZeroDivisionError` when core
   auto-detection resolves oddly here; pass `cores=1` explicitly to
   `pm.sample()` always, in this environment. Also means PyMC's "N
   chains in parallel" is actually sequential here regardless of the
   `chains` argument - timing estimates should assume no parallelism.
8. **Multi-chain label-switching, confirmed on Day 6.** Any PyMC model
   with K interchangeable regimes/components and no ordering constraint
   built in will let independent chains land on different arbitrary
   permutations of the K labels, producing catastrophic R-hat (seen:
   1.4-2.4) with zero divergences - not a sampling failure, a labelling
   one. Day 5's univariate HMM avoided this with an `ordered` transform
   on a single scalar (`mu`); that trick doesn't generalise to a
   multi-dimensional regime parameter (Day 6's VAR intercept `c` is
   6-dimensional, no natural single ordering). Fix used: post-hoc
   relabelling, per posterior draw, by sorting on one chosen scalar
   projection (Day 6 used the Nifty-return intercept). Check for this
   symmetry whenever a new model has K>1 unordered latent
   regimes/components, before trusting R-hat at face value - low
   divergences and bad R-hat together is the specific signature to
   watch for.
9. **Torch is broken in the main environment (regression, Day 7).** A
   plain `pip install tensorflow` in the main environment made pip
   "complete" torch's dependency tree, which had been skipped on Day 3
   with `--no-deps`. That pulled several GB of CUDA packages, ran the disk
   out of space mid-install and left corrupted and half-installed
   packages. Cleanup restored numpy 1.26.4 and scikit-learn 1.3.2 (so
   hmmlearn and giotto-tda work again) and freed the disk, but
   `import torch` still fails: this wheel's `libtorch_global_deps.so`
   links `libcudart` unconditionally, and its fallback path then insists
   on the `nvidia-*` pip libraries one at a time (cublas alone is 439MB,
   with cudnn and others still to come). Repairing it means installing
   several GB of CUDA libraries on a CPU-only machine for a package Day 7
   does not use, so it was left broken on purpose. Consequences:
   `src/features/sector_gnn.py` (Day 3) cannot be imported, and its 3
   GCN tests skip with an explicit reason. Day 3's saved results and
   notebook outputs are unaffected. Lesson: never run an unpinned `pip
   install` in an environment holding a `--no-deps` package, and check
   `df -h /` first.
10. **TensorFlow and TFP live in `.venv-bayesian`, not the main
    environment.** They need numpy 2.x, which giotto-tda's pinned
    scikit-learn forbids there. TFP also needs the `tf-keras` shim
    (Keras 3 broke its expected API) and `TF_USE_LEGACY_KERAS=1`, set in
    each `src/models/bdl` module. Training ran as scripts, one stage per
    tool call. Bridge files (features and HMM labels) are CSV in
    `artifacts_data/`, because no parquet engine is installed.
11. **Day 8 is blocked independent of torch.** Hugging Face returns `403`
    with `x-deny-reason: host_not_allowed` from this sandbox (tested on
    Day 7). Chronos, TimesFM, Lag-Llama and Moirai all fetch pretrained
    weights from there, so the zero-shot embeddings Day 8 asks for cannot
    be produced here. A working torch is a second, separate blocker.
    Decide the Day 8 approach with the user before starting: options are
    a clearly labelled substitute (for example a small transformer
    trained from scratch on the synthetic panel, which is not a
    foundation model and would not answer the task's question), running
    the provided code on a machine with normal internet or skipping the
    day with the reason documented.

    **Resolution (Day 8).** The user instructed: use synthetic data and
    implement the model. Two architecture-level re-implementations
    (Chronos-style and TimesFM-style) were built in TensorFlow and
    pretrained on a synthetic corpus. They are not the real checkpoints and
    the Day 8 results do not speak to real Chronos or TimesFM. Running the
    real models on a machine with internet remains the only way to answer
    that. Lag-Llama and Moirai were not implemented; the task requires
    Chronos plus at least one other model, which the TimesFM-style model
    covers.
12. **Recovered work from an earlier part of the Day 8 session.** When Day
    8 resumed, `src/models/foundation/`, four `scripts/run_day8_*.py`
    files and `artifacts_data/day8/` (pretrained weights, embeddings, two
    finished sample-efficiency arms) already existed but were not in the
    working context. They were audited (code read, outputs inspected)
    before being adopted rather than rebuilt. The five foundation-model
    sample-efficiency arms had not been run and were run then; the
    no-regime-switch control was added.

"""The combined regime output contract, Section A10.4: one record per
day, whatever combination method produced it.

A10.4 lists five fields beyond the probability vector itself: a
conformalised prediction set, the dominant model and its weight, the
epistemic/aleatoric split, and a model-health flag (BOCPD changepoint,
online/batch reconciliation gap, out-of-distribution score). Three of
those depend on work this project has not built yet:

- Conformal calibration is Day 11's deliverable. `conformal_lower` /
  `conformal_upper` below use a PLACEHOLDER (a fixed-width band from the
  cross-member spread, not a calibrated conformal set) and are marked
  `conformal_is_placeholder=True` in the record so a caller cannot
  mistake this for the real thing.
- BOCPD is Day 10's deliverable. `changepoint_flag` is always `None` here.
- Online/batch reconciliation and an out-of-distribution score depend on
  a streaming/online inference path this project has not built (Day 10's
  scope also). `reconciliation_gap` and `ood_score` are always `None`.

Populating the schema now, even with explicit placeholders, is the
point: A10.4 defines a CONTRACT (field names and types other code
depends on), and getting that contract right, with honest gaps marked
as gaps, is separable from having every upstream model that fills it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class RegimeOutputRecord:
    date: str
    regime_id: int
    regime_label: str
    probability: float
    conformal_lower: float
    conformal_upper: float
    conformal_is_placeholder: bool
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    dominant_model: str
    dominant_model_weight: float
    changepoint_flag: bool | None
    reconciliation_gap: float | None
    ood_score: float | None
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_output_contract(date: str, combined_probs: np.ndarray, labels: list[str],
                           member_probs: np.ndarray, member_names: list[str],
                           weights: np.ndarray, source: str,
                           placeholder_band_width: float = 0.15) -> RegimeOutputRecord:
    """One day's combined-output record.

    combined_probs: (K,) the ensembled probability vector.
    member_probs: (M, K) each member's probability vector for this day -
      used for the epistemic (cross-member std of the dominant class'
      probability) / aleatoric (mean per-member entropy) split, the same
      two decompositions used throughout Days 7-8, applied here to
      MODELS rather than stochastic forward passes or ensemble members
      of one network. This conflates two different notions of
      "epistemic" (model-form uncertainty vs. within-model weight
      uncertainty); flagged here rather than presented as identical to
      Day 7's decomposition.
    weights: (M,) the combination weights used (BMA or stacking).
    placeholder_band_width: the PLACEHOLDER conformal band's half-width
      in probability units, applied around the dominant class'
      probability and clipped to [0, 1]. Not calibrated; see module
      docstring.
    """
    regime_id = int(np.argmax(combined_probs))
    p = float(combined_probs[regime_id])
    dominant_idx = int(np.argmax(weights))

    member_p_for_class = member_probs[:, regime_id]
    epistemic = float(member_p_for_class.std())
    clipped = np.clip(member_probs, 1e-12, 1.0)
    aleatoric = float((-(clipped * np.log(clipped)).sum(axis=1)).mean())

    lower = max(0.0, p - placeholder_band_width)
    upper = min(1.0, p + placeholder_band_width)

    return RegimeOutputRecord(
        date=date, regime_id=regime_id, regime_label=labels[regime_id], probability=p,
        conformal_lower=lower, conformal_upper=upper, conformal_is_placeholder=True,
        epistemic_uncertainty=epistemic, aleatoric_uncertainty=aleatoric,
        dominant_model=member_names[dominant_idx], dominant_model_weight=float(weights[dominant_idx]),
        changepoint_flag=None, reconciliation_gap=None, ood_score=None, source=source,
    )


def build_output_contract_series(dates: list[str], combined_probs: np.ndarray, labels: list[str],
                                  member_probs: np.ndarray, member_names: list[str],
                                  weights: np.ndarray, source: str) -> list[dict]:
    """`build_output_contract` applied over every day; weights are
    per-day if `weights` is (N, M), or shared across all days if (M,)."""
    N = len(dates)
    per_day_weights = weights if weights.ndim == 2 else np.tile(weights, (N, 1))
    return [
        build_output_contract(dates[t], combined_probs[t], labels, member_probs[:, t, :],
                               member_names, per_day_weights[t], source).to_dict()
        for t in range(N)
    ]

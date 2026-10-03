import numpy as np
import pytest

from src.models.ensemble.contract import build_output_contract, build_output_contract_series, RegimeOutputRecord


LABELS = ["Risk-On", "Transitional", "Risk-Off"]


def _toy():
    member_probs = np.array([[0.8, 0.15, 0.05], [0.6, 0.3, 0.1], [0.9, 0.05, 0.05]])
    weights = np.array([0.5, 0.3, 0.2])
    combined = weights @ member_probs
    return member_probs, weights, combined


def test_output_contract_probability_matches_weighted_combination():
    member_probs, weights, combined = _toy()
    rec = build_output_contract("2025-06-27", combined, LABELS, member_probs, ["a", "b", "c"], weights, "stacking")
    assert rec.regime_id == 0
    assert rec.regime_label == "Risk-On"
    assert rec.probability == pytest.approx(combined[0])


def test_output_contract_dominant_model_is_highest_weight_not_highest_probability():
    member_probs = np.array([[0.5, 0.3, 0.2], [0.9, 0.05, 0.05]])
    weights = np.array([0.9, 0.1])   # member 0 has lower confidence but the higher weight
    combined = weights @ member_probs
    rec = build_output_contract("d", combined, LABELS, member_probs, ["low_conf_high_weight", "high_conf_low_weight"], weights, "bma")
    assert rec.dominant_model == "low_conf_high_weight"
    assert rec.dominant_model_weight == pytest.approx(0.9)


def test_output_contract_epistemic_zero_when_members_agree():
    member_probs = np.tile([0.7, 0.2, 0.1], (4, 1))   # every member identical
    weights = np.full(4, 0.25)
    combined = weights @ member_probs
    rec = build_output_contract("d", combined, LABELS, member_probs, list("abcd"), weights, "bma")
    assert rec.epistemic_uncertainty == pytest.approx(0.0, abs=1e-9)


def test_output_contract_epistemic_positive_when_members_disagree():
    member_probs = np.array([[0.9, 0.05, 0.05], [0.1, 0.1, 0.8]])
    weights = np.array([0.5, 0.5])
    combined = weights @ member_probs
    rec = build_output_contract("d", combined, LABELS, member_probs, ["a", "b"], weights, "bma")
    assert rec.epistemic_uncertainty > 0.1


def test_output_contract_conformal_band_is_flagged_as_placeholder_and_clipped():
    member_probs, weights, combined = _toy()
    rec = build_output_contract("d", combined, LABELS, member_probs, ["a", "b", "c"], weights, "bma",
                                 placeholder_band_width=0.9)
    assert rec.conformal_is_placeholder is True
    assert 0.0 <= rec.conformal_lower <= rec.probability <= rec.conformal_upper <= 1.0


def test_output_contract_unbuilt_fields_are_explicitly_none():
    """Day 10/11 fields must be None, not silently omitted or defaulted
    to something a caller could mistake for a real value."""
    member_probs, weights, combined = _toy()
    rec = build_output_contract("d", combined, LABELS, member_probs, ["a", "b", "c"], weights, "bma")
    assert rec.changepoint_flag is None
    assert rec.reconciliation_gap is None
    assert rec.ood_score is None


def test_output_contract_series_matches_per_day_calls():
    member_probs, weights, combined = _toy()
    N = 3
    dates = [f"d{i}" for i in range(N)]
    combined_series = np.tile(combined, (N, 1))
    member_probs_series = np.tile(member_probs[:, None, :], (1, N, 1))
    series = build_output_contract_series(dates, combined_series, LABELS, member_probs_series,
                                           ["a", "b", "c"], weights, "stacking")
    assert len(series) == N
    single = build_output_contract(dates[1], combined, LABELS, member_probs, ["a", "b", "c"], weights, "stacking")
    assert series[1] == single.to_dict()


def test_output_contract_series_accepts_per_day_weights():
    member_probs, _, _ = _toy()
    N = 2
    per_day_weights = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # day 0 -> member a, day 1 -> member c
    combined_series = np.stack([per_day_weights[t] @ member_probs for t in range(N)])
    member_probs_series = np.tile(member_probs[:, None, :], (1, N, 1))
    series = build_output_contract_series(["d0", "d1"], combined_series, LABELS, member_probs_series,
                                           ["a", "b", "c"], per_day_weights, "stacking")
    assert series[0]["dominant_model"] == "a"
    assert series[1]["dominant_model"] == "c"


def test_to_dict_round_trips_all_fields():
    rec = RegimeOutputRecord(
        date="d", regime_id=0, regime_label="Risk-On", probability=0.5,
        conformal_lower=0.3, conformal_upper=0.7, conformal_is_placeholder=True,
        epistemic_uncertainty=0.1, aleatoric_uncertainty=0.2,
        dominant_model="a", dominant_model_weight=1.0,
        changepoint_flag=None, reconciliation_gap=None, ood_score=None, source="test",
    )
    d = rec.to_dict()
    assert set(d.keys()) == {f.name for f in rec.__dataclass_fields__.values()}

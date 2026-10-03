"""Reconciling regime label spaces across ensemble members.

Days 4, 5, 7 and 8 all classify into 5 regimes (Risk-On, Late-Cycle,
Transitional, Post-Shock, Risk-Off). Day 6's multivariate RS-VAR only
supports K=3 (Risk-On, Transitional, Risk-Off) at feasible compute -
documented there as a compute-budget limitation, not a design choice.

An ensemble needs every member on the same output space. Rather than
fabricate a 5-way split for RS-VAR that has no basis, every member here
is collapsed to the 3-way space, using the same mapping already
established in `docs/day6_rsvar_notes.md`'s ground-truth comparison:
Risk-On absorbs Post-Shock (both are positive-drift states in this
project's regime taxonomy), Transitional absorbs Late-Cycle. This means
Day 9's ensemble answers a coarser question (3 regimes, not 5) than
Days 4/5/7/8 individually did. That is a real scope reduction, stated
here once so every caller inherits the same, single definition instead
of each script rolling its own mapping.
"""
from __future__ import annotations

import numpy as np

LABELS_3 = ["Risk-On", "Transitional", "Risk-Off"]

# Index i of LABELS_5 collapses into LABELS_3[j] where j = index of the
# mapped name below.
LABELS_5 = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]
COLLAPSE_5_TO_3 = {
    "Risk-On": "Risk-On",
    "Post-Shock": "Risk-On",
    "Late-Cycle": "Transitional",
    "Transitional": "Transitional",
    "Risk-Off": "Risk-Off",
}


def collapse_probs_5_to_3(probs_5: np.ndarray, label_order_5: list[str]) -> np.ndarray:
    """(N, 5) probabilities, in the order given by `label_order_5`, summed
    into (N, 3) probabilities in `LABELS_3` order. Summing (not
    re-normalising afterwards) is exact for a probability vector: the
    collapsed classes' probability mass is exactly the sum of the
    5-way classes that map to them, since the 5-way rows already sum to
    1 and the mapping partitions them."""
    out = np.zeros((probs_5.shape[0], 3))
    for i, label in enumerate(label_order_5):
        j = LABELS_3.index(COLLAPSE_5_TO_3[label])
        out[:, j] += probs_5[:, i]
    return out


def to_display_label(label) -> str:
    """`src/data/synthetic.py` uses snake_case regime names
    (risk_on, late_cycle, ...); the rest of this project displays them
    Title-Case-With-Hyphens (Risk-On, Late-Cycle, ...). Accepts either,
    idempotently, so callers do not need to know which one a given
    upstream array happens to use."""
    if label in LABELS_5:
        return label
    return label.replace("_", " ").title().replace(" ", "-")


def collapse_labels_5_to_3(labels_5) -> np.ndarray:
    """Hard-label version, for turning HMM pseudo-labels or synthetic
    ground truth (5-way, either naming convention) into the 3-way space
    that comparisons are made in."""
    return np.array([COLLAPSE_5_TO_3[to_display_label(l)] for l in labels_5])

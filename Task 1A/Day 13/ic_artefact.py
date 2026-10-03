"""The Investment Committee artefact, Section A13.3: Monte Carlo output
as testable conditional statements, each carrying explicit lineage (the
model, data window, and seed that produced it) so the statement can be
traced and reproduced rather than taken on faith.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Lineage:
    model_name: str
    model_fit_window: str
    model_fit_seed: int
    n_valid_restarts: int
    simulation_seed: int
    n_sims: int
    horizon_days: int
    as_of_date: str


@dataclass
class ICArtefact:
    lineage: Lineage
    current_regime_probabilities: dict
    prediction_interval_90: tuple
    prob_negative_return: float
    worst_case_5th_percentile: float
    var_5pct: float
    cvar_5pct: float
    conditional_statement: str

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def render_ic_statements(self) -> list[str]:
        """The four example-style statements from Section A13.3,
        populated from this artefact's own numbers rather than
        hand-written, so they cannot drift out of sync with the
        underlying computation."""
        lo, hi = self.prediction_interval_90
        statements = [
            f"Conditional on current regime probabilities "
            f"({', '.join(f'{k}: {v:.0%}' for k, v in self.current_regime_probabilities.items())}), "
            f"the 90% one-year prediction interval for the scheme is {lo:+.0%} to {hi:+.0%}.",
            f"Probability of negative one-year return: {self.prob_negative_return:.0%}.",
            f"Worst-case (5th percentile) one-year return: {self.worst_case_5th_percentile:+.0%}.",
            self.conditional_statement,
        ]
        return statements


def build_ic_artefact(lineage: Lineage, final_regime_dist: dict, p90_interval: tuple,
                       prob_negative: float, p5_return: float, var5: float, cvar5: float,
                       conditional_statement: str) -> ICArtefact:
    return ICArtefact(
        lineage=lineage, current_regime_probabilities=final_regime_dist,
        prediction_interval_90=p90_interval, prob_negative_return=prob_negative,
        worst_case_5th_percentile=p5_return, var_5pct=var5, cvar_5pct=cvar5,
        conditional_statement=conditional_statement,
    )


def conditional_on_state_statement(paths: np.ndarray, states: np.ndarray, target_state: int,
                                    window_days: int, min_days_in_state: int,
                                    state_name: str, alpha: float = 0.05) -> tuple[str, float, float]:
    """Builds the fourth example statement's pattern directly: "if the
    regime transitions to X within N days (probability: p%), worst-case
    drawdown rises to -q%". Returns (statement, probability, worst_case)
    rather than just a string, so the two numbers can be checked or
    reused without re-parsing the sentence.
    """
    in_state_mask = (states[:, :window_days] == target_state).sum(axis=1) >= min_days_in_state
    p_transition = float(in_state_mask.mean())
    final_returns = paths[:, -1] - 1.0
    if in_state_mask.sum() >= 20:
        worst_case = float(np.percentile(final_returns[in_state_mask], alpha * 100))
        statement = (f"If the regime spends at least {min_days_in_state} of the first {window_days} days "
                     f"in {state_name} (probability: {p_transition:.0%}), the worst-case "
                     f"({alpha:.0%}-percentile) one-year return is {worst_case:+.0%}.")
    else:
        worst_case = float("nan")
        statement = (f"If the regime spends at least {min_days_in_state} of the first {window_days} days "
                     f"in {state_name} (probability: {p_transition:.0%}): too few simulated paths "
                     f"({int(in_state_mask.sum())}) meet this condition to estimate a reliable worst case.")
    return statement, p_transition, worst_case

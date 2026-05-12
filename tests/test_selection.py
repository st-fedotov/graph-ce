"""Tests for Wagner-faithful percentile selection with tie cap.

The tie cap is the part that distinguishes our implementation from a plain
``scores >= threshold`` filter. Without it, many sessions tying at the
percentile boundary flood the elite pool.
"""
from __future__ import annotations

import numpy as np
import pytest

from graph_ce.cem import select_by_percentile_with_tiebreak


def test_toy_example_from_documentation():
    """10 sessions, percentile=70, budget=3: tied-at-cutoff sessions should
    be capped to exactly the budget."""
    scores = np.array([5, 4, 3, 3, 3, 3, 3, 2, 1, 0], dtype=float)
    idx = select_by_percentile_with_tiebreak(scores, percentile=70.0, budget=3)
    assert idx.tolist() == [0, 1, 2]


def test_well_separated_scores_match_naive_filter():
    """When there are no ties at the boundary, tie cap and naive >= filter agree."""
    scores = np.array([10.0, 8.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.0])
    idx = select_by_percentile_with_tiebreak(scores, percentile=70.0, budget=3)
    threshold = np.percentile(scores, 70.0)
    naive = np.flatnonzero(scores >= threshold)
    assert idx.tolist() == naive.tolist()


def test_all_scores_identical_returns_budget():
    """Pathological case: everyone is tied. Without the cap our pool would
    take all 100; with the cap we stop at budget."""
    scores = np.full(100, fill_value=-2.18)
    idx = select_by_percentile_with_tiebreak(scores, percentile=93.0, budget=7)
    assert len(idx) == 7
    # And they're the first 7 in iteration order, matching Wagner's
    # iteration-order tie-breaking.
    assert idx.tolist() == list(range(7))


def test_strictly_above_threshold_always_included():
    """A session strictly above the percentile threshold should be selected
    even if the budget is already exhausted by earlier ties."""
    # First 5 sessions are at the threshold; next 5 are below; then a
    # spike strictly above. Budget = 3.
    scores = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 9.0])
    idx = select_by_percentile_with_tiebreak(scores, percentile=50.0, budget=3)
    # threshold = 50th percentile ≈ 3.0; budget exhausted by first 3 ties;
    # session 9 is strictly above and must be admitted regardless.
    assert 9 in idx.tolist()
    # And exactly 3 of the tied sessions made it in.
    tied_kept = [i for i in idx.tolist() if scores[i] == 3.0]
    assert len(tied_kept) == 3


def test_empty_input():
    idx = select_by_percentile_with_tiebreak(np.array([], dtype=float), 50.0, 5)
    assert idx.shape == (0,)


def test_budget_larger_than_population():
    scores = np.array([1.0, 2.0, 3.0])
    # percentile=0 → threshold = min = 1.0, so every score passes the cutoff.
    idx = select_by_percentile_with_tiebreak(scores, percentile=0.0, budget=100)
    assert idx.tolist() == [0, 1, 2]


def test_budget_zero_admits_only_strictly_above():
    # 5 at threshold, 1 strictly above. With budget=0, only the strict one survives.
    scores = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 9.0])
    idx = select_by_percentile_with_tiebreak(scores, percentile=50.0, budget=0)
    assert idx.tolist() == [5]

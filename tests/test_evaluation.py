"""
test_evaluation.py – Unit tests for src/evaluation.py

Uses deterministic synthetic examples so expected metric values can be
verified by hand without running the full spaCy model.

Run with:  pytest tests/test_evaluation.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd

from src.evaluation import (
    compute_metrics, _safe_div,
    validate_annotations, load_annotated_data,
    REQUIRED_COLUMNS,
)


# ── _safe_div ─────────────────────────────────────────────────────────────────

class TestSafeDiv:
    def test_normal_division(self):
        assert _safe_div(3, 4) == pytest.approx(0.75)

    def test_zero_denominator_returns_zero(self):
        assert _safe_div(5, 0) == 0.0

    def test_zero_numerator(self):
        assert _safe_div(0, 5) == 0.0

    def test_both_zero(self):
        assert _safe_div(0, 0) == 0.0


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_perfect_predictions(self):
        m = compute_metrics(tp=10, fp=0, fn=0)
        assert m["precision"] == 1.0
        assert m["recall"]    == 1.0
        assert m["f1"]        == 1.0

    def test_no_predictions(self):
        m = compute_metrics(tp=0, fp=0, fn=5)
        assert m["precision"] == 0.0
        assert m["recall"]    == 0.0
        assert m["f1"]        == 0.0

    def test_all_extra_no_correct(self):
        m = compute_metrics(tp=0, fp=5, fn=0)
        assert m["precision"] == 0.0
        assert m["recall"]    == 0.0
        assert m["f1"]        == 0.0

    def test_all_zero_counts(self):
        m = compute_metrics(tp=0, fp=0, fn=0)
        assert m["precision"] == 0.0
        assert m["recall"]    == 0.0
        assert m["f1"]        == 0.0

    def test_partial_match_tp2_fp1_fn1(self):
        # P = 2/3, R = 2/3, F1 = 2/3
        m = compute_metrics(tp=2, fp=1, fn=1)
        assert m["precision"] == pytest.approx(2/3, abs=1e-4)
        assert m["recall"]    == pytest.approx(2/3, abs=1e-4)
        assert m["f1"]        == pytest.approx(2/3, abs=1e-4)

    def test_high_precision_low_recall(self):
        # TP=1, FP=0, FN=9 → P=1.0, R=0.1
        m = compute_metrics(tp=1, fp=0, fn=9)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"]    == pytest.approx(0.1)

    def test_low_precision_high_recall(self):
        # TP=1, FP=9, FN=0 → P=0.1, R=1.0
        m = compute_metrics(tp=1, fp=9, fn=0)
        assert m["precision"] == pytest.approx(0.1)
        assert m["recall"]    == pytest.approx(1.0)

    def test_f1_harmonic_mean_formula(self):
        # P=1.0, R=0.5 → F1 = 2*1.0*0.5/1.5 = 2/3
        m = compute_metrics(tp=1, fp=0, fn=1)
        expected_f1 = 2 * 1.0 * 0.5 / (1.0 + 0.5)
        assert m["f1"] == pytest.approx(expected_f1, abs=1e-4)

    def test_returned_dict_has_all_keys(self):
        m = compute_metrics(tp=3, fp=1, fn=2)
        for key in ("precision", "recall", "f1", "tp", "fp", "fn"):
            assert key in m

    def test_raw_counts_preserved(self):
        m = compute_metrics(tp=7, fp=2, fn=3)
        assert m["tp"] == 7
        assert m["fp"] == 2
        assert m["fn"] == 3


# ── validate_annotations ─────────────────────────────────────────────────────

class TestValidateAnnotations:
    def _df(self, rows):
        return pd.DataFrame(rows)

    def test_valid_row_no_warnings(self):
        df = self._df([{
            "text_id": "1",
            "text": "Apple Inc. is great.",
            "entity_text": "Apple Inc.",
            "entity_label": "ORG",
            "start_char": 0,
            "end_char": 10,
        }])
        assert validate_annotations(df) == []

    def test_bad_offset_generates_warning(self):
        df = self._df([{
            "text_id": "1",
            "text": "Apple Inc. is great.",
            "entity_text": "Apple Inc.",
            "entity_label": "ORG",
            "start_char": 5,    # wrong
            "end_char": 15,
        }])
        warnings = validate_annotations(df)
        assert len(warnings) == 1
        assert "offset mismatch" in warnings[0]

    def test_no_entity_row_skipped(self):
        df = self._df([{
            "text_id": "2",
            "text": "The sky is blue.",
            "entity_text": "NO_ENTITY",
            "entity_label": "",
            "start_char": 0,
            "end_char": 0,
        }])
        assert validate_annotations(df) == []

    def test_multiple_valid_rows(self):
        df = self._df([
            {"text_id": "1", "text": "Steve Jobs founded Apple.",
             "entity_text": "Steve Jobs", "entity_label": "PERSON",
             "start_char": 0, "end_char": 10},
            {"text_id": "1", "text": "Steve Jobs founded Apple.",
             "entity_text": "Apple", "entity_label": "ORG",
             "start_char": 19, "end_char": 24},
        ])
        assert validate_annotations(df) == []

"""
evaluation.py – Evaluation metrics for NER Explorer.

Implements exact-span entity matching: a prediction is correct only when
both the character span (start_char, end_char) and the entity label match
the annotated reference exactly.

Metrics
-------
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-score  = 2 × P × R / (P + R)

where TP = entities correctly predicted,
      FP = extra predictions not in the reference,
      FN = reference entities not predicted.

Both micro-averaged (overall) and per-entity-type metrics are provided.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from spacy.language import Language

from .ner_engine import extract_entities


# ── Required CSV columns ──────────────────────────────────────────────────────

REQUIRED_COLUMNS: set[str] = {
    "text_id", "text", "entity_text", "entity_label",
    "start_char", "end_char",
}


# ── Reference data loading & validation ──────────────────────────────────────

def load_annotated_data(path: str) -> pd.DataFrame:
    """
    Load and validate an annotated CSV dataset.

    Expected columns
    ----------------
    text_id      : unique identifier for each document / sentence
    text         : the full input text for that document
    entity_text  : the entity surface form, or 'NO_ENTITY' if none
    entity_label : the entity type label (PERSON, ORG, …)
    start_char   : character offset of the entity start (0-indexed)
    end_char     : character offset of the entity end (exclusive)

    Rows where entity_text == 'NO_ENTITY' represent documents with
    no annotated entities and are skipped during evaluation.
    """
    df = pd.read_csv(path, dtype={"text_id": str})
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Annotated CSV is missing required columns: {missing}\n"
            f"Required: {REQUIRED_COLUMNS}"
        )
    return df


def validate_annotations(df: pd.DataFrame) -> list[str]:
    """
    Check annotation consistency.

    Verifies that text[start_char:end_char] == entity_text for every
    non-NO_ENTITY row. Returns a list of warning strings so that
    partial datasets can still be used after warnings are displayed.
    """
    warnings: list[str] = []
    for _, row in df.iterrows():
        et = str(row.get("entity_text", "") or "")
        if not et or et.strip().upper() == "NO_ENTITY":
            continue
        text   = str(row["text"])
        start  = int(row["start_char"])
        end    = int(row["end_char"])
        found  = text[start:end]
        if found != et:
            warnings.append(
                f"text_id={row['text_id']}: offset mismatch – "
                f"expected '{et}', found '{found}' at [{start}:{end}]"
            )
    return warnings


# ── Internal helpers ──────────────────────────────────────────────────────────

def _span_key(label: str, start: int, end: int) -> tuple:
    """Canonical key for exact-span matching: (LABEL, start, end)."""
    return (label.upper(), int(start), int(end))


def _ref_spans(df: pd.DataFrame, text_id: Any) -> set[tuple]:
    rows = df[df["text_id"].astype(str) == str(text_id)]
    spans: set[tuple] = set()
    for _, row in rows.iterrows():
        et = str(row.get("entity_text", "") or "")
        if not et or et.strip().upper() == "NO_ENTITY":
            continue
        spans.add(_span_key(row["entity_label"], row["start_char"], row["end_char"]))
    return spans


def _pred_spans(text: str, nlp: Language) -> set[tuple]:
    result = extract_entities(text, nlp)
    return {
        _span_key(e["label"], e["start_char"], e["end_char"])
        for e in result["entities"]
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def compute_metrics(tp: int, fp: int, fn: int) -> dict:
    """
    Compute precision, recall, and F1-score from raw counts.

    Division by zero is handled safely (returns 0.0).
    """
    precision = _safe_div(tp, tp + fp)
    recall    = _safe_div(tp, tp + fn)
    f1        = _safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ── Main evaluator ────────────────────────────────────────────────────────────

def evaluate_dataset(df: pd.DataFrame, nlp: Language) -> dict:
    """
    Evaluate the NER model against the annotated dataset.

    Parameters
    ----------
    df  : DataFrame from load_annotated_data()
    nlp : loaded spaCy Language model

    Returns
    -------
    dict with:
        'overall'   – micro-averaged metrics (precision, recall, F1, TP, FP, FN)
        'per_label' – dict of label → metrics
        'details'   – list of per-document dicts for error drilling
    """
    total_tp = total_fp = total_fn = 0
    label_counts: dict[str, dict[str, int]] = {}
    details: list[dict] = []

    for text_id in df["text_id"].astype(str).unique():
        text       = df[df["text_id"].astype(str) == str(text_id)]["text"].iloc[0]
        ref_spans  = _ref_spans(df, text_id)
        pred_spans = _pred_spans(text, nlp)

        tp_spans = ref_spans & pred_spans
        fp_spans = pred_spans - ref_spans
        fn_spans = ref_spans - pred_spans

        tp, fp, fn = len(tp_spans), len(fp_spans), len(fn_spans)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        for (label, *_) in tp_spans:
            label_counts.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
        for (label, *_) in fp_spans:
            label_counts.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
        for (label, *_) in fn_spans:
            label_counts.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1

        details.append({
            "text_id":  text_id,
            "text":     text[:100] + ("…" if len(text) > 100 else ""),
            "tp": tp, "fp": fp, "fn": fn,
            "tp_spans": sorted(tp_spans),
            "fp_spans": sorted(fp_spans),
            "fn_spans": sorted(fn_spans),
        })

    per_label = {
        lbl: compute_metrics(c["tp"], c["fp"], c["fn"])
        for lbl, c in sorted(label_counts.items())
    }

    return {
        "overall":   compute_metrics(total_tp, total_fp, total_fn),
        "per_label": per_label,
        "details":   details,
    }

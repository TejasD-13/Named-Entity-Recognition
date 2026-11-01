"""
error_analysis.py – Error analysis for NER Explorer.

Compares predicted entities against reference annotations and assigns
each discrepancy to one of five categories:

CORRECT          – span and label both match exactly (true positive)
MISSED           – reference entity absent from predictions (false negative)
EXTRA            – predicted entity absent from reference (false positive)
WRONG_LABEL      – span matches but the label differs
BOUNDARY_MISMATCH– overlapping but non-identical spans
"""
from __future__ import annotations

import pandas as pd
from spacy.language import Language

from .ner_engine import extract_entities


# ── Helpers ───────────────────────────────────────────────────────────────────

def _entity_map(entities: list[dict]) -> dict[tuple, dict]:
    """Map (start_char, end_char) → entity dict for fast lookup."""
    return {(e["start_char"], e["end_char"]): e for e in entities}


def _spans_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    return s1 < e2 and s2 < e1


def _find_overlapping_pred(
    start: int, end: int, pred_map: dict[tuple, dict]
) -> dict | None:
    for (s, e), ent in pred_map.items():
        if _spans_overlap(start, end, s, e):
            return ent
    return None


def _context(text: str, start: int, end: int, window: int = 40) -> str:
    """Return a short snippet of text around the entity."""
    left   = max(0, start - window)
    right  = min(len(text), end + window)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(text) else ""
    return prefix + text[left:right] + suffix


# ── Main analysis function ────────────────────────────────────────────────────

def analyse_errors(
    text: str,
    reference_entities: list[dict],
    nlp: Language,
) -> list[dict]:
    """
    Compare predicted entities against reference annotations.

    Parameters
    ----------
    text                : input text
    reference_entities  : list of dicts with keys text, label,
                          start_char, end_char (from reference data)
    nlp                 : loaded spaCy pipeline

    Returns
    -------
    list of error-record dicts, one per entity (reference or predicted).
    Each record has keys:
        category, ref_text, ref_label, pred_text, pred_label, context
    """
    result    = extract_entities(text, nlp)
    predicted = result["entities"]

    ref_map  = _entity_map(reference_entities)
    pred_map = _entity_map(predicted)

    errors: list[dict] = []
    handled_pred_spans: set[tuple] = set()

    # ── Iterate over every reference entity ──────────────────────────────────
    for (s, e), ref in ref_map.items():
        if (s, e) in pred_map:
            pred = pred_map[(s, e)]
            handled_pred_spans.add((s, e))
            if pred["label"] == ref["label"]:
                cat = "CORRECT"
            else:
                cat = "WRONG_LABEL"
            errors.append({
                "category":   cat,
                "ref_text":   ref["text"],
                "ref_label":  ref["label"],
                "pred_text":  pred["text"],
                "pred_label": pred["label"],
                "context":    _context(text, s, e),
            })
        else:
            # Check for partial boundary overlap
            overlap = _find_overlapping_pred(s, e, pred_map)
            if overlap:
                overlap_key = (overlap["start_char"], overlap["end_char"])
                handled_pred_spans.add(overlap_key)
                errors.append({
                    "category":   "BOUNDARY_MISMATCH",
                    "ref_text":   ref["text"],
                    "ref_label":  ref["label"],
                    "pred_text":  overlap["text"],
                    "pred_label": overlap["label"],
                    "context":    _context(text, s, e),
                })
            else:
                errors.append({
                    "category":   "MISSED",
                    "ref_text":   ref["text"],
                    "ref_label":  ref["label"],
                    "pred_text":  "—",
                    "pred_label": "—",
                    "context":    _context(text, s, e),
                })

    # ── Extra predictions (false positives) ──────────────────────────────────
    for (s, e), pred in pred_map.items():
        if (s, e) not in handled_pred_spans:
            errors.append({
                "category":   "EXTRA",
                "ref_text":   "—",
                "ref_label":  "—",
                "pred_text":  pred["text"],
                "pred_label": pred["label"],
                "context":    _context(text, s, e),
            })

    return errors


def summarise_errors(errors: list[dict]) -> pd.DataFrame:
    """Convert a list of error records to a labelled DataFrame."""
    if not errors:
        return pd.DataFrame(columns=[
            "Category", "Reference Text", "Ref Label",
            "Predicted Text", "Pred Label", "Context",
        ])
    return pd.DataFrame([{
        "Category":       e["category"],
        "Reference Text": e["ref_text"],
        "Ref Label":      e["ref_label"],
        "Predicted Text": e["pred_text"],
        "Pred Label":     e["pred_label"],
        "Context":        e["context"],
    } for e in errors])


# ── Possible error causes ─────────────────────────────────────────────────────

ERROR_EXPLANATIONS = {
    "CORRECT":           "Correctly predicted — span and label match the reference.",
    "MISSED":            "False negative — the model did not detect this entity. "
                         "Possible causes: unseen name, unusual context, domain mismatch.",
    "EXTRA":             "False positive — the model predicted an entity not in the "
                         "reference. Possible causes: over-generalisation, ambiguous word.",
    "WRONG_LABEL":       "Span is correct but the label is wrong. "
                         "Possible causes: ambiguity (e.g. 'Washington' as PERSON vs GPE).",
    "BOUNDARY_MISMATCH": "Overlapping spans — the model found part of the entity "
                         "or included extra tokens.",
}

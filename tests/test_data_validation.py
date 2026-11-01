"""
test_data_validation.py – Validates the annotated demonstration dataset.

Run with:  pytest tests/test_data_validation.py -v

Requires:  data/annotated_test_data.csv (included in the project)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd

from src.evaluation import load_annotated_data, validate_annotations, REQUIRED_COLUMNS

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "annotated_test_data.csv",
)
DATASET_EXISTS = os.path.isfile(DATASET_PATH)
_SKIP_IF_MISSING = pytest.mark.skipif(
    not DATASET_EXISTS,
    reason=f"Dataset not found: {DATASET_PATH}",
)


@_SKIP_IF_MISSING
def test_dataset_file_exists():
    assert os.path.isfile(DATASET_PATH)


@_SKIP_IF_MISSING
def test_required_columns_present():
    df = pd.read_csv(DATASET_PATH)
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


@_SKIP_IF_MISSING
def test_text_id_not_empty():
    df = pd.read_csv(DATASET_PATH)
    assert df["text_id"].notna().all(), "Some text_id values are null"


@_SKIP_IF_MISSING
def test_text_column_not_empty():
    df = pd.read_csv(DATASET_PATH)
    assert df["text"].notna().all(), "Some text values are null"


@_SKIP_IF_MISSING
def test_annotation_offsets_valid():
    df = load_annotated_data(DATASET_PATH)
    warnings = validate_annotations(df)
    assert warnings == [], (
        f"{len(warnings)} annotation offset mismatch(es) found:\n"
        + "\n".join(warnings)
    )


@_SKIP_IF_MISSING
def test_entity_start_less_than_end():
    df = pd.read_csv(DATASET_PATH)
    entity_rows = df[
        df["entity_text"].notna() &
        ~df["entity_text"].str.strip().str.upper().eq("NO_ENTITY")
    ]
    bad = entity_rows[entity_rows["start_char"].astype(int) >= entity_rows["end_char"].astype(int)]
    assert bad.empty, f"Rows with start_char >= end_char:\n{bad[['text_id','entity_text','start_char','end_char']]}"


@_SKIP_IF_MISSING
def test_dataset_has_multiple_label_types():
    df = pd.read_csv(DATASET_PATH)
    entity_rows = df[
        df["entity_text"].notna() &
        ~df["entity_text"].str.strip().str.upper().eq("NO_ENTITY")
    ]
    unique_labels = entity_rows["entity_label"].nunique()
    assert unique_labels >= 4, (
        f"Expected at least 4 different entity labels, found {unique_labels}"
    )


@_SKIP_IF_MISSING
def test_dataset_contains_no_entity_row():
    df = pd.read_csv(DATASET_PATH)
    has_no_entity = df["entity_text"].str.strip().str.upper().eq("NO_ENTITY").any()
    assert has_no_entity, "Dataset should contain at least one NO_ENTITY row"

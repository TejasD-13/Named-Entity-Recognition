"""
test_ner_engine.py – Unit tests for src/ner_engine.py

Requires en_core_web_sm:  python -m spacy download en_core_web_sm
Run with:  pytest tests/test_ner_engine.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from src.ner_engine import (
    load_model, extract_entities, get_model_info,
    ModelNotFoundError, LABEL_DESCRIPTIONS,
)

# ── Model availability check ──────────────────────────────────────────────────
try:
    _nlp = load_model("en_core_web_sm")
    MODEL_AVAILABLE = True
except ModelNotFoundError:
    MODEL_AVAILABLE = False

_SKIP = pytest.mark.skipif(not MODEL_AVAILABLE, reason="en_core_web_sm not installed")


# ── load_model ────────────────────────────────────────────────────────────────

def test_load_nonexistent_model_raises():
    with pytest.raises(ModelNotFoundError):
        load_model("nonexistent_model_xyz_999")


@_SKIP
def test_load_model_returns_object():
    assert _nlp is not None


# ── extract_entities ──────────────────────────────────────────────────────────

@_SKIP
class TestExtractEntities:
    def test_returns_required_keys(self):
        result = extract_entities("Hello world.", _nlp)
        assert "text" in result
        assert "entities" in result
        assert "doc" in result

    def test_text_field_matches_input(self):
        text = "Steve Jobs founded Apple."
        assert extract_entities(text, _nlp)["text"] == text

    def test_entities_is_list(self):
        assert isinstance(extract_entities("Hello.", _nlp)["entities"], list)

    def test_entity_dict_has_required_fields(self):
        result = extract_entities("Apple Inc. is in California.", _nlp)
        for ent in result["entities"]:
            assert "text"        in ent
            assert "label"       in ent
            assert "description" in ent
            assert "start_char"  in ent
            assert "end_char"    in ent

    def test_character_offsets_are_correct(self):
        text = "Steve Jobs founded Apple Inc. in California."
        result = extract_entities(text, _nlp)
        for ent in result["entities"]:
            s, e = ent["start_char"], ent["end_char"]
            extracted = text[s:e]
            assert extracted == ent["text"], (
                f"Offset mismatch for '{ent['text']}': "
                f"text[{s}:{e}] = '{extracted}'"
            )

    def test_end_greater_than_start(self):
        result = extract_entities("Barack Obama was the 44th President.", _nlp)
        for ent in result["entities"]:
            assert ent["end_char"] > ent["start_char"]

    def test_no_entities_gives_empty_list(self):
        # Very generic sentence – may or may not have entities; just check type
        result = extract_entities("The process runs smoothly.", _nlp)
        assert isinstance(result["entities"], list)

    def test_multiple_entities_detected(self):
        text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
        result = extract_entities(text, _nlp)
        assert len(result["entities"]) >= 2

    def test_description_is_string(self):
        result = extract_entities("Apple was founded in California.", _nlp)
        for ent in result["entities"]:
            assert isinstance(ent["description"], str)


# ── get_model_info ────────────────────────────────────────────────────────────

@_SKIP
def test_get_model_info_keys():
    info = get_model_info(_nlp)
    for key in ("model_name", "version", "lang", "pipeline", "ner_labels"):
        assert key in info


@_SKIP
def test_ner_labels_is_list():
    info = get_model_info(_nlp)
    assert isinstance(info["ner_labels"], list)
    assert len(info["ner_labels"]) > 0


# ── LABEL_DESCRIPTIONS ────────────────────────────────────────────────────────

def test_label_descriptions_not_empty():
    assert len(LABEL_DESCRIPTIONS) > 0


def test_label_descriptions_values_are_strings():
    for k, v in LABEL_DESCRIPTIONS.items():
        assert isinstance(v, str), f"Description for {k} is not a string"

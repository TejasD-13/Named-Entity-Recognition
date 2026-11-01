"""
test_preprocessing.py – Unit tests for src/preprocessing.py

Run with:  pytest tests/test_preprocessing.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from src.preprocessing import (
    preprocess_text, PreprocessingError,
    validate_input, remove_control_characters, normalize_whitespace,
)


# ── validate_input ────────────────────────────────────────────────────────────

class TestValidateInput:
    def test_empty_string_raises(self):
        with pytest.raises(PreprocessingError, match="empty"):
            validate_input("")

    def test_whitespace_only_raises(self):
        with pytest.raises(PreprocessingError, match="empty"):
            validate_input("   \n\t  ")

    def test_non_string_raises(self):
        with pytest.raises(PreprocessingError, match="string"):
            validate_input(None)  # type: ignore

    def test_integer_raises(self):
        with pytest.raises(PreprocessingError):
            validate_input(42)  # type: ignore

    def test_too_long_raises(self):
        with pytest.raises(PreprocessingError, match="too long"):
            validate_input("a" * 101, max_chars=100)

    def test_exactly_max_length_passes(self):
        validate_input("a" * 100, max_chars=100)   # should not raise

    def test_valid_text_passes(self):
        validate_input("Hello, world!")             # should not raise


# ── remove_control_characters ─────────────────────────────────────────────────

class TestRemoveControlCharacters:
    def test_preserves_normal_text(self):
        text = "Hello, World! 123"
        assert remove_control_characters(text) == text

    def test_preserves_newlines(self):
        assert "\n" in remove_control_characters("Line 1\nLine 2")

    def test_preserves_tabs(self):
        assert "\t" in remove_control_characters("Col1\tCol2")

    def test_removes_null_byte(self):
        result = remove_control_characters("Hello\x00World")
        assert "\x00" not in result
        assert "Hello" in result
        assert "World" in result

    def test_removes_bell_character(self):
        assert "\x07" not in remove_control_characters("Hello\x07World")

    def test_removes_escape_character(self):
        assert "\x1b" not in remove_control_characters("Esc\x1bSeq")

    def test_preserves_dollar_sign(self):
        assert "$" in remove_control_characters("$100")

    def test_preserves_euro_sign(self):
        assert "€" in remove_control_characters("€200")

    def test_preserves_rupee_sign(self):
        assert "₹" in remove_control_characters("₹500")

    def test_preserves_unicode_letters(self):
        text = "Café résumé naïve"
        assert remove_control_characters(text) == text

    def test_empty_string_returns_empty(self):
        assert remove_control_characters("") == ""


# ── normalize_whitespace ──────────────────────────────────────────────────────

class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("Hello   world") == "Hello world"

    def test_collapses_mixed_spaces_tabs(self):
        assert normalize_whitespace("Hello \t  world") == "Hello world"

    def test_strips_leading_trailing(self):
        assert normalize_whitespace("  hello  ") == "hello"

    def test_preserves_single_newline(self):
        result = normalize_whitespace("Line 1\nLine 2")
        assert "\n" in result

    def test_collapses_three_newlines_to_two(self):
        result = normalize_whitespace("A\n\n\nB")
        assert "\n\n\n" not in result
        assert "\n\n" in result

    def test_empty_string_returns_empty(self):
        assert normalize_whitespace("") == ""


# ── preprocess_text (full pipeline) ──────────────────────────────────────────

class TestPreprocessText:
    def test_empty_raises(self):
        with pytest.raises(PreprocessingError):
            preprocess_text("")

    def test_whitespace_only_raises(self):
        with pytest.raises(PreprocessingError):
            preprocess_text("   ")

    def test_normal_text_preserved(self):
        text = "Apple Inc. was founded by Steve Jobs in California."
        assert preprocess_text(text) == text

    def test_multiple_spaces_collapsed(self):
        assert preprocess_text("Hello   world") == "Hello world"

    def test_control_chars_removed(self):
        result = preprocess_text("Hello\x00World")
        assert "\x00" not in result

    def test_preserves_capitalization(self):
        text = "Steve Jobs founded Apple."
        result = preprocess_text(text)
        assert "Steve Jobs" in result
        assert "Apple" in result

    def test_preserves_punctuation(self):
        text = "Hello, world! How are you?"
        result = preprocess_text(text)
        assert "," in result
        assert "!" in result
        assert "?" in result

    def test_preserves_date_string(self):
        text = "Founded on 01/01/2024 at 3:30 PM."
        result = preprocess_text(text)
        assert "01/01/2024" in result
        assert "3:30 PM" in result

    def test_preserves_currency(self):
        text = "Valued at $3 trillion."
        result = preprocess_text(text)
        assert "$3 trillion" in result

    def test_multiword_entity_preserved(self):
        text = "The United Nations was founded in San Francisco."
        result = preprocess_text(text)
        assert "United Nations" in result
        assert "San Francisco" in result

    def test_unicode_name_preserved(self):
        text = "Narendra Modi visited New Delhi."
        assert "Narendra Modi" in preprocess_text(text)

    def test_too_long_raises(self):
        with pytest.raises(PreprocessingError, match="too long"):
            preprocess_text("a" * 100_001)

    def test_newlines_and_tabs_not_stripped(self):
        text = "Para 1.\n\nPara 2."
        result = preprocess_text(text)
        assert "\n" in result

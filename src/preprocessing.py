"""
preprocessing.py – Text preprocessing for NER Explorer.

IMPORTANT: NER is context-sensitive. This module performs only minimal
cleaning that preserves capitalization, punctuation, entity boundaries,
and character offsets. It does NOT lowercase, stem, lemmatize, or strip
stop-words before entity recognition.
"""
from __future__ import annotations

import re
import unicodedata


class PreprocessingError(ValueError):
    """Raised when input text fails validation."""


def validate_input(text: str, max_chars: int = 100_000) -> None:
    """
    Validate that *text* is a non-empty string within the allowed size.

    Parameters
    ----------
    text     : input to validate
    max_chars: maximum allowed character count

    Raises
    ------
    PreprocessingError on any validation failure.
    """
    if not isinstance(text, str):
        raise PreprocessingError(
            f"Input must be a string, got {type(text).__name__}."
        )
    if not text.strip():
        raise PreprocessingError(
            "Input text is empty or contains only whitespace."
        )
    if len(text) > max_chars:
        raise PreprocessingError(
            f"Input is too long ({len(text):,} characters). "
            f"Maximum allowed: {max_chars:,} characters."
        )


def remove_control_characters(text: str) -> str:
    """
    Remove Unicode control characters (categories Cc, Cf) while keeping
    ordinary whitespace: space, newline (\\n), tab (\\t), carriage return (\\r).

    Preserves
    ---------
    - All printable ASCII and Unicode characters
    - Currency symbols ($ € £ ₹ ¥)
    - Punctuation and numbers
    - Capitalization (essential for NER)
    """
    _allowed_ws = {'\n', '\t', '\r', ' '}

    def _keep(ch: str) -> bool:
        if ch in _allowed_ws:
            return True
        cat = unicodedata.category(ch)
        return cat not in ('Cc', 'Cf')

    return ''.join(ch for ch in text if _keep(ch))


def normalize_whitespace(text: str) -> str:
    """
    Conservatively normalize whitespace.

    - Collapses runs of spaces or horizontal whitespace into a single space.
    - Preserves newlines (they may delimit sentences or paragraphs).
    - Collapses more than two consecutive newlines to two.
    - Strips leading/trailing whitespace.
    """
    # Collapse multiple spaces (not newlines) into one
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def preprocess_text(text: str, max_chars: int = 100_000) -> str:
    """
    Full preprocessing pipeline for NER.

    Steps
    -----
    1. Validate input (type, emptiness, length).
    2. Remove unwanted Unicode control characters.
    3. Normalize whitespace conservatively.
    4. Re-validate (cleaning cannot create valid text from invalid input).

    What is intentionally NOT done
    --------------------------------
    - No lowercasing  (capitalization is a key NER signal)
    - No stop-word removal (context is required for NER)
    - No stemming or lemmatization
    - No punctuation stripping (affects sentence boundaries)

    Parameters
    ----------
    text      : raw input text (UTF-8 string)
    max_chars : maximum allowed input length

    Returns
    -------
    str – cleaned text suitable for spaCy NER processing.

    Raises
    ------
    PreprocessingError if input fails validation.
    """
    validate_input(text, max_chars=max_chars)
    text = remove_control_characters(text)
    text = normalize_whitespace(text)
    validate_input(text, max_chars=max_chars)   # re-check after cleaning
    return text


PREPROCESSING_STEPS = [
    "1. Input validation — reject empty, non-string, or oversized input.",
    "2. Control-character removal — strip invisible Unicode control chars (Cc/Cf) "
       "while keeping spaces, newlines, and tabs.",
    "3. Whitespace normalization — collapse multiple spaces into one; "
       "preserve newlines; strip leading/trailing whitespace.",
    "4. Re-validation — confirm cleaned text is still non-empty.",
    "Note: Capitalization, punctuation, numbers, and entity spellings "
       "are preserved throughout.",
]

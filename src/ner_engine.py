"""
ner_engine.py – NER processing engine for NER Explorer.

Loads a spaCy language model and extracts named entities from text.
All predictions come directly from the spaCy pipeline – no hand-crafted
rules or hardcoded entity lists are substituted for model inference.
"""
from __future__ import annotations

import spacy
from spacy.language import Language


# Friendly descriptions for standard OntoNotes / spaCy entity labels
LABEL_DESCRIPTIONS: dict[str, str] = {
    "PERSON":      "People, including fictional characters",
    "ORG":         "Companies, agencies, institutions",
    "GPE":         "Countries, cities, states",
    "LOC":         "Non-GPE locations (mountain ranges, water bodies, etc.)",
    "DATE":        "Absolute or relative dates or periods",
    "TIME":        "Times smaller than a day",
    "MONEY":       "Monetary values, including unit",
    "PERCENT":     "Percentage values",
    "PRODUCT":     "Objects, vehicles, foods, etc.",
    "EVENT":       "Named events (hurricanes, battles, sports events, etc.)",
    "WORK_OF_ART": "Titles of books, songs, films, etc.",
    "LAW":         "Named documents enacted into law",
    "LANGUAGE":    "Any named language",
    "NORP":        "Nationalities or religious / political groups",
    "FAC":         "Buildings, airports, highways, bridges, etc.",
    "QUANTITY":    "Measurements, weights, or amounts",
    "ORDINAL":     "Ordinal numbers (first, second, etc.)",
    "CARDINAL":    "Numerals not classified elsewhere",
}

# Default model name – change here or pass explicitly to load_model()
DEFAULT_MODEL = "en_core_web_sm"


class ModelNotFoundError(RuntimeError):
    """Raised when a spaCy model cannot be loaded."""


def load_model(model_name: str = DEFAULT_MODEL) -> Language:
    """
    Load a spaCy language model by name.

    Parameters
    ----------
    model_name : str
        Name of the installed spaCy pipeline, e.g. 'en_core_web_sm'.

    Returns
    -------
    spacy.Language – the loaded pipeline.

    Raises
    ------
    ModelNotFoundError if the model is not installed.
    """
    try:
        return spacy.load(model_name)
    except OSError:
        raise ModelNotFoundError(
            f"spaCy model '{model_name}' is not installed.\n"
            f"Install it with:\n  python -m spacy download {model_name}"
        )


def extract_entities(text: str, nlp: Language) -> dict:
    """
    Run the spaCy NLP pipeline on *text* and extract named entities.

    Parameters
    ----------
    text : str
        Preprocessed input text.
    nlp  : spacy.Language
        Loaded spaCy pipeline.

    Returns
    -------
    dict:
        'text'     – original text passed in
        'entities' – list of entity dicts, each containing:
                       text, label, description, start_char, end_char
        'doc'      – the spaCy Doc object (for displaCy rendering)

    Notes
    -----
    - Character offsets are taken directly from spaCy's Span objects.
    - The description is looked up in LABEL_DESCRIPTIONS; if the label
      is unknown, spacy.explain() is used as a fallback.
    """
    doc = nlp(text)
    entities: list[dict] = []
    for ent in doc.ents:
        entities.append({
            "text":        ent.text,
            "label":       ent.label_,
            "description": LABEL_DESCRIPTIONS.get(
                               ent.label_,
                               spacy.explain(ent.label_) or ent.label_
                           ),
            "start_char":  ent.start_char,
            "end_char":    ent.end_char,
        })
    return {
        "text":     text,
        "entities": entities,
        "doc":      doc,
    }


def get_model_info(nlp: Language) -> dict:
    """
    Return metadata about the loaded spaCy model.

    Returns
    -------
    dict with keys:
        model_name, version, lang, pipeline, ner_labels, description
    """
    ner_labels: list[str] = []
    if nlp.has_pipe("ner"):
        ner_labels = sorted(nlp.get_pipe("ner").labels)
    return {
        "model_name":  nlp.meta.get("name", "unknown"),
        "version":     nlp.meta.get("version", "unknown"),
        "lang":        nlp.meta.get("lang", "unknown"),
        "pipeline":    list(nlp.pipe_names),
        "ner_labels":  ner_labels,
        "description": nlp.meta.get("description", ""),
        "model_type":  "CNN-based spaCy pipeline (en_core_web_sm). "
                       "Not transformer-based.",
    }

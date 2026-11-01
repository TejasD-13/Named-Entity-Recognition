# Data Directory

This directory contains the NER Explorer demonstration dataset.

## Files

### `sample_inputs.txt`
Ten diverse English sentences used as quick test inputs in the app.
Covers people, organizations, locations, dates, monetary values, Indian names,
and a sentence with no named entities.

### `annotated_test_data.csv`
Manually annotated reference dataset for evaluation and error analysis.

#### Format

| Column        | Type   | Description                                              |
|---------------|--------|----------------------------------------------------------|
| `text_id`     | int    | Unique document identifier                               |
| `text`        | str    | Full input sentence                                      |
| `entity_text` | str    | Entity surface form, or `NO_ENTITY` if none present      |
| `entity_label`| str    | Entity type label (PERSON, ORG, GPE, …)                  |
| `start_char`  | int    | Character offset of entity start (0-indexed, inclusive)  |
| `end_char`    | int    | Character offset of entity end (exclusive)               |

#### Annotation rules
- One row per entity.
- `text[start_char:end_char]` must equal `entity_text` exactly.
- Rows with `entity_text = NO_ENTITY` represent entity-free sentences.
- Labels follow the OntoNotes 5 / spaCy standard.

#### Important disclaimer
This is a **demonstration dataset** (10 sentences, ~31 annotated entities).
Results obtained on this dataset **cannot** be generalised as benchmark
performance of en_core_web_sm on real-world text.

#### Validating offsets
```python
from src.evaluation import load_annotated_data, validate_annotations
df = load_annotated_data("data/annotated_test_data.csv")
warnings = validate_annotations(df)
print(warnings or "All offsets valid.")
```

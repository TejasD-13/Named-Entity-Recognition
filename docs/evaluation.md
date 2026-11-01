# Evaluation Guide

## Overview

NER Explorer evaluates the spaCy model using **exact-span matching** against
a manually annotated reference dataset.

---

## Dataset Format

The annotated CSV (`data/annotated_test_data.csv`) has these columns:

| Column | Type | Description |
|---|---|---|
| `text_id` | str/int | Unique document identifier |
| `text` | str | Full input sentence |
| `entity_text` | str | Surface form of the entity, or `NO_ENTITY` |
| `entity_label` | str | Label: PERSON, ORG, GPE, DATE, etc. |
| `start_char` | int | Character offset (0-indexed, inclusive) |
| `end_char` | int | Character offset (exclusive) |

One row per entity. For entity-free sentences, use `entity_text = NO_ENTITY`.

### Offset rule
`text[start_char : end_char]` must equal `entity_text` exactly.

---

## Evaluation Steps

1. Load the CSV with `load_annotated_data(path)`
2. Validate offsets with `validate_annotations(df)` — check warnings
3. Call `evaluate_dataset(df, nlp)` which:
   - Iterates over each unique `text_id`
   - Runs spaCy on the full `text`
   - Matches predicted spans against reference spans (label + start + end)
   - Accumulates TP, FP, FN counts

---

## Metrics Definitions

```
True Positive  (TP): model predicted a span that exists in the reference
False Positive (FP): model predicted a span NOT in the reference
False Negative (FN): reference span NOT predicted by the model

Precision  = TP / (TP + FP)    → how many predictions are correct
Recall     = TP / (TP + FN)    → how many reference entities were found
F1-score   = 2 × P × R / (P + R)  → harmonic mean of P and R
```

Division by zero → returns 0.0 (handled by `_safe_div`).

---

## Micro vs Per-Label

- **Micro-averaged** (Overall): TP, FP, FN summed across all entity types, then metrics computed from totals. Treats every entity equally regardless of type.
- **Per-label**: TP, FP, FN computed separately for each label. Useful for identifying which entity types the model struggles with.

---

## Demonstration Dataset Results

> **Important:** The included demonstration dataset contains only 10 sentences
> (~31 entities). Results on this dataset **do not represent** general NER
> accuracy on real-world text.

To obtain actual results:
1. Run the **📐 Evaluation** tab in the app
2. Click **▶ Run Evaluation** with the demo dataset enabled
3. Record the precision, recall, and F1 values shown

---

## Extending the Evaluation

To evaluate on your own data:
1. Create a CSV following the format above
2. Validate offsets:
   ```python
   from src.evaluation import load_annotated_data, validate_annotations
   df = load_annotated_data("your_file.csv")
   print(validate_annotations(df))
   ```
3. Upload the CSV in the app's Evaluation tab

---

## Improving Accuracy

| Approach | Notes |
|---|---|
| Use `en_core_web_lg` | Larger model, higher accuracy |
| Use `en_core_web_trf` | Transformer-based, best accuracy, slower |
| Fine-tune on domain data | Requires annotated in-domain corpus |

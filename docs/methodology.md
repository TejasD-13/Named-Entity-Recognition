# Methodology

## NER Explorer: Named Entity Recognition System

### System Architecture

```
User Input (Text / File)
        │
        ▼
  Input Validation
  (type, length, emptiness)
        │
        ▼
  Control-Character Removal
  (strip Unicode Cc/Cf, keep whitespace)
        │
        ▼
  Whitespace Normalization
  (collapse spaces, preserve newlines)
        │
        ▼
  spaCy NLP Pipeline  [en_core_web_sm]
  ┌─────────────────────────────────┐
  │  Tokenizer → Tagger → Parser   │
  │         → NER (CNN)            │
  └─────────────────────────────────┘
        │
        ▼
  Entity Span Extraction
  (text, label, start_char, end_char)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  Highlighted Text              Optional Evaluation
  + Entity Table                Against Annotated CSV
  + Statistics                  (Precision / Recall / F1)
                                        │
                                        ▼
                                  Error Analysis
                                  (CORRECT / MISSED /
                                   EXTRA / WRONG_LABEL /
                                   BOUNDARY_MISMATCH)
```

---

### Preprocessing Design Decisions

NER is fundamentally context-dependent. The following operations are
**intentionally excluded** from preprocessing:

| Operation | Reason excluded |
|---|---|
| Lowercasing | Capital letters signal proper nouns (key NER feature) |
| Stop-word removal | Context words help the model determine entity type |
| Stemming / Lemmatization | Alters surface forms; breaks character offsets |
| Punctuation stripping | Sentence boundaries help the parser |

Only minimal cleaning is performed:
1. Remove invisible Unicode control characters (Cc / Cf categories)
2. Normalize excessive whitespace (multiple spaces → one)
3. Preserve newlines (paragraph structure aids tokenization)

---

### spaCy Pipeline (en_core_web_sm)

The selected model is a **CNN-based** (not transformer-based) English NLP
pipeline trained on the **OntoNotes 5** corpus.

Pipeline components:
- **tok2vec** – shared token-to-vector representation (CNN)
- **tagger** – part-of-speech tagging
- **parser** – dependency parsing (used implicitly)
- **ner** – named entity recognition using B-I-O tagging

#### B-I-O Tagging Scheme

Each token receives one of:
- `B-LABEL` – **B**eginning of an entity of type LABEL
- `I-LABEL` – **I**nside a continuing entity
- `O` – **O**utside any entity

Adjacent tokens with matching B/I labels are merged into a single span.

Example:
```
Token      Tag
Steve    → B-PERSON
Jobs     → I-PERSON
founded  → O
Apple    → B-ORG
```
→ Entities: "Steve Jobs" (PERSON), "Apple" (ORG)

---

### Evaluation Protocol

- **Matching criterion:** exact span — `(label, start_char, end_char)` must all match
- **Micro-averaging:** counts are summed across all documents before computing ratios
- **Per-label metrics:** computed independently for each entity type
- **Safe division:** 0.0 returned when denominator is zero

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × P × R / (P + R)
```

---

### Error Categories

| Category | Description |
|---|---|
| CORRECT | Span and label match exactly |
| MISSED | Reference entity not predicted (FN) |
| EXTRA | Predicted entity not in reference (FP) |
| WRONG_LABEL | Span matches but label differs |
| BOUNDARY_MISMATCH | Overlapping but non-identical spans |

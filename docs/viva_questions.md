# Viva Questions & Answers

## NER Explorer — NLP Laboratory Assignment

---

### 1. What is Named Entity Recognition?

**A:** Named Entity Recognition (NER) is an NLP task that locates and classifies named entities in text into predefined categories such as PERSON, ORG, GPE, DATE, and MONEY. For example, in "Apple was founded by Steve Jobs in 1976", NER identifies "Apple" as an ORG, "Steve Jobs" as a PERSON, and "1976" as a DATE.

---

### 2. Which model does your project use and why?

**A:** The project uses `en_core_web_sm`, a small CNN-based English NLP pipeline from spaCy trained on the OntoNotes 5 corpus. It was chosen because it is lightweight, fast, requires no GPU, installs with a single command, and is well-suited for a demonstration on a standard laptop. It is **not** a transformer-based model. For higher accuracy, `en_core_web_lg` or `en_core_web_trf` could be used.

---

### 3. What is the B-I-O tagging scheme?

**A:** B-I-O (Beginning, Inside, Outside) is a token-level labeling scheme used in sequence labeling.
- `B-LABEL`: the first token of an entity of that type
- `I-LABEL`: a continuation token inside the same entity
- `O`: a token that is not part of any entity

Adjacent B/I tokens with matching labels are merged into a single entity span.

---

### 4. Why do you not lowercase the text before NER?

**A:** Capitalization is one of the most important signals for NER. "Apple" (company) vs "apple" (fruit), or "Washington" (person) vs "washington" (unlikely). Lowercasing destroys this signal and significantly reduces NER accuracy. Our preprocessing intentionally preserves capitalization.

---

### 5. What preprocessing steps do you apply?

**A:** Only three minimal steps:
1. Input validation (reject empty, non-string, oversized input)
2. Remove invisible Unicode control characters (Cc/Cf categories) while keeping spaces, newlines, and tabs
3. Normalize excessive whitespace (multiple spaces → one space)

We do NOT lowercase, stem, lemmatize, remove stop words, or strip punctuation — all of these would hurt NER performance.

---

### 6. How do you evaluate the NER model?

**A:** We use **exact-span matching**: a prediction is correct only when both the character span (`start_char`, `end_char`) and the entity label exactly match the annotated reference. We compute:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-score = 2 × P × R / (P + R)

Both micro-averaged (overall) and per-entity-type metrics are reported.

---

### 7. What is the difference between Precision and Recall?

**A:**
- **Precision**: Of all entities the model predicted, what fraction were correct? (measures false positives)
- **Recall**: Of all entities in the reference, what fraction did the model find? (measures false negatives)
- A model can have high precision with low recall (conservative, misses many) or high recall with low precision (greedy, predicts too many).
- F1-score is the harmonic mean that balances both.

---

### 8. What error categories do you identify?

**A:** Five categories:
- **CORRECT**: span and label both match
- **MISSED** (FN): reference entity not predicted by the model
- **EXTRA** (FP): prediction not in the reference
- **WRONG_LABEL**: span matches but the label is wrong (e.g., "Washington" predicted as PERSON instead of GPE)
- **BOUNDARY_MISMATCH**: overlapping but non-identical spans (e.g., model predicted "Steve" instead of "Steve Jobs")

---

### 9. What are the limitations of en_core_web_sm?

**A:**
- Small CNN model — lower accuracy than large or transformer-based models
- Trained on newswire/broadcast data — lower performance on domain-specific text
- May miss rare names (Indian names, uncommon organizations)
- No confidence scores per prediction
- Boundary detection may fail for complex multi-word entities
- Some entity types (FAC, WORK_OF_ART, LAW) have fewer training examples

---

### 10. How does context affect NER predictions?

**A:** The same word can be different entity types depending on context:
- "Washington" in "Washington signed the treaty" → PERSON
- "Washington" in "I live in Washington" → GPE (city/state)
- "Apple" in "I ate an apple" → not an entity
- "Apple" in "Apple released a new product" → ORG

spaCy's CNN reads surrounding tokens before assigning a label, which is why we preserve full sentence context during preprocessing.

---

### 11. What is the difference between micro and macro averaging?

**A:**
- **Micro-averaging**: Aggregate TP, FP, FN across all classes first, then compute metrics from totals. Gives equal weight to each entity instance. Dominated by frequent entity types.
- **Macro-averaging**: Compute metrics per class, then average. Gives equal weight to each entity type regardless of frequency.
- This project reports **micro-averaged** overall metrics and **per-label** (not macro-averaged) metrics for individual types.

---

### 12. How would you improve this system?

**A:**
- Use a larger model (`en_core_web_lg` or `en_core_web_trf`)
- Fine-tune on domain-specific annotated data
- Add custom NER rules for entities missed by the model (spaCy EntityRuler)
- Expand the annotated evaluation dataset for more reliable metrics
- Add support for multiple languages

---

### 13. What is OntoNotes 5?

**A:** OntoNotes 5 is a large annotated corpus of English text from multiple genres (newswire, broadcast news, web blogs, telephone conversations). It is the primary training dataset for spaCy's English models. The NER annotations follow a standardized label set including PERSON, ORG, GPE, DATE, and 14 other entity types.

---

### 14. Explain your project's folder structure.

**A:**
```
app.py              – Main Streamlit application (6 tabs)
src/
  preprocessing.py  – Input validation and text cleaning
  ner_engine.py     – Model loading and entity extraction
  evaluation.py     – Exact-span matching and metrics
  error_analysis.py – Error categorization
  visualization.py  – Plotly charts
data/
  annotated_test_data.csv  – 10-sentence annotated demo dataset
  sample_inputs.txt        – Quick test sentences
tests/
  test_preprocessing.py    – 20+ unit tests for preprocessing
  test_ner_engine.py       – Model and extraction tests
  test_evaluation.py       – Metric calculation tests
  test_data_validation.py  – Dataset offset validation
docs/
  methodology.md    – Architecture and design decisions
  evaluation.md     – Evaluation protocol guide
  viva_questions.md – This file
```

---

### 15. Why do you use Streamlit instead of Flask?

**A:** Streamlit requires significantly less code for the same interactive UI. It handles state management, widget rendering, and live reloading automatically. Flask requires manual routing, HTML templates, and JavaScript. For a data-science or NLP prototype, Streamlit is faster to develop, easier to deploy (Streamlit Cloud), and more readable for other data scientists.

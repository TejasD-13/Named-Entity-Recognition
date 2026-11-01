# 🔍 NER Explorer
### A Named Entity Recognition System Using Python and spaCy

> *Identify, analyze, and evaluate named entities in text using Natural Language Processing.*

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-3.x-09A3D5)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Abstract

NER Explorer is an interactive Named Entity Recognition (NER) application that accepts unstructured English text, identifies and classifies named entities using a pretrained spaCy CNN pipeline, and presents results through entity highlighting, structured tables, and quantitative evaluation metrics. The system also supports evaluation against manually annotated reference data, error analysis across five error categories, and an academic documentation suite.

---

## ✨ Features

| Tab | Functionality |
|---|---|
| 🏠 Input | File upload or text paste, preprocessing pipeline display |
| 🏷️ Entities | Color-coded displaCy highlighting, filterable entity table |
| 📊 Statistics | Entity counts, distribution chart, per-label breakdown |
| 📐 Evaluation | Upload annotated CSV → Precision / Recall / F1-score |
| 🔬 Error Analysis | CORRECT / MISSED / EXTRA / WRONG_LABEL / BOUNDARY_MISMATCH |
| ℹ️ Model Info | Model metadata, supported labels, known limitations |

---

## 📁 Project Structure

```
Named-Entity-Recognition/
│
├── app.py                      ← Main Streamlit application (6 tabs)
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        ← Input validation & minimal text cleaning
│   ├── ner_engine.py           ← spaCy model loading & entity extraction
│   ├── evaluation.py           ← Exact-span P/R/F1 evaluation
│   ├── error_analysis.py       ← Error categorization
│   └── visualization.py        ← Plotly charts
│
├── data/
│   ├── sample_inputs.txt       ← 10 diverse test sentences
│   ├── annotated_test_data.csv ← 10-sentence annotated demo dataset
│   └── README.md
│
├── tests/
│   ├── test_preprocessing.py   ← 20+ unit tests
│   ├── test_ner_engine.py      ← Model & extraction tests
│   ├── test_evaluation.py      ← Metric calculation tests
│   └── test_data_validation.py ← Dataset offset validation
│
└── docs/
    ├── methodology.md          ← Architecture & design decisions
    ├── evaluation.md           ← Evaluation protocol guide
    └── viva_questions.md       ← 15 Q&A for college viva
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/TejasD-13/Named-Entity-Recognition.git
cd Named-Entity-Recognition
```

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
```

### Step 3 — Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Download the spaCy model
```bash
python -m spacy download en_core_web_sm
```

### Step 5 — Run the app
```bash
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected output:
- `test_preprocessing.py` — all 20 tests pass (no model required)
- `test_evaluation.py` — all 10 tests pass (no model required)
- `test_data_validation.py` — all 7 tests pass (no model required)
- `test_ner_engine.py` — 11 tests pass (requires `en_core_web_sm`)

---

## 📐 Evaluation

### Using the demo dataset
1. Open the **📐 Evaluation** tab
2. Check **"Use included demonstration dataset"**
3. Click **▶ Run Evaluation**
4. View Precision, Recall, F1-score, and per-label breakdown

### Using your own dataset
Create a CSV with these columns:

| Column | Description |
|---|---|
| `text_id` | Unique document ID |
| `text` | Full input sentence |
| `entity_text` | Entity surface form or `NO_ENTITY` |
| `entity_label` | PERSON, ORG, GPE, DATE, etc. |
| `start_char` | Start character offset (0-indexed) |
| `end_char` | End character offset (exclusive) |

Then upload it in the **📐 Evaluation** tab.

> ⚠️ The demo dataset (10 sentences) is for demonstration only.
> Results are **not** representative of general NER accuracy.

---

## 🏷️ Supported Entity Types

| Label | Description |
|---|---|
| PERSON | People, including fictional |
| ORG | Companies, agencies, institutions |
| GPE | Countries, cities, states |
| LOC | Non-GPE locations |
| DATE | Dates or periods |
| TIME | Times smaller than a day |
| MONEY | Monetary values |
| PERCENT | Percentages |
| PRODUCT | Objects, vehicles, foods |
| EVENT | Named events |
| WORK_OF_ART | Titles of books, songs, films |
| LAW | Named legal documents |
| LANGUAGE | Named languages |
| NORP | Nationalities or groups |
| FAC | Buildings, airports, bridges |
| QUANTITY | Measurements and amounts |
| ORDINAL | First, second, etc. |
| CARDINAL | Other numerals |

---

## 🧠 How It Works

1. **Input** — paste text or upload a `.txt` file
2. **Preprocessing** — control characters removed, whitespace normalized; capitalization preserved
3. **spaCy Pipeline** — tokenizer → tagger → parser → CNN NER
4. **B-I-O Tagging** — tokens tagged B-LABEL / I-LABEL / O, merged into spans
5. **Output** — highlighted text, entity table, statistics
6. **Evaluation** — exact-span matching against annotated reference data

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select `TejasD-13/Named-Entity-Recognition`, branch `main`, file `app.py`
4. Click **Deploy** — dependencies from `requirements.txt` install automatically

---

## 🛠️ Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'streamlit'` | `pip install streamlit` |
| `Model 'en_core_web_sm' not found` | `python -m spacy download en_core_web_sm` |
| `No module named 'plotly'` | `pip install plotly` |
| App crashes on upload | Ensure file is UTF-8 encoded `.txt` |
| `python -m streamlit` not found | Use full path: `& "C:\...\python.exe" -m streamlit run app.py` |

---

## 📚 References

- spaCy Documentation: https://spacy.io/usage
- OntoNotes 5 Corpus: https://catalog.ldc.upenn.edu/LDC2013T19
- Streamlit Documentation: https://docs.streamlit.io
- Plotly Documentation: https://plotly.com/python/
- Tjong Kim Sang, E.F. & De Meulder, F. (2003). Introduction to the CoNLL-2003 Shared Task

---

## 👤 Author

**Tejas Divekar**
GitHub: [@TejasD-13](https://github.com/TejasD-13)

---

## 📄 License

MIT License

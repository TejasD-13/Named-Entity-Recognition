# 🔍 Named Entity Recognition (NER) Explorer

A clean, interactive **Named Entity Recognition** web app built with **Streamlit** and **spaCy**. Upload a text file or paste any text and instantly highlight named entities — people, places, organizations, dates, and much more.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-3.x-09A3D5?logo=spacy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- 📄 **Upload `.txt` files** or **paste text directly** into the app
- 🏷️ **Color-coded entity highlighting** powered by spaCy's `displacy`
- 📊 **Live stats** — total entities, unique types, word count
- 📋 **Sortable entity table** with label and description for each entity
- 🗂️ **Entity type reference legend** covering all 18 NER categories
- 🎨 **Clean, bright UI** — white theme, blue accents, Inter font

---

## 🖼️ Demo

| Input Panel | Results Panel |
|---|---|
| Upload file or paste text | Highlighted entities + stats table |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/TejasD-13/Named-Entity-Recognition.git
cd Named-Entity-Recognition
```

### 2. Install dependencies

```bash
pip install streamlit spacy pandas
```

### 3. Download the spaCy English model

```bash
python -m spacy download en_core_web_sm
```

### 4. Run the app

```bash
python -m streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web app framework |
| `spacy` | NLP engine for NER |
| `en_core_web_sm` | Pre-trained English language model |
| `pandas` | Entity table display |

---

## 🏷️ Supported Entity Types

| Label | Description |
|---|---|
| `PERSON` | People, including fictional |
| `ORG` | Companies, agencies, institutions |
| `GPE` | Countries, cities, states |
| `LOC` | Non-GPE locations, mountain ranges, water bodies |
| `DATE` | Absolute or relative dates or periods |
| `TIME` | Times smaller than a day |
| `MONEY` | Monetary values, including unit |
| `PERCENT` | Percentage |
| `PRODUCT` | Objects, vehicles, foods, etc. |
| `EVENT` | Named hurricanes, battles, wars, sports events |
| `WORK_OF_ART` | Titles of books, songs, etc. |
| `LAW` | Named documents made into laws |
| `LANGUAGE` | Any named language |
| `NORP` | Nationalities or religious/political groups |
| `FAC` | Buildings, airports, highways, bridges |
| `QUANTITY` | Measurements, weights or amounts |
| `ORDINAL` | First, second, etc. |
| `CARDINAL` | Numerals that do not fall under another type |

---

## 📁 Project Structure

```
Named-Entity-Recognition/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository and set `app.py` as the entry point
5. Click **Deploy** — done! 🚀

> **Note:** Streamlit Cloud will automatically install packages from `requirements.txt`. The spaCy model `en_core_web_sm` is listed there and will be downloaded automatically.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 👤 Author

**Tejas Divekar**  
GitHub: [@TejasD-13](https://github.com/TejasD-13)

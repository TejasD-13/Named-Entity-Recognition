import streamlit as st
import spacy
from spacy import displacy
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NER Explorer",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS  (bright, clean, human-feel theme) ─────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Base background ── */
    .stApp {
        background-color: #f5f7fa;
        min-height: 100vh;
    }

    /* ── Top hero banner ── */
    .hero {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 2.2rem 2rem 1.8rem;
        text-align: center;
        margin-bottom: 1.8rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a202c;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .hero h1 span {
        color: #3b82f6;
    }
    .hero p {
        color: #64748b;
        font-size: 1rem;
        margin: 0;
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #3b82f6;
        margin-bottom: 0.5rem;
    }

    /* ── Cards ── */
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.6rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }

    /* ── Stat badges ── */
    .stat-row {
        display: flex;
        gap: 0.9rem;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    .stat-badge {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 0.7rem 1.3rem;
        text-align: center;
        min-width: 100px;
    }
    .stat-badge .num {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0369a1;
        line-height: 1;
    }
    .stat-badge .label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 3px;
    }

    /* ── NER highlighted output ── */
    .ner-output {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        line-height: 2.4;
        font-size: 1rem;
        color: #1e293b;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border-radius: 8px;
        color: #64748b;
        border: 1px solid #e2e8f0;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: #eff6ff !important;
        color: #2563eb !important;
        border-color: #bfdbfe !important;
        font-weight: 600 !important;
    }

    /* ── Analyse button ── */
    .stButton > button {
        background: #2563eb;
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.8rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: background 0.18s, box-shadow 0.18s, transform 0.18s;
        width: 100%;
    }
    .stButton > button:hover {
        background: #1d4ed8;
        box-shadow: 0 6px 18px rgba(37,99,235,0.25);
        transform: translateY(-1px);
    }

    /* ── File uploader ── */
    .stFileUploader > div {
        background: #ffffff;
        border: 2px dashed #93c5fd;
        border-radius: 12px;
        color: #64748b;
    }

    /* ── Text area ── */
    .stTextArea textarea {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        color: #1e293b !important;
        font-size: 0.95rem !important;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.04) !important;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
    }

    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 1.5rem 0;
    }

    /* ── Dataframe ── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }

    /* ── Success / warning boxes ── */
    .stSuccess {
        background: #f0fdf4;
        border-color: #86efac;
        color: #166534;
        border-radius: 10px;
    }
    .stWarning {
        background: #fffbeb;
        border-color: #fcd34d;
        color: #92400e;
        border-radius: 10px;
    }
    .stInfo {
        background: #eff6ff;
        border-color: #93c5fd;
        color: #1e40af;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load spaCy model ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# ── Entity label → description ────────────────────────────────────────────────
LABEL_DESCRIPTIONS = {
    "PERSON":     "People, including fictional",
    "ORG":        "Companies, agencies, institutions",
    "GPE":        "Countries, cities, states",
    "LOC":        "Non-GPE locations, mountain ranges, water bodies",
    "DATE":       "Absolute or relative dates or periods",
    "TIME":       "Times smaller than a day",
    "MONEY":      "Monetary values, including unit",
    "PERCENT":    "Percentage",
    "PRODUCT":    "Objects, vehicles, foods, etc.",
    "EVENT":      "Named hurricanes, battles, wars, sports events",
    "WORK_OF_ART":"Titles of books, songs, etc.",
    "LAW":        "Named documents made into laws",
    "LANGUAGE":   "Any named language",
    "NORP":       "Nationalities or religious/political groups",
    "FAC":        "Buildings, airports, highways, bridges",
    "QUANTITY":   "Measurements, weights or amounts",
    "ORDINAL":    "First, second, etc.",
    "CARDINAL":   "Numerals that do not fall under another type",
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔍 NER <span>Explorer</span></h1>
    <p>Named Entity Recognition powered by spaCy &nbsp;·&nbsp; Identify people, places, organizations &amp; more</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col_input, col_results = st.columns([1, 1.4], gap="large")

with col_input:
    st.markdown('<div class="section-label">📥 Your Input</div>', unsafe_allow_html=True)
    tab_file, tab_text = st.tabs(["📄 Upload File", "✍️ Paste Text"])

    text_input = ""

    with tab_file:
        uploaded = st.file_uploader(
            "Drop a .txt file here",
            type=["txt"],
            label_visibility="collapsed",
        )
        if uploaded:
            text_input = uploaded.read().decode("utf-8", errors="ignore")
            st.success(f"✅ **{uploaded.name}** loaded — {len(text_input):,} characters")

    with tab_text:
        sample = (
            "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California in 1976. "
            "The company is headquartered in the United States and is valued at over $3 trillion."
        )
        text_input = st.text_area(
            "Enter text",
            value=text_input or sample,
            height=220,
            label_visibility="collapsed",
            placeholder="Paste your text here…",
        )

    analyse = st.button("🔍 Analyse Entities", use_container_width=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyse and text_input.strip():
    doc = nlp(text_input)
    entities = [
        (ent.text, ent.label_, LABEL_DESCRIPTIONS.get(ent.label_, ""), ent.start_char, ent.end_char)
        for ent in doc.ents
    ]

    with col_results:
        # Stat badges
        unique_labels = set(e[1] for e in entities)
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-badge"><div class="num">{len(entities)}</div><div class="label">Entities</div></div>
            <div class="stat-badge"><div class="num">{len(unique_labels)}</div><div class="label">Types</div></div>
            <div class="stat-badge"><div class="num">{len(text_input.split())}</div><div class="label">Words</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Highlighted NER output
        st.markdown('<div class="section-label">🏷️ Highlighted Text</div>', unsafe_allow_html=True)
        ner_html = displacy.render(doc, style="ent", page=False)
        ner_html = ner_html.replace(
            'style="line-height: 2.5; direction: ltr"',
            'style="line-height: 2.5; direction: ltr; background: transparent;"'
        )
        st.markdown(f'<div class="ner-output">{ner_html}</div>', unsafe_allow_html=True)

        # Entity table
        if entities:
            st.markdown('<div class="section-label" style="margin-top:1.5rem">📋 Entity Table</div>', unsafe_allow_html=True)
            df = pd.DataFrame(entities, columns=["Entity", "Label", "Description", "Start", "End"])
            df = df.drop(columns=["Start", "End"])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No named entities were detected in this text.")

elif analyse and not text_input.strip():
    with col_results:
        st.warning("⚠️ Please provide some text or upload a file first.")

else:
    with col_results:
        st.markdown("""
        <div class="card" style="margin-top:3rem; text-align:center; padding: 3rem;">
            <div style="font-size:3rem">🧠</div>
            <div style="color:#64748b; margin-top:1rem; font-size:1rem; line-height:1.7;">
                Upload a file or paste text on the left,<br>
                then click <strong style="color:#2563eb;">Analyse Entities</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer legend ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">🗂️ Entity Type Reference</div>', unsafe_allow_html=True)
legend_df = pd.DataFrame([{"Label": k, "Description": v} for k, v in LABEL_DESCRIPTIONS.items()])
st.dataframe(legend_df, use_container_width=True, hide_index=True, height=250)

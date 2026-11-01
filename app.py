"""
app.py – NER Explorer: main Streamlit application.

Tabs
----
1. 🏠 Input              – text / file input, preprocessing display
2. 🏷️ Entity Recognition – highlighted text, entity table, filtering
3. 📊 Statistics          – summary cards, entity distribution chart
4. 📐 Evaluation          – upload annotated CSV, compute P / R / F1
5. 🔬 Error Analysis      – per-sentence error breakdown
6. ℹ️ Model Info          – loaded model metadata and label reference
"""
from __future__ import annotations

import os
import sys
import spacy as _spacy


def spacy_explain_safe(label: str) -> str:
    """Return a human-readable explanation for a spaCy label, or the label itself."""
    return _spacy.explain(label) or label

import pandas as pd
import streamlit as st
from spacy import displacy

# ── Make the src/ package importable when running from the project root ──────
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import preprocess_text, PreprocessingError, PREPROCESSING_STEPS
from src.ner_engine import (
    load_model, extract_entities, get_model_info,
    ModelNotFoundError, LABEL_DESCRIPTIONS, DEFAULT_MODEL,
)
from src.evaluation import (
    load_annotated_data, validate_annotations, evaluate_dataset,
    REQUIRED_COLUMNS,
)
from src.error_analysis import analyse_errors, summarise_errors, ERROR_EXPLANATIONS
from src.visualization import (
    entity_distribution_chart, per_label_metrics_chart, error_category_chart,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NER Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MAX_INPUT_CHARS = 50_000
MODEL_NAME = DEFAULT_MODEL   # change to 'en_core_web_lg' for higher accuracy

# ─────────────────────────────────────────────────────────────────────────────
# CSS – clean bright white theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #f5f7fa; }

/* ── Hero ── */
.hero {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 2rem 2rem 1.6rem;
    text-align: center;
    margin-bottom: 1.8rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.hero h1 { font-size:2.2rem; font-weight:700; color:#1a202c; margin:0 0 .3rem 0; letter-spacing:-.5px; }
.hero h1 span { color:#2563eb; }
.hero p  { color:#64748b; font-size:.95rem; margin:0; }

/* ── Section labels ── */
.slabel {
    font-size:.7rem; font-weight:600; text-transform:uppercase;
    letter-spacing:.1em; color:#2563eb; margin-bottom:.4rem;
}

/* ── Stat cards ── */
.stat-row { display:flex; gap:.8rem; flex-wrap:wrap; margin-bottom:1.2rem; }
.stat-card {
    background:#eff6ff; border:1px solid #bfdbfe;
    border-radius:12px; padding:.65rem 1.1rem;
    text-align:center; min-width:90px; flex:1;
}
.stat-card .num  { font-size:1.7rem; font-weight:700; color:#1d4ed8; line-height:1; }
.stat-card .lbl  { font-size:.7rem; color:#64748b; text-transform:uppercase;
                    letter-spacing:.06em; margin-top:3px; }

/* ── NER output box ── */
.ner-box {
    background:#fff; border:1px solid #e2e8f0; border-radius:14px;
    padding:1.2rem 1.5rem; line-height:2.4; font-size:.97rem; color:#1e293b;
    box-shadow:0 1px 4px rgba(0,0,0,0.04);
}

/* ── Metric cards ── */
.metric-grid { display:flex; gap:.8rem; margin-bottom:1.2rem; flex-wrap:wrap; }
.metric-card {
    flex:1; min-width:110px; background:#fff;
    border:1px solid #e2e8f0; border-radius:12px;
    padding:.8rem 1rem; text-align:center;
    box-shadow:0 1px 4px rgba(0,0,0,0.04);
}
.metric-card .mval { font-size:1.8rem; font-weight:700; }
.metric-card .mlbl { font-size:.7rem; color:#64748b; text-transform:uppercase; letter-spacing:.06em; margin-top:2px; }
.blue  { color:#2563eb; } .green { color:#059669; } .amber { color:#d97706; }

/* ── General card ── */
.card {
    background:#fff; border:1px solid #e2e8f0; border-radius:14px;
    padding:1.4rem; box-shadow:0 1px 5px rgba(0,0,0,0.04); margin-bottom:.8rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap:6px; background:transparent; }
.stTabs [data-baseweb="tab"] {
    background:#fff; border-radius:8px; color:#64748b;
    border:1px solid #e2e8f0; font-size:.88rem;
}
.stTabs [aria-selected="true"] {
    background:#eff6ff !important; color:#2563eb !important;
    border-color:#bfdbfe !important; font-weight:600 !important;
}

/* ── Button ── */
.stButton > button {
    background:#2563eb; color:#fff; border:none;
    border-radius:10px; font-weight:600; font-size:.95rem;
    transition:background .18s, box-shadow .18s, transform .15s;
    width:100%;
}
.stButton > button:hover {
    background:#1d4ed8;
    box-shadow:0 6px 18px rgba(37,99,235,.25);
    transform:translateY(-1px);
}

/* ── Inputs ── */
.stTextArea textarea {
    background:#fff !important; border:1px solid #cbd5e1 !important;
    border-radius:10px !important; color:#1e293b !important;
    font-size:.93rem !important;
}
.stFileUploader > div {
    background:#fff; border:2px dashed #93c5fd;
    border-radius:12px; color:#64748b;
}

/* ── Error category colours ── */
.err-CORRECT { color:#059669; font-weight:600; }
.err-MISSED  { color:#dc2626; font-weight:600; }
.err-EXTRA   { color:#d97706; font-weight:600; }
.err-WRONG_LABEL { color:#7c3aed; font-weight:600; }
.err-BOUNDARY_MISMATCH { color:#2563eb; font-weight:600; }

/* ── Misc ── */
#MainMenu, footer, header { visibility:hidden; }
hr { border:none; border-top:1px solid #e2e8f0; margin:1.2rem 0; }
.stDataFrame { border-radius:12px; overflow:hidden; border:1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NLP model…")
def get_nlp(model_name: str):
    return load_model(model_name)


try:
    nlp = get_nlp(MODEL_NAME)
    model_ok = True
except ModelNotFoundError as e:
    model_ok = False
    model_error = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔍 NER <span>Explorer</span></h1>
  <p>Identify, analyze, and evaluate named entities in text using Natural Language Processing.</p>
</div>
""", unsafe_allow_html=True)

if not model_ok:
    st.error(
        f"⚠️ spaCy model not found.\n\n"
        f"Run this command in your terminal, then restart the app:\n\n"
        f"```\npython -m spacy download {MODEL_NAME}\n```\n\n"
        f"Details: {model_error}"
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────────────────────────────────────
def _clear_analysis() -> None:
    for key in ("ner_result", "clean_text", "raw_text"):
        st.session_state.pop(key, None)


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Input",
    "🏷️ Entities",
    "📊 Statistics",
    "📐 Evaluation",
    "🔬 Error Analysis",
    "ℹ️ Model Info",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – INPUT
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    col_in, col_prep = st.columns([1.1, 1], gap="large")

    with col_in:
        st.markdown('<div class="slabel">📥 Text Input</div>', unsafe_allow_html=True)
        input_tab_file, input_tab_text = st.tabs(["📄 Upload File", "✍️ Paste Text"])

        raw_text = ""
        with input_tab_file:
            uploaded = st.file_uploader(
                "Upload a UTF-8 .txt file",
                type=["txt"],
                label_visibility="collapsed",
                key="uploader",
            )
            if uploaded:
                if uploaded.size == 0:
                    st.warning("The uploaded file is empty.")
                else:
                    raw_text = uploaded.read().decode("utf-8", errors="ignore")
                    st.success(
                        f"✅ **{uploaded.name}** — {len(raw_text):,} characters"
                    )

        with input_tab_text:
            sample_text = (
                "Apple Inc. was founded by Steve Jobs and Steve Wozniak "
                "in Cupertino, California in 1976. The company is "
                "headquartered in the United States and is valued at "
                "over $3 trillion."
            )
            raw_text = st.text_area(
                "Enter text",
                value=raw_text or sample_text,
                height=200,
                label_visibility="collapsed",
                placeholder="Paste your text here…",
                max_chars=MAX_INPUT_CHARS,
                key="text_area",
            )

        analyse_clicked = st.button("🔍 Analyse Entities", use_container_width=True)

        if analyse_clicked:
            _clear_analysis()
            if not raw_text or not raw_text.strip():
                st.warning("⚠️ Please enter some text or upload a file first.")
            else:
                try:
                    clean = preprocess_text(raw_text, max_chars=MAX_INPUT_CHARS)
                    result = extract_entities(clean, nlp)
                    st.session_state["raw_text"]   = raw_text
                    st.session_state["clean_text"] = clean
                    st.session_state["ner_result"] = result
                    st.success(
                        f"✅ Detected **{len(result['entities'])}** entities. "
                        "Switch to the **🏷️ Entities** tab to see results."
                    )
                except PreprocessingError as e:
                    st.error(f"Input error: {e}")
                except Exception as e:
                    st.error(f"Processing error: {e}")

    with col_prep:
        st.markdown('<div class="slabel">⚙️ Preprocessing Pipeline</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for step in PREPROCESSING_STEPS:
            st.markdown(f"- {step}")
        st.markdown('</div>', unsafe_allow_html=True)

        if "clean_text" in st.session_state:
            st.markdown('<div class="slabel" style="margin-top:.8rem">📄 Preprocessed Text Preview</div>', unsafe_allow_html=True)
            preview = st.session_state["clean_text"][:500]
            if len(st.session_state["clean_text"]) > 500:
                preview += "…"
            st.code(preview, language=None)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – ENTITIES
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    if "ner_result" not in st.session_state:
        st.info("👈 Enter text in the **🏠 Input** tab and click **Analyse Entities**.")
    else:
        result   = st.session_state["ner_result"]
        entities = result["entities"]
        doc      = result["doc"]

        # Highlighted output
        st.markdown('<div class="slabel">🏷️ Entity-Highlighted Text</div>', unsafe_allow_html=True)
        if entities:
            ner_html = displacy.render(doc, style="ent", page=False)
            ner_html = ner_html.replace(
                'style="line-height: 2.5; direction: ltr"',
                'style="line-height: 2.5; direction: ltr; background: transparent;"',
            )
            st.markdown(f'<div class="ner-box">{ner_html}</div>', unsafe_allow_html=True)
        else:
            st.info("ℹ️ No named entities detected in this text.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Entity table with filtering
        if entities:
            st.markdown('<div class="slabel">📋 Entity Table</div>', unsafe_allow_html=True)
            df_ent = pd.DataFrame(entities).drop(columns=["start_char", "end_char"], errors="ignore")
            df_ent.columns = [c.capitalize() for c in df_ent.columns]

            labels_available = sorted(df_ent["Label"].unique())
            selected_labels  = st.multiselect(
                "Filter by entity type",
                options=labels_available,
                default=labels_available,
                key="ent_filter",
            )
            df_filtered = df_ent[df_ent["Label"].isin(selected_labels)]
            st.dataframe(df_filtered, use_container_width=True, hide_index=True)
            st.caption(f"Showing {len(df_filtered)} of {len(df_ent)} entities.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    if "ner_result" not in st.session_state:
        st.info("👈 Analyse some text first.")
    else:
        result   = st.session_state["ner_result"]
        entities = result["entities"]
        text     = result["text"]

        unique_labels = set(e["label"] for e in entities)

        # Stat cards
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="num">{len(entities)}</div><div class="lbl">Entities</div></div>
          <div class="stat-card"><div class="num">{len(unique_labels)}</div><div class="lbl">Types</div></div>
          <div class="stat-card"><div class="num">{len(text.split())}</div><div class="lbl">Words</div></div>
          <div class="stat-card"><div class="num">{len(text):,}</div><div class="lbl">Characters</div></div>
        </div>
        """, unsafe_allow_html=True)

        col_chart, col_table = st.columns([1.4, 1], gap="large")

        with col_chart:
            st.markdown('<div class="slabel">📊 Entity Distribution</div>', unsafe_allow_html=True)
            fig = entity_distribution_chart(entities)
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.markdown('<div class="slabel">📋 Count by Label</div>', unsafe_allow_html=True)
            if entities:
                df_cnt = (
                    pd.DataFrame(entities)["label"]
                    .value_counts()
                    .reset_index()
                )
                # handle both old and new pandas column naming
                df_cnt.columns = ["Label", "Count"] if len(df_cnt.columns) == 2 else df_cnt.columns
                if "label" in df_cnt.columns:
                    df_cnt = df_cnt.rename(columns={"label": "Label", "count": "Count"})
                st.dataframe(df_cnt, use_container_width=True, hide_index=True)
            else:
                st.info("No entities to summarise.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="slabel">📐 Evaluation Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <strong>Evaluation methodology</strong><br>
    A prediction is counted as correct only when the entity span
    (<code>start_char</code>, <code>end_char</code>) <em>and</em> the entity label
    match the annotated reference exactly (exact-span matching).<br><br>
    <strong>Formulas:</strong><br>
    Precision&nbsp;=&nbsp;TP / (TP + FP)&nbsp;&nbsp;|&nbsp;&nbsp;
    Recall&nbsp;=&nbsp;TP / (TP + FN)&nbsp;&nbsp;|&nbsp;&nbsp;
    F1&nbsp;=&nbsp;2 × P × R / (P + R)
    </div>
    """, unsafe_allow_html=True)

    col_up, col_res = st.columns([1, 1.5], gap="large")

    with col_up:
        st.markdown('<div class="slabel">Upload Annotated CSV</div>', unsafe_allow_html=True)
        eval_file = st.file_uploader(
            "Upload annotated CSV",
            type=["csv"],
            label_visibility="collapsed",
            key="eval_uploader",
        )
        use_demo = st.checkbox(
            "Use included demonstration dataset (`data/annotated_test_data.csv`)",
            value=True,
        )

        run_eval = st.button("▶ Run Evaluation", use_container_width=True)

    with col_res:
        if run_eval:
            try:
                if eval_file is not None:
                    df_eval = load_annotated_data(eval_file)
                elif use_demo:
                    demo_path = os.path.join(os.path.dirname(__file__), "data", "annotated_test_data.csv")
                    if not os.path.isfile(demo_path):
                        st.error("Demo dataset not found. Upload a CSV file instead.")
                        st.stop()
                    df_eval = load_annotated_data(demo_path)
                else:
                    st.warning("Please upload a CSV file or enable the demo dataset.")
                    st.stop()

                # Validate annotations
                warnings = validate_annotations(df_eval)
                if warnings:
                    with st.expander(f"⚠️ {len(warnings)} annotation warning(s)"):
                        for w in warnings:
                            st.text(w)

                with st.spinner("Running model on evaluation data…"):
                    eval_result = evaluate_dataset(df_eval, nlp)
                st.session_state["eval_result"] = eval_result
                st.session_state["eval_df"]     = df_eval

            except Exception as e:
                st.error(f"Evaluation error: {e}")

        if "eval_result" in st.session_state:
            ev = st.session_state["eval_result"]
            ov = ev["overall"]

            # Overall metric cards
            p_pct  = f"{ov['precision']:.1%}"
            r_pct  = f"{ov['recall']:.1%}"
            f1_pct = f"{ov['f1']:.1%}"
            st.markdown(f"""
            <div class="metric-grid">
              <div class="metric-card"><div class="mval blue">{p_pct}</div><div class="mlbl">Precision</div></div>
              <div class="metric-card"><div class="mval green">{r_pct}</div><div class="mlbl">Recall</div></div>
              <div class="metric-card"><div class="mval amber">{f1_pct}</div><div class="mlbl">F1-Score</div></div>
              <div class="metric-card"><div class="mval">{ov['tp']}</div><div class="mlbl">TP</div></div>
              <div class="metric-card"><div class="mval">{ov['fp']}</div><div class="mlbl">FP</div></div>
              <div class="metric-card"><div class="mval">{ov['fn']}</div><div class="mlbl">FN</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.caption(
                "⚠️ Results above are on the **demonstration dataset** (10 sentences). "
                "They do not represent general-purpose NER accuracy."
            )

            # Per-label metrics chart
            if ev["per_label"]:
                st.markdown('<div class="slabel" style="margin-top:.6rem">Per-Label Metrics</div>', unsafe_allow_html=True)
                fig_pl = per_label_metrics_chart(ev["per_label"])
                st.plotly_chart(fig_pl, use_container_width=True)

                # Per-label table
                rows = []
                for lbl, m in ev["per_label"].items():
                    rows.append({
                        "Label": lbl,
                        "Precision": f"{m['precision']:.3f}",
                        "Recall":    f"{m['recall']:.3f}",
                        "F1":        f"{m['f1']:.3f}",
                        "TP": m["tp"], "FP": m["fp"], "FN": m["fn"],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="slabel">🔬 Error Analysis</div>', unsafe_allow_html=True)

    if "eval_df" not in st.session_state:
        st.info("Run an **Evaluation** (📐 tab) first to enable error analysis.")
    else:
        df_ea = st.session_state["eval_df"]
        all_errors: list[dict] = []

        for text_id in df_ea["text_id"].astype(str).unique():
            text_row   = df_ea[df_ea["text_id"].astype(str) == str(text_id)]
            text_str   = text_row["text"].iloc[0]
            ref_ents   = []
            for _, row in text_row.iterrows():
                et = str(row.get("entity_text", "") or "")
                if et and et.strip().upper() != "NO_ENTITY":
                    ref_ents.append({
                        "text":       et,
                        "label":      str(row["entity_label"]),
                        "start_char": int(row["start_char"]),
                        "end_char":   int(row["end_char"]),
                    })
            errs = analyse_errors(text_str, ref_ents, nlp)
            for e in errs:
                e["text_id"] = text_id
            all_errors.extend(errs)

        st.session_state["all_errors"] = all_errors

        # Summary chart
        col_pie, col_leg = st.columns([1, 1.3], gap="large")
        with col_pie:
            fig_err = error_category_chart(all_errors)
            st.plotly_chart(fig_err, use_container_width=True)

        with col_leg:
            st.markdown('<div class="slabel">Category Definitions</div>', unsafe_allow_html=True)
            for cat, desc in ERROR_EXPLANATIONS.items():
                st.markdown(f"**{cat}** – {desc}")

        # Filter & table
        st.markdown('<div class="slabel" style="margin-top:.6rem">Error Details</div>', unsafe_allow_html=True)
        cats = sorted(set(e["category"] for e in all_errors))
        selected_cats = st.multiselect(
            "Filter by category", cats, default=cats, key="err_filter"
        )
        filtered_errs = [e for e in all_errors if e["category"] in selected_cats]
        df_err = summarise_errors(filtered_errs)
        if not df_err.empty:
            st.dataframe(df_err, use_container_width=True, hide_index=True)
            st.caption(f"Showing {len(df_err)} error records.")
        else:
            st.success("No errors in the selected categories.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 – MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    info = get_model_info(nlp)
    col_meta, col_labels = st.columns([1, 1.4], gap="large")

    with col_meta:
        st.markdown('<div class="slabel">🤖 Model Metadata</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
        <b>Name:</b> {info['model_name']}<br>
        <b>Version:</b> {info['version']}<br>
        <b>Language:</b> {info['lang']}<br>
        <b>Pipeline:</b> {', '.join(info['pipeline'])}<br>
        <b>Type:</b> {info['model_type']}<br>
        <b>NER labels:</b> {len(info['ner_labels'])}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="slabel" style="margin-top:.8rem">⚠️ Known Limitations</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <ul style="margin:0; padding-left:1.2rem; color:#374151;">
        <li>en_core_web_sm is a small CNN-based model — not transformer-based.</li>
        <li>Lower accuracy on rare names, domain-specific text, or non-standard spelling.</li>
        <li>Entity support varies by label type; rarer labels (FAC, WORK_OF_ART) have fewer training examples.</li>
        <li>Does not assign confidence scores per prediction.</li>
        <li>May miss entities that were uncommon in training data (e.g. Indian names).</li>
        <li>Boundary detection may be imperfect for compound multi-word entities.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_labels:
        st.markdown('<div class="slabel">🏷️ Supported Entity Labels (model)</div>', unsafe_allow_html=True)
        label_rows = [
            {
                "Label": lbl,
                "Description": LABEL_DESCRIPTIONS.get(lbl, spacy_explain_safe(lbl)),
            }
            for lbl in info["ner_labels"]
        ]
        st.dataframe(pd.DataFrame(label_rows), use_container_width=True, hide_index=True, height=460)

        st.markdown('<div class="slabel" style="margin-top:.8rem">📖 About Pretrained NER</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card" style="font-size:.9rem; color:#374151; line-height:1.7;">
        <b>en_core_web_sm</b> is a small English NLP pipeline trained on the
        <b>OntoNotes 5</b> corpus (annotated newswire, broadcast news, web text).<br><br>
        It uses a CNN architecture for token classification and applies the
        <b>B-I-O tagging scheme</b>: each token is tagged as
        <code>B-LABEL</code> (beginning), <code>I-LABEL</code> (inside), or
        <code>O</code> (outside). Adjacent B/I tokens with the same label are
        merged into one entity span.<br><br>
        The model predicts from contextual signals — the same word can be labeled
        differently depending on surrounding tokens.
        </div>
        """, unsafe_allow_html=True)




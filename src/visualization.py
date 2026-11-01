"""
visualization.py – Chart helpers for NER Explorer.

Uses Plotly for interactive charts rendered inside Streamlit.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_PALETTE = px.colors.qualitative.Safe


# ── Entity distribution ───────────────────────────────────────────────────────

def entity_distribution_chart(entities: list[dict]) -> go.Figure:
    """Horizontal bar chart of detected entity counts by label."""
    if not entities:
        fig = go.Figure()
        fig.add_annotation(
            text="No entities detected",
            showarrow=False,
            font=dict(size=15, color="#64748b"),
        )
        fig.update_layout(
            height=200, paper_bgcolor="white", plot_bgcolor="white",
        )
        return fig

    df = (
        pd.DataFrame(entities)["label"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Label", "label": "Count"})
        .sort_values("Count", ascending=True)
    )
    # Newer pandas may have different column naming after value_counts
    if "label" in df.columns and "count" in df.columns:
        df = df.rename(columns={"label": "Label", "count": "Count"})

    fig = px.bar(
        df,
        x="Count",
        y="Label",
        orientation="h",
        color="Label",
        color_discrete_sequence=_PALETTE,
        title="Entity Type Distribution",
        labels={"Count": "Number of Entities", "Label": "Entity Type"},
    )
    fig.update_layout(
        showlegend=False,
        height=max(280, 38 * len(df) + 80),
        margin=dict(l=10, r=20, t=45, b=20),
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        font=dict(family="Inter, sans-serif", size=13),
        title_font_size=14,
    )
    fig.update_traces(marker_line_width=0)
    return fig


# ── Per-label evaluation metrics ─────────────────────────────────────────────

def per_label_metrics_chart(per_label: dict) -> go.Figure:
    """Grouped bar chart of per-label precision, recall, and F1."""
    if not per_label:
        fig = go.Figure()
        fig.add_annotation(
            text="No per-label evaluation data",
            showarrow=False,
            font=dict(size=15, color="#64748b"),
        )
        fig.update_layout(height=250, paper_bgcolor="white", plot_bgcolor="white")
        return fig

    labels  = list(per_label.keys())
    colors  = ["#3b82f6", "#10b981", "#f59e0b"]
    metrics = ["precision", "recall", "f1"]

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=labels,
            y=[per_label[lbl][metric] for lbl in labels],
            marker_color=color,
        ))

    fig.update_layout(
        barmode="group",
        title="Per-Label Evaluation Metrics",
        yaxis=dict(range=[0, 1.05], tickformat=".0%", title="Score"),
        xaxis_title="Entity Label",
        height=400,
        legend_title="Metric",
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        font=dict(family="Inter, sans-serif", size=13),
        title_font_size=14,
        margin=dict(l=10, r=20, t=50, b=20),
    )
    return fig


# ── Error category pie chart ──────────────────────────────────────────────────

def error_category_chart(errors: list[dict]) -> go.Figure:
    """Pie chart of error categories from error analysis."""
    if not errors:
        fig = go.Figure()
        fig.add_annotation(
            text="No error data",
            showarrow=False,
            font=dict(size=15, color="#64748b"),
        )
        fig.update_layout(height=250, paper_bgcolor="white", plot_bgcolor="white")
        return fig

    from collections import Counter
    counts = Counter(e["category"] for e in errors)

    color_map = {
        "CORRECT":           "#10b981",
        "MISSED":            "#ef4444",
        "EXTRA":             "#f59e0b",
        "WRONG_LABEL":       "#8b5cf6",
        "BOUNDARY_MISMATCH": "#3b82f6",
    }

    labels = list(counts.keys())
    values = list(counts.values())
    colors = [color_map.get(l, "#94a3b8") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.4,
    ))
    fig.update_layout(
        title="Error Category Breakdown",
        height=350,
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=13),
        title_font_size=14,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig

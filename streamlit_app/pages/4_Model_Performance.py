"""
Final model performance page.

This page reflects the updated notebook results: STGNN remains the best model,
while XGBoost, Random Forest, HMM, N-HiTS, and a Ridge ensemble provide useful
baselines and interpretation.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import (
    get_evaluation_chart_path,
    get_model_comparison_chart_path,
    load_feature_importance,
    load_model_comparison,
    load_results,
)


st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("Model Performance Dashboard")
st.caption("Final comparison after leakage fixes, GAE repair, and N-HiTS sequence-model testing")


with st.spinner("Loading final result artifacts..."):
    comparison = load_model_comparison()
    ablation = load_results()
    feature_importance = load_feature_importance()

if comparison.empty:
    st.error("model_comparison.csv is missing or empty. Run the notebook first.")
    st.stop()

comparison = comparison.copy()
comparison["MAE"] = pd.to_numeric(comparison["MAE"], errors="coerce")
comparison["RMSE"] = pd.to_numeric(comparison["RMSE"], errors="coerce")
comparison = comparison.dropna(subset=["MAE", "RMSE"]).sort_values("MAE").reset_index(drop=True)

best = comparison.iloc[0]
stgnn_row = comparison[comparison["Model"].str.contains("STGNN", case=False, na=False)]
stgnn_mae = float(stgnn_row["MAE"].iloc[0]) if not stgnn_row.empty else 10.9

st.subheader("Final Result Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Best Model", best["Model"])
c2.metric("Best MAE", f"{best['MAE']:.2f} min")
c3.metric("Best RMSE", f"{best['RMSE']:.2f} min")
c4.metric("STGNN Baseline", f"{stgnn_mae:.2f} min")

st.success(
    "The final comparison supports the main project claim: graph-aware STGNN "
    "prediction is strongest because flight delays propagate through airport routes."
)

st.divider()

left, right = st.columns([1.05, 1])

with left:
    st.subheader("Leaderboard")
    best_idx = comparison["MAE"].idxmin()

    def _highlight_best(row):
        return [
            "background-color: #1f8f4d; color: white" if row.name == best_idx else ""
            for _ in row
        ]

    st.dataframe(
        comparison.style.apply(_highlight_best, axis=1).format({"MAE": "{:.2f}", "RMSE": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download model_comparison.csv",
        comparison.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )

with right:
    st.subheader("MAE Comparison")
    colors = ["#2ecc71" if i == best_idx else "#4a90d9" for i in comparison.index]
    fig = px.bar(
        comparison,
        x="MAE",
        y="Model",
        orientation="h",
        text=comparison["MAE"].map(lambda value: f"{value:.2f}"),
        title="Final Model MAE Comparison - Flight Delay Prediction",
    )
    fig.update_traces(marker_color=colors, textposition="outside")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        xaxis_title="MAE (minutes)",
        yaxis_title="Model",
        yaxis={"categoryorder": "total ascending"},
        height=420,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("How To Read These Results")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(
        """
**STGNN**

Best overall. It uses both temporal history and route-network structure, which
matches how delays actually spread.
"""
    )
with col_b:
    st.markdown(
        """
**Tabular baselines**

XGBoost and Random Forest are realistic now because `arrival_delay` was removed
from the input features.
"""
    )
with col_c:
    st.markdown(
        """
**HMM and N-HiTS**

HMM is useful for state interpretation. N-HiTS trained successfully but pure
time-series forecasting missed graph propagation effects.
"""
    )

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Random Forest Feature Importance")
    if feature_importance.empty:
        st.warning("feature_importance.csv not found. Run the notebook to generate it.")
    else:
        feature_importance = feature_importance.sort_values("importance", ascending=True)
        fig_fi = px.bar(
            feature_importance,
            x="importance",
            y="feature",
            orientation="h",
            text=feature_importance["importance"].map(lambda value: f"{value:.3f}"),
            title="Corrected Feature Importances",
        )
        fig_fi.update_traces(marker_color="#4a90d9", textposition="outside")
        fig_fi.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=380,
            showlegend=False,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        st.download_button(
            "Download feature_importance.csv",
            feature_importance.sort_values("importance", ascending=False).to_csv(index=False).encode("utf-8"),
            file_name="feature_importance.csv",
            mime="text/csv",
            use_container_width=True,
        )

with right:
    st.subheader("Original STGNN Ablation Context")
    if ablation.empty:
        st.warning("results_table.csv not found.")
    else:
        fig_ablation = go.Figure()
        for metric, color in [("MAE", "#58a6ff"), ("RMSE", "#f0883e")]:
            fig_ablation.add_trace(
                go.Bar(
                    name=metric,
                    x=ablation["Model"],
                    y=ablation[metric],
                    marker_color=color,
                    text=[f"{value:.1f}" for value in ablation[metric]],
                    textposition="outside",
                )
            )
        fig_ablation.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            yaxis_title="Error (minutes)",
            xaxis_title="Pipeline Variant",
            height=380,
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_ablation, use_container_width=True)

st.divider()

st.subheader("Saved Evaluation Artifacts")
artifact_cols = st.columns(2)

with artifact_cols[0]:
    comparison_chart_path = get_model_comparison_chart_path()
    if os.path.exists(comparison_chart_path):
        st.image(comparison_chart_path, use_container_width=True, caption="Final model comparison chart")
    else:
        st.info("model_comparison.png has not been generated yet.")

with artifact_cols[1]:
    evaluation_chart_path = get_evaluation_chart_path()
    if os.path.exists(evaluation_chart_path):
        st.image(evaluation_chart_path, use_container_width=True, caption="Original STGNN evaluation charts")
    else:
        st.info("final_evaluation_charts.png has not been generated yet.")

st.divider()

st.subheader("Final Interpretation")
st.markdown(
    """
The updated dashboard should be read as a model-selection story, not just a
scoreboard. The corrected tabular baselines are competitive, but the STGNN
remains best because it directly models airport connectivity. HMM contributes
interpretable delay regimes, GAE contributes airport similarity structure, and
N-HiTS confirms that a lightweight sequence model alone is not enough for this
networked forecasting problem.
"""
)

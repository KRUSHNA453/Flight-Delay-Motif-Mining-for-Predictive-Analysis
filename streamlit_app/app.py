"""
Main entry point for the Flight Delay Motif Mining dashboard.

Run from the project root with:
    streamlit run streamlit_app/app.py
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.data_loader import load_model_comparison, load_motifs


st.set_page_config(
    page_title="Flight Delay Motif Mining",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    a { color: #58a6ff !important; }
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px;
    }
    [data-testid="stMetricLabel"] { color: #8b949e; }
    [data-testid="stMetricValue"] { color: #58a6ff; }
    h1, h2, h3 { color: #e6edf3 !important; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load_home_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_model_comparison(), load_motifs()


comparison, motifs = _load_home_data()
comparison = comparison.sort_values("MAE", ascending=True).reset_index(drop=True)
best = comparison.iloc[0] if not comparison.empty else None

with st.sidebar:
    st.markdown("## Flight Delay Motif Mining")
    st.caption("Temporal graph mining and delay prediction")
    st.divider()

    st.markdown("### Project Pages")
    st.page_link("pages/1_Network_Explorer.py", label="Network Explorer", icon="🗺️")
    st.page_link("pages/2_Motif_Viewer.py", label="Motif Viewer", icon="🔗")
    st.page_link("pages/3_Delay_Prediction.py", label="Delay Prediction", icon="⏱️")
    st.page_link("pages/4_Model_Performance.py", label="Model Performance", icon="📈")
    st.page_link("pages/5_Causal_Graph.py", label="Causal Graph", icon="🕸️")
    st.divider()

    st.markdown("### Final Run")
    if best is not None:
        st.metric("Best Model", str(best["Model"]))
        st.metric("Best MAE", f"{best['MAE']:.2f} min")
    st.caption("Updated after the final notebook comparison.")


st.title("Flight Delay Motif Mining for Predictive Analysis")
st.markdown(
    """
This dashboard summarizes a flight-delay forecasting project where airports are
modeled as a temporal graph. The central result is that **graph-aware delay
propagation modeling beats pure tabular and pure time-series baselines**.
"""
)

st.divider()

if best is not None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", str(best["Model"]))
    c2.metric("Best MAE", f"{best['MAE']:.2f} min")
    c3.metric("Best RMSE", f"{best['RMSE']:.2f} min")
    c4.metric("Motifs Mined", f"{len(motifs):,}")

st.info(
    "Final finding: the original motif-aware STGNN remains the strongest model "
    "because it learns route-network delay propagation directly."
)

st.divider()

left, right = st.columns([1.15, 1])

with left:
    st.subheader("Final Model Leaderboard")
    if comparison.empty:
        st.warning("model_comparison.csv was not found. Run the notebook to generate final results.")
    else:
        st.dataframe(
            comparison.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

        colors = ["#2ecc71" if i == 0 else "#4a90d9" for i in range(len(comparison))]
        fig = px.bar(
            comparison,
            x="MAE",
            y="Model",
            orientation="h",
            text=comparison["MAE"].map(lambda v: f"{v:.2f}"),
            title="Final Model MAE Comparison",
        )
        fig.update_traces(marker_color=colors, textposition="outside")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            xaxis_title="MAE (minutes)",
            yaxis_title="Model",
            yaxis={"categoryorder": "total ascending"},
            height=390,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Project Pipeline")
    st.markdown(
        """
1. **Raw flights**: 2023 US flight records  
2. **Preprocessing**: top airports, delayed flights, 6-hour windows  
3. **Temporal graph**: airports as nodes, routes as directed edges  
4. **Motif mining**: repeated delay chains such as `A -> B -> C`  
5. **Causal filtering**: motif-pruned PCMCI benchmark  
6. **Prediction**: STGNN compared with XGBoost, RF, HMM, N-HiTS, ensemble  
"""
    )

    st.subheader("Why STGNN Wins")
    st.markdown(
        """
- XGBoost and Random Forest see tabular features only.  
- HMM explains delay states but is weak as a predictor.  
- N-HiTS sees temporal history but not route propagation.  
- STGNN sees both **airport connectivity** and **time dynamics**.  
"""
    )

st.divider()

st.subheader("Important Modeling Note")
st.markdown(
    """
The corrected tabular baselines remove `arrival_delay` from their feature matrix.
That prevents data leakage because arrival delay is the value being predicted and
would not be known at prediction time in production.
"""
)

st.caption("Use the sidebar to explore motifs, airport maps, prediction simulation, and detailed performance.")

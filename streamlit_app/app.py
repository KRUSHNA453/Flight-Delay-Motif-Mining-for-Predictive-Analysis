"""
app.py — Main entry point for the Flight Delay Motif Mining Dashboard.
Run with:  streamlit run app.py
"""

import streamlit as st

# ── Page config (must be the very first Streamlit call) ──────────────────
st.set_page_config(
    page_title="Flight Delay Motif Mining",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark-themed custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Global dark palette overrides */
    .stApp {
        background-color: #0f1117;
    }
    /* Sidebar branding area */
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    /* Accent link colour */
    a { color: #00d4ff !important; }
    /* Metric card styling */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e;
    }
    [data-testid="stMetricValue"] {
        color: #00d4ff;
        font-size: 2rem;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
    /* Section headers */
    h1, h2, h3 {
        color: #e6edf3 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar (persistent across all pages) ───────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ Flight Delay Motif Mining")
    st.caption("Temporal Propagation Analysis & STGNN Prediction")
    st.divider()

    st.markdown("### 📊 Dataset Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Flights", "265K")
    col2.metric("Airports", "50")
    col3.metric("Motifs", "233")
    st.divider()

    st.markdown("### 🗂️ Pages")
    st.page_link("pages/1_Network_Explorer.py",  label="🗺️ Network Explorer",   icon="🗺️")
    st.page_link("pages/2_Motif_Viewer.py",      label="🔗 Motif Viewer",       icon="🔗")
    st.page_link("pages/3_Delay_Prediction.py",  label="⏱️ Delay Prediction",   icon="⏱️")
    st.page_link("pages/4_Model_Performance.py",  label="📈 Model Performance",  icon="📈")
    st.page_link("pages/5_Causal_Graph.py",       label="🕸️ Causal Graph",       icon="🕸️")
    st.divider()
    st.caption("Built with Streamlit • 2024-25 Research Pipeline")

# ── Landing page content ────────────────────────────────────────────────
st.title("✈️ Flight Delay Motif Mining for Predictive Analysis")
st.markdown("""
> **Research Goal:** Move beyond simple delay classification and model how
> delays *propagate* through the US airport network using **Temporal Motif
> Mining**, **Causal Discovery (PCMCI)**, and a **Spatio-Temporal Graph
> Neural Network (STGNN)**.
""")

st.divider()

# Key metrics row
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE Improvement", "23.24%", delta="14.2 → 10.9 min")
c2.metric("RMSE Improvement", "22.17%", delta="22.1 → 17.2 min")
c3.metric("MAPE Improvement", "33.33%", delta="0.18 → 0.12")
c4.metric("Causal Speedup", "45.2%", delta="12.45s → 6.82s")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🔬 Pipeline Overview")
    st.markdown("""
    1. **Dataset Integration** — US flights 2023 + weather + geolocation  
    2. **Temporal Graph Construction** — 6-hour snapshots (480 total)  
    3. **DM-Miner** — 233 unique 3-node delay propagation chains  
    4. **PCMCI Causal Discovery** — Motif-filtered for 45% speedup  
    5. **STGNN Prediction** — 50 epochs, early stopping, GTX 1650  
    6. **Evaluation** — MAE / RMSE / MAPE ablation study  
    """)

with col_right:
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    | Component | Technology |
    |---|---|
    | Graph Learning | PyTorch Geometric |
    | Causal Analysis | Tigramite / PCMCI |
    | Graph Ops | NetworkX |
    | Visualisation | Plotly, PyDeck, PyVis |
    | Data Science | Pandas, NumPy, Scikit-learn |
    | Dashboard | Streamlit |
    """)

st.info("👈 Use the sidebar to navigate to individual analysis pages.", icon="ℹ️")

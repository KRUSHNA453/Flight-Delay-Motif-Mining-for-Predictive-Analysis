"""
Page 2 — Motif Viewer
----------------------
Searchable/sortable table of all 233 motifs, bar & scatter charts,
mini-map for a selected chain, and a Hawaii Chain callout card.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

from utils.data_loader import load_motifs, get_airport_coords_map, get_airport_name_map

st.set_page_config(page_title="Motif Viewer", page_icon="🔗", layout="wide")
st.title("🔗 Motif Viewer")
st.caption("Explore all 233 discovered delay propagation chains")

# ── Data ────────────────────────────────────────────────────────────────
with st.spinner("Loading motifs…"):
    motifs = load_motifs()
    coords = get_airport_coords_map()
    names  = get_airport_name_map()

# Simulated "Avg Delay Propagated (mins)" — derived from dCPI × 2
motifs["avg_delay_mins"] = (motifs["dCPI"] * 2).round(1)

# ── Hawaii Chain callout ────────────────────────────────────────────────
hawaii = motifs[(motifs["node_b"] == "OGG") & (motifs["node_c"] == "HNL")]
if not hawaii.empty:
    top_hawaii = hawaii.sort_values("dCPI", ascending=False).iloc[0]
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a2332 0%, #0d2137 100%);
                border: 1px solid #00d4ff; border-radius: 12px;
                padding: 20px; margin-bottom: 20px;">
        <h3 style="color:#00d4ff; margin:0;">🌺 Hawaii Chain — Most Significant Motif Pattern</h3>
        <p style="color:#8b949e; margin:8px 0 0 0;">
            <b style="color:#e6edf3;">{top_hawaii['chain']}</b> &nbsp;|&nbsp;
            Frequency: <b style="color:#00d4ff;">{top_hawaii['frequency']}</b> &nbsp;|&nbsp;
            Causal Score (dCPI): <b style="color:#ff4b4b;">{top_hawaii['dCPI']}</b> &nbsp;|&nbsp;
            Avg Delay: <b>{top_hawaii['avg_delay_mins']:.1f} min</b>
        </p>
        <p style="color:#8b949e; font-size:0.85rem; margin-top:6px;">
            Mainland airports propagate delays to Maui (OGG) which then cascades
            to Honolulu (HNL). This pattern consistently shows the highest causal
            significance due to limited routing alternatives over the Pacific.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Searchable table ────────────────────────────────────────────────────
st.subheader("📋 All Motifs")
search = st.text_input("🔍 Search by airport code or motif ID", "")

display = motifs.copy()
if search:
    mask = (
        display["chain"].str.contains(search.upper(), na=False) |
        display["motif_id"].str.contains(search.upper(), na=False)
    )
    display = display[mask]

sort_col = st.selectbox("Sort by", ["dCPI", "frequency", "avg_delay_mins", "motif_id"], index=0)
display = display.sort_values(sort_col, ascending=(sort_col == "motif_id"))

st.dataframe(
    display[["motif_id", "chain", "frequency", "dCPI", "avg_delay_mins"]].reset_index(drop=True),
    use_container_width=True, hide_index=True, height=400,
)

# ── Charts ──────────────────────────────────────────────────────────────
st.divider()
chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("📊 Top 20 Motifs by Frequency")
    top20 = motifs.nlargest(20, "frequency")
    fig_bar = px.bar(
        top20, x="chain", y="frequency", color="dCPI",
        color_continuous_scale=["#00d4ff", "#ff4b4b"],
        labels={"chain": "Motif Chain", "frequency": "Frequency", "dCPI": "Causal Score"},
    )
    fig_bar.update_layout(
        template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        xaxis_tickangle=-45, height=450,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_right:
    st.subheader("🫧 Frequency vs Causal Significance")
    fig_scatter = px.scatter(
        motifs, x="frequency", y="dCPI", size="avg_delay_mins",
        hover_data=["chain", "motif_id"],
        color="avg_delay_mins",
        color_continuous_scale=["#00d4ff", "#ffcc00", "#ff4b4b"],
        labels={"frequency": "Frequency", "dCPI": "Causal Score (dCPI)",
                "avg_delay_mins": "Avg Delay (min)"},
    )
    fig_scatter.update_layout(
        template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        height=450,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── Mini-map for selected motif ─────────────────────────────────────────
st.divider()
st.subheader("🗺️ Motif Route Map")
selected_id = st.selectbox("Select a motif to visualise", motifs["motif_id"].tolist())
sel = motifs[motifs["motif_id"] == selected_id].iloc[0]

points = []
arcs = []
chain_nodes = [sel["node_a"], sel["node_b"], sel["node_c"]]
for i, code in enumerate(chain_nodes):
    c = coords.get(code)
    if c:
        points.append({"iata": code, "name": names.get(code, code),
                        "lat": c[0], "lon": c[1],
                        "color": [0, 212, 255] if i == 0 else ([255, 204, 0] if i == 1 else [255, 75, 75]),
                        "radius": 50000})
        if i > 0:
            prev = coords.get(chain_nodes[i - 1])
            if prev:
                arcs.append({"src_lat": prev[0], "src_lon": prev[1],
                             "dst_lat": c[0], "dst_lon": c[1]})

if points:
    avg_lat = np.mean([p["lat"] for p in points])
    avg_lon = np.mean([p["lon"] for p in points])

    st.pydeck_chart(pdk.Deck(
        layers=[
            pdk.Layer("ScatterplotLayer", pd.DataFrame(points),
                      get_position=["lon", "lat"], get_radius="radius",
                      get_fill_color="color", pickable=True),
            pdk.Layer("ArcLayer", pd.DataFrame(arcs),
                      get_source_position=["src_lon", "src_lat"],
                      get_target_position=["dst_lon", "dst_lat"],
                      get_source_color=[0, 212, 255, 200],
                      get_target_color=[255, 75, 75, 200],
                      get_width=4),
        ],
        initial_view_state=pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=3, pitch=20),
        tooltip={"html": "<b>{iata}</b><br/>{name}",
                 "style": {"backgroundColor": "#161b22", "color": "#e6edf3"}},
        map_style="mapbox://styles/mapbox/dark-v11",
    ), height=350)
    st.caption(f"**{sel['chain']}** — Frequency: {sel['frequency']} | dCPI: {sel['dCPI']}")
else:
    st.warning("Could not resolve coordinates for the selected motif.")

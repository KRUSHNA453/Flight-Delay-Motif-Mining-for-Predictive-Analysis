"""
Page 1 — Network Explorer
--------------------------
Interactive US map showing Top-50 airports as coloured nodes,
with motif chains overlaid as directional arcs.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

from utils.data_loader import get_top50_airports, load_motifs, get_airport_coords_map, get_airport_name_map
from utils.motif_utils import motifs_for_airport, delay_color_rgb

st.set_page_config(page_title="Network Explorer", page_icon="🗺️", layout="wide")
st.title("🗺️ Network Explorer")
st.caption("Interactive map of the Top-50 US airports and their delay propagation motifs")

# ── Load data ───────────────────────────────────────────────────────────
with st.spinner("Loading airport and motif data…"):
    airports = get_top50_airports()
    motifs   = load_motifs()
    coords   = get_airport_coords_map()
    names    = get_airport_name_map()

# ── Compute per-airport average dCPI (proxy for delay severity) ─────────
airport_dcpi = {}
for _, row in motifs.iterrows():
    for node in [row["node_a"], row["node_b"], row["node_c"]]:
        airport_dcpi.setdefault(node, []).append(row["dCPI"])
avg_dcpi = {k: np.mean(v) for k, v in airport_dcpi.items()}

# ── Sidebar controls ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗺️ Network Explorer")
    snapshot = st.slider("Time Snapshot Window", 0, 2, 0,
                         help="T=0 is the current window; T+1 and T+2 are the next 6-hour windows")
    top_n_motifs = st.slider("Motif arcs to display", 5, 50, 20)
    selected_airport = st.selectbox(
        "Inspect airport",
        sorted(airports["IATA_CODE"].tolist()),
        index=0,
    )

# ── Build node layer ────────────────────────────────────────────────────
node_data = []
for _, ap in airports.iterrows():
    code = ap["IATA_CODE"]
    dcpi = avg_dcpi.get(code, 5.0)
    node_data.append({
        "iata": code,
        "name": ap.get("AIRPORT", code),
        "lat": ap["LATITUDE"],
        "lon": ap["LONGITUDE"],
        "avg_delay": round(dcpi, 2),
        "color": delay_color_rgb(dcpi),
        "radius": 40000 + int((dcpi - 5.0) * 30000),
    })

node_df = pd.DataFrame(node_data)

node_layer = pdk.Layer(
    "ScatterplotLayer",
    data=node_df,
    get_position=["lon", "lat"],
    get_radius="radius",
    get_fill_color="color",
    pickable=True,
    auto_highlight=True,
)

# ── Build arc layer (top-N motifs) ──────────────────────────────────────
sorted_motifs = motifs.sort_values("dCPI", ascending=False).head(top_n_motifs)
arc_data = []
for _, m in sorted_motifs.iterrows():
    nodes = [m["node_a"], m["node_b"], m["node_c"]]
    # Only draw arcs for the selected snapshot step
    if snapshot == 0:
        pairs = [(nodes[0], nodes[1])]
    elif snapshot == 1:
        pairs = [(nodes[1], nodes[2])]
    else:
        pairs = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]

    for src, dst in pairs:
        src_coord = coords.get(src)
        dst_coord = coords.get(dst)
        if src_coord and dst_coord:
            arc_data.append({
                "src_lat": src_coord[0], "src_lon": src_coord[1],
                "dst_lat": dst_coord[0], "dst_lon": dst_coord[1],
                "chain": m["chain"],
                "dCPI": m["dCPI"],
            })

arc_df = pd.DataFrame(arc_data) if arc_data else pd.DataFrame(
    columns=["src_lat", "src_lon", "dst_lat", "dst_lon", "chain", "dCPI"])

arc_layer = pdk.Layer(
    "ArcLayer",
    data=arc_df,
    get_source_position=["src_lon", "src_lat"],
    get_target_position=["dst_lon", "dst_lat"],
    get_source_color=[0, 212, 255, 180],
    get_target_color=[255, 75, 75, 180],
    get_width=2,
    pickable=True,
)

# ── Render map ──────────────────────────────────────────────────────────
tooltip = {
    "html": "<b>{iata}</b><br/>{name}<br/>Avg dCPI: {avg_delay}",
    "style": {"backgroundColor": "#161b22", "color": "#e6edf3", "fontSize": "13px"},
}

view = pdk.ViewState(latitude=39.5, longitude=-98.35, zoom=3.4, pitch=30)

st.pydeck_chart(pdk.Deck(
    layers=[node_layer, arc_layer],
    initial_view_state=view,
    tooltip=tooltip,
    map_style="mapbox://styles/mapbox/dark-v11",
))

# ── Airport detail card ─────────────────────────────────────────────────
st.divider()
st.subheader(f"🔍 Airport Detail — {selected_airport}")

col1, col2 = st.columns([1, 2])
with col1:
    ap_name = names.get(selected_airport, selected_airport)
    ap_dcpi = avg_dcpi.get(selected_airport, 0)
    st.metric("Airport", ap_name)
    st.metric("Avg Causal Score (dCPI)", f"{ap_dcpi:.2f}")
with col2:
    top3 = motifs_for_airport(motifs, selected_airport, top_n=3)
    if top3.empty:
        st.info("No motifs found for this airport.")
    else:
        st.markdown("**Top Connected Motifs:**")
        st.dataframe(top3[["chain", "frequency", "dCPI"]].reset_index(drop=True),
                     use_container_width=True, hide_index=True)

"""
Page 5 — Causal Graph
----------------------
Interactive causal graph built from motif dCPI scores, rendered with
pyvis.  Controls for threshold filtering, ego-graph view, and edge
label toggling.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import tempfile

from utils.data_loader import load_motifs, get_top50_airports, get_airport_name_map
from utils.graph_builder import build_causal_graph, get_ego_graph, top_spreaders

st.set_page_config(page_title="Causal Graph", page_icon="🕸️", layout="wide")
st.title("🕸️ Causal Graph")
st.caption("Explore causal delay-propagation relationships between airports")

# ── Data ────────────────────────────────────────────────────────────────
with st.spinner("Building causal graph…"):
    motifs   = load_motifs()
    airports = get_top50_airports()
    names    = get_airport_name_map()

# ── Sidebar controls ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🕸️ Causal Graph Controls")
    min_dcpi = st.slider("Minimum causal strength (dCPI)",
                         float(motifs["dCPI"].min()),
                         float(motifs["dCPI"].max()),
                         5.5, 0.05)
    all_codes = sorted(set(motifs["node_a"]) | set(motifs["node_b"]) | set(motifs["node_c"]))
    focus_airport = st.selectbox("Focus airport (ego-graph)", ["All"] + all_codes)
    show_labels = st.toggle("Show edge labels", value=False)

# ── Build graph ─────────────────────────────────────────────────────────
G = build_causal_graph(motifs, min_dcpi=min_dcpi)

if focus_airport != "All" and focus_airport in G:
    G = get_ego_graph(G, focus_airport, radius=1)

if len(G.nodes) == 0:
    st.warning("No nodes remaining at this threshold. Lower the minimum causal strength.")
    st.stop()

# ── Compute node metrics ────────────────────────────────────────────────
spreaders = top_spreaders(G, n=5)
in_degrees  = dict(G.in_degree())
out_degrees = dict(G.out_degree())
max_degree  = max(max(in_degrees.values(), default=1), max(out_degrees.values(), default=1))

# ── Build pyvis network ────────────────────────────────────────────────
net = Network(height="620px", width="100%", bgcolor="#0f1117",
              font_color="#e6edf3", directed=True)
net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=200)

for node in G.nodes():
    in_d  = in_degrees.get(node, 0)
    out_d = out_degrees.get(node, 0)
    total = in_d + out_d

    # Size by degree centrality
    size = 10 + (total / max_degree) * 40

    # Colour: more out-degree → red (spreader), more in-degree → blue (receiver)
    if total > 0:
        ratio = out_d / total  # 0 = pure receiver, 1 = pure spreader
    else:
        ratio = 0.5
    r = int(255 * ratio)
    b = int(255 * (1 - ratio))
    color = f"#{r:02x}30{b:02x}"

    # Gold border for top-5 spreaders
    border_color = "#ffd700" if node in spreaders else "#30363d"
    border_width = 3 if node in spreaders else 1

    label = f"{node}\n({names.get(node, '')})"
    title = (f"<b>{node}</b> — {names.get(node, '')}<br/>"
             f"In-degree: {in_d} | Out-degree: {out_d}")

    net.add_node(node, label=label, title=title, size=size,
                 color={"background": color, "border": border_color},
                 borderWidth=border_width,
                 font={"size": 12, "color": "#e6edf3"})

for u, v, data in G.edges(data=True):
    weight = data.get("weight", 1)
    freq   = data.get("frequency", 0)
    width  = 0.5 + (weight / 20)  # scale

    label = f"{weight:.1f}" if show_labels else ""
    title = f"{u}→{v}<br/>Cumulative dCPI: {weight:.1f}<br/>Freq: {freq}"

    net.add_edge(u, v, value=width, title=title, label=label,
                 color="#30536d", arrows="to",
                 font={"size": 9, "color": "#8b949e"})

# ── Render ──────────────────────────────────────────────────────────────
# Save to a temp file and embed as HTML
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
net.save_graph(tmp.name)
tmp.close()

with open(tmp.name, "r", encoding="utf-8") as f:
    html = f.read()

components.html(html, height=640, scrolling=False)

# Clean up
try:
    os.unlink(tmp.name)
except OSError:
    pass

# ── Legend ──────────────────────────────────────────────────────────────
st.divider()
legend_cols = st.columns(4)
legend_cols[0].markdown("🔴 **Red nodes** = delay *spreaders* (high out-degree)")
legend_cols[1].markdown("🔵 **Blue nodes** = delay *receivers* (high in-degree)")
legend_cols[2].markdown("🟡 **Gold border** = Top-5 most influential spreaders")
legend_cols[3].markdown("📏 **Node size** = total degree centrality")

# ── Top spreaders table ─────────────────────────────────────────────────
st.divider()
st.subheader("🏅 Top 5 Delay Spreader Airports")
spreader_data = []
for s in spreaders:
    spreader_data.append({
        "Airport": f"{s} — {names.get(s, '')}",
        "Out-Degree": out_degrees.get(s, 0),
        "In-Degree": in_degrees.get(s, 0),
        "Total Connections": out_degrees.get(s, 0) + in_degrees.get(s, 0),
    })
st.dataframe(pd.DataFrame(spreader_data), use_container_width=True, hide_index=True)

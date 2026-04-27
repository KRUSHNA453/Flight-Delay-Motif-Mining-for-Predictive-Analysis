"""
Page 3 — Delay Prediction
---------------------------
Select an origin/destination pair, time of day, and optional weather
inputs.  A simulated prediction (weighted-average of motif scores)
is shown along with activated motif chains and a route mini-map.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

from utils.data_loader import (load_motifs, get_top50_airports,
                                get_airport_coords_map, get_airport_name_map)
from utils.motif_utils import predict_delay

st.set_page_config(page_title="Delay Prediction", page_icon="⏱️", layout="wide")
st.title("⏱️ Delay Prediction")
st.caption("Simulate flight delay predictions using motif-weighted analysis")

# ── Data ────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    motifs   = load_motifs()
    airports = get_top50_airports()
    coords   = get_airport_coords_map()
    names    = get_airport_name_map()

codes = sorted(airports["IATA_CODE"].tolist())

# ── Input controls ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⏱️ Prediction Inputs")

col1, col2, col3 = st.columns(3)
with col1:
    origin = st.selectbox("🛫 Origin Airport", codes, index=codes.index("DEN") if "DEN" in codes else 0)
with col2:
    dest = st.selectbox("🛬 Destination Airport", codes, index=codes.index("LAS") if "LAS" in codes else 1)
with col3:
    tod = st.selectbox("🕐 Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

# Optional weather
with st.expander("🌤️ Optional Weather Conditions"):
    wcol1, wcol2, wcol3 = st.columns(3)
    wind_speed = wcol1.slider("Wind Speed (km/h)", 0, 100, 15)
    precip     = wcol2.slider("Precipitation (mm)", 0.0, 50.0, 0.0, 0.5)
    pressure   = wcol3.slider("Pressure (hPa)", 980, 1040, 1013)

    # Simple weather factor: higher wind/precip or lower pressure → more delay
    weather_factor = 1.0
    if wind_speed > 40:
        weather_factor += 0.3
    if precip > 10:
        weather_factor += 0.4
    if pressure < 1000:
        weather_factor += 0.2

predict_btn = st.button("🔮 Predict Delay", use_container_width=True, type="primary")

# ── Prediction ──────────────────────────────────────────────────────────
if predict_btn:
    if origin == dest:
        st.error("Origin and destination must be different airports.")
    else:
        with st.spinner("Running prediction…"):
            result = predict_delay(motifs, origin, dest, tod, weather_factor)

        predicted = result["predicted_delay"]
        confidence = result["confidence"]
        activated = result["activated_motifs"]

        # Colour the metric by severity
        if predicted < 10:
            delta_color = "normal"
        elif predicted < 20:
            delta_color = "normal"
        else:
            delta_color = "inverse"

        st.divider()
        st.subheader("📋 Prediction Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Delay", f"{predicted:.1f} min")

        # Confidence bar
        conf_colors = {"High": "#22c55e", "Medium": "#eab308", "Low": "#ff4b4b"}
        conf_widths = {"High": "100%", "Medium": "60%", "Low": "30%"}
        m2.markdown(f"""
        <div style="margin-top:8px;">
            <span style="color:#8b949e;">Confidence</span><br/>
            <span style="font-size:1.5rem; font-weight:700; color:{conf_colors[confidence]};">{confidence}</span>
            <div style="background:#30363d; border-radius:6px; height:8px; margin-top:6px;">
                <div style="background:{conf_colors[confidence]}; width:{conf_widths[confidence]};
                            height:8px; border-radius:6px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        m3.metric("Weather Factor", f"×{weather_factor:.1f}")

        # Activated motifs
        st.divider()
        st.subheader("🔗 Activated Motif Chains")
        if activated.empty:
            st.info("No direct motif chains found for this route. Prediction is based on global averages.")
        else:
            st.dataframe(
                activated[["motif_id", "chain", "frequency", "dCPI"]].reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )

        # Route mini-map
        st.divider()
        st.subheader("🗺️ Route Map")
        o_coord = coords.get(origin)
        d_coord = coords.get(dest)

        if o_coord and d_coord:
            map_points = pd.DataFrame([
                {"iata": origin, "name": names.get(origin, origin),
                 "lat": o_coord[0], "lon": o_coord[1],
                 "color": [0, 212, 255], "radius": 50000},
                {"iata": dest, "name": names.get(dest, dest),
                 "lat": d_coord[0], "lon": d_coord[1],
                 "color": [255, 75, 75], "radius": 50000},
            ])
            map_arcs = pd.DataFrame([{
                "src_lat": o_coord[0], "src_lon": o_coord[1],
                "dst_lat": d_coord[0], "dst_lon": d_coord[1],
            }])
            mid_lat = (o_coord[0] + d_coord[0]) / 2
            mid_lon = (o_coord[1] + d_coord[1]) / 2

            st.pydeck_chart(pdk.Deck(
                layers=[
                    pdk.Layer("ScatterplotLayer", map_points,
                              get_position=["lon", "lat"], get_radius="radius",
                              get_fill_color="color", pickable=True),
                    pdk.Layer("ArcLayer", map_arcs,
                              get_source_position=["src_lon", "src_lat"],
                              get_target_position=["dst_lon", "dst_lat"],
                              get_source_color=[0, 212, 255, 200],
                              get_target_color=[255, 75, 75, 200],
                              get_width=4),
                ],
                initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lon,
                                                  zoom=3.5, pitch=20),
                tooltip={"html": "<b>{iata}</b><br/>{name}",
                         "style": {"backgroundColor": "#161b22", "color": "#e6edf3"}},
                map_style="mapbox://styles/mapbox/dark-v11",
            ), height=350)
        else:
            st.warning("Could not resolve coordinates for the selected airports.")

st.divider()
st.caption("⚠️ Predictions are simulated using motif-weighted averages. "
           "For production use, load the trained STGNN model checkpoint.")

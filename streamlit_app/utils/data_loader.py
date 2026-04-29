"""
utils/data_loader.py
--------------------
Centralized data-loading module for the Flight Delay Motif Mining dashboard.
Every public function uses @st.cache_data so files are read only once per
session, keeping the app fast even on large CSVs.
"""

import json
import os
import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path helpers – all paths are relative to *this* file so the app works
# regardless of the working directory Streamlit is launched from.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_DATASET_DIR = os.path.join(_PROJECT_ROOT, "Dataset")

def _dataset_path(filename: str) -> str:
    """Return the absolute path to a file inside the Dataset/ folder."""
    return os.path.join(_DATASET_DIR, filename)

def _project_path(filename: str) -> str:
    """Return the absolute path to a file in the project root."""
    return os.path.join(_PROJECT_ROOT, filename)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_airports() -> pd.DataFrame:
    """Load airports_geolocation.csv and return a clean DataFrame."""
    path = _dataset_path("airports_geolocation.csv")
    df = pd.read_csv(path)
    # Standardise column names
    df.columns = [c.strip() for c in df.columns]
    # Drop rows with missing coordinates
    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])
    return df


@st.cache_data(show_spinner=False)
def load_motifs() -> pd.DataFrame:
    """Load motifs.json and return a tidy DataFrame with columns:
       motif_id, node_a, node_b, node_c, chain, frequency, dCPI
    """
    path = _project_path("motifs.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for m in raw:
        nodes = m["nodes"]
        rows.append({
            "motif_id": m["motif_id"],
            "node_a": nodes[0],
            "node_b": nodes[1],
            "node_c": nodes[2],
            "chain": " → ".join(nodes),
            "frequency": m["frequency"],
            "dCPI": m["dCPI"],
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    """Load results_table.csv with model comparison metrics."""
    path = _project_path("results_table.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    """Load the final model comparison table if it exists.

    Falls back to the original ablation table so the dashboard still works
    before the notebook has generated the final comparison artifact.
    """
    path = _project_path("model_comparison.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = load_results()
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    """Load Random Forest feature importance results if available."""
    path = _project_path("feature_importance.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["feature", "importance"])
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def get_model_comparison_chart_path() -> str:
    """Return the path to the final model comparison PNG."""
    return _project_path("model_comparison.png")


@st.cache_data(show_spinner=False)
def load_weather() -> pd.DataFrame:
    """Load weather_meteo_by_airport.csv."""
    path = _dataset_path("weather_meteo_by_airport.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def get_evaluation_chart_path() -> str:
    """Return the path to the final evaluation chart PNG."""
    return _project_path("final_evaluation_charts.png")


# ---------------------------------------------------------------------------
# Derived / aggregated helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_top50_airports() -> pd.DataFrame:
    """Return the airports DataFrame filtered to only the airports that
    appear in at least one motif chain (i.e. the Top-50 used in the study).
    """
    motifs = load_motifs()
    airports = load_airports()

    # Collect unique IATA codes from motifs
    codes = set(motifs["node_a"]) | set(motifs["node_b"]) | set(motifs["node_c"])

    top50 = airports[airports["IATA_CODE"].isin(codes)].copy()
    return top50


@st.cache_data(show_spinner=False)
def get_airport_coords_map() -> dict:
    """Return a dict  IATA_CODE -> (lat, lon)  for quick lookups."""
    df = load_airports()
    return {
        row.IATA_CODE: (row.LATITUDE, row.LONGITUDE)
        for row in df.itertuples()
    }


@st.cache_data(show_spinner=False)
def get_airport_name_map() -> dict:
    """Return a dict  IATA_CODE -> full airport name."""
    df = load_airports()
    return dict(zip(df["IATA_CODE"], df["AIRPORT"]))

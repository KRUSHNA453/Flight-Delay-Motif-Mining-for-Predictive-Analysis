"""
utils/motif_utils.py
--------------------
Helper functions that operate on the motif DataFrame:
 – finding motifs for a given airport pair
 – computing simulated delay predictions (fallback when model is absent)
 – colour scales for delay magnitudes
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Motif look-ups
# ---------------------------------------------------------------------------

def motifs_for_route(motifs_df: pd.DataFrame,
                     origin: str, destination: str) -> pd.DataFrame:
    """Return all motif chains that contain the leg origin→destination
    (either as A→B or B→C)."""
    mask = (
        ((motifs_df["node_a"] == origin) & (motifs_df["node_b"] == destination)) |
        ((motifs_df["node_b"] == origin) & (motifs_df["node_c"] == destination))
    )
    return motifs_df[mask].sort_values("dCPI", ascending=False)


def motifs_for_airport(motifs_df: pd.DataFrame,
                       iata: str, top_n: int = 3) -> pd.DataFrame:
    """Return the top-N motifs that involve *iata* (in any position)."""
    mask = (
        (motifs_df["node_a"] == iata) |
        (motifs_df["node_b"] == iata) |
        (motifs_df["node_c"] == iata)
    )
    return motifs_df[mask].sort_values("dCPI", ascending=False).head(top_n)


# ---------------------------------------------------------------------------
# Simulated delay prediction (weighted-average fallback)
# ---------------------------------------------------------------------------

TIME_MULTIPLIER = {
    "Night":     0.7,
    "Morning":   1.0,
    "Afternoon": 1.2,
    "Evening":   1.1,
}


def predict_delay(motifs_df: pd.DataFrame,
                  origin: str, destination: str,
                  time_of_day: str = "Morning",
                  weather_factor: float = 1.0) -> dict:
    """
    Simulate a delay prediction when the real STGNN model is unavailable.
    Uses the weighted average of dCPI scores from activated motif chains,
    scaled by time-of-day and weather multipliers.

    Returns a dict with:
      predicted_delay  – minutes (float)
      confidence       – "Low" / "Medium" / "High"
      activated_motifs – DataFrame of matching chains
    """
    activated = motifs_for_route(motifs_df, origin, destination)

    if activated.empty:
        # No direct motif — fall back to a global average
        base = motifs_df["dCPI"].mean() * 2.0
        confidence = "Low"
    else:
        # Weighted average: weight = frequency
        weights = activated["frequency"].values.astype(float)
        scores  = activated["dCPI"].values.astype(float)
        base = float(np.average(scores, weights=weights)) * 2.0

        total_freq = int(weights.sum())
        if total_freq >= 100:
            confidence = "High"
        elif total_freq >= 30:
            confidence = "Medium"
        else:
            confidence = "Low"

    time_mult = TIME_MULTIPLIER.get(time_of_day, 1.0)
    predicted = base * time_mult * weather_factor

    return {
        "predicted_delay": round(predicted, 1),
        "confidence": confidence,
        "activated_motifs": activated,
    }


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def delay_color_hex(dcpi: float, vmin: float = 5.0, vmax: float = 7.0) -> str:
    """Map a dCPI value to a green→yellow→red hex colour."""
    t = np.clip((dcpi - vmin) / (vmax - vmin), 0, 1)
    r = int(255 * min(1, 2 * t))
    g = int(255 * min(1, 2 * (1 - t)))
    return f"#{r:02x}{g:02x}00"


def delay_color_rgb(dcpi: float, vmin: float = 5.0, vmax: float = 7.0) -> list:
    """Map a dCPI value to an [R, G, B, A] list for pydeck."""
    t = np.clip((dcpi - vmin) / (vmax - vmin), 0, 1)
    r = int(255 * min(1, 2 * t))
    g = int(255 * min(1, 2 * (1 - t)))
    return [r, g, 0, 200]

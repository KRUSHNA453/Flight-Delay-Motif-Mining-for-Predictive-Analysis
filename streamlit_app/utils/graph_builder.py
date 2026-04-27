"""
utils/graph_builder.py
----------------------
Builds NetworkX directed graphs from the motif data for causal-graph
visualisation and analysis.
"""

import networkx as nx
import pandas as pd


def build_causal_graph(motifs_df: pd.DataFrame,
                       min_dcpi: float = 0.0) -> nx.DiGraph:
    """
    Build a directed graph where:
      - Nodes  = airports (IATA codes)
      - Edges  = causal links derived from motif chains
      - Weight = dCPI (delay Causal Propagation Index)

    Each 3-node chain A→B→C contributes TWO directed edges:
      A→B  and  B→C, each weighted by the motif's dCPI.
    If the same edge appears in multiple motifs the weights are
    summed (stronger cumulative causal link).
    """
    filtered = motifs_df[motifs_df["dCPI"] >= min_dcpi]

    G = nx.DiGraph()

    for _, row in filtered.iterrows():
        a, b, c = row["node_a"], row["node_b"], row["node_c"]
        w = row["dCPI"]
        freq = row["frequency"]

        for src, dst in [(a, b), (b, c)]:
            if G.has_edge(src, dst):
                G[src][dst]["weight"] += w
                G[src][dst]["frequency"] += freq
                G[src][dst]["count"] += 1
            else:
                G.add_edge(src, dst, weight=w, frequency=freq, count=1)

    return G


def get_ego_graph(G: nx.DiGraph, center: str, radius: int = 1) -> nx.DiGraph:
    """Return the ego-graph (sub-graph) centred on *center* airport."""
    if center not in G:
        return nx.DiGraph()
    return nx.ego_graph(G, center, radius=radius)


def top_spreaders(G: nx.DiGraph, n: int = 5) -> list:
    """Return the *n* airports with the highest out-degree (delay spreaders)."""
    out_deg = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in out_deg[:n]]

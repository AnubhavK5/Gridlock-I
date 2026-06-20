from __future__ import annotations
import pandas as pd
import networkx as nx
from itertools import combinations
from typing import Tuple, List

CSV_PATH      = "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
DEFAULT_EDGE_WEIGHT = 5.0
INCIDENT_EDGE_WEIGHT = 120.0


def _build_graph_from_csv() -> Tuple[nx.DiGraph, List[str]]:
    try:
        df = pd.read_csv(CSV_PATH, usecols=["junction", "police_station"])
        df = df.dropna(subset=["junction"])
        df["junction"] = df["junction"].astype(str).str.strip()
        df["police_station"] = df["police_station"].fillna("Unknown").astype(str).str.strip()
    except Exception as e:
        print(f"[diversion_engine] CSV load failed ({e}). Using minimal fallback graph.")
        df = pd.DataFrame({
            "junction":       [f"J{i}" for i in range(10)],
            "police_station": [f"PS_{i//3}" for i in range(10)],
        })

    G = nx.DiGraph()
    junctions = df["junction"].unique().tolist()
    G.add_nodes_from(junctions)

    # Group junctions by police_station and create edges between co-located junctions
    station_groups = df.groupby("police_station")["junction"].apply(list).to_dict()
    added_edges = set()
    for station, juncs in station_groups.items():
        unique_juncs = list(dict.fromkeys(juncs))  # preserve order, deduplicate
        for u, v in combinations(unique_juncs[:6], 2):  # cap at 6 per station to avoid explosion
            if u == v:
                continue
            for src, dst in [(u, v), (v, u)]:
                if (src, dst) not in added_edges:
                    G.add_edge(src, dst, travel_time=DEFAULT_EDGE_WEIGHT,
                               police_station=station)
                    added_edges.add((src, dst))

    # Ensure connectivity: chain all junctions sequentially as fallback spine
    if len(junctions) > 1:
        for i in range(len(junctions) - 1):
            u, v = junctions[i], junctions[i + 1]
            for src, dst in [(u, v), (v, u)]:
                if not G.has_edge(src, dst):
                    G.add_edge(src, dst, travel_time=DEFAULT_EDGE_WEIGHT,
                               police_station="spine")

    return G, junctions


class DiversionEngine:
    def __init__(self):
        self.G, self.junctions = _build_graph_from_csv()
        print(f"[diversion_engine] Graph built: {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges.")

    def generate_detour(
        self,
        origin_junc: str,
        dest_junc: str,
        incident_junc: str,
    ) -> Tuple[List[str], List[str], str]:
        """
        Compute baseline path and crisis-detour path around an incident junction.

        Returns
        -------
        (baseline_path, detour_path, status_message)
        """
        G = self.G.copy()

        # --- Validate nodes ---
        for node, label in [(origin_junc, "Origin"), (dest_junc, "Destination"),
                             (incident_junc, "Incident")]:
            if node not in G:
                return [], [], f"ERROR: {label} junction '{node}' not found in graph."

        if origin_junc == dest_junc:
            return [origin_junc], [origin_junc], "Origin and destination are identical."

        # --- Baseline path (no incident inflation) ---
        try:
            baseline_path = nx.shortest_path(G, origin_junc, dest_junc, weight="travel_time")
        except nx.NetworkXNoPath:
            baseline_path = []

        # --- Inflate edges touching the incident junction ---
        for u, v, data in list(G.edges(incident_junc, data=True)):
            G[u][v]["travel_time"] = INCIDENT_EDGE_WEIGHT
        for u, v, data in list(G.in_edges(incident_junc, data=True)):
            G[u][v]["travel_time"] = INCIDENT_EDGE_WEIGHT

        # --- Crisis detour path (with inflated weights) ---
        try:
            detour_path = nx.shortest_path(G, origin_junc, dest_junc, weight="travel_time")
        except nx.NetworkXNoPath:
            detour_path = []

        # --- Build status message ---
        if not baseline_path and not detour_path:
            status = "No path exists between origin and destination."
        elif not detour_path:
            status = "WARNING: No viable detour found. All routes pass through incident junction."
        elif detour_path == baseline_path:
            status = ("Detour is identical to baseline — incident junction is not on the "
                      "primary route. No diversion needed.")
        else:
            b_cost = _path_cost(self.G, baseline_path)
            d_cost = _path_cost(self.G, detour_path)
            delta  = ((d_cost - b_cost) / b_cost * 100) if b_cost > 0 else 0
            status = (f"Detour active. Travel time increase: +{delta:.1f}% "
                      f"({b_cost:.0f} min baseline → {d_cost:.0f} min via detour). "
                      f"Incident junction '{incident_junc}' bypassed.")

        return baseline_path, detour_path, status


def _path_cost(G: nx.DiGraph, path: List[str]) -> float:
    return sum(G[u][v].get("travel_time", DEFAULT_EDGE_WEIGHT)
               for u, v in zip(path[:-1], path[1:]))


# Module-level singleton
_engine_instance: DiversionEngine = None


def get_engine() -> DiversionEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DiversionEngine()
    return _engine_instance


def generate_detour(origin_junc: str, dest_junc: str, incident_junc: str
                    ) -> Tuple[List[str], List[str], str]:
    return get_engine().generate_detour(origin_junc, dest_junc, incident_junc)


if __name__ == "__main__":
    engine = DiversionEngine()
    jl = engine.junctions
    if len(jl) >= 3:
        b, d, s = engine.generate_detour(jl[0], jl[-1], jl[len(jl)//2])
        print("Baseline:", " → ".join(b[:6]), "..." if len(b) > 6 else "")
        print("Detour  :", " → ".join(d[:6]), "..." if len(d) > 6 else "")
        print("Status  :", s)
